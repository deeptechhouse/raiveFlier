"""Parallel research orchestrator for all entities on a rave flier.

Accepts extracted entities (artists, venue, promoter, date) and dispatches
concurrent research tasks via ``asyncio.gather``, collecting results from
all four domain-specific researchers in parallel.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from datetime import date

import structlog

from src.models.entities import EntityType
from src.models.flier import ExtractedEntities, ExtractedEntity
from src.models.research import ResearchResult
from src.services.artist_researcher import ArtistResearcher
from src.services.date_context_researcher import DateContextResearcher
from src.services.event_name_researcher import EventNameResearcher
from src.services.promoter_researcher import PromoterResearcher
from src.services.venue_researcher import VenueResearcher
from src.utils.logging import get_logger

# Attempt to use python-dateutil for flexible date parsing; fall back to
# manual parsing if the package is not installed.
try:
    from dateutil import parser as dateutil_parser

    _HAS_DATEUTIL = True
except ImportError:  # pragma: no cover
    _HAS_DATEUTIL = False

# Ordinal suffix pattern used during manual date parsing
_ORDINAL_RE = re.compile(r"(\d+)(st|nd|rd|th)\b", re.IGNORECASE)


class ResearchService:
    """Orchestrates parallel research across all entity types on a flier.

    Each researcher (artist, venue, promoter, date-context) is injected at
    construction time.  The main entry point, :meth:`research_all`, fans out
    concurrent ``asyncio`` tasks — one per artist plus one each for venue,
    promoter, and date — so a flier with five artists, a venue, a promoter,
    and a date produces eight simultaneous research jobs.

    All external service dependencies are injected through the individual
    researcher instances, following the adapter pattern (CLAUDE.md Section 6).
    """

    def __init__(
        self,
        artist_researcher: ArtistResearcher,
        venue_researcher: VenueResearcher,
        promoter_researcher: PromoterResearcher,
        date_context_researcher: DateContextResearcher,
        event_name_researcher: EventNameResearcher | None = None,
    ) -> None:
        self._artist_researcher = artist_researcher
        self._venue_researcher = venue_researcher
        self._promoter_researcher = promoter_researcher
        self._date_context_researcher = date_context_researcher
        self._event_name_researcher = event_name_researcher
        self._logger: structlog.BoundLogger = get_logger(__name__)

    # -- Public API -----------------------------------------------------------

    async def research_all(
        self,
        entities: ExtractedEntities,
        event_date: date | None = None,
    ) -> list[ResearchResult]:
        """Run all entity research concurrently and return collected results.

        Parameters
        ----------
        entities:
            The entities extracted from the flier's OCR output.
        event_date:
            An explicit event date.  When ``None``, the method will attempt
            to parse a date from ``entities.date``.

        Returns
        -------
        list[ResearchResult]
            One :class:`ResearchResult` per successfully-researched entity,
            plus placeholder results (confidence=0) for any entity whose
            research raised an exception.
        """
        # Resolve the event date from the extracted entity if not provided
        if event_date is None and entities.date is not None:
            event_date = await self._parse_event_date(entities.date)

        # Derive the city from the venue entity for context researchers
        city: str | None = None
        if entities.venue is not None:
            city = self._extract_city_hint(entities.venue.text)

        self._logger.info(
            "Starting parallel research",
            artists=len(entities.artists),
            has_venue=entities.venue is not None,
            has_promoter=entities.promoter is not None,
            has_event_name=entities.event_name is not None,
            has_date=event_date is not None,
            event_date=str(event_date) if event_date else None,
        )

        # -- Build task list --------------------------------------------------
        tasks: list[asyncio.Task[ResearchResult]] = []
        task_labels: list[str] = []

        # 1. One task per artist
        for artist_entity in entities.artists:
            tasks.append(
                asyncio.ensure_future(
                    self._artist_researcher.research(
                        artist_name=artist_entity.text,
                        before_date=event_date,
                        city=city,
                    )
                )
            )
            task_labels.append(f"artist:{artist_entity.text}")

        # 2. Venue task
        if entities.venue is not None:
            tasks.append(
                asyncio.ensure_future(
                    self._venue_researcher.research(
                        venue_name=entities.venue.text,
                        city=city,
                    )
                )
            )
            task_labels.append(f"venue:{entities.venue.text}")

        # 3. Promoter task
        if entities.promoter is not None:
            tasks.append(
                asyncio.ensure_future(
                    self._promoter_researcher.research(
                        promoter_name=entities.promoter.text,
                        city=city,
                    )
                )
            )
            task_labels.append(f"promoter:{entities.promoter.text}")

        # 4. Date-context task
        if event_date is not None:
            tasks.append(
                asyncio.ensure_future(
                    self._date_context_researcher.research(
                        event_date=event_date,
                        city=city,
                    )
                )
            )
            task_labels.append(f"date:{event_date.isoformat()}")

        # 5. Event-name task
        if entities.event_name is not None and self._event_name_researcher is not None:
            tasks.append(
                asyncio.ensure_future(
                    self._event_name_researcher.research(
                        event_name=entities.event_name.text,
                        promoter_name=entities.promoter.text if entities.promoter else None,
                        city=city,
                    )
                )
            )
            task_labels.append(f"event:{entities.event_name.text}")

        if not tasks:
            self._logger.warning("No researchable entities found on flier")
            return []

        # -- Execute all tasks concurrently -----------------------------------
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # -- Collect results --------------------------------------------------
        results: list[ResearchResult] = []
        for idx, raw in enumerate(raw_results):
            label = task_labels[idx]

            if isinstance(raw, Exception):
                self._logger.error(
                    "Research task failed",
                    task=label,
                    error=str(raw),
                    error_type=type(raw).__name__,
                )
                # Build a fallback ResearchResult so callers always get an
                # entry for every entity that was submitted.
                entity_type, entity_name = label.split(":", 1)
                results.append(
                    ResearchResult(
                        entity_type=EntityType(entity_type.upper()),
                        entity_name=entity_name,
                        sources_consulted=[],
                        confidence=0.0,
                        warnings=[f"Research failed: {type(raw).__name__}: {raw}"],
                    )
                )
            else:
                results.append(raw)

        self._logger.info(
            "Parallel research complete",
            total_tasks=len(tasks),
            successful=sum(1 for r in results if r.confidence > 0),
            failed=sum(1 for r in results if r.confidence == 0),
        )

        return results

    # -- Private helpers ------------------------------------------------------

    async def _parse_event_date(self, date_entity: ExtractedEntity | None) -> date | None:
        """Parse a date string from OCR text into a Python :class:`date`.

        Handles many common flier formats, including:
        - ``Saturday March 15th 1997``
        - ``03/15/97``
        - ``15.03.1997``
        - ``March 15, 1997``
        - ``15 March 1997``

        Uses ``python-dateutil`` when available for robust fuzzy parsing;
        otherwise falls back to manual regex parsing.
        """
        if date_entity is None:
            return None

        raw_text = date_entity.text.strip()
        if not raw_text:
            return None

        # Strip common prefixes and day-of-week names
        cleaned = re.sub(
            r"^(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s*,?\s*",
            "",
            raw_text,
            flags=re.IGNORECASE,
        ).strip()

        # Remove ordinal suffixes (1st → 1, 2nd → 2, etc.)
        cleaned = _ORDINAL_RE.sub(r"\1", cleaned)

        # Attempt dateutil first
        if _HAS_DATEUTIL:
            with contextlib.suppress(Exception):
                parsed = dateutil_parser.parse(cleaned, fuzzy=True)
                return parsed.date()

        # Manual fallback — try several patterns
        return self._manual_date_parse(cleaned)

    @staticmethod
    def _manual_date_parse(text: str) -> date | None:
        """Attempt to parse a date string using common format patterns."""
        month_map: dict[str, int] = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }

        # Pattern: "March 15 1997" or "March 15, 1997"
        match = re.match(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", text)
        if match:
            month_str, day_str, year_str = match.groups()
            month = month_map.get(month_str.lower())
            if month:
                with contextlib.suppress(ValueError):
                    return date(int(year_str), month, int(day_str))

        # Pattern: "15 March 1997"
        match = re.match(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", text)
        if match:
            day_str, month_str, year_str = match.groups()
            month = month_map.get(month_str.lower())
            if month:
                with contextlib.suppress(ValueError):
                    return date(int(year_str), month, int(day_str))

        # Pattern: MM/DD/YY or MM/DD/YYYY
        match = re.match(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", text)
        if match:
            m_str, d_str, y_str = match.groups()
            year = int(y_str)
            if year < 100:
                year += 1900 if year > 50 else 2000
            with contextlib.suppress(ValueError):
                return date(year, int(m_str), int(d_str))

        # Pattern: DD.MM.YYYY
        match = re.match(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", text)
        if match:
            d_str, m_str, y_str = match.groups()
            with contextlib.suppress(ValueError):
                return date(int(y_str), int(m_str), int(d_str))

        # Pattern: YYYY-MM-DD (ISO)
        match = re.match(r"(\d{4})-(\d{2})-(\d{2})", text)
        if match:
            y_str, m_str, d_str = match.groups()
            with contextlib.suppress(ValueError):
                return date(int(y_str), int(m_str), int(d_str))

        return None

    @staticmethod
    def _extract_city_hint(venue_text: str) -> str | None:
        """Attempt to extract a city name from venue text.

        Many flier venues include the city after a comma or dash:
        ``"The Warehouse, Chicago"`` → ``"Chicago"``.
        """
        for separator in [",", " - ", " — ", " – "]:
            if separator in venue_text:
                parts = venue_text.split(separator)
                candidate = parts[-1].strip()
                if candidate and len(candidate) > 2:
                    return candidate
        return None

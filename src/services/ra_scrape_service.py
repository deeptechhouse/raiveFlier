"""Orchestrates the full RA.co event scrape across cities and date ranges.

Manages the scrape lifecycle: iterates through cities and monthly date
ranges, saves intermediate JSON checkpoints for resume capability, and
writes human-readable corpus text files for RAG ingestion.

Usage via CLI::

    python -m src.cli.scrape_ra scrape --city chicago --start-year 2016
    python -m src.cli.scrape_ra scrape --all --start-year 2016
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Callable

import structlog

from src.models.ra_event import RAEvent, RAScrapeProgress
from src.providers.event.ra_graphql_provider import (
    CITY_DISPLAY_NAMES,
    RA_AREA_IDS,
    RAGraphQLProvider,
)

logger = structlog.get_logger(logger_name=__name__)

_CHECKPOINT_DIR = "data/ra_scrape"
_CORPUS_DIR = "data/reference_corpus"
_EARLIEST_YEAR = 2003


class RAScrapeService:
    """Orchestrates a full historical RA.co event scrape with checkpointing.

    Parameters
    ----------
    provider:
        The :class:`RAGraphQLProvider` handling HTTP requests.
    checkpoint_dir:
        Directory for intermediate JSON files (default: ``data/ra_scrape/``).
    corpus_dir:
        Directory for generated corpus text files (default: ``data/reference_corpus/``).
    """

    def __init__(
        self,
        provider: RAGraphQLProvider,
        checkpoint_dir: str = _CHECKPOINT_DIR,
        corpus_dir: str = _CORPUS_DIR,
    ) -> None:
        self._provider = provider
        self._checkpoint_dir = Path(checkpoint_dir)
        self._corpus_dir = Path(corpus_dir)
        self._logger = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def scrape_city(
        self,
        city: str,
        start_year: int = _EARLIEST_YEAR,
        end_year: int | None = None,
        on_progress: Callable[[str, int, int, int], None] | None = None,
    ) -> list[RAEvent]:
        """Scrape all events for a single city, month by month.

        Saves a checkpoint after each month.  Resumes from the last
        checkpoint if one exists.

        Parameters
        ----------
        city:
            City key (must exist in ``RA_AREA_IDS``).
        start_year:
            First year to scrape (default 2003).
        end_year:
            Last year to scrape (default: current year).
        on_progress:
            Callback ``(city, events_so_far, year, month)``.

        Returns
        -------
        list[RAEvent]
            All events scraped for this city.
        """
        if end_year is None:
            end_year = date.today().year

        city_key = city.lower().replace(" ", "_")
        area_id = RA_AREA_IDS.get(city_key)
        if area_id is None:
            self._logger.error("ra_unknown_city", city=city_key)
            return []

        display_name = CITY_DISPLAY_NAMES.get(city_key, city.title())

        # Load existing events (shared across all date ranges for this city).
        existing_events = self._load_events(city_key)

        # Load progress for THIS specific date range.
        progress = self._load_progress(city_key, start_year, end_year)

        # Determine resume point within our date range.
        resume_year = start_year
        resume_month = 1
        if progress and progress.last_completed_year:
            resume_year = progress.last_completed_year
            resume_month = (progress.last_completed_month or 12) + 1
            if resume_month > 12:
                resume_year += 1
                resume_month = 1

        if progress and progress.is_complete:
            self._logger.info(
                "ra_range_already_complete",
                city=city_key,
                start_year=start_year,
                end_year=end_year,
                events=len(existing_events),
            )
            return existing_events

        all_events = list(existing_events)

        # Track seen RA event IDs to avoid duplicates when resuming.
        seen_ids: set[str] = {e.ra_id for e in all_events}

        for year in range(resume_year, end_year + 1):
            month_start = resume_month if year == resume_year else 1

            # Don't scrape future months.
            today = date.today()
            month_end = 12
            if year == today.year:
                month_end = today.month

            for month in range(month_start, month_end + 1):
                start_date = f"{year}-{month:02d}-01"
                if month == 12:
                    end_date = f"{year}-12-31"
                else:
                    end_date = f"{year}-{month + 1:02d}-01"

                try:
                    events = await self._provider.fetch_all_events(
                        area_id=area_id,
                        start_date=start_date,
                        end_date=end_date,
                        city=display_name,
                    )
                except Exception as exc:
                    self._logger.error(
                        "ra_month_failed",
                        city=city_key,
                        year=year,
                        month=month,
                        error=str(exc),
                    )
                    events = []

                # Deduplicate — process one at a time to handle
                # duplicates within the same month's response.
                new_events: list[RAEvent] = []
                for e in events:
                    if e.ra_id not in seen_ids:
                        seen_ids.add(e.ra_id)
                        new_events.append(e)
                all_events.extend(new_events)

                # Checkpoint.
                self._save_events(city_key, all_events)
                self._save_progress(
                    city_key, area_id, len(all_events), year, month,
                    start_year=start_year, end_year=end_year,
                )

                if on_progress:
                    on_progress(city_key, len(all_events), year, month)

                self._logger.info(
                    "ra_month_complete",
                    city=city_key,
                    year=year,
                    month=month,
                    events_this_month=len(new_events),
                    total_events=len(all_events),
                )

        # Mark this date range complete.
        self._save_progress(
            city_key, area_id, len(all_events),
            end_year, 12, is_complete=True,
            start_year=start_year, end_year=end_year,
        )

        self._logger.info(
            "ra_city_complete",
            city=city_key,
            total_events=len(all_events),
        )
        return all_events

    async def scrape_all_cities(
        self,
        cities: list[str] | None = None,
        start_year: int = _EARLIEST_YEAR,
        end_year: int | None = None,
        on_progress: Callable[[str, int, int, int], None] | None = None,
    ) -> dict[str, list[RAEvent]]:
        """Scrape events for all (or specified) cities sequentially.

        Cities are processed one at a time to avoid multiplying request
        rates to RA.

        Returns
        -------
        dict[str, list[RAEvent]]
            Mapping of city key to scraped events.
        """
        if cities is None:
            cities = list(RA_AREA_IDS.keys())

        results: dict[str, list[RAEvent]] = {}
        for city in cities:
            city_key = city.lower().replace(" ", "_")
            self._logger.info("ra_scrape_city_start", city=city_key)

            events = await self.scrape_city(
                city=city_key,
                start_year=start_year,
                end_year=end_year,
                on_progress=on_progress,
            )
            results[city_key] = events

            self._logger.info(
                "ra_scrape_city_done",
                city=city_key,
                event_count=len(events),
            )

        return results

    def generate_corpus_file(self, city: str) -> Path | None:
        """Generate a corpus text file from scraped JSON data.

        Reads the checkpoint JSON for *city* and writes a structured
        ``.txt`` file to ``data/reference_corpus/ra_events_{city}.txt``.

        Returns the path to the generated file, or ``None`` if no data.
        """
        city_key = city.lower().replace(" ", "_")
        events = self._load_events(city_key)
        if not events:
            self._logger.warning("ra_corpus_no_data", city=city_key)
            return None

        display_name = CITY_DISPLAY_NAMES.get(city_key, city.title())
        sorted_events = sorted(events, key=lambda e: e.event_date)

        lines: list[str] = []
        lines.append(f"Resident Advisor Event History: {display_name}")
        lines.append("")
        lines.append("Source: Resident Advisor (ra.co)")
        if sorted_events:
            first_year = sorted_events[0].event_date.year
            last_year = sorted_events[-1].event_date.year
            lines.append(f"Coverage: {first_year}-{last_year}")
        lines.append(f"Total events: {len(sorted_events):,}")
        lines.append("")

        for event in sorted_events:
            lines.append("---")
            lines.append("")
            lines.append(f"Event: {event.title}")
            lines.append(f"Date: {event.event_date.isoformat()}")

            if event.venue:
                venue_str = event.venue.name
                if event.venue.area_name:
                    venue_str += f" ({event.venue.area_name})"
                lines.append(f"Venue: {venue_str}")

            if event.artists:
                artist_names = ", ".join(a.name for a in event.artists)
                lines.append(f"Artists: {artist_names}")

            if event.promoters:
                promoter_names = ", ".join(p.name for p in event.promoters)
                lines.append(f"Promoter: {promoter_names}")

            lines.append(f"City: {display_name}")

            if event.content_url:
                lines.append(f"URL: https://ra.co{event.content_url}")

            if event.attending > 0:
                lines.append(f"Attending: {event.attending:,}")

            lines.append("")

        self._corpus_dir.mkdir(parents=True, exist_ok=True)
        corpus_path = self._corpus_dir / f"ra_events_{city_key}.txt"
        corpus_path.write_text("\n".join(lines), encoding="utf-8")

        self._logger.info(
            "ra_corpus_generated",
            city=city_key,
            events=len(sorted_events),
            path=str(corpus_path),
        )
        return corpus_path

    def get_scrape_status(self) -> list[dict]:
        """Return scrape progress for all known cities.

        Each city gets one entry per date-range progress file (Phase A,
        Phase B, etc.).  Cities with no progress files appear once with
        zeroed-out fields.

        Returns
        -------
        list[dict]
            One dict per city/range with keys: ``city``, ``area_id``,
            ``events``, ``range``, ``last_year``, ``last_month``,
            ``complete``.
        """
        status: list[dict] = []
        for city_key in RA_AREA_IDS:
            area_id = RA_AREA_IDS[city_key]
            pattern = f"ra_progress_{city_key}_*.json"
            progress_files = sorted(self._checkpoint_dir.glob(pattern))

            if not progress_files:
                status.append({
                    "city": city_key,
                    "area_id": area_id,
                    "events": 0,
                    "range": "",
                    "last_year": 0,
                    "last_month": 0,
                    "complete": False,
                })
                continue

            event_count = len(self._load_events(city_key))

            for pf in progress_files:
                try:
                    prog = RAScrapeProgress.model_validate_json(
                        pf.read_text(encoding="utf-8")
                    )
                    status.append({
                        "city": city_key,
                        "area_id": area_id,
                        "events": event_count,
                        "range": (
                            f"{prog.scrape_start_year}-"
                            f"{prog.scrape_end_year}"
                        ),
                        "last_year": prog.last_completed_year or 0,
                        "last_month": prog.last_completed_month or 0,
                        "complete": prog.is_complete,
                    })
                except Exception:
                    self._logger.warning(
                        "ra_progress_parse_failed",
                        city=city_key,
                        path=str(pf),
                    )
        return status

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _ensure_dir(self) -> None:
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _events_path(self, city: str) -> Path:
        return self._checkpoint_dir / f"ra_events_{city}.json"

    def _progress_path(self, city: str, start_year: int = 0, end_year: int = 0) -> Path:
        """Return the progress file path, keyed by city and date range."""
        if start_year and end_year:
            return self._checkpoint_dir / f"ra_progress_{city}_{start_year}_{end_year}.json"
        return self._checkpoint_dir / f"ra_progress_{city}.json"

    def _save_events(self, city: str, events: list[RAEvent]) -> None:
        """Write the full event list for *city* to JSON."""
        self._ensure_dir()
        path = self._events_path(city)
        data = [e.model_dump(mode="json") for e in events]
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def _load_events(self, city: str) -> list[RAEvent]:
        """Load previously scraped events from the checkpoint JSON.

        Deduplicates by ``ra_id`` to clean up data from before the
        in-scrape dedup fix.
        """
        path = self._events_path(city)
        if not path.exists():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            seen: set[str] = set()
            events: list[RAEvent] = []
            for item in raw:
                evt = RAEvent.model_validate(item)
                if evt.ra_id not in seen:
                    seen.add(evt.ra_id)
                    events.append(evt)
            return events
        except Exception:
            self._logger.warning("ra_checkpoint_load_failed", city=city)
            return []

    def _save_progress(
        self,
        city: str,
        area_id: int,
        total: int,
        year: int,
        month: int,
        is_complete: bool = False,
        start_year: int = 0,
        end_year: int = 0,
    ) -> None:
        """Write progress checkpoint for a city + date range."""
        self._ensure_dir()
        now = datetime.now().isoformat()
        progress = RAScrapeProgress(
            city=city,
            area_id=area_id,
            scrape_start_year=start_year,
            scrape_end_year=end_year,
            total_events_scraped=total,
            last_completed_year=year,
            last_completed_month=month,
            is_complete=is_complete,
            started_at=now,
            updated_at=now,
        )
        self._progress_path(city, start_year, end_year).write_text(
            progress.model_dump_json(indent=2), encoding="utf-8"
        )

    def _load_progress(
        self, city: str, start_year: int = 0, end_year: int = 0
    ) -> RAScrapeProgress | None:
        """Load the progress checkpoint for a city + date range."""
        path = self._progress_path(city, start_year, end_year)
        if not path.exists():
            return None
        try:
            return RAScrapeProgress.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception:
            return None

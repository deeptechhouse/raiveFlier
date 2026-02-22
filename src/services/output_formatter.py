"""Structured output formatting for the raiveFlier analysis pipeline.

Transforms :class:`PipelineState` into clean JSON-serialisable dictionaries
optimised for frontend consumption.  Two output modes are supported:

- **Full analysis** — complete structured output with all research, citations,
  and interconnection data.
- **Summary** — abbreviated key-findings view suitable for status endpoints.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.models.analysis import Citation, InterconnectionMap
from src.models.entities import ConfidenceLevel
from src.models.flier import ExtractedEntities, OCRResult
from src.models.pipeline import PipelineState
from src.models.research import DateContext, ResearchResult
from src.utils.logging import get_logger


def _confidence_level(score: float) -> str:
    """Map a numeric confidence score to a qualitative level string."""
    if score >= 0.8:
        return ConfidenceLevel.HIGH.value
    if score >= 0.5:
        return ConfidenceLevel.MEDIUM.value
    if score >= 0.3:
        return ConfidenceLevel.LOW.value
    return ConfidenceLevel.UNCERTAIN.value


class OutputFormatter:
    """Transforms pipeline state into structured output dictionaries.

    All public methods return plain ``dict`` objects that are directly
    JSON-serialisable (no Pydantic models or datetime objects remain).
    """

    def __init__(self) -> None:
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def format_full_analysis(self, state: PipelineState) -> dict[str, Any]:
        """Transform a :class:`PipelineState` into a complete JSON structure.

        The output is optimised for frontend rendering and includes every
        section of the analysis pipeline.

        Parameters
        ----------
        state:
            The pipeline state to format.

        Returns
        -------
        dict[str, Any]
            Structured output dictionary.
        """
        warnings = self._collect_warnings(state)

        result: dict[str, Any] = {
            "session_id": state.session_id,
            "flier": self._format_flier(state),
            "ocr": self._format_ocr(state.ocr_result),
            "entities": self._format_entities(state.confirmed_entities or state.extracted_entities),
            "research": self._format_research(state.research_results),
            "interconnections": self._format_interconnections(state.interconnection_map),
            "citations": self._format_citations(state),
            "warnings": warnings,
            "completed_at": self._iso_timestamp(state.completed_at),
        }

        self._logger.info(
            "full_analysis_formatted",
            session_id=state.session_id,
            sections=list(result.keys()),
        )

        return result

    def format_summary(self, state: PipelineState) -> dict[str, Any]:
        """Produce an abbreviated summary of key findings.

        Suitable for status endpoints where the full analysis payload
        would be too large.

        Parameters
        ----------
        state:
            The pipeline state to summarise.

        Returns
        -------
        dict[str, Any]
            Abbreviated output dictionary.
        """
        entities = state.confirmed_entities or state.extracted_entities
        artist_names: list[str] = []
        if entities:
            artist_names = [a.text for a in entities.artists]

        venue_name: str | None = None
        if entities and entities.venue:
            venue_name = entities.venue.text

        relationship_count = 0
        pattern_count = 0
        narrative: str | None = None
        if state.interconnection_map:
            relationship_count = len(state.interconnection_map.edges)
            pattern_count = len(state.interconnection_map.patterns)
            narrative = state.interconnection_map.narrative

        result: dict[str, Any] = {
            "session_id": state.session_id,
            "phase": state.current_phase.value,
            "progress_percent": state.progress_percent,
            "artists": artist_names,
            "venue": venue_name,
            "relationship_count": relationship_count,
            "pattern_count": pattern_count,
            "narrative_preview": (
                (narrative[:300] + "...") if narrative and len(narrative) > 300 else narrative
            ),
            "warnings_count": len(state.errors),
            "completed_at": self._iso_timestamp(state.completed_at),
        }

        self._logger.debug(
            "summary_formatted",
            session_id=state.session_id,
        )

        return result

    # ------------------------------------------------------------------
    # Private formatters
    # ------------------------------------------------------------------

    @staticmethod
    def _format_flier(state: PipelineState) -> dict[str, Any]:
        """Format the flier metadata section."""
        ocr = state.ocr_result
        return {
            "filename": None,
            "upload_time": state.started_at.isoformat() if state.started_at else None,
            "has_ocr": ocr is not None,
        }

    @staticmethod
    def _format_ocr(ocr: OCRResult | None) -> dict[str, Any] | None:
        """Format the OCR result section."""
        if ocr is None:
            return None
        return {
            "raw_text": ocr.raw_text,
            "confidence": ocr.confidence,
            "provider": ocr.provider_used,
        }

    @staticmethod
    def _format_entities(
        entities: ExtractedEntities | None,
    ) -> dict[str, Any]:
        """Format the extracted entities section."""
        if entities is None:
            return {
                "artists": [],
                "venue": None,
                "date": None,
                "promoter": None,
                "event_name": None,
            }

        artists = [
            {
                "name": a.text,
                "confidence": a.confidence,
                "confidence_level": _confidence_level(a.confidence),
            }
            for a in entities.artists
        ]

        venue: dict[str, Any] | None = None
        if entities.venue:
            venue = {
                "name": entities.venue.text,
                "confidence": entities.venue.confidence,
            }

        date_info: dict[str, Any] | None = None
        if entities.date:
            date_info = {
                "text": entities.date.text,
                "parsed_date": None,
            }

        promoter: dict[str, Any] | None = None
        if entities.promoter:
            promoter = {
                "name": entities.promoter.text,
                "confidence": entities.promoter.confidence,
            }

        event_name: dict[str, Any] | None = None
        if entities.event_name:
            event_name = {
                "name": entities.event_name.text,
                "confidence": entities.event_name.confidence,
            }

        return {
            "artists": artists,
            "venue": venue,
            "date": date_info,
            "promoter": promoter,
            "event_name": event_name,
        }

    def _format_research(self, results: list[ResearchResult]) -> dict[str, Any]:
        """Format the research results section."""
        artists: list[dict[str, Any]] = []
        venue_data: dict[str, Any] | None = None
        promoter_data: dict[str, Any] | None = None
        date_context_data: dict[str, Any] | None = None

        for result in results:
            if result.artist:
                artists.append(self._format_artist_research(result))
            if result.venue:
                venue_data = self._format_venue_research(result)
            if result.promoter:
                promoter_data = self._format_promoter_research(result)
            if result.date_context:
                date_context_data = self._format_date_context(result.date_context)

        return {
            "artists": artists,
            "venue": venue_data,
            "promoter": promoter_data,
            "date_context": date_context_data,
        }

    @staticmethod
    def _format_artist_research(result: ResearchResult) -> dict[str, Any]:
        """Format a single artist research result."""
        artist = result.artist
        if artist is None:
            return {"name": result.entity_name, "confidence": result.confidence}

        discogs_url: str | None = None
        if artist.discogs_id:
            discogs_url = f"https://www.discogs.com/artist/{artist.discogs_id}"

        musicbrainz_url: str | None = None
        if artist.musicbrainz_id:
            musicbrainz_url = f"https://musicbrainz.org/artist/{artist.musicbrainz_id}"

        labels = list({lb.name for lb in artist.labels})

        return {
            "name": artist.name,
            "discogs_url": discogs_url,
            "musicbrainz_url": musicbrainz_url,
            "releases_count": len(artist.releases),
            "labels": labels,
            "confidence": result.confidence,
        }

    @staticmethod
    def _format_venue_research(result: ResearchResult) -> dict[str, Any] | None:
        """Format venue research data."""
        venue = result.venue
        if venue is None:
            return None

        articles = [
            {
                "title": a.title,
                "source": a.source,
                "url": a.url,
            }
            for a in venue.articles[:20]
        ]

        return {
            "name": venue.name,
            "history": venue.history,
            "notable_events": venue.notable_events[:20],
            "articles": articles,
        }

    @staticmethod
    def _format_promoter_research(result: ResearchResult) -> dict[str, Any] | None:
        """Format promoter research data."""
        promoter = result.promoter
        if promoter is None:
            return None

        articles = [
            {
                "title": a.title,
                "source": a.source,
                "url": a.url,
            }
            for a in promoter.articles[:20]
        ]

        return {
            "name": promoter.name,
            "event_history": promoter.event_history[:20],
            "articles": articles,
        }

    @staticmethod
    def _format_date_context(dc: DateContext) -> dict[str, Any]:
        """Format date context research data."""
        return {
            "scene": dc.scene_context,
            "city": dc.city_context,
            "cultural": dc.cultural_context,
        }

    def _format_interconnections(self, imap: InterconnectionMap | None) -> dict[str, Any]:
        """Format the interconnection analysis section."""
        if imap is None:
            return {
                "relationships": [],
                "patterns": [],
                "narrative": None,
            }

        relationships = [
            {
                "source": edge.source,
                "target": edge.target,
                "type": edge.relationship_type,
                "details": edge.details,
                "citation": edge.citations[0].text if edge.citations else None,
                "confidence": edge.confidence,
            }
            for edge in imap.edges
        ]

        patterns = [
            {
                "type": p.pattern_type,
                "description": p.description,
                "entities": p.involved_entities,
            }
            for p in imap.patterns
        ]

        return {
            "relationships": relationships,
            "patterns": patterns,
            "narrative": imap.narrative,
        }

    def _format_citations(self, state: PipelineState) -> list[dict[str, Any]]:
        """Collect and format all citations from the interconnection map."""
        if state.interconnection_map is None:
            return []

        return [self._format_single_citation(c) for c in state.interconnection_map.citations]

    @staticmethod
    def _format_single_citation(citation: Citation) -> dict[str, Any]:
        """Format a single citation for the output payload."""
        return {
            "text": citation.text,
            "source": citation.source_name,
            "url": citation.source_url,
            "tier": citation.tier,
            "accessible": None,
        }

    @staticmethod
    def _collect_warnings(state: PipelineState) -> list[str]:
        """Collect warning messages from pipeline errors and research results."""
        warnings: list[str] = []

        for err in state.errors:
            warnings.append(f"[{err.phase.value}] {err.message}")

        for result in state.research_results:
            for w in result.warnings:
                warnings.append(f"[{result.entity_name}] {w}")

        return warnings

    @staticmethod
    def _iso_timestamp(dt: datetime | None) -> str | None:
        """Convert a datetime to an ISO-format string, or return ``None``."""
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)  # noqa: UP017
        return dt.isoformat()

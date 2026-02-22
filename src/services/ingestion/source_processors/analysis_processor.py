"""Source processor that converts completed pipeline analysis results into chunks.

Enables the **feedback loop**: every finished flier analysis is re-ingested
into the vector store so that future analyses benefit from accumulated knowledge.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import structlog

from src.models.rag import DocumentChunk

if TYPE_CHECKING:
    from src.models.pipeline import PipelineState

logger = structlog.get_logger(logger_name=__name__)


class AnalysisProcessor:
    """Converts a completed :class:`PipelineState` into :class:`DocumentChunk` objects.

    The processor creates summary chunks for:

    * Each researched artist (discography highlights, appearances, articles found).
    * Venue and promoter summaries (if present).
    * The interconnection narrative produced by Phase 5.

    All chunks carry ``source_type = "analysis"`` and ``citation_tier = 5``.
    """

    def process(self, pipeline_state: PipelineState) -> list[DocumentChunk]:
        """Generate document chunks from a completed pipeline analysis.

        Parameters
        ----------
        pipeline_state:
            A pipeline session state that has reached the OUTPUT phase.

        Returns
        -------
        list[DocumentChunk]
            Chunks summarising artist research, venue/promoter context,
            and the interconnection narrative.
        """
        source_id = f"analysis-{pipeline_state.session_id}"
        chunks: list[DocumentChunk] = []
        entity_names: list[str] = []

        # -- Artist research chunks ----------------------------------------
        for result in pipeline_state.research_results:
            if result.artist:
                artist = result.artist
                entity_names.append(artist.name)
                text = self._build_artist_summary(artist)
                if text.strip():
                    chunks.append(
                        DocumentChunk(
                            chunk_id=str(uuid.uuid4()),
                            text=text,
                            source_id=source_id,
                            source_title=f"Analysis: {artist.name}",
                            source_type="analysis",
                            citation_tier=5,
                            entity_tags=[artist.name, *artist.aliases],
                        )
                    )

            if result.venue:
                venue = result.venue
                entity_names.append(venue.name)
                text = self._build_venue_summary(venue)
                if text.strip():
                    geo_tags: list[str] = []
                    if venue.city:
                        geo_tags.append(venue.city)
                    if venue.country:
                        geo_tags.append(venue.country)
                    chunks.append(
                        DocumentChunk(
                            chunk_id=str(uuid.uuid4()),
                            text=text,
                            source_id=source_id,
                            source_title=f"Analysis: {venue.name}",
                            source_type="analysis",
                            citation_tier=5,
                            entity_tags=[venue.name],
                            geographic_tags=geo_tags,
                        )
                    )

            if result.promoter:
                promoter = result.promoter
                entity_names.append(promoter.name)
                text = self._build_promoter_summary(promoter)
                if text.strip():
                    chunks.append(
                        DocumentChunk(
                            chunk_id=str(uuid.uuid4()),
                            text=text,
                            source_id=source_id,
                            source_title=f"Analysis: {promoter.name}",
                            source_type="analysis",
                            citation_tier=5,
                            entity_tags=[promoter.name],
                        )
                    )

        # -- Interconnection narrative chunk --------------------------------
        imap = pipeline_state.interconnection_map
        if imap and imap.narrative:
            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=imap.narrative,
                    source_id=source_id,
                    source_title=f"Interconnection: session {pipeline_state.session_id}",
                    source_type="analysis",
                    citation_tier=5,
                    entity_tags=list(set(entity_names)),
                )
            )

        logger.info(
            "analysis_processed",
            session_id=pipeline_state.session_id,
            chunks_created=len(chunks),
            entities=entity_names,
        )
        return chunks

    # ------------------------------------------------------------------
    # Summary builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_artist_summary(artist) -> str:  # noqa: ANN001 – avoids circular
        """Build a text summary of an artist's research data."""
        parts: list[str] = [f"Artist: {artist.name}"]

        if artist.aliases:
            parts.append(f"Also known as: {', '.join(artist.aliases)}")

        if artist.profile_summary:
            parts.append(artist.profile_summary)

        if artist.releases:
            release_lines = []
            for r in artist.releases[:15]:
                line = f"  - {r.title}"
                if r.label:
                    line += f" ({r.label})"
                if r.year:
                    line += f" [{r.year}]"
                release_lines.append(line)
            parts.append("Key releases:\n" + "\n".join(release_lines))


        return "\n\n".join(parts)

    @staticmethod
    def _build_venue_summary(venue) -> str:  # noqa: ANN001
        """Build a text summary of a venue's research data."""
        parts: list[str] = [f"Venue: {venue.name}"]
        if venue.city:
            parts.append(
                f"Location: {venue.city}" + (f", {venue.country}" if venue.country else "")
            )
        if venue.history:
            parts.append(venue.history)
        if venue.cultural_significance:
            parts.append(f"Cultural significance: {venue.cultural_significance}")
        if venue.notable_events:
            parts.append("Notable events: " + ", ".join(venue.notable_events[:10]))
        return "\n\n".join(parts)

    @staticmethod
    def _build_promoter_summary(promoter) -> str:  # noqa: ANN001
        """Build a text summary of a promoter's research data."""
        parts: list[str] = [f"Promoter: {promoter.name}"]
        if promoter.event_history:
            parts.append("Events: " + ", ".join(promoter.event_history[:10]))
        if promoter.affiliated_artists:
            parts.append("Affiliated artists: " + ", ".join(promoter.affiliated_artists[:10]))
        if promoter.affiliated_venues:
            parts.append("Affiliated venues: " + ", ".join(promoter.affiliated_venues[:10]))
        return "\n\n".join(parts)

"""Source processor that converts RA.co event data into DocumentChunks.

Generates structured text from :class:`~src.models.ra_event.RAEvent` models
(scraped via the RA CLI) and wraps them in :class:`~src.models.rag.DocumentChunk`
objects.  Events are sorted by date and grouped into batches of ~8 per chunk
to stay within the ~500-token embedding target.

Key optimization: Because RA events are already structured data (not free text),
entity tags (artist names, venue names, promoter names) and geographic tags
(city) are **pre-extracted directly from the model fields** -- no LLM call
needed.  This allows the expensive MetadataExtractor tagging step to be
skipped with ``--skip-tagging`` during bulk event ingestion.

Events are assigned ``citation_tier = 3`` (event listings) and
``source_type = "event"``.
"""

from __future__ import annotations

import hashlib
import uuid

import structlog

from src.models.ra_event import RAEvent
from src.models.rag import DocumentChunk

logger = structlog.get_logger(logger_name=__name__)

# Number of events per document chunk -- tuned to produce ~500-token chunks.
_EVENTS_PER_CHUNK = 8


class RAEventProcessor:
    """Converts batches of :class:`RAEvent` objects into :class:`DocumentChunk` objects.

    Parameters
    ----------
    events_per_chunk:
        Number of events to group per document chunk (default 8).
    """

    def __init__(self, events_per_chunk: int = _EVENTS_PER_CHUNK) -> None:
        self._events_per_chunk = events_per_chunk

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_events(
        self,
        events: list[RAEvent],
        city: str,
    ) -> list[DocumentChunk]:
        """Convert a list of RA events into document chunks.

        Events are sorted by date, grouped into batches, and each batch
        becomes a single ``DocumentChunk`` with pre-populated entity and
        geographic tags.

        Parameters
        ----------
        events:
            Scraped ``RAEvent`` objects for a single city.
        city:
            City name used for geographic tagging and ``source_title``.

        Returns
        -------
        list[DocumentChunk]
            One chunk per batch of events, ready for the ingestion pipeline.
        """
        if not events:
            return []

        sorted_events = sorted(events, key=lambda e: e.event_date)

        chunks: list[DocumentChunk] = []
        for i in range(0, len(sorted_events), self._events_per_chunk):
            batch = sorted_events[i : i + self._events_per_chunk]
            chunk = self._build_chunk(batch, city)
            if chunk:
                chunks.append(chunk)

        logger.info(
            "ra_events_processed",
            city=city,
            events=len(events),
            chunks=len(chunks),
        )
        return chunks

    @staticmethod
    def event_to_text(event: RAEvent) -> str:
        """Render a single ``RAEvent`` as structured text.

        The format uses labeled fields on separate lines, which works
        well with the existing :class:`TextChunker` paragraph splitting.
        """
        lines: list[str] = []
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

        if event.city:
            lines.append(f"City: {event.city}")

        if event.content_url:
            lines.append(f"URL: https://ra.co{event.content_url}")

        if event.attending > 0:
            lines.append(f"Attending: {event.attending:,}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_chunk(
        self,
        events: list[RAEvent],
        city: str,
    ) -> DocumentChunk | None:
        """Build a single :class:`DocumentChunk` from a batch of events."""
        if not events:
            return None

        text_parts = [self.event_to_text(e) for e in events]
        text = "\n\n---\n\n".join(text_parts)

        # Pre-extract tags from structured data.
        entity_tags: set[str] = set()
        for e in events:
            for a in e.artists:
                entity_tags.add(a.name)
            if e.venue:
                entity_tags.add(e.venue.name)
            for p in e.promoters:
                entity_tags.add(p.name)

        first_date = events[0].event_date
        last_date = events[-1].event_date

        source_id = hashlib.sha256(
            f"ra_events_{city}_{first_date}_{last_date}_{len(events)}".encode()
        ).hexdigest()

        return DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            text=text,
            source_id=source_id,
            source_title=f"RA Events: {city.title()} ({first_date} to {last_date})",
            source_type="event",
            citation_tier=3,
            publication_date=last_date,
            entity_tags=sorted(entity_tags),
            geographic_tags=[city.title()],
        )

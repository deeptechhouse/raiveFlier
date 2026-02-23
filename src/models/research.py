"""Research result models for the raiveFlier pipeline.

Defines Pydantic v2 models for date context and research results produced
by the entity research phase. All models use frozen config to enforce immutability.

These models wrap the output of Phase 2 (RESEARCH). For each entity confirmed
by the user, the research service dispatches to a specialist researcher:
    - ARTIST  → artist_researcher.py  → populates ResearchResult.artist
    - VENUE   → venue_researcher.py   → populates ResearchResult.venue
    - PROMOTER → promoter_researcher.py → populates ResearchResult.promoter
    - DATE    → date_context_researcher.py → populates ResearchResult.date_context
    - EVENT   → event_name_researcher.py → populates ResearchResult.event_history

Each ResearchResult carries exactly ONE populated entity field (the rest are None).
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from src.models.entities import (
    ArticleReference,
    Artist,
    EntityType,
    EventSeriesHistory,
    Promoter,
    Venue,
)


# ---------------------------------------------------------------------------
# DateContext — enriches a flier date with scene/cultural context.
# ---------------------------------------------------------------------------
class DateContext(BaseModel):
    """Contextual information about the date/era of an event on a flier.

    When a flier has a date, the date_context_researcher.py uses web search
    and the RAG corpus to describe what was happening in the music scene at
    that time and place. This transforms a bare date like "15 Nov 2003" into
    rich context like "peak of the UK garage-to-grime transition in London."
    """

    model_config = ConfigDict(frozen=True)

    # The parsed event date from the flier.
    event_date: date
    # What was happening in the broader electronic music scene at this time.
    scene_context: str | None = None
    # What was happening in the specific city's scene at this time.
    city_context: str | None = None
    # Broader cultural context (e.g. political events, venue closures, movements).
    cultural_context: str | None = None
    # Other notable events happening around the same date/city.
    nearby_events: list[str] = Field(default_factory=list)
    # Articles and references used to construct this context.
    sources: list[ArticleReference] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# ResearchResult — the unified output container for any entity type.
# ---------------------------------------------------------------------------
class ResearchResult(BaseModel):
    """The result of researching a single entity extracted from a flier.

    This is a "union-like" model: exactly one of (artist, venue, promoter,
    date_context, event_history) will be populated based on entity_type.
    Using a single model simplifies the pipeline — PipelineState holds a
    flat list[ResearchResult] regardless of entity mix.

    The research_service.py creates one ResearchResult per confirmed entity
    and appends them to PipelineState.research_results during the RESEARCH phase.
    """

    model_config = ConfigDict(frozen=True)

    # What type of entity was researched — determines which field below is set.
    entity_type: EntityType
    # The entity name as it appeared on the flier (e.g. "DJ Shadow").
    entity_name: str
    # Exactly one of these will be populated (the rest remain None):
    artist: Artist | None = None                    # Set when entity_type == ARTIST
    venue: Venue | None = None                      # Set when entity_type == VENUE
    promoter: Promoter | None = None                # Set when entity_type == PROMOTER
    date_context: DateContext | None = None          # Set when entity_type == DATE
    event_history: EventSeriesHistory | None = None  # Set when entity_type == EVENT
    # Which data sources were queried (e.g. ["discogs", "musicbrainz", "duckduckgo"]).
    # Useful for debugging why certain information is missing.
    sources_consulted: list[str] = Field(default_factory=list)
    # Overall confidence in the research result (aggregated from sub-sources).
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # Non-fatal issues encountered during research (e.g. "Discogs rate limit hit",
    # "Multiple artists with this name found — used best match").
    warnings: list[str] = Field(default_factory=list)

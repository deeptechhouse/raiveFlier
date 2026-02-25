"""Pydantic request/response schemas for the RaiveFlier API.

Defines the public contract for all REST endpoints — upload, confirmation,
status polling, full results, health, and provider listing.

# ─── HOW SCHEMAS WORK (Junior Developer Guide) ────────────────────────
#
# These Pydantic models define the *shape* of every HTTP request body
# and response body in the API.  FastAPI uses them for:
#
#   1. **Validation** — Incoming JSON is automatically validated against
#      the schema.  Invalid requests get a 422 error with details.
#   2. **Serialization** — Outgoing objects are automatically converted
#      to JSON matching the schema (via response_model=...).
#   3. **Documentation** — FastAPI generates OpenAPI/Swagger docs from
#      these schemas automatically (visible at /docs).
#
# Convention: Request schemas end with "Request", response schemas
# end with "Response".  Field(...) adds constraints (min_length, ge/le)
# and descriptions for the API docs.
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.models.flier import ExtractedEntities


class EntityInput(BaseModel):
    """A single entity submitted by the user during confirmation.

    This is the frontend's representation of an entity — just a name
    and type string.  Simpler than the internal ExtractedEntity model.
    """

    name: str
    entity_type: str


class DuplicateMatch(BaseModel):
    """Metadata about a previously analyzed flier that visually matches the upload."""

    previous_session_id: str
    similarity: float = Field(ge=0.0, le=1.0, description="1.0 = exact visual match")
    analyzed_at: str
    artists: list[str] = Field(default_factory=list)
    venue: str | None = None
    event_name: str | None = None
    event_date: str | None = None
    hamming_distance: int = Field(description="Perceptual hash Hamming distance (0 = identical)")
    times_analyzed: int = Field(default=1, description="How many times this flier image has been analyzed")


class FlierUploadResponse(BaseModel):
    """Response returned after uploading and processing a flier image."""

    session_id: str
    extracted_entities: ExtractedEntities
    ocr_confidence: float
    provider_used: str
    duplicate_match: DuplicateMatch | None = None
    times_analyzed: int = Field(default=1, description="Total times this flier image has been analyzed (including this time)")


class ConfirmEntitiesRequest(BaseModel):
    """User-confirmed entities submitted to start the research pipeline."""

    artists: list[EntityInput] = Field(default_factory=list)
    venue: EntityInput | None = None
    date: EntityInput | None = None
    promoter: EntityInput | None = None
    event_name: EntityInput | None = None
    genre_tags: list[str] = Field(default_factory=list)
    ticket_price: str | None = None


class ConfirmResponse(BaseModel):
    """Response after confirming entities and starting research."""

    session_id: str
    status: str = "research_started"
    message: str


class PipelineStatusResponse(BaseModel):
    """Current pipeline progress status for a session."""

    session_id: str
    phase: str
    progress: float
    message: str | None = None
    errors: list[dict[str, Any]] = Field(default_factory=list)


class FlierAnalysisResponse(BaseModel):
    """Full analysis results for a pipeline session."""

    session_id: str
    status: str
    extracted_entities: ExtractedEntities | None = None
    research_results: list[dict[str, Any]] | None = None
    interconnection_map: dict[str, Any] | None = None
    completed_at: datetime | None = None


class HealthResponse(BaseModel):
    """Application health check response."""

    status: str
    version: str
    providers: dict[str, Any]


class ProvidersResponse(BaseModel):
    """List of configured service providers and their availability."""

    providers: list[dict[str, Any]]


class CorpusStatsResponse(BaseModel):
    """RAG corpus statistics.

    Includes genre and time period lists for populating frontend filter
    dropdowns dynamically from the actual corpus contents.
    """

    total_chunks: int = 0
    total_sources: int = 0
    sources_by_type: dict[str, int] = Field(default_factory=dict)
    entity_tag_count: int = 0
    geographic_tag_count: int = 0
    # New fields for dynamic filter population
    genre_tags: list[str] = Field(default_factory=list)
    time_periods: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error: str
    detail: str | None = None


class AskQuestionRequest(BaseModel):
    """User question about analysis results."""

    question: str = Field(..., min_length=1, max_length=1000)
    entity_type: str | None = None
    entity_name: str | None = None


class RelatedFact(BaseModel):
    """A contextual fact related to the flier's entities."""

    text: str
    category: str | None = None  # LABEL, HISTORY, SCENE, VENUE, ARTIST, CONNECTION
    entity_name: str | None = None


class AskQuestionResponse(BaseModel):
    """Response to a user question with answer, citations, and related facts."""

    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    related_facts: list[RelatedFact] = Field(default_factory=list)


class CorpusSearchRequest(BaseModel):
    """User query for corpus semantic search.

    Supports pagination via offset/page_size — the backend computes a full
    ranked candidate pool (up to top_k) then returns a page of results.
    New filters (genre, era, quality floor, relevance floor) are all optional
    with safe defaults so existing callers are unaffected.
    """

    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=50, ge=1, le=100)
    source_type: list[str] | None = Field(
        default=None,
        description='Filter by source types: "book", "article", "interview", etc.',
    )
    entity_tag: str | None = Field(
        default=None,
        description="Filter to chunks mentioning a specific entity name.",
    )
    geographic_tag: str | None = Field(
        default=None,
        description="Filter to chunks referencing a specific city/region.",
    )
    # --- New filters for enhanced corpus search ---
    genre_tags: list[str] | None = Field(
        default=None,
        description='Filter by genre tags: "techno", "house", "jungle", etc.',
    )
    time_period: str | None = Field(
        default=None,
        description='Filter by era/time period: "1990s", "1988-1992", or named era.',
    )
    min_citation_tier: int | None = Field(
        default=None,
        ge=1,
        le=6,
        description="Minimum citation tier (1=best, 6=unverified). Only return chunks at this tier or better.",
    )
    min_similarity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold. Results below this are excluded.",
    )
    # --- Pagination ---
    offset: int = Field(
        default=0,
        ge=0,
        description="Pagination offset — number of results to skip for load-more.",
    )
    page_size: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Number of results per page (used with offset for pagination).",
    )


class CorpusSearchChunk(BaseModel):
    """A single corpus search result chunk."""

    text: str
    source_title: str
    source_type: str
    author: str | None = None
    citation_tier: int = 6
    page_number: str | None = None
    similarity_score: float = 0.0
    formatted_citation: str = ""
    entity_tags: list[str] = Field(default_factory=list)
    entity_types: list[str] = Field(default_factory=list)
    geographic_tags: list[str] = Field(default_factory=list)
    genre_tags: list[str] = Field(default_factory=list)
    time_period: str | None = None


class CorpusSearchResponse(BaseModel):
    """Response containing corpus search results with pagination metadata.

    The backend computes the full ranked result set then returns a page
    controlled by offset/page_size.  The has_more flag tells the frontend
    whether a "Load More" button should be rendered.
    """

    query: str
    total_results: int
    results: list[CorpusSearchChunk] = Field(default_factory=list)
    # Pagination metadata — safe defaults preserve backward compatibility.
    offset: int = 0
    page_size: int = 20
    has_more: bool = False


class SubmitRatingRequest(BaseModel):
    """Submit a thumbs up/down rating for a result item."""

    item_type: str = Field(
        ...,
        description="ARTIST, VENUE, PROMOTER, DATE, EVENT, CONNECTION, PATTERN, QA, CORPUS, RELEASE, LABEL, RECOMMENDATION",
    )
    item_key: str = Field(..., min_length=1, max_length=500)
    rating: int = Field(..., description="+1 for thumbs up, -1 for thumbs down")


class RatingResponse(BaseModel):
    """Response after submitting a rating."""

    id: int
    session_id: str
    item_type: str
    item_key: str
    rating: int
    created_at: str
    updated_at: str


class SessionRatingsResponse(BaseModel):
    """All ratings for a session."""

    session_id: str
    ratings: list[RatingResponse] = Field(default_factory=list)
    total: int = 0


class RatingSummaryResponse(BaseModel):
    """Aggregate rating statistics."""

    total_ratings: int = 0
    positive: int = 0
    negative: int = 0
    by_type: dict[str, dict[str, int]] = Field(default_factory=dict)


class DismissConnectionRequest(BaseModel):
    """Request to dismiss a specific interconnection edge."""

    source: str
    target: str
    relationship_type: str
    reason: str | None = None


class DismissConnectionResponse(BaseModel):
    """Response after dismissing a connection."""

    session_id: str
    dismissed_count: int
    message: str


class RecommendedArtistResponse(BaseModel):
    """A single recommended artist."""

    artist_name: str
    genres: list[str] = Field(default_factory=list)
    reason: str = ""
    source_tier: str = "llm_suggestion"
    connection_strength: float = 0.5
    connected_to: list[str] = Field(default_factory=list)
    label_name: str | None = None
    event_name: str | None = None


class RecommendationsResponse(BaseModel):
    """Response containing artist recommendations based on flier analysis."""

    session_id: str
    recommendations: list[RecommendedArtistResponse] = Field(default_factory=list)
    flier_artists: list[str] = Field(default_factory=list)
    genres_analyzed: list[str] = Field(default_factory=list)
    total: int = 0
    is_partial: bool = False

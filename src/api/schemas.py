"""Pydantic request/response schemas for the RaiveFlier API.

Defines the public contract for all REST endpoints — upload, confirmation,
status polling, full results, health, and provider listing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.models.flier import ExtractedEntities


class EntityInput(BaseModel):
    """A single entity submitted by the user during confirmation."""

    name: str
    entity_type: str


class FlierUploadResponse(BaseModel):
    """Response returned after uploading and processing a flier image."""

    session_id: str
    extracted_entities: ExtractedEntities
    ocr_confidence: float
    provider_used: str


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
    """RAG corpus statistics."""

    total_chunks: int = 0
    total_sources: int = 0
    sources_by_type: dict[str, int] = Field(default_factory=dict)
    entity_tag_count: int = 0
    geographic_tag_count: int = 0


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
    """User query for corpus semantic search."""

    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=50)
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
    geographic_tags: list[str] = Field(default_factory=list)
    genre_tags: list[str] = Field(default_factory=list)


class CorpusSearchResponse(BaseModel):
    """Response containing corpus search results."""

    query: str
    total_results: int
    results: list[CorpusSearchChunk] = Field(default_factory=list)


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

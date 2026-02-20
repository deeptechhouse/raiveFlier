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

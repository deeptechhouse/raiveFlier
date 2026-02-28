"""Request and response schemas for the Rave Stories API.

# ─── ARCHITECTURE ROLE ───────────────────────────────────────────────
#
# Layer: API (Pydantic v2 schemas for request validation and response
#        serialization).
# Pattern: All schemas use ``frozen=True`` for immutability, matching
#          the convention used throughout the raiveFlier codebase.
#
# These schemas define the contract between the frontend and the
# StoryService.  They are separate from the domain models in
# src/models/story.py — domain models represent business state,
# while API schemas represent the HTTP request/response interface.
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.models.story import StoryStatus


# ─── Request schemas ──────────────────────────────────────────────────

class StoryMetadataRequest(BaseModel):
    """Metadata submitted with a story — at least one field required."""

    model_config = ConfigDict(frozen=True)

    event_name: str | None = Field(default=None, description="Name of the rave/event.")
    event_year: int | None = Field(default=None, ge=1980, le=2030, description="Year of the event.")
    city: str | None = Field(default=None, description="City where the event took place.")
    genre: str | None = Field(default=None, description="Primary genre.")
    promoter: str | None = Field(default=None, description="Promoter or crew.")
    artist: str | None = Field(default=None, description="Featured artist.")
    other: str | None = Field(default=None, description="Extra context.")


class SubmitStoryRequest(BaseModel):
    """Request body for submitting a text story."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(
        description="The story text (20–15,000 characters).",
        min_length=20,
        max_length=15000,
    )
    metadata: StoryMetadataRequest = Field(description="Event context (at least one field required).")


class SearchStoriesRequest(BaseModel):
    """Request body for semantic search across stories."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(description="Natural language search query.", min_length=3, max_length=500)
    limit: int = Field(default=10, ge=1, le=50, description="Max results to return.")


# ─── Response schemas ─────────────────────────────────────────────────

class StoryMetadataResponse(BaseModel):
    """Metadata fields in a story response."""

    model_config = ConfigDict(frozen=True)

    event_name: str | None = None
    event_year: int | None = None
    city: str | None = None
    genre: str | None = None
    promoter: str | None = None
    artist: str | None = None
    other: str | None = None


class StoryResponse(BaseModel):
    """Response for a single story."""

    model_config = ConfigDict(frozen=True)

    story_id: str = Field(description="UUID of the story.")
    text: str = Field(description="The story text.")
    word_count: int = Field(default=0, description="Word count.")
    input_mode: str = Field(default="text", description="'text' or 'audio'.")
    audio_duration: float | None = Field(default=None, description="Audio duration in seconds.")
    status: StoryStatus = Field(description="Moderation status.")
    created_at: str = Field(description="Submission date (YYYY-MM-DD).")
    metadata: StoryMetadataResponse = Field(description="Event context.")
    entity_tags: list[str] = Field(default_factory=list, description="Extracted entity tags.")
    genre_tags: list[str] = Field(default_factory=list, description="Extracted genre tags.")
    geographic_tags: list[str] = Field(default_factory=list, description="Extracted geographic tags.")


class SubmitStoryResponse(BaseModel):
    """Response after submitting a story."""

    model_config = ConfigDict(frozen=True)

    story_id: str | None = Field(default=None, description="UUID if story was created.")
    status: str = Field(description="Result status: APPROVED, REJECTED, or error.")
    created_at: str | None = Field(default=None, description="Submission date.")
    moderation_flags: list[str] = Field(default_factory=list, description="Moderation flags.")
    moderation_reason: str | None = Field(default=None, description="Rejection reason if applicable.")
    error: str | None = Field(default=None, description="Error message if validation failed.")


class EventSummaryResponse(BaseModel):
    """Summary of an event with story count."""

    model_config = ConfigDict(frozen=True)

    event_name: str = Field(description="Event name.")
    event_year: int | None = Field(default=None, description="Event year.")
    city: str | None = Field(default=None, description="Event city.")
    story_count: int = Field(default=0, description="Number of approved stories.")


class EventCollectionResponse(BaseModel):
    """Full event collection with stories and optional narrative."""

    model_config = ConfigDict(frozen=True)

    event_name: str = Field(description="Event name.")
    event_year: int | None = Field(default=None, description="Event year.")
    city: str | None = Field(default=None, description="Event city.")
    story_count: int = Field(default=0, description="Number of approved stories.")
    stories: list[StoryResponse] = Field(default_factory=list, description="Stories for this event.")
    narrative: str | None = Field(default=None, description="Collective narrative.")
    themes: list[str] = Field(default_factory=list, description="Recurring themes.")
    narrative_generated_at: str | None = Field(default=None, description="When narrative was generated.")


class NarrativeResponse(BaseModel):
    """Response for a collective narrative."""

    model_config = ConfigDict(frozen=True)

    event_name: str = Field(description="Event name.")
    event_year: int | None = Field(default=None, description="Event year.")
    narrative: str | None = Field(default=None, description="The collective narrative.")
    themes: list[str] = Field(default_factory=list, description="Recurring themes.")
    story_count: int = Field(default=0, description="Stories used to generate narrative.")
    generated_at: str | None = Field(default=None, description="Generation date.")
    error: str | None = Field(default=None, description="Error message if generation failed.")


class SearchResultResponse(BaseModel):
    """A single semantic search result."""

    model_config = ConfigDict(frozen=True)

    story_id: str = Field(description="Story UUID.")
    text_excerpt: str = Field(description="First 500 chars of matching story.")
    similarity_score: float = Field(description="Cosine similarity score (0–1).")
    entity_tags: list[str] = Field(default_factory=list)
    genre_tags: list[str] = Field(default_factory=list)
    geographic_tags: list[str] = Field(default_factory=list)
    time_period: str | None = Field(default=None)


class StatsResponse(BaseModel):
    """Aggregate story statistics."""

    model_config = ConfigDict(frozen=True)

    total_stories: int = Field(default=0)
    approved_stories: int = Field(default=0)
    total_events: int = Field(default=0)
    total_entity_tags: int = Field(default=0)
    total_genre_tags: int = Field(default=0)
    total_geographic_tags: int = Field(default=0)
    stories_by_status: dict[str, int] = Field(default_factory=dict)
    stories_by_input_mode: dict[str, int] = Field(default_factory=dict)

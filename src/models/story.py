"""Rave Stories domain models — anonymous first-person rave experience accounts.

# ─── ARCHITECTURE ROLE ───────────────────────────────────────────────
#
# Layer: Models (bottom of the dependency graph — no imports from upper layers).
#
# These frozen Pydantic v2 models represent the core data structures for the
# Rave Stories feature.  Stories are anonymous first-person accounts of rave
# experiences, tagged to events, genres, cities, artists, and promoters.
#
# Key design decisions:
#   - **Anonymity by design**: No user ID, session ID, or IP address fields
#     exist on any model.  ``created_at`` stores only the date (YYYY-MM-DD),
#     never a timestamp, so individual submissions can't be correlated by time.
#   - **Immutable state**: All models use ``frozen=True``.  Status transitions
#     (e.g. PENDING_MODERATION → APPROVED) use ``model_copy(update={...})``.
#   - **Decoupled metadata**: ``StoryMetadata`` is a separate model from
#     ``RaveStory`` to keep the story text cleanly separated from the
#     user-supplied event/artist/genre context.
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


# ─── StoryStatus ─────────────────────────────────────────────────────
# Tracks the lifecycle of a submitted story through the moderation pipeline.
# Inherits from (str, Enum) for automatic JSON serialization as a string.
class StoryStatus(str, Enum):
    """Lifecycle states for a submitted rave story."""

    PENDING_MODERATION = "PENDING_MODERATION"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


# ─── StoryMetadata ───────────────────────────────────────────────────
# User-supplied context about the event the story describes.  At least one
# field must be filled for a valid submission (enforced at the API layer).
class StoryMetadata(BaseModel):
    """Optional metadata supplied by the story author.

    Captures the event context: which event, when, where, what genre,
    which promoter/artist.  Used for tagging and grouping stories into
    event collections.
    """

    model_config = ConfigDict(frozen=True)

    event_name: str | None = Field(default=None, description="Name of the rave/event.")
    event_year: int | None = Field(default=None, ge=1980, le=2030, description="Year the event occurred.")
    city: str | None = Field(default=None, description="City where the event took place.")
    genre: str | None = Field(default=None, description="Primary genre (e.g. techno, jungle, house).")
    promoter: str | None = Field(default=None, description="Promoter or crew who organized the event.")
    artist: str | None = Field(default=None, description="Featured artist the story centers on.")
    other: str | None = Field(default=None, description="Freeform extra context from the author.")

    def has_any_field(self) -> bool:
        """Return True if at least one metadata field is populated."""
        return any([
            self.event_name, self.event_year, self.city,
            self.genre, self.promoter, self.artist, self.other,
        ])


# ─── ModerationResult ────────────────────────────────────────────────
# Output of the LLM + profanity-filter moderation pipeline.
class ModerationResult(BaseModel):
    """Result of the content moderation pipeline for a story submission.

    The pipeline runs in two stages:
      1. Sync profanity filter (better-profanity) catches slurs/hate speech.
      2. Async LLM moderation checks for PII, threats, illegal content, spam.

    If PII is detected, ``sanitized_text`` contains the cleaned version.
    ``flags`` lists all issues found (e.g. ["pii_detected", "phone_number"]).
    """

    model_config = ConfigDict(frozen=True)

    is_safe: bool = Field(description="True if the story passes moderation.")
    flags: list[str] = Field(
        default_factory=list,
        description="List of moderation flags raised (e.g. 'pii_detected', 'hate_speech').",
    )
    sanitized_text: str | None = Field(
        default=None,
        description="Cleaned text with PII redacted.  None if no PII was found.",
    )
    reason: str | None = Field(
        default=None,
        description="Human-readable explanation of rejection, if rejected.",
    )


# ─── RaveStory ────────────────────────────────────────────────────────
# The central domain model — a single anonymous rave experience account.
class RaveStory(BaseModel):
    """A single anonymous rave experience story.

    Stories are submitted as text or transcribed audio, moderated, tagged
    with entities extracted from the text, and indexed in ChromaDB for
    semantic search.  Multiple stories for the same event coalesce into
    collective narratives.

    Anonymity guarantee: no user ID, IP, session ID, or sub-second timestamp
    fields.  ``created_at`` stores only the date (YYYY-MM-DD).
    """

    model_config = ConfigDict(frozen=True)

    story_id: str = Field(description="UUID identifying this story.")
    text: str = Field(description="The story text (sanitized, moderated).")
    word_count: int = Field(default=0, ge=0, description="Word count of the story text.")
    input_mode: str = Field(
        default="text",
        description="How the story was submitted: 'text' or 'audio'.",
    )
    audio_duration: float | None = Field(
        default=None,
        description="Duration of the original audio in seconds (audio submissions only).",
    )
    status: StoryStatus = Field(
        default=StoryStatus.PENDING_MODERATION,
        description="Current moderation status.",
    )
    moderation_flags: list[str] = Field(
        default_factory=list,
        description="Flags raised during moderation (empty if clean).",
    )
    # Date only — no timestamp — to prevent correlation of submissions.
    created_at: str = Field(description="Submission date (YYYY-MM-DD only, no timestamp).")
    moderated_at: str | None = Field(default=None, description="Date moderation completed (YYYY-MM-DD).")
    metadata: StoryMetadata = Field(
        default_factory=StoryMetadata,
        description="User-supplied event context.",
    )
    # Entity tags extracted from the story text by LLM entity extraction.
    entity_tags: list[str] = Field(
        default_factory=list,
        description="Artist, venue, and label names extracted from the story.",
    )
    genre_tags: list[str] = Field(
        default_factory=list,
        description="Genre tags extracted from the story.",
    )
    geographic_tags: list[str] = Field(
        default_factory=list,
        description="City and region tags extracted from the story.",
    )


# ─── EventStoryCollection ────────────────────────────────────────────
# Groups all approved stories for a single event, with an optional
# LLM-generated collective narrative.
class EventStoryCollection(BaseModel):
    """All approved stories for a specific event, with optional collective narrative.

    The collective narrative is generated on-demand when >= 3 stories exist
    for an event.  It synthesizes individual accounts into a crowd-perspective
    description of what happened, without attributing details to individual
    stories (preserving anonymity).
    """

    model_config = ConfigDict(frozen=True)

    event_name: str = Field(description="Name of the event.")
    event_year: int | None = Field(default=None, description="Year of the event.")
    city: str | None = Field(default=None, description="City of the event.")
    story_count: int = Field(default=0, ge=0, description="Number of approved stories.")
    stories: list[RaveStory] = Field(
        default_factory=list,
        description="Approved stories for this event.",
    )
    narrative: str | None = Field(
        default=None,
        description="LLM-generated collective narrative (requires >= 3 stories).",
    )
    themes: list[str] = Field(
        default_factory=list,
        description="Recurring themes identified across stories.",
    )
    narrative_generated_at: str | None = Field(
        default=None,
        description="Date the narrative was last generated (YYYY-MM-DD).",
    )

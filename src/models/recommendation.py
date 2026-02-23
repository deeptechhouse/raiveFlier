"""Recommendation models for the raiveFlier pipeline.

Defines Pydantic v2 models for co-appearances, recommended artists, and
recommendation results. All models use frozen config to enforce immutability
(encapsulation per CLAUDE.md Section 28).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CoAppearance(BaseModel):
    """Records that an artist appeared on another flier alongside a flier artist."""

    model_config = ConfigDict(frozen=True)

    artist_name: str
    shared_with: str
    event_name: str | None = None
    venue: str | None = None
    times_seen: int = 1


class RecommendedArtist(BaseModel):
    """A single artist recommendation with provenance and connection metadata."""

    model_config = ConfigDict(frozen=True)

    artist_name: str
    genres: list[str] = Field(default_factory=list)
    reason: str
    source_tier: str
    connection_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    connected_to: list[str] = Field(default_factory=list)
    label_name: str | None = None
    event_name: str | None = None


class RecommendationResult(BaseModel):
    """The full recommendation response for a flier analysis."""

    model_config = ConfigDict(frozen=True)

    recommendations: list[RecommendedArtist] = Field(default_factory=list)
    flier_artists: list[str] = Field(default_factory=list)
    genres_analyzed: list[str] = Field(default_factory=list)
    generated_at: datetime | None = None

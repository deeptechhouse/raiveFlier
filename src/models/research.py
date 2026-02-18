"""Research result models for the raiveFlier pipeline.

Defines Pydantic v2 models for date context and research results produced
by the entity research phase. All models use frozen config to enforce immutability.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from src.models.entities import (
    ArticleReference,
    Artist,
    EntityType,
    Promoter,
    Venue,
)


class DateContext(BaseModel):
    """Contextual information about the date/era of an event on a flier."""

    model_config = ConfigDict(frozen=True)

    event_date: date
    scene_context: str | None = None
    city_context: str | None = None
    cultural_context: str | None = None
    nearby_events: list[str] = Field(default_factory=list)
    sources: list[ArticleReference] = Field(default_factory=list)


class ResearchResult(BaseModel):
    """The result of researching a single entity extracted from a flier."""

    model_config = ConfigDict(frozen=True)

    entity_type: EntityType
    entity_name: str
    artist: Artist | None = None
    venue: Venue | None = None
    promoter: Promoter | None = None
    date_context: DateContext | None = None
    sources_consulted: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)

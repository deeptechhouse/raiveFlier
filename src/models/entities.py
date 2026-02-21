"""Core domain entities for the raiveFlier analysis pipeline.

Defines enums and Pydantic v2 models for artists, venues, promoters, releases,
labels, event appearances, and article references. All models use frozen config
to enforce immutability (encapsulation per CLAUDE.md Section 28).
"""

from __future__ import annotations

import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class EntityType(str, Enum):  # noqa: UP042 — StrEnum requires Python 3.11+
    """Types of entities that can be extracted from a rave flier."""

    ARTIST = "ARTIST"
    VENUE = "VENUE"
    PROMOTER = "PROMOTER"
    DATE = "DATE"
    GENRE = "GENRE"
    LABEL = "LABEL"
    EVENT = "EVENT"


class ConfidenceLevel(str, Enum):  # noqa: UP042 — StrEnum requires Python 3.11+
    """Qualitative confidence levels derived from numeric confidence scores.

    Thresholds:
        HIGH:      >= 0.8
        MEDIUM:    >= 0.5
        LOW:       >= 0.3
        UNCERTAIN: <  0.3
    """

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNCERTAIN = "UNCERTAIN"


class Label(BaseModel):
    """A record label associated with an artist or release."""

    model_config = ConfigDict(frozen=True)

    name: str
    discogs_id: int | None = None
    discogs_url: str | None = None


class Release(BaseModel):
    """A music release (single, EP, album) associated with an artist."""

    model_config = ConfigDict(frozen=True)

    title: str
    label: str
    catalog_number: str | None = None
    year: int | None = None
    format: str | None = None
    discogs_url: str | None = None
    bandcamp_url: str | None = None
    beatport_url: str | None = None
    genres: list[str] = Field(default_factory=list)
    styles: list[str] = Field(default_factory=list)


class EventAppearance(BaseModel):
    """A record of an artist appearing at an event."""

    model_config = ConfigDict(frozen=True)

    event_name: str | None = None
    venue: str | None = None
    city: str | None = None
    date: datetime.date | None = None
    source: str | None = None
    source_url: str | None = None


class ArticleReference(BaseModel):
    """A reference to an article, interview, review, or forum post."""

    model_config = ConfigDict(frozen=True)

    title: str
    source: str
    url: str | None = None
    date: datetime.date | None = None
    article_type: str = "article"
    snippet: str | None = None
    citation_tier: int = Field(default=6, ge=1, le=6)


class EventInstance(BaseModel):
    """A single historical instance of a named event/party."""

    model_config = ConfigDict(frozen=True)

    event_name: str
    promoter: str | None = None
    venue: str | None = None
    city: str | None = None
    date: str | None = None
    source_url: str | None = None


class EventSeriesHistory(BaseModel):
    """Historical research on a named event series (e.g. 'Bugged Out!', 'BLOC').

    Groups past instances by promoter to reveal promoter name changes
    or multiple promoters using similar event names.
    """

    model_config = ConfigDict(frozen=True)

    event_name: str
    instances: list[EventInstance] = Field(default_factory=list)
    promoter_groups: dict[str, list[EventInstance]] = Field(default_factory=dict)
    promoter_name_changes: list[str] = Field(default_factory=list)
    total_found: int = 0
    articles: list[ArticleReference] = Field(default_factory=list)


class Artist(BaseModel):
    """An artist/DJ/performer extracted from or researched for a rave flier."""

    model_config = ConfigDict(frozen=True)

    name: str
    aliases: list[str] = Field(default_factory=list)
    discogs_id: int | None = None
    musicbrainz_id: str | None = None
    bandcamp_url: str | None = None
    beatport_url: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    releases: list[Release] = Field(default_factory=list)
    labels: list[Label] = Field(default_factory=list)
    appearances: list[EventAppearance] = Field(default_factory=list)
    articles: list[ArticleReference] = Field(default_factory=list)
    profile_summary: str | None = None
    city: str | None = None
    region: str | None = None
    country: str | None = None

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Return the qualitative confidence level based on the numeric score."""
        if self.confidence >= 0.8:
            return ConfidenceLevel.HIGH
        if self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        if self.confidence >= 0.3:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.UNCERTAIN


class Venue(BaseModel):
    """A venue extracted from or researched for a rave flier."""

    model_config = ConfigDict(frozen=True)

    name: str
    location: str | None = None
    city: str | None = None
    country: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    history: str | None = None
    notable_events: list[str] = Field(default_factory=list)
    cultural_significance: str | None = None
    articles: list[ArticleReference] = Field(default_factory=list)


class Promoter(BaseModel):
    """A promoter/event organizer extracted from or researched for a rave flier."""

    model_config = ConfigDict(frozen=True)

    name: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    event_history: list[str] = Field(default_factory=list)
    affiliated_artists: list[str] = Field(default_factory=list)
    affiliated_venues: list[str] = Field(default_factory=list)
    articles: list[ArticleReference] = Field(default_factory=list)
    city: str | None = None
    region: str | None = None
    country: str | None = None

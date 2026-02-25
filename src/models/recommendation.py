"""Recommendation models for the raiveFlier pipeline.

Defines Pydantic v2 models for co-appearances, recommended artists, and
recommendation results. All models use frozen config to enforce immutability
(encapsulation per CLAUDE.md Section 28).

The recommendation system works after the main analysis is complete. It uses
three strategies to find artists the user might enjoy:
    1. Co-appearance — artists who have appeared on other fliers alongside
       the current flier's artists (from flier_history SQLite DB).
    2. Label-mates — artists who share record labels with flier artists
       (from Discogs/MusicBrainz data).
    3. Genre similarity — artists in the same genre/style tags.

See src/services/recommendation_service.py for the implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# CoAppearance — tracks when two artists have been on the same flier before.
# ---------------------------------------------------------------------------
class CoAppearance(BaseModel):
    """Records that an artist appeared on another flier alongside a flier artist.

    Built from the flier_history SQLite database (src/providers/flier_history/).
    If multiple past fliers feature both "Artist A" and "Artist B", the
    times_seen count increases, indicating a stronger connection.
    """

    model_config = ConfigDict(frozen=True)

    # The recommended artist's name (the one NOT on the current flier).
    artist_name: str
    # Which artist on the current flier they co-appeared with.
    shared_with: str
    # The event where they co-appeared (if known).
    event_name: str | None = None
    # The venue where they co-appeared (if known).
    venue: str | None = None
    # How many times they've co-appeared across all tracked fliers.
    times_seen: int = 1


# ---------------------------------------------------------------------------
# RecommendedArtist — a single recommendation with reasoning.
# ---------------------------------------------------------------------------
class RecommendedArtist(BaseModel):
    """A single artist recommendation with provenance and connection metadata.

    Each recommendation includes a human-readable reason explaining WHY
    this artist is being recommended, plus metadata about the connection
    so the frontend can display it meaningfully.
    """

    model_config = ConfigDict(frozen=True)

    # The recommended artist's name.
    artist_name: str
    # Genres associated with this artist (for display and filtering).
    genres: list[str] = Field(default_factory=list)
    # Human-readable explanation (e.g. "Shares label Warp Records with Aphex Twin").
    reason: str
    # How the recommendation was discovered:
    #   "co_appearance" — appeared on same flier
    #   "label_mate"    — shares a record label
    #   "genre_match"   — same genre/style
    source_tier: str
    # Strength of connection (0.0–1.0) — used to sort recommendations.
    # Co-appearances with high times_seen score highest.
    connection_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    # Which flier artist(s) this recommendation connects to.
    connected_to: list[str] = Field(default_factory=list)
    # If this is a label-mate recommendation, which label they share.
    label_name: str | None = None
    # If this is a co-appearance recommendation, which event connected them.
    event_name: str | None = None


# ---------------------------------------------------------------------------
# RecommendationResult — the complete recommendation response.
# ---------------------------------------------------------------------------
class RecommendationResult(BaseModel):
    """The full recommendation response for a flier analysis.

    Returned by the /api/v1/fliers/{session_id}/recommendations endpoint.
    Contains a sorted list of recommended artists plus metadata about
    what was analyzed to produce them.
    """

    model_config = ConfigDict(frozen=True)

    # Sorted list of recommendations (highest connection_strength first).
    recommendations: list[RecommendedArtist] = Field(default_factory=list)
    # The artist names from the current flier that drove the recommendations.
    flier_artists: list[str] = Field(default_factory=list)
    # All genre tags that were considered during recommendation.
    genres_analyzed: list[str] = Field(default_factory=list)
    # When the recommendations were generated (for caching/freshness).
    generated_at: datetime | None = None


# ---------------------------------------------------------------------------
# PreloadedTier1 — cached Tier 1 discovery results from background preload.
# ---------------------------------------------------------------------------
@dataclass
class PreloadedTier1:
    """Cached Tier 1 discovery results from the background preload task.

    After the main pipeline completes (research + interconnection), the
    background task calls RecommendationService.preload_tier1() to run all
    three Tier 1 discovery methods (label-mates, shared-flier, shared-
    lineup) in parallel.  Results are stored on app.state._reco_preload
    keyed by session_id.

    When the frontend later requests /recommendations or /recommendations/
    quick, the endpoints pass this cached data to the service, which skips
    the redundant discovery calls — delivering results in <1 second instead
    of 5-15+ seconds.

    Uses a plain dataclass rather than Pydantic because the inner dicts
    are mutable intermediate data (candidate dicts from the discovery
    methods) and this is an internal cache object, not an API-facing schema.
    """

    label_mates: list[dict[str, Any]] = field(default_factory=list)
    shared_flier: list[dict[str, Any]] = field(default_factory=list)
    shared_lineup: list[dict[str, Any]] = field(default_factory=list)

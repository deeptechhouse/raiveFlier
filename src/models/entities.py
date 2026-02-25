"""Core domain entities for the raiveFlier analysis pipeline.

Defines enums and Pydantic v2 models for artists, venues, promoters, releases,
labels, event appearances, and article references. All models use frozen config
to enforce immutability (encapsulation per CLAUDE.md Section 28).

This is the richest model file in the project — it contains the data structures
that the Research phase populates after looking up entities in music databases
(Discogs, MusicBrainz, Bandcamp, Beatport) and web searches.

Key relationships:
    - Artist has many Release objects and Label objects
    - Venue and Promoter carry ArticleReference lists from web research
    - EventSeriesHistory groups EventInstance records by promoter
    - All models are consumed by src/services/output_formatter.py for display
"""

from __future__ import annotations

import datetime
# Enum is the standard library base for creating enumeration types. We inherit
# from (str, Enum) so enum values serialize as plain strings in JSON, which
# Pydantic and FastAPI handle natively. The noqa comment suppresses a linter
# warning about StrEnum which requires Python 3.11+ (this project targets 3.10+).
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums — used throughout the codebase to classify entities and confidence.
# ---------------------------------------------------------------------------

class EntityType(str, Enum):  # noqa: UP042 — StrEnum requires Python 3.11+
    """Types of entities that can be extracted from a rave flier.

    Used by:
        - ExtractedEntity (src/models/flier.py) — tags each OCR-extracted entity
        - ResearchResult (src/models/research.py) — identifies what was researched
        - EntityNode (src/models/analysis.py) — nodes in the interconnection graph
        - entity_extractor.py prompt — tells the LLM what categories to look for

    If you add a new entity type, you must also add handling for it in:
        - src/services/entity_extractor.py (extraction prompt)
        - src/services/research_service.py (dispatch to the right researcher)
        - src/pipeline/orchestrator.py (if the new type needs special pipeline logic)
    """

    ARTIST = "ARTIST"       # DJ, musician, live act, or band
    VENUE = "VENUE"         # Club, warehouse, park, or other event location
    PROMOTER = "PROMOTER"   # Event organizer / promotion crew
    DATE = "DATE"           # Event date (parsed from text like "SAT 15 NOV 2003")
    GENRE = "GENRE"         # Musical genre tag (techno, house, drum & bass, etc.)
    LABEL = "LABEL"         # Record label (may appear on flier or found via research)
    EVENT = "EVENT"         # Named event series (e.g. "Bugged Out!", "BLOC")


class ConfidenceLevel(str, Enum):  # noqa: UP042 — StrEnum requires Python 3.11+
    """Qualitative confidence levels derived from numeric confidence scores.

    Used by the Artist.confidence_level property to convert a float (0.0–1.0)
    into a human-readable tier. Displayed in the frontend results view and
    used by the interconnection service to weight relationship edges.

    Thresholds:
        HIGH:      >= 0.8  — Strong match from a reliable music database
        MEDIUM:    >= 0.5  — Reasonable match, may need human verification
        LOW:       >= 0.3  — Weak match, likely needs manual confirmation
        UNCERTAIN: <  0.3  — Very low confidence, may be a false positive
    """

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNCERTAIN = "UNCERTAIN"


# ---------------------------------------------------------------------------
# Supporting models — used as nested fields within Artist, Venue, Promoter.
# ---------------------------------------------------------------------------

class Label(BaseModel):
    """A record label associated with an artist or release.

    Populated by music DB providers (Discogs, MusicBrainz) during artist research.
    The discogs_id enables direct linking to the label's Discogs page.
    """

    model_config = ConfigDict(frozen=True)

    name: str                           # Label name (e.g. "Warp Records")
    discogs_id: int | None = None       # Discogs numeric label ID (if found)
    discogs_url: str | None = None      # Full URL to the label's Discogs page


class Release(BaseModel):
    """A music release (single, EP, album) associated with an artist.

    Collected from multiple sources — each source may populate different fields.
    For example, Discogs provides catalog_number and discogs_url while Bandcamp
    provides bandcamp_url. The output formatter merges these for display.
    """

    model_config = ConfigDict(frozen=True)

    title: str                          # Release title (e.g. "Windowlicker")
    artist: str | None = None           # Artist name from the release (used by label-mate discovery)
    label: str                          # Label name (plain string, not a Label object)
    catalog_number: str | None = None   # Catalog number (e.g. "WAP105") from Discogs
    year: int | None = None             # Release year
    format: str | None = None           # Physical format ("12\"", "CD", "Digital", etc.)
    discogs_url: str | None = None      # Link to Discogs release page
    bandcamp_url: str | None = None     # Link to Bandcamp release page
    beatport_url: str | None = None     # Link to Beatport release page
    # Genre and style tags help the recommendation service find similar artists.
    genres: list[str] = Field(default_factory=list)   # Broad genres ("Electronic")
    styles: list[str] = Field(default_factory=list)   # Specific styles ("Acid Techno")


class EventAppearance(BaseModel):
    """A record of an artist appearing at an event.

    Used by the recommendation service to find co-appearing artists —
    if two artists have EventAppearances at the same venue/event, they
    may be musically related and worth recommending together.

    .. note::
        This model is currently not populated by any researcher.  The
        interconnection service discovers shared events via direct
        ChromaDB RA event queries instead of through this model.

    .. todo::
        Wire EventAppearance population into ArtistResearcher so artist
        research results include event history from RA corpus data.
        This would also enable ``_extract_artist_geography()`` to infer
        an artist's geographic base from their appearance patterns.
    """

    model_config = ConfigDict(frozen=True)

    event_name: str | None = None       # Name of the event (if known)
    venue: str | None = None            # Venue where the event was held
    city: str | None = None             # City of the event
    date: datetime.date | None = None   # Date of the event
    source: str | None = None           # Where this data came from ("discogs", "ra", etc.)
    source_url: str | None = None       # URL to the source page


class ArticleReference(BaseModel):
    """A reference to an article, interview, review, or forum post.

    Collected during web research for venues, promoters, and artists.
    The citation_tier field ranks source reliability on a 1-6 scale:
        1 = Published book (e.g. Energy Flash by Simon Reynolds)
        2 = Academic paper / peer-reviewed journal
        3 = Major music publication (Resident Advisor, Pitchfork, Mixmag)
        4 = Blog or independent publication
        5 = Forum post or social media
        6 = Unverified web content (default)
    """

    model_config = ConfigDict(frozen=True)

    title: str                              # Article headline or title
    source: str                             # Publication name (e.g. "Resident Advisor")
    url: str | None = None                  # Direct link to the article
    date: datetime.date | None = None       # Publication date
    article_type: str = "article"           # "article", "interview", "review", "forum"
    snippet: str | None = None              # Brief excerpt or summary
    # citation_tier: 1 (most reliable) to 6 (least reliable).
    # ge=1 and le=6 enforce this range at Pydantic validation time.
    citation_tier: int = Field(default=6, ge=1, le=6)


# ---------------------------------------------------------------------------
# Event series models — track the history of named recurring events.
# ---------------------------------------------------------------------------

class EventInstance(BaseModel):
    """A single historical instance of a named event/party.

    For example, one edition of "Bugged Out!" at Fabric on 2003-11-15.
    Multiple EventInstance objects are grouped into EventSeriesHistory.
    """

    model_config = ConfigDict(frozen=True)

    event_name: str                     # The event series name
    promoter: str | None = None         # Who promoted this specific instance
    venue: str | None = None            # Where it was held
    city: str | None = None             # City
    date: str | None = None             # Date as free-text (may not parse to a date)
    source_url: str | None = None       # Where this info was found


class EventSeriesHistory(BaseModel):
    """Historical research on a named event series (e.g. 'Bugged Out!', 'BLOC').

    Groups past instances by promoter to reveal promoter name changes
    or multiple promoters using similar event names. This is valuable for
    rave history research — event series often change promoters, venues,
    and even cities over decades.
    """

    model_config = ConfigDict(frozen=True)

    event_name: str                     # The event series being researched
    # All discovered instances of this event, regardless of promoter.
    instances: list[EventInstance] = Field(default_factory=list)
    # Instances grouped by promoter name — helps identify if different
    # promoters ran events with the same name (common in rave culture).
    promoter_groups: dict[str, list[EventInstance]] = Field(default_factory=dict)
    # Detected name changes for the promoter (e.g. "Bugged Out" → "Bugged Out!")
    promoter_name_changes: list[str] = Field(default_factory=list)
    # Total number of instances found across all sources.
    total_found: int = 0
    # Articles/references about the event series history.
    articles: list[ArticleReference] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Primary entity models — the main output of the Research phase.
# These are the "rich" versions of entities, populated with data from
# music databases, web searches, and article scraping.
# ---------------------------------------------------------------------------

class Artist(BaseModel):
    """An artist/DJ/performer extracted from or researched for a rave flier.

    This is the most data-rich model in the system. The artist researcher
    (src/services/artist_researcher.py) populates it by querying multiple
    music databases in priority order: Discogs → MusicBrainz → Bandcamp →
    Beatport → web search. Each source may contribute different fields.
    """

    model_config = ConfigDict(frozen=True)

    name: str                                   # Primary artist name
    # Alternative names / aliases (DJs often perform under multiple names).
    aliases: list[str] = Field(default_factory=list)
    # Database IDs for cross-referencing and linking to source pages.
    discogs_id: int | None = None
    musicbrainz_id: str | None = None
    bandcamp_url: str | None = None
    beatport_url: str | None = None
    # Confidence that this is the correct artist (set by the researcher
    # based on match quality from music databases).
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # Discography and label affiliations — key data for understanding
    # an artist's place in the music ecosystem.
    releases: list[Release] = Field(default_factory=list)
    labels: list[Label] = Field(default_factory=list)
    # LLM-generated summary of the artist's profile (style, history, significance).
    profile_summary: str | None = None
    # Geographic info — helps the interconnection service map scene relationships.
    city: str | None = None
    region: str | None = None
    country: str | None = None

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Return the qualitative confidence level based on the numeric score.

        This property converts the float confidence score into a human-readable
        tier. Used by the frontend to color-code results and by the
        interconnection service to weight relationship edges.
        """
        if self.confidence >= 0.8:
            return ConfidenceLevel.HIGH
        if self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        if self.confidence >= 0.3:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.UNCERTAIN


class Venue(BaseModel):
    """A venue extracted from or researched for a rave flier.

    Populated by src/services/venue_researcher.py using web search and
    article scraping. Venues are important in rave culture — iconic clubs
    (Tresor, Fabric, Berghain) have their own cultural significance.
    """

    model_config = ConfigDict(frozen=True)

    name: str                                   # Venue name (e.g. "Fabric")
    location: str | None = None                 # Street address or general location
    city: str | None = None                     # City (e.g. "London")
    country: str | None = None                  # Country (e.g. "UK")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # LLM-generated narrative about the venue's history and role in the scene.
    history: str | None = None
    # Notable events held at this venue (e.g. ["Fabric Live", "FABRICLIVE"]).
    notable_events: list[str] = Field(default_factory=list)
    # Why this venue matters to rave/club culture.
    cultural_significance: str | None = None
    # Web articles and references about this venue.
    articles: list[ArticleReference] = Field(default_factory=list)


class Promoter(BaseModel):
    """A promoter/event organizer extracted from or researched for a rave flier.

    Populated by src/services/promoter_researcher.py. Promoters are the
    connective tissue of rave culture — they curate lineups, book venues,
    and shape local scenes. Tracking promoter affiliations reveals the
    social network behind the music.
    """

    model_config = ConfigDict(frozen=True)

    name: str                                   # Promoter/crew name
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # Names of past events this promoter has organized.
    event_history: list[str] = Field(default_factory=list)
    # Artists this promoter frequently books — reveals curatorial taste.
    affiliated_artists: list[str] = Field(default_factory=list)
    # Venues this promoter commonly uses.
    affiliated_venues: list[str] = Field(default_factory=list)
    # Web articles and references about this promoter.
    articles: list[ArticleReference] = Field(default_factory=list)
    # Geographic info — most promoters operate in a specific city/region.
    city: str | None = None
    region: str | None = None
    country: str | None = None

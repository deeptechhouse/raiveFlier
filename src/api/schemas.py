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
    dropdowns dynamically from the actual corpus contents.  Also exposes
    full entity and geographic tag lists for autocomplete suggestions.
    """

    total_chunks: int = 0
    total_sources: int = 0
    sources_by_type: dict[str, int] = Field(default_factory=dict)
    entity_tag_count: int = 0
    geographic_tag_count: int = 0
    # Dynamic filter population — sorted lists for frontend dropdowns
    genre_tags: list[str] = Field(default_factory=list)
    time_periods: list[str] = Field(default_factory=list)
    # Full tag lists for autocomplete / fuzzy-match suggestions
    entity_tags: list[str] = Field(
        default_factory=list,
        description="Sorted list of distinct entity tag strings for autocomplete.",
    )
    geographic_tags: list[str] = Field(
        default_factory=list,
        description="Sorted list of distinct geographic tag strings for autocomplete.",
    )


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


# ---------------------------------------------------------------------------
# Smart Search schemas — query parsing, autocomplete, and faceted counts.
# These power the "smarter sidebar" features: natural-language filter
# detection, fuzzy autocomplete, and per-dimension result counts.
# ---------------------------------------------------------------------------


class ParseQueryRequest(BaseModel):
    """Request body for the parse-query endpoint.

    Accepts free-text input and returns structured filter signals detected
    via domain_knowledge functions (genre extraction, temporal detection,
    geographic scene mapping, artist alias resolution).
    """

    query: str = Field(..., min_length=1, max_length=500)


class ParsedQueryFilters(BaseModel):
    """Structured filters auto-detected from a natural-language query.

    Returned by POST /api/v1/corpus/parse-query and also embedded in
    CorpusSearchResponse so the frontend can show "auto-detected" badges
    on filter controls that were filled by the parser.

    Each field is optional — only populated when the parser detects that
    signal in the query text.
    """

    genres: list[str] = Field(default_factory=list)
    time_period: str | None = None
    geographic_tags: list[str] = Field(default_factory=list)
    artist_canonical: str | None = None
    artist_aliases: list[str] = Field(default_factory=list)


class SuggestResponse(BaseModel):
    """Autocomplete suggestions for a filter field.

    Returned by GET /api/v1/corpus/suggest.  Combines prefix matching,
    substring matching, and fuzzy matching (via difflib) to tolerate
    typos and partial input.
    """

    field: str
    prefix: str
    suggestions: list[str] = Field(default_factory=list)


class FacetCounts(BaseModel):
    """Per-dimension counts from the search candidate pool.

    Computed after vector retrieval and per-source dedup but BEFORE
    user-applied filters, so they reflect what is available in the
    query's semantic neighborhood regardless of active filter settings.
    The frontend uses these to annotate filter dropdowns with counts
    like "Techno (42)" or "Detroit (17)".
    """

    source_types: dict[str, int] = Field(default_factory=dict)
    genre_tags: dict[str, int] = Field(default_factory=dict)
    time_periods: dict[str, int] = Field(default_factory=dict)
    # Truncated to top-30 / top-20 by count to bound response size
    entity_tags: dict[str, int] = Field(default_factory=dict)
    geographic_tags: dict[str, int] = Field(default_factory=dict)
    citation_tiers: dict[str, int] = Field(default_factory=dict)


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


class WebSearchResult(BaseModel):
    """A single web search result from the corpus sidebar's web-search tier.

    Returned alongside RAG corpus chunks when the tiered search strategy
    augments local knowledge with live web results.  Only results that pass
    the music-relevance filter (known domains or keyword signals) are included.
    """

    title: str
    url: str
    snippet: str = ""
    source_domain: str = ""


class CorpusSearchCitation(BaseModel):
    """A single citation referenced in the synthesized answer.

    Maps a citation marker (e.g. "[1]") to its source metadata so the
    frontend can render inline citation links or a bibliography section.
    The ``source_tier`` field indicates which search tier produced this
    citation (ra_exchange, book, corpus, or web).
    """

    index: int = Field(description="1-based citation number matching [N] markers in the answer text.")
    source_title: str
    author: str | None = None
    citation_tier: int = 6
    page_number: str | None = None
    excerpt: str = Field(default="", description="Short excerpt from the source passage used.")
    url: str | None = Field(default=None, description="URL for web-sourced citations.")
    source_tier: str | None = Field(
        default=None,
        description='Which search tier produced this citation: "ra_exchange", "book", "event_listing", "corpus", or "web".',
    )


class CorpusSearchResponse(BaseModel):
    """Response containing corpus search results with pagination metadata.

    The backend computes the full ranked result set then returns a page
    controlled by offset/page_size.  The has_more flag tells the frontend
    whether a "Load More" button should be rendered.

    Optional smart-search fields (facets, parsed_filters) are populated
    when the backend detects structured signals in the query or computes
    facet counts from the candidate pool.  Old clients ignore them safely.

    The synthesized_answer field contains an LLM-generated natural language
    response synthesized from the top retrieved chunks.  It includes inline
    citation markers like [1], [2] that map to the answer_citations list.
    """

    query: str
    total_results: int
    results: list[CorpusSearchChunk] = Field(default_factory=list)
    # Pagination metadata — safe defaults preserve backward compatibility.
    offset: int = 0
    page_size: int = 20
    has_more: bool = False
    # Smart-search additions — both default to None for backward compat.
    facets: FacetCounts | None = Field(
        default=None,
        description="Per-dimension counts from the candidate pool (pre-filter).",
    )
    parsed_filters: ParsedQueryFilters | None = Field(
        default=None,
        description="Filters auto-detected from the query text via domain knowledge.",
    )
    # LLM-synthesized natural language answer from retrieved chunks.
    # Only populated on fresh searches (offset=0), not on "Load More" pages.
    synthesized_answer: str | None = Field(
        default=None,
        description="Cohesive NL answer synthesized by LLM from the top retrieved chunks.",
    )
    answer_citations: list[CorpusSearchCitation] = Field(
        default_factory=list,
        description="Numbered citations referenced by [N] markers in synthesized_answer.",
    )
    # Tiered search additions — web results and tier tracking.
    web_results: list[WebSearchResult] = Field(
        default_factory=list,
        description="Music-relevant web search results from the web-search tier (tier 4).",
    )
    search_tiers_used: list[int] = Field(
        default_factory=list,
        description="Which search tiers returned results: 1=RA Exchange, 2=Books, 3=Other corpus, 4=Web.",
    )


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


# ---------------------------------------------------------------------------
# Persistent Analysis Storage schemas — stored analysis retrieval,
# annotations, and listing endpoints.
# ---------------------------------------------------------------------------


class StoredAnalysisResponse(BaseModel):
    """Response containing a stored analysis snapshot."""

    session_id: str
    flier_id: int
    interconnection_map: dict[str, Any]
    research_results: list[dict[str, Any]] | None = None
    revision: int = 1
    created_at: str | None = None


class AnalysisListResponse(BaseModel):
    """Paginated list of stored analyses."""

    analyses: list[dict[str, Any]] = Field(default_factory=list)
    total: int = 0
    offset: int = 0
    limit: int = 50


class AddAnnotationRequest(BaseModel):
    """Request to add a user annotation to a stored analysis."""

    note: str = Field(..., min_length=1, max_length=2000)
    target_type: str = Field(default="analysis", description="analysis, entity, or edge")
    target_key: str | None = Field(default=None, description="Entity name or source->target for edges")


class AnnotationResponse(BaseModel):
    """Response after adding an annotation."""

    id: int
    flier_id: int
    target_type: str
    target_key: str | None = None
    note: str
    created_at: str | None = None
    updated_at: str | None = None


class AnnotationListResponse(BaseModel):
    """List of annotations for a session's analysis."""

    session_id: str
    annotations: list[dict[str, Any]] = Field(default_factory=list)
    total: int = 0

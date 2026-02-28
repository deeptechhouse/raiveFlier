"""REST API routes for the Rave Stories feature.

# ─── ARCHITECTURE ROLE ───────────────────────────────────────────────
#
# Layer: API (FastAPI route handlers).
# Pattern: Routes access services via ``request.app.state.story_service``
#          — same pattern as the main raiveFlier API and raiveFeeder.
#          No ``Depends()`` for singleton services; direct ``getattr``.
#
# Endpoints (order matters — literal routes before catch-all):
#   POST /api/v1/stories/submit        — Submit text story
#   POST /api/v1/stories/submit-audio  — Upload + transcribe audio story
#   GET  /api/v1/stories/              — List stories (paginated + filterable)
#   GET  /api/v1/stories/events        — List event collections
#   GET  /api/v1/stories/events/{name} — Get all stories for an event
#   GET  /api/v1/stories/events/{name}/narrative — Get collective narrative
#   POST /api/v1/stories/search        — Semantic search
#   GET  /api/v1/stories/tags/{type}   — List tag values
#   GET  /api/v1/stories/stats         — Aggregate stats
#   GET  /api/v1/stories/{story_id}    — Get single story (LAST — catch-all)
#
# Anonymity middleware: The StoryAnonymityMiddleware (defined in the
# sub-app factory) strips identifying headers before any logging on
# story routes.
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form

from src.models.story import StoryMetadata
from src.services.story_service import StoryService
from src.stories.api.schemas import (
    EventCollectionResponse,
    EventSummaryResponse,
    NarrativeResponse,
    SearchResultResponse,
    SearchStoriesRequest,
    StatsResponse,
    StoryMetadataResponse,
    StoryResponse,
    SubmitStoryRequest,
    SubmitStoryResponse,
)

logger = structlog.get_logger(logger_name=__name__)

router = APIRouter(prefix="/api/v1/stories", tags=["stories"])


# ── Service accessor ──────────────────────────────────────────────────
def _get_story_service(request: Request) -> StoryService:
    """Retrieve StoryService from app state; raise 503 if unavailable."""
    svc = getattr(request.app.state, "story_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Story service unavailable")
    return svc


# ── Helper: convert RaveStory to StoryResponse ────────────────────────
def _story_to_response(story: Any) -> StoryResponse:
    """Map a RaveStory domain model to the API response schema."""
    return StoryResponse(
        story_id=story.story_id,
        text=story.text,
        word_count=story.word_count,
        input_mode=story.input_mode,
        audio_duration=story.audio_duration,
        status=story.status,
        created_at=story.created_at,
        metadata=StoryMetadataResponse(
            event_name=story.metadata.event_name,
            event_year=story.metadata.event_year,
            city=story.metadata.city,
            genre=story.metadata.genre,
            promoter=story.metadata.promoter,
            artist=story.metadata.artist,
            other=story.metadata.other,
        ),
        entity_tags=story.entity_tags,
        genre_tags=story.genre_tags,
        geographic_tags=story.geographic_tags,
    )


# ── Submit text story ─────────────────────────────────────────────────
@router.post("/submit", response_model=SubmitStoryResponse)
async def submit_story(request: Request, body: SubmitStoryRequest) -> SubmitStoryResponse:
    """Submit a text story through the moderation + extraction pipeline."""
    svc = _get_story_service(request)

    metadata = StoryMetadata(
        event_name=body.metadata.event_name,
        event_year=body.metadata.event_year,
        city=body.metadata.city,
        genre=body.metadata.genre,
        promoter=body.metadata.promoter,
        artist=body.metadata.artist,
        other=body.metadata.other,
    )

    result = await svc.submit_text_story(body.text, metadata)

    return SubmitStoryResponse(
        story_id=result.get("story_id"),
        status=result.get("status", "REJECTED"),
        created_at=result.get("created_at"),
        moderation_flags=result.get("moderation_flags", []),
        moderation_reason=result.get("moderation_reason"),
        error=result.get("error"),
    )


# ── Submit audio story ────────────────────────────────────────────────
@router.post("/submit-audio", response_model=SubmitStoryResponse)
async def submit_audio_story(
    request: Request,
    audio: UploadFile = File(..., description="Audio file (mp3/m4a/wav/webm/ogg, max 25MB)"),
    event_name: str | None = Form(default=None),
    event_year: int | None = Form(default=None),
    city: str | None = Form(default=None),
    genre: str | None = Form(default=None),
    promoter: str | None = Form(default=None),
    artist: str | None = Form(default=None),
    other: str | None = Form(default=None),
) -> SubmitStoryResponse:
    """Upload and transcribe an audio story."""
    svc = _get_story_service(request)

    metadata = StoryMetadata(
        event_name=event_name,
        event_year=event_year,
        city=city,
        genre=genre,
        promoter=promoter,
        artist=artist,
        other=other,
    )

    # Read audio data from upload.
    audio_data = await audio.read()
    filename = audio.filename or "recording.webm"

    result = await svc.submit_audio_story(audio_data, filename, metadata)

    return SubmitStoryResponse(
        story_id=result.get("story_id"),
        status=result.get("status", "REJECTED"),
        created_at=result.get("created_at"),
        moderation_flags=result.get("moderation_flags", []),
        moderation_reason=result.get("moderation_reason"),
        error=result.get("error"),
    )


# ── List stories ──────────────────────────────────────────────────────
@router.get("/", response_model=list[StoryResponse])
async def list_stories(
    request: Request,
    tag_type: str | None = None,
    tag_value: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[StoryResponse]:
    """List approved stories with optional tag filtering and pagination."""
    # Validate pagination to prevent abuse.
    limit = max(1, min(limit, 50))
    offset = max(0, offset)

    svc = _get_story_service(request)
    stories = await svc.list_stories(
        tag_type=tag_type,
        tag_value=tag_value,
        limit=limit,
        offset=offset,
    )
    return [_story_to_response(s) for s in stories]


# ── List events ───────────────────────────────────────────────────────
@router.get("/events", response_model=list[EventSummaryResponse])
async def list_events(
    request: Request,
    limit: int = 50,
    offset: int = 0,
) -> list[EventSummaryResponse]:
    """List events that have approved stories, with counts."""
    limit = max(1, min(limit, 100))
    offset = max(0, offset)

    svc = _get_story_service(request)
    events = await svc.list_events(limit=limit, offset=offset)
    return [
        EventSummaryResponse(
            event_name=e["event_name"],
            event_year=e.get("event_year"),
            city=e.get("city"),
            story_count=e.get("story_count", 0),
        )
        for e in events
    ]


# ── Get event stories ─────────────────────────────────────────────────
@router.get("/events/{event_name}", response_model=EventCollectionResponse)
async def get_event_stories(
    request: Request,
    event_name: str,
    event_year: int | None = None,
) -> EventCollectionResponse:
    """Get all approved stories for an event."""
    svc = _get_story_service(request)
    collection = await svc.get_event_stories(event_name, event_year)

    return EventCollectionResponse(
        event_name=collection.event_name,
        event_year=collection.event_year,
        city=collection.city,
        story_count=collection.story_count,
        stories=[_story_to_response(s) for s in collection.stories],
        narrative=collection.narrative,
        themes=collection.themes,
        narrative_generated_at=collection.narrative_generated_at,
    )


# ── Get or generate narrative ──────────────────────────────────────────
@router.get("/events/{event_name}/narrative", response_model=NarrativeResponse)
async def get_event_narrative(
    request: Request,
    event_name: str,
    event_year: int | None = None,
) -> NarrativeResponse:
    """Get or generate a collective narrative for an event (requires >= 3 stories)."""
    svc = _get_story_service(request)
    result = await svc.get_or_generate_narrative(event_name, event_year)

    return NarrativeResponse(
        event_name=result.get("event_name", event_name),
        event_year=result.get("event_year", event_year),
        narrative=result.get("narrative"),
        themes=result.get("themes", []),
        story_count=result.get("story_count", 0),
        generated_at=result.get("generated_at"),
        error=result.get("error"),
    )


# ── Semantic search ───────────────────────────────────────────────────
@router.post("/search", response_model=list[SearchResultResponse])
async def search_stories(
    request: Request,
    body: SearchStoriesRequest,
) -> list[SearchResultResponse]:
    """Semantic search across approved stories via ChromaDB."""
    svc = _get_story_service(request)
    results = await svc.search_stories(body.query, body.limit)

    return [
        SearchResultResponse(
            story_id=r["story_id"],
            text_excerpt=r["text_excerpt"],
            similarity_score=r["similarity_score"],
            entity_tags=r.get("entity_tags", []),
            genre_tags=r.get("genre_tags", []),
            geographic_tags=r.get("geographic_tags", []),
            time_period=r.get("time_period"),
        )
        for r in results
    ]


# ── List tags ─────────────────────────────────────────────────────────
@router.get("/tags/{tag_type}", response_model=list[str])
async def get_tags(request: Request, tag_type: str) -> list[str]:
    """List available tag values for a given type (entity, genre, geographic)."""
    if tag_type not in ("entity", "genre", "geographic"):
        raise HTTPException(status_code=400, detail="tag_type must be 'entity', 'genre', or 'geographic'")
    svc = _get_story_service(request)
    return await svc.get_tags(tag_type)


# ── Stats ─────────────────────────────────────────────────────────────
@router.get("/stats", response_model=StatsResponse)
async def get_stats(request: Request) -> StatsResponse:
    """Return aggregate story statistics."""
    svc = _get_story_service(request)
    stats = await svc.get_stats()
    return StatsResponse(**stats)


# ── Get single story (MUST be last — catch-all path parameter) ────────
# This route uses a bare path parameter ``{story_id}`` which matches any
# single path segment.  If placed before literal routes like ``/events``,
# ``/stats``, or ``/tags/{tag_type}``, FastAPI would interpret those
# literal segments as a ``story_id`` value and return 404.  Keeping this
# route at the very end of the file ensures all literal prefixes are
# checked first.
@router.get("/{story_id}", response_model=StoryResponse)
async def get_story(request: Request, story_id: str) -> StoryResponse:
    """Get a single approved story by UUID."""
    svc = _get_story_service(request)
    story = await svc.get_story(story_id)
    if story is None:
        raise HTTPException(status_code=404, detail="Story not found")
    return _story_to_response(story)

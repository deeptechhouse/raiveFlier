"""FastAPI API routes for the RaiveFlier pipeline.

Provides REST endpoints for flier upload, entity confirmation, progress
polling, result retrieval, health checks, and provider listing.  Service
dependencies are resolved from ``app.state`` via FastAPI's ``Depends``
using the ``Annotated`` pattern.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, UploadFile

from src.api.schemas import (
    AskQuestionRequest,
    AskQuestionResponse,
    ConfirmEntitiesRequest,
    ConfirmResponse,
    CorpusSearchChunk,
    CorpusSearchRequest,
    CorpusSearchResponse,
    CorpusStatsResponse,
    DismissConnectionRequest,
    DismissConnectionResponse,
    DuplicateMatch,
    ErrorResponse,
    FlierAnalysisResponse,
    FlierUploadResponse,
    HealthResponse,
    PipelineStatusResponse,
    ProvidersResponse,
    RatingResponse,
    RatingSummaryResponse,
    RecommendationsResponse,
    RecommendedArtistResponse,
    RelatedFact,
    SessionRatingsResponse,
    SubmitRatingRequest,
)
from src.models.entities import EntityType
from src.models.flier import ExtractedEntities, ExtractedEntity, FlierImage, OCRResult
from src.models.pipeline import PipelineState
from src.pipeline.confirmation_gate import ConfirmationGate
from src.pipeline.orchestrator import FlierAnalysisPipeline
from src.pipeline.progress_tracker import ProgressTracker
from src.utils.logging import get_logger

_logger: structlog.BoundLogger = get_logger(__name__)

router = APIRouter(prefix="/api/v1")

_ALLOWED_CONTENT_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
_PHASH_DUPLICATE_THRESHOLD = 10  # Hamming distance — lower = stricter


def _compute_perceptual_hash(image_data: bytes) -> str | None:
    """Compute a perceptual hash (pHash) of the image for duplicate detection.

    Returns the hex-encoded 64-bit pHash string, or ``None`` if hashing fails.
    """
    try:
        from io import BytesIO

        import imagehash
        from PIL import Image

        img = Image.open(BytesIO(image_data))
        return str(imagehash.phash(img))
    except Exception as exc:
        _logger.warning("phash_computation_failed", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Dependency injection helpers — resolve singletons from app.state
# ---------------------------------------------------------------------------


def _get_pipeline(request: Request) -> FlierAnalysisPipeline:
    """Return the pipeline orchestrator from application state."""
    return request.app.state.pipeline


def _get_confirmation_gate(request: Request) -> ConfirmationGate:
    """Return the confirmation gate from application state."""
    return request.app.state.confirmation_gate


def _get_progress_tracker(request: Request) -> ProgressTracker:
    """Return the progress tracker from application state."""
    return request.app.state.progress_tracker


def _get_session_states(request: Request) -> dict[str, PipelineState]:
    """Return the in-memory session state store from application state."""
    return request.app.state.session_states


# Annotated dependency types (avoids B008 / function-call-in-default-argument)
PipelineDep = Annotated[FlierAnalysisPipeline, Depends(_get_pipeline)]
GateDep = Annotated[ConfirmationGate, Depends(_get_confirmation_gate)]
TrackerDep = Annotated[ProgressTracker, Depends(_get_progress_tracker)]
SessionStatesDep = Annotated[dict[str, PipelineState], Depends(_get_session_states)]


def _get_qa_service(request: Request) -> Any:
    """Return the Q&A service from application state, or ``None``."""
    return getattr(request.app.state, "qa_service", None)


def _get_vector_store(request: Request) -> Any:
    """Return the vector store from application state, or ``None``."""
    return getattr(request.app.state, "vector_store", None)


QAServiceDep = Annotated[Any, Depends(_get_qa_service)]
VectorStoreDep = Annotated[Any, Depends(_get_vector_store)]


def _get_feedback_provider(request: Request) -> Any:
    """Return the feedback provider from application state, or ``None``."""
    return getattr(request.app.state, "feedback_provider", None)


FeedbackDep = Annotated[Any, Depends(_get_feedback_provider)]


def _get_flier_history(request: Request) -> Any:
    """Return the flier history provider from application state, or ``None``."""
    return getattr(request.app.state, "flier_history", None)


FlierHistoryDep = Annotated[Any, Depends(_get_flier_history)]


def _js_simple_hash(s: str) -> str:
    """Replicate the frontend ``Rating.simpleHash()`` — 32-bit DJB-style hash as 8-char hex."""
    h = 0
    for ch in s:
        # JS << operates on signed Int32; convert before shifting to match.
        h_signed = h if h < 0x80000000 else h - 0x100000000
        h = (((h_signed << 5) & 0xFFFFFFFF) - h + ord(ch)) & 0xFFFFFFFF
    # JS: Math.abs() on the signed 32-bit result
    h_signed = h if h < 0x80000000 else h - 0x100000000
    return format(abs(h_signed), "x").zfill(8)


def _corpus_item_key(source_title: str, text: str) -> str:
    """Build the same ``item_key`` the frontend uses for corpus rating widgets."""
    return (source_title or "") + "::" + _js_simple_hash(text or "")


# ---------------------------------------------------------------------------
# Background task helpers
# ---------------------------------------------------------------------------


async def _run_phases_2_through_5(
    pipeline: FlierAnalysisPipeline,
    state: PipelineState,
    session_states: dict[str, PipelineState],
    flier_history: Any | None = None,
) -> None:
    """Execute research-through-output phases and persist the final state."""
    try:
        result = await pipeline.run_phases_2_through_5(state)
        session_states[result.session_id] = result

        # Log flier data for cross-flier recommendations
        if flier_history is not None:
            try:
                confirmed = result.confirmed_entities or result.extracted_entities
                await flier_history.log_flier(
                    session_id=result.session_id,
                    artists=[a.text if hasattr(a, 'text') else str(a) for a in (confirmed.artists or [])],
                    venue=confirmed.venue.text if confirmed.venue and hasattr(confirmed.venue, 'text') else None,
                    promoter=confirmed.promoter.text if confirmed.promoter and hasattr(confirmed.promoter, 'text') else None,
                    event_name=confirmed.event_name.text if confirmed.event_name and hasattr(confirmed.event_name, 'text') else None,
                    event_date=str(confirmed.date.text) if confirmed.date and hasattr(confirmed.date, 'text') else None,
                    genre_tags=confirmed.genre_tags if hasattr(confirmed, 'genre_tags') else [],
                )
            except Exception as exc:
                _logger.warning("flier_history_log_failed", session_id=result.session_id, error=str(exc))
    except Exception as exc:
        _logger.error(
            "background_phases_failed",
            session_id=state.session_id,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Flier endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/fliers/upload",
    response_model=FlierUploadResponse,
    responses={413: {"model": ErrorResponse}, 415: {"model": ErrorResponse}},
    summary="Upload a flier image for OCR and entity extraction",
)
async def upload_flier(
    file: UploadFile,
    pipeline: PipelineDep,
    gate: GateDep,
    session_states: SessionStatesDep,
    flier_history: FlierHistoryDep,
) -> FlierUploadResponse:
    """Accept a flier image, run OCR + entity extraction, and return results for review."""
    # --- Validate content type ---
    content_type = file.content_type or ""
    if content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type: {content_type}. "
                f"Allowed: {', '.join(sorted(_ALLOWED_CONTENT_TYPES))}"
            ),
        )

    # --- Read and validate file size ---
    image_data = await file.read()
    if len(image_data) > _MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(image_data)} bytes. Maximum: {_MAX_FILE_SIZE} bytes.",
        )

    # --- Compute perceptual hash for duplicate detection ---
    image_phash = _compute_perceptual_hash(image_data)
    duplicate_match: DuplicateMatch | None = None

    if image_phash and flier_history is not None:
        try:
            match = await flier_history.find_duplicate_by_phash(
                image_phash, threshold=_PHASH_DUPLICATE_THRESHOLD,
            )
            if match is not None:
                duplicate_match = DuplicateMatch(
                    previous_session_id=match["session_id"],
                    similarity=match["similarity"],
                    analyzed_at=match["analyzed_at"],
                    artists=match.get("artists", []),
                    venue=match.get("venue"),
                    event_name=match.get("event_name"),
                    event_date=match.get("event_date"),
                    hamming_distance=match["hamming_distance"],
                )
        except Exception as exc:
            _logger.warning("duplicate_check_failed", error=str(exc))

    # --- Build FlierImage ---
    session_id = str(uuid.uuid4())
    image_hash = hashlib.sha256(image_data).hexdigest()

    flier_image = FlierImage(
        id=session_id,
        filename=file.filename or "unknown",
        content_type=content_type,
        file_size=len(image_data),
        image_hash=image_hash,
        image_phash=image_phash,
    )
    # Attach raw bytes to the private attr on a frozen model
    flier_image.__pydantic_private__["_image_data"] = image_data

    # --- Register perceptual hash for future duplicate detection ---
    if image_phash and flier_history is not None:
        try:
            await flier_history.register_image_hash(session_id, image_phash)
        except Exception as exc:
            _logger.warning("phash_registration_failed", error=str(exc))

    # --- Run Phase 1 (OCR + Entity Extraction) ---
    state = PipelineState(session_id=session_id, flier=flier_image)
    state = await pipeline.run_phase_1(state)
    session_states[session_id] = state

    # --- Submit to confirmation gate for user review ---
    await gate.submit_for_review(state)

    ocr_result = state.ocr_result
    return FlierUploadResponse(
        session_id=session_id,
        extracted_entities=state.extracted_entities,
        ocr_confidence=ocr_result.confidence if ocr_result else 0.0,
        provider_used=ocr_result.provider_used if ocr_result else "unknown",
        duplicate_match=duplicate_match,
    )


@router.post(
    "/fliers/{session_id}/confirm",
    response_model=ConfirmResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Confirm extracted entities and start research",
)
async def confirm_entities(
    session_id: str,
    body: ConfirmEntitiesRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    pipeline: PipelineDep,
    gate: GateDep,
    session_states: SessionStatesDep,
) -> ConfirmResponse:
    """Confirm (and optionally edit) extracted entities, then kick off research."""
    # --- Retrieve pending state ---
    pending_state = await gate.get_pending(session_id)
    if pending_state is None:
        raise HTTPException(status_code=404, detail=f"No pending session: {session_id}")

    # --- Build ExtractedEntities from request ---
    raw_ocr = pending_state.ocr_result or OCRResult(
        raw_text="", confidence=0.0, provider_used="unknown", processing_time=0.0
    )

    artists = [
        ExtractedEntity(text=a.name, entity_type=EntityType(a.entity_type)) for a in body.artists
    ]
    venue = (
        ExtractedEntity(text=body.venue.name, entity_type=EntityType.VENUE) if body.venue else None
    )
    date_entity = (
        ExtractedEntity(text=body.date.name, entity_type=EntityType.DATE) if body.date else None
    )
    promoter = (
        ExtractedEntity(text=body.promoter.name, entity_type=EntityType.PROMOTER)
        if body.promoter
        else None
    )
    event_name = (
        ExtractedEntity(text=body.event_name.name, entity_type=EntityType.EVENT)
        if body.event_name
        else None
    )

    confirmed_entities = ExtractedEntities(
        artists=artists,
        venue=venue,
        date=date_entity,
        promoter=promoter,
        event_name=event_name,
        genre_tags=body.genre_tags,
        ticket_price=body.ticket_price,
        raw_ocr=raw_ocr,
    )

    # --- Confirm through the gate ---
    confirmed_state = await gate.confirm(session_id, confirmed_entities)
    session_states[session_id] = confirmed_state

    # --- Run phases 2-5 in background ---
    flier_history = getattr(request.app.state, "flier_history", None)
    background_tasks.add_task(
        _run_phases_2_through_5,
        pipeline,
        confirmed_state,
        session_states,
        flier_history,
    )

    return ConfirmResponse(
        session_id=session_id,
        status="research_started",
        message="Entities confirmed. Research pipeline started.",
    )


@router.post(
    "/fliers/{session_id}/ask",
    response_model=AskQuestionResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    summary="Ask a question about analysis results",
)
async def ask_question(
    session_id: str,
    body: AskQuestionRequest,
    session_states: SessionStatesDep,
    qa_service: QAServiceDep,
) -> AskQuestionResponse:
    """Ask a follow-up question about the analysis results, answered via RAG + LLM."""
    if qa_service is None:
        raise HTTPException(status_code=503, detail="Q&A service not available")

    state = session_states.get(session_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}. Sessions are cleared on server restart. Please re-upload the flier.",
        )

    # Build session context dict from PipelineState
    session_context: dict[str, Any] = {
        "session_id": session_id,
        "extracted_entities": state.confirmed_entities or state.extracted_entities,
        "research_results": state.research_results,
        "interconnection_map": state.interconnection_map,
    }

    result = await qa_service.ask(
        question=body.question,
        session_context=session_context,
        entity_type=body.entity_type,
        entity_name=body.entity_name,
    )

    facts = [
        RelatedFact(
            text=f.get("text", f) if isinstance(f, dict) else str(f),
            category=f.get("category") if isinstance(f, dict) else None,
            entity_name=f.get("entity_name") if isinstance(f, dict) else None,
        )
        for f in result.related_facts
    ]

    return AskQuestionResponse(
        answer=result.answer,
        citations=result.citations,
        related_facts=facts,
    )


# ---------------------------------------------------------------------------
# Connection correction endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/fliers/{session_id}/dismiss-connection",
    response_model=DismissConnectionResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Dismiss an incorrect interconnection",
)
async def dismiss_connection(
    session_id: str,
    body: DismissConnectionRequest,
    session_states: SessionStatesDep,
) -> DismissConnectionResponse:
    """Mark a specific interconnection relationship as dismissed/incorrect."""
    state = session_states.get(session_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}. Sessions are cleared on server restart. Please re-upload the flier.",
        )

    if state.interconnection_map is None:
        raise HTTPException(status_code=404, detail="No interconnection data available")

    # Find and dismiss matching edges
    dismissed_count = 0
    updated_edges = []
    for edge in state.interconnection_map.edges:
        if (
            edge.source.lower() == body.source.lower()
            and edge.target.lower() == body.target.lower()
            and edge.relationship_type.lower() == body.relationship_type.lower()
        ):
            edge = edge.model_copy(update={"dismissed": True})
            dismissed_count += 1
        updated_edges.append(edge)

    # Update the state with the modified interconnection map
    updated_map = state.interconnection_map.model_copy(update={"edges": updated_edges})
    updated_state = state.model_copy(update={"interconnection_map": updated_map})
    session_states[session_id] = updated_state

    return DismissConnectionResponse(
        session_id=session_id,
        dismissed_count=dismissed_count,
        message=f"Dismissed {dismissed_count} connection(s).",
    )


# ---------------------------------------------------------------------------
# Rating endpoints
# ---------------------------------------------------------------------------

_VALID_RATING_TYPES = frozenset({
    "ARTIST", "VENUE", "PROMOTER", "DATE", "EVENT",
    "CONNECTION", "PATTERN", "QA", "CORPUS",
    "RELEASE", "LABEL", "RECOMMENDATION",
})


@router.post(
    "/fliers/{session_id}/rate",
    response_model=RatingResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    summary="Submit a thumbs up/down rating for a result item",
)
async def submit_rating(
    session_id: str,
    body: SubmitRatingRequest,
    feedback: FeedbackDep,
) -> RatingResponse:
    """Rate a specific result item with thumbs up (+1) or thumbs down (-1)."""
    if feedback is None:
        raise HTTPException(status_code=503, detail="Feedback service not available")
    if body.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="Rating must be +1 or -1")
    if body.item_type.upper() not in _VALID_RATING_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid item_type: {body.item_type}. "
            f"Allowed: {', '.join(sorted(_VALID_RATING_TYPES))}",
        )

    result = await feedback.submit_rating(
        session_id=session_id,
        item_type=body.item_type,
        item_key=body.item_key,
        rating=body.rating,
    )
    return RatingResponse(**result)


@router.get(
    "/fliers/{session_id}/ratings",
    response_model=SessionRatingsResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Get all ratings for a session",
)
async def get_session_ratings(
    session_id: str,
    feedback: FeedbackDep,
) -> SessionRatingsResponse:
    """Return all ratings submitted for this session."""
    if feedback is None:
        raise HTTPException(status_code=503, detail="Feedback service not available")

    ratings = await feedback.get_ratings(session_id)
    return SessionRatingsResponse(
        session_id=session_id,
        ratings=[RatingResponse(**r) for r in ratings],
        total=len(ratings),
    )


@router.get(
    "/ratings/summary",
    response_model=RatingSummaryResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Get aggregate rating statistics",
)
async def get_rating_summary(
    feedback: FeedbackDep,
    item_type: str | None = None,
) -> RatingSummaryResponse:
    """Return aggregate rating statistics across all sessions."""
    if feedback is None:
        raise HTTPException(status_code=503, detail="Feedback service not available")

    summary = await feedback.get_rating_summary(item_type=item_type)
    return RatingSummaryResponse(**summary)


# ---------------------------------------------------------------------------
# Status & results endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/fliers/{session_id}/status",
    response_model=PipelineStatusResponse,
    summary="Get current pipeline progress",
)
async def get_status(
    session_id: str,
    tracker: TrackerDep,
    session_states: SessionStatesDep,
) -> PipelineStatusResponse:
    """Return the current phase, progress percentage, and any errors."""
    status = tracker.get_status(session_id)
    state = session_states.get(session_id)

    errors: list[dict[str, Any]] = []
    if state is not None:
        errors = [
            {
                "phase": e.phase.value,
                "message": e.message,
                "recoverable": e.recoverable,
            }
            for e in state.errors
        ]

    return PipelineStatusResponse(
        session_id=session_id,
        phase=status["phase"],
        progress=status["progress"],
        message=status["message"] or None,
        errors=errors,
    )


@router.get(
    "/fliers/{session_id}/results",
    response_model=FlierAnalysisResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get full analysis results",
)
async def get_results(
    session_id: str,
    session_states: SessionStatesDep,
) -> FlierAnalysisResponse:
    """Return full analysis results if complete, or current progress status."""
    state = session_states.get(session_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}. Sessions are cleared on server restart. Please re-upload the flier.",
        )

    entities = state.confirmed_entities or state.extracted_entities

    if state.completed_at is not None:
        return FlierAnalysisResponse(
            session_id=session_id,
            status="completed",
            extracted_entities=entities,
            research_results=(
                [r.model_dump() for r in state.research_results] if state.research_results else None
            ),
            interconnection_map=(
                state.interconnection_map.model_dump() if state.interconnection_map else None
            ),
            completed_at=state.completed_at,
        )

    return FlierAnalysisResponse(
        session_id=session_id,
        status=state.current_phase.value,
        extracted_entities=entities,
    )


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Application health check",
)
async def health_check(request: Request) -> HealthResponse:
    """Return application health, version, and provider availability."""
    providers: dict[str, Any] = {}
    if hasattr(request.app.state, "provider_registry"):
        providers = dict(request.app.state.provider_registry)

    # Actively verify RAG corpus readiness
    vector_store = getattr(request.app.state, "vector_store", None)
    if vector_store is not None:
        try:
            stats = await vector_store.get_stats()
            providers["rag"] = stats.total_chunks > 0
            providers["rag_chunks"] = stats.total_chunks
        except Exception:
            providers["rag"] = False
            providers["rag_chunks"] = 0

    all_critical_ok = providers.get("llm", False) and providers.get("ocr", False)
    rag_expected = getattr(request.app.state, "rag_enabled", False)
    rag_ok = providers.get("rag", False) or not rag_expected

    if all_critical_ok and rag_ok:
        status = "healthy"
    elif all_critical_ok:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        version="0.1.0",
        providers=providers,
    )


@router.get(
    "/providers",
    response_model=ProvidersResponse,
    summary="List configured providers",
)
async def list_providers(request: Request) -> ProvidersResponse:
    """List all configured providers, their types, and availability status."""
    providers: list[dict[str, Any]] = []
    if hasattr(request.app.state, "provider_list"):
        providers = request.app.state.provider_list

    return ProvidersResponse(providers=providers)


@router.get(
    "/corpus/stats",
    response_model=CorpusStatsResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get RAG corpus statistics",
)
async def corpus_stats(request: Request) -> CorpusStatsResponse:
    """Return aggregate statistics about the RAG vector-store corpus.

    Returns 404 if RAG is not enabled.
    """
    rag_enabled = getattr(request.app.state, "rag_enabled", False)
    if not rag_enabled:
        raise HTTPException(status_code=404, detail="RAG is not enabled")

    ingestion_service = getattr(request.app.state, "ingestion_service", None)
    if ingestion_service is None:
        raise HTTPException(status_code=404, detail="Ingestion service not available")

    stats = await ingestion_service.get_corpus_stats()
    return CorpusStatsResponse(
        total_chunks=stats.total_chunks,
        total_sources=stats.total_sources,
        sources_by_type=stats.sources_by_type,
        entity_tag_count=stats.entity_tag_count,
        geographic_tag_count=stats.geographic_tag_count,
    )


_ARTIST_QUERY_RE = re.compile(
    r"""(?ix)                     # case-insensitive, verbose
    \b(?:artist|artists|dj|djs|producer|producers|musician|musicians|act|acts)
      \s+(?:from|in|who|that|like)\b
    | \bsimilar\s+to\b
    | \bsounds?\s+like\b
    | \bwho\s+(?:play|plays|perform|performs|spins?)\b
    """,
)


def _is_artist_query(query: str) -> bool:
    """Return *True* when the query is explicitly asking for artists."""
    return _ARTIST_QUERY_RE.search(query) is not None


@router.post(
    "/corpus/search",
    response_model=CorpusSearchResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Search the RAG corpus",
)
async def corpus_search(
    body: CorpusSearchRequest,
    request: Request,
    vector_store: VectorStoreDep,
    feedback: FeedbackDep,
) -> CorpusSearchResponse:
    """Perform semantic search against the RAG vector-store corpus.

    Does not require a session — available at any time when RAG is enabled.
    """
    rag_enabled = getattr(request.app.state, "rag_enabled", False)
    if not rag_enabled or vector_store is None:
        raise HTTPException(status_code=503, detail="RAG corpus not available")

    # Build filters from request
    filters: dict[str, Any] = {}
    if body.source_type:
        filters["source_type"] = {"$in": body.source_type}
    if body.entity_tag:
        filters["entity_tags"] = {"$contains": body.entity_tag}
    if body.geographic_tag:
        filters["geographic_tags"] = {"$contains": body.geographic_tag}

    chunks = await vector_store.query(
        query_text=body.query,
        top_k=body.top_k,
        filters=filters if filters else None,
    )

    # Safety-net dedup by source_id — keep top 3 chunks per source
    _MAX_PER_SOURCE = 3
    source_chunks: dict[str, list[CorpusSearchChunk]] = {}
    for c in chunks:
        sid = c.chunk.source_id
        candidate = CorpusSearchChunk(
            text=c.chunk.text,
            source_title=c.chunk.source_title,
            source_type=c.chunk.source_type,
            author=c.chunk.author,
            citation_tier=c.chunk.citation_tier,
            page_number=c.chunk.page_number,
            similarity_score=round(c.similarity_score, 3),
            formatted_citation=c.formatted_citation,
            entity_tags=c.chunk.entity_tags,
            geographic_tags=c.chunk.geographic_tags,
            genre_tags=c.chunk.genre_tags,
        )
        source_chunks.setdefault(sid, []).append(candidate)

    deduped: list[CorpusSearchChunk] = []
    for entries in source_chunks.values():
        entries.sort(key=lambda r: r.similarity_score, reverse=True)
        deduped.extend(entries[:_MAX_PER_SOURCE])

    # Filter out results the user previously thumbs-downed.
    if feedback is not None:
        try:
            negative_keys = await feedback.get_negative_item_keys("CORPUS", "")
            if negative_keys:
                deduped = [
                    r for r in deduped
                    if _corpus_item_key(r.source_title, r.text) not in negative_keys
                ]
        except Exception:
            _logger.debug("Feedback lookup failed for corpus search, skipping filter")

    # When the query is NOT explicitly asking for artists, diversify results
    # so a single entity doesn't dominate (max 2 results per entity tag).
    if not _is_artist_query(body.query):
        _MAX_PER_ENTITY = 2
        entity_counts: dict[str, int] = {}
        diversified: list[CorpusSearchChunk] = []
        # Process in similarity order so we keep the best hits per entity.
        deduped.sort(key=lambda r: r.similarity_score, reverse=True)
        for r in deduped:
            if not r.entity_tags:
                diversified.append(r)
                continue
            # A chunk can have multiple entity tags; gate on the most-seen one.
            capped = False
            for tag in r.entity_tags:
                if entity_counts.get(tag, 0) >= _MAX_PER_ENTITY:
                    capped = True
                    break
            if not capped:
                diversified.append(r)
                for tag in r.entity_tags:
                    entity_counts[tag] = entity_counts.get(tag, 0) + 1
        deduped = diversified

    results = sorted(
        deduped,
        key=lambda r: r.similarity_score,
        reverse=True,
    )

    return CorpusSearchResponse(
        query=body.query,
        total_results=len(results),
        results=results,
    )


# ---------------------------------------------------------------------------
# Recommendation endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/fliers/{session_id}/recommendations",
    response_model=RecommendationsResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    summary="Get artist recommendations based on flier analysis",
)
async def get_recommendations(
    session_id: str,
    request: Request,
    session_states: SessionStatesDep,
) -> RecommendationsResponse:
    """Generate artist recommendations based on the completed flier analysis.

    Uses a three-tier approach: label-mates, shared-flier history, shared
    lineups (from RAG corpus), then LLM reasoning for remaining slots.
    """
    recommendation_service = getattr(request.app.state, "recommendation_service", None)
    if recommendation_service is None:
        raise HTTPException(status_code=503, detail="Recommendation service not available")

    state = session_states.get(session_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}. Sessions are cleared on server restart.",
        )

    if not state.research_results:
        raise HTTPException(status_code=404, detail="No research data available yet. Wait for analysis to complete.")

    entities = state.confirmed_entities or state.extracted_entities

    try:
        result = await recommendation_service.recommend(
            research_results=state.research_results,
            entities=entities,
            interconnection_map=state.interconnection_map,
        )
    except Exception as exc:
        _logger.error("recommendation_failed", session_id=session_id, error=str(exc))
        raise HTTPException(status_code=503, detail="Failed to generate recommendations") from exc

    return RecommendationsResponse(
        session_id=session_id,
        recommendations=[
            RecommendedArtistResponse(
                artist_name=r.artist_name,
                genres=r.genres,
                reason=r.reason,
                source_tier=r.source_tier,
                connection_strength=r.connection_strength,
                connected_to=r.connected_to,
                label_name=r.label_name,
                event_name=r.event_name,
            )
            for r in result.recommendations
        ],
        flier_artists=result.flier_artists,
        genres_analyzed=result.genres_analyzed,
        total=len(result.recommendations),
        is_partial=False,
    )


@router.get(
    "/fliers/{session_id}/recommendations/quick",
    response_model=RecommendationsResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    summary="Get fast label-mate recommendations (no LLM)",
)
async def get_recommendations_quick(
    session_id: str,
    request: Request,
    session_states: SessionStatesDep,
) -> RecommendationsResponse:
    """Return label-mate recommendations only, without any LLM calls.

    Designed for instant display while the full recommendation pipeline
    runs in background.  Returns ``is_partial=True`` so the frontend
    knows to backfill with the full endpoint.
    """
    recommendation_service = getattr(request.app.state, "recommendation_service", None)
    if recommendation_service is None:
        raise HTTPException(status_code=503, detail="Recommendation service not available")

    state = session_states.get(session_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}. Sessions are cleared on server restart.",
        )

    if not state.research_results:
        raise HTTPException(
            status_code=404,
            detail="No research data available yet. Wait for analysis to complete.",
        )

    entities = state.confirmed_entities or state.extracted_entities

    try:
        result = await recommendation_service.recommend_quick(
            research_results=state.research_results,
            entities=entities,
        )
    except Exception as exc:
        _logger.error("quick_recommendation_failed", session_id=session_id, error=str(exc))
        raise HTTPException(status_code=503, detail="Failed to generate quick recommendations") from exc

    return RecommendationsResponse(
        session_id=session_id,
        recommendations=[
            RecommendedArtistResponse(
                artist_name=r.artist_name,
                genres=r.genres,
                reason=r.reason,
                source_tier=r.source_tier,
                connection_strength=r.connection_strength,
                connected_to=r.connected_to,
                label_name=r.label_name,
                event_name=r.event_name,
            )
            for r in result.recommendations
        ],
        flier_artists=result.flier_artists,
        genres_analyzed=result.genres_analyzed,
        total=len(result.recommendations),
        is_partial=True,
    )

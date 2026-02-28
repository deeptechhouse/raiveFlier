"""FastAPI API routes for the RaiveFlier pipeline.

Provides REST endpoints for flier upload, entity confirmation, progress
polling, result retrieval, health checks, and provider listing.  Service
dependencies are resolved from ``app.state`` via FastAPI's ``Depends``
using the ``Annotated`` pattern.

# ─── API ROUTE MAP (Junior Developer Guide) ───────────────────────────
#
# Endpoint                              Method  Description
# ─────────────────────────────────────────────────────────────────────
# /api/v1/fliers/upload                 POST    Upload flier → OCR → extraction
# /api/v1/fliers/{sid}/confirm          POST    Confirm entities → start research
# /api/v1/fliers/{sid}/status           GET     Poll pipeline progress
# /api/v1/fliers/{sid}/results          GET     Fetch full analysis results
# /api/v1/fliers/{sid}/ask              POST    Ask Q&A about results (RAG)
# /api/v1/fliers/{sid}/dismiss-connection POST   Dismiss bad interconnection edge
# /api/v1/fliers/{sid}/rate             POST    Submit thumbs up/down rating
# /api/v1/fliers/{sid}/ratings          GET     Get all ratings for session
# /api/v1/fliers/{sid}/recommendations  GET     Artist recommendations (on-demand)
# /api/v1/ratings/summary               GET     Aggregate rating stats
# /api/v1/health                        GET     Health check + provider status
# /api/v1/providers                     GET     List all configured providers
# /api/v1/corpus/stats                  GET     RAG corpus statistics
# /api/v1/corpus/search                 POST    Semantic search of RAG corpus
#
# DEPENDENCY INJECTION PATTERN:
# Each route function declares its dependencies as type-annotated params.
# FastAPI resolves these via Depends() which calls helper functions that
# read from app.state (populated at startup in main.py's _build_all).
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import uuid
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, UploadFile

from src.api.schemas import (
    AskQuestionRequest,
    AskQuestionResponse,
    ConfirmEntitiesRequest,
    ConfirmResponse,
    CorpusSearchChunk,
    CorpusSearchCitation,
    CorpusSearchRequest,
    CorpusSearchResponse,
    CorpusStatsResponse,
    DismissConnectionRequest,
    DismissConnectionResponse,
    DuplicateMatch,
    ErrorResponse,
    FacetCounts,
    FlierAnalysisResponse,
    FlierUploadResponse,
    HealthResponse,
    ParseQueryRequest,
    ParsedQueryFilters,
    PipelineStatusResponse,
    ProvidersResponse,
    RatingResponse,
    RatingSummaryResponse,
    RecommendationsResponse,
    RecommendedArtistResponse,
    RelatedFact,
    SessionRatingsResponse,
    SubmitRatingRequest,
    StoredAnalysisResponse,
    AnalysisListResponse,
    AddAnnotationRequest,
    AnnotationResponse,
    AnnotationListResponse,
    SuggestResponse,
    WebSearchResult,
)
from src.models.entities import EntityType
from src.models.flier import ExtractedEntities, ExtractedEntity, FlierImage, OCRResult
from src.models.pipeline import PipelineState
from src.pipeline.confirmation_gate import ConfirmationGate
from src.pipeline.orchestrator import FlierAnalysisPipeline
from src.pipeline.progress_tracker import ProgressTracker
from src.utils.logging import get_logger

_logger: structlog.BoundLogger = get_logger(__name__)

# All routes in this file are prefixed with /api/v1.
# Example: @router.post("/fliers/upload") → POST /api/v1/fliers/upload
router = APIRouter(prefix="/api/v1")

# --- Upload validation constants ---
# frozenset is an immutable set — perfect for constants that should never change.
_ALLOWED_CONTENT_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
_MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB — lowered from 10 MB to reduce peak memory on 512 MB Render

# Chunk size for streaming uploads — read in 64 KB increments to reject
# oversized files early without buffering the entire payload into memory.
_UPLOAD_CHUNK_SIZE = 64 * 1024  # 64 KB

# Maximum image dimension (pixels) for OCR processing.  Images larger than
# this are downscaled immediately after upload to cap uncompressed bitmap
# memory.  OCR accuracy does not benefit from >2048px — Tesseract and LLM
# Vision both work well at this resolution.
_MAX_IMAGE_DIM = 2048

# Perceptual hash duplicate threshold: Hamming distance ≤ 10 means the
# images look visually similar.  Lower = stricter (0 = pixel-identical).
# A threshold of 10 allows for minor cropping, compression, color shifts.
_PHASH_DUPLICATE_THRESHOLD = 10  # Hamming distance — lower = stricter

# ── Memory protection: upload processing semaphore ──────────────────
# Limits concurrent flier upload+OCR processing to 1 at a time.  On the
# 512 MB Render Starter instance, a single upload can spike ~150-200 MB
# during image preprocessing (7 OCR variants from a high-res photo).
# Two concurrent uploads would exceed the memory budget and trigger an
# OOM kill.  Subsequent uploads queue behind the semaphore — latency
# increases but the process stays alive.
_UPLOAD_SEMAPHORE = asyncio.Semaphore(1)

# Recommendation endpoint timeouts.
# Quick mode: SQLite + simple LLM only (~3-5 seconds expected).
_RECO_QUICK_TIMEOUT = 15.0  # seconds
# Full mode: Discogs + RAG + full LLM fill+explain (~10-15 seconds expected).
_RECO_TIMEOUT = 60.0  # seconds


def _downscale_if_oversized(image_data: bytes, max_dim: int) -> bytes:
    """Downscale an image if its largest dimension exceeds *max_dim* pixels.

    Returns the original bytes unchanged if the image is already within
    bounds.  Re-encodes as high-quality JPEG (95%) after downscaling to
    minimize the byte buffer held in memory downstream.

    This runs BEFORE OCR preprocessing so the uncompressed bitmap that
    PIL/OpenCV create from the bytes is bounded to ~2048×2048 (~12 MB)
    instead of potentially 4000×3000 (~36 MB).
    """
    import io as _io

    from PIL import Image as _PILImage

    try:
        img = _PILImage.open(_io.BytesIO(image_data)).convert("RGB")
        largest = max(img.size)
        if largest <= max_dim:
            return image_data

        scale = max_dim / largest
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, _PILImage.LANCZOS)

        buf = _io.BytesIO()
        # JPEG at quality=95 keeps OCR-relevant detail while being much
        # smaller than PNG for photographic content.
        img.save(buf, format="JPEG", quality=95)
        _logger.info(
            "image_downscaled_for_memory",
            original_largest_dim=largest,
            new_size=new_size,
            original_bytes=len(image_data),
            new_bytes=buf.tell(),
        )
        return buf.getvalue()
    except Exception as exc:
        # If downscaling fails for any reason, proceed with the original
        # bytes — OCR will still work, just with higher memory usage.
        _logger.warning("image_downscale_failed", error=str(exc))
        return image_data


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
# JUNIOR DEV NOTE — FastAPI Dependency Injection
# -----------------------------------------------
# FastAPI uses Depends() to inject services into route handlers.
# The pattern:
#   1. Write a helper function that extracts a service from app.state
#   2. Create an Annotated type alias: XDep = Annotated[XType, Depends(helper)]
#   3. Declare XDep as a route param → FastAPI calls helper() automatically
#
# This avoids importing app.state directly in route functions, making
# them easier to test (just pass mock objects as params).
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


# Annotated dependency types.  PEP 593 Annotated[T, Depends(fn)] is the
# modern FastAPI pattern — avoids B008 linter warnings about function
# calls in default argument values.
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


def _get_llm_provider(request: Request) -> Any:
    """Return the LLM provider from application state, or ``None``."""
    return getattr(request.app.state, "primary_llm", None)


LLMProviderDep = Annotated[Any, Depends(_get_llm_provider)]


def _get_web_search(request: Request) -> Any:
    """Return the web search provider from application state, or ``None``.

    Used by the corpus sidebar's web-search tier to augment RAG results
    with live DuckDuckGo results filtered for music relevance.
    """
    return getattr(request.app.state, "web_search", None)


WebSearchDep = Annotated[Any, Depends(_get_web_search)]


def _get_flier_history(request: Request) -> Any:
    """Return the flier history provider from application state, or ``None``."""
    return getattr(request.app.state, "flier_history", None)


FlierHistoryDep = Annotated[Any, Depends(_get_flier_history)]


def _js_simple_hash(s: str) -> str:
    """Replicate the frontend ``Rating.simpleHash()`` — 32-bit DJB-style hash as 8-char hex.

    # JUNIOR DEV NOTE — Why replicate a JS hash in Python?
    # The frontend generates item_key values using a simple hash function
    # in rating.js.  The backend needs to match those keys exactly when
    # filtering out negatively-rated corpus items.  The bit manipulation
    # mimics JavaScript's signed 32-bit integer overflow behavior.
    """
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
    """Execute research-through-output phases and persist the final state.

    # JUNIOR DEV NOTE — BackgroundTasks
    # This function runs as a FastAPI BackgroundTask, meaning it executes
    # AFTER the HTTP response has been sent to the client.  The user gets
    # an instant "research_started" response, then this runs in the
    # background while the frontend polls /status or listens on WebSocket.
    """
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

            # Store the full InterconnectionMap for permanent retention.
            if result.interconnection_map is not None:
                try:
                    map_dict = result.interconnection_map.model_dump()
                    research_dicts = None
                    if result.research_results:
                        research_dicts = [r.model_dump() for r in result.research_results]
                    await flier_history.store_analysis(
                        session_id=result.session_id,
                        interconnection_map=map_dict,
                        research_results=research_dicts,
                    )
                except Exception as exc:
                    _logger.warning("analysis_store_failed", session_id=result.session_id, error=str(exc))

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
    # --- Validate content type (security: reject non-image files) ---
    content_type = file.content_type or ""
    if content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type: {content_type}. "
                f"Allowed: {', '.join(sorted(_ALLOWED_CONTENT_TYPES))}"
            ),
        )

    # --- Stream upload in chunks — reject oversized files early -------
    # Reading in 64 KB chunks means a 50 MB upload is rejected after
    # buffering only 5 MB, instead of loading all 50 MB into memory.
    chunks: list[bytes] = []
    total_size = 0
    while True:
        chunk = await file.read(_UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > _MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large: >{_MAX_FILE_SIZE // (1024 * 1024)} MB. "
                    f"Maximum: {_MAX_FILE_SIZE} bytes."
                ),
            )
        chunks.append(chunk)
    image_data = b"".join(chunks)
    # Free the chunk list immediately — image_data now owns the bytes.
    del chunks

    # --- Compute hashes on ORIGINAL bytes (before any downscaling) ----
    # Perceptual hash and SHA-256 must be computed on the original image
    # so duplicate detection works consistently regardless of resize logic.
    image_phash = _compute_perceptual_hash(image_data)
    image_hash = hashlib.sha256(image_data).hexdigest()

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
                    times_analyzed=match.get("times_analyzed", 1),
                )
        except Exception as exc:
            _logger.warning("duplicate_check_failed", error=str(exc))

    # --- Early resize: cap image dimensions before OCR processing -----
    # A 4000×3000 JPEG (5 MB compressed) becomes ~36 MB as an uncompressed
    # RGB bitmap, and then 7 OCR preprocessing variants push peak memory
    # to ~250 MB.  Downscaling to 2048px max before any processing cuts
    # uncompressed size to ~12 MB and peak preprocessing to ~85 MB.
    image_data = _downscale_if_oversized(image_data, _MAX_IMAGE_DIM)

    # --- Acquire upload semaphore for memory-intensive processing ------
    # Only one upload goes through OCR + preprocessing at a time to
    # prevent concurrent memory spikes from exceeding the 512 MB budget.
    async with _UPLOAD_SEMAPHORE:
        # --- Build FlierImage ---
        session_id = str(uuid.uuid4())

        flier_image = FlierImage(
            id=session_id,
            filename=file.filename or "unknown",
            content_type=content_type,
            file_size=len(image_data),
            image_hash=image_hash,
            image_phash=image_phash,
        )
        # Attach raw image bytes to the PrivateAttr on the frozen Pydantic model.
        # PrivateAttr fields bypass Pydantic's immutability checks, allowing us
        # to store the binary data without it appearing in serialization (JSON).
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
    # times_analyzed: previous analyses + this one (1 for first-time fliers)
    times_analyzed = (duplicate_match.times_analyzed + 1) if duplicate_match else 1
    return FlierUploadResponse(
        session_id=session_id,
        extracted_entities=state.extracted_entities,
        ocr_confidence=ocr_result.confidence if ocr_result else 0.0,
        provider_used=ocr_result.provider_used if ocr_result else "unknown",
        duplicate_match=duplicate_match,
        times_analyzed=times_analyzed,
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
    request: Request,
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

    # Persist dismissal permanently in flier_history.
    flier_history = getattr(request.app.state, "flier_history", None)
    if flier_history is not None:
        try:
            await flier_history.persist_edge_dismissal(
                session_id=session_id,
                source=body.source,
                target=body.target,
                relationship_type=body.relationship_type,
                reason=body.reason,
            )
        except Exception as exc:
            _logger.warning("edge_dismissal_persist_failed", session_id=session_id, error=str(exc))

    return DismissConnectionResponse(
        session_id=session_id,
        dismissed_count=dismissed_count,
        message=f"Dismissed {dismissed_count} connection(s).",
    )


# ---------------------------------------------------------------------------
# Persistent analysis storage endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/fliers/{session_id}/analysis",
    response_model=StoredAnalysisResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Retrieve stored analysis for a session",
)
async def get_stored_analysis(
    session_id: str,
    request: Request,
    include_research: bool = False,
) -> StoredAnalysisResponse:
    """Retrieve the permanently stored analysis snapshot for a session."""
    flier_history = getattr(request.app.state, "flier_history", None)
    if flier_history is None:
        raise HTTPException(status_code=503, detail="Flier history service unavailable")

    analysis = await flier_history.get_analysis(session_id, include_research=include_research)
    if analysis is None:
        raise HTTPException(status_code=404, detail=f"No stored analysis for session: {session_id}")

    return StoredAnalysisResponse(
        session_id=analysis["session_id"],
        flier_id=analysis["flier_id"],
        interconnection_map=analysis["interconnection_map"],
        research_results=analysis.get("research_results"),
        revision=analysis.get("revision", 1),
        created_at=analysis.get("created_at"),
    )


@router.get(
    "/analyses",
    response_model=AnalysisListResponse,
    summary="List all stored analyses",
)
async def list_analyses(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> AnalysisListResponse:
    """List all permanently stored analyses with pagination."""
    flier_history = getattr(request.app.state, "flier_history", None)
    if flier_history is None:
        raise HTTPException(status_code=503, detail="Flier history service unavailable")

    analyses = await flier_history.list_analyses(limit=limit, offset=offset)
    return AnalysisListResponse(
        analyses=analyses,
        total=len(analyses),
        offset=offset,
        limit=limit,
    )


@router.post(
    "/fliers/{session_id}/analysis/annotate",
    response_model=AnnotationResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Add annotation to stored analysis",
)
async def add_analysis_annotation(
    session_id: str,
    body: AddAnnotationRequest,
    request: Request,
) -> AnnotationResponse:
    """Add a user annotation to a stored analysis."""
    flier_history = getattr(request.app.state, "flier_history", None)
    if flier_history is None:
        raise HTTPException(status_code=503, detail="Flier history service unavailable")

    try:
        result = await flier_history.add_annotation(
            session_id=session_id,
            note=body.note,
            target_type=body.target_type,
            target_key=body.target_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return AnnotationResponse(**result)


@router.get(
    "/fliers/{session_id}/analysis/annotations",
    summary="Get annotations for stored analysis",
)
async def get_analysis_annotations(
    session_id: str,
    request: Request,
) -> AnnotationListResponse:
    """Get all annotations for a session's stored analysis."""
    flier_history = getattr(request.app.state, "flier_history", None)
    if flier_history is None:
        raise HTTPException(status_code=503, detail="Flier history service unavailable")

    annotations = await flier_history.get_annotations(session_id)
    return AnnotationListResponse(
        session_id=session_id,
        annotations=annotations,
        total=len(annotations),
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
        # Genre tags and time periods for frontend filter dropdowns
        genre_tags=stats.genre_tags,
        time_periods=stats.time_periods,
        # Full tag lists for autocomplete suggestions
        entity_tags=stats.entity_tags_list,
        geographic_tags=stats.geographic_tags_list,
    )


# ---------------------------------------------------------------------------
# Parse-query endpoint — extracts structured filters from free-text queries.
# Uses domain_knowledge functions only (no LLM, no I/O), so response is <5ms.
# ---------------------------------------------------------------------------


@router.post(
    "/corpus/parse-query",
    response_model=ParsedQueryFilters,
    summary="Parse natural language query into structured filters",
)
async def parse_query(body: ParseQueryRequest) -> ParsedQueryFilters:
    """Extract genre, temporal, geographic, and artist signals from free text.

    Uses domain_knowledge functions (pure, no I/O, no LLM calls) for fast
    (<5ms) parsing.  Called by the frontend on debounce to auto-fill filter
    controls with detected signals before the search request fires.
    """
    try:
        from src.config.domain_knowledge import (
            detect_temporal_signal,
            expand_aliases,
            extract_query_genres,
            get_canonical_name,
            get_scene_geographies,
        )
    except ImportError:
        # domain_knowledge module not available — return empty result
        return ParsedQueryFilters()

    query = body.query

    # Genre extraction — longest-match-first prevents partial matches
    genres = sorted(extract_query_genres(query))

    # Temporal signal — named eras, decade strings, year ranges
    time_period = detect_temporal_signal(query)

    # Geographic signals — scene keywords mapped to cities/regions
    geo_tags = sorted(get_scene_geographies(query))

    # Artist alias resolution — canonical name + all known aliases
    canonical = get_canonical_name(query.strip())
    aliases = sorted(expand_aliases(canonical)) if canonical else []

    return ParsedQueryFilters(
        genres=genres,
        time_period=time_period,
        geographic_tags=geo_tags,
        artist_canonical=canonical,
        artist_aliases=aliases,
    )


# ---------------------------------------------------------------------------
# Suggest endpoint — autocomplete with prefix + fuzzy matching for filter
# text inputs.  Uses cached tag lists from get_stats() plus difflib for
# typo tolerance.  No new dependencies required.
# ---------------------------------------------------------------------------


@router.get(
    "/corpus/suggest",
    response_model=SuggestResponse,
    summary="Autocomplete suggestions for filter fields",
)
async def corpus_suggest(
    field: str,
    prefix: str = "",
    request: Request = ...,
) -> SuggestResponse:
    """Return autocomplete suggestions for a corpus filter field.

    Combines prefix matching, substring matching, and fuzzy matching
    (via stdlib difflib.get_close_matches) to tolerate typos.
    For the entity_tag field, also searches artist alias names from
    domain_knowledge.

    Depends on the cached stats from get_stats() — no additional DB
    queries are issued, keeping response time under ~5ms.
    """
    import difflib

    rag_enabled = getattr(request.app.state, "rag_enabled", False)
    if not rag_enabled:
        return SuggestResponse(field=field, prefix=prefix, suggestions=[])

    # Resolve the vector store from app.state (same pattern as corpus_search)
    vector_store = getattr(request.app.state, "vector_store", None)
    if vector_store is None:
        return SuggestResponse(field=field, prefix=prefix, suggestions=[])

    try:
        stats = await vector_store.get_stats()
    except Exception:
        return SuggestResponse(field=field, prefix=prefix, suggestions=[])

    # Select the tag list based on the requested field
    tag_lists: dict[str, list[str]] = {
        "entity_tag": stats.entity_tags_list,
        "geographic_tag": stats.geographic_tags_list,
        "genre_tag": stats.genre_tags,
    }
    candidates = tag_lists.get(field, [])

    if not prefix:
        # Empty prefix — return most common tags (already sorted alphabetically)
        return SuggestResponse(
            field=field, prefix=prefix, suggestions=candidates[:30]
        )

    prefix_lower = prefix.lower()

    # Phase 1: Prefix match (case-insensitive startswith)
    prefix_matches = [t for t in candidates if t.lower().startswith(prefix_lower)]

    # Phase 2: Contains match (substring, excluding already-matched)
    prefix_set = set(prefix_matches)
    contains_matches = [
        t
        for t in candidates
        if prefix_lower in t.lower() and t not in prefix_set
    ]

    combined = prefix_matches + contains_matches

    # Phase 3: Fuzzy match via difflib for typo tolerance — only if we
    # haven't found enough results from exact/substring matching.
    if len(combined) < 10:
        fuzzy_hits = difflib.get_close_matches(
            prefix_lower,
            [t.lower() for t in candidates],
            n=10,
            cutoff=0.6,
        )
        # Map lowercase fuzzy hits back to original-case tag strings
        lower_to_original = {t.lower(): t for t in candidates}
        combined_set = set(combined)
        for hit in fuzzy_hits:
            original = lower_to_original.get(hit)
            if original and original not in combined_set:
                combined.append(original)
                combined_set.add(original)

    # For entity_tag field: also search artist aliases from domain_knowledge
    if field == "entity_tag" and prefix_lower:
        try:
            from src.config.domain_knowledge import get_all_artist_names

            all_names = get_all_artist_names()
            combined_set = set(combined)
            alias_matches = sorted(
                n
                for n in all_names
                if n.lower().startswith(prefix_lower) and n not in combined_set
            )
            combined.extend(alias_matches[:5])
        except ImportError:
            pass

    return SuggestResponse(
        field=field, prefix=prefix, suggestions=combined[:20]
    )


_ARTIST_QUERY_RE = re.compile(
    r"""(?ix)                     # case-insensitive, verbose
    \b(?:artist|artists|dj|djs|producer|producers|musician|musicians|act|acts)
      \s+(?:from|in|who|that|like)\b
    | \bsimilar\s+to\b
    | \bsounds?\s+like\b
    | \bwho\s+(?:play|plays|perform|performs|spins?)\b
    | \bwho\s+is\b
    | \btell\s+me\s+about\b
    | \bbiography\b
    | \bprofile\s+of\b
    | \bcareer\s+of\b
    """,
)

# Cached set of known artist names — populated lazily on first search.
_artist_name_cache: set[str] | None = None


def _build_artist_name_cache() -> set[str]:
    """Build (or return cached) set of known artist names from the alias table."""
    global _artist_name_cache  # noqa: PLW0603
    if _artist_name_cache is not None:
        return _artist_name_cache
    try:
        from src.config.domain_knowledge import get_all_artist_names
        _artist_name_cache = {n.lower() for n in get_all_artist_names()}
    except ImportError:
        _artist_name_cache = set()
    return _artist_name_cache


def _is_artist_query(query: str) -> bool:
    """Return *True* when the query is explicitly asking for artists.

    Uses both regex pattern matching and a lookup against known artist
    names from the alias table.
    """
    if _ARTIST_QUERY_RE.search(query):
        return True
    # Check if query matches a known artist name (exact or substring).
    query_lower = query.strip().lower()
    for name in _build_artist_name_cache():
        if name == query_lower or (len(name) > 4 and name in query_lower):
            return True
    return False


_EXPAND_SYSTEM_PROMPT = "You are a rave and electronic music culture search assistant."
_EXPAND_USER_PROMPT = (
    "Expand this search query with 2-3 related terms from electronic/dance music culture. "
    "Return ONLY the expanded query on a single line, no explanation.\n\nQuery: {query}"
)
# LRU cache for query expansions -- avoids repeated LLM calls for identical queries.
_expansion_cache: dict[str, str] = {}
_EXPANSION_CACHE_MAX = 128


async def _expand_query(llm: Any, query: str) -> str:
    """Use the LLM to expand a short query with related rave/electronic music terms.

    Skips expansion for long queries (>80 chars) or when LLM is unavailable.
    Results are cached in an LRU dict to avoid repeated LLM round-trips.
    """
    if llm is None or len(query) > 80:
        return query
    if query in _expansion_cache:
        return _expansion_cache[query]
    try:
        expanded = await llm.complete(
            system_prompt=_EXPAND_SYSTEM_PROMPT,
            user_prompt=_EXPAND_USER_PROMPT.format(query=query),
            temperature=0.0,
            max_tokens=100,
        )
        result = expanded.strip() if expanded else query
    except Exception:
        _logger.debug("query_expansion_failed", query_preview=query[:50])
        result = query
    # LRU eviction: remove oldest entry if cache is full.
    if len(_expansion_cache) >= _EXPANSION_CACHE_MAX:
        _expansion_cache.pop(next(iter(_expansion_cache)))
    _expansion_cache[query] = result
    return result


def _semantic_dedup(
    results: list[CorpusSearchChunk],
    threshold: float = 0.85,
) -> list[CorpusSearchChunk]:
    """Remove near-duplicate chunks based on word-level Jaccard similarity.

    Compares each candidate against already-kept results.  If the token
    overlap exceeds *threshold*, the candidate is dropped (the kept result
    already covers the same content and has a better score/tier).

    Parameters
    ----------
    results:
        Chunks pre-sorted by score (best first).
    threshold:
        Jaccard similarity above which two chunks are considered duplicates.
    """
    if len(results) <= 1:
        return results

    kept: list[CorpusSearchChunk] = [results[0]]
    for candidate in results[1:]:
        c_tokens = set(candidate.text.lower().split())
        if not c_tokens:
            kept.append(candidate)
            continue
        is_dup = False
        for existing in kept:
            e_tokens = set(existing.text.lower().split())
            if not e_tokens:
                continue
            intersection = len(c_tokens & e_tokens)
            union = len(c_tokens | e_tokens)
            if union > 0 and intersection / union > threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(candidate)
    return kept


def _compute_facets(chunks: list[CorpusSearchChunk]) -> FacetCounts:
    """Count metadata values across the candidate pool for faceted navigation.

    Called after vector retrieval + per-source dedup but BEFORE user-applied
    filters, so facets show what is available in the query's semantic
    neighborhood regardless of active filter settings.  Entity and geographic
    facets are truncated to the top-30 / top-20 by count to bound response
    size; source_type, genre, time_period, and citation_tier facets are
    exhaustive (small cardinality).
    """
    source_types: dict[str, int] = {}
    genre_tags: dict[str, int] = {}
    time_periods: dict[str, int] = {}
    entity_tags: dict[str, int] = {}
    geographic_tags: dict[str, int] = {}
    citation_tiers: dict[str, int] = {}

    for c in chunks:
        if c.source_type:
            source_types[c.source_type] = source_types.get(c.source_type, 0) + 1

        for g in c.genre_tags:
            genre_tags[g] = genre_tags.get(g, 0) + 1

        if c.time_period:
            time_periods[c.time_period] = time_periods.get(c.time_period, 0) + 1

        for e in c.entity_tags:
            entity_tags[e] = entity_tags.get(e, 0) + 1

        for geo in c.geographic_tags:
            geographic_tags[geo] = geographic_tags.get(geo, 0) + 1

        key = f"T{c.citation_tier}"
        citation_tiers[key] = citation_tiers.get(key, 0) + 1

    # Truncate high-cardinality facets to top-N by count
    top_entities = dict(sorted(entity_tags.items(), key=lambda x: -x[1])[:30])
    top_geos = dict(sorted(geographic_tags.items(), key=lambda x: -x[1])[:20])

    return FacetCounts(
        source_types=source_types,
        genre_tags=genre_tags,
        time_periods=time_periods,
        entity_tags=top_entities,
        geographic_tags=top_geos,
        citation_tiers=citation_tiers,
    )


# ---------------------------------------------------------------------------
# Tiered corpus query — priority-ordered ChromaDB retrieval
# ---------------------------------------------------------------------------
# The tiered strategy queries ChromaDB four times in sequence, each time
# targeting a different source priority:
#   Tier 1: RA Exchange podcast transcripts (source_type="interview",
#           source_title starts with "EX.")
#   Tier 2: Books (source_type="book")
#   Tier 3a: Event listings (source_type="event") — dedicated
#            query with its own budget so the 69K event chunks don't
#            crowd out other sources in the catch-all tier.
#   Tier 3b: Catch-all — everything not already captured by T1/T2/T3a,
#            including reference docs and any future source types.
# Chunks are deduped across tiers via a seen_ids set so the same
# source never appears in two tiers.
#
# Returns (all_chunks, chunk_tier_map, tiers_used) where chunk_tier_map
# maps id(chunk) → tier number for downstream provenance labeling.


async def _tiered_corpus_query(
    vector_store: Any,
    query_text: str,
    user_filters: dict[str, Any] | None,
) -> tuple[list[Any], dict[int, int], list[int]]:
    """Execute priority-ordered ChromaDB queries across four sub-tiers.

    The query budget is split so that the 69K event chunks
    (79% of the corpus) cannot crowd out smaller but higher-authority
    source types.  Each tier has its own fetch/keep budget:

        T1  interviews (RA Exchange)   fetch 25 → keep 10
        T2  books                       fetch 20 → keep 10
        T3a event listings              fetch 25 → keep 10
        T3b catch-all (reference, etc.) fetch 25 → keep 12

    T3a and T3b both map to tier number 3 for downstream labeling.

    Parameters
    ----------
    vector_store:
        The ChromaDB vector store instance.
    query_text:
        The (possibly expanded) search query text.
    user_filters:
        User-applied ChromaDB where-clause filters (entity, geo, genre,
        etc.) to be merged into each tier's query.  When the user's
        source_type filter excludes a tier's designated type, that tier
        is skipped entirely.

    Returns
    -------
    tuple of (all_chunks, chunk_tier_map, tiers_used)
        - all_chunks: flat list of retrieval results across all tiers
        - chunk_tier_map: ``{id(chunk): tier_number}`` for provenance
        - tiers_used: list of tier numbers that produced results
    """
    all_chunks: list[Any] = []
    chunk_tier_map: dict[int, int] = {}
    seen_ids: set[str] = set()
    tiers_used: list[int] = []

    # Helper to merge user filters with tier-specific filters.
    # Tier-specific source_type filters are authoritative — if the user
    # also supplied a source_type filter, the intersection is used (the
    # tier only returns its designated type, further narrowed by the user
    # selection).  If the user's $in list doesn't include the tier's
    # type, the tier query is skipped (returns None to signal "skip").
    def _merge_filters(tier_filter: dict[str, Any]) -> dict[str, Any] | None:
        merged = dict(tier_filter)
        if not user_filters:
            return merged if merged else None

        # source_type needs special merge: tier filter is authoritative,
        # user filter narrows within it.
        tier_st = tier_filter.get("source_type")
        user_st = user_filters.get("source_type")
        # Copy non-source_type user filters first
        for k, v in user_filters.items():
            if k != "source_type":
                merged[k] = v
        # Merge source_type: if tier specifies a type and user also
        # filters by type, check compatibility.
        if tier_st and user_st and isinstance(user_st, dict):
            user_in = user_st.get("$in", [])
            if isinstance(tier_st, str):
                # Tier wants one specific type — keep it only if user
                # allows it.  If user's $in list doesn't include it,
                # this tier is irrelevant.
                if user_in and tier_st not in user_in:
                    return None  # signal: skip this tier entirely
                # Otherwise tier's exact filter is more specific; keep it
            elif isinstance(tier_st, dict) and "$nin" in tier_st:
                # Tier 3b exclusion list — intersect with user's $in.
                # If no user types survive the exclusion, skip this tier.
                if user_in:
                    surviving = [
                        t for t in user_in
                        if t not in tier_st["$nin"]
                    ]
                    if not surviving:
                        return None  # all user types excluded by this tier
                    merged["source_type"] = {"$in": surviving}
                # If no user $in, keep the tier's $nin as-is
        elif not tier_st and user_st:
            # No tier filter, just apply user's
            merged["source_type"] = user_st

        return merged if merged else None

    # --- Tier 1: RA Exchange interviews ---
    # ChromaDB query filtered to source_type="interview"; post-filter to
    # source_title starting with "EX." (the RA Exchange episode prefix).
    # _merge_filters returns None when the user's source_type filter
    # excludes this tier's type — skip the query entirely in that case.
    t1_filters = _merge_filters({"source_type": "interview"})
    if t1_filters is not None:
        try:
            t1_raw = await vector_store.query(
                query_text=query_text, top_k=25, filters=t1_filters,
            )
            # Post-filter: only keep RA Exchange episodes (source_title starts with "EX.")
            t1_chunks = [
                c for c in t1_raw
                if getattr(c.chunk, "source_title", "").startswith("EX.")
                and c.chunk.source_id not in seen_ids
            ][:10]
            for c in t1_chunks:
                seen_ids.add(c.chunk.source_id)
                chunk_tier_map[id(c)] = 1
            all_chunks.extend(t1_chunks)
            if t1_chunks:
                tiers_used.append(1)
        except Exception:
            _logger.debug("tiered_query_tier1_failed")

    # --- Tier 2: Books ---
    t2_filters = _merge_filters({"source_type": "book"})
    if t2_filters is not None:
        try:
            t2_raw = await vector_store.query(
                query_text=query_text, top_k=20, filters=t2_filters,
            )
            t2_chunks = [
                c for c in t2_raw
                if c.chunk.source_id not in seen_ids
            ][:10]
            for c in t2_chunks:
                seen_ids.add(c.chunk.source_id)
                chunk_tier_map[id(c)] = 2
            all_chunks.extend(t2_chunks)
            if t2_chunks:
                tiers_used.append(2)
        except Exception:
            _logger.debug("tiered_query_tier2_failed")

    # --- Tier 3a: Event listings (dedicated budget) ---
    # Event listings comprise ~69K chunks (79% of corpus).  Without a
    # dedicated query, they dominate the unfiltered catch-all and crowd
    # out reference docs, analysis, and other smaller source types.
    # Giving them their own tier with a capped budget (keep 10) ensures
    # relevant event data surfaces without monopolising results.
    t3a_filters = _merge_filters({"source_type": "event"})
    if t3a_filters is not None:
        try:
            t3a_raw = await vector_store.query(
                query_text=query_text, top_k=25, filters=t3a_filters,
            )
            t3a_chunks = [
                c for c in t3a_raw
                if c.chunk.source_id not in seen_ids
            ][:10]
            for c in t3a_chunks:
                seen_ids.add(c.chunk.source_id)
                chunk_tier_map[id(c)] = 3
            all_chunks.extend(t3a_chunks)
            if t3a_chunks:
                tiers_used.append(3)
        except Exception:
            _logger.debug("tiered_query_tier3a_event_failed")

    # --- Tier 3b: Catch-all (everything not in T1/T2/T3a) ---
    # Covers reference docs, analysis, and any future source types.
    # Uses $nin to exclude types already handled by dedicated tiers,
    # preventing duplicate retrieval of interview/book/event
    # chunks that would just be deduped away by seen_ids.
    t3b_base: dict[str, Any] = {
        "source_type": {"$nin": ["interview", "book", "event"]},
    }
    t3b_filters = _merge_filters(t3b_base)
    if t3b_filters is not None:
        try:
            t3b_raw = await vector_store.query(
                query_text=query_text, top_k=25, filters=t3b_filters,
            )
            t3b_chunks = [
                c for c in t3b_raw
                if c.chunk.source_id not in seen_ids
            ][:12]
            for c in t3b_chunks:
                seen_ids.add(c.chunk.source_id)
                chunk_tier_map[id(c)] = 3
            all_chunks.extend(t3b_chunks)
            # T3b shares tier number 3 with T3a — no separate tiers_used entry
        except Exception:
            _logger.debug("tiered_query_tier3b_catchall_failed")

    return all_chunks, chunk_tier_map, tiers_used


# ---------------------------------------------------------------------------
# Web search tier — live DuckDuckGo results filtered for music relevance
# ---------------------------------------------------------------------------
# Tier 4: augments the corpus with live web results when the web search
# provider is available.  Runs in parallel with ChromaDB queries (via
# asyncio.create_task) to minimize latency.  Has a 3-second hard timeout
# for graceful degradation when DDG is slow or rate-limited.

_WEB_SEARCH_TIMEOUT = 3.0  # seconds — hard ceiling for DDG call


async def _web_search_tier(
    web_search: Any,
    query: str,
    corpus_chunks: list[Any],
) -> list[WebSearchResult]:
    """Execute a music-filtered web search to augment corpus results.

    Builds an enriched query from the user's original query plus entity/genre
    tags extracted from the corpus hits, then filters results through the
    shared music-relevance check.

    Parameters
    ----------
    web_search:
        An IWebSearchProvider instance (typically DuckDuckGoSearchProvider).
    query:
        The user's original search query.
    corpus_chunks:
        Chunks from tiers 1-3, used to extract entity/genre context.

    Returns
    -------
    list[WebSearchResult]
        Music-relevant web results, or empty list on failure/timeout.
    """
    if web_search is None:
        return []

    from src.utils.music_relevance import is_music_relevant
    from urllib.parse import urlparse

    # Build enriched query: original + top entity/genre tags from corpus
    context_tags: list[str] = []
    for c in corpus_chunks[:5]:
        chunk = c.chunk if hasattr(c, "chunk") else c
        if hasattr(chunk, "entity_tags"):
            context_tags.extend(chunk.entity_tags[:2])
        if hasattr(chunk, "genre_tags"):
            context_tags.extend(chunk.genre_tags[:1])
    # Deduplicate and limit context expansion
    unique_tags = list(dict.fromkeys(context_tags))[:4]
    enriched_query = query
    if unique_tags:
        enriched_query = f"{query} {' '.join(unique_tags)} electronic music"
    else:
        enriched_query = f"{query} electronic music"

    try:
        raw_results = await asyncio.wait_for(
            web_search.search(enriched_query, num_results=5),
            timeout=_WEB_SEARCH_TIMEOUT,
        )
    except (asyncio.TimeoutError, Exception):
        _logger.debug("web_search_tier_failed_or_timeout", query_preview=query[:50])
        return []

    # Filter for music relevance and convert to schema objects
    web_results: list[WebSearchResult] = []
    for r in raw_results:
        if not is_music_relevant(r.url, r.title or "", r.snippet or ""):
            continue
        domain = urlparse(r.url).netloc.removeprefix("www.")
        web_results.append(WebSearchResult(
            title=r.title or "",
            url=r.url,
            snippet=r.snippet or "",
            source_domain=domain,
        ))

    return web_results


# ---------------------------------------------------------------------------
# Corpus search — LLM synthesis prompts and helper
# ---------------------------------------------------------------------------
# After vector retrieval ranks chunks, the top results are passed to the LLM
# to produce a cohesive natural-language answer with inline citation markers
# like [1], [2].  This transforms raw chunk excerpts into a readable response
# similar to other RAG-powered assistants (Perplexity, ChatGPT search).

_SYNTHESIS_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant specializing in electronic music, "
    "rave culture, and the history of DJs, venues, labels, and promoters.\n\n"
    "You will be given a user's search query and a set of numbered source passages "
    "retrieved from a curated knowledge base of books, articles, interviews, "
    "and event listings.\n\n"
    "Your job is to synthesize these passages into a clear, cohesive, and "
    "informative answer. Follow these rules:\n"
    "- Write a natural, flowing response (2-4 paragraphs) that directly answers "
    "the query.\n"
    "- Cite sources inline using numbered markers like [1], [2], etc. that "
    "correspond to the passage numbers provided.\n"
    "- Only cite passages that actually support a claim — do not cite every passage.\n"
    "- If the passages do not contain enough information to answer the query, "
    "say so honestly and provide what you can.\n"
    "- Stay strictly on topic: electronic music, rave culture, DJs, labels, "
    "venues, promoters, and related culture.\n"
    "- Do NOT invent facts not supported by the provided passages.\n"
    "- Write in a knowledgeable but accessible tone."
)

# Maximum number of top-ranked chunks to feed into the synthesis prompt.
# More chunks = richer context but higher token cost and latency.
_SYNTHESIS_MAX_CHUNKS = 8

# LRU cache for synthesis results — avoids repeated LLM calls for identical
# query + chunk combinations.  Keyed by SHA-256 of (query + chunk texts).
_synthesis_cache: dict[str, tuple[str, list[dict[str, Any]]]] = {}
_SYNTHESIS_CACHE_MAX = 64


async def _synthesize_answer(
    llm: Any,
    query: str,
    chunks: list[CorpusSearchChunk],
    web_results: list[WebSearchResult] | None = None,
    chunk_tiers: dict[int, str] | None = None,
) -> tuple[str | None, list[CorpusSearchCitation]]:
    """Synthesize a natural-language answer from tiered corpus chunks and web results.

    Passes the user's query and numbered source passages to the LLM, which
    returns a cohesive response with inline [N] citation markers.  Passage
    labels include tier provenance so the LLM can prioritize authoritative
    sources (RA Exchange > Books > Corpus > Web).

    Parameters
    ----------
    llm:
        The LLM provider instance.
    query:
        The user's search query.
    chunks:
        Ranked corpus search chunks from tiers 1-3.
    web_results:
        Optional music-relevant web search results from tier 4.
    chunk_tiers:
        Maps ``id(chunk)`` → tier label string (e.g. "RA Exchange", "Book").

    Returns (answer_text, citations_list).  Returns (None, []) if synthesis
    fails or LLM is unavailable — the frontend gracefully falls back to
    showing raw chunk cards.
    """
    if llm is None or (not chunks and not web_results):
        return None, []

    # Select top chunks for synthesis context
    top_chunks = chunks[:_SYNTHESIS_MAX_CHUNKS]

    # Build cache key from query + chunk texts + web presence
    web_sig = "".join(w.url[:40] for w in (web_results or [])[:3])
    cache_raw = query.lower().strip() + "||" + "||".join(c.text[:200] for c in top_chunks) + "||" + web_sig
    cache_key = hashlib.sha256(cache_raw.encode()).hexdigest()[:32]
    if cache_key in _synthesis_cache:
        cached_answer, cached_cits = _synthesis_cache[cache_key]
        return cached_answer, [CorpusSearchCitation(**c) for c in cached_cits]

    # Track which passage index maps to which source (chunk or web) for citations
    # passage_sources: list of (type, chunk_or_web, tier_label)
    passage_sources: list[tuple[str, Any, str]] = []

    # Build numbered passage list for the LLM — corpus chunks first
    passage_lines: list[str] = []
    for chunk in top_chunks:
        idx = len(passage_lines) + 1
        source_label = chunk.source_title or "Unknown source"
        if chunk.author:
            source_label += f" by {chunk.author}"
        if chunk.page_number:
            source_label += f", p. {chunk.page_number}"
        # Resolve tier label from chunk_tiers map
        tier_label = (chunk_tiers or {}).get(id(chunk), "Corpus")
        passage_lines.append(
            f"[{idx}] ({tier_label}: {source_label}, Tier {chunk.citation_tier})\n{chunk.text}"
        )
        passage_sources.append(("chunk", chunk, tier_label))

    # Append web results only when corpus coverage is thin — prevents
    # web snippets from diluting rich corpus passages in the LLM prompt.
    # Web results remain in the API response for frontend card display.
    _WEB_IN_SYNTHESIS_THRESHOLD = 4
    if len(top_chunks) < _WEB_IN_SYNTHESIS_THRESHOLD:
        for wr in (web_results or [])[:3]:
            idx = len(passage_lines) + 1
            passage_lines.append(
                f"[{idx}] (Web: {wr.source_domain})\n{wr.title}\n{wr.snippet}"
            )
            passage_sources.append(("web", wr, "Web"))

    user_prompt = (
        f"## Search Query\n{query}\n\n"
        f"## Retrieved Passages\n\n"
        + "\n\n---\n\n".join(passage_lines)
        + "\n\n## Task\n"
        "Synthesize a clear, informative answer to the search query using "
        "the passages above. Cite sources with [N] markers."
    )

    try:
        raw_answer = await llm.complete(
            system_prompt=_SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=1500,
        )
        answer_text = raw_answer.strip() if raw_answer else None
    except Exception:
        _logger.debug("corpus_synthesis_failed", query_preview=query[:50])
        return None, []

    if not answer_text:
        return None, []

    # Build citation objects for passages actually cited in the answer
    citations: list[CorpusSearchCitation] = []
    for i, (src_type, src_obj, tier_label) in enumerate(passage_sources, 1):
        if f"[{i}]" not in answer_text:
            continue

        if src_type == "chunk":
            citations.append(CorpusSearchCitation(
                index=i,
                source_title=src_obj.source_title,
                author=src_obj.author,
                citation_tier=src_obj.citation_tier,
                page_number=src_obj.page_number,
                excerpt=src_obj.text[:200] + ("..." if len(src_obj.text) > 200 else ""),
                source_tier=tier_label.lower().replace(" ", "_"),
            ))
        else:
            # Web result citation
            citations.append(CorpusSearchCitation(
                index=i,
                source_title=src_obj.title,
                citation_tier=5,
                excerpt=src_obj.snippet[:200] + ("..." if len(src_obj.snippet) > 200 else ""),
                url=src_obj.url,
                source_tier="web",
            ))

    # Cache the result
    if len(_synthesis_cache) >= _SYNTHESIS_CACHE_MAX:
        _synthesis_cache.pop(next(iter(_synthesis_cache)))
    _synthesis_cache[cache_key] = (
        answer_text,
        [c.model_dump() for c in citations],
    )

    return answer_text, citations


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
    llm: LLMProviderDep,
    web_search: WebSearchDep,
) -> CorpusSearchResponse:
    """Perform semantic search against the RAG vector-store corpus.

    Does not require a session — available at any time when RAG is enabled.
    """
    rag_enabled = getattr(request.app.state, "rag_enabled", False)
    if not rag_enabled or vector_store is None:
        raise HTTPException(status_code=503, detail="RAG corpus not available")

    # Build filters from request — pushed to ChromaDB where possible
    # for early pruning; others applied as post-filters after retrieval.
    filters: dict[str, Any] = {}
    if body.source_type:
        filters["source_type"] = {"$in": body.source_type}
    if body.entity_tag:
        filters["entity_tags"] = {"$contains": body.entity_tag}
    if body.geographic_tag:
        filters["geographic_tags"] = {"$contains": body.geographic_tag}
    # Genre filter — ChromaDB $contains handles one genre; multi-genre
    # precision is handled by a post-filter after retrieval.
    if body.genre_tags:
        filters["genre_tags"] = {"$contains": body.genre_tags[0]}
    # Citation tier quality floor — lower number = better quality,
    # so $lte returns chunks "at least this good" (e.g. $lte:3 → T1-T3).
    if body.min_citation_tier is not None:
        filters["citation_tier"] = {"$lte": body.min_citation_tier}

    # --- Pre-query processing: expand query for better retrieval ---
    query_text = body.query

    # Alias expansion: if query matches a known alias, append canonical + all names
    # Also capture parsed signals for the ParsedQueryFilters response field.
    _parsed_canonical: str | None = None
    _parsed_aliases: list[str] = []
    _parsed_genres: list[str] = []
    _parsed_time_period: str | None = None
    _parsed_geo_tags: list[str] = []

    try:
        from src.config.domain_knowledge import (
            detect_temporal_signal,
            expand_aliases,
            extract_query_genres,
            get_canonical_name,
            get_scene_geographies,
        )

        canonical = get_canonical_name(query_text.strip())
        if canonical:
            all_names = expand_aliases(canonical)
            query_text = f"{query_text} ({' '.join(all_names)})"
            _parsed_canonical = canonical
            _parsed_aliases = sorted(all_names)

        # Also expand entity_tag filter if it matches an alias
        if body.entity_tag:
            tag_canonical = get_canonical_name(body.entity_tag)
            if tag_canonical:
                # Rewrite entity filter to use canonical name
                filters["entity_tags"] = {"$contains": tag_canonical}

        # --- Smart filter auto-detection from query text ---
        # Detected signals are returned in the parsed_filters response
        # field so the frontend can show "auto" badges, and the domain-
        # aware boost section (below) already promotes matching results.
        # We intentionally do NOT inject auto-detected values into the
        # hard `filters` dict — only user-explicit filters should hard-
        # filter results.  Auto-detected signals are soft (boost only).

        _parsed_genres = sorted(extract_query_genres(body.query))
        _parsed_time_period = detect_temporal_signal(body.query)
        _parsed_geo_tags = sorted(get_scene_geographies(body.query))

    except ImportError:
        pass  # domain_knowledge not yet available

    # HyDE-lite: expand short/vague queries with LLM
    raw_query = query_text
    query_text = await _expand_query(llm, query_text)

    # --- Launch unified synthesis query in parallel with tiered query ---
    # The unified query ignores source_type partitioning so the LLM gets
    # the globally most relevant chunks for NL synthesis, while the tiered
    # query (below) still drives the card-display results.
    _SYNTHESIS_UNIFIED_TOP_K = 20
    synthesis_query_task: asyncio.Task | None = None
    if body.offset == 0:
        synthesis_query_task = asyncio.create_task(
            vector_store.query(
                query_text=query_text,
                top_k=_SYNTHESIS_UNIFIED_TOP_K,
                filters=filters if filters else None,
            )
        )

    # --- Tiered corpus retrieval ---
    tiered_chunks, chunk_tier_map, tiers_used = await _tiered_corpus_query(
        vector_store, query_text, filters if filters else None,
    )

    # Fallback: if tiered query returned nothing and expanded query differs,
    # retry with the raw query.
    if not tiered_chunks and query_text != raw_query:
        _logger.debug(
            "tiered_query_empty_fallback_to_raw",
            expanded=query_text[:80],
            raw=raw_query[:80],
        )
        tiered_chunks, chunk_tier_map, tiers_used = await _tiered_corpus_query(
            vector_store, raw_query, filters if filters else None,
        )

    # Recall safety net: if tiered approach returned some but too few
    # results, supplement with a single unified query to restore recall.
    # Skipped when tiered returned zero (corpus has nothing) or when
    # user has explicit source_type filter (tiered already respected it).
    _MIN_TIERED_RECALL = 15
    _user_has_source_filter = bool(filters and "source_type" in filters)
    if (
        0 < len(tiered_chunks) < _MIN_TIERED_RECALL
        and not _user_has_source_filter
    ):
        _logger.debug(
            "tiered_recall_low_supplementing",
            tiered_count=len(tiered_chunks),
            threshold=_MIN_TIERED_RECALL,
        )
        seen_chunk_ids = {c.chunk.source_id for c in tiered_chunks}
        supplement = await vector_store.query(
            query_text=query_text,
            top_k=body.top_k,
            filters=filters if filters else None,
        )
        for c in supplement:
            if c.chunk.source_id not in seen_chunk_ids:
                seen_chunk_ids.add(c.chunk.source_id)
                st = getattr(c.chunk, "source_type", "")
                title = getattr(c.chunk, "source_title", "")
                if st == "interview" and title.startswith("EX."):
                    chunk_tier_map[id(c)] = 1
                elif st == "book":
                    chunk_tier_map[id(c)] = 2
                else:
                    chunk_tier_map[id(c)] = 3
                tiered_chunks.append(c)

    # --- Web search (single launch, after corpus results are available) ---
    # Fires once with corpus context for enriched query building.
    # 3-second hard timeout ensures it won't block the response.
    web_task: asyncio.Task[list[WebSearchResult]] | None = None
    if body.offset == 0 and web_search is not None:
        web_task = asyncio.create_task(
            _web_search_tier(web_search, body.query, tiered_chunks)
        )

    chunks = tiered_chunks

    # Safety-net dedup by source_id — keep top N chunks per source.
    _MAX_PER_SOURCE = 5
    source_chunks: dict[str, list[CorpusSearchChunk]] = {}
    # Map from CorpusSearchChunk id → tier label for synthesis provenance.
    # Labels are resolved from source_type (authoritative) with the
    # query-tier number as fallback for unlabeled source types.
    chunk_tier_labels: dict[int, str] = {}
    _TIER_LABELS = {1: "RA Exchange", 2: "Book", 3: "Corpus"}
    _SOURCE_TYPE_LABELS: dict[str, str] = {
        "interview": "RA Exchange",
        "book": "Book",
        "event": "Event",
    }

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
            entity_types=c.chunk.entity_types,
            geographic_tags=c.chunk.geographic_tags,
            genre_tags=c.chunk.genre_tags,
            time_period=c.chunk.time_period,
        )
        source_chunks.setdefault(sid, []).append(candidate)
        # Resolve label from source_type first (reliable), then fall
        # back to query-tier number.  This avoids mislabeling when the
        # id()-based tier map loses track of object identity.
        st = getattr(c.chunk, "source_type", "")
        label = _SOURCE_TYPE_LABELS.get(st)
        if label is None:
            raw_tier = chunk_tier_map.get(id(c), 3)
            label = _TIER_LABELS.get(raw_tier, "Corpus")
        chunk_tier_labels[id(candidate)] = label

    deduped: list[CorpusSearchChunk] = []
    for entries in source_chunks.values():
        entries.sort(key=lambda r: r.similarity_score, reverse=True)
        deduped.extend(entries[:_MAX_PER_SOURCE])

    # Exclude "analysis" source-type chunks — internal pipeline outputs.
    deduped = [r for r in deduped if r.source_type != "analysis"]

    # --- Facet computation: count metadata values in the candidate pool ---
    facet_counts = _compute_facets(deduped)

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

    # --- Post-filters for fields that need Python-side logic ---
    _entity_filter = filters.get("entity_tags", {}).get("$contains") if filters else None
    if _entity_filter:
        _entity_lower = _entity_filter.lower()
        deduped = [
            r for r in deduped
            if any(_entity_lower in t.lower() for t in r.entity_tags)
        ]

    _geo_filter = filters.get("geographic_tags", {}).get("$contains") if filters else None
    if _geo_filter:
        _geo_lower = _geo_filter.lower()
        deduped = [
            r for r in deduped
            if any(_geo_lower in t.lower() for t in r.geographic_tags)
        ]

    _genre_filter = filters.get("genre_tags", {}).get("$contains") if filters else None
    if body.genre_tags and len(body.genre_tags) > 1:
        genre_set = {g.lower() for g in body.genre_tags}
        deduped = [
            r for r in deduped
            if any(gt.lower() in genre_set for gt in r.genre_tags)
        ]
    elif _genre_filter:
        _genre_lower = _genre_filter.lower()
        deduped = [
            r for r in deduped
            if any(
                _genre_lower in gt.lower() or gt.lower() in _genre_lower
                for gt in r.genre_tags
            )
        ]

    if body.time_period:
        try:
            from src.config.domain_knowledge import detect_temporal_signal, temporal_overlap
            normalized = detect_temporal_signal(body.time_period) or body.time_period
            deduped = [
                r for r in deduped
                if r.time_period and temporal_overlap(normalized, r.time_period)
            ]
        except ImportError:
            pass

    # --- Score boosting: domain-aware re-ranking ---
    try:
        from src.config.domain_knowledge import (
            detect_temporal_signal,
            extract_query_genres,
            get_adjacent_genres,
            get_scene_geographies,
            temporal_overlap,
        )
        _dk_available = True
    except ImportError:
        _dk_available = False

    boosted: dict[int, float] = {}
    query_temporal = detect_temporal_signal(body.query) if _dk_available else None
    query_geos = get_scene_geographies(body.query) if _dk_available else set()
    query_genres = extract_query_genres(body.query) if _dk_available else set()

    for r in deduped:
        boost = 0.0
        boost += max(0.0, (6 - r.citation_tier) * 0.024)
        if body.query.lower() in r.text.lower():
            boost += 0.03

        # Tier priority boost: T1 (RA Exchange) → +0.06, T2 (Book) → +0.04, T3 → +0.00
        # Gives higher-authority sources a ranking lift within the merged pool.
        tier_label = chunk_tier_labels.get(id(r), "Corpus")
        if tier_label == "RA Exchange":
            boost += 0.06
        elif tier_label == "Book":
            boost += 0.04

        if _dk_available:
            if query_genres and r.genre_tags:
                for qg in query_genres:
                    adjacent = get_adjacent_genres(qg)
                    for rtag in r.genre_tags:
                        if rtag.lower() == qg.lower():
                            boost += 0.05
                            break
                        if rtag.lower() in adjacent:
                            boost += 0.03
                            break

            if query_temporal:
                if r.time_period and temporal_overlap(query_temporal, r.time_period):
                    boost += 0.04

            if query_geos and r.geographic_tags:
                for geo in r.geographic_tags:
                    if geo in query_geos:
                        boost += 0.03
                        break

        boosted[id(r)] = min(1.0, r.similarity_score + boost)

    def _score(r: CorpusSearchChunk) -> float:
        return boosted.get(id(r), r.similarity_score)

    # --- Diversification: entity-type-aware caps ---
    if not _is_artist_query(body.query):
        _CAP_BY_TYPE = {"ARTIST": 4, "VENUE": 5, "LABEL": 5, "EVENT": 5, "COLLECTIVE": 4}
        _DEFAULT_CAP = 4
        entity_counts: dict[str, int] = {}
        diversified: list[CorpusSearchChunk] = []
        deduped.sort(key=_score, reverse=True)

        for r in deduped:
            if not r.entity_tags:
                diversified.append(r)
                continue

            has_types = r.entity_types and len(r.entity_types) == len(r.entity_tags)

            if has_types:
                capped = False
                for tag, etype in zip(r.entity_tags, r.entity_types):
                    cap = _CAP_BY_TYPE.get(etype, _DEFAULT_CAP)
                    if entity_counts.get(tag, 0) >= cap:
                        capped = True
                        break
                if not capped:
                    diversified.append(r)
                    for tag in r.entity_tags:
                        entity_counts[tag] = entity_counts.get(tag, 0) + 1
            else:
                capped = False
                for tag in r.entity_tags:
                    if entity_counts.get(tag, 0) >= _DEFAULT_CAP:
                        capped = True
                        break
                if not capped:
                    diversified.append(r)
                    for tag in r.entity_tags:
                        entity_counts[tag] = entity_counts.get(tag, 0) + 1

        deduped = diversified

    results = sorted(
        deduped,
        key=_score,
        reverse=True,
    )

    results = _semantic_dedup(results, threshold=0.80)

    if body.min_similarity is not None and body.min_similarity > 0:
        results = [r for r in results if _score(r) >= body.min_similarity]

    # --- Pagination ---
    total_available = len(results)
    page_start = min(body.offset, total_available)
    page_end = min(page_start + body.page_size, total_available)
    page_results = results[page_start:page_end]

    # --- Await web search results (launched in parallel earlier) ---
    web_results: list[WebSearchResult] = []
    if web_task is not None:
        try:
            web_results = await web_task
            if web_results:
                tiers_used.append(4)
        except (asyncio.CancelledError, Exception):
            _logger.debug("web_search_task_failed")

    # --- Build synthesis-specific chunk list (unified, not tiered) ---
    # The unified query returns the globally best chunks by pure semantic
    # relevance — no forced source_type diversity.  Event listings are only
    # included as fallback when the entity has thin non-event coverage.
    synthesis_chunks: list[CorpusSearchChunk] = []
    synthesis_tier_labels: dict[int, str] = {}

    if synthesis_query_task is not None:
        try:
            unified_raw = await synthesis_query_task
        except Exception:
            unified_raw = []

        # Convert raw query results to CorpusSearchChunk, split by source_type
        non_event: list[CorpusSearchChunk] = []
        event_chunks: list[CorpusSearchChunk] = []
        for c in unified_raw:
            chunk_obj = CorpusSearchChunk(
                text=c.chunk.text,
                source_title=c.chunk.source_title,
                source_type=c.chunk.source_type,
                author=c.chunk.author,
                citation_tier=c.chunk.citation_tier,
                page_number=c.chunk.page_number,
                similarity_score=round(c.similarity_score, 3),
                formatted_citation=c.formatted_citation,
                entity_tags=c.chunk.entity_tags,
                entity_types=c.chunk.entity_types,
                geographic_tags=c.chunk.geographic_tags,
                genre_tags=c.chunk.genre_tags,
                time_period=c.chunk.time_period,
            )
            st = getattr(c.chunk, "source_type", "")
            label = _SOURCE_TYPE_LABELS.get(st)
            if label is None:
                label = "Corpus"
            synthesis_tier_labels[id(chunk_obj)] = label

            if st == "event":
                event_chunks.append(chunk_obj)
            else:
                non_event.append(chunk_obj)

        # Primary context: top non-event chunks ranked by semantic relevance
        synthesis_chunks = non_event[:_SYNTHESIS_MAX_CHUNKS]

        # Event-listing fallback: if the queried entity/name has thin coverage
        # in books/articles/interviews, supplement with event listings that
        # actually mention the entity.
        _entity_name = (_parsed_canonical or body.query).lower()
        _mentions_in_non_event = sum(
            1 for ch in synthesis_chunks
            if _entity_name in ch.text.lower()
            or any(_entity_name in t.lower() for t in ch.entity_tags)
        )
        _EVENT_SUPPLEMENT_THRESHOLD = 2
        _MAX_EVENT_SUPPLEMENT = 3

        if _mentions_in_non_event < _EVENT_SUPPLEMENT_THRESHOLD and event_chunks:
            relevant_events = [
                ch for ch in event_chunks
                if _entity_name in ch.text.lower()
                or any(_entity_name in t.lower() for t in ch.entity_tags)
            ][:_MAX_EVENT_SUPPLEMENT]
            synthesis_chunks.extend(relevant_events)

    # --- LLM synthesis: generate NL answer from unified chunks + web results ---
    synthesized_answer: str | None = None
    answer_citations: list[CorpusSearchCitation] = []
    if body.offset == 0 and (synthesis_chunks or results or web_results):
        synthesized_answer, answer_citations = await _synthesize_answer(
            llm, body.query,
            synthesis_chunks if synthesis_chunks else results,
            web_results=web_results if web_results else None,
            chunk_tiers=synthesis_tier_labels if synthesis_chunks else chunk_tier_labels,
        )

    return CorpusSearchResponse(
        query=body.query,
        total_results=total_available,
        results=page_results,
        offset=body.offset,
        page_size=body.page_size,
        has_more=page_end < total_available,
        facets=facet_counts,
        parsed_filters=ParsedQueryFilters(
            genres=_parsed_genres,
            time_period=_parsed_time_period,
            geographic_tags=_parsed_geo_tags,
            artist_canonical=_parsed_canonical,
            artist_aliases=_parsed_aliases,
        ),
        synthesized_answer=synthesized_answer,
        answer_citations=answer_citations,
        web_results=web_results,
        search_tiers_used=sorted(tiers_used),
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
    mode: str = "full",
) -> RecommendationsResponse:
    """Generate artist recommendations based on the completed flier analysis.

    Supports two modes via the ``mode`` query parameter:

    - **quick** — Fast path (~3-5s): shared-flier SQLite lookup + simple
      LLM suggestions.  Returns ``is_partial=True`` so the frontend knows
      to fetch full results next.
    - **full** (default) — Full pipeline (~10-60s): Discogs label-mates,
      RAG shared-lineup, shared-flier, LLM fill + explanation.  Returns
      ``is_partial=False``.

    The frontend calls quick first for immediate display, then fetches
    full in the background to replace quick results with richer data.
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

    # Route to quick or full pipeline based on mode query param.
    is_quick = mode == "quick"
    timeout = _RECO_QUICK_TIMEOUT if is_quick else _RECO_TIMEOUT

    try:
        if is_quick:
            result = await asyncio.wait_for(
                recommendation_service.recommend_quick(
                    research_results=state.research_results,
                    entities=entities,
                    interconnection_map=state.interconnection_map,
                ),
                timeout=timeout,
            )
        else:
            result = await asyncio.wait_for(
                recommendation_service.recommend(
                    research_results=state.research_results,
                    entities=entities,
                    interconnection_map=state.interconnection_map,
                ),
                timeout=timeout,
            )
    except asyncio.TimeoutError:
        _logger.warning(
            "recommendation_timeout",
            session_id=session_id,
            mode=mode,
            timeout=timeout,
        )
        raise HTTPException(
            status_code=503,
            detail=f"Recommendations ({mode}) timed out after {timeout:.0f} seconds",
        )
    except Exception as exc:
        _logger.error("recommendation_failed", session_id=session_id, mode=mode, error=str(exc))
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
        is_partial=is_quick,
    )

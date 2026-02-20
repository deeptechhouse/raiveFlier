"""FastAPI API routes for the RaiveFlier pipeline.

Provides REST endpoints for flier upload, entity confirmation, progress
polling, result retrieval, health checks, and provider listing.  Service
dependencies are resolved from ``app.state`` via FastAPI's ``Depends``
using the ``Annotated`` pattern.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, UploadFile

from src.api.schemas import (
    AskQuestionRequest,
    AskQuestionResponse,
    ConfirmEntitiesRequest,
    ConfirmResponse,
    CorpusStatsResponse,
    ErrorResponse,
    FlierAnalysisResponse,
    FlierUploadResponse,
    HealthResponse,
    PipelineStatusResponse,
    ProvidersResponse,
    RelatedFact,
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


QAServiceDep = Annotated[Any, Depends(_get_qa_service)]


# ---------------------------------------------------------------------------
# Background task helpers
# ---------------------------------------------------------------------------


async def _run_phases_2_through_5(
    pipeline: FlierAnalysisPipeline,
    state: PipelineState,
    session_states: dict[str, PipelineState],
) -> None:
    """Execute research-through-output phases and persist the final state."""
    try:
        result = await pipeline.run_phases_2_through_5(state)
        session_states[result.session_id] = result
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

    # --- Build FlierImage ---
    session_id = str(uuid.uuid4())
    image_hash = hashlib.sha256(image_data).hexdigest()

    flier_image = FlierImage(
        id=session_id,
        filename=file.filename or "unknown",
        content_type=content_type,
        file_size=len(image_data),
        image_hash=image_hash,
    )
    # Attach raw bytes to the private attr on a frozen model
    flier_image.__pydantic_private__["_image_data"] = image_data

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
    background_tasks.add_task(
        _run_phases_2_through_5,
        pipeline,
        confirmed_state,
        session_states,
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
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

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
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

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
        providers = request.app.state.provider_registry

    return HealthResponse(
        status="healthy",
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

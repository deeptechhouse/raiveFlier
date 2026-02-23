"""Pipeline state management models for the raiveFlier pipeline.

Defines Pydantic v2 models for pipeline phases, errors, and session state.
All models use frozen config to enforce immutability — state transitions
produce new PipelineState instances via model_copy(update={...}).

Architecture note:
    PipelineState is the "single source of truth" for an analysis session.
    The orchestrator (src/pipeline/orchestrator.py) holds one PipelineState
    per session and advances it through phases by creating new copies with
    updated fields. This functional/immutable approach means any intermediate
    state can be serialized to SQLite for persistence and crash recovery
    (see src/providers/session/).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

# Import the models that PipelineState references — these represent the
# output of each pipeline phase.
from src.models.analysis import InterconnectionMap
from src.models.flier import ExtractedEntities, FlierImage, OCRResult
from src.models.research import ResearchResult


# ---------------------------------------------------------------------------
# PipelinePhase — the state machine that drives the analysis workflow.
# ---------------------------------------------------------------------------
class PipelinePhase(str, Enum):  # noqa: UP042 — StrEnum requires Python 3.11+
    """Phases of the flier analysis pipeline.

    The pipeline progresses through these phases in order:
        UPLOAD → OCR → ENTITY_EXTRACTION → USER_CONFIRMATION → RESEARCH →
        INTERCONNECTION → OUTPUT

    The orchestrator (src/pipeline/orchestrator.py) checks current_phase
    to decide what to do next. The WebSocket progress tracker sends phase
    updates to the frontend so the user sees real-time progress.
    """

    UPLOAD = "UPLOAD"                       # Image received and validated
    OCR = "OCR"                             # Text extraction in progress
    ENTITY_EXTRACTION = "ENTITY_EXTRACTION" # LLM parsing entities from OCR text
    USER_CONFIRMATION = "USER_CONFIRMATION" # Waiting for user to review/edit entities
    RESEARCH = "RESEARCH"                   # Querying music DBs and web for each entity
    INTERCONNECTION = "INTERCONNECTION"     # Mapping relationships between entities
    OUTPUT = "OUTPUT"                       # Formatting and returning final results


# ---------------------------------------------------------------------------
# PipelineError — records errors without crashing the pipeline.
# ---------------------------------------------------------------------------
class PipelineError(BaseModel):
    """An error that occurred during pipeline processing.

    Errors are appended to PipelineState.errors rather than raising exceptions.
    This allows the pipeline to continue processing other entities even if one
    fails. The 'recoverable' flag hints whether the pipeline can proceed.
    """

    model_config = ConfigDict(frozen=True)

    # Which phase the error occurred in — helps debugging and display.
    phase: PipelinePhase
    # Human-readable error description.
    message: str
    # When the error happened (UTC).
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)  # noqa: UP017
    )
    # If True, the pipeline can continue despite this error (e.g. one artist
    # lookup failed but others succeeded). If False, the pipeline should halt.
    recoverable: bool = True


# ---------------------------------------------------------------------------
# PipelineState — the complete snapshot of an analysis session.
# ---------------------------------------------------------------------------
class PipelineState(BaseModel):
    """The current state of a flier analysis pipeline session.

    Immutable — use model_copy(update={...}) to produce new states.
    Example:
        new_state = state.model_copy(update={
            "current_phase": PipelinePhase.RESEARCH,
            "confirmed_entities": confirmed,
        })

    This model is serialized to SQLite (src/providers/session/) for crash
    recovery and persisted on the /data disk in production (Render).
    """

    model_config = ConfigDict(frozen=True)

    # Unique session identifier — used in API URLs and WebSocket channels.
    session_id: str
    # The uploaded flier image metadata (set at UPLOAD phase).
    flier: FlierImage
    # Current phase — determines what the orchestrator does next.
    current_phase: PipelinePhase = PipelinePhase.UPLOAD
    # Set after OCR phase completes — contains the raw extracted text.
    ocr_result: OCRResult | None = None
    # Set after ENTITY_EXTRACTION — the LLM's initial entity parse.
    extracted_entities: ExtractedEntities | None = None
    # Set after USER_CONFIRMATION — may be identical to extracted_entities
    # if the user accepted without changes, or modified if they edited.
    confirmed_entities: ExtractedEntities | None = None
    # Accumulated during RESEARCH phase — one ResearchResult per entity.
    research_results: list[ResearchResult] = Field(default_factory=list)
    # Set after INTERCONNECTION phase — the relationship graph.
    interconnection_map: InterconnectionMap | None = None
    # Session timing — used for performance monitoring and display.
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)  # noqa: UP017
    )
    completed_at: datetime | None = None
    # Non-fatal errors accumulated during processing.
    errors: list[PipelineError] = Field(default_factory=list)
    # Overall progress percentage (0.0–100.0) — sent via WebSocket to the
    # frontend progress bar.
    progress_percent: float = Field(default=0.0, ge=0.0, le=100.0)

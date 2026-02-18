"""Pipeline state management models for the raiveFlier pipeline.

Defines Pydantic v2 models for pipeline phases, errors, and session state.
All models use frozen config to enforce immutability — state transitions
produce new PipelineState instances via model_copy(update={...}).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from src.models.analysis import InterconnectionMap
from src.models.flier import ExtractedEntities, OCRResult
from src.models.research import ResearchResult


class PipelinePhase(str, Enum):  # noqa: UP042 — StrEnum requires Python 3.11+
    """Phases of the flier analysis pipeline."""

    UPLOAD = "UPLOAD"
    OCR = "OCR"
    ENTITY_EXTRACTION = "ENTITY_EXTRACTION"
    USER_CONFIRMATION = "USER_CONFIRMATION"
    RESEARCH = "RESEARCH"
    INTERCONNECTION = "INTERCONNECTION"
    OUTPUT = "OUTPUT"


class PipelineError(BaseModel):
    """An error that occurred during pipeline processing."""

    model_config = ConfigDict(frozen=True)

    phase: PipelinePhase
    message: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)  # noqa: UP017
    )
    recoverable: bool = True


class PipelineState(BaseModel):
    """The current state of a flier analysis pipeline session.

    Immutable — use model_copy(update={...}) to produce new states.
    """

    model_config = ConfigDict(frozen=True)

    session_id: str
    current_phase: PipelinePhase = PipelinePhase.UPLOAD
    ocr_result: OCRResult | None = None
    extracted_entities: ExtractedEntities | None = None
    confirmed_entities: ExtractedEntities | None = None
    research_results: list[ResearchResult] = Field(default_factory=list)
    interconnection_map: InterconnectionMap | None = None
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)  # noqa: UP017
    )
    completed_at: datetime | None = None
    errors: list[PipelineError] = Field(default_factory=list)
    progress_percent: float = Field(default=0.0, ge=0.0, le=100.0)

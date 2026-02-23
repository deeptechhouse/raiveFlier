"""Central orchestrator for the five-phase flier analysis pipeline.

Coordinates OCR, entity extraction, user confirmation, research,
interconnection analysis, and citation ranking into a coherent pipeline.
Each phase updates a frozen :class:`PipelineState` via ``model_copy``
and broadcasts progress through the injected :class:`ProgressTracker`.

ARCHITECTURE NOTE (for junior developers):
    This orchestrator follows the "Pipeline" pattern — it coordinates
    multiple services in a fixed sequence, passing data from one phase
    to the next through the immutable PipelineState object.

    The pipeline is split into TWO entry points:
        - run_phase_1()         → OCR + Entity Extraction (automatic)
        - run_phases_2_through_5() → Research + Interconnection + Output

    This split exists because of the HUMAN-IN-THE-LOOP step: after Phase 1
    extracts entities, the user gets a chance to review and edit them before
    the expensive research phase begins. The API layer (routes.py) manages
    this pause/resume flow.

    Each phase follows the same pattern:
        1. Update state with new phase + progress percentage
        2. Broadcast progress via ProgressTracker (WebSocket to frontend)
        3. Call the appropriate service
        4. Update state with results
        5. Handle errors (recoverable vs. fatal)

    The frozen state pattern: instead of mutating state, each phase creates
    a NEW PipelineState via model_copy(update={...}). This means any
    intermediate state can be serialized for crash recovery without worrying
    about partially-mutated objects.
"""

from __future__ import annotations

import contextlib
from datetime import date, datetime, timezone
# TYPE_CHECKING is a special constant that is False at runtime but True
# when type checkers (mypy) analyze the code. Imports inside this block
# are only used for type annotations and don't cause circular imports.
from typing import TYPE_CHECKING

import structlog

from src.models.flier import ExtractedEntities
# Aliased import: PipelineError (the Pydantic model for error records stored
# in state) vs PipelineError (the exception class used for flow control).
# The model is aliased as PipelineErrorModel to avoid name collision.
from src.models.pipeline import (
    PipelineError as PipelineErrorModel,
)
from src.models.pipeline import (
    PipelinePhase,
    PipelineState,
)
from src.pipeline.progress_tracker import ProgressTracker
from src.services.citation_service import CitationService
from src.services.entity_extractor import EntityExtractor
from src.services.interconnection_service import InterconnectionService
from src.services.ocr_service import OCRService
from src.services.research_service import ResearchService
# PipelineError (the exception) is raised to signal fatal pipeline failures
# that should stop the pipeline. NOT the same as PipelineErrorModel above.
from src.utils.errors import PipelineError
from src.utils.logging import get_logger

if TYPE_CHECKING:
    # IngestionService is only imported for type hints to avoid a circular
    # dependency — the ingestion service depends on models that the pipeline
    # also depends on.
    from src.services.ingestion.ingestion_service import IngestionService


class FlierAnalysisPipeline:
    """Central orchestrator for the five-phase flier analysis pipeline.

    Phases:
        1. OCR — extract raw text from the flier image.
        2. Entity Extraction — parse artists, venue, date, promoter.
        3. User Confirmation — pause for human review and editing.
        4. Research — deep research on every confirmed entity.
        5. Interconnection — trace links between entities.
        6. Output — verify citations and finalise results.

    All service dependencies are injected at construction time.
    The ``interconnection_service`` may be ``None`` (Phase F not yet
    implemented) and is handled gracefully.
    """

    def __init__(
        self,
        ocr_service: OCRService,
        entity_extractor: EntityExtractor,
        research_service: ResearchService,
        interconnection_service: InterconnectionService | None,
        citation_service: CitationService,
        progress_tracker: ProgressTracker,
        ingestion_service: IngestionService | None = None,
    ) -> None:
        # All services are injected — the orchestrator never creates them.
        # This makes testing easy: inject mocks for any service.
        self._ocr_service = ocr_service               # Phase 1a: text extraction
        self._entity_extractor = entity_extractor       # Phase 1b: entity parsing
        self._research_service = research_service       # Phase 2: music DB + web research
        self._interconnection_service = interconnection_service  # Phase 3: relationship mapping
        self._citation_service = citation_service       # Phase 4: citation verification
        self._progress_tracker = progress_tracker       # WebSocket broadcast
        self._ingestion_service = ingestion_service     # Phase 5: RAG feedback (optional)
        self._logger: structlog.BoundLogger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Phase 1: OCR + Entity Extraction
    # ------------------------------------------------------------------

    async def run_phase_1(self, state: PipelineState) -> PipelineState:
        """Run Phase 1: OCR + Entity Extraction.

        Returns a :class:`PipelineState` paused at the
        ``USER_CONFIRMATION`` phase so the user can review extracted
        entities before research begins.

        Parameters
        ----------
        state:
            Initial pipeline state (typically at UPLOAD phase).

        Returns
        -------
        PipelineState
            Updated state paused at USER_CONFIRMATION with OCR and
            entity extraction results populated.

        Raises
        ------
        PipelineError
            If the entire phase fails irrecoverably (both OCR and
            extraction must succeed for the pipeline to proceed).
        """
        session_id = state.session_id
        errors: list[PipelineErrorModel] = list(state.errors)

        # --- OCR ---
        # model_copy(update={...}) creates a NEW frozen PipelineState with
        # the specified fields changed. The original state is untouched.
        # This is the functional/immutable state management pattern used
        # throughout the pipeline — no state mutation, only new copies.
        state = state.model_copy(
            update={
                "current_phase": PipelinePhase.OCR,
                "progress_percent": 10.0,
            }
        )
        await self._progress_tracker.update(
            session_id, PipelinePhase.OCR, 10.0, "Running OCR extraction..."
        )

        try:
            self._logger.info("phase1_ocr_start", session_id=session_id)
            ocr_result = await self._ocr_service.extract_text(state.flier)
            state = state.model_copy(update={"ocr_result": ocr_result})
            self._logger.info(
                "phase1_ocr_complete",
                session_id=session_id,
                confidence=round(ocr_result.confidence, 4),
                provider=ocr_result.provider_used,
            )
        except Exception as exc:
            self._logger.error(
                "phase1_ocr_failed",
                session_id=session_id,
                error=str(exc),
            )
            errors.append(
                PipelineErrorModel(
                    phase=PipelinePhase.OCR,
                    message=f"OCR extraction failed: {exc}",
                    recoverable=False,
                )
            )
            state = state.model_copy(update={"errors": errors})
            raise PipelineError(message=f"OCR phase failed: {exc}") from exc

        # --- Entity Extraction ---
        state = state.model_copy(
            update={
                "current_phase": PipelinePhase.ENTITY_EXTRACTION,
                "progress_percent": 30.0,
            }
        )
        await self._progress_tracker.update(
            session_id,
            PipelinePhase.ENTITY_EXTRACTION,
            30.0,
            "Extracting entities from OCR text...",
        )

        try:
            self._logger.info("phase1_entity_extraction_start", session_id=session_id)
            extracted_entities = await self._entity_extractor.extract(state.ocr_result)
            state = state.model_copy(update={"extracted_entities": extracted_entities})
            self._logger.info(
                "phase1_entity_extraction_complete",
                session_id=session_id,
                artists=len(extracted_entities.artists),
                has_venue=extracted_entities.venue is not None,
                has_date=extracted_entities.date is not None,
            )
        except Exception as exc:
            self._logger.error(
                "phase1_entity_extraction_failed",
                session_id=session_id,
                error=str(exc),
            )
            errors.append(
                PipelineErrorModel(
                    phase=PipelinePhase.ENTITY_EXTRACTION,
                    message=f"Entity extraction failed: {exc}",
                    recoverable=False,
                )
            )
            state = state.model_copy(update={"errors": errors})
            raise PipelineError(message=f"Entity extraction phase failed: {exc}") from exc

        # --- Pause at User Confirmation ---
        state = state.model_copy(
            update={
                "current_phase": PipelinePhase.USER_CONFIRMATION,
                "progress_percent": 40.0,
                "errors": errors,
            }
        )
        await self._progress_tracker.update(
            session_id,
            PipelinePhase.USER_CONFIRMATION,
            40.0,
            "Awaiting user confirmation of extracted entities...",
        )

        self._logger.info(
            "phase1_complete_awaiting_confirmation",
            session_id=session_id,
        )

        return state

    # ------------------------------------------------------------------
    # Phases 2–5: Research → Interconnection → Output
    # ------------------------------------------------------------------

    async def run_phases_2_through_5(self, state: PipelineState) -> PipelineState:
        """Run phases 2–5 after user confirmation.

        Executes Research, Interconnection, and Output phases
        sequentially.  Non-fatal errors (e.g. a single artist's research
        failing) are recorded in ``state.errors`` with
        ``recoverable=True`` and the pipeline continues.

        Parameters
        ----------
        state:
            Pipeline state with ``confirmed_entities`` populated by the
            user confirmation step.

        Returns
        -------
        PipelineState
            Final state with research results, interconnection map,
            ranked citations, and completion timestamp.

        Raises
        ------
        PipelineError
            If ``confirmed_entities`` is missing or if an entire phase
            fails irrecoverably.
        """
        session_id = state.session_id
        errors: list[PipelineErrorModel] = list(state.errors)

        # --- Validate confirmed entities ---
        if state.confirmed_entities is None:
            raise PipelineError(message="Cannot run phases 2-5 without confirmed entities")

        confirmed: ExtractedEntities = state.confirmed_entities

        # --- Parse event date ---
        event_date = await self._parse_event_date(confirmed)

        # --- Phase 2: Research ---
        state = state.model_copy(
            update={
                "current_phase": PipelinePhase.RESEARCH,
                "progress_percent": 45.0,
            }
        )
        await self._progress_tracker.update(
            session_id,
            PipelinePhase.RESEARCH,
            45.0,
            "Researching entities...",
        )

        try:
            self._logger.info("phase2_research_start", session_id=session_id)

            research_results = await self._research_service.research_all(confirmed, event_date)

            state = state.model_copy(
                update={
                    "research_results": research_results,
                    "progress_percent": 75.0,
                }
            )

            await self._progress_tracker.update(
                session_id,
                PipelinePhase.RESEARCH,
                75.0,
                f"Research complete — {len(research_results)} entities researched.",
            )

            # Record warnings from individual research results as
            # recoverable errors
            for result in research_results:
                if result.confidence == 0.0 and result.warnings:
                    errors.append(
                        PipelineErrorModel(
                            phase=PipelinePhase.RESEARCH,
                            message=f"Research for {result.entity_name}: "
                            + "; ".join(result.warnings),
                            recoverable=True,
                        )
                    )

            self._logger.info(
                "phase2_research_complete",
                session_id=session_id,
                results=len(research_results),
                failed=sum(1 for r in research_results if r.confidence == 0.0),
            )

        except Exception as exc:
            self._logger.error(
                "phase2_research_failed",
                session_id=session_id,
                error=str(exc),
            )
            errors.append(
                PipelineErrorModel(
                    phase=PipelinePhase.RESEARCH,
                    message=f"Research phase failed: {exc}",
                    recoverable=False,
                )
            )
            state = state.model_copy(update={"errors": errors})
            raise PipelineError(message=f"Research phase failed: {exc}") from exc

        # --- Phase 3: Interconnection ---
        # This phase uses an LLM to discover relationships BETWEEN entities —
        # e.g. "Artist X released on a label run by Promoter Y" or "all three
        # artists have played at this venue." Errors here are recoverable
        # because the core research results are already complete.
        state = state.model_copy(
            update={
                "current_phase": PipelinePhase.INTERCONNECTION,
                "progress_percent": 80.0,
            }
        )
        await self._progress_tracker.update(
            session_id,
            PipelinePhase.INTERCONNECTION,
            80.0,
            "Analyzing interconnections between entities...",
        )

        if self._interconnection_service is not None:
            try:
                self._logger.info("phase3_interconnection_start", session_id=session_id)

                interconnection_map = await self._interconnection_service.analyze(
                    state.research_results, confirmed
                )

                state = state.model_copy(update={"interconnection_map": interconnection_map})

                self._logger.info(
                    "phase3_interconnection_complete",
                    session_id=session_id,
                    edges=len(interconnection_map.edges),
                    patterns=len(interconnection_map.patterns),
                )

            except Exception as exc:
                self._logger.warning(
                    "phase3_interconnection_failed",
                    session_id=session_id,
                    error=str(exc),
                )
                errors.append(
                    PipelineErrorModel(
                        phase=PipelinePhase.INTERCONNECTION,
                        message=f"Interconnection analysis failed: {exc}",
                        recoverable=True,
                    )
                )
        else:
            self._logger.info(
                "phase3_interconnection_skipped",
                session_id=session_id,
                reason="interconnection_service not available",
            )

        # --- Phase 4: Output (Citation Verification) ---
        state = state.model_copy(
            update={
                "current_phase": PipelinePhase.OUTPUT,
                "progress_percent": 95.0,
            }
        )
        await self._progress_tracker.update(
            session_id,
            PipelinePhase.OUTPUT,
            95.0,
            "Verifying and ranking citations...",
        )

        try:
            self._logger.info("phase4_output_start", session_id=session_id)

            # Collect all citations from the interconnection map
            if state.interconnection_map is not None:
                all_citations = list(state.interconnection_map.citations)

                # Rank citations by tier and date
                ranked = self._citation_service.rank_citations(all_citations)

                # Rebuild the interconnection map with ranked citations
                state = state.model_copy(
                    update={
                        "interconnection_map": state.interconnection_map.model_copy(
                            update={"citations": ranked}
                        )
                    }
                )

                self._logger.info(
                    "phase4_citations_ranked",
                    session_id=session_id,
                    total_citations=len(ranked),
                )

        except Exception as exc:
            self._logger.warning(
                "phase4_citation_ranking_failed",
                session_id=session_id,
                error=str(exc),
            )
            errors.append(
                PipelineErrorModel(
                    phase=PipelinePhase.OUTPUT,
                    message=f"Citation ranking failed: {exc}",
                    recoverable=True,
                )
            )

        # --- Phase 5: RAG Feedback Loop (if enabled) ---
        # If RAG is enabled, the completed analysis itself is ingested into
        # the vector store as a new source document. This means future flier
        # analyses can reference past analyses — a self-improving knowledge base.
        if self._ingestion_service is not None:
            try:
                ingestion_result = await self._ingestion_service.ingest_analysis(state)
                self._logger.info(
                    "rag_feedback_ingestion_complete",
                    session_id=session_id,
                    chunks=ingestion_result.chunks_created,
                )
            except Exception as exc:
                self._logger.warning(
                    "rag_feedback_ingestion_failed",
                    session_id=session_id,
                    error=str(exc),
                )
                errors.append(
                    PipelineErrorModel(
                        phase=PipelinePhase.OUTPUT,
                        message=f"RAG feedback ingestion failed: {exc}",
                        recoverable=True,
                    )
                )

        # --- Finalise ---
        state = state.model_copy(
            update={
                "completed_at": datetime.now(tz=timezone.utc),  # noqa: UP017
                "progress_percent": 100.0,
                "errors": errors,
            }
        )
        await self._progress_tracker.update(
            session_id,
            PipelinePhase.OUTPUT,
            100.0,
            "Pipeline complete.",
        )

        self._logger.info(
            "pipeline_complete",
            session_id=session_id,
            total_errors=len(errors),
            recoverable_errors=sum(1 for e in errors if e.recoverable),
        )

        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _parse_event_date(self, entities: ExtractedEntities) -> date | None:
        """Extract a Python date from confirmed entities' date field.

        Delegates to :class:`ResearchService` internal date parsing
        via its ``_parse_event_date`` method if available, otherwise
        returns ``None`` (the research service will parse it itself).

        Parameters
        ----------
        entities:
            The user-confirmed extracted entities.

        Returns
        -------
        date or None
            Parsed event date, or ``None`` if parsing is deferred.
        """
        if entities.date is None:
            return None

        # Attempt a direct ISO parse for clean date strings
        raw_text = entities.date.text.strip()
        with contextlib.suppress(ValueError):
            return date.fromisoformat(raw_text)

        # Let the research service handle complex date parsing
        return None

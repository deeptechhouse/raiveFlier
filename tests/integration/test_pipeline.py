"""Integration tests for the FlierAnalysisPipeline orchestrator."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.analysis import (
    Citation,
    EntityNode,
    InterconnectionMap,
    PatternInsight,
    RelationshipEdge,
)
from src.models.entities import EntityType
from src.models.flier import ExtractedEntities, ExtractedEntity, FlierImage, OCRResult
from src.models.pipeline import PipelinePhase, PipelineState
from src.models.research import ResearchResult
from src.pipeline.confirmation_gate import ConfirmationGate
from src.pipeline.orchestrator import FlierAnalysisPipeline
from src.pipeline.progress_tracker import ProgressTracker
from src.services.citation_service import CitationService
from src.services.entity_extractor import EntityExtractor
from src.services.interconnection_service import InterconnectionService
from src.services.ocr_service import OCRService
from src.services.research_service import ResearchService
from src.utils.errors import PipelineError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_flier() -> FlierImage:
    """Create a minimal FlierImage fixture."""
    return FlierImage(
        id="test-pipeline-001",
        filename="test_flier.jpg",
        content_type="image/jpeg",
        file_size=5000,
        image_hash="sha256_test_hash",
    )


def _make_ocr_result() -> OCRResult:
    return OCRResult(
        raw_text="CARL COX\nJEFF MILLS\nTRESOR BERLIN\nMARCH 15 1997",
        confidence=0.88,
        provider_used="tesseract",
        processing_time=0.8,
    )


def _make_extracted_entities(ocr: OCRResult) -> ExtractedEntities:
    return ExtractedEntities(
        artists=[
            ExtractedEntity(text="Carl Cox", entity_type=EntityType.ARTIST, confidence=0.95),
            ExtractedEntity(text="Jeff Mills", entity_type=EntityType.ARTIST, confidence=0.92),
        ],
        venue=ExtractedEntity(
            text="Tresor, Berlin", entity_type=EntityType.VENUE, confidence=0.88
        ),
        date=ExtractedEntity(
            text="March 15, 1997", entity_type=EntityType.DATE, confidence=0.85
        ),
        promoter=None,
        raw_ocr=ocr,
    )


def _make_research_results() -> list[ResearchResult]:
    return [
        ResearchResult(
            entity_type=EntityType.ARTIST,
            entity_name="Carl Cox",
            sources_consulted=["discogs", "ra.co"],
            confidence=0.9,
        ),
        ResearchResult(
            entity_type=EntityType.ARTIST,
            entity_name="Jeff Mills",
            sources_consulted=["discogs", "wikipedia"],
            confidence=0.88,
        ),
        ResearchResult(
            entity_type=EntityType.VENUE,
            entity_name="Tresor",
            sources_consulted=["wikipedia", "ra.co"],
            confidence=0.85,
        ),
    ]


def _make_interconnection_map() -> InterconnectionMap:
    return InterconnectionMap(
        nodes=[
            EntityNode(entity_type=EntityType.ARTIST, name="Carl Cox", properties={}),
            EntityNode(entity_type=EntityType.ARTIST, name="Jeff Mills", properties={}),
            EntityNode(entity_type=EntityType.VENUE, name="Tresor", properties={}),
        ],
        edges=[
            RelationshipEdge(
                source="Carl Cox",
                target="Tresor",
                relationship_type="performed_at",
                details="Residency in the 1990s",
                citations=[],
                confidence=0.8,
            ),
        ],
        patterns=[
            PatternInsight(
                pattern_type="scene_connection",
                description="Both artists associated with Berlin techno scene",
                involved_entities=["Carl Cox", "Jeff Mills"],
                citations=[],
            ),
        ],
        narrative="Carl Cox and Jeff Mills were pivotal figures at Tresor Berlin.",
        citations=[
            Citation(
                text="Carl Cox at Tresor",
                source_type="website",
                source_name="RA",
                source_url="https://ra.co/features/carl-cox",
                tier=2,
            ),
        ],
    )


def _make_pipeline(
    ocr_result: OCRResult | None = None,
    extracted_entities: ExtractedEntities | None = None,
    research_results: list[ResearchResult] | None = None,
    interconnection_map: InterconnectionMap | None = None,
    research_raises: Exception | None = None,
) -> FlierAnalysisPipeline:
    """Create a FlierAnalysisPipeline with fully mocked services."""
    ocr = ocr_result or _make_ocr_result()
    entities = extracted_entities or _make_extracted_entities(ocr)
    research = research_results or _make_research_results()
    imap = interconnection_map or _make_interconnection_map()

    # OCR Service mock
    ocr_service = MagicMock(spec=OCRService)
    ocr_service.extract_text = AsyncMock(return_value=ocr)

    # Entity Extractor mock
    entity_extractor = MagicMock(spec=EntityExtractor)
    entity_extractor.extract = AsyncMock(return_value=entities)

    # Research Service mock
    research_service = MagicMock(spec=ResearchService)
    if research_raises is not None:
        research_service.research_all = AsyncMock(side_effect=research_raises)
    else:
        research_service.research_all = AsyncMock(return_value=research)

    # Interconnection Service mock
    interconnection_service = MagicMock(spec=InterconnectionService)
    interconnection_service.analyze = AsyncMock(return_value=imap)

    # Citation Service (real — lightweight, no external deps)
    citation_service = CitationService()

    # Progress Tracker (real)
    progress_tracker = ProgressTracker()

    return FlierAnalysisPipeline(
        ocr_service=ocr_service,
        entity_extractor=entity_extractor,
        research_service=research_service,
        interconnection_service=interconnection_service,
        citation_service=citation_service,
        progress_tracker=progress_tracker,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPipelinePhase1:
    """Tests for Phase 1: OCR + Entity Extraction + User Confirmation pause."""

    @pytest.mark.asyncio()
    async def test_phase_1_flow(self) -> None:
        """Upload image -> OCR -> entity extraction -> state paused at USER_CONFIRMATION."""
        pipeline = _make_pipeline()
        flier = _make_flier()
        state = PipelineState(session_id="test-001", flier=flier)

        result_state = await pipeline.run_phase_1(state)

        assert result_state.current_phase == PipelinePhase.USER_CONFIRMATION
        assert result_state.progress_percent == pytest.approx(40.0)
        assert result_state.ocr_result is not None
        assert result_state.ocr_result.confidence > 0
        assert result_state.extracted_entities is not None
        assert len(result_state.extracted_entities.artists) == 2
        assert result_state.extracted_entities.venue is not None

    @pytest.mark.asyncio()
    async def test_phase_1_ocr_failure_raises(self) -> None:
        """If OCR fails, PipelineError is raised with error recorded."""
        ocr_service = MagicMock(spec=OCRService)
        ocr_service.extract_text = AsyncMock(
            side_effect=RuntimeError("All OCR providers failed")
        )
        entity_extractor = MagicMock(spec=EntityExtractor)
        research_service = MagicMock(spec=ResearchService)
        interconnection_service = MagicMock(spec=InterconnectionService)
        citation_service = CitationService()
        progress_tracker = ProgressTracker()

        pipeline = FlierAnalysisPipeline(
            ocr_service=ocr_service,
            entity_extractor=entity_extractor,
            research_service=research_service,
            interconnection_service=interconnection_service,
            citation_service=citation_service,
            progress_tracker=progress_tracker,
        )

        flier = _make_flier()
        state = PipelineState(session_id="test-fail", flier=flier)

        with pytest.raises(PipelineError):
            await pipeline.run_phase_1(state)


class TestConfirmationGate:
    """Tests for the ConfirmationGate submit/retrieve/confirm workflow."""

    @pytest.mark.asyncio()
    async def test_confirmation_gate(self) -> None:
        """Submit state, retrieve pending, confirm with edited entities."""
        gate = ConfirmationGate()
        pipeline = _make_pipeline()
        flier = _make_flier()
        state = PipelineState(session_id="test-gate", flier=flier)

        # Phase 1
        state = await pipeline.run_phase_1(state)

        # Submit for review
        await gate.submit_for_review(state)

        # Retrieve pending
        pending = await gate.get_pending("test-gate")
        assert pending is not None
        assert pending.current_phase == PipelinePhase.USER_CONFIRMATION

        # Confirm with edited entities (user removes one artist, adds promoter)
        ocr = state.ocr_result or _make_ocr_result()
        edited = ExtractedEntities(
            artists=[
                ExtractedEntity(text="Carl Cox", entity_type=EntityType.ARTIST, confidence=0.95),
            ],
            venue=ExtractedEntity(
                text="Tresor, Berlin", entity_type=EntityType.VENUE, confidence=0.88
            ),
            date=ExtractedEntity(
                text="March 15, 1997", entity_type=EntityType.DATE, confidence=0.85
            ),
            promoter=ExtractedEntity(
                text="Tresor Records", entity_type=EntityType.PROMOTER, confidence=0.7
            ),
            raw_ocr=ocr,
        )

        confirmed_state = await gate.confirm("test-gate", edited)

        assert confirmed_state.confirmed_entities is not None
        assert len(confirmed_state.confirmed_entities.artists) == 1
        assert confirmed_state.confirmed_entities.promoter is not None
        assert confirmed_state.confirmed_entities.promoter.text == "Tresor Records"

        # Pending should be cleared
        assert await gate.get_pending("test-gate") is None

    @pytest.mark.asyncio()
    async def test_confirm_nonexistent_raises(self) -> None:
        """Confirming a non-existent session raises KeyError."""
        gate = ConfirmationGate()
        ocr = _make_ocr_result()
        entities = _make_extracted_entities(ocr)

        with pytest.raises(KeyError):
            await gate.confirm("nonexistent", entities)

    @pytest.mark.asyncio()
    async def test_cancel_removes_pending(self) -> None:
        """Cancelling removes the session from pending."""
        gate = ConfirmationGate()
        pipeline = _make_pipeline()
        flier = _make_flier()
        state = PipelineState(session_id="test-cancel", flier=flier)

        state = await pipeline.run_phase_1(state)
        await gate.submit_for_review(state)

        cancelled = await gate.cancel("test-cancel")
        assert cancelled is True

        assert await gate.get_pending("test-cancel") is None


class TestFullPipeline:
    """Tests for the complete Phase 1 -> confirm -> Phase 2-5 flow."""

    @pytest.mark.asyncio()
    async def test_full_pipeline(self) -> None:
        """Phase 1 -> confirm -> Phase 2-5 -> results contain research + interconnections."""
        pipeline = _make_pipeline()
        gate = ConfirmationGate()
        flier = _make_flier()
        state = PipelineState(session_id="full-001", flier=flier)

        # Phase 1
        state = await pipeline.run_phase_1(state)
        assert state.current_phase == PipelinePhase.USER_CONFIRMATION

        # Submit and confirm
        await gate.submit_for_review(state)
        ocr = state.ocr_result or _make_ocr_result()
        confirmed_entities = state.extracted_entities or _make_extracted_entities(ocr)
        state = await gate.confirm("full-001", confirmed_entities)

        # Phases 2-5
        final_state = await pipeline.run_phases_2_through_5(state)

        assert final_state.completed_at is not None
        assert final_state.progress_percent == pytest.approx(100.0)
        assert final_state.research_results is not None
        assert len(final_state.research_results) > 0
        assert final_state.interconnection_map is not None
        assert len(final_state.interconnection_map.edges) > 0
        assert len(final_state.interconnection_map.patterns) > 0

    @pytest.mark.asyncio()
    async def test_pipeline_error_recovery(self) -> None:
        """Research phase failure raises PipelineError with error info."""
        pipeline = _make_pipeline(
            research_raises=RuntimeError("External API down"),
        )
        gate = ConfirmationGate()
        flier = _make_flier()
        state = PipelineState(session_id="error-001", flier=flier)

        # Phase 1 succeeds
        state = await pipeline.run_phase_1(state)
        await gate.submit_for_review(state)
        ocr = state.ocr_result or _make_ocr_result()
        confirmed_entities = state.extracted_entities or _make_extracted_entities(ocr)
        state = await gate.confirm("error-001", confirmed_entities)

        # Phases 2-5 — research failure raises PipelineError
        with pytest.raises(PipelineError):
            await pipeline.run_phases_2_through_5(state)

    @pytest.mark.asyncio()
    async def test_phases_2_5_without_confirmed_entities(self) -> None:
        """Calling run_phases_2_through_5 without confirmed entities raises PipelineError."""
        pipeline = _make_pipeline()
        flier = _make_flier()
        state = PipelineState(session_id="no-confirm", flier=flier)

        with pytest.raises(PipelineError, match="confirmed entities"):
            await pipeline.run_phases_2_through_5(state)

    @pytest.mark.asyncio()
    async def test_interconnection_failure_recoverable(self) -> None:
        """Interconnection service failure is recorded as recoverable error."""
        ocr = _make_ocr_result()
        entities = _make_extracted_entities(ocr)
        research = _make_research_results()

        # Build pipeline with failing interconnection service
        ocr_service = MagicMock(spec=OCRService)
        ocr_service.extract_text = AsyncMock(return_value=ocr)
        entity_extractor = MagicMock(spec=EntityExtractor)
        entity_extractor.extract = AsyncMock(return_value=entities)
        research_service = MagicMock(spec=ResearchService)
        research_service.research_all = AsyncMock(return_value=research)
        interconnection_service = MagicMock(spec=InterconnectionService)
        interconnection_service.analyze = AsyncMock(
            side_effect=RuntimeError("LLM unavailable")
        )
        citation_service = CitationService()
        progress_tracker = ProgressTracker()

        pipeline = FlierAnalysisPipeline(
            ocr_service=ocr_service,
            entity_extractor=entity_extractor,
            research_service=research_service,
            interconnection_service=interconnection_service,
            citation_service=citation_service,
            progress_tracker=progress_tracker,
        )

        flier = _make_flier()
        state = PipelineState(session_id="intercon-fail", flier=flier)
        state = await pipeline.run_phase_1(state)

        gate = ConfirmationGate()
        await gate.submit_for_review(state)
        confirmed_entities = state.extracted_entities or entities
        state = await gate.confirm("intercon-fail", confirmed_entities)

        final_state = await pipeline.run_phases_2_through_5(state)

        # Pipeline should complete despite interconnection failure
        assert final_state.completed_at is not None
        assert final_state.progress_percent == pytest.approx(100.0)

        # Error should be recorded as recoverable
        intercon_errors = [
            e for e in final_state.errors
            if e.phase == PipelinePhase.INTERCONNECTION
        ]
        assert len(intercon_errors) == 1
        assert intercon_errors[0].recoverable is True

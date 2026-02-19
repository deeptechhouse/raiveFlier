"""Integration tests for FastAPI API endpoints using TestClient."""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from src.api.routes import router as api_router
from src.models.analysis import InterconnectionMap
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ocr_result() -> OCRResult:
    return OCRResult(
        raw_text="CARL COX\nTRESOR BERLIN\nMARCH 15 1997",
        confidence=0.88,
        provider_used="tesseract",
        processing_time=0.8,
    )


def _make_extracted_entities(ocr: OCRResult) -> ExtractedEntities:
    return ExtractedEntities(
        artists=[
            ExtractedEntity(text="Carl Cox", entity_type=EntityType.ARTIST, confidence=0.95),
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


def _make_flier_image(session_id: str = "api-test-001") -> FlierImage:
    return FlierImage(
        id=session_id,
        filename="test.jpg",
        content_type="image/jpeg",
        file_size=5000,
        image_hash="sha256_test",
    )


def _create_test_image_bytes(width: int = 100, height: int = 100) -> bytes:
    """Create a minimal JPEG image in memory."""
    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _create_test_app() -> tuple[FastAPI, ConfirmationGate, ProgressTracker, dict]:
    """Create a FastAPI app with mocked pipeline dependencies for testing."""
    app = FastAPI()
    app.include_router(api_router)

    ocr = _make_ocr_result()
    entities = _make_extracted_entities(ocr)

    # Build a mock pipeline that performs Phase 1 deterministically
    ocr_service = MagicMock(spec=OCRService)
    ocr_service.extract_text = AsyncMock(return_value=ocr)

    entity_extractor = MagicMock(spec=EntityExtractor)
    entity_extractor.extract = AsyncMock(return_value=entities)

    research_service = MagicMock(spec=ResearchService)
    research_service.research_all = AsyncMock(return_value=[])

    interconnection_service = MagicMock(spec=InterconnectionService)
    interconnection_service.analyze = AsyncMock(
        return_value=InterconnectionMap(
            nodes=[], edges=[], patterns=[], citations=[]
        )
    )

    citation_service = CitationService()
    progress_tracker = ProgressTracker()
    confirmation_gate = ConfirmationGate()
    session_states: dict[str, PipelineState] = {}

    pipeline = FlierAnalysisPipeline(
        ocr_service=ocr_service,
        entity_extractor=entity_extractor,
        research_service=research_service,
        interconnection_service=interconnection_service,
        citation_service=citation_service,
        progress_tracker=progress_tracker,
    )

    app.state.pipeline = pipeline
    app.state.confirmation_gate = confirmation_gate
    app.state.progress_tracker = progress_tracker
    app.state.session_states = session_states
    app.state.provider_registry = {
        "llm": True,
        "ocr": True,
        "music_db": True,
        "search": True,
    }

    return app, confirmation_gate, progress_tracker, session_states


@pytest.fixture
def test_app():
    """Create a test app and return (TestClient, gate, tracker, session_states)."""
    app, gate, tracker, states = _create_test_app()
    client = TestClient(app)
    return client, gate, tracker, states


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUploadEndpoint:
    """Tests for POST /api/v1/fliers/upload."""

    def test_upload_endpoint(self, test_app) -> None:
        """POST a valid JPEG file -> 200 with session_id and extracted entities."""
        client, gate, tracker, states = test_app
        image_data = _create_test_image_bytes()

        response = client.post(
            "/api/v1/fliers/upload",
            files={"file": ("rave_flier.jpg", io.BytesIO(image_data), "image/jpeg")},
        )

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["session_id"]  # non-empty
        assert data["extracted_entities"] is not None
        assert data["ocr_confidence"] > 0
        assert data["provider_used"] == "tesseract"

    def test_upload_invalid_file(self, test_app) -> None:
        """POST a non-image file -> 415 error."""
        client, gate, tracker, states = test_app

        response = client.post(
            "/api/v1/fliers/upload",
            files={"file": ("document.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        )

        assert response.status_code == 415
        data = response.json()
        assert "detail" in data
        assert "Unsupported file type" in data["detail"]

    def test_upload_too_large(self, test_app) -> None:
        """POST a file >10MB -> 413 error."""
        client, gate, tracker, states = test_app

        # Create oversized data (> 10MB)
        oversized_data = b"\xff\xd8\xff\xe0" + b"\x00" * (11 * 1024 * 1024)

        response = client.post(
            "/api/v1/fliers/upload",
            files={"file": ("huge.jpg", io.BytesIO(oversized_data), "image/jpeg")},
        )

        assert response.status_code == 413
        data = response.json()
        assert "detail" in data
        assert "too large" in data["detail"].lower()


class TestConfirmEndpoint:
    """Tests for POST /api/v1/fliers/{session_id}/confirm."""

    def test_confirm_endpoint(self, test_app) -> None:
        """POST confirm with valid entities -> 200 with research_started status."""
        client, gate, tracker, states = test_app

        # First upload
        image_data = _create_test_image_bytes()
        upload_resp = client.post(
            "/api/v1/fliers/upload",
            files={"file": ("flier.jpg", io.BytesIO(image_data), "image/jpeg")},
        )
        session_id = upload_resp.json()["session_id"]

        # Confirm entities
        confirm_body = {
            "artists": [{"name": "Carl Cox", "entity_type": "ARTIST"}],
            "venue": {"name": "Tresor, Berlin", "entity_type": "VENUE"},
            "date": {"name": "March 15, 1997", "entity_type": "DATE"},
            "promoter": None,
            "genre_tags": ["techno"],
            "ticket_price": "10 DM",
        }

        response = client.post(
            f"/api/v1/fliers/{session_id}/confirm",
            json=confirm_body,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert data["status"] == "research_started"
        assert "confirmed" in data["message"].lower() or "research" in data["message"].lower()

    def test_confirm_nonexistent_session(self, test_app) -> None:
        """POST confirm for unknown session -> 404."""
        client, gate, tracker, states = test_app

        confirm_body = {
            "artists": [{"name": "Carl Cox", "entity_type": "ARTIST"}],
        }

        response = client.post(
            "/api/v1/fliers/nonexistent-session/confirm",
            json=confirm_body,
        )

        assert response.status_code == 404


class TestStatusEndpoint:
    """Tests for GET /api/v1/fliers/{session_id}/status."""

    def test_status_endpoint(self, test_app) -> None:
        """GET status after upload -> returns current phase and progress."""
        client, gate, tracker, states = test_app

        # Upload first
        image_data = _create_test_image_bytes()
        upload_resp = client.post(
            "/api/v1/fliers/upload",
            files={"file": ("flier.jpg", io.BytesIO(image_data), "image/jpeg")},
        )
        session_id = upload_resp.json()["session_id"]

        response = client.get(f"/api/v1/fliers/{session_id}/status")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert "phase" in data
        assert "progress" in data

    def test_status_unknown_session(self, test_app) -> None:
        """GET status for unknown session -> returns default status."""
        client, gate, tracker, states = test_app

        response = client.get("/api/v1/fliers/unknown-id/status")

        # ProgressTracker returns default status for unknown sessions
        assert response.status_code == 200


class TestResultsEndpoint:
    """Tests for GET /api/v1/fliers/{session_id}/results."""

    def test_results_endpoint(self, test_app) -> None:
        """GET results after upload -> returns current analysis state."""
        client, gate, tracker, states = test_app

        # Upload
        image_data = _create_test_image_bytes()
        upload_resp = client.post(
            "/api/v1/fliers/upload",
            files={"file": ("flier.jpg", io.BytesIO(image_data), "image/jpeg")},
        )
        session_id = upload_resp.json()["session_id"]

        response = client.get(f"/api/v1/fliers/{session_id}/results")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert data["status"] in [
            "USER_CONFIRMATION",
            "completed",
            "OCR",
            "ENTITY_EXTRACTION",
            "RESEARCH",
        ]

    def test_results_nonexistent_session(self, test_app) -> None:
        """GET results for unknown session -> 404."""
        client, gate, tracker, states = test_app

        response = client.get("/api/v1/fliers/nonexistent-id/results")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_results_completed_session(self, test_app) -> None:
        """GET results for completed session -> full data with completed_at."""
        client, gate, tracker, states = test_app

        # Manually inject a completed state
        flier = _make_flier_image("completed-session")
        ocr = _make_ocr_result()
        entities = _make_extracted_entities(ocr)

        completed_state = PipelineState(
            session_id="completed-session",
            flier=flier,
            current_phase=PipelinePhase.OUTPUT,
            ocr_result=ocr,
            extracted_entities=entities,
            confirmed_entities=entities,
            research_results=[
                ResearchResult(
                    entity_type=EntityType.ARTIST,
                    entity_name="Carl Cox",
                    sources_consulted=["discogs"],
                    confidence=0.9,
                ),
            ],
            completed_at=datetime.now(tz=timezone.utc),
            progress_percent=100.0,
        )
        states["completed-session"] = completed_state

        response = client.get("/api/v1/fliers/completed-session/results")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["completed_at"] is not None
        assert data["research_results"] is not None
        assert len(data["research_results"]) == 1


class TestHealthEndpoint:
    """Tests for GET /api/v1/health."""

    def test_health_endpoint(self, test_app) -> None:
        """GET health -> 200 with version and provider status."""
        client, gate, tracker, states = test_app

        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert "providers" in data
        assert data["providers"]["llm"] is True


class TestEndToEndFlow:
    """End-to-end test through upload -> confirm -> results."""

    def test_upload_then_confirm_flow(self, test_app) -> None:
        """Upload image, then confirm entities — verifies full request flow."""
        client, gate, tracker, states = test_app

        # 1. Upload
        image_data = _create_test_image_bytes()
        upload_resp = client.post(
            "/api/v1/fliers/upload",
            files={"file": ("flier.jpg", io.BytesIO(image_data), "image/jpeg")},
        )
        assert upload_resp.status_code == 200
        session_id = upload_resp.json()["session_id"]

        # 2. Check status (should be at USER_CONFIRMATION)
        status_resp = client.get(f"/api/v1/fliers/{session_id}/status")
        assert status_resp.status_code == 200

        # 3. Confirm entities
        confirm_body = {
            "artists": [{"name": "Carl Cox", "entity_type": "ARTIST"}],
            "venue": {"name": "Tresor, Berlin", "entity_type": "VENUE"},
            "date": {"name": "March 15, 1997", "entity_type": "DATE"},
        }
        confirm_resp = client.post(
            f"/api/v1/fliers/{session_id}/confirm",
            json=confirm_body,
        )
        assert confirm_resp.status_code == 200
        assert confirm_resp.json()["status"] == "research_started"

        # 4. Check results (may still be processing in background)
        results_resp = client.get(f"/api/v1/fliers/{session_id}/results")
        assert results_resp.status_code == 200

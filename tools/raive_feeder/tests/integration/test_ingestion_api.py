"""Integration tests for raiveFeeder ingestion API endpoints.

Tests the full request → route → service flow using FastAPI's TestClient.
Services and providers are mocked to avoid external dependencies.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tools.raive_feeder.config.settings import FeederSettings
from tools.raive_feeder.main import create_app


@pytest.fixture
def mock_settings():
    """Settings with no real API keys to prevent provider initialization."""
    return FeederSettings(
        openai_api_key="",
        anthropic_api_key="",
        chromadb_persist_dir=tempfile.mkdtemp(),
    )


@pytest.fixture
def app(mock_settings):
    """Create the FastAPI app with mocked services."""
    application = create_app(mock_settings)

    # Mock the ingestion service on app.state.
    mock_ingestion = MagicMock()
    mock_result = MagicMock()
    mock_result.source_id = "test-source-123"
    mock_result.source_title = "Test Document"
    mock_result.chunks_created = 42
    mock_result.total_tokens = 8000
    mock_result.ingestion_time = 1.5
    mock_ingestion.ingest_book = AsyncMock(return_value=mock_result)
    mock_ingestion.ingest_pdf = AsyncMock(return_value=mock_result)
    mock_ingestion.ingest_epub = AsyncMock(return_value=mock_result)
    mock_ingestion.ingest_article = AsyncMock(return_value=mock_result)

    application.state.ingestion_service = mock_ingestion
    application.state.ingestion_status = "Mock: test"
    application.state.vector_store = None
    application.state.llm_provider = None
    application.state.ocr_providers = []
    application.state.embedding_provider = None

    # Mock batch processor.
    from tools.raive_feeder.services.batch_processor import BatchProcessor
    application.state.batch_processor = BatchProcessor(
        ingestion_service=mock_ingestion,
        max_concurrency=1,
    )

    return application


@pytest.fixture
def client(app):
    """TestClient that bypasses the lifespan (deps already mocked)."""
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    """Tests for GET /api/v1/health."""

    def test_health_returns_ok(self, client):
        """Health endpoint should return 200 with provider status."""
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["ingestion_available"] is True


class TestDocumentIngestion:
    """Tests for POST /api/v1/ingest/document."""

    def test_ingest_txt_file(self, client):
        """Uploading a TXT file should return ingestion results."""
        file_content = b"This is a test document about rave culture."
        resp = client.post(
            "/api/v1/ingest/document",
            files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")},
            data={
                "title": "Test Document",
                "author": "Test Author",
                "year": "2024",
                "source_type": "book",
                "citation_tier": "2",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["chunks_created"] == 42
        assert data["source_id"] == "test-source-123"

    def test_ingest_pdf_file(self, client):
        """Uploading a PDF file should dispatch to ingest_pdf."""
        resp = client.post(
            "/api/v1/ingest/document",
            files={"file": ("test.pdf", io.BytesIO(b"fake pdf"), "application/pdf")},
            data={"title": "PDF Test"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"

    def test_ingest_unsupported_format(self, client):
        """Uploading an unsupported format should return a failed status."""
        resp = client.post(
            "/api/v1/ingest/document",
            files={"file": ("test.xyz", io.BytesIO(b"data"), "application/octet-stream")},
            data={"title": "Bad Format"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert "Unsupported" in data["error"]


class TestProvidersEndpoint:
    """Tests for GET /api/v1/providers."""

    def test_list_providers(self, client):
        """Should return a list (possibly empty) of providers."""
        resp = client.get("/api/v1/providers")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestJobsEndpoint:
    """Tests for GET /api/v1/jobs."""

    def test_list_jobs_empty(self, client):
        """Jobs list should start empty."""
        resp = client.get("/api/v1/jobs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_nonexistent_job(self, client):
        """Getting a non-existent job should return 404."""
        resp = client.get("/api/v1/jobs/nonexistent-id")
        assert resp.status_code == 404

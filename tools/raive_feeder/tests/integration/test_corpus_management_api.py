"""Integration tests for raiveFeeder corpus management API endpoints.

Tests corpus stats, search, source listing, and delete operations
through the FastAPI TestClient with a mocked vector store.
"""

from __future__ import annotations

import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.models.rag import CorpusStats
from tools.raive_feeder.config.settings import FeederSettings
from tools.raive_feeder.main import create_app


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store with test data."""
    vs = MagicMock()

    mock_stats = CorpusStats(
        total_chunks=100,
        total_sources=5,
        sources_by_type={"book": 3, "article": 2},
        entity_tag_count=50,
        geographic_tag_count=20,
        genre_tag_count=15,
        genre_tags=["techno", "house", "ambient"],
        time_periods=["1990s", "2000s"],
    )
    vs.get_stats = AsyncMock(return_value=mock_stats)
    vs.is_available.return_value = True
    vs.get_provider_name.return_value = "mock_chromadb"
    vs.delete_by_source = AsyncMock(return_value=10)
    vs.delete_by_source_type = AsyncMock(return_value=25)
    vs.query = AsyncMock(return_value=[])
    vs.update_chunk_metadata = AsyncMock(return_value=True)

    # list_all_metadata() — public API used by CorpusManager instead of
    # reaching into _collection internals.
    _all_chunks = [
        ("c1", {"source_id": "s1", "source_title": "Energy Flash", "source_type": "book", "citation_tier": 1}, "Text 1"),
        ("c2", {"source_id": "s1", "source_title": "Energy Flash", "source_type": "book", "citation_tier": 1}, "Text 2"),
        ("c3", {"source_id": "s2", "source_title": "RA Article", "source_type": "article", "citation_tier": 3}, "Text 3"),
    ]

    async def _list_all_metadata(include_documents=False, where=None, page_size=5000):
        results = _all_chunks
        if where and "source_id" in where:
            target_sid = where["source_id"]
            results = [(cid, m, d) for cid, m, d in results if m.get("source_id") == target_sid]
        if not include_documents:
            results = [(cid, m, None) for cid, m, _d in results]
        return results

    vs.list_all_metadata = _list_all_metadata

    return vs


@pytest.fixture
def app(mock_vector_store):
    settings = FeederSettings(chromadb_persist_dir=tempfile.mkdtemp())
    application = create_app(settings)

    application.state.vector_store = mock_vector_store
    application.state.ingestion_service = None
    application.state.ingestion_status = "Mock"
    application.state.llm_provider = None
    application.state.ocr_providers = []
    application.state.embedding_provider = None

    from tools.raive_feeder.services.batch_processor import BatchProcessor
    application.state.batch_processor = BatchProcessor(ingestion_service=None)

    return application


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestCorpusStats:
    """Tests for GET /api/v1/corpus/stats."""

    def test_returns_stats(self, client):
        """Should return corpus statistics."""
        resp = client.get("/api/v1/corpus/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_chunks"] == 100
        assert data["total_sources"] == 5
        assert data["genre_tags"] == ["techno", "house", "ambient"]


class TestCorpusSources:
    """Tests for GET /api/v1/corpus/sources."""

    def test_list_sources(self, client):
        """Should return grouped source summaries."""
        resp = client.get("/api/v1/corpus/sources")
        assert resp.status_code == 200
        sources = resp.json()
        assert len(sources) == 2

        titles = {s["source_title"] for s in sources}
        assert "Energy Flash" in titles
        assert "RA Article" in titles


class TestDeleteSource:
    """Tests for DELETE /api/v1/corpus/sources/{source_id}."""

    def test_delete_source(self, client, mock_vector_store):
        """Deleting a source should call vector store and return count."""
        resp = client.delete("/api/v1/corpus/sources/s1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["source_id"] == "s1"
        assert data["deleted_chunks"] == 10
        mock_vector_store.delete_by_source.assert_called_once_with("s1")


class TestDeleteSourceType:
    """Tests for DELETE /api/v1/corpus/type/{source_type}."""

    def test_delete_type(self, client, mock_vector_store):
        """Deleting by type should call vector store and return count."""
        resp = client.delete("/api/v1/corpus/type/article")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted_chunks"] == 25


class TestCorpusSearch:
    """Tests for POST /api/v1/corpus/search."""

    def test_search_returns_results(self, client):
        """Semantic search should return (possibly empty) results."""
        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "Berlin techno", "top_k": 10},
        )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestUpdateChunkMetadata:
    """Tests for PATCH /api/v1/corpus/chunks/{chunk_id}."""

    def test_update_metadata(self, client, mock_vector_store):
        """Updating chunk metadata should call vector store."""
        resp = client.patch(
            "/api/v1/corpus/chunks/c1",
            json={
                "entity_tags": ["Aphex Twin", "Warp Records"],
                "genre_tags": ["ambient", "IDM"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["updated"] is True
        mock_vector_store.update_chunk_metadata.assert_called_once()

"""Integration tests for the corpus search API endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import router as api_router
from src.models.rag import DocumentChunk, RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_retrieved_chunk(
    text: str = "A passage about Detroit techno.",
    source_title: str = "Energy Flash",
    source_type: str = "book",
    author: str = "Simon Reynolds",
    citation_tier: int = 1,
    similarity: float = 0.88,
) -> RetrievedChunk:
    chunk = DocumentChunk(
        chunk_id="chunk-001",
        text=text,
        token_count=50,
        source_id="src-001",
        source_title=source_title,
        source_type=source_type,
        author=author,
        citation_tier=citation_tier,
        entity_tags=["Juan Atkins", "Derrick May"],
        geographic_tags=["Detroit"],
        genre_tags=["techno"],
    )
    return RetrievedChunk(
        chunk=chunk,
        similarity_score=similarity,
        formatted_citation=f"{source_title}, {author} [Tier {citation_tier}]",
    )


def _create_test_app(
    *,
    rag_enabled: bool = True,
    vector_store: object | None = None,
) -> FastAPI:
    """Build a minimal FastAPI app with mocked state for testing."""
    app = FastAPI()
    app.include_router(api_router)

    # Attach required state
    app.state.rag_enabled = rag_enabled
    app.state.vector_store = vector_store
    app.state.session_states = {}
    app.state.pipeline = MagicMock()
    app.state.confirmation_gate = MagicMock()
    app.state.progress_tracker = MagicMock()

    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCorpusSearchEndpoint:
    """Test POST /api/v1/corpus/search."""

    def test_search_returns_results(self) -> None:
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[
            _make_retrieved_chunk(),
            _make_retrieved_chunk(
                text="Another passage about house music.",
                source_title="Last Night a DJ Saved My Life",
                source_type="book",
                author="Bill Brewster",
                citation_tier=2,
                similarity=0.75,
            ),
        ])

        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "Detroit techno origins"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "Detroit techno origins"
        assert data["total_results"] == 2
        assert len(data["results"]) == 2

        first = data["results"][0]
        assert first["source_title"] == "Energy Flash"
        assert first["source_type"] == "book"
        assert first["author"] == "Simon Reynolds"
        assert first["citation_tier"] == 1
        assert first["similarity_score"] == 0.88
        assert "Juan Atkins" in first["entity_tags"]
        assert "Detroit" in first["geographic_tags"]

    def test_search_rag_disabled_returns_503(self) -> None:
        app = _create_test_app(rag_enabled=False, vector_store=None)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "techno"},
        )

        assert resp.status_code == 503

    def test_search_no_vector_store_returns_503(self) -> None:
        app = _create_test_app(rag_enabled=True, vector_store=None)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "techno"},
        )

        assert resp.status_code == 503

    def test_search_passes_source_type_filter(self) -> None:
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[])

        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "acid house", "source_type": ["book"]},
        )

        assert resp.status_code == 200
        call_kwargs = mock_store.query.call_args
        assert call_kwargs.kwargs.get("filters") == {"source_type": {"$in": ["book"]}}

    def test_search_passes_entity_tag_filter(self) -> None:
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[])

        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "career", "entity_tag": "Carl Cox"},
        )

        assert resp.status_code == 200
        call_kwargs = mock_store.query.call_args
        filters = call_kwargs.kwargs.get("filters")
        assert filters["entity_tags"] == {"$contains": "Carl Cox"}

    def test_search_passes_geographic_tag_filter(self) -> None:
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[])

        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "clubs", "geographic_tag": "Berlin"},
        )

        assert resp.status_code == 200
        call_kwargs = mock_store.query.call_args
        filters = call_kwargs.kwargs.get("filters")
        assert filters["geographic_tags"] == {"$contains": "Berlin"}

    def test_search_combined_filters(self) -> None:
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[])

        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={
                "query": "warehouse parties",
                "source_type": ["article", "interview"],
                "entity_tag": "Jeff Mills",
                "geographic_tag": "Detroit",
            },
        )

        assert resp.status_code == 200
        call_kwargs = mock_store.query.call_args
        filters = call_kwargs.kwargs.get("filters")
        assert filters["source_type"] == {"$in": ["article", "interview"]}
        assert filters["entity_tags"] == {"$contains": "Jeff Mills"}
        assert filters["geographic_tags"] == {"$contains": "Detroit"}

    def test_search_custom_top_k(self) -> None:
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[])

        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "techno", "top_k": 5},
        )

        assert resp.status_code == 200
        call_kwargs = mock_store.query.call_args
        assert call_kwargs.kwargs.get("top_k") == 5

    def test_search_empty_query_returns_422(self) -> None:
        mock_store = AsyncMock()
        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": ""},
        )

        assert resp.status_code == 422

    def test_search_query_too_long_returns_422(self) -> None:
        mock_store = AsyncMock()
        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "x" * 501},
        )

        assert resp.status_code == 422

    def test_search_no_filters_passes_none(self) -> None:
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[])

        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "techno"},
        )

        assert resp.status_code == 200
        call_kwargs = mock_store.query.call_args
        assert call_kwargs.kwargs.get("filters") is None

    def test_search_empty_results(self) -> None:
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[])

        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "extremely obscure topic with no matches"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_results"] == 0
        assert data["results"] == []

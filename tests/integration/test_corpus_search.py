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
    source_id: str = "src-001",
    chunk_id: str = "chunk-001",
) -> RetrievedChunk:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        token_count=50,
        source_id=source_id,
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
                source_id="src-002",
                chunk_id="chunk-002",
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

    # -----------------------------------------------------------------------
    # Single-artist filtering (non-artist queries)
    # -----------------------------------------------------------------------

    def test_non_artist_query_filters_single_entity_chunks(self) -> None:
        """Chunks with exactly 1 entity tag are dropped for non-artist queries."""
        single_entity = DocumentChunk(
            chunk_id="chunk-single",
            text="Carl Cox biography and career overview.",
            token_count=40,
            source_id="src-bio",
            source_title="DJ Mag Profile",
            source_type="article",
            author="Staff",
            citation_tier=4,
            entity_tags=["Carl Cox"],         # single entity → filtered
            geographic_tags=["London"],
            genre_tags=["techno"],
        )
        multi_entity = DocumentChunk(
            chunk_id="chunk-multi",
            text="The Detroit-Berlin axis shaped techno globally.",
            token_count=45,
            source_id="src-history",
            source_title="Energy Flash",
            source_type="book",
            author="Simon Reynolds",
            citation_tier=1,
            entity_tags=["Juan Atkins", "Tresor"],  # 2 entities → kept
            geographic_tags=["Detroit", "Berlin"],
            genre_tags=["techno"],
        )
        no_entity = DocumentChunk(
            chunk_id="chunk-none",
            text="Warehouse parties defined the 1990s rave scene.",
            token_count=35,
            source_id="src-scene",
            source_title="Rave Culture",
            source_type="book",
            author="Unknown",
            citation_tier=2,
            entity_tags=[],                    # no entity → kept
            geographic_tags=[],
            genre_tags=["rave"],
        )

        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[
            RetrievedChunk(chunk=single_entity, similarity_score=0.95,
                           formatted_citation="DJ Mag Profile [T4]"),
            RetrievedChunk(chunk=multi_entity, similarity_score=0.85,
                           formatted_citation="Energy Flash [T1]"),
            RetrievedChunk(chunk=no_entity, similarity_score=0.80,
                           formatted_citation="Rave Culture [T2]"),
        ])

        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "techno in the 90s"},  # NOT an artist query
        )

        assert resp.status_code == 200
        data = resp.json()
        titles = [r["source_title"] for r in data["results"]]
        # Single-entity "DJ Mag Profile" should be filtered out
        assert "DJ Mag Profile" not in titles
        # Multi-entity and no-entity results should remain
        assert "Energy Flash" in titles
        assert "Rave Culture" in titles
        assert data["total_results"] == 2

    def test_artist_query_keeps_single_entity_chunks(self) -> None:
        """Chunks with exactly 1 entity tag are kept for artist queries."""
        single_entity = DocumentChunk(
            chunk_id="chunk-single",
            text="Carl Cox biography and career overview.",
            token_count=40,
            source_id="src-bio",
            source_title="DJ Mag Profile",
            source_type="article",
            author="Staff",
            citation_tier=4,
            entity_tags=["Carl Cox"],
            geographic_tags=["London"],
            genre_tags=["techno"],
        )

        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[
            RetrievedChunk(chunk=single_entity, similarity_score=0.95,
                           formatted_citation="DJ Mag Profile [T4]"),
        ])

        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "artists similar to Jeff Mills"},  # IS an artist query
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_results"] == 1
        assert data["results"][0]["source_title"] == "DJ Mag Profile"

    def test_non_artist_query_keeps_no_entity_and_multi_entity(self) -> None:
        """No-entity and multi-entity chunks pass through for any query."""
        no_entity = DocumentChunk(
            chunk_id="chunk-none",
            text="The origins of acid house.",
            token_count=30,
            source_id="src-a",
            source_title="Acid Primer",
            source_type="article",
            author="Staff",
            citation_tier=3,
            entity_tags=[],
            geographic_tags=["Chicago"],
            genre_tags=["acid house"],
        )
        three_entities = DocumentChunk(
            chunk_id="chunk-three",
            text="Knuckles, Hardy, and Jefferson laid the groundwork.",
            token_count=40,
            source_id="src-b",
            source_title="House Origins",
            source_type="book",
            author="Author",
            citation_tier=1,
            entity_tags=["Frankie Knuckles", "Ron Hardy", "Marshall Jefferson"],
            geographic_tags=["Chicago"],
            genre_tags=["house"],
        )

        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[
            RetrievedChunk(chunk=no_entity, similarity_score=0.90,
                           formatted_citation="Acid Primer [T3]"),
            RetrievedChunk(chunk=three_entities, similarity_score=0.85,
                           formatted_citation="House Origins [T1]"),
        ])

        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "house music chicago"},  # NOT an artist query
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_results"] == 2
        titles = [r["source_title"] for r in data["results"]]
        assert "Acid Primer" in titles
        assert "House Origins" in titles

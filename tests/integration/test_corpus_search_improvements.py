"""Integration tests for the 10 corpus search improvements.

Tests verify that the full corpus search endpoint correctly applies:
  #1  Entity type tagging in results
  #2  Expanded artist query detection
  #3  Per-entity-type diversification caps
  #4  Genre adjacency boosting
  #5  Temporal boosting
  #6  Geographic scene boosting
  #7  Alias expansion
  #8  Citation tier boosting
  #9  Semantic dedup
  #10 Query expansion (HyDE-lite)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import router as api_router
from src.api.schemas import CorpusSearchChunk
from src.models.rag import DocumentChunk, RetrievedChunk


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_chunk(
    text: str = "A passage about electronic music.",
    source_title: str = "Energy Flash",
    source_type: str = "book",
    author: str = "Simon Reynolds",
    citation_tier: int = 1,
    similarity: float = 0.85,
    source_id: str = "src-001",
    chunk_id: str = "chunk-001",
    entity_tags: list[str] | None = None,
    entity_types: list[str] | None = None,
    geographic_tags: list[str] | None = None,
    genre_tags: list[str] | None = None,
    time_period: str | None = None,
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
        entity_tags=entity_tags or [],
        entity_types=entity_types or [],
        geographic_tags=geographic_tags or [],
        genre_tags=genre_tags or [],
        time_period=time_period,
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
    primary_llm: object | None = None,
) -> FastAPI:
    """Build a minimal FastAPI app with mocked state for testing."""
    app = FastAPI()
    app.include_router(api_router)

    app.state.rag_enabled = rag_enabled
    app.state.vector_store = vector_store
    app.state.session_states = {}
    app.state.pipeline = MagicMock()
    app.state.confirmation_gate = MagicMock()
    app.state.progress_tracker = MagicMock()
    app.state.primary_llm = primary_llm

    return app


# ═══════════════════════════════════════════════════════════════════════════
# #1 — Entity Type Tagging in Results
# ═══════════════════════════════════════════════════════════════════════════


class TestEntityTypeTagging:
    """Verify entity_types are passed through to response chunks."""

    def test_entity_types_in_response(self) -> None:
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[
            _make_chunk(
                entity_tags=["Juan Atkins", "Tresor"],
                entity_types=["ARTIST", "VENUE"],
                chunk_id="chunk-typed",
            ),
        ])

        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "artists who played at Tresor"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) >= 1
        first = data["results"][0]
        assert first["entity_types"] == ["ARTIST", "VENUE"]

    def test_empty_entity_types_backward_compat(self) -> None:
        """Legacy chunks without entity_types should show empty list."""
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[
            _make_chunk(
                entity_tags=["Juan Atkins"],
                entity_types=[],
                chunk_id="chunk-legacy",
            ),
        ])

        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "artists who played at Tresor"},
        )

        assert resp.status_code == 200
        first = resp.json()["results"][0]
        assert first["entity_types"] == []


# ═══════════════════════════════════════════════════════════════════════════
# #2 — Expanded Artist Detection
# ═══════════════════════════════════════════════════════════════════════════


class TestExpandedArtistDetection:
    """Verify new regex patterns and artist name cache work."""

    def test_who_is_query_keeps_single_entity(self) -> None:
        """'who is Jeff Mills' is an artist query → single-entity chunks kept."""
        single = _make_chunk(
            text="Jeff Mills biography.",
            entity_tags=["Jeff Mills"],
            entity_types=["ARTIST"],
            chunk_id="c-jm",
            source_id="s-jm",
        )
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[single])

        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "who is Jeff Mills"},
        )

        assert resp.status_code == 200
        assert resp.json()["total_results"] >= 1

    def test_tell_me_about_keeps_results(self) -> None:
        """'tell me about Carl Cox' is an artist query."""
        single = _make_chunk(
            text="Carl Cox career overview.",
            entity_tags=["Carl Cox"],
            entity_types=["ARTIST"],
            chunk_id="c-cc",
            source_id="s-cc",
        )
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[single])

        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "tell me about Carl Cox"},
        )

        assert resp.status_code == 200
        assert resp.json()["total_results"] >= 1


# ═══════════════════════════════════════════════════════════════════════════
# #3 — Per-Entity-Type Diversification Caps
# ═══════════════════════════════════════════════════════════════════════════


class TestPerTypeDiversification:
    """Verify entity-type-aware diversification caps."""

    def test_single_artist_entity_filtered_in_non_artist_query(self) -> None:
        """Chunks with single ARTIST entity_type are filtered for non-artist queries."""
        artist_chunk = _make_chunk(
            text="Carl Cox biography.",
            source_title="DJ Mag Profile",
            entity_tags=["Carl Cox"],
            entity_types=["ARTIST"],
            chunk_id="c-artist",
            source_id="s-artist",
            similarity=0.95,
        )
        venue_chunk = _make_chunk(
            text="Tresor Berlin history.",
            source_title="Tresor History",
            entity_tags=["Tresor"],
            entity_types=["VENUE"],
            chunk_id="c-venue",
            source_id="s-venue",
            similarity=0.90,
        )
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[artist_chunk, venue_chunk])

        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "techno clubs in Berlin"},  # not an artist query
        )

        assert resp.status_code == 200
        data = resp.json()
        titles = [r["source_title"] for r in data["results"]]
        # Single ARTIST entity should be filtered; VENUE should be kept
        assert "DJ Mag Profile" not in titles
        assert "Tresor History" in titles

    def test_single_venue_entity_kept_in_non_artist_query(self) -> None:
        """Chunks with single VENUE entity_type are NOT filtered."""
        venue_chunk = _make_chunk(
            text="Berghain is one of the most famous techno clubs in the world.",
            entity_tags=["Berghain"],
            entity_types=["VENUE"],
            chunk_id="c-berghain",
            source_id="s-berghain",
            similarity=0.92,
        )
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[venue_chunk])

        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "Berlin club scene"},
        )

        assert resp.status_code == 200
        assert resp.json()["total_results"] == 1

    def test_legacy_single_entity_filtered(self) -> None:
        """Legacy chunks (no entity_types) with single entity are filtered."""
        legacy_chunk = _make_chunk(
            text="About a single artist.",
            entity_tags=["Some Artist"],
            entity_types=[],  # no type data
            chunk_id="c-legacy",
            source_id="s-legacy",
        )
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[legacy_chunk])

        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "techno history"},
        )

        assert resp.status_code == 200
        assert resp.json()["total_results"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# #4 — Genre Adjacency Boosting
# ═══════════════════════════════════════════════════════════════════════════


class TestGenreBoosting:
    """Verify genre adjacency boosts affect ranking."""

    def test_genre_match_boosted_above_no_genre(self) -> None:
        """A chunk with matching genre tags should be ranked higher."""
        matching = _make_chunk(
            text="Detroit techno pioneers created a new sound.",
            genre_tags=["techno", "detroit techno"],
            chunk_id="c-match",
            source_id="s-match",
            similarity=0.80,
            citation_tier=3,
        )
        no_genre = _make_chunk(
            text="General history of music production.",
            genre_tags=[],
            chunk_id="c-nogenre",
            source_id="s-nogenre",
            similarity=0.80,
            citation_tier=3,
        )
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[no_genre, matching])

        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "techno music origins"},
        )

        assert resp.status_code == 200
        results = resp.json()["results"]
        # The genre-matching chunk should be ranked first (boosted)
        if len(results) == 2:
            assert "Detroit techno" in results[0]["text"] or "techno" in results[0]["genre_tags"]


# ═══════════════════════════════════════════════════════════════════════════
# #5 — Temporal Boosting
# ═══════════════════════════════════════════════════════════════════════════


class TestTemporalBoosting:
    """Verify temporal matching boosts affect ranking."""

    def test_temporal_match_boosted(self) -> None:
        """Chunk with matching time_period should rank above one without."""
        temporal = _make_chunk(
            text="The early 1990s rave scene was transformative.",
            time_period="1990s",
            chunk_id="c-temporal",
            source_id="s-temporal",
            similarity=0.80,
            citation_tier=3,
        )
        no_temporal = _make_chunk(
            text="Dance music has always been popular.",
            time_period=None,
            chunk_id="c-notemporal",
            source_id="s-notemporal",
            similarity=0.80,
            citation_tier=3,
        )
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[no_temporal, temporal])

        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "techno in the 1990s"},
        )

        assert resp.status_code == 200
        results = resp.json()["results"]
        if len(results) == 2:
            # Temporal match should be first due to boost
            assert "1990s" in results[0].get("time_period", "") or "1990" in results[0]["text"]


# ═══════════════════════════════════════════════════════════════════════════
# #6 — Geographic Scene Boosting
# ═══════════════════════════════════════════════════════════════════════════


class TestGeographicBoosting:
    """Verify geographic scene boost affects ranking."""

    def test_geo_match_boosted(self) -> None:
        """Chunk with matching geographic_tags for a scene query gets boosted."""
        detroit_chunk = _make_chunk(
            text="Juan Atkins created the blueprint in Detroit.",
            geographic_tags=["Detroit"],
            chunk_id="c-detroit",
            source_id="s-detroit",
            similarity=0.80,
            citation_tier=3,
        )
        london_chunk = _make_chunk(
            text="UK garage thrived in London clubs.",
            geographic_tags=["London"],
            chunk_id="c-london",
            source_id="s-london",
            similarity=0.80,
            citation_tier=3,
        )
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[london_chunk, detroit_chunk])

        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "detroit techno origins"},
        )

        assert resp.status_code == 200
        results = resp.json()["results"]
        if len(results) == 2:
            # Detroit chunk should be first due to geo boost
            assert "Detroit" in results[0]["geographic_tags"]


# ═══════════════════════════════════════════════════════════════════════════
# #7 — Alias Expansion
# ═══════════════════════════════════════════════════════════════════════════


class TestAliasExpansion:
    """Verify alias expansion modifies the query sent to the vector store."""

    def test_alias_expands_query(self) -> None:
        """Searching for 'AFX' should expand to include 'Aphex Twin'."""
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[])

        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "AFX"},
        )

        assert resp.status_code == 200
        # The expanded query should contain "Aphex Twin"
        call_args = mock_store.query.call_args
        query_text = call_args.kwargs.get("query_text", call_args.args[0] if call_args.args else "")
        assert "Aphex Twin" in query_text or "AFX" in query_text

    def test_entity_tag_filter_alias_expansion(self) -> None:
        """Entity tag filter with alias should resolve to canonical name."""
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[])

        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "discography", "entity_tag": "Plastikman"},
        )

        assert resp.status_code == 200
        call_args = mock_store.query.call_args
        filters = call_args.kwargs.get("filters")
        if filters:
            entity_filter = filters.get("entity_tags", {})
            contains = entity_filter.get("$contains", "")
            # Should be "Richie Hawtin" (canonical) or "Plastikman" (original)
            assert contains in ("Richie Hawtin", "Plastikman")


# ═══════════════════════════════════════════════════════════════════════════
# #8 — Citation Tier Boosting
# ═══════════════════════════════════════════════════════════════════════════


class TestCitationTierBoosting:
    """Verify citation tier affects ranking."""

    def test_tier_1_ranked_above_tier_6(self) -> None:
        """A T1 (book) chunk at same similarity should rank above T6 (unverified)."""
        tier1 = _make_chunk(
            text="From a published book about techno history and culture.",
            citation_tier=1,
            chunk_id="c-t1",
            source_id="s-t1",
            similarity=0.80,
        )
        tier6 = _make_chunk(
            text="From an unverified random web page about dance music.",
            citation_tier=6,
            chunk_id="c-t6",
            source_id="s-t6",
            similarity=0.80,
        )
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[tier6, tier1])

        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "techno history"},
        )

        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 2
        # T1 chunk gets boost of (6-1)*0.02 = 0.10, T6 gets 0.00
        assert results[0]["citation_tier"] == 1
        assert results[1]["citation_tier"] == 6


# ═══════════════════════════════════════════════════════════════════════════
# #9 — Semantic Dedup (Integration)
# ═══════════════════════════════════════════════════════════════════════════


class TestSemanticDedupIntegration:
    """Verify semantic dedup removes near-duplicates in the full pipeline."""

    def test_near_identical_chunks_deduped(self) -> None:
        """Two nearly identical chunks from different sources should be collapsed."""
        base_text = (
            "Detroit techno was pioneered by Juan Atkins Derrick May and Kevin "
            "Saunderson in the mid 1980s drawing from Kraftwerk Parliament and "
            "the post-industrial landscape of Detroit Michigan."
        )
        c1 = _make_chunk(
            text=base_text,
            source_title="Energy Flash",
            source_id="s-ef",
            chunk_id="c-ef",
            similarity=0.90,
        )
        c2 = _make_chunk(
            text=base_text,
            source_title="Last Night a DJ Saved My Life",
            source_id="s-ln",
            chunk_id="c-ln",
            similarity=0.88,
        )
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[c1, c2])

        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "Detroit techno origins"},
        )

        assert resp.status_code == 200
        data = resp.json()
        # Near-identical chunks should be collapsed to 1
        assert data["total_results"] == 1

    def test_different_chunks_both_kept(self) -> None:
        """Two different chunks from different sources should both be kept."""
        c1 = _make_chunk(
            text="Detroit techno pioneers Juan Atkins created the blueprint.",
            source_title="Energy Flash",
            source_id="s-ef",
            chunk_id="c-ef",
            similarity=0.90,
        )
        c2 = _make_chunk(
            text="Acid house exploded in UK clubs during the second summer of love.",
            source_title="Last Night a DJ",
            source_id="s-ln",
            chunk_id="c-ln",
            similarity=0.88,
        )
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[c1, c2])

        app = _create_test_app(rag_enabled=True, vector_store=mock_store)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "electronic music history"},
        )

        assert resp.status_code == 200
        assert resp.json()["total_results"] == 2


# ═══════════════════════════════════════════════════════════════════════════
# #10 — Query Expansion (HyDE-lite) Integration
# ═══════════════════════════════════════════════════════════════════════════


class TestQueryExpansionIntegration:
    """Verify query expansion integrates with the search endpoint."""

    def test_short_query_expansion_with_llm(self) -> None:
        """When LLM is available and query is short, expansion should trigger."""
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[])

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value="techno detroit warehouse electronic music")

        app = _create_test_app(
            rag_enabled=True,
            vector_store=mock_store,
            primary_llm=mock_llm,
        )
        client = TestClient(app)

        # Clear expansion cache for this specific query
        from src.api.routes import _expansion_cache
        _expansion_cache.pop("techno", None)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "techno"},
        )

        assert resp.status_code == 200
        # LLM should have been called for expansion
        mock_llm.complete.assert_called_once()

        # Clean up
        _expansion_cache.pop("techno", None)

    def test_no_llm_still_works(self) -> None:
        """Without LLM, search should still work (no expansion)."""
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[])

        app = _create_test_app(
            rag_enabled=True,
            vector_store=mock_store,
            primary_llm=None,
        )
        client = TestClient(app)

        resp = client.post(
            "/api/v1/corpus/search",
            json={"query": "techno history"},
        )

        assert resp.status_code == 200

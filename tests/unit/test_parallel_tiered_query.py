"""Unit tests for Phase 2 optimizations: parallel tiered corpus query and
pre-computed embedding logic.

# ─── MODULE OVERVIEW ───────────────────────────────────────────────────
#
# Tests the following Phase 2 optimizations in the raiveFlier search pipeline:
#
#   1. IVectorStoreProvider.embed_query() — delegates to the injected
#      embedding provider to pre-compute a query embedding once.
#   2. ChromaDBProvider.query(query_embedding=...) — accepts an optional
#      pre-computed embedding to skip redundant embed_single() calls
#      when the same query is sent to multiple tier queries concurrently.
#   3. _tiered_corpus_query() — module-level function in routes.py that
#      executes 4 tier queries against ChromaDB with dedup.
#
# These tests verify the embedding pre-computation and pass-through
# logic without requiring a real ChromaDB instance or embedding model.
# All external dependencies are mocked via unittest.mock.
#
# Tests for ChromaDBProvider (sections 1 & 2) require the `chromadb`
# package to be importable; they are automatically skipped via
# pytest.importorskip when the local environment has a broken chromadb
# installation (e.g., OpenTelemetry version mismatch).
#
# Architectural layer: Testing layer
# Depends on: IVectorStoreProvider, IEmbeddingProvider interfaces,
#             ChromaDBProvider concrete adapter, routes._tiered_corpus_query
# ───────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.rag import DocumentChunk, RetrievedChunk

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_retrieved_chunk(
    source_id: str = "s1",
    source_title: str = "EX.001",
    source_type: str = "interview",
    text: str = "RA Exchange podcast content",
    similarity: float = 0.90,
) -> RetrievedChunk:
    """Build a RetrievedChunk for test assertions.

    Wraps a DocumentChunk inside a RetrievedChunk with a formatted
    citation, mirroring the structure returned by ChromaDBProvider.query().
    """
    chunk = DocumentChunk(
        chunk_id=f"chunk_{source_id}",
        source_id=source_id,
        source_type=source_type,
        source_title=source_title,
        text=text,
        citation_tier=1,
    )
    return RetrievedChunk(
        chunk=chunk,
        similarity_score=similarity,
        formatted_citation=f"{source_title} [Tier 1]",
    )


def _make_mock_vector_store(
    query_results: list[RetrievedChunk] | None = None,
) -> MagicMock:
    """Create a mock IVectorStoreProvider with sensible defaults.

    The mock's query() returns the provided results for all calls,
    and embed_query() returns a fixed 3-dimensional embedding vector.
    """
    mock = MagicMock()
    mock.query = AsyncMock(return_value=query_results or [])
    # embed_query returns a fixed embedding vector so tests can verify
    # it was called and its return value was passed to query().
    mock.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return mock


# ═══════════════════════════════════════════════════════════════════════════
# 1. embed_query — delegates to the injected embedding provider
# ═══════════════════════════════════════════════════════════════════════════

# Guard: ChromaDB-dependent test classes (TestEmbedQuery and
# TestQueryWithPrecomputedEmbedding) are marked to skip when chromadb
# cannot be imported — e.g., broken OpenTelemetry transitive deps.
# The TestTieredCorpusQuery class does NOT depend on chromadb (it uses
# mock vector stores) and runs unconditionally.
_has_chromadb = True
try:
    import chromadb as _chromadb  # noqa: F401
except (ImportError, Exception):
    _has_chromadb = False

_skip_no_chromadb = pytest.mark.skipif(
    not _has_chromadb,
    reason="chromadb not importable (OpenTelemetry dep conflict)",
)


@_skip_no_chromadb
class TestEmbedQuery:
    """Verify that embed_query() on the ChromaDB provider delegates
    to the injected IEmbeddingProvider.embed_single() correctly."""

    @pytest.mark.asyncio()
    async def test_embed_query_delegates_to_provider(self, tmp_path) -> None:
        """embed_query() should call through to the embedding provider's
        embed_single() method and return its result unchanged."""
        from src.interfaces.embedding_provider import IEmbeddingProvider
        from src.providers.vector_store.chromadb_provider import ChromaDBProvider

        # Arrange: mock embedding provider returning a known vector
        expected_embedding = [0.5, 0.6, 0.7, 0.8]
        mock_emb = MagicMock(spec=IEmbeddingProvider)
        mock_emb.embed_single = AsyncMock(return_value=expected_embedding)
        mock_emb.get_dimension.return_value = 4
        mock_emb.get_provider_name.return_value = "mock"
        mock_emb.is_available.return_value = True

        # Build ChromaDBProvider with the mock embedding provider.
        # Uses tmp_path for isolated ChromaDB persistence.
        provider = ChromaDBProvider(
            embedding_provider=mock_emb,
            persist_directory=str(tmp_path / "chroma"),
            collection_name="test_embed_query",
        )

        # Act
        result = await provider.embed_query("acid house origins")

        # Assert: embed_single was called with the query text
        mock_emb.embed_single.assert_awaited_once_with("acid house origins")
        assert result == expected_embedding


# ═══════════════════════════════════════════════════════════════════════════
# 2. query(query_embedding=...) — pre-computed embedding bypass
# ═══════════════════════════════════════════════════════════════════════════


@_skip_no_chromadb
class TestQueryWithPrecomputedEmbedding:
    """Verify that ChromaDBProvider.query() correctly skips or performs
    internal embedding based on the query_embedding parameter."""

    @pytest.fixture()
    def mock_embedding_provider(self):
        """Mock IEmbeddingProvider for ChromaDBProvider initialization."""
        from src.interfaces.embedding_provider import IEmbeddingProvider

        mock = MagicMock(spec=IEmbeddingProvider)
        mock.embed_single = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        mock.get_dimension.return_value = 3
        mock.get_provider_name.return_value = "mock"
        mock.is_available.return_value = True
        return mock

    @pytest.mark.asyncio()
    async def test_query_with_precomputed_embedding_skips_embed(
        self, mock_embedding_provider, tmp_path
    ) -> None:
        """When query_embedding is provided, the provider should NOT call
        embed_single() — the pre-computed vector is used directly."""
        from src.providers.vector_store.chromadb_provider import ChromaDBProvider

        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma"),
            collection_name="test_precomputed",
        )

        precomputed = [0.9, 0.8, 0.7]
        await provider.query(
            query_text="detroit techno",
            top_k=5,
            query_embedding=precomputed,
        )

        # embed_single should NOT have been called because we passed
        # the embedding directly.
        mock_embedding_provider.embed_single.assert_not_awaited()

    @pytest.mark.asyncio()
    async def test_query_without_embedding_embeds_internally(
        self, mock_embedding_provider, tmp_path
    ) -> None:
        """When query_embedding is None (the default), the provider should
        call embed_single() internally to compute the query vector."""
        from src.providers.vector_store.chromadb_provider import ChromaDBProvider

        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma"),
            collection_name="test_internal_embed",
        )

        await provider.query(
            query_text="warehouse parties",
            top_k=5,
        )

        # embed_single SHOULD have been called because no pre-computed
        # embedding was provided.
        mock_embedding_provider.embed_single.assert_awaited_once_with(
            "warehouse parties"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3. _tiered_corpus_query — 4-tier query with dedup
# ═══════════════════════════════════════════════════════════════════════════


class TestTieredCorpusQuery:
    """Test the _tiered_corpus_query() module-level function from routes.py.

    These tests verify tier assignment, cross-tier dedup via seen_ids, and
    proper handling of query failures in individual tiers."""

    @pytest.mark.asyncio()
    async def test_returns_chunks_from_all_tiers(self) -> None:
        """When all 4 tiers return results, chunks from each tier
        should appear in the output with correct tier labels."""
        from src.api.routes import _tiered_corpus_query

        # Build distinct chunks for each tier
        t1_chunk = _make_retrieved_chunk(
            source_id="ra1", source_title="EX.001",
            source_type="interview", text="RA Exchange content",
        )
        t2_chunk = _make_retrieved_chunk(
            source_id="book1", source_title="Energy Flash",
            source_type="book", text="Book content",
        )
        t3a_chunk = _make_retrieved_chunk(
            source_id="evt1", source_title="Berghain Opening",
            source_type="event", text="Event listing",
        )
        t3b_chunk = _make_retrieved_chunk(
            source_id="ref1", source_title="Reference Doc",
            source_type="reference", text="Reference content",
        )

        # Configure mock to return tier-appropriate chunks based on
        # the filters argument.
        async def _mock_query(
            query_text: str,
            top_k: int = 20,
            filters: dict | None = None,
            **kwargs,
        ) -> list[RetrievedChunk]:
            if filters and filters.get("source_type") == "interview":
                return [t1_chunk]
            if filters and filters.get("source_type") == "book":
                return [t2_chunk]
            if filters and filters.get("source_type") == "event":
                return [t3a_chunk]
            # Tier 3b catch-all
            return [t3b_chunk]

        mock_vs = MagicMock()
        mock_vs.query = AsyncMock(side_effect=_mock_query)

        all_chunks, tier_map, tiers_used = await _tiered_corpus_query(
            mock_vs, "techno history", None,
        )

        assert len(all_chunks) == 4
        # Tiers 1 and 2 should be present; tier 3 covers both 3a and 3b.
        assert 1 in tiers_used
        assert 2 in tiers_used
        assert 3 in tiers_used

    @pytest.mark.asyncio()
    async def test_dedup_across_tiers(self) -> None:
        """Chunks with the same source_id appearing in multiple tiers
        should only be kept from the first tier that returns them."""
        from src.api.routes import _tiered_corpus_query

        # Same source_id in both T1 and T2 results
        shared_chunk = _make_retrieved_chunk(
            source_id="shared_src",
            source_title="EX.002",
            source_type="interview",
        )

        call_count = 0

        async def _mock_query(
            query_text: str,
            top_k: int = 20,
            filters: dict | None = None,
            **kwargs,
        ) -> list[RetrievedChunk]:
            nonlocal call_count
            call_count += 1
            if filters and filters.get("source_type") == "interview":
                return [shared_chunk]
            if filters and filters.get("source_type") == "book":
                # Return the same chunk from T2 — should be deduped out
                return [shared_chunk]
            return []

        mock_vs = MagicMock()
        mock_vs.query = AsyncMock(side_effect=_mock_query)

        all_chunks, _, _ = await _tiered_corpus_query(
            mock_vs, "techno", None,
        )

        # The shared chunk should appear only once (from T1, the first tier).
        source_ids = [c.chunk.source_id for c in all_chunks]
        assert source_ids.count("shared_src") == 1

    @pytest.mark.asyncio()
    async def test_tier_failure_does_not_block_others(self) -> None:
        """If one tier's query raises an exception, the other tiers
        should still return their results."""
        from src.api.routes import _tiered_corpus_query

        t2_chunk = _make_retrieved_chunk(
            source_id="book1", source_title="Last Night a DJ",
            source_type="book",
        )

        async def _mock_query(
            query_text: str,
            top_k: int = 20,
            filters: dict | None = None,
            **kwargs,
        ) -> list[RetrievedChunk]:
            if filters and filters.get("source_type") == "interview":
                raise RuntimeError("T1 database timeout")
            if filters and filters.get("source_type") == "book":
                return [t2_chunk]
            return []

        mock_vs = MagicMock()
        mock_vs.query = AsyncMock(side_effect=_mock_query)

        all_chunks, _, tiers_used = await _tiered_corpus_query(
            mock_vs, "acid house", None,
        )

        # T1 failed but T2 should still have results
        assert any(c.chunk.source_type == "book" for c in all_chunks)
        assert 2 in tiers_used

    @pytest.mark.asyncio()
    async def test_empty_results(self) -> None:
        """When no tier returns results, the function should return
        empty collections without errors."""
        from src.api.routes import _tiered_corpus_query

        mock_vs = _make_mock_vector_store(query_results=[])

        all_chunks, tier_map, tiers_used = await _tiered_corpus_query(
            mock_vs, "obscure query", None,
        )

        assert all_chunks == []
        assert tier_map == {}
        assert tiers_used == []

"""Unit tests for reciprocal rank fusion (hybrid search merging).

# ─── MODULE OVERVIEW ────────────────────────────────────────────────
# Tests the reciprocal_rank_fusion() function that merges semantic
# vector search results (CorpusSearchChunk) with BM25 keyword search
# results (BM25Result) into a single ranked list.
#
# RRF scoring: for item at rank r, score = weight / (k + r).
# Items appearing in both lists get their scores summed, naturally
# boosting results that both retrieval methods agree on.
#
# These tests verify: empty inputs, single-source results, overlap
# boosting, and weight asymmetry (semantic > BM25 by default).
# ────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import pytest

from src.api.schemas import CorpusSearchChunk
from src.providers.vector_store.bm25_provider import BM25Result
from src.utils.rank_fusion import reciprocal_rank_fusion

# ─── Helpers ───────────────────────────────────────────────────────


def _semantic_chunk(text: str, score: float = 0.9) -> CorpusSearchChunk:
    """Build a minimal CorpusSearchChunk for testing."""
    return CorpusSearchChunk(
        text=text,
        source_title="Test Source",
        source_type="book",
        citation_tier=2,
        similarity_score=score,
    )


def _bm25_result(text: str, score: float = 5.0) -> BM25Result:
    """Build a minimal BM25Result for testing."""
    return BM25Result(
        chunk_id="bm25-chunk",
        source_id="src-test",
        source_title="Test Source",
        source_type="book",
        text=text,
        citation_tier=2,
        bm25_score=score,
    )


# ─── Tests ─────────────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_rrf_empty_inputs_returns_empty() -> None:
    """Both inputs empty should return an empty list."""
    result = reciprocal_rank_fusion([], [])
    assert result == []


@pytest.mark.asyncio()
async def test_rrf_semantic_only() -> None:
    """When only semantic results are provided, they are returned ranked."""
    chunks = [
        _semantic_chunk("Detroit techno history", score=0.95),
        _semantic_chunk("Chicago house origins", score=0.80),
    ]
    result = reciprocal_rank_fusion(chunks, [])

    assert len(result) == 2
    # Results should be present (order preserved by RRF rank)
    texts = [r.text for r in result]
    assert "Detroit techno history" in texts
    assert "Chicago house origins" in texts


@pytest.mark.asyncio()
async def test_rrf_bm25_only() -> None:
    """When only BM25 results are provided, they are converted and returned."""
    bm25 = [
        _bm25_result("Tresor club opened in 1991", score=8.0),
        _bm25_result("Berlin techno after the wall", score=5.0),
    ]
    result = reciprocal_rank_fusion([], bm25)

    assert len(result) == 2
    # BM25 results should be converted to CorpusSearchChunk
    assert all(isinstance(r, CorpusSearchChunk) for r in result)
    texts = [r.text for r in result]
    assert "Tresor club opened in 1991" in texts


@pytest.mark.asyncio()
async def test_rrf_overlap_boosts_score() -> None:
    """An item appearing in both lists scores higher than items in only one.

    RRF sums scores from both lists for overlapping items.  An item at
    rank 1 in both lists gets semantic_weight/(k+1) + bm25_weight/(k+1),
    which is always higher than either alone.
    """
    overlap_text = "Carl Cox played at Tresor Berlin in 1997"
    semantic_only_text = "Jeff Mills Detroit techno pioneer"
    bm25_only_text = "Adam Beyer Drumcode label founder"

    semantic = [
        _semantic_chunk(overlap_text, score=0.9),
        _semantic_chunk(semantic_only_text, score=0.85),
    ]
    bm25 = [
        _bm25_result(overlap_text, score=7.0),
        _bm25_result(bm25_only_text, score=5.0),
    ]

    result = reciprocal_rank_fusion(semantic, bm25)

    # Find the overlap item — it should have the highest fused score
    overlap_result = next(r for r in result if "Carl Cox" in r.text)
    other_scores = [r.similarity_score for r in result if "Carl Cox" not in r.text]

    # The overlap item's fused score should exceed all non-overlap items
    assert all(overlap_result.similarity_score > s for s in other_scores)


@pytest.mark.asyncio()
async def test_rrf_weights_matter() -> None:
    """Semantic weight 1.0 vs BM25 weight 0.5: semantic rank-1 outscores BM25 rank-1.

    With default weights, a semantic-only item at rank 1 gets
    1.0 / (60 + 1) = 0.01639, while a BM25-only item at rank 1 gets
    0.5 / (60 + 1) = 0.00820.  The semantic item should rank higher.
    """
    sem_text = "Semantic-only result about acid house"
    bm25_text = "BM25-only result about warehouse parties"

    semantic = [_semantic_chunk(sem_text, score=0.9)]
    bm25 = [_bm25_result(bm25_text, score=10.0)]

    result = reciprocal_rank_fusion(
        semantic, bm25,
        semantic_weight=1.0,
        bm25_weight=0.5,
    )

    assert len(result) == 2
    # Semantic-only at rank 1 should score higher than BM25-only at rank 1
    sem_result = next(r for r in result if "acid house" in r.text)
    bm25_result_item = next(r for r in result if "warehouse" in r.text)
    assert sem_result.similarity_score > bm25_result_item.similarity_score

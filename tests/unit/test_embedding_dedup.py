"""Unit tests for embedding-based semantic dedup and cosine similarity.

# ─── MODULE OVERVIEW ───────────────────────────────────────────────────
#
# Tests the Phase 2 dedup optimizations in the corpus search pipeline:
#
#   1. _cosine_similarity(a, b) — pure-math helper that computes the
#      cosine similarity between two float vectors.  Returns a value
#      in [-1, 1].
#   2. _semantic_dedup(results, threshold) — near-duplicate removal
#      for CorpusSearchChunk lists.  When chunks carry a pre-computed
#      `.embedding` vector, cosine similarity is used (threshold ~0.93).
#      When embeddings are absent, the function falls back to word-level
#      Jaccard similarity (threshold default 0.85).
#
# Both functions are module-level helpers in src/api/routes.py and are
# imported directly.  CorpusSearchChunk objects are built from the
# schema in src/api/schemas.py.
#
# Architectural layer: Testing layer
# Depends on: src.api.routes (_cosine_similarity, _semantic_dedup),
#             src.api.schemas (CorpusSearchChunk)
# ───────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math

import pytest

from src.api.routes import _cosine_similarity, _semantic_dedup
from src.api.schemas import CorpusSearchChunk

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_chunk(
    text: str = "A passage about techno.",
    source_title: str = "Energy Flash",
    source_type: str = "book",
    citation_tier: int = 1,
    similarity_score: float = 0.90,
    embedding: list[float] | None = None,
) -> CorpusSearchChunk:
    """Build a CorpusSearchChunk for dedup tests.

    The optional embedding parameter lets tests attach a pre-computed
    vector to switch _semantic_dedup into cosine-similarity mode for
    that chunk pair.
    """
    return CorpusSearchChunk(
        text=text,
        source_title=source_title,
        source_type=source_type,
        citation_tier=citation_tier,
        similarity_score=similarity_score,
        embedding=embedding,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. _cosine_similarity — vector math
# ═══════════════════════════════════════════════════════════════════════════


class TestCosineSimilarity:
    """Test the _cosine_similarity() pure function."""

    def test_cosine_identical_vectors(self) -> None:
        """Identical vectors should have cosine similarity of 1.0."""
        result = _cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert result == pytest.approx(1.0)

    def test_cosine_orthogonal_vectors(self) -> None:
        """Orthogonal (perpendicular) vectors should have cosine similarity
        of 0.0 — they share no directional component."""
        result = _cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        assert result == pytest.approx(0.0)

    def test_cosine_opposite_vectors(self) -> None:
        """Opposite vectors should have cosine similarity of -1.0 — they
        point in exactly opposite directions."""
        result = _cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        assert result == pytest.approx(-1.0)

    def test_cosine_non_unit_vectors(self) -> None:
        """Cosine similarity should be magnitude-independent — scaling
        a vector should not change the similarity value."""
        sim_unit = _cosine_similarity([1.0, 0.0], [0.0, 1.0])
        sim_scaled = _cosine_similarity([5.0, 0.0], [0.0, 10.0])
        assert sim_unit == pytest.approx(sim_scaled)

    def test_cosine_zero_vector_returns_zero(self) -> None:
        """A zero-magnitude vector should return 0.0 rather than
        raising a division-by-zero error."""
        result = _cosine_similarity([0.0, 0.0], [1.0, 0.0])
        assert result == pytest.approx(0.0)

    def test_cosine_similar_vectors(self) -> None:
        """Vectors pointing in nearly the same direction should have
        high cosine similarity (close to 1.0)."""
        # Two vectors at a ~10-degree angle
        a = [1.0, 0.0]
        b = [math.cos(math.radians(10)), math.sin(math.radians(10))]
        result = _cosine_similarity(a, b)
        assert result > 0.98


# ═══════════════════════════════════════════════════════════════════════════
# 2. _semantic_dedup — embedding-aware near-duplicate removal
# ═══════════════════════════════════════════════════════════════════════════


class TestSemanticDedupWithEmbeddings:
    """Test _semantic_dedup() with and without pre-computed embeddings.

    When chunks have the .embedding field set, dedup uses cosine
    similarity (threshold 0.93).  When embeddings are absent, it falls
    back to word-level Jaccard (threshold 0.85)."""

    def test_dedup_identical_embeddings_removes_one(self) -> None:
        """Two chunks with the same embedding vector should be treated
        as near-duplicates — only the first (higher-scored) survives."""
        emb = [1.0, 0.0, 0.0]
        c1 = _make_chunk(
            text="Detroit techno pioneers",
            source_title="Book A",
            similarity_score=0.95,
            embedding=emb,
        )
        c2 = _make_chunk(
            text="Different text but same embedding",
            source_title="Book B",
            similarity_score=0.90,
            embedding=emb,
        )
        result = _semantic_dedup([c1, c2])
        assert len(result) == 1
        assert result[0].source_title == "Book A"

    def test_dedup_different_embeddings_keeps_both(self) -> None:
        """Two chunks with orthogonal embeddings should both be kept —
        they represent genuinely different content."""
        c1 = _make_chunk(
            text="Detroit techno pioneers",
            source_title="Book A",
            similarity_score=0.95,
            embedding=[1.0, 0.0, 0.0],
        )
        c2 = _make_chunk(
            text="UK acid house scene",
            source_title="Book B",
            similarity_score=0.90,
            embedding=[0.0, 1.0, 0.0],
        )
        result = _semantic_dedup([c1, c2])
        assert len(result) == 2

    def test_dedup_no_embeddings_uses_jaccard(self) -> None:
        """Chunks without embedding fields should fall back to word-level
        Jaccard similarity.  Identical text (Jaccard = 1.0) should be
        deduped even without embeddings."""
        text = "Detroit techno emerged in the mid 1980s with Juan Atkins"
        c1 = _make_chunk(text=text, source_title="Book A", similarity_score=0.95)
        c2 = _make_chunk(text=text, source_title="Book B", similarity_score=0.90)
        # Both chunks have embedding=None (default)
        result = _semantic_dedup([c1, c2])
        assert len(result) == 1
        assert result[0].source_title == "Book A"

    def test_dedup_no_embeddings_different_text_kept(self) -> None:
        """Without embeddings, chunks with completely different text
        should both be kept (low Jaccard overlap)."""
        c1 = _make_chunk(
            text="Detroit techno pioneers Juan Atkins Derrick May",
            source_title="Book A",
        )
        c2 = _make_chunk(
            text="Acid house exploded in UK clubs during second summer of love",
            source_title="Book B",
        )
        result = _semantic_dedup([c1, c2])
        assert len(result) == 2

    def test_dedup_mixed_some_with_embeddings(self) -> None:
        """When some chunks have embeddings and some do not, the function
        should use cosine for pairs where both have embeddings, and
        Jaccard for pairs where at least one is missing an embedding."""
        # c1 and c2 have identical embeddings — should be deduped by cosine.
        emb = [1.0, 0.0, 0.0]
        c1 = _make_chunk(
            text="Techno in Detroit",
            source_title="Source A",
            similarity_score=0.95,
            embedding=emb,
        )
        c2 = _make_chunk(
            text="Totally different text but same vector",
            source_title="Source B",
            similarity_score=0.90,
            embedding=emb,
        )
        # c3 has no embedding — should use Jaccard against c1, and since
        # the text is completely different, it should survive.
        c3 = _make_chunk(
            text="UK garage scene developed in south London warehouses",
            source_title="Source C",
            similarity_score=0.85,
            embedding=None,
        )

        result = _semantic_dedup([c1, c2, c3])

        # c2 deduped against c1 via cosine; c3 kept via Jaccard fallback
        assert len(result) == 2
        titles = [r.source_title for r in result]
        assert "Source A" in titles
        assert "Source C" in titles
        assert "Source B" not in titles

    def test_dedup_mixed_no_embedding_jaccard_dup(self) -> None:
        """When a chunk without embedding has identical text to a kept
        chunk (also without embedding), Jaccard should catch it."""
        text = "warehouse raves in the 1990s"
        c1 = _make_chunk(
            text=text,
            source_title="Source A",
            similarity_score=0.95,
            embedding=[1.0, 0.0],
        )
        # c2 has the same text as c1 but no embedding — Jaccard fallback
        # should compare against c1's text and detect the duplicate.
        c2 = _make_chunk(
            text=text,
            source_title="Source B",
            similarity_score=0.85,
            embedding=None,
        )

        result = _semantic_dedup([c1, c2])
        assert len(result) == 1
        assert result[0].source_title == "Source A"

    def test_dedup_near_identical_embeddings(self) -> None:
        """Embeddings that are very close (cosine > 0.93 threshold)
        should trigger dedup even if the text differs."""
        # Two vectors with cosine similarity > 0.93
        c1 = _make_chunk(
            text="First passage about Berlin clubs",
            source_title="Book A",
            similarity_score=0.95,
            embedding=[1.0, 0.0, 0.0],
        )
        # Slightly rotated vector — cosine ~0.995
        c2 = _make_chunk(
            text="Completely different words here",
            source_title="Book B",
            similarity_score=0.90,
            embedding=[0.999, 0.05, 0.0],
        )
        result = _semantic_dedup([c1, c2])
        assert len(result) == 1

    def test_dedup_empty_list(self) -> None:
        """Empty input should return an empty list."""
        result = _semantic_dedup([])
        assert result == []

    def test_dedup_single_chunk(self) -> None:
        """A single chunk should be returned unchanged."""
        c = _make_chunk(embedding=[1.0, 0.0])
        result = _semantic_dedup([c])
        assert len(result) == 1
        assert result[0] is c

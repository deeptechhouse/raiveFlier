"""Reciprocal Rank Fusion for merging semantic and BM25 retrieval results.

Combines two ranked lists into one using the formula:

    fused_score(doc) = sum(weight / (k + rank))  across all lists containing doc

where k is a constant (default 60) that controls how much rank position
matters relative to absolute score differences.

# ─── WHY RECIPROCAL RANK FUSION? ──────────────────────────────────────
# Semantic search (ChromaDB cosine similarity) and BM25 keyword search
# produce scores on completely different scales — cosine similarity is
# [0, 1] while BM25 scores are unbounded negative numbers.  Directly
# combining these scores is meaningless.
#
# RRF sidesteps the score-normalization problem entirely by using only
# RANK positions.  A document ranked #1 by both systems gets a high
# fused score regardless of the raw score values.  The k parameter
# (default 60) dampens the contribution of low-ranked results so that
# a document at rank 100 barely affects the final ordering.
#
# Weight parameters let us express a preference: semantic_weight=1.0
# and bm25_weight=0.5 means semantic rank matters twice as much as
# BM25 rank, reflecting that semantic search is our primary signal
# and BM25 is a supplementary recall booster.
#
# Reference: Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion
# outperforms Condorcet and individual Rank Learning Methods" (SIGIR 2009)
#
# Layer: Utility (stateless function, no dependencies)
# Called by: routes.py corpus_search() after parallel retrieval
# ───────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(logger_name=__name__)


def _normalize_text_key(text: str) -> str:
    """Create a stable hash key from chunk text for cross-list matching.

    Uses the first 200 characters, lowercased and whitespace-normalized,
    as the matching key.  This is sufficient to identify the same chunk
    across semantic and BM25 result lists even when metadata differs.
    """
    return " ".join(text[:200].lower().split())


def reciprocal_rank_fusion(
    semantic_results: list[Any],
    bm25_results: list[Any],
    k: int = 60,
    semantic_weight: float = 1.0,
    bm25_weight: float = 0.5,
) -> list[Any]:
    """Fuse semantic and BM25 results using Reciprocal Rank Fusion.

    Parameters
    ----------
    semantic_results:
        Ranked list of CorpusSearchChunk objects from ChromaDB semantic search.
        Expected to have .text, .similarity_score, and metadata attributes.
    bm25_results:
        Ranked list of BM25Result objects from SQLite FTS5 search.
        Expected to have .text, .source_title, .source_type, .citation_tier,
        .bm25_score attributes.
    k:
        RRF constant controlling rank position dampening.  Higher k means
        rank differences matter less.  The standard value of 60 works well
        across most IR benchmarks.
    semantic_weight:
        Multiplicative weight for semantic results' rank contribution.
    bm25_weight:
        Multiplicative weight for BM25 results' rank contribution.
        Default 0.5 (half semantic weight) because semantic search is
        the primary retrieval signal; BM25 supplements with exact matches.

    Returns
    -------
    list[CorpusSearchChunk]
        The input semantic_results re-sorted by fused RRF score, with
        BM25-only results (not found in semantic list) appended as new
        CorpusSearchChunk objects.  The similarity_score field on each
        chunk is NOT modified — the fused score is used only for ordering.
    """
    if not bm25_results:
        # No BM25 results to fuse — return semantic results unchanged
        return list(semantic_results)

    if not semantic_results:
        # No semantic results — convert BM25-only results to CorpusSearchChunk
        return _bm25_only_to_chunks(bm25_results)

    # ── Build RRF score map keyed by normalized text ──────────────────
    # Each entry tracks the cumulative RRF score and the source object.
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, Any] = {}

    # Score semantic results (rank is 1-based)
    for rank, chunk in enumerate(semantic_results, start=1):
        key = _normalize_text_key(chunk.text)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + semantic_weight / (k + rank)
        chunk_map[key] = chunk

    # Score BM25 results (rank is 1-based)
    bm25_only_keys: list[str] = []
    for rank, bm25_hit in enumerate(bm25_results, start=1):
        key = _normalize_text_key(bm25_hit.text)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + bm25_weight / (k + rank)
        # Track BM25-only results (not present in semantic results)
        if key not in chunk_map:
            bm25_only_keys.append(key)
            # Convert BM25Result to a CorpusSearchChunk-like dict for later
            chunk_map[key] = _bm25_to_chunk(bm25_hit)

    # ── Re-sort all results by fused RRF score (descending) ───────────
    sorted_keys = sorted(rrf_scores.keys(), key=lambda k_: rrf_scores[k_], reverse=True)
    fused_results = [chunk_map[key] for key in sorted_keys if key in chunk_map]

    logger.debug(
        "rank_fusion_complete",
        semantic_count=len(semantic_results),
        bm25_count=len(bm25_results),
        fused_count=len(fused_results),
        bm25_only_count=len(bm25_only_keys),
    )

    return fused_results


def _bm25_to_chunk(bm25_hit: Any) -> Any:
    """Convert a BM25Result to a CorpusSearchChunk object.

    Deferred import avoids circular dependency between utils and api layers.
    The constructed chunk has a similarity_score of 0.0 since BM25 scores
    are not on the same scale as cosine similarity — the RRF score handles
    ranking instead.
    """
    from src.api.schemas import CorpusSearchChunk

    return CorpusSearchChunk(
        text=bm25_hit.text,
        source_title=bm25_hit.source_title,
        source_type=bm25_hit.source_type,
        citation_tier=bm25_hit.citation_tier,
        # BM25-only chunks get a baseline similarity score since they
        # weren't returned by semantic search.  The domain-aware boost
        # in corpus_search() will adjust this further.
        similarity_score=0.0,
        formatted_citation=f"{bm25_hit.source_title} [Tier {bm25_hit.citation_tier}]",
    )


def _bm25_only_to_chunks(bm25_results: list[Any]) -> list[Any]:
    """Convert a list of BM25Results to CorpusSearchChunk objects.

    Used when semantic search returned zero results but BM25 found matches.
    """
    return [_bm25_to_chunk(hit) for hit in bm25_results]

"""Unit tests for corpus search helper functions in src/api/routes.py.

Tests cover:
  - _is_artist_query() — regex + artist name cache detection
  - _expand_query() — HyDE-lite LLM query expansion
  - _semantic_dedup() — Jaccard-based near-duplicate removal
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
    entity_tags: list[str] | None = None,
    entity_types: list[str] | None = None,
    geographic_tags: list[str] | None = None,
    genre_tags: list[str] | None = None,
    time_period: str | None = None,
) -> CorpusSearchChunk:
    return CorpusSearchChunk(
        text=text,
        source_title=source_title,
        source_type=source_type,
        citation_tier=citation_tier,
        similarity_score=similarity_score,
        entity_tags=entity_tags or [],
        entity_types=entity_types or [],
        geographic_tags=geographic_tags or [],
        genre_tags=genre_tags or [],
        time_period=time_period,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. _is_artist_query
# ═══════════════════════════════════════════════════════════════════════════


class TestIsArtistQuery:
    """Test the _is_artist_query() function."""

    def _call(self, query: str) -> bool:
        from src.api.routes import _is_artist_query
        return _is_artist_query(query)

    def test_artists_from_pattern(self) -> None:
        assert self._call("artists from Detroit") is True

    def test_dj_who_pattern(self) -> None:
        assert self._call("DJs who play acid techno") is True

    def test_similar_to_pattern(self) -> None:
        assert self._call("similar to Jeff Mills") is True

    def test_sounds_like_pattern(self) -> None:
        assert self._call("sounds like Carl Craig") is True

    def test_who_is_pattern(self) -> None:
        assert self._call("who is Jeff Mills") is True

    def test_tell_me_about_pattern(self) -> None:
        assert self._call("tell me about Carl Cox") is True

    def test_biography_pattern(self) -> None:
        assert self._call("Carl Cox biography") is True

    def test_profile_of_pattern(self) -> None:
        assert self._call("profile of Derrick May") is True

    def test_career_of_pattern(self) -> None:
        assert self._call("career of Juan Atkins") is True

    def test_non_artist_query(self) -> None:
        assert self._call("techno in the 90s") is False

    def test_generic_query(self) -> None:
        assert self._call("warehouse parties detroit") is False

    def test_known_artist_name_match(self) -> None:
        """Queries containing a known artist name from the alias table should match."""
        # "Aphex Twin" is in the alias table and >4 chars
        assert self._call("Aphex Twin") is True

    def test_short_name_exact_match(self) -> None:
        """Short names (<=4 chars) should require exact match."""
        # "AFX" is 3 chars — only matches if exact
        # The function checks len(name) > 4 for substring matching
        # So "AFX" needs to be exactly the query
        result = self._call("AFX")
        # AFX is lowered to "afx" in the cache; the query "AFX" is lowered to "afx"
        # Since len("afx") <= 4, substring matching is skipped;
        # exact match "afx" == "afx" should work
        assert result is True


# ═══════════════════════════════════════════════════════════════════════════
# 2. _expand_query
# ═══════════════════════════════════════════════════════════════════════════


class TestExpandQuery:
    """Test the _expand_query() async function."""

    @pytest.mark.asyncio()
    async def test_no_llm_returns_original(self) -> None:
        from src.api.routes import _expand_query
        result = await _expand_query(None, "techno history")
        assert result == "techno history"

    @pytest.mark.asyncio()
    async def test_long_query_skips_expansion(self) -> None:
        from src.api.routes import _expand_query
        long_query = "x" * 81
        mock_llm = AsyncMock()
        result = await _expand_query(mock_llm, long_query)
        assert result == long_query
        mock_llm.complete.assert_not_called()

    @pytest.mark.asyncio()
    async def test_short_query_calls_llm(self) -> None:
        from src.api.routes import _expand_query, _expansion_cache
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value="techno history detroit warehouse music")

        # Clear cache to force LLM call
        test_query = "test_unique_query_for_expansion_123"
        _expansion_cache.pop(test_query, None)

        result = await _expand_query(mock_llm, test_query)
        assert result == "techno history detroit warehouse music"
        mock_llm.complete.assert_called_once()

        # Clean up
        _expansion_cache.pop(test_query, None)

    @pytest.mark.asyncio()
    async def test_llm_failure_returns_original(self) -> None:
        from src.api.routes import _expand_query, _expansion_cache
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))

        test_query = "test_unique_query_for_failure_456"
        _expansion_cache.pop(test_query, None)

        result = await _expand_query(mock_llm, test_query)
        assert result == test_query

        # Clean up
        _expansion_cache.pop(test_query, None)

    @pytest.mark.asyncio()
    async def test_cached_result_returned(self) -> None:
        from src.api.routes import _expand_query, _expansion_cache
        test_query = "test_unique_query_for_cache_789"
        _expansion_cache[test_query] = "cached expansion"

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock()

        result = await _expand_query(mock_llm, test_query)
        assert result == "cached expansion"
        mock_llm.complete.assert_not_called()

        # Clean up
        _expansion_cache.pop(test_query, None)


# ═══════════════════════════════════════════════════════════════════════════
# 3. _semantic_dedup
# ═══════════════════════════════════════════════════════════════════════════


class TestSemanticDedup:
    """Test the _semantic_dedup() function."""

    def _call(
        self,
        results: list[CorpusSearchChunk],
        threshold: float = 0.85,
    ) -> list[CorpusSearchChunk]:
        from src.api.routes import _semantic_dedup
        return _semantic_dedup(results, threshold)

    def test_empty_list(self) -> None:
        assert self._call([]) == []

    def test_single_item(self) -> None:
        chunk = _make_chunk(text="Only one chunk here.")
        result = self._call([chunk])
        assert len(result) == 1

    def test_identical_texts_deduped(self) -> None:
        text = "Detroit techno emerged in the mid 1980s with Juan Atkins Derrick May and Kevin Saunderson"
        c1 = _make_chunk(text=text, source_title="Book A", similarity_score=0.95)
        c2 = _make_chunk(text=text, source_title="Book B", similarity_score=0.90)
        result = self._call([c1, c2])
        assert len(result) == 1
        assert result[0].source_title == "Book A"  # First (higher score) kept

    def test_different_texts_kept(self) -> None:
        c1 = _make_chunk(
            text="Detroit techno was pioneered by Juan Atkins in the mid 1980s",
            source_title="Book A",
        )
        c2 = _make_chunk(
            text="Acid house exploded in the UK clubs during the second summer of love",
            source_title="Book B",
        )
        result = self._call([c1, c2])
        assert len(result) == 2

    def test_near_duplicate_texts_deduped(self) -> None:
        # Same content with minor differences
        base_words = "Detroit techno emerged in the mid 1980s pioneers Juan Atkins Derrick May Kevin Saunderson Belleville Three futuristic electronic music"
        c1 = _make_chunk(text=base_words, source_title="Book A", similarity_score=0.95)
        c2 = _make_chunk(
            text=base_words + " extra word",
            source_title="Book B",
            similarity_score=0.90,
        )
        result = self._call([c1, c2])
        assert len(result) == 1

    def test_threshold_respected(self) -> None:
        # With a very low threshold, even similar texts should be kept
        text = "Detroit techno emerged in the mid 1980s with Juan Atkins Derrick May and Kevin Saunderson"
        c1 = _make_chunk(text=text, source_title="Book A")
        c2 = _make_chunk(text=text, source_title="Book B")
        result = self._call([c1, c2], threshold=0.999)
        # Identical texts have Jaccard = 1.0 > 0.999
        assert len(result) == 1

    def test_empty_text_kept(self) -> None:
        c1 = _make_chunk(text="Normal text about techno.", similarity_score=0.95)
        c2 = _make_chunk(text="", source_title="Empty", similarity_score=0.80)
        result = self._call([c1, c2])
        assert len(result) == 2  # Empty text is kept (no tokens -> always appended)

    def test_preserves_order(self) -> None:
        chunks = [
            _make_chunk(text=f"Unique passage number {i} about electronic music history", similarity_score=0.95 - i * 0.05)
            for i in range(5)
        ]
        result = self._call(chunks)
        assert len(result) == 5
        # Order should be preserved
        for i, r in enumerate(result):
            assert f"number {i}" in r.text

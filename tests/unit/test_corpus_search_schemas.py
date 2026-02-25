"""Unit tests for corpus search Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import CorpusSearchChunk, CorpusSearchRequest, CorpusSearchResponse


class TestCorpusSearchRequest:
    """Validate CorpusSearchRequest schema constraints."""

    def test_valid_minimal_request(self) -> None:
        req = CorpusSearchRequest(query="techno history")
        assert req.query == "techno history"
        # Default top_k raised from 10→50 for larger candidate pool
        assert req.top_k == 50
        assert req.source_type is None
        assert req.entity_tag is None
        assert req.geographic_tag is None
        # New filter fields default to None/0
        assert req.genre_tags is None
        assert req.time_period is None
        assert req.min_citation_tier is None
        assert req.min_similarity is None
        assert req.offset == 0
        assert req.page_size == 20

    def test_valid_full_request(self) -> None:
        req = CorpusSearchRequest(
            query="acid house",
            top_k=25,
            source_type=["book", "article"],
            entity_tag="Carl Cox",
            geographic_tag="Detroit",
        )
        assert req.top_k == 25
        assert req.source_type == ["book", "article"]
        assert req.entity_tag == "Carl Cox"
        assert req.geographic_tag == "Detroit"

    def test_empty_query_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CorpusSearchRequest(query="")

    def test_query_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CorpusSearchRequest(query="x" * 501)

    def test_query_at_max_length_accepted(self) -> None:
        req = CorpusSearchRequest(query="x" * 500)
        assert len(req.query) == 500

    def test_top_k_below_min_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CorpusSearchRequest(query="test", top_k=0)

    def test_top_k_above_max_rejected(self) -> None:
        # Max top_k raised from 50→100 for larger candidate pool
        with pytest.raises(ValidationError):
            CorpusSearchRequest(query="test", top_k=101)

    def test_top_k_at_boundaries_accepted(self) -> None:
        req_min = CorpusSearchRequest(query="test", top_k=1)
        req_max = CorpusSearchRequest(query="test", top_k=100)
        assert req_min.top_k == 1
        assert req_max.top_k == 100


class TestCorpusSearchChunk:
    """Validate CorpusSearchChunk schema."""

    def test_valid_chunk(self) -> None:
        chunk = CorpusSearchChunk(
            text="A passage about acid house in Chicago.",
            source_title="Energy Flash",
            source_type="book",
            author="Simon Reynolds",
            citation_tier=1,
            page_number="p.42",
            similarity_score=0.92,
            formatted_citation="Energy Flash, Simon Reynolds, p.42, 1998 [Tier 1]",
            entity_tags=["Frankie Knuckles", "Ron Hardy"],
            geographic_tags=["Chicago"],
            genre_tags=["acid house", "house"],
        )
        assert chunk.source_title == "Energy Flash"
        assert chunk.citation_tier == 1
        assert len(chunk.entity_tags) == 2

    def test_minimal_chunk(self) -> None:
        chunk = CorpusSearchChunk(
            text="Some text.",
            source_title="Unknown",
            source_type="article",
        )
        assert chunk.author is None
        assert chunk.citation_tier == 6
        assert chunk.similarity_score == 0.0
        assert chunk.entity_tags == []


class TestCorpusSearchResponse:
    """Validate CorpusSearchResponse schema."""

    def test_valid_response(self) -> None:
        resp = CorpusSearchResponse(
            query="techno",
            total_results=2,
            results=[
                CorpusSearchChunk(
                    text="Passage one.",
                    source_title="Book A",
                    source_type="book",
                ),
                CorpusSearchChunk(
                    text="Passage two.",
                    source_title="Article B",
                    source_type="article",
                ),
            ],
        )
        assert resp.total_results == 2
        assert len(resp.results) == 2

    def test_empty_response(self) -> None:
        resp = CorpusSearchResponse(query="obscure topic", total_results=0)
        assert resp.results == []

    def test_pagination_defaults(self) -> None:
        """Verify pagination metadata has safe defaults for backward compatibility."""
        resp = CorpusSearchResponse(query="test", total_results=5)
        assert resp.offset == 0
        assert resp.page_size == 20
        assert resp.has_more is False

    def test_pagination_with_more(self) -> None:
        resp = CorpusSearchResponse(
            query="test",
            total_results=42,
            offset=20,
            page_size=20,
            has_more=True,
        )
        assert resp.has_more is True
        assert resp.offset == 20


class TestCorpusSearchRequestNewFilters:
    """Validate new filter fields added for enhanced corpus search."""

    def test_genre_tags_filter(self) -> None:
        req = CorpusSearchRequest(
            query="electronic music",
            genre_tags=["techno", "house"],
        )
        assert req.genre_tags == ["techno", "house"]

    def test_time_period_filter(self) -> None:
        req = CorpusSearchRequest(query="rave", time_period="1990s")
        assert req.time_period == "1990s"

    def test_min_citation_tier_valid(self) -> None:
        req = CorpusSearchRequest(query="test", min_citation_tier=3)
        assert req.min_citation_tier == 3

    def test_min_citation_tier_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            CorpusSearchRequest(query="test", min_citation_tier=0)
        with pytest.raises(ValidationError):
            CorpusSearchRequest(query="test", min_citation_tier=7)

    def test_min_similarity_valid(self) -> None:
        req = CorpusSearchRequest(query="test", min_similarity=0.5)
        assert req.min_similarity == 0.5

    def test_min_similarity_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            CorpusSearchRequest(query="test", min_similarity=1.5)

    def test_pagination_fields(self) -> None:
        req = CorpusSearchRequest(query="test", offset=20, page_size=10)
        assert req.offset == 20
        assert req.page_size == 10

    def test_page_size_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            CorpusSearchRequest(query="test", page_size=0)
        with pytest.raises(ValidationError):
            CorpusSearchRequest(query="test", page_size=51)

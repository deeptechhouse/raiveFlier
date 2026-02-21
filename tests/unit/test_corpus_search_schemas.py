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
        assert req.top_k == 10
        assert req.source_type is None
        assert req.entity_tag is None
        assert req.geographic_tag is None

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
        with pytest.raises(ValidationError):
            CorpusSearchRequest(query="test", top_k=51)

    def test_top_k_at_boundaries_accepted(self) -> None:
        req_min = CorpusSearchRequest(query="test", top_k=1)
        req_max = CorpusSearchRequest(query="test", top_k=50)
        assert req_min.top_k == 1
        assert req_max.top_k == 50


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

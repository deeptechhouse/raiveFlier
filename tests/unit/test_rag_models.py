"""Unit tests for RAG pipeline Pydantic models."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from src.models.rag import CorpusStats, DocumentChunk, IngestionResult, RetrievedChunk


class TestDocumentChunk:
    """Validate DocumentChunk creation and field constraints."""

    def test_document_chunk_creation(self) -> None:
        chunk = DocumentChunk(
            chunk_id="chunk-001",
            text="Carl Cox played at Space Ibiza.",
            source_id="src-001",
            source_title="Energy Flash",
            source_type="book",
            author="Simon Reynolds",
            publication_date=date(1998, 4, 1),
            citation_tier=1,
            page_number="142",
            entity_tags=["Carl Cox", "Space Ibiza"],
            geographic_tags=["Ibiza", "Spain"],
            genre_tags=["techno", "house"],
        )

        assert chunk.chunk_id == "chunk-001"
        assert chunk.text == "Carl Cox played at Space Ibiza."
        assert chunk.source_id == "src-001"
        assert chunk.source_title == "Energy Flash"
        assert chunk.source_type == "book"
        assert chunk.author == "Simon Reynolds"
        assert chunk.publication_date == date(1998, 4, 1)
        assert chunk.citation_tier == 1
        assert chunk.page_number == "142"
        assert "Carl Cox" in chunk.entity_tags
        assert "Ibiza" in chunk.geographic_tags
        assert "techno" in chunk.genre_tags

    def test_citation_tier_range_valid(self) -> None:
        for tier in (1, 2, 3, 4, 5, 6):
            chunk = DocumentChunk(
                chunk_id=f"chunk-{tier}",
                text="text",
                source_id="src",
                source_title="title",
                source_type="book",
                citation_tier=tier,
            )
            assert chunk.citation_tier == tier

    def test_citation_tier_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            DocumentChunk(
                chunk_id="bad",
                text="text",
                source_id="src",
                source_title="title",
                source_type="book",
                citation_tier=0,
            )

        with pytest.raises(ValidationError):
            DocumentChunk(
                chunk_id="bad",
                text="text",
                source_id="src",
                source_title="title",
                source_type="book",
                citation_tier=7,
            )

    def test_defaults(self) -> None:
        chunk = DocumentChunk(
            chunk_id="d",
            text="t",
            source_id="s",
            source_title="st",
            source_type="article",
        )
        assert chunk.citation_tier == 6
        assert chunk.author is None
        assert chunk.publication_date is None
        assert chunk.page_number is None
        assert chunk.entity_tags == []
        assert chunk.geographic_tags == []
        assert chunk.genre_tags == []

    def test_frozen(self) -> None:
        chunk = DocumentChunk(
            chunk_id="f",
            text="frozen",
            source_id="s",
            source_title="st",
            source_type="book",
        )
        with pytest.raises(ValidationError):
            chunk.text = "modified"  # type: ignore[misc]


class TestRetrievedChunk:
    """Validate RetrievedChunk with similarity score constraints."""

    def _make_chunk(self) -> DocumentChunk:
        return DocumentChunk(
            chunk_id="rc-001",
            text="Test chunk text.",
            source_id="src-001",
            source_title="Test Source",
            source_type="book",
            citation_tier=2,
        )

    def test_retrieved_chunk_with_score(self) -> None:
        chunk = self._make_chunk()
        retrieved = RetrievedChunk(
            chunk=chunk,
            similarity_score=0.85,
            formatted_citation="Test Source, p.1, 2020 [Tier 2]",
        )

        assert retrieved.chunk.chunk_id == "rc-001"
        assert retrieved.similarity_score == 0.85
        assert "Tier 2" in retrieved.formatted_citation

    def test_similarity_score_bounds(self) -> None:
        chunk = self._make_chunk()

        # Valid boundary values
        for score in (0.0, 0.5, 1.0):
            rc = RetrievedChunk(chunk=chunk, similarity_score=score)
            assert rc.similarity_score == score

        # Out-of-range values
        with pytest.raises(ValidationError):
            RetrievedChunk(chunk=chunk, similarity_score=-0.1)

        with pytest.raises(ValidationError):
            RetrievedChunk(chunk=chunk, similarity_score=1.1)

    def test_defaults(self) -> None:
        chunk = self._make_chunk()
        rc = RetrievedChunk(chunk=chunk)
        assert rc.similarity_score == 0.0
        assert rc.formatted_citation == ""


class TestIngestionResult:
    """Validate IngestionResult fields."""

    def test_ingestion_result(self) -> None:
        result = IngestionResult(
            source_id="ing-001",
            source_title="Energy Flash",
            chunks_created=42,
            total_tokens=21000,
            ingestion_time=3.14,
        )

        assert result.source_id == "ing-001"
        assert result.source_title == "Energy Flash"
        assert result.chunks_created == 42
        assert result.total_tokens == 21000
        assert result.ingestion_time == 3.14

    def test_non_negative_constraints(self) -> None:
        with pytest.raises(ValidationError):
            IngestionResult(
                source_id="bad",
                source_title="bad",
                chunks_created=-1,
            )

        with pytest.raises(ValidationError):
            IngestionResult(
                source_id="bad",
                source_title="bad",
                total_tokens=-1,
            )

        with pytest.raises(ValidationError):
            IngestionResult(
                source_id="bad",
                source_title="bad",
                ingestion_time=-0.5,
            )

    def test_defaults(self) -> None:
        result = IngestionResult(source_id="x", source_title="y")
        assert result.chunks_created == 0
        assert result.total_tokens == 0
        assert result.ingestion_time == 0.0


class TestCorpusStats:
    """Validate CorpusStats fields."""

    def test_corpus_stats(self) -> None:
        stats = CorpusStats(
            total_chunks=150,
            total_sources=5,
            sources_by_type={"book": 3, "article": 2},
            entity_tag_count=42,
            geographic_tag_count=10,
        )

        assert stats.total_chunks == 150
        assert stats.total_sources == 5
        assert stats.sources_by_type == {"book": 3, "article": 2}
        assert stats.entity_tag_count == 42
        assert stats.geographic_tag_count == 10

    def test_sources_by_type_dict(self) -> None:
        stats = CorpusStats(
            total_chunks=10,
            total_sources=2,
            sources_by_type={"book": 5, "article": 3, "analysis": 2},
        )
        assert isinstance(stats.sources_by_type, dict)
        assert stats.sources_by_type["book"] == 5
        assert sum(stats.sources_by_type.values()) == 10

    def test_defaults(self) -> None:
        stats = CorpusStats()
        assert stats.total_chunks == 0
        assert stats.total_sources == 0
        assert stats.sources_by_type == {}
        assert stats.entity_tag_count == 0
        assert stats.geographic_tag_count == 0

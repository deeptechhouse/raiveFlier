"""Integration tests for the document ingestion pipeline.

Verifies end-to-end ingestion of books, articles, and analysis feedback
using mock embedding and vector store providers (no real API calls).
"""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.interfaces.article_provider import ArticleContent
from src.models.analysis import InterconnectionMap
from src.models.entities import Artist, EntityType
from src.models.flier import ExtractedEntities, ExtractedEntity, FlierImage, OCRResult
from src.models.pipeline import PipelinePhase, PipelineState
from src.models.research import ResearchResult
from src.services.ingestion.chunker import TextChunker
from src.services.ingestion.ingestion_service import IngestionService
from src.services.ingestion.metadata_extractor import MetadataExtractor
from tests.conftest import MockEmbeddingProvider, MockVectorStore

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_ingestion_service(
    mock_embedding: MockEmbeddingProvider,
    mock_store: MockVectorStore,
    article_scraper: Any | None = None,
) -> IngestionService:
    """Construct an IngestionService wired to mock providers."""
    # Use a mock LLM that returns empty tags (avoid real LLM calls)
    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value='{"entities": [], "places": [], "genres": []}')

    chunker = TextChunker(chunk_size=200, overlap=40)
    metadata_extractor = MetadataExtractor(llm=mock_llm)

    return IngestionService(
        chunker=chunker,
        metadata_extractor=metadata_extractor,
        embedding_provider=mock_embedding,
        vector_store=mock_store,
        article_scraper=article_scraper,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBookIngestionE2E:
    """Create temp text file -> ingest via IngestionService -> verify chunks in store."""

    @pytest.mark.asyncio
    async def test_book_ingestion_e2e(
        self,
        tmp_path,
        mock_embedding_provider: MockEmbeddingProvider,
        mock_vector_store: MockVectorStore,
    ) -> None:
        # Write a test book file
        book_text = (
            "Chapter 1: The Birth of Techno\n\n"
            "Detroit techno emerged in the mid-1980s, pioneered by the "
            "Belleville Three. Juan Atkins created the template with Cybotron "
            "and Model 500.\n\n"
            "Derrick May's Strings of Life became an anthem that transcended "
            "the genre. Its piano-driven euphoria captured the spirit of a "
            "new musical movement.\n\n"
            "Kevin Saunderson brought a funkier edge through Inner City, "
            "scoring mainstream hits with Big Fun and Good Life.\n\n"
            "Chapter 2: The UK Explosion\n\n"
            "Acid house arrived in the UK via Ibiza. DJs returning from the "
            "island brought with them a new sound and a new culture that would "
            "transform British nightlife forever."
        )
        book_file = tmp_path / "test_book.txt"
        book_file.write_text(book_text)

        service = _build_ingestion_service(mock_embedding_provider, mock_vector_store)
        result = await service.ingest_book(
            file_path=str(book_file),
            title="History of Techno",
            author="Test Author",
            year=2020,
        )

        assert result.chunks_created > 0
        assert result.total_tokens > 0
        assert result.ingestion_time >= 0.0
        assert result.source_title == "History of Techno"

        # Verify chunks are actually in the mock store
        stats = await mock_vector_store.get_stats()
        assert stats.total_chunks == result.chunks_created
        assert stats.total_sources == 1

    @pytest.mark.asyncio
    async def test_empty_book_returns_empty_result(
        self,
        tmp_path,
        mock_embedding_provider: MockEmbeddingProvider,
        mock_vector_store: MockVectorStore,
    ) -> None:
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        service = _build_ingestion_service(mock_embedding_provider, mock_vector_store)
        result = await service.ingest_book(
            file_path=str(empty_file),
            title="Empty Book",
            author="Nobody",
            year=2020,
        )

        assert result.chunks_created == 0
        assert result.total_tokens == 0


class TestArticleIngestion:
    """Mock article scraper -> ingest URL -> verify chunks stored."""

    @pytest.mark.asyncio
    async def test_article_ingestion(
        self,
        mock_embedding_provider: MockEmbeddingProvider,
        mock_vector_store: MockVectorStore,
    ) -> None:
        mock_scraper = AsyncMock()
        mock_scraper.extract_content = AsyncMock(
            return_value=ArticleContent(
                title="Carl Cox Interview",
                text=(
                    "In this exclusive interview, Carl Cox discusses his "
                    "legendary residency at Space Ibiza and the evolution of "
                    "techno music over three decades. He reflects on the early "
                    "days of the UK rave scene and how it shaped his career."
                ),
                author="DJ Mag Staff",
                date=date(2020, 6, 15),
                url="https://djmag.com/carl-cox-interview",
            )
        )

        service = _build_ingestion_service(
            mock_embedding_provider, mock_vector_store, article_scraper=mock_scraper
        )
        result = await service.ingest_article("https://djmag.com/carl-cox-interview")

        assert result.chunks_created > 0
        assert result.total_tokens > 0

        stats = await mock_vector_store.get_stats()
        assert stats.total_chunks > 0

    @pytest.mark.asyncio
    async def test_article_ingestion_no_scraper(
        self,
        mock_embedding_provider: MockEmbeddingProvider,
        mock_vector_store: MockVectorStore,
    ) -> None:
        service = _build_ingestion_service(
            mock_embedding_provider, mock_vector_store, article_scraper=None
        )
        result = await service.ingest_article("https://example.com/article")

        assert result.chunks_created == 0


class TestAnalysisFeedback:
    """Create mock PipelineState with research results -> ingest -> verify chunks."""

    @pytest.mark.asyncio
    async def test_analysis_feedback(
        self,
        mock_embedding_provider: MockEmbeddingProvider,
        mock_vector_store: MockVectorStore,
    ) -> None:
        # Build a minimal PipelineState with research results
        artist = Artist(
            name="Carl Cox",
            aliases=["The Three Deck Wizard"],
            confidence=0.9,
        )
        research_result = ResearchResult(
            entity_type=EntityType.ARTIST,
            entity_name="Carl Cox",
            artist=artist,
            sources_consulted=["web_search_press", "music_databases"],
            confidence=0.85,
        )

        flier = FlierImage(
            filename="test.jpg",
            content_type="image/jpeg",
            file_size=1024,
            image_hash="abc123",
        )

        ocr = OCRResult(
            raw_text="Carl Cox",
            confidence=0.9,
            provider_used="mock",
            processing_time=0.1,
        )

        entities = ExtractedEntities(
            artists=[
                ExtractedEntity(text="Carl Cox", entity_type=EntityType.ARTIST, confidence=0.9)
            ],
            raw_ocr=ocr,
        )

        imap = InterconnectionMap(narrative="Carl Cox is a legendary DJ known for three-deck sets.")

        state = PipelineState(
            session_id="test-session-001",
            flier=flier,
            current_phase=PipelinePhase.OUTPUT,
            ocr_result=ocr,
            extracted_entities=entities,
            research_results=[research_result],
            interconnection_map=imap,
        )

        service = _build_ingestion_service(mock_embedding_provider, mock_vector_store)
        result = await service.ingest_analysis(state)

        assert result.chunks_created > 0
        assert "test-session-001" in result.source_title

        stats = await mock_vector_store.get_stats()
        assert stats.total_chunks > 0
        assert "analysis" in stats.sources_by_type


class TestCorpusStats:
    """Ingest 3 sources -> get_stats() -> verify counts."""

    @pytest.mark.asyncio
    async def test_corpus_stats(
        self,
        tmp_path,
        mock_embedding_provider: MockEmbeddingProvider,
        mock_vector_store: MockVectorStore,
    ) -> None:
        service = _build_ingestion_service(mock_embedding_provider, mock_vector_store)

        # Source 1: book
        book1 = tmp_path / "book1.txt"
        book1.write_text("Chapter 1\n\nDetroit techno emerged from the Belleville Three.")
        await service.ingest_book(str(book1), "Book One", "Author A", 1998)

        # Source 2: another book
        book2 = tmp_path / "book2.txt"
        book2.write_text("Chapter 1\n\nAcid house exploded in the UK Summer of Love 1988.")
        await service.ingest_book(str(book2), "Book Two", "Author B", 2005)

        # Source 3: article via directory ingestion
        article_dir = tmp_path / "articles"
        article_dir.mkdir()
        (article_dir / "article1.txt").write_text(
            "Carl Cox discusses his legendary career in this interview."
        )
        await service.ingest_directory(str(article_dir), "article")

        stats = await service.get_corpus_stats()
        assert stats.total_chunks >= 3
        assert stats.total_sources >= 3
        assert len(stats.sources_by_type) >= 1

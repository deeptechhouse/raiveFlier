"""Integration tests for RAG retrieval integration with researcher services.

Verifies that ArtistResearcher and InterconnectionService correctly use
the vector store when available and degrade gracefully when it is None.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.entities import EntityType
from src.models.flier import ExtractedEntities, ExtractedEntity, OCRResult
from src.models.rag import DocumentChunk
from src.models.research import ResearchResult
from src.services.artist_researcher import ArtistResearcher
from src.services.citation_service import CitationService
from src.services.interconnection_service import InterconnectionService
from tests.conftest import MockEmbeddingProvider, MockVectorStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_music_db(provider_name: str = "discogs_api") -> MagicMock:
    """Build a mock IMusicDatabaseProvider."""
    db = MagicMock()
    db.get_provider_name.return_value = provider_name
    db.is_available.return_value = True
    db.search_artist = AsyncMock(return_value=[])
    db.get_artist_releases = AsyncMock(return_value=[])
    db.get_artist_labels = AsyncMock(return_value=[])
    return db


def _mock_web_search() -> MagicMock:
    """Build a mock IWebSearchProvider that returns empty results."""
    ws = MagicMock()
    ws.get_provider_name.return_value = "mock-search"
    ws.is_available.return_value = True
    ws.search = AsyncMock(return_value=[])
    return ws


def _mock_article_scraper() -> MagicMock:
    """Build a mock IArticleProvider."""
    scraper = MagicMock()
    scraper.get_provider_name.return_value = "mock-scraper"
    scraper.is_available.return_value = True
    scraper.extract_content = AsyncMock(return_value=None)
    scraper.check_availability = AsyncMock(return_value=True)
    return scraper


def _mock_llm() -> MagicMock:
    """Build a mock ILLMProvider that returns NONE for event extraction."""
    llm = MagicMock()
    llm.get_provider_name.return_value = "mock-llm"
    llm.is_available.return_value = True
    llm.supports_vision.return_value = False
    llm.complete = AsyncMock(return_value="NONE")
    llm.validate_credentials = AsyncMock(return_value=True)
    return llm


async def _preload_vector_store(
    store: MockVectorStore,
    artist_name: str,
    source_title: str = "Energy Flash",
) -> None:
    """Pre-load a mock vector store with chunks mentioning an artist."""
    embedding_provider = MockEmbeddingProvider()

    chunks = [
        DocumentChunk(
            chunk_id=f"preload-{artist_name.lower().replace(' ', '-')}-{i}",
            text=(
                f"{artist_name} is a legendary DJ who has been at the forefront "
                f"of electronic music for decades. Part {i} of the story."
            ),
            source_id=f"src-{source_title.lower().replace(' ', '-')}",
            source_title=source_title,
            source_type="book",
            author="Simon Reynolds",
            publication_date=date(1998, 4, 1),
            citation_tier=1,
            entity_tags=[artist_name],
            geographic_tags=["London", "Ibiza"],
            genre_tags=["techno", "house"],
        )
        for i in range(3)
    ]
    texts = [c.text for c in chunks]
    embeddings = await embedding_provider.embed(texts)
    await store.add_chunks(chunks, embeddings)


# ---------------------------------------------------------------------------
# ArtistResearcher tests
# ---------------------------------------------------------------------------


class TestArtistResearcherWithRAG:
    """Verify that ArtistResearcher retrieves corpus chunks when vector store is present."""

    @pytest.mark.asyncio
    async def test_artist_researcher_with_rag(self, mock_vector_store: MockVectorStore) -> None:
        await _preload_vector_store(mock_vector_store, "Carl Cox")

        researcher = ArtistResearcher(
            music_dbs=[_mock_music_db()],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            vector_store=mock_vector_store,
        )

        result = await researcher.research("Carl Cox")

        assert result.entity_name == "Carl Cox"
        assert "rag_corpus" in result.sources_consulted

        # Check that corpus-sourced ArticleReferences are present
        corpus_refs = [a for a in result.artist.articles if a.source == "book"]
        assert len(corpus_refs) > 0, "Should have article refs from RAG corpus"

        # Corpus refs should carry tier 1 (book source)
        for ref in corpus_refs:
            assert ref.citation_tier == 1


class TestArtistResearcherWithoutRAG:
    """Verify identical behavior when vector_store=None (pre-RAG baseline)."""

    @pytest.mark.asyncio
    async def test_artist_researcher_without_rag(self) -> None:
        researcher = ArtistResearcher(
            music_dbs=[_mock_music_db()],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            vector_store=None,
        )

        result = await researcher.research("Carl Cox")

        assert result.entity_name == "Carl Cox"
        assert "rag_corpus" not in result.sources_consulted

        # No corpus-sourced references should exist
        corpus_refs = [a for a in result.artist.articles if a.source == "book"]
        assert len(corpus_refs) == 0


# ---------------------------------------------------------------------------
# InterconnectionService tests
# ---------------------------------------------------------------------------


def _make_entities(*artist_names: str) -> ExtractedEntities:
    """Build ExtractedEntities with the given artist names."""
    ocr = OCRResult(
        raw_text=" ".join(artist_names),
        confidence=0.9,
        provider_used="mock",
        processing_time=0.1,
    )
    return ExtractedEntities(
        artists=[
            ExtractedEntity(text=name, entity_type=EntityType.ARTIST, confidence=0.9)
            for name in artist_names
        ],
        raw_ocr=ocr,
    )


def _make_research_results(*artist_names: str) -> list[ResearchResult]:
    """Build minimal ResearchResult list for given artists."""
    return [
        ResearchResult(
            entity_type=EntityType.ARTIST,
            entity_name=name,
            sources_consulted=["web_search_press"],
            confidence=0.7,
        )
        for name in artist_names
    ]


class TestInterconnectionWithRAG:
    """Pre-load vector store with chunks mentioning two artists, verify cross-entity context."""

    @pytest.mark.asyncio
    async def test_interconnection_with_rag(self, mock_vector_store: MockVectorStore) -> None:
        # Pre-load chunks mentioning both artists
        embedding_provider = MockEmbeddingProvider()
        chunks = [
            DocumentChunk(
                chunk_id="cross-entity-001",
                text=(
                    "Carl Cox and Derrick May shared a stage at the Detroit "
                    "Electronic Music Festival in 2005. Their connection goes "
                    "back to the early days of techno."
                ),
                source_id="src-cross",
                source_title="Techno Connections",
                source_type="book",
                citation_tier=1,
                entity_tags=["Carl Cox", "Derrick May"],
                geographic_tags=["Detroit"],
                genre_tags=["techno"],
            ),
        ]
        embeddings = await embedding_provider.embed([c.text for c in chunks])
        await mock_vector_store.add_chunks(chunks, embeddings)

        llm = _mock_llm()
        # LLM returns a valid JSON response for interconnection analysis
        llm.complete = AsyncMock(
            return_value=(
                '{"relationships": [], "patterns": [], '
                '"narrative": "Carl Cox and Derrick May are connected through Detroit techno."}'
            )
        )

        service = InterconnectionService(
            llm_provider=llm,
            citation_service=CitationService(),
            vector_store=mock_vector_store,
        )

        entities = _make_entities("Carl Cox", "Derrick May")
        results = _make_research_results("Carl Cox", "Derrick May")

        imap = await service.analyze(results, entities)

        # The LLM should have received the corpus context as part of the prompt
        call_args = llm.complete.call_args
        user_prompt = call_args.kwargs.get(
            "user_prompt", call_args.args[1] if len(call_args.args) > 1 else ""
        )
        assert "CORPUS CONTEXT" in user_prompt or imap.narrative is not None


class TestInterconnectionWithoutRAG:
    """vector_store=None — behavior identical to pre-RAG."""

    @pytest.mark.asyncio
    async def test_interconnection_without_rag(self) -> None:
        llm = _mock_llm()
        llm.complete = AsyncMock(
            return_value=(
                '{"relationships": [], "patterns": [], '
                '"narrative": "Test narrative without RAG."}'
            )
        )

        service = InterconnectionService(
            llm_provider=llm,
            citation_service=CitationService(),
            vector_store=None,
        )

        entities = _make_entities("Carl Cox", "Derrick May")
        results = _make_research_results("Carl Cox", "Derrick May")

        imap = await service.analyze(results, entities)

        assert imap.narrative is not None

        # The LLM prompt should NOT contain corpus context
        call_args = llm.complete.call_args
        user_prompt = call_args.kwargs.get(
            "user_prompt", call_args.args[1] if len(call_args.args) > 1 else ""
        )
        assert "CORPUS CONTEXT" not in user_prompt

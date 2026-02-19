"""Backward-compatibility tests proving RAG does not break existing functionality.

CRITICAL: These tests verify that all pre-RAG behavior remains intact when
RAG is disabled or when vector_store=None is passed to services.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.models.entities import EntityType
from src.models.flier import ExtractedEntities, ExtractedEntity, OCRResult
from src.services.artist_researcher import ArtistResearcher
from src.services.citation_service import CitationService
from src.services.interconnection_service import InterconnectionService
from src.services.promoter_researcher import PromoterResearcher
from src.services.venue_researcher import VenueResearcher

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llm() -> MagicMock:
    """Build a mock ILLMProvider."""
    llm = MagicMock()
    llm.get_provider_name.return_value = "mock-llm"
    llm.is_available.return_value = True
    llm.supports_vision.return_value = False
    llm.complete = AsyncMock(return_value="NONE")
    llm.validate_credentials = AsyncMock(return_value=True)
    return llm


def _mock_web_search() -> MagicMock:
    """Build a mock IWebSearchProvider."""
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


def _mock_music_db(provider_name: str = "mock-discogs") -> MagicMock:
    """Build a mock IMusicDatabaseProvider."""
    db = MagicMock()
    db.get_provider_name.return_value = provider_name
    db.is_available.return_value = True
    db.search_artist = AsyncMock(return_value=[])
    db.get_artist_releases = AsyncMock(return_value=[])
    db.get_artist_labels = AsyncMock(return_value=[])
    return db


# ---------------------------------------------------------------------------
# Test: All researchers accept None vector_store
# ---------------------------------------------------------------------------


class TestAllResearchersAcceptNoneVectorStore:
    """Instantiate every researcher with vector_store=None, call research(), no errors."""

    @pytest.mark.asyncio
    async def test_artist_researcher_none_vector_store(self) -> None:
        researcher = ArtistResearcher(
            music_dbs=[_mock_music_db()],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            vector_store=None,
        )
        result = await researcher.research("Test Artist")
        assert result.entity_name == "Test Artist"
        assert "rag_corpus" not in result.sources_consulted

    @pytest.mark.asyncio
    async def test_venue_researcher_none_vector_store(self) -> None:
        researcher = VenueResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            vector_store=None,
        )
        result = await researcher.research("Test Venue")
        assert result.entity_name == "Test Venue"

    @pytest.mark.asyncio
    async def test_promoter_researcher_none_vector_store(self) -> None:
        researcher = PromoterResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            vector_store=None,
        )
        result = await researcher.research("Test Promoter")
        assert result.entity_name == "Test Promoter"

    @pytest.mark.asyncio
    async def test_interconnection_none_vector_store(self) -> None:
        llm = _mock_llm()
        llm.complete = AsyncMock(
            return_value=(
                '{"relationships": [], "patterns": [], ' '"narrative": "Test narrative."}'
            )
        )

        service = InterconnectionService(
            llm_provider=llm,
            citation_service=CitationService(),
            vector_store=None,
        )

        ocr = OCRResult(
            raw_text="Test",
            confidence=0.9,
            provider_used="mock",
            processing_time=0.1,
        )
        entities = ExtractedEntities(
            artists=[
                ExtractedEntity(
                    text="Test Artist",
                    entity_type=EntityType.ARTIST,
                    confidence=0.9,
                )
            ],
            raw_ocr=ocr,
        )

        from src.models.research import ResearchResult

        results = [
            ResearchResult(
                entity_type=EntityType.ARTIST,
                entity_name="Test Artist",
                sources_consulted=["web_search_press"],
                confidence=0.7,
            )
        ]

        imap = await service.analyze(results, entities)
        assert imap is not None
        assert imap.narrative == "Test narrative."


# ---------------------------------------------------------------------------
# Test: Full pipeline with RAG disabled
# ---------------------------------------------------------------------------


class TestFullPipelineRAGDisabled:
    """Run the pipeline structure with RAG_ENABLED=false and verify output."""

    @pytest.mark.asyncio
    async def test_full_pipeline_rag_disabled(self) -> None:
        """Verify pipeline produces valid output with no RAG components."""
        llm = _mock_llm()
        llm.complete = AsyncMock(
            return_value=(
                '{"relationships": [], "patterns": [], '
                '"narrative": "A test narrative about the rave scene."}'
            )
        )

        # Build researcher with no vector store
        researcher = ArtistResearcher(
            music_dbs=[_mock_music_db()],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=llm,
            vector_store=None,
        )

        # Research should complete without errors
        result = await researcher.research("Carl Cox")
        assert result.entity_name == "Carl Cox"
        assert result.entity_type == EntityType.ARTIST

        # Verify structure matches expected pre-RAG output shape
        assert hasattr(result, "artist")
        assert hasattr(result, "sources_consulted")
        assert hasattr(result, "confidence")
        assert hasattr(result, "warnings")
        assert "rag_corpus" not in result.sources_consulted

        # InterconnectionService without RAG
        service = InterconnectionService(
            llm_provider=llm,
            citation_service=CitationService(),
            vector_store=None,
        )

        ocr = OCRResult(
            raw_text="Carl Cox",
            confidence=0.9,
            provider_used="mock",
            processing_time=0.1,
        )
        entities = ExtractedEntities(
            artists=[
                ExtractedEntity(
                    text="Carl Cox",
                    entity_type=EntityType.ARTIST,
                    confidence=0.9,
                )
            ],
            raw_ocr=ocr,
        )

        from src.models.research import ResearchResult

        imap = await service.analyze(
            [
                ResearchResult(
                    entity_type=EntityType.ARTIST,
                    entity_name="Carl Cox",
                    sources_consulted=["web_search_press"],
                    confidence=0.7,
                )
            ],
            entities,
        )
        assert imap.narrative is not None
        assert hasattr(imap, "edges")
        assert hasattr(imap, "nodes")
        assert hasattr(imap, "patterns")
        assert hasattr(imap, "citations")


# ---------------------------------------------------------------------------
# Test: App startup with and without RAG
# ---------------------------------------------------------------------------


class TestMainAppStartup:
    """Verify app startup succeeds with both RAG enabled and disabled."""

    def test_main_app_starts_without_rag(self) -> None:
        """Import app with RAG_ENABLED=false and verify startup + 404 on corpus stats."""
        with patch.dict(
            "os.environ",
            {"RAG_ENABLED": "false", "APP_ENV": "test"},
            clear=False,
        ):
            # Re-import to pick up the test environment
            from src.config.settings import Settings

            test_settings = Settings(rag_enabled=False)
            assert test_settings.rag_enabled is False

            # Build the app and test the corpus stats endpoint
            from src.main import create_app

            app = create_app()
            client = TestClient(app)

            # corpus/stats should return 404 when RAG is disabled
            response = client.get("/api/v1/corpus/stats")
            assert response.status_code == 404

    def test_main_app_starts_with_rag(self) -> None:
        """Set RAG_ENABLED=true with mock providers and verify startup + 200 on corpus stats."""
        from src.main import create_app

        app = create_app()

        # Manually inject mock RAG components onto app state
        from src.services.ingestion.chunker import TextChunker
        from src.services.ingestion.ingestion_service import IngestionService
        from src.services.ingestion.metadata_extractor import MetadataExtractor
        from tests.conftest import MockEmbeddingProvider, MockVectorStore

        mock_embedding = MockEmbeddingProvider()
        mock_store = MockVectorStore()
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value='{"entities": [], "places": [], "genres": []}')

        chunker = TextChunker()
        metadata_extractor = MetadataExtractor(llm=mock_llm)
        ingestion_service = IngestionService(
            chunker=chunker,
            metadata_extractor=metadata_extractor,
            embedding_provider=mock_embedding,
            vector_store=mock_store,
        )

        # Inject RAG state
        app.state.rag_enabled = True
        app.state.ingestion_service = ingestion_service

        client = TestClient(app)

        # corpus/stats should now return 200
        response = client.get("/api/v1/corpus/stats")
        assert response.status_code == 200

        data = response.json()
        assert "total_chunks" in data
        assert "total_sources" in data
        assert "sources_by_type" in data
        assert data["total_chunks"] == 0  # empty store

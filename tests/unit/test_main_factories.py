"""Unit tests for factory functions in src/main.py.

Tests the LLM provider selection, embedding provider selection,
build_pipeline assembly, create_app factory, and run_pipeline
orchestration — all with mocked external dependencies so no
real network calls or API keys are required.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI

from src.config.settings import Settings


# ======================================================================
# Shared helpers
# ======================================================================


def _settings(**overrides) -> Settings:
    """Build a Settings instance with safe defaults and optional overrides.

    All API keys default to empty strings so the Ollama/None fallback
    is exercised unless explicitly overridden.
    """
    defaults = {
        "openai_api_key": "",
        "openai_base_url": "",
        "openai_text_model": "",
        "openai_vision_model": "",
        "openai_embedding_model": "",
        "anthropic_api_key": "",
        "ollama_base_url": "http://localhost:11434",
        "discogs_consumer_key": "",
        "discogs_consumer_secret": "",
        "musicbrainz_app_name": "raiveFlier",
        "musicbrainz_app_version": "0.1.0",
        "musicbrainz_contact": "",
        "rag_enabled": False,
        "app_env": "test",
    }
    defaults.update(overrides)
    return Settings(**defaults)


# ======================================================================
# _build_llm_provider
# ======================================================================


class TestBuildLLMProvider:
    """Tests for the _build_llm_provider factory — provider priority order."""

    def test_openai_priority(self) -> None:
        """When openai_api_key is set, return OpenAILLMProvider
        regardless of whether anthropic_api_key is also set."""
        from src.main import _build_llm_provider
        from src.providers.llm.openai_provider import OpenAILLMProvider

        s = _settings(anthropic_api_key="test-anthropic-key", openai_api_key="sk-also-set")
        result = _build_llm_provider(s)
        assert isinstance(result, OpenAILLMProvider)

    def test_anthropic_fallback(self) -> None:
        """When only anthropic_api_key is set (no openai), return AnthropicLLMProvider."""
        from src.main import _build_llm_provider
        from src.providers.llm.anthropic_provider import AnthropicLLMProvider

        s = _settings(anthropic_api_key="test-anthropic-key")
        result = _build_llm_provider(s)
        assert isinstance(result, AnthropicLLMProvider)

    def test_ollama_default(self) -> None:
        """When no API keys are set, return OllamaLLMProvider as the default."""
        from src.main import _build_llm_provider
        from src.providers.llm.ollama_provider import OllamaLLMProvider

        s = _settings()
        result = _build_llm_provider(s)
        assert isinstance(result, OllamaLLMProvider)


# ======================================================================
# _build_embedding_provider
# ======================================================================


class TestBuildEmbeddingProvider:
    """Tests for the _build_embedding_provider factory — lazy imports & availability."""

    @pytest.mark.asyncio()
    async def test_openai_embedding_when_available(self) -> None:
        """When openai_api_key is set and the provider reports available,
        return the OpenAIEmbeddingProvider instance."""
        from src.main import _build_embedding_provider

        s = _settings(openai_api_key="sk-test-key")

        mock_instance = MagicMock()
        mock_instance.is_available.return_value = True
        mock_instance.embed_single = AsyncMock(return_value=[0.1] * 1024)

        with patch(
            "src.providers.embedding.openai_embedding_provider.OpenAIEmbeddingProvider",
            return_value=mock_instance,
        ):
            result = await _build_embedding_provider(s)

        assert result is mock_instance
        mock_instance.is_available.assert_called_once()

    @pytest.mark.asyncio()
    async def test_fastembed_fallback_when_openai_unavailable(self) -> None:
        """When openai_api_key is set but the OpenAI embedding provider
        reports unavailable, fall through to FastEmbedEmbeddingProvider."""
        from src.main import _build_embedding_provider

        s = _settings(openai_api_key="sk-test-key")

        mock_openai = MagicMock()
        mock_openai.is_available.return_value = False

        mock_fe = MagicMock()
        mock_fe.is_available.return_value = True

        with patch(
            "src.providers.embedding.openai_embedding_provider.OpenAIEmbeddingProvider",
            return_value=mock_openai,
        ), patch(
            "src.providers.embedding.fastembed_embedding_provider.FastEmbedEmbeddingProvider",
            return_value=mock_fe,
        ):
            result = await _build_embedding_provider(s)

        assert result is mock_fe
        mock_fe.is_available.assert_called_once()

    @pytest.mark.asyncio()
    async def test_fastembed_fallback_when_openai_api_error(self) -> None:
        """When openai_api_key is set but embed_single raises (e.g. 402),
        fall back to FastEmbedEmbeddingProvider."""
        from src.main import _build_embedding_provider

        s = _settings(openai_api_key="sk-test-key")

        mock_openai = MagicMock()
        mock_openai.is_available.return_value = True
        mock_openai.embed_single = AsyncMock(side_effect=Exception("402 Payment Required"))

        mock_fe = MagicMock()
        mock_fe.is_available.return_value = True

        with patch(
            "src.providers.embedding.openai_embedding_provider.OpenAIEmbeddingProvider",
            return_value=mock_openai,
        ), patch(
            "src.providers.embedding.fastembed_embedding_provider.FastEmbedEmbeddingProvider",
            return_value=mock_fe,
        ):
            result = await _build_embedding_provider(s)

        assert result is mock_fe

    @pytest.mark.asyncio()
    async def test_nomic_fallback_when_no_local_providers(self) -> None:
        """When no openai_api_key and both fastembed/sentence-transformers
        are unavailable, fall back to NomicEmbeddingProvider."""
        from src.main import _build_embedding_provider

        s = _settings()

        mock_fe = MagicMock()
        mock_fe.is_available.return_value = False

        mock_st = MagicMock()
        mock_st.is_available.return_value = False

        mock_nomic = MagicMock()
        mock_nomic.is_available.return_value = True

        with patch(
            "src.providers.embedding.fastembed_embedding_provider.FastEmbedEmbeddingProvider",
            return_value=mock_fe,
        ), patch(
            "src.providers.embedding.sentence_transformer_embedding_provider.SentenceTransformerEmbeddingProvider",
            return_value=mock_st,
        ), patch(
            "src.providers.embedding.nomic_embedding_provider.NomicEmbeddingProvider",
            return_value=mock_nomic,
        ):
            result = await _build_embedding_provider(s)

        assert result is mock_nomic

    @pytest.mark.asyncio()
    async def test_none_when_nothing_available(self) -> None:
        """When no embedding provider reports available, return None."""
        from src.main import _build_embedding_provider

        s = _settings()

        mock_fe = MagicMock()
        mock_fe.is_available.return_value = False

        mock_st = MagicMock()
        mock_st.is_available.return_value = False

        mock_nomic = MagicMock()
        mock_nomic.is_available.return_value = False

        with patch(
            "src.providers.embedding.fastembed_embedding_provider.FastEmbedEmbeddingProvider",
            return_value=mock_fe,
        ), patch(
            "src.providers.embedding.sentence_transformer_embedding_provider.SentenceTransformerEmbeddingProvider",
            return_value=mock_st,
        ), patch(
            "src.providers.embedding.nomic_embedding_provider.NomicEmbeddingProvider",
            return_value=mock_nomic,
        ):
            result = await _build_embedding_provider(s)

        assert result is None


# ======================================================================
# build_pipeline
# ======================================================================


class TestBuildPipeline:
    """Tests for the build_pipeline helper used by CLI / scripting usage."""

    def test_returns_expected_keys(self) -> None:
        """build_pipeline returns a dict containing all expected service keys."""
        from src.main import build_pipeline

        mock_llm = MagicMock()
        mock_llm.supports_vision.return_value = False
        mock_llm.get_provider_name.return_value = "mock"

        with patch("src.main._build_llm_provider", return_value=mock_llm), \
             patch("src.main._EASYOCR_AVAILABLE", False):
            result = build_pipeline(_settings())

        expected_keys = {
            "ocr_service",
            "entity_extractor",
            "artist_researcher",
            "venue_researcher",
            "promoter_researcher",
            "date_context_researcher",
            "event_name_researcher",
            "citation_service",
            "interconnection_service",
            "settings",
        }
        assert set(result.keys()) == expected_keys

    def test_with_custom_settings(self) -> None:
        """build_pipeline uses custom_settings when provided instead of module-level settings."""
        from src.main import build_pipeline

        custom = _settings(ollama_base_url="http://custom:11434")

        mock_llm = MagicMock()
        mock_llm.supports_vision.return_value = False
        mock_llm.get_provider_name.return_value = "mock"

        with patch("src.main._build_llm_provider", return_value=mock_llm) as mock_build, \
             patch("src.main._EASYOCR_AVAILABLE", False):
            result = build_pipeline(custom)

        # The custom settings should have been forwarded to _build_llm_provider
        mock_build.assert_called_once_with(custom)
        # And the returned dict stores those settings
        assert result["settings"] is custom

    def test_settings_fallback_to_module_level(self) -> None:
        """build_pipeline uses module-level settings when custom_settings is None."""
        from src.main import build_pipeline

        mock_llm = MagicMock()
        mock_llm.supports_vision.return_value = False
        mock_llm.get_provider_name.return_value = "mock"

        with patch("src.main._build_llm_provider", return_value=mock_llm) as mock_build, \
             patch("src.main._EASYOCR_AVAILABLE", False):
            result = build_pipeline(None)

        # Should have been called with the module-level settings object
        from src.main import settings as module_settings
        mock_build.assert_called_once_with(module_settings)
        assert result["settings"] is module_settings

    def test_ocr_service_in_result(self) -> None:
        """The returned ocr_service is an OCRService instance."""
        from src.main import build_pipeline
        from src.services.ocr_service import OCRService

        mock_llm = MagicMock()
        mock_llm.supports_vision.return_value = False
        mock_llm.get_provider_name.return_value = "mock"

        with patch("src.main._build_llm_provider", return_value=mock_llm), \
             patch("src.main._EASYOCR_AVAILABLE", False):
            result = build_pipeline(_settings())

        assert isinstance(result["ocr_service"], OCRService)

    def test_entity_extractor_in_result(self) -> None:
        """The returned entity_extractor is an EntityExtractor instance."""
        from src.main import build_pipeline
        from src.services.entity_extractor import EntityExtractor

        mock_llm = MagicMock()
        mock_llm.supports_vision.return_value = False
        mock_llm.get_provider_name.return_value = "mock"

        with patch("src.main._build_llm_provider", return_value=mock_llm), \
             patch("src.main._EASYOCR_AVAILABLE", False):
            result = build_pipeline(_settings())

        assert isinstance(result["entity_extractor"], EntityExtractor)

    def test_vision_llm_adds_vision_ocr_provider(self) -> None:
        """When the LLM provider supports vision, an LLMVisionOCRProvider is included."""
        from src.main import build_pipeline

        mock_llm = MagicMock()
        mock_llm.supports_vision.return_value = True
        mock_llm.get_provider_name.return_value = "mock"

        with patch("src.main._build_llm_provider", return_value=mock_llm), \
             patch("src.main._EASYOCR_AVAILABLE", False):
            result = build_pipeline(_settings())

        # The OCR service should have at least 2 providers: LLMVision + Tesseract
        ocr_svc = result["ocr_service"]
        provider_types = [type(p).__name__ for p in ocr_svc._providers]
        assert "LLMVisionOCRProvider" in provider_types
        assert "TesseractOCRProvider" in provider_types


# ======================================================================
# create_app
# ======================================================================


class TestCreateApp:
    """Tests for the create_app FastAPI application factory."""

    def test_returns_fastapi_instance(self) -> None:
        """create_app() returns a FastAPI application object."""
        from src.main import create_app

        application = create_app()
        assert isinstance(application, FastAPI)

    def test_app_has_correct_title(self) -> None:
        """The app title is 'raiveFlier API'."""
        from src.main import create_app

        application = create_app()
        assert application.title == "raiveFlier API"

    def test_app_has_correct_version(self) -> None:
        """The app version is '0.1.0'."""
        from src.main import create_app

        application = create_app()
        assert application.version == "0.1.0"

    def test_app_has_api_routes(self) -> None:
        """The app includes routes from the API router."""
        from src.main import create_app

        application = create_app()
        # Collect all route paths defined on the app
        paths = [route.path for route in application.routes]
        # The API router should contribute at least /upload and /health
        assert any("/upload" in p for p in paths), f"Expected /upload route in {paths}"
        assert any("/health" in p for p in paths), f"Expected /health route in {paths}"

    def test_app_has_websocket_route(self) -> None:
        """The app includes the WebSocket progress endpoint."""
        from src.main import create_app

        application = create_app()
        paths = [route.path for route in application.routes]
        assert any("ws" in p and "progress" in p for p in paths), (
            f"Expected WebSocket progress route in {paths}"
        )

    def test_module_level_app_is_fastapi(self) -> None:
        """The module-level `app` object is the result of create_app()."""
        from src.main import app

        assert isinstance(app, FastAPI)
        assert app.title == "raiveFlier API"


# ======================================================================
# run_pipeline
# ======================================================================


class TestRunPipeline:
    """Tests for the async run_pipeline function — full pipeline orchestration."""

    @pytest.mark.asyncio
    async def test_run_pipeline_returns_completed_state(self) -> None:
        """run_pipeline returns a PipelineState with completed_at set and OUTPUT phase."""
        from src.models.entities import EntityType
        from src.models.flier import ExtractedEntities, ExtractedEntity, FlierImage, OCRResult
        from src.models.pipeline import PipelinePhase, PipelineState

        # -- Build a minimal FlierImage --
        flier = FlierImage(
            id="test-run",
            filename="test.jpg",
            content_type="image/jpeg",
            file_size=1000,
            image_hash="abc123",
        )

        # -- Mock OCR result --
        mock_ocr_result = OCRResult(
            raw_text="CARL COX\nTRESOR BERLIN\n1997",
            confidence=0.9,
            provider_used="mock",
            processing_time=0.1,
        )

        # -- Mock extracted entities (at least one artist) --
        mock_entities = ExtractedEntities(
            artists=[
                ExtractedEntity(
                    text="Carl Cox",
                    entity_type=EntityType.ARTIST,
                    confidence=0.95,
                ),
            ],
            venue=ExtractedEntity(
                text="Tresor Berlin",
                entity_type=EntityType.VENUE,
                confidence=0.88,
            ),
            date=ExtractedEntity(
                text="1997",
                entity_type=EntityType.DATE,
                confidence=0.80,
            ),
            promoter=None,
            raw_ocr=mock_ocr_result,
        )

        # -- Mock research result --
        from src.models.research import ResearchResult

        mock_research = ResearchResult(
            entity_type=EntityType.ARTIST,
            entity_name="Carl Cox",
            confidence=0.9,
        )

        # -- Mock interconnection map --
        from src.models.analysis import InterconnectionMap

        mock_interconnection = InterconnectionMap(
            nodes=[],
            edges=[],
            patterns=[],
            narrative="Mock narrative",
            citations=[],
        )

        # -- Build mock services dict --
        mock_ocr_svc = AsyncMock()
        mock_ocr_svc.extract_text = AsyncMock(return_value=mock_ocr_result)

        mock_entity_ext = AsyncMock()
        mock_entity_ext.extract = AsyncMock(return_value=mock_entities)

        mock_artist_res = AsyncMock()
        mock_artist_res.research = AsyncMock(return_value=mock_research)

        mock_venue_res = AsyncMock()
        mock_venue_res.research = AsyncMock(return_value=mock_research)

        mock_date_res = AsyncMock()
        mock_date_res.research = AsyncMock(return_value=mock_research)

        mock_promoter_res = AsyncMock()
        mock_promoter_res.research = AsyncMock(return_value=mock_research)

        mock_interconnection_svc = AsyncMock()
        mock_interconnection_svc.analyze = AsyncMock(return_value=mock_interconnection)

        mock_services = {
            "ocr_service": mock_ocr_svc,
            "entity_extractor": mock_entity_ext,
            "artist_researcher": mock_artist_res,
            "venue_researcher": mock_venue_res,
            "promoter_researcher": mock_promoter_res,
            "date_context_researcher": mock_date_res,
            "event_name_researcher": AsyncMock(),
            "citation_service": MagicMock(),
            "interconnection_service": mock_interconnection_svc,
            "settings": _settings(),
        }

        with patch("src.main.build_pipeline", return_value=mock_services):
            from src.main import run_pipeline

            state = await run_pipeline(flier)

        assert isinstance(state, PipelineState)
        assert state.current_phase == PipelinePhase.OUTPUT
        assert state.completed_at is not None
        assert state.progress_percent == 100.0

    @pytest.mark.asyncio
    async def test_run_pipeline_calls_services_in_order(self) -> None:
        """run_pipeline calls OCR, entity extraction, research, and
        interconnection in the expected sequence."""
        from src.models.entities import EntityType
        from src.models.flier import ExtractedEntities, ExtractedEntity, FlierImage, OCRResult

        flier = FlierImage(
            id="test-order",
            filename="test.jpg",
            content_type="image/jpeg",
            file_size=500,
            image_hash="def456",
        )

        mock_ocr_result = OCRResult(
            raw_text="DJ RUSH",
            confidence=0.85,
            provider_used="mock",
            processing_time=0.05,
        )

        mock_entities = ExtractedEntities(
            artists=[
                ExtractedEntity(
                    text="DJ Rush",
                    entity_type=EntityType.ARTIST,
                    confidence=0.90,
                ),
            ],
            venue=None,
            date=None,
            promoter=None,
            raw_ocr=mock_ocr_result,
        )

        from src.models.analysis import InterconnectionMap
        from src.models.research import ResearchResult

        mock_research = ResearchResult(
            entity_type=EntityType.ARTIST,
            entity_name="DJ Rush",
            confidence=0.88,
        )
        mock_interconnection = InterconnectionMap(
            nodes=[], edges=[], patterns=[], narrative="", citations=[]
        )

        call_order: list[str] = []

        async def track_ocr(*args, **kwargs):
            call_order.append("ocr")
            return mock_ocr_result

        async def track_extract(*args, **kwargs):
            call_order.append("entity_extraction")
            return mock_entities

        async def track_research(*args, **kwargs):
            call_order.append("research")
            return mock_research

        async def track_interconnection(*args, **kwargs):
            call_order.append("interconnection")
            return mock_interconnection

        mock_services = {
            "ocr_service": MagicMock(extract_text=track_ocr),
            "entity_extractor": MagicMock(extract=track_extract),
            "artist_researcher": MagicMock(research=track_research),
            "venue_researcher": AsyncMock(),
            "promoter_researcher": AsyncMock(),
            "date_context_researcher": AsyncMock(),
            "event_name_researcher": AsyncMock(),
            "citation_service": MagicMock(),
            "interconnection_service": MagicMock(analyze=track_interconnection),
            "settings": _settings(),
        }

        with patch("src.main.build_pipeline", return_value=mock_services):
            from src.main import run_pipeline

            await run_pipeline(flier)

        assert call_order == ["ocr", "entity_extraction", "research", "interconnection"]

    @pytest.mark.asyncio
    async def test_run_pipeline_session_id_is_set(self) -> None:
        """The returned PipelineState has a non-empty session_id."""
        from src.models.entities import EntityType
        from src.models.flier import ExtractedEntities, ExtractedEntity, FlierImage, OCRResult

        flier = FlierImage(
            id="test-sid",
            filename="test.jpg",
            content_type="image/jpeg",
            file_size=100,
            image_hash="ghi789",
        )

        mock_ocr = OCRResult(
            raw_text="", confidence=0.5, provider_used="mock", processing_time=0.01
        )
        mock_entities = ExtractedEntities(
            artists=[], venue=None, date=None, promoter=None, raw_ocr=mock_ocr
        )
        from src.models.analysis import InterconnectionMap

        mock_interconnection = InterconnectionMap(
            nodes=[], edges=[], patterns=[], narrative="", citations=[]
        )

        mock_services = {
            "ocr_service": MagicMock(extract_text=AsyncMock(return_value=mock_ocr)),
            "entity_extractor": MagicMock(extract=AsyncMock(return_value=mock_entities)),
            "artist_researcher": AsyncMock(),
            "venue_researcher": AsyncMock(),
            "promoter_researcher": AsyncMock(),
            "date_context_researcher": AsyncMock(),
            "event_name_researcher": AsyncMock(),
            "citation_service": MagicMock(),
            "interconnection_service": MagicMock(
                analyze=AsyncMock(return_value=mock_interconnection)
            ),
            "settings": _settings(),
        }

        with patch("src.main.build_pipeline", return_value=mock_services):
            from src.main import run_pipeline

            state = await run_pipeline(flier)

        assert state.session_id
        assert len(state.session_id) > 0


# ======================================================================
# _build_all
# ======================================================================


class TestBuildAll:
    """Tests for the _build_all DI assembly function.

    All external provider constructors are patched so the function can
    run without API keys, network access, or heavy library imports.
    """

    @staticmethod
    def _patch_all():
        """Return a dict of context-managers that mock every provider
        constructor called inside ``_build_all``.
        """
        mock_llm = MagicMock()
        mock_llm.supports_vision.return_value = False
        mock_llm.get_provider_name.return_value = "mock-llm"

        patches = {
            "llm": patch("src.main._build_llm_provider", return_value=mock_llm),
            "easyocr_flag": patch("src.main._EASYOCR_AVAILABLE", False),
            "tesseract": patch(
                "src.main.TesseractOCRProvider",
                return_value=MagicMock(),
            ),
            "duckduckgo": patch(
                "src.main.DuckDuckGoSearchProvider",
                return_value=MagicMock(),
            ),
            "web_scraper": patch(
                "src.main.WebScraperProvider",
                return_value=MagicMock(),
            ),
            "wayback": patch(
                "src.main.WaybackProvider",
                return_value=MagicMock(),
            ),
            "memory_cache": patch(
                "src.main.MemoryCacheProvider",
                return_value=MagicMock(),
            ),
            "feedback": patch(
                "src.main.SQLiteFeedbackProvider",
                return_value=MagicMock(),
            ),
            "httpx_client": patch(
                "src.main.httpx.AsyncClient",
                return_value=MagicMock(),
            ),
            "image_preprocessor": patch(
                "src.main.ImagePreprocessor",
                return_value=MagicMock(),
            ),
        }
        return patches, mock_llm

    @pytest.mark.asyncio()
    async def test_returns_dict_with_expected_keys(self) -> None:
        """_build_all returns a dict with all required component keys."""
        from src.main import _build_all

        patches, _ = self._patch_all()
        with (
            patches["llm"],
            patches["easyocr_flag"],
            patches["tesseract"],
            patches["duckduckgo"],
            patches["web_scraper"],
            patches["wayback"],
            patches["memory_cache"],
            patches["feedback"],
            patches["httpx_client"],
            patches["image_preprocessor"],
        ):
            result = await _build_all(_settings())

        expected_keys = {
            "http_client",
            "pipeline",
            "confirmation_gate",
            "progress_tracker",
            "session_states",
            "provider_registry",
            "provider_list",
            "primary_llm_name",
            "ingestion_service",
            "vector_store",
            "rag_enabled",
            "qa_service",
            "feedback_provider",
        }
        assert set(result.keys()) == expected_keys

    @pytest.mark.asyncio()
    async def test_rag_disabled_path(self) -> None:
        """When rag_enabled=False, vector_store and ingestion_service are None."""
        from src.main import _build_all

        patches, _ = self._patch_all()
        with (
            patches["llm"],
            patches["easyocr_flag"],
            patches["tesseract"],
            patches["duckduckgo"],
            patches["web_scraper"],
            patches["wayback"],
            patches["memory_cache"],
            patches["feedback"],
            patches["httpx_client"],
            patches["image_preprocessor"],
        ):
            result = await _build_all(_settings(rag_enabled=False))

        assert result["vector_store"] is None
        assert result["ingestion_service"] is None
        assert result["rag_enabled"] is False

    @pytest.mark.asyncio()
    async def test_rag_enabled_with_available_embedding(self) -> None:
        """When rag_enabled=True and embedding provider is available,
        vector_store and ingestion_service are populated."""
        from src.main import _build_all

        patches, mock_llm = self._patch_all()

        mock_embedding = MagicMock()
        mock_embedding.is_available.return_value = True
        mock_embedding.get_provider_name.return_value = "mock-embedding"

        mock_chromadb = MagicMock()
        mock_ingestion = MagicMock()

        async def _fake_build_embedding(_s):
            return mock_embedding

        with (
            patches["llm"],
            patches["easyocr_flag"],
            patches["tesseract"],
            patches["duckduckgo"],
            patches["web_scraper"],
            patches["wayback"],
            patches["memory_cache"],
            patches["feedback"],
            patches["httpx_client"],
            patches["image_preprocessor"],
            patch("src.main._build_embedding_provider", side_effect=_fake_build_embedding),
            patch(
                "src.providers.vector_store.chromadb_provider.ChromaDBProvider",
                return_value=mock_chromadb,
            ),
            patch(
                "src.services.ingestion.ingestion_service.IngestionService",
                return_value=mock_ingestion,
            ),
            patch("src.services.ingestion.chunker.TextChunker", return_value=MagicMock()),
            patch(
                "src.services.ingestion.metadata_extractor.MetadataExtractor",
                return_value=MagicMock(),
            ),
        ):
            result = await _build_all(_settings(rag_enabled=True))

        assert result["vector_store"] is mock_chromadb
        assert result["ingestion_service"] is mock_ingestion
        assert result["rag_enabled"] is True

    @pytest.mark.asyncio()
    async def test_rag_enabled_without_embedding_provider(self) -> None:
        """When rag_enabled=True but no embedding provider is available,
        vector_store stays None and rag_enabled is False."""
        from src.main import _build_all

        patches, _ = self._patch_all()

        async def _fake_build_embedding(_s):
            return None

        with (
            patches["llm"],
            patches["easyocr_flag"],
            patches["tesseract"],
            patches["duckduckgo"],
            patches["web_scraper"],
            patches["wayback"],
            patches["memory_cache"],
            patches["feedback"],
            patches["httpx_client"],
            patches["image_preprocessor"],
            patch("src.main._build_embedding_provider", side_effect=_fake_build_embedding),
        ):
            result = await _build_all(_settings(rag_enabled=True))

        assert result["vector_store"] is None
        assert result["ingestion_service"] is None
        assert result["rag_enabled"] is False

    @pytest.mark.asyncio()
    async def test_provider_registry_flags(self) -> None:
        """The provider_registry dict has correct boolean flags."""
        from src.main import _build_all

        patches, _ = self._patch_all()
        with (
            patches["llm"],
            patches["easyocr_flag"],
            patches["tesseract"],
            patches["duckduckgo"],
            patches["web_scraper"],
            patches["wayback"],
            patches["memory_cache"],
            patches["feedback"],
            patches["httpx_client"],
            patches["image_preprocessor"],
        ):
            result = await _build_all(_settings())

        registry = result["provider_registry"]
        assert registry["llm"] is True
        assert registry["ocr"] is True  # Tesseract always present
        assert registry["music_db"] is True
        assert registry["search"] is True
        assert registry["article"] is True
        assert registry["cache"] is True
        assert registry["rag"] is False  # RAG disabled by default

    @pytest.mark.asyncio()
    async def test_provider_list_contains_core_providers(self) -> None:
        """The provider_list contains at least Ollama, OCR, music DB, and search entries."""
        from src.main import _build_all

        patches, _ = self._patch_all()
        with (
            patches["llm"],
            patches["easyocr_flag"],
            patches["tesseract"],
            patches["duckduckgo"],
            patches["web_scraper"],
            patches["wayback"],
            patches["memory_cache"],
            patches["feedback"],
            patches["httpx_client"],
            patches["image_preprocessor"],
        ):
            result = await _build_all(_settings())

        provider_list = result["provider_list"]
        names = [p["name"] for p in provider_list]
        # Should include at least the LLM fallback and search providers
        assert "ollama" in names
        assert "DuckDuckGoSearchProvider" in names
        assert "WebScraperProvider" in names
        assert "WaybackProvider" in names

    @pytest.mark.asyncio()
    async def test_primary_llm_name_returned(self) -> None:
        """The result dict includes the primary LLM provider name string."""
        from src.main import _build_all

        patches, _ = self._patch_all()
        with (
            patches["llm"],
            patches["easyocr_flag"],
            patches["tesseract"],
            patches["duckduckgo"],
            patches["web_scraper"],
            patches["wayback"],
            patches["memory_cache"],
            patches["feedback"],
            patches["httpx_client"],
            patches["image_preprocessor"],
        ):
            result = await _build_all(_settings())

        assert result["primary_llm_name"] == "mock-llm"

    @pytest.mark.asyncio()
    async def test_session_states_initialized_empty(self) -> None:
        """session_states dict starts empty."""
        from src.main import _build_all

        patches, _ = self._patch_all()
        with (
            patches["llm"],
            patches["easyocr_flag"],
            patches["tesseract"],
            patches["duckduckgo"],
            patches["web_scraper"],
            patches["wayback"],
            patches["memory_cache"],
            patches["feedback"],
            patches["httpx_client"],
            patches["image_preprocessor"],
        ):
            result = await _build_all(_settings())

        assert result["session_states"] == {}

    @pytest.mark.asyncio()
    async def test_vision_llm_adds_vision_ocr(self) -> None:
        """When the LLM supports vision, LLMVisionOCRProvider is added to OCR stack."""
        from src.main import _build_all

        patches, mock_llm = self._patch_all()
        mock_llm.supports_vision.return_value = True

        with (
            patches["llm"],
            patches["easyocr_flag"],
            patches["tesseract"],
            patches["duckduckgo"],
            patches["web_scraper"],
            patches["wayback"],
            patches["memory_cache"],
            patches["feedback"],
            patches["httpx_client"],
            patches["image_preprocessor"],
        ):
            result = await _build_all(_settings())

        # The provider_list should include at least one 'ocr' type entry
        # that corresponds to LLMVisionOCRProvider
        ocr_entries = [p for p in result["provider_list"] if p["type"] == "ocr"]
        ocr_names = [p["name"] for p in ocr_entries]
        assert "LLMVisionOCRProvider" in ocr_names


# ======================================================================
# _auto_ingest_reference_corpus
# ======================================================================


class TestAutoIngestCorpus:
    """Tests for _auto_ingest_reference_corpus startup hook."""

    @pytest.mark.asyncio
    async def test_no_ingestion_service_returns_immediately(self) -> None:
        """When ingestion_service is not set on app.state, the function
        returns without error (RAG not enabled path)."""
        from src.main import _auto_ingest_reference_corpus

        app = FastAPI()
        # Do NOT set ingestion_service or vector_store on state
        await _auto_ingest_reference_corpus(app)
        # Should complete silently — no exception raised

    @pytest.mark.asyncio
    async def test_vector_store_has_data_skips_ingestion(self) -> None:
        """When the vector store already contains chunks, ingestion is skipped."""
        from src.main import _auto_ingest_reference_corpus
        from src.models.rag import CorpusStats

        app = FastAPI()
        mock_vs = AsyncMock()
        mock_vs.get_stats = AsyncMock(
            return_value=CorpusStats(total_chunks=100, total_sources=5)
        )
        mock_ingestion = AsyncMock()

        app.state.vector_store = mock_vs
        app.state.ingestion_service = mock_ingestion

        await _auto_ingest_reference_corpus(app)

        # get_stats was called to check, but ingest_directory was NOT called
        mock_vs.get_stats.assert_awaited_once()
        mock_ingestion.ingest_directory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_corpus_dir_not_found_skips(self) -> None:
        """When the reference corpus directory does not exist, ingestion is skipped."""
        from src.main import _auto_ingest_reference_corpus
        from src.models.rag import CorpusStats

        app = FastAPI()
        mock_vs = AsyncMock()
        mock_vs.get_stats = AsyncMock(
            return_value=CorpusStats(total_chunks=0, total_sources=0)
        )
        mock_ingestion = AsyncMock()

        app.state.vector_store = mock_vs
        app.state.ingestion_service = mock_ingestion

        # Point to a non-existent directory
        with patch(
            "src.main._REFERENCE_CORPUS_DIR",
            Path("/tmp/nonexistent_corpus_dir_99999"),
        ):
            await _auto_ingest_reference_corpus(app)

        mock_ingestion.ingest_directory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_success_path_ingests_directory(self, tmp_path: Path) -> None:
        """When vector store is empty and corpus dir exists, ingest_directory is called."""
        from src.main import _auto_ingest_reference_corpus
        from src.models.rag import CorpusStats, IngestionResult

        # Create a temporary corpus directory with a file
        corpus_dir = tmp_path / "reference_corpus"
        corpus_dir.mkdir()
        (corpus_dir / "test.txt").write_text("Berlin techno history")

        app = FastAPI()
        mock_vs = AsyncMock()
        mock_vs.get_stats = AsyncMock(
            return_value=CorpusStats(total_chunks=0, total_sources=0)
        )
        mock_ingestion = AsyncMock()
        mock_ingestion.ingest_directory = AsyncMock(
            return_value=[
                IngestionResult(
                    source_id="src-1",
                    source_title="test.txt",
                    chunks_created=3,
                    total_tokens=100,
                    ingestion_time=0.5,
                ),
            ]
        )

        app.state.vector_store = mock_vs
        app.state.ingestion_service = mock_ingestion

        with patch("src.main._REFERENCE_CORPUS_DIR", corpus_dir):
            await _auto_ingest_reference_corpus(app)

        mock_ingestion.ingest_directory.assert_awaited_once_with(
            str(corpus_dir), source_type="reference"
        )

    @pytest.mark.asyncio
    async def test_get_stats_exception_proceeds_to_ingest(self, tmp_path: Path) -> None:
        """When get_stats raises an exception, ingestion proceeds anyway."""
        from src.main import _auto_ingest_reference_corpus
        from src.models.rag import IngestionResult

        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        (corpus_dir / "data.txt").write_text("Some text")

        app = FastAPI()
        mock_vs = AsyncMock()
        mock_vs.get_stats = AsyncMock(side_effect=RuntimeError("DB error"))
        mock_ingestion = AsyncMock()
        mock_ingestion.ingest_directory = AsyncMock(
            return_value=[
                IngestionResult(
                    source_id="src-2",
                    source_title="data.txt",
                    chunks_created=1,
                    total_tokens=20,
                    ingestion_time=0.1,
                ),
            ]
        )

        app.state.vector_store = mock_vs
        app.state.ingestion_service = mock_ingestion

        with patch("src.main._REFERENCE_CORPUS_DIR", corpus_dir):
            await _auto_ingest_reference_corpus(app)

        mock_ingestion.ingest_directory.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ingestion_exception_is_caught(self, tmp_path: Path) -> None:
        """When ingest_directory raises an exception, it is caught and logged
        (no unhandled exception propagation)."""
        from src.main import _auto_ingest_reference_corpus
        from src.models.rag import CorpusStats

        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        (corpus_dir / "boom.txt").write_text("Will fail")

        app = FastAPI()
        mock_vs = AsyncMock()
        mock_vs.get_stats = AsyncMock(
            return_value=CorpusStats(total_chunks=0, total_sources=0)
        )
        mock_ingestion = AsyncMock()
        mock_ingestion.ingest_directory = AsyncMock(
            side_effect=RuntimeError("Embedding service down")
        )

        app.state.vector_store = mock_vs
        app.state.ingestion_service = mock_ingestion

        with patch("src.main._REFERENCE_CORPUS_DIR", corpus_dir):
            # Should NOT raise — error is caught internally
            await _auto_ingest_reference_corpus(app)

    @pytest.mark.asyncio
    async def test_no_vector_store_returns_immediately(self) -> None:
        """When vector_store is None (but ingestion_service is set),
        the function returns immediately."""
        from src.main import _auto_ingest_reference_corpus

        app = FastAPI()
        app.state.ingestion_service = AsyncMock()
        app.state.vector_store = None

        await _auto_ingest_reference_corpus(app)
        # No exception — clean exit


# ======================================================================
# _lifespan
# ======================================================================


class TestLifespan:
    """Tests for the _lifespan async context manager (startup/shutdown)."""

    @pytest.mark.asyncio
    async def test_lifespan_sets_state_and_cleans_up(self) -> None:
        """_lifespan sets all component keys on app.state during startup
        and closes the httpx client on shutdown."""
        from src.main import _lifespan

        mock_http = AsyncMock()
        mock_feedback = AsyncMock()
        mock_feedback.initialize = AsyncMock()

        mock_components = {
            "http_client": mock_http,
            "pipeline": MagicMock(),
            "confirmation_gate": MagicMock(),
            "progress_tracker": MagicMock(),
            "session_states": {},
            "provider_registry": {"llm": True},
            "provider_list": [{"name": "ollama", "type": "llm", "available": True}],
            "primary_llm_name": "ollama",
            "ingestion_service": None,
            "vector_store": None,
            "rag_enabled": False,
            "qa_service": MagicMock(),
            "feedback_provider": mock_feedback,
        }

        app = FastAPI()

        with patch("src.main._build_all", return_value=mock_components), \
             patch("src.main._auto_ingest_reference_corpus", new_callable=AsyncMock):
            async with _lifespan(app):
                # During lifespan, all keys should be set on state
                assert app.state.pipeline is mock_components["pipeline"]
                assert app.state.primary_llm_name == "ollama"
                assert app.state.rag_enabled is False
                assert app.state.feedback_provider is mock_feedback

            # After shutdown, httpx client should be closed
            mock_http.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_lifespan_calls_auto_ingest(self) -> None:
        """_lifespan calls _auto_ingest_reference_corpus during startup."""
        from src.main import _lifespan

        mock_components = {
            "http_client": AsyncMock(),
            "pipeline": MagicMock(),
            "confirmation_gate": MagicMock(),
            "progress_tracker": MagicMock(),
            "session_states": {},
            "provider_registry": {},
            "provider_list": [],
            "primary_llm_name": "ollama",
            "ingestion_service": None,
            "vector_store": None,
            "rag_enabled": False,
            "qa_service": MagicMock(),
            "feedback_provider": AsyncMock(initialize=AsyncMock()),
        }

        app = FastAPI()
        mock_auto_ingest = AsyncMock()

        with patch("src.main._build_all", return_value=mock_components), \
             patch("src.main._auto_ingest_reference_corpus", mock_auto_ingest):
            async with _lifespan(app):
                pass

        mock_auto_ingest.assert_awaited_once_with(app)

    @pytest.mark.asyncio
    async def test_lifespan_initializes_feedback_db(self) -> None:
        """_lifespan calls feedback_provider.initialize() on startup."""
        from src.main import _lifespan

        mock_feedback = AsyncMock()
        mock_feedback.initialize = AsyncMock()

        mock_components = {
            "http_client": AsyncMock(),
            "pipeline": MagicMock(),
            "confirmation_gate": MagicMock(),
            "progress_tracker": MagicMock(),
            "session_states": {},
            "provider_registry": {},
            "provider_list": [],
            "primary_llm_name": "ollama",
            "ingestion_service": None,
            "vector_store": None,
            "rag_enabled": False,
            "qa_service": MagicMock(),
            "feedback_provider": mock_feedback,
        }

        app = FastAPI()

        with patch("src.main._build_all", return_value=mock_components), \
             patch("src.main._auto_ingest_reference_corpus", new_callable=AsyncMock):
            async with _lifespan(app):
                pass

        mock_feedback.initialize.assert_awaited_once()

"""RaiveFlier FastAPI application entry point.

Wires together all providers, services, and routes via dependency injection.
Loads configuration from ``.env`` and ``config/config.yaml``, configures
structured logging, and mounts static files for the single-page frontend.

Also retains the standalone ``build_pipeline`` / ``run_pipeline`` helpers
for CLI or scripting usage outside the web server.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import structlog
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.middleware import (
    ErrorHandlingMiddleware,
    RequestLoggingMiddleware,
    configure_cors,
)
from src.api.routes import router as api_router
from src.api.websocket import websocket_progress
from src.config.loader import load_config
from src.config.settings import Settings
from src.interfaces.llm_provider import ILLMProvider
from src.models.flier import FlierImage
from src.models.pipeline import PipelinePhase, PipelineState
from src.models.research import ResearchResult
from src.pipeline.confirmation_gate import ConfirmationGate
from src.pipeline.orchestrator import FlierAnalysisPipeline
from src.pipeline.progress_tracker import ProgressTracker
from src.providers.article.wayback_provider import WaybackProvider
from src.providers.article.web_scraper_provider import WebScraperProvider
from src.providers.cache.memory_cache import MemoryCacheProvider
from src.providers.feedback.sqlite_feedback_provider import SQLiteFeedbackProvider
from src.providers.llm.anthropic_provider import AnthropicLLMProvider
from src.providers.llm.ollama_provider import OllamaLLMProvider
from src.providers.llm.openai_provider import OpenAILLMProvider
from src.providers.music_db.bandcamp_provider import BandcampProvider
from src.providers.music_db.beatport_provider import BeatportProvider
from src.providers.music_db.discogs_api_provider import DiscogsAPIProvider
from src.providers.music_db.discogs_scrape_provider import DiscogsScrapeProvider
from src.providers.music_db.musicbrainz_provider import MusicBrainzProvider
from src.providers.ocr.llm_vision_provider import LLMVisionOCRProvider
from src.providers.ocr.tesseract_provider import TesseractOCRProvider

# EasyOCR is optional — it pulls in PyTorch (~2 GB) which exceeds RAM on
# lightweight deployments (e.g. Render free tier 512 MB).
try:
    from src.providers.ocr.easyocr_provider import EasyOCRProvider

    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False
from src.providers.search.duckduckgo_provider import DuckDuckGoSearchProvider
from src.services.artist_researcher import ArtistResearcher
from src.services.citation_service import CitationService
from src.services.date_context_researcher import DateContextResearcher
from src.services.entity_extractor import EntityExtractor
from src.services.event_name_researcher import EventNameResearcher
from src.services.interconnection_service import InterconnectionService
from src.services.ocr_service import OCRService
from src.services.promoter_researcher import PromoterResearcher
from src.services.research_service import ResearchService
from src.services.venue_researcher import VenueResearcher
from src.utils.image_preprocessor import ImagePreprocessor
from src.utils.logging import configure_logging, get_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# ---------------------------------------------------------------------------
# Module-level settings & logging
# ---------------------------------------------------------------------------

settings = Settings()
config = load_config()

configure_logging(
    log_level=settings.log_level,
    json_output=(settings.app_env == "production"),
)
_logger: structlog.BoundLogger = get_logger(__name__)


# ---------------------------------------------------------------------------
# LLM provider selection
# ---------------------------------------------------------------------------


def _build_llm_provider(app_settings: Settings) -> ILLMProvider:
    """Select the first available LLM provider based on configured API keys.

    Priority order: Anthropic -> OpenAI -> Ollama (always available).
    """
    if app_settings.anthropic_api_key:
        return AnthropicLLMProvider(settings=app_settings)
    if app_settings.openai_api_key:
        return OpenAILLMProvider(settings=app_settings)
    return OllamaLLMProvider(settings=app_settings)


def _build_embedding_provider(app_settings: Settings):  # noqa: ANN202
    """Select the first available embedding provider.

    Priority: OpenAI/OpenAI-compatible (if API key set) ->
              Nomic/Ollama (if reachable).
    Returns ``None`` if no embedding provider is available.
    """
    from src.interfaces.embedding_provider import IEmbeddingProvider

    if app_settings.openai_api_key:
        from src.providers.embedding.openai_embedding_provider import (
            OpenAIEmbeddingProvider,
        )

        provider: IEmbeddingProvider = OpenAIEmbeddingProvider(settings=app_settings)
        if provider.is_available():
            return provider

    from src.providers.embedding.nomic_embedding_provider import (
        NomicEmbeddingProvider,
    )

    provider = NomicEmbeddingProvider(settings=app_settings)
    if provider.is_available():
        return provider

    return None


# ---------------------------------------------------------------------------
# Full DI assembly for the FastAPI application
# ---------------------------------------------------------------------------


def _build_all(app_settings: Settings) -> dict[str, Any]:
    """Construct every provider and service instance for the application.

    Returns a flat dict of named components to be stored on ``app.state``.
    """
    # -- Shared resources --
    http_client = httpx.AsyncClient(timeout=30.0)
    preprocessor = ImagePreprocessor()

    # -- LLM --
    primary_llm = _build_llm_provider(app_settings)

    # -- LLM provider metadata --
    llm_meta: list[dict[str, Any]] = []
    if app_settings.anthropic_api_key:
        llm_meta.append({"name": "anthropic", "type": "llm", "available": True})
    if app_settings.openai_api_key:
        llm_meta.append({"name": "openai", "type": "llm", "available": True})
    llm_meta.append({"name": "ollama", "type": "llm", "available": True})

    # -- OCR providers (ordered by priority) --
    ocr_providers = []
    if primary_llm.supports_vision():
        ocr_providers.append(
            LLMVisionOCRProvider(llm_provider=primary_llm, preprocessor=preprocessor)
        )
    if _EASYOCR_AVAILABLE:
        ocr_providers.append(EasyOCRProvider(preprocessor=preprocessor))
    ocr_providers.append(TesseractOCRProvider(preprocessor=preprocessor))

    # -- Music DB providers --
    music_dbs = []
    if app_settings.discogs_consumer_key and app_settings.discogs_consumer_secret:
        music_dbs.append(DiscogsAPIProvider(settings=app_settings))
    music_dbs.append(DiscogsScrapeProvider(http_client=http_client))
    music_dbs.append(MusicBrainzProvider(settings=app_settings))
    music_dbs.append(BandcampProvider(http_client=http_client))
    music_dbs.append(BeatportProvider(http_client=http_client))

    # -- Search providers --
    primary_search = DuckDuckGoSearchProvider()

    # -- Article providers --
    primary_article = WebScraperProvider(http_client=http_client)
    _wayback = WaybackProvider(http_client=http_client)  # noqa: F841

    # -- Cache --
    cache = MemoryCacheProvider()

    # -- RAG components (only when RAG_ENABLED=True) --
    vector_store = None
    ingestion_service = None

    if app_settings.rag_enabled:
        embedding_provider = _build_embedding_provider(app_settings)
        if embedding_provider is not None and embedding_provider.is_available():
            from src.providers.vector_store.chromadb_provider import ChromaDBProvider
            from src.services.ingestion.chunker import TextChunker
            from src.services.ingestion.ingestion_service import IngestionService
            from src.services.ingestion.metadata_extractor import MetadataExtractor

            vector_store = ChromaDBProvider(
                embedding_provider=embedding_provider,
                persist_directory=app_settings.chromadb_persist_dir,
                collection_name=app_settings.chromadb_collection,
            )

            chunker = TextChunker()
            metadata_extractor = MetadataExtractor(llm=primary_llm)
            ingestion_service = IngestionService(
                chunker=chunker,
                metadata_extractor=metadata_extractor,
                embedding_provider=embedding_provider,
                vector_store=vector_store,
                article_scraper=primary_article,
            )

            _logger.info(
                "rag_enabled",
                embedding_provider=embedding_provider.get_provider_name(),
                vector_store="chromadb",
                persist_dir=app_settings.chromadb_persist_dir,
            )
        else:
            _logger.warning(
                "rag_enabled_but_no_embedding_provider",
                msg="RAG_ENABLED=True but no embedding provider available. RAG disabled.",
            )

    # -- Q&A Service --
    from src.services.qa_service import QAService

    qa_service = QAService(
        llm=primary_llm,
        vector_store=vector_store,
        cache=cache,
    )

    # -- Services --
    ocr_min_conf = config.get("ocr", {}).get("min_confidence", 0.7)
    ocr_service = OCRService(providers=ocr_providers, min_confidence=ocr_min_conf)
    entity_extractor = EntityExtractor(llm_provider=primary_llm)

    artist_researcher = ArtistResearcher(
        music_dbs=music_dbs,
        web_search=primary_search,
        article_scraper=primary_article,
        llm=primary_llm,
        cache=cache,
        vector_store=vector_store,
    )
    venue_researcher = VenueResearcher(
        web_search=primary_search,
        article_scraper=primary_article,
        llm=primary_llm,
        cache=cache,
        vector_store=vector_store,
    )
    promoter_researcher = PromoterResearcher(
        web_search=primary_search,
        article_scraper=primary_article,
        llm=primary_llm,
        cache=cache,
        vector_store=vector_store,
    )
    date_researcher = DateContextResearcher(
        web_search=primary_search,
        article_scraper=primary_article,
        llm=primary_llm,
        cache=cache,
    )
    event_name_researcher = EventNameResearcher(
        web_search=primary_search,
        article_scraper=primary_article,
        llm=primary_llm,
        cache=cache,
        vector_store=vector_store,
    )
    research_service = ResearchService(
        artist_researcher=artist_researcher,
        venue_researcher=venue_researcher,
        promoter_researcher=promoter_researcher,
        date_context_researcher=date_researcher,
        event_name_researcher=event_name_researcher,
    )

    citation_service = CitationService()
    interconnection_service = InterconnectionService(
        llm_provider=primary_llm,
        citation_service=citation_service,
        vector_store=vector_store,
    )
    progress_tracker = ProgressTracker()
    confirmation_gate = ConfirmationGate()

    pipeline = FlierAnalysisPipeline(
        ocr_service=ocr_service,
        entity_extractor=entity_extractor,
        research_service=research_service,
        interconnection_service=interconnection_service,
        citation_service=citation_service,
        progress_tracker=progress_tracker,
        ingestion_service=ingestion_service,
    )

    # -- Provider registry for /health --
    provider_registry: dict[str, bool] = {
        "llm": True,
        "ocr": len(ocr_providers) > 0,
        "music_db": len(music_dbs) > 0,
        "search": True,
        "article": True,
        "cache": True,
        "rag": vector_store is not None,
    }

    # -- Provider list for /providers --
    provider_list: list[dict[str, Any]] = list(llm_meta)
    for ocr_p in ocr_providers:
        provider_list.append({"name": type(ocr_p).__name__, "type": "ocr", "available": True})
    for mdb in music_dbs:
        provider_list.append({"name": type(mdb).__name__, "type": "music_db", "available": True})
    provider_list.append({"name": "DuckDuckGoSearchProvider", "type": "search", "available": True})
    provider_list.append({"name": "WebScraperProvider", "type": "article", "available": True})
    provider_list.append({"name": "WaybackProvider", "type": "article", "available": True})

    # -- Feedback provider (SQLite-backed ratings persistence) --
    feedback_provider = SQLiteFeedbackProvider(db_path="data/feedback.db")

    return {
        "http_client": http_client,
        "pipeline": pipeline,
        "confirmation_gate": confirmation_gate,
        "progress_tracker": progress_tracker,
        "session_states": {},
        "provider_registry": provider_registry,
        "provider_list": provider_list,
        "primary_llm_name": primary_llm.get_provider_name(),
        "ingestion_service": ingestion_service,
        "vector_store": vector_store,
        "rag_enabled": app_settings.rag_enabled and vector_store is not None,
        "qa_service": qa_service,
        "feedback_provider": feedback_provider,
    }


# ---------------------------------------------------------------------------
# Standalone pipeline helpers (CLI / scripting)
# ---------------------------------------------------------------------------


def build_pipeline(custom_settings: Settings | None = None) -> dict[str, Any]:
    """Construct and return all pipeline services with injected dependencies.

    Parameters
    ----------
    custom_settings:
        Application settings.  Uses module-level ``settings`` if not provided.

    Returns
    -------
    dict
        A dictionary of service instances keyed by role name.
    """
    s = custom_settings or settings

    llm = _build_llm_provider(s)
    cache = MemoryCacheProvider()
    web_search = DuckDuckGoSearchProvider()
    http_client = httpx.AsyncClient()
    preprocessor = ImagePreprocessor()
    article_scraper = WebScraperProvider(http_client=http_client)

    music_dbs = []
    if s.discogs_consumer_key:
        music_dbs.append(DiscogsAPIProvider(settings=s))
    music_dbs.append(DiscogsScrapeProvider(http_client=http_client))
    music_dbs.append(MusicBrainzProvider(settings=s))
    music_dbs.append(BandcampProvider(http_client=http_client))
    music_dbs.append(BeatportProvider(http_client=http_client))

    ocr_providers = []
    if llm.supports_vision():
        ocr_providers.append(LLMVisionOCRProvider(llm_provider=llm))
    if _EASYOCR_AVAILABLE:
        ocr_providers.append(EasyOCRProvider(preprocessor=preprocessor))
    ocr_providers.append(TesseractOCRProvider(preprocessor=preprocessor))

    ocr_min_conf = config.get("ocr", {}).get("min_confidence", 0.7)
    ocr_service = OCRService(providers=ocr_providers, min_confidence=ocr_min_conf)
    entity_extractor = EntityExtractor(llm_provider=llm)

    artist_researcher = ArtistResearcher(
        music_dbs=music_dbs,
        web_search=web_search,
        article_scraper=article_scraper,
        llm=llm,
        cache=cache,
    )
    venue_researcher = VenueResearcher(
        web_search=web_search,
        article_scraper=article_scraper,
        llm=llm,
        cache=cache,
    )
    promoter_researcher = PromoterResearcher(
        web_search=web_search,
        article_scraper=article_scraper,
        llm=llm,
        cache=cache,
    )
    date_context_researcher = DateContextResearcher(
        web_search=web_search,
        article_scraper=article_scraper,
        llm=llm,
        cache=cache,
    )
    event_name_researcher = EventNameResearcher(
        web_search=web_search,
        article_scraper=article_scraper,
        llm=llm,
        cache=cache,
    )

    citation_service = CitationService()
    interconnection_service = InterconnectionService(
        llm_provider=llm,
        citation_service=citation_service,
    )

    return {
        "ocr_service": ocr_service,
        "entity_extractor": entity_extractor,
        "artist_researcher": artist_researcher,
        "venue_researcher": venue_researcher,
        "promoter_researcher": promoter_researcher,
        "date_context_researcher": date_context_researcher,
        "event_name_researcher": event_name_researcher,
        "citation_service": citation_service,
        "interconnection_service": interconnection_service,
        "settings": s,
    }


async def run_pipeline(flier: FlierImage) -> PipelineState:
    """Execute the full flier-analysis pipeline (standalone / CLI usage).

    Phases: UPLOAD -> OCR -> ENTITY_EXTRACTION -> RESEARCH ->
    INTERCONNECTION -> OUTPUT.

    Parameters
    ----------
    flier:
        The uploaded flier image to analyse.

    Returns
    -------
    PipelineState
        Final pipeline state containing all results.
    """
    services = build_pipeline()
    session_id = str(uuid4())

    state = PipelineState(session_id=session_id, flier=flier)

    # -- Phase 1: OCR --
    _logger.info("pipeline_phase", phase="OCR", session=session_id)
    state = state.model_copy(update={"current_phase": PipelinePhase.OCR})
    ocr_result = await services["ocr_service"].extract_text(flier)
    state = state.model_copy(update={"ocr_result": ocr_result})

    # -- Phase 2: Entity Extraction --
    _logger.info("pipeline_phase", phase="ENTITY_EXTRACTION", session=session_id)
    state = state.model_copy(update={"current_phase": PipelinePhase.ENTITY_EXTRACTION})
    extracted = await services["entity_extractor"].extract(ocr_result, flier)
    state = state.model_copy(
        update={
            "extracted_entities": extracted,
            "confirmed_entities": extracted,
        }
    )

    # -- Phase 3: Research --
    _logger.info("pipeline_phase", phase="RESEARCH", session=session_id)
    state = state.model_copy(update={"current_phase": PipelinePhase.RESEARCH})

    research_results: list[ResearchResult] = []
    for artist_entity in extracted.artists:
        result = await services["artist_researcher"].research(artist_entity.text)
        research_results.append(result)
    if extracted.venue:
        result = await services["venue_researcher"].research(extracted.venue.text)
        research_results.append(result)
    if extracted.promoter:
        result = await services["promoter_researcher"].research(extracted.promoter.text)
        research_results.append(result)
    if extracted.date:
        result = await services["date_context_researcher"].research(extracted.date.text)
        research_results.append(result)

    state = state.model_copy(update={"research_results": research_results})

    # -- Phase 4: Interconnection Analysis --
    _logger.info("pipeline_phase", phase="INTERCONNECTION", session=session_id)
    state = state.model_copy(update={"current_phase": PipelinePhase.INTERCONNECTION})
    interconnection_map = await services["interconnection_service"].analyze(
        research_results=research_results,
        entities=extracted,
    )
    state = state.model_copy(update={"interconnection_map": interconnection_map})

    # -- Phase 5: Output --
    _logger.info("pipeline_phase", phase="OUTPUT", session=session_id)
    state = state.model_copy(
        update={
            "current_phase": PipelinePhase.OUTPUT,
            "completed_at": datetime.now(tz=timezone.utc),  # noqa: UP017
            "progress_percent": 100.0,
        }
    )

    _logger.info(
        "pipeline_complete",
        session=session_id,
        entities=len(extracted.artists),
        edges=len(interconnection_map.edges),
        patterns=len(interconnection_map.patterns),
    )

    return state


# ---------------------------------------------------------------------------
# Application lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(application: FastAPI):  # noqa: ANN201
    """Initialise all providers and services on startup, clean up on shutdown."""
    components = _build_all(settings)

    for key, value in components.items():
        setattr(application.state, key, value)

    # Initialize feedback database (creates table if needed)
    if hasattr(application.state, "feedback_provider"):
        await application.state.feedback_provider.initialize()

    _logger.info(
        "app_startup",
        version="0.1.0",
        environment=settings.app_env,
        primary_llm=components["primary_llm_name"],
        providers=len(components["provider_list"]),
    )

    yield

    # -- Shutdown: close shared httpx client --
    http_client: httpx.AsyncClient = components["http_client"]
    await http_client.aclose()
    _logger.info("app_shutdown", message="HTTP client closed")


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    application = FastAPI(
        title="raiveFlier API",
        version="0.1.0",
        description=(
            "Upload a rave flier image, extract artist/venue/promoter information "
            "via OCR and LLM analysis, then deep-research every entity with "
            "citations from music databases, web search, and article scraping."
        ),
        lifespan=_lifespan,
    )

    # -- Middleware (order matters: last added = first executed) --
    application.add_middleware(ErrorHandlingMiddleware)
    application.add_middleware(RequestLoggingMiddleware)
    configure_cors(application)

    # -- API routes --
    application.include_router(api_router)

    # -- WebSocket --
    @application.websocket("/ws/progress/{session_id}")
    async def ws_progress(websocket: WebSocket, session_id: str) -> None:
        await websocket_progress(websocket, session_id)

    # -- Frontend static files --
    if _FRONTEND_DIR.exists():
        if (_FRONTEND_DIR / "css").exists():
            application.mount(
                "/css",
                StaticFiles(directory=str(_FRONTEND_DIR / "css")),
                name="css",
            )
        if (_FRONTEND_DIR / "js").exists():
            application.mount(
                "/js",
                StaticFiles(directory=str(_FRONTEND_DIR / "js")),
                name="js",
            )

        @application.get("/", include_in_schema=False)
        async def serve_index() -> FileResponse:
            return FileResponse(str(_FRONTEND_DIR / "index.html"))

    return application


app = create_app()

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=(settings.app_env == "development"),
    )

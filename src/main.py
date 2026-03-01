"""RaiveFlier FastAPI application entry point.

Wires together all providers, services, and routes via dependency injection.
Loads configuration from ``.env`` and ``config/config.yaml``, configures
structured logging, and mounts static files for the single-page frontend.

Also retains the standalone ``build_pipeline`` / ``run_pipeline`` helpers
for CLI or scripting usage outside the web server.

# ─── HOW THIS FILE WORKS (Junior Developer Guide) ─────────────────────
#
# This is the "composition root" — the single place where every provider,
# service, and pipeline component is created and wired together via
# **dependency injection (DI)**.  No service in the codebase creates its
# own dependencies; they are all constructed here and passed in.
#
# Key concepts:
#   1. **Provider selection** — The app picks the best available LLM,
#      OCR, embedding, and music-DB provider based on which API keys
#      are configured in the environment (.env).
#   2. **Lifespan context** — FastAPI's `lifespan` hook runs at startup
#      and shutdown.  Startup wires everything; shutdown cleans up.
#   3. **app.state** — All singletons are stored on `app.state` so
#      route handlers can retrieve them via FastAPI Depends().
#   4. **Two usage modes** — The same DI logic powers both:
#        • The web server (create_app → uvicorn)
#        • The CLI / scripting path (build_pipeline → run_pipeline)
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
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

# ── Application layers ────────────────────────────────────────────────
# Imports are grouped by layer: middleware → config → interfaces →
# models → pipeline → providers → services → utils.  This ordering
# mirrors the dependency graph (lower layers first).
# ──────────────────────────────────────────────────────────────────────
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
from src.providers.session.sqlite_session_store import PersistentSessionStore
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

# ── Conditional import: EasyOCR ──────────────────────────────────────
# EasyOCR is optional — it pulls in PyTorch (~2 GB) which exceeds RAM on
# lightweight deployments (e.g. Render free tier 512 MB).
# The try/except "graceful import" pattern lets the app start without it;
# the _EASYOCR_AVAILABLE flag is checked later when building the OCR
# provider chain so EasyOCR is only added if importable.
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

# Resolve the frontend/ directory relative to *this file* (src/main.py),
# going up one level to the project root, then into "frontend/".
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# ---------------------------------------------------------------------------
# Module-level settings & logging
# ---------------------------------------------------------------------------

# Settings() reads .env + environment variables automatically via
# pydantic-settings.  config loads config/config.yaml with env overrides.
# Both are created once at module import time (singleton pattern).
settings = Settings()
config = load_config()

# Structured logging: human-readable in dev, JSON in production.
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

    Priority order: OpenAI/TogetherAI -> Anthropic -> Ollama (always available).

    # JUNIOR DEV NOTE — Provider Priority Chain
    # ------------------------------------------
    # This is a common "fallback chain" pattern:
    #   1. Check if the user has an OpenAI key → use OpenAI (or a compatible
    #      API like TogetherAI via OPENAI_BASE_URL override).
    #   2. Else check Anthropic key → use Claude.
    #   3. Else default to Ollama (runs locally, no API key needed).
    #
    # Every provider implements ILLMProvider, so the rest of the code
    # doesn't care which one was selected.  This is the Strategy pattern.
    """
    if app_settings.openai_api_key:
        return OpenAILLMProvider(settings=app_settings)
    if app_settings.anthropic_api_key:
        return AnthropicLLMProvider(settings=app_settings)
    # Ollama is the "always available" local fallback — no API key required,
    # but the user must have the Ollama server running at ollama_base_url.
    return OllamaLLMProvider(settings=app_settings)


async def _build_embedding_provider(app_settings: Settings):  # noqa: ANN202
    """Select the first available embedding provider.

    Priority: OpenAI/OpenAI-compatible (if API key set and reachable) ->
              FastEmbed (ONNX, lightweight) ->
              SentenceTransformer (local, PyTorch-based) ->
              Nomic/Ollama (if reachable).
    Returns ``None`` if no embedding provider is available.

    # JUNIOR DEV NOTE — Why so many fallbacks?
    # -----------------------------------------
    # Embeddings convert text → numeric vectors for similarity search (RAG).
    # Different environments have different constraints:
    #   - Cloud (Render 512 MB)   → FastEmbed (~50 MB) is the sweet spot
    #   - Local with GPU           → SentenceTransformer gives best quality
    #   - API-only (no local model)→ OpenAI embedding endpoint
    #   - Ollama running locally   → Nomic via Ollama
    #
    # Imports are deferred (inside the function) to avoid ImportError if
    # an optional dependency (fastembed, sentence-transformers) isn't installed.
    # Each provider's is_available() check verifies that its dependencies
    # actually exist before we try to use it.
    """
    from src.interfaces.embedding_provider import IEmbeddingProvider

    # --- Tier 1: OpenAI embedding API (best quality, costs money) ---
    if app_settings.openai_api_key:
        from src.providers.embedding.openai_embedding_provider import (
            OpenAIEmbeddingProvider,
        )

        provider: IEmbeddingProvider = OpenAIEmbeddingProvider(settings=app_settings)
        if provider.is_available():
            try:
                # Smoke-test: embed a single word to verify the API key and
                # endpoint are actually reachable before committing to this provider.
                await provider.embed_single("test")
                return provider
            except Exception as exc:
                _logger.warning(
                    "openai_embedding_unavailable",
                    error=str(exc)[:200],
                    msg="Falling back to local embedding provider.",
                )

    # --- Tier 2: FastEmbed (ONNX, no PyTorch, ~50 MB) ---
    # Default for Docker production — lightweight and fast.
    from src.providers.embedding.fastembed_embedding_provider import (
        FastEmbedEmbeddingProvider,
    )

    fe_provider: IEmbeddingProvider = FastEmbedEmbeddingProvider()
    if fe_provider.is_available():
        return fe_provider

    # --- Tier 3: SentenceTransformer (PyTorch, ~500 MB) ---
    # Better quality than FastEmbed but too heavy for Render's free tier.
    from src.providers.embedding.sentence_transformer_embedding_provider import (
        SentenceTransformerEmbeddingProvider,
    )

    st_provider: IEmbeddingProvider = SentenceTransformerEmbeddingProvider()
    if st_provider.is_available():
        return st_provider

    # --- Tier 4: Nomic via Ollama (free, local, requires Ollama server) ---
    from src.providers.embedding.nomic_embedding_provider import (
        NomicEmbeddingProvider,
    )

    provider = NomicEmbeddingProvider(settings=app_settings)
    if provider.is_available():
        return provider

    # All providers exhausted — RAG will be disabled.
    return None


# ---------------------------------------------------------------------------
# Full DI assembly for the FastAPI application
# ---------------------------------------------------------------------------


def _build_story_components(
    app_settings: Settings,
    primary_llm: Any,
    vector_store: Any,
    embedding_provider: Any,
) -> dict[str, Any]:
    """Build DI components for the Rave Stories sub-app.

    Returns a dict of named components that will be wired into the
    stories sub-app's ``app.state`` during its lifespan startup.
    The story_store.initialize() is called by the stories sub-app lifespan.
    """
    from src.providers.story.sqlite_story_provider import SQLiteStoryProvider
    from src.services.story_service import StoryService

    story_store = SQLiteStoryProvider(db_path=app_settings.story_db_path)

    # Transcription providers (optional — only needed for audio submissions).
    transcription = None
    transcription_fallback = None
    try:
        from src.providers.transcription.whisper_api_provider import WhisperAPIProvider
        if app_settings.openai_api_key:
            transcription = WhisperAPIProvider(api_key=app_settings.openai_api_key)
    except ImportError:
        pass
    try:
        from src.providers.transcription.whisper_local_provider import WhisperLocalProvider
        fallback = WhisperLocalProvider(model_size="tiny")
        if fallback.is_available():
            transcription_fallback = fallback
    except ImportError:
        pass

    story_service = StoryService(
        llm=primary_llm,
        story_store=story_store,
        transcription=transcription,
        transcription_fallback=transcription_fallback,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    return {
        "story_store": story_store,
        "story_service": story_service,
    }


async def _build_all(app_settings: Settings) -> dict[str, Any]:
    """Construct every provider and service instance for the application.

    This is the **Composition Root** — the single function that wires
    together the entire dependency graph for the web server.  Every
    provider and service is instantiated here, then stored on
    ``app.state`` so route handlers can access them via FastAPI Depends().

    Returns a flat dict of named components to be stored on ``app.state``.
    """
    # -- Shared resources --
    # A single httpx client is shared across all providers that make
    # outbound HTTP calls (Bandcamp, Beatport, Discogs scrape, Wayback,
    # web scraper).  Reusing one client = connection pooling + one
    # cleanup point at shutdown.
    http_client = httpx.AsyncClient(timeout=30.0)
    preprocessor = ImagePreprocessor()

    # -- LLM --
    # Select the best available LLM based on configured API keys.
    primary_llm = _build_llm_provider(app_settings)

    # -- LLM provider metadata --
    # Build a list of *all* available providers (not just the primary one)
    # for the /providers endpoint.  This lets the frontend show what's configured.
    llm_meta: list[dict[str, Any]] = []
    if app_settings.anthropic_api_key:
        llm_meta.append({"name": "anthropic", "type": "llm", "available": True})
    if app_settings.openai_api_key:
        llm_meta.append({"name": "openai", "type": "llm", "available": True})
    llm_meta.append({"name": "ollama", "type": "llm", "available": True})

    # -- OCR providers (ordered by priority) --
    # The list order determines the OCR fallback chain:
    #   1. LLM Vision — most context-aware (reads stylized rave text)
    #   2. EasyOCR    — PyTorch-based, good accuracy (if installed)
    #   3. Tesseract  — always available, good baseline
    # OCRService tries each provider in order until one succeeds.
    ocr_providers = []
    if primary_llm.supports_vision():
        ocr_providers.append(
            LLMVisionOCRProvider(llm_provider=primary_llm, preprocessor=preprocessor)
        )
    if _EASYOCR_AVAILABLE:
        ocr_providers.append(EasyOCRProvider(preprocessor=preprocessor))
    ocr_providers.append(TesseractOCRProvider(preprocessor=preprocessor))

    # -- Music DB providers --
    # Each provider queries a different music database.  Researchers try
    # them in order and aggregate results across all sources.
    #   - DiscogsAPI:     Official API (requires consumer key/secret)
    #   - DiscogsScrape:  HTML scraping fallback (no key needed)
    #   - MusicBrainz:    Open database (rate-limited, no key needed)
    #   - Bandcamp:       Artist page scraping
    #   - Beatport:       DJ/electronic-focused catalog scraping
    music_dbs = []
    if app_settings.discogs_consumer_key and app_settings.discogs_consumer_secret:
        music_dbs.append(DiscogsAPIProvider(settings=app_settings))
    music_dbs.append(DiscogsScrapeProvider(http_client=http_client))
    music_dbs.append(MusicBrainzProvider(settings=app_settings))
    music_dbs.append(BandcampProvider(http_client=http_client))
    music_dbs.append(BeatportProvider(http_client=http_client))

    # -- Search providers --
    # DuckDuckGo is the only search provider (free, no API key).
    # To add Google/Bing, create a new IWebSearchProvider implementation.
    primary_search = DuckDuckGoSearchProvider()

    # -- Article providers --
    # WebScraperProvider fetches and extracts text from web pages.
    # WaybackProvider queries the Internet Archive for historical versions.
    primary_article = WebScraperProvider(http_client=http_client)
    _wayback = WaybackProvider(http_client=http_client)  # noqa: F841

    # -- Cache --
    # In-memory TTL cache — avoids redundant API calls within a session.
    # Not shared across processes; swap for Redis in multi-worker deploys.
    cache = MemoryCacheProvider()

    # -- RAG components (only when RAG_ENABLED=True) --
    # RAG = Retrieval-Augmented Generation.  When enabled, the app can
    # search a curated reference corpus (books, articles, interviews about
    # rave culture) and inject relevant excerpts into LLM prompts for
    # richer, citation-backed analysis.
    #
    # The RAG pipeline requires TWO components:
    #   1. An embedding provider (to convert text → vectors)
    #   2. A vector store (ChromaDB — stores and searches those vectors)
    # If either is unavailable, RAG is silently disabled.
    vector_store = None
    ingestion_service = None
    embedding_provider = None

    if app_settings.rag_enabled:
        embedding_provider = await _build_embedding_provider(app_settings)
        if embedding_provider is not None and embedding_provider.is_available():
            # Deferred imports — these pull in chromadb and fastembed,
            # which we don't want to load unless RAG is actually enabled.
            from src.providers.vector_store.chromadb_provider import ChromaDBProvider
            from src.services.ingestion.chunker import TextChunker
            from src.services.ingestion.ingestion_service import IngestionService
            from src.services.ingestion.metadata_extractor import MetadataExtractor

            vector_store = ChromaDBProvider(
                embedding_provider=embedding_provider,
                persist_directory=app_settings.chromadb_persist_dir,
                collection_name=app_settings.chromadb_collection,
            )

            # The ingestion pipeline: text → chunks → metadata → embeddings → store
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
                embedding_dimension=embedding_provider.get_dimension(),
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

    # Web search fallback gives the Q&A sidebar access to live DuckDuckGo
    # results when the RAG corpus doesn't have relevant passages.
    qa_service = QAService(
        llm=primary_llm,
        vector_store=vector_store,
        cache=cache,
        web_search=primary_search,
    )

    # -- Feedback provider (SQLite-backed ratings persistence) --
    feedback_provider = SQLiteFeedbackProvider(db_path=app_settings.feedback_db_path)

    # -- Flier history provider (SQLite-backed flier data persistence) --
    from src.providers.flier_history.sqlite_flier_history_provider import SQLiteFlierHistoryProvider

    flier_history = SQLiteFlierHistoryProvider(db_path=app_settings.flier_history_db_path)
    await flier_history.initialize()

    # -- Graph aggregation service (combines all stored analyses) --
    from src.services.graph_aggregation_service import GraphAggregationService

    graph_aggregation = GraphAggregationService(flier_history=flier_history)

    # -- Services --
    # Services are the business-logic layer.  Each service receives its
    # dependencies (providers) via constructor injection — never via globals.
    # This makes testing easy: inject mocks instead of real providers.

    # OCR: orchestrates the OCR fallback chain, filtering results below
    # min_confidence (default 0.7 = 70%).
    ocr_min_conf = config.get("ocr", {}).get("min_confidence", 0.7)
    ocr_service = OCRService(providers=ocr_providers, min_confidence=ocr_min_conf)

    # Entity Extraction: uses the LLM to parse OCR text into structured
    # entities (artists, venue, promoter, date, event name).
    entity_extractor = EntityExtractor(llm_provider=primary_llm)

    # --- Researchers: one per entity type ---
    # Each researcher uses web search + article scraping + music DBs + LLM
    # to build a rich profile.  The vector_store param enables RAG-enhanced
    # research; pass None to disable RAG for that researcher.
    artist_researcher = ArtistResearcher(
        music_dbs=music_dbs,
        web_search=primary_search,
        article_scraper=primary_article,
        llm=primary_llm,
        cache=cache,
        vector_store=vector_store,
        feedback=feedback_provider,
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

    # ResearchService is a facade that dispatches to the correct
    # researcher based on entity type (artist → ArtistResearcher, etc.)
    research_service = ResearchService(
        artist_researcher=artist_researcher,
        venue_researcher=venue_researcher,
        promoter_researcher=promoter_researcher,
        date_context_researcher=date_researcher,
        event_name_researcher=event_name_researcher,
    )

    # CitationService formats source references into academic-style citations.
    citation_service = CitationService()
    # InterconnectionService uses the LLM to discover relationships between
    # the researched entities (e.g., "Artist X has released on Venue Y's label").
    interconnection_service = InterconnectionService(
        llm_provider=primary_llm,
        citation_service=citation_service,
        vector_store=vector_store,
    )

    # -- Recommendation service --
    from src.services.recommendation_service import RecommendationService

    recommendation_service = RecommendationService(
        llm_provider=primary_llm,
        music_dbs=music_dbs,
        vector_store=vector_store,
        flier_history=flier_history,
    )

    # -- Persistent session stores (survive container restarts) --
    # PersistentSessionStore implements MutableMapping[str, PipelineState]
    # backed by SQLite.  Two separate tables:
    #   • "sessions"         — completed/in-progress pipeline states (72h TTL)
    #   • "pending_sessions" — states waiting for user confirmation (24h TTL)
    # This means if Render's container restarts, users can resume where
    # they left off (within the TTL window).
    session_store = PersistentSessionStore(
        db_path=app_settings.session_db_path,
        table_name="sessions",
        max_age_hours=72,
    )
    pending_store = PersistentSessionStore(
        db_path=app_settings.session_db_path,
        table_name="pending_sessions",
        max_age_hours=24,
    )
    session_store.initialize()
    pending_store.initialize()

    # ProgressTracker pushes phase/percent updates to WebSocket listeners.
    progress_tracker = ProgressTracker()
    # ConfirmationGate manages the human-in-the-loop pause point between
    # Phase 1 (OCR + extraction) and Phase 2 (research).
    confirmation_gate = ConfirmationGate(pending_store=pending_store)

    # The pipeline orchestrator runs the 5 analysis phases in order.
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
    # Simple bool map: "is this capability available?"  Used by the
    # /health endpoint to report "healthy", "degraded", or "unhealthy".
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
    # Detailed list of every concrete provider and its type, used by the
    # /providers endpoint for debugging / frontend display.
    provider_list: list[dict[str, Any]] = list(llm_meta)
    for ocr_p in ocr_providers:
        provider_list.append({"name": type(ocr_p).__name__, "type": "ocr", "available": True})
    for mdb in music_dbs:
        provider_list.append({"name": type(mdb).__name__, "type": "music_db", "available": True})
    provider_list.append({"name": "DuckDuckGoSearchProvider", "type": "search", "available": True})
    provider_list.append({"name": "WebScraperProvider", "type": "article", "available": True})
    provider_list.append({"name": "WaybackProvider", "type": "article", "available": True})

    return {
        "http_client": http_client,
        "pipeline": pipeline,
        "confirmation_gate": confirmation_gate,
        "progress_tracker": progress_tracker,
        "session_states": session_store,
        "provider_registry": provider_registry,
        "provider_list": provider_list,
        "primary_llm_name": primary_llm.get_provider_name(),
        "primary_llm": primary_llm,
        "ingestion_service": ingestion_service,
        "vector_store": vector_store,
        "rag_enabled": app_settings.rag_enabled and vector_store is not None,
        "qa_service": qa_service,
        "feedback_provider": feedback_provider,
        "flier_history": flier_history,
        "graph_aggregation": graph_aggregation,
        "recommendation_service": recommendation_service,
        # Web search provider — exposed for the corpus sidebar's web-search
        # tier, which augments RAG results with live DDG web results filtered
        # for electronic music relevance.
        "web_search": primary_search,
        # Rave Stories components — pre-built here so the stories sub-app
        # can use the same LLM, vector store, and embedding providers as
        # the main app (no duplicate provider instances).
        "story_components": _build_story_components(
            app_settings, primary_llm, vector_store,
            embedding_provider if app_settings.rag_enabled else None,
        ),
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

    cli_feedback = SQLiteFeedbackProvider(db_path="data/feedback.db")

    artist_researcher = ArtistResearcher(
        music_dbs=music_dbs,
        web_search=web_search,
        article_scraper=article_scraper,
        llm=llm,
        cache=cache,
        feedback=cli_feedback,
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

    # JUNIOR DEV NOTE — CLI vs Web pipeline
    # This function runs all phases sequentially with NO human-in-the-loop
    # confirmation step.  It's the CLI shortcut — upload a flier, get results.
    # The web path (routes.py) splits this into two calls with a user-review
    # pause between Phase 1 and Phase 2 (see ConfirmationGate).

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
    # Extract raw text from the flier image using the OCR fallback chain.
    _logger.info("pipeline_phase", phase="OCR", session=session_id)
    state = state.model_copy(update={"current_phase": PipelinePhase.OCR})
    ocr_result = await services["ocr_service"].extract_text(flier)
    state = state.model_copy(update={"ocr_result": ocr_result})

    # -- Phase 2: Entity Extraction --
    # Use the LLM to parse raw OCR text into structured entities.
    # In CLI mode, extracted_entities == confirmed_entities (no user review).
    _logger.info("pipeline_phase", phase="ENTITY_EXTRACTION", session=session_id)
    state = state.model_copy(update={"current_phase": PipelinePhase.ENTITY_EXTRACTION})
    extracted = await services["entity_extractor"].extract(ocr_result, flier)
    state = state.model_copy(
        update={
            "extracted_entities": extracted,
            "confirmed_entities": extracted,
        }
    )

    # -- Phase 3: Research (parallel) --
    # Research ALL entities concurrently using asyncio.gather.
    # Each entity type gets its own researcher, and all run in parallel
    # for speed.  Failures in one researcher don't block the others
    # (return_exceptions=True captures them instead of raising).
    _logger.info("pipeline_phase", phase="RESEARCH", session=session_id)
    state = state.model_copy(update={"current_phase": PipelinePhase.RESEARCH})

    research_tasks: list[asyncio.Task[ResearchResult]] = []
    for artist_entity in extracted.artists:
        research_tasks.append(
            asyncio.ensure_future(
                services["artist_researcher"].research(artist_entity.text)
            )
        )
    if extracted.venue:
        research_tasks.append(
            asyncio.ensure_future(
                services["venue_researcher"].research(extracted.venue.text)
            )
        )
    if extracted.promoter:
        research_tasks.append(
            asyncio.ensure_future(
                services["promoter_researcher"].research(extracted.promoter.text)
            )
        )
    if extracted.date:
        research_tasks.append(
            asyncio.ensure_future(
                services["date_context_researcher"].research(extracted.date.text)
            )
        )

    # asyncio.gather runs all tasks concurrently and returns results in order.
    # return_exceptions=True means exceptions are returned as values (not raised).
    raw_research = await asyncio.gather(*research_tasks, return_exceptions=True)
    research_results: list[ResearchResult] = []
    for raw in raw_research:
        if isinstance(raw, Exception):
            _logger.error("CLI research task failed", error=str(raw))
        else:
            research_results.append(raw)

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
# Auto-ingest corpus on startup (multi-phase, priority-ordered)
# ---------------------------------------------------------------------------
#
# Ingests content from three source directories in priority order:
#   Phase 1: RA Exchange interviews  (tier 1, highest search value)
#   Phase 2: Books                   (tier 1, user-provided .txt files)
#   Phase 3: Reference corpus        (tier 5 refs / tier 3 events)
#
# Each phase is memory-aware — the governor inside ingest_directory
# pauses ingestion before OOM.  On restart, skip_source_ids ensures
# already-ingested files are skipped and the next phase picks up.

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REFERENCE_CORPUS_DIR = _PROJECT_ROOT / "data" / "reference_corpus"
_INTERVIEW_DIR = _PROJECT_ROOT / "transcripts" / "ra_exchange"
_BOOKS_DIR = _PROJECT_ROOT / "data" / "books"

# Source types that participate in auto-ingest.  The set is used to
# build a comprehensive skip list so files are never re-ingested.
_INGEST_SOURCE_TYPES = ("reference", "event", "interview", "book")


async def _auto_ingest_reference_corpus(application: FastAPI) -> None:
    """Incrementally ingest corpus files into the RAG vector store.

    Runs on every startup but only processes NEW files.  Uses the vector
    store's ``get_source_ids`` method to find which sources are already
    ingested, then passes that set to ``ingest_directory`` so already-
    ingested files are skipped (no wasted embedding calls).

    Three source directories are processed in priority order:

    1. **RA Exchange interviews** (``transcripts/ra_exchange/``)
       489 long-form artist interview transcripts.  Tier 1 (highest
       authority).  These power the T1 tier in sidebar search and
       are the most valuable content for artist-specific queries.

    2. **Books** (``data/books/``)
       User-provided ``.txt`` files of electronic music books.  Tier 1.
       Place plain-text book files here and they'll be auto-ingested.

    3. **Reference corpus** (``data/reference_corpus/``)
       Curated knowledge files, RBMA lectures, and RA event listings.
       Knowledge files get tier 5 (reference); event files get tier 3.

    The memory governor inside ``ingest_directory`` pauses ingestion
    before hitting the 512 MB container limit.  Each restart cycle
    picks up where the previous one left off via ``skip_source_ids``.
    """
    ingestion_service = getattr(application.state, "ingestion_service", None)
    vector_store = getattr(application.state, "vector_store", None)
    if ingestion_service is None or vector_store is None:
        return  # RAG not enabled

    # Build a comprehensive skip list from ALL source types so files
    # ingested in previous cycles (possibly from a different phase)
    # are never re-processed.
    existing_ids: set[str] = set()
    try:
        for st in _INGEST_SOURCE_TYPES:
            existing_ids |= await vector_store.get_source_ids(source_type=st)
    except Exception:
        _logger.debug("get_source_ids_failed_proceeding_with_full_ingest")

    _logger.info(
        "corpus_ingest_check",
        existing_sources=len(existing_ids),
    )

    # Guard: skip ingestion if the corpus already has enough sources.
    # The memory governor provides the real OOM protection; this is a
    # secondary safeguard.  Threshold accommodates ~489 interviews +
    # ~34 reference files + books.
    _MIN_EXISTING_SOURCES = 550
    if len(existing_ids) >= _MIN_EXISTING_SOURCES:
        _logger.info(
            "corpus_sufficient",
            existing_sources=len(existing_ids),
            threshold=_MIN_EXISTING_SOURCES,
            message="Corpus fully ingested; skipping",
        )
        try:
            final_stats = await vector_store.get_stats()
            _logger.info(
                "corpus_readiness",
                total_chunks=final_stats.total_chunks,
                total_sources=final_stats.total_sources,
                ready=final_stats.total_chunks > 0,
            )
        except Exception as exc:
            _logger.error("corpus_readiness_check_failed", error=str(exc))
        return

    total_new_chunks = 0

    # --- Phase 1: RA Exchange interviews (highest priority) ---
    # These are tier 1 content — long-form artist interviews that
    # power the T1 tier in sidebar search.  Ingested before reference
    # corpus so they get memory priority over the large event files.
    if _INTERVIEW_DIR.is_dir():
        try:
            _logger.info("corpus_phase_start", phase=1, source="interviews")
            results = await ingestion_service.ingest_directory(
                str(_INTERVIEW_DIR),
                source_type="interview",
                default_tier=1,
                skip_source_ids=existing_ids if existing_ids else None,
                skip_tagging=True,
            )
            phase_chunks = sum(r.chunks_created for r in results)
            total_new_chunks += phase_chunks
            if results:
                _logger.info(
                    "corpus_phase_complete",
                    phase=1,
                    new_files=len(results),
                    new_chunks=phase_chunks,
                )
        except Exception as exc:
            _logger.error("corpus_interview_ingestion_failed", error=str(exc))

    # --- Phase 2: Books (user-provided .txt files) ---
    # Place plain-text book files in data/books/ and they'll be
    # auto-ingested as tier 1 book sources.
    if _BOOKS_DIR.is_dir():
        try:
            _logger.info("corpus_phase_start", phase=2, source="books")
            results = await ingestion_service.ingest_directory(
                str(_BOOKS_DIR),
                source_type="book",
                default_tier=1,
                skip_source_ids=existing_ids if existing_ids else None,
                skip_tagging=True,
            )
            phase_chunks = sum(r.chunks_created for r in results)
            total_new_chunks += phase_chunks
            if results:
                _logger.info(
                    "corpus_phase_complete",
                    phase=2,
                    new_files=len(results),
                    new_chunks=phase_chunks,
                )
        except Exception as exc:
            _logger.error("corpus_book_ingestion_failed", error=str(exc))

    # --- Phase 3: Reference corpus (knowledge files + events) ---
    # Curated text files and RA event listings.  Event files are the
    # largest (up to 20 MB each) and lowest priority — the memory
    # governor typically pauses during this phase.
    if _REFERENCE_CORPUS_DIR.is_dir():
        try:
            _logger.info("corpus_phase_start", phase=3, source="reference")
            results = await ingestion_service.ingest_directory(
                str(_REFERENCE_CORPUS_DIR),
                source_type="reference",
                skip_source_ids=existing_ids if existing_ids else None,
                skip_tagging=True,
            )
            phase_chunks = sum(r.chunks_created for r in results)
            total_new_chunks += phase_chunks
            if results:
                _logger.info(
                    "corpus_phase_complete",
                    phase=3,
                    new_files=len(results),
                    new_chunks=phase_chunks,
                )
        except Exception as exc:
            _logger.error("corpus_reference_ingestion_failed", error=str(exc))

    if total_new_chunks:
        _logger.info("corpus_ingestion_total", new_chunks=total_new_chunks)
    else:
        _logger.info("corpus_up_to_date")

    # Always log final corpus state for operational visibility
    try:
        final_stats = await vector_store.get_stats()
        _logger.info(
            "corpus_readiness",
            total_chunks=final_stats.total_chunks,
            total_sources=final_stats.total_sources,
            ready=final_stats.total_chunks > 0,
        )
    except Exception as exc:
        _logger.error("corpus_readiness_check_failed", error=str(exc))


# ---------------------------------------------------------------------------
# Application lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(application: FastAPI):  # noqa: ANN201
    """Initialise all providers and services on startup, clean up on shutdown.

    # JUNIOR DEV NOTE — FastAPI Lifespan
    # -----------------------------------
    # This is an async context manager that FastAPI calls once:
    #   STARTUP:  everything before `yield` runs when the server starts
    #   SHUTDOWN: everything after `yield` runs when the server stops
    #
    # The `yield` line is the boundary — the server is running between
    # startup and shutdown.  This replaces the older @app.on_event("startup")
    # / @app.on_event("shutdown") pattern.
    """
    # --- STARTUP ---
    # Build the entire dependency graph and attach to app.state.
    components = await _build_all(settings)

    # Attach every component to app.state so route handlers can access
    # them via FastAPI's Depends() mechanism (see routes.py).
    for key, value in components.items():
        setattr(application.state, key, value)

    # Initialize feedback database (creates table if needed)
    if hasattr(application.state, "feedback_provider"):
        await application.state.feedback_provider.initialize()

    # Schedule corpus ingestion as a background task so it doesn't block
    # the health check.  Large corpus files (12 city event files, ~72MB)
    # can take many minutes to chunk, tag, and embed — blocking startup
    # causes Render's health check to fail and the container to restart
    # in an infinite loop.  The app starts serving immediately with
    # whatever corpus is already in the vector store; new files are
    # ingested in the background.
    application.state._ingestion_task = asyncio.create_task(
        _auto_ingest_reference_corpus(application),
        name="auto_ingest_reference_corpus",
    )

    _logger.info(
        "app_startup",
        version="0.1.0",
        environment=settings.app_env,
        primary_llm=components["primary_llm_name"],
        providers=len(components["provider_list"]),
    )

    yield

    # -- Shutdown --
    # Cancel background ingestion if still running
    ingestion_task = getattr(application.state, "_ingestion_task", None)
    if ingestion_task and not ingestion_task.done():
        ingestion_task.cancel()
        _logger.info("app_shutdown", message="Cancelled background ingestion task")

    http_client: httpx.AsyncClient = components["http_client"]
    await http_client.aclose()
    _logger.info("app_shutdown", message="HTTP client closed")


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Build and configure the FastAPI application.

    This is the **application factory** — it creates a new FastAPI instance,
    attaches middleware, routes, WebSocket handlers, and static file mounts.
    Called once at module load to create the ``app`` singleton below.
    """
    application = FastAPI(
        title="raiveFlier API",
        version="0.1.0",
        description=(
            "Upload a rave flier image, extract artist/venue/promoter information "
            "via OCR and LLM analysis, then deep-research every entity with "
            "citations from music databases, web search, and article scraping."
        ),
        lifespan=_lifespan,  # ← hooks _build_all at startup, cleanup at shutdown
    )

    # -- Middleware (order matters: last added = first executed) --
    # Starlette processes middleware as a stack (LIFO), so:
    #   Request  → RequestLogging → ErrorHandling → route handler
    #   Response ← RequestLogging ← ErrorHandling ← route handler
    application.add_middleware(ErrorHandlingMiddleware)
    application.add_middleware(RequestLoggingMiddleware)

    # In production, restrict CORS to the deployed domain only.
    # In development, allow all origins for local testing.
    if settings.app_env == "production":
        configure_cors(application, allowed_origins=[
            "https://raiveflier.onrender.com",
        ])
    else:
        configure_cors(application)

    # -- API routes --
    # All REST endpoints are defined in src/api/routes.py under /api/v1.
    application.include_router(api_router)

    # -- WebSocket --
    # Real-time progress updates pushed to the frontend during analysis.
    # The frontend opens a WebSocket to /ws/progress/{session_id} and
    # receives JSON messages with phase/progress/message on each update.
    @application.websocket("/ws/progress/{session_id}")
    async def ws_progress(websocket: WebSocket, session_id: str) -> None:
        await websocket_progress(websocket, session_id)

    # -- raiveFeeder sub-app --
    # Mount the raiveFeeder companion app at /feeder/ so both apps share
    # the same Uvicorn process, port, and ChromaDB data.  This must come
    # BEFORE the catch-all static file mounts below.
    try:
        from tools.raive_feeder.main import create_app as create_feeder_app
        application.mount("/feeder", create_feeder_app())
        _logger.info("raive_feeder_mounted", path="/feeder")
    except Exception as exc:
        _logger.warning("raive_feeder_mount_failed", error=str(exc))

    # -- Rave Stories sub-app --
    # Mount the Rave Stories companion app at /stories/ so it shares
    # the same Uvicorn process, LLM providers, and ChromaDB data.
    try:
        from src.stories.main import create_stories_app
        story_components = getattr(application.state, "story_components", None)
        application.mount("/stories", create_stories_app(components=story_components))
        _logger.info("rave_stories_mounted", path="/stories")
    except Exception as exc:
        _logger.warning("rave_stories_mount_failed", error=str(exc))

    # -- Frontend static files --
    # Mount the vanilla JS/CSS/HTML frontend as static files.
    # Each subdirectory (css/, js/, assets/) is mounted separately so
    # FastAPI can serve them at the matching URL path.
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
        if (_FRONTEND_DIR / "assets").exists():
            application.mount(
                "/assets",
                StaticFiles(directory=str(_FRONTEND_DIR / "assets")),
                name="assets",
            )

        # Serve the SPA entry point at the root URL.
        # include_in_schema=False hides this from the OpenAPI docs.
        @application.get("/", include_in_schema=False)
        async def serve_index() -> FileResponse:
            return FileResponse(str(_FRONTEND_DIR / "index.html"))

    return application


# Create the singleton FastAPI app at module load time.
# Uvicorn references this as "src.main:app".
app = create_app()

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

# When run directly (python -m src.main), start the Uvicorn dev server.
# In production, Uvicorn is started by the entrypoint script instead.
if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.app_host,
        port=settings.app_port,
        # Hot-reload enabled in dev — Uvicorn watches for file changes
        # and restarts the server automatically.
        reload=(settings.app_env == "development"),
    )

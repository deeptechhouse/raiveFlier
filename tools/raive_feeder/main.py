"""raiveFeeder FastAPI application entry point.

# ─── HOW THIS FILE WORKS (Junior Developer Guide) ─────────────────────
#
# This is the composition root for raiveFeeder — the single place where
# every provider, service, and route component is created and wired via
# dependency injection (DI).  The pattern mirrors src/main.py exactly.
#
# Key concepts:
#   1. **Provider selection** — Reuses the same fallback chains as
#      raiveFlier (embedding, LLM, OCR, vector store) so both apps
#      talk to the same ChromaDB corpus with compatible embeddings.
#   2. **Lifespan context** — FastAPI's lifespan hook runs at startup
#      (wires everything) and shutdown (cleans up).
#   3. **app.state** — All singletons stored on app.state for access
#      via FastAPI Depends() in route handlers.
#   4. **Separate port** — raiveFeeder runs on :8001 while raiveFlier
#      runs on :8000.  Both can run simultaneously.
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from tools.raive_feeder.config.settings import FeederSettings

logger = structlog.get_logger(logger_name=__name__)

# ─── Directory paths ───────────────────────────────────────────────────
# Resolve relative to this file so paths work regardless of cwd.
_THIS_DIR = Path(__file__).resolve().parent
_FRONTEND_DIR = _THIS_DIR / "frontend"


# ─── Provider factories ───────────────────────────────────────────────
# These mirror src/cli/ingest.py's factories to ensure the same provider
# selection and embedding dimensions.  raiveFeeder must produce embeddings
# compatible with raiveFlier's existing ChromaDB corpus.

def _build_embedding_provider(settings: FeederSettings) -> Any | None:
    """Select the first available embedding provider (same chain as raiveFlier)."""
    if settings.openai_api_key:
        from src.providers.embedding.openai_embedding_provider import (
            OpenAIEmbeddingProvider,
        )
        provider = OpenAIEmbeddingProvider(settings=settings)
        if provider.is_available():
            return provider

    from src.providers.embedding.fastembed_embedding_provider import (
        FastEmbedEmbeddingProvider,
    )
    fe_provider = FastEmbedEmbeddingProvider()
    if fe_provider.is_available():
        return fe_provider

    from src.providers.embedding.sentence_transformer_embedding_provider import (
        SentenceTransformerEmbeddingProvider,
    )
    st_provider = SentenceTransformerEmbeddingProvider()
    if st_provider.is_available():
        return st_provider

    from src.providers.embedding.nomic_embedding_provider import (
        NomicEmbeddingProvider,
    )
    provider = NomicEmbeddingProvider(settings=settings)
    if provider.is_available():
        return provider

    return None


def _build_llm_provider(settings: FeederSettings) -> Any | None:
    """Select the first available LLM provider for metadata extraction."""
    if settings.anthropic_api_key:
        from src.providers.llm.anthropic_provider import AnthropicLLMProvider
        return AnthropicLLMProvider(settings=settings)
    if settings.openai_api_key:
        from src.providers.llm.openai_provider import OpenAILLMProvider
        return OpenAILLMProvider(settings=settings)
    from src.providers.llm.ollama_provider import OllamaLLMProvider
    return OllamaLLMProvider(settings=settings)


def _build_ocr_providers(settings: FeederSettings) -> list[Any]:
    """Build the OCR provider fallback chain for image ingestion."""
    providers: list[Any] = []
    # LLM Vision OCR — best for stylized flier text.
    llm = _build_llm_provider(settings)
    if llm is not None:
        try:
            from src.providers.ocr.llm_vision_ocr_provider import LLMVisionOCRProvider
            providers.append(LLMVisionOCRProvider(llm_provider=llm))
        except Exception:
            pass
    # Tesseract — good for clean printed text (book/magazine scans).
    try:
        from src.providers.ocr.tesseract_provider import TesseractOCRProvider
        prov = TesseractOCRProvider()
        if prov.is_available():
            providers.append(prov)
    except Exception:
        pass
    return providers


def _build_ingestion_service(settings: FeederSettings) -> tuple[Any | None, str]:
    """Construct the full ingestion pipeline (mirrors src/cli/ingest.py).

    Returns the IngestionService and a status string, or (None, error_message).
    """
    embedding_provider = _build_embedding_provider(settings)
    if embedding_provider is None:
        return None, "No embedding provider available"

    from src.providers.vector_store.chromadb_provider import ChromaDBProvider
    from src.services.ingestion.chunker import TextChunker
    from src.services.ingestion.ingestion_service import IngestionService
    from src.services.ingestion.metadata_extractor import MetadataExtractor

    llm_provider = _build_llm_provider(settings)
    vector_store = ChromaDBProvider(
        embedding_provider=embedding_provider,
        persist_directory=settings.chromadb_persist_dir,
        collection_name=settings.chromadb_collection,
    )
    chunker = TextChunker()
    metadata_extractor = MetadataExtractor(llm=llm_provider)

    # Article scraper for URL ingestion — uses trafilatura under the hood.
    article_scraper = None
    try:
        import httpx
        from src.providers.article.web_scraper_provider import WebScraperProvider
        http_client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        article_scraper = WebScraperProvider(http_client=http_client)
    except Exception:
        pass

    service = IngestionService(
        chunker=chunker,
        metadata_extractor=metadata_extractor,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        article_scraper=article_scraper,
    )

    provider_name = embedding_provider.get_provider_name()
    return service, f"Embedding: {provider_name} | Store: chromadb"


def _build_all(settings: FeederSettings) -> dict[str, Any]:
    """Construct the entire dependency graph for raiveFeeder.

    This is the composition root — all providers and services are created
    here and injected into each other.  No service creates its own deps.
    """
    components: dict[str, Any] = {"settings": settings}

    # Ingestion service (shared with raiveFlier's pipeline).
    ingestion_service, status = _build_ingestion_service(settings)
    components["ingestion_service"] = ingestion_service
    components["ingestion_status"] = status

    # Embedding provider (needed independently for corpus management).
    components["embedding_provider"] = _build_embedding_provider(settings)

    # LLM provider (for web crawler relevance scoring and metadata tagging).
    components["llm_provider"] = _build_llm_provider(settings)

    # OCR providers (for image ingestion tab).
    components["ocr_providers"] = _build_ocr_providers(settings)

    # Vector store (for corpus management operations).
    if components["embedding_provider"] is not None:
        from src.providers.vector_store.chromadb_provider import ChromaDBProvider
        components["vector_store"] = ChromaDBProvider(
            embedding_provider=components["embedding_provider"],
            persist_directory=settings.chromadb_persist_dir,
            collection_name=settings.chromadb_collection,
        )
    else:
        components["vector_store"] = None

    # Batch processor — manages concurrent ingestion jobs.
    from tools.raive_feeder.services.batch_processor import BatchProcessor
    components["batch_processor"] = BatchProcessor(
        ingestion_service=ingestion_service,
        max_concurrency=settings.batch_max_concurrency,
    )

    logger.info(
        "feeder_components_built",
        ingestion=status,
        has_llm=components["llm_provider"] is not None,
        ocr_count=len(components["ocr_providers"]),
        has_vector_store=components["vector_store"] is not None,
    )
    return components


# ─── FastAPI lifespan ──────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup: build dependency graph.  Shutdown: clean up."""
    settings: FeederSettings = app.state.settings
    components = _build_all(settings)

    # Attach all singletons to app.state for route handler access.
    for key, value in components.items():
        setattr(app.state, key, value)

    logger.info(
        "raive_feeder_started",
        port=settings.feeder_port,
        ingestion_status=components["ingestion_status"],
    )

    yield

    # Shutdown cleanup.
    batch_processor = getattr(app.state, "batch_processor", None)
    if batch_processor is not None:
        await batch_processor.shutdown()
    logger.info("raive_feeder_shutdown")


# ─── App factory ───────────────────────────────────────────────────────

def create_app(settings: FeederSettings | None = None) -> FastAPI:
    """Create and configure the raiveFeeder FastAPI application.

    Parameters
    ----------
    settings:
        Optional pre-built settings.  If None, settings are loaded from
        environment variables / .env file.
    """
    if settings is None:
        settings = FeederSettings()

    app = FastAPI(
        title="raiveFeeder",
        description="Corpus ingestion & database management GUI for raiveFlier",
        version="0.1.0",
        lifespan=_lifespan,
    )

    # Store settings before lifespan runs so _lifespan can read them.
    app.state.settings = settings

    # CORS — allow the frontend (served from same origin) and raiveFlier.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routes.
    from tools.raive_feeder.api.routes import router as api_router
    app.include_router(api_router, prefix="/api/v1")

    # Mount WebSocket endpoint.
    from tools.raive_feeder.api.websocket import websocket_progress
    app.add_api_websocket_route("/ws/progress/{job_id}", websocket_progress)

    # Mount static frontend files.
    if _FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")

    return app


# ─── Entry point ───────────────────────────────────────────────────────

def main() -> None:
    """Launch raiveFeeder on the configured port (default 8001)."""
    settings = FeederSettings()
    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.feeder_host,
        port=settings.feeder_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()

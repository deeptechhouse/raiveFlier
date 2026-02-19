# raiveFlier — Navigation Map

> Upload a rave flier image, extract artist/venue/promoter entities via OCR and LLM analysis, then deep-research every entity with citations from music databases, web search, and article scraping.

## Architecture Overview

raiveFlier follows a **layered architecture with adapter pattern** for all external service integrations. The system is organized into five distinct layers: data models, abstract interfaces, concrete provider implementations, business-logic services, and a pipeline orchestrator that coordinates the 5-phase analysis workflow. All external dependencies (LLMs, music databases, search engines, OCR engines) are abstracted behind interfaces so providers can be swapped without changing business logic.

## Directory Map

| Directory | Layer/Role | Description |
|---|---|---|
| `src/` | Application Root | Python package root; contains `main.py` application entry point |
| `src/models/` | Data Layer | Pydantic data models for fliers, entities, pipeline state, research results, and RAG |
| `src/interfaces/` | Abstraction Layer | Abstract base classes (interfaces) for all provider types — ensures swappability |
| `src/providers/` | Integration Layer | Concrete implementations of each interface — grouped by capability |
| `src/providers/ocr/` | Integration | OCR providers: LLM Vision, EasyOCR, Tesseract |
| `src/providers/llm/` | Integration | LLM providers: Anthropic, OpenAI, Ollama |
| `src/providers/music_db/` | Integration | Music database providers: Discogs API, Discogs Scrape, MusicBrainz |
| `src/providers/search/` | Integration | Web search providers: DuckDuckGo |
| `src/providers/article/` | Integration | Article/content providers: Web Scraper, Wayback Machine |
| `src/providers/cache/` | Integration | Cache providers: In-memory cache |
| `src/providers/embedding/` | Integration | Embedding providers: OpenAI Embeddings, Nomic (Ollama) |
| `src/providers/vector_store/` | Integration | Vector store providers: ChromaDB |
| `src/services/` | Business Logic | Core research services — artist, venue, promoter, date context researchers |
| `src/services/ingestion/` | Business Logic | RAG ingestion pipeline — chunking, metadata extraction, source processing |
| `src/services/ingestion/source_processors/` | Business Logic | Source-specific processors for articles, books, and analysis documents |
| `src/pipeline/` | Orchestration | Pipeline orchestrator, confirmation gate (human-in-the-loop), progress tracker |
| `src/api/` | Interface (HTTP) | FastAPI routes, request/response schemas, middleware, WebSocket handler |
| `src/cli/` | Interface (CLI) | CLI entry point for RAG corpus ingestion |
| `src/config/` | Configuration | YAML config loader and Pydantic settings (env vars) |
| `src/utils/` | Cross-Cutting | Shared utilities — logging, error types, image preprocessing, text normalization |
| `frontend/` | Presentation | Single-page HTML/CSS/JS frontend with drag-and-drop upload |
| `frontend/css/` | Presentation | Bunker + Amethyst design system stylesheet |
| `frontend/js/` | Presentation | Client-side modules — upload, entity confirmation, WebSocket progress |
| `config/` | Configuration | Runtime YAML configs — app settings, logging |
| `tests/` | Testing | pytest test suite — unit, integration, E2E, fixtures |
| `tests/unit/` | Testing | Unit tests for individual functions and classes |
| `tests/integration/` | Testing | Integration tests for API endpoints and service interactions |
| `tests/e2e/` | Testing | End-to-end tests for full pipeline flows |
| `tests/fixtures/` | Testing | Test fixtures — mock responses and sample flier images |

## Key Entry Points

- **Application start:** `src/main.py` — FastAPI app factory, dependency injection, lifespan management
- **API routes:** `src/api/routes.py` — all REST endpoint definitions (`/api/v1/...`)
- **WebSocket:** `src/api/websocket.py` — real-time progress updates via `/ws/progress/{session_id}`
- **Pipeline orchestrator:** `src/pipeline/orchestrator.py` — coordinates the 5-phase analysis
- **CLI ingestion:** `src/cli/ingest.py` — `python -m src.cli.ingest` for RAG corpus loading
- **App config:** `config/config.yaml` — provider priorities, rate limits, cache TTL
- **Env config:** `.env` — API keys, RAG settings, app host/port

## Module Relationships

- **API layer** (`src/api/`) calls **Pipeline** (`src/pipeline/`), never accesses providers directly
- **Pipeline orchestrator** coordinates **Services** (`src/services/`) through a 5-phase workflow
- **Services** depend on **Interfaces** (`src/interfaces/`) — never on concrete providers
- **Providers** (`src/providers/`) implement **Interfaces** and handle all external I/O
- **Models** (`src/models/`) are pure data containers used across all layers
- **Config** (`src/config/`) is injected at startup; providers receive settings via constructor
- **Utils** (`src/utils/`) are shared helpers available to all layers
- **Frontend** communicates with the API via REST + WebSocket; no direct service access

## External Dependencies

| Dependency | Purpose | Swappable? |
|---|---|---|
| OpenAI API | LLM text analysis, vision OCR, embeddings | Yes (`ILLMProvider` / `IEmbeddingProvider` interface) |
| Anthropic API | LLM text analysis, vision OCR | Yes (`ILLMProvider` interface) |
| Ollama | Local LLM + Nomic embeddings (free, no API key) | Yes (`ILLMProvider` / `IEmbeddingProvider` interface) |
| Tesseract | Local OCR engine | Yes (`IOCRProvider` interface) |
| EasyOCR | Local OCR engine (ML-based) | Yes (`IOCRProvider` interface) |
| Discogs API | Music database lookups | Yes (`IMusicDBProvider` interface) |
| MusicBrainz | Music database lookups (free, no key) | Yes (`IMusicDBProvider` interface) |
| DuckDuckGo | Web search (free, no key) | Yes (`IWebSearchProvider` interface) |
| Serper | Web search (paid, optional) | Yes (`IWebSearchProvider` interface) |
| ChromaDB | Vector store for RAG | Yes (`IVectorStoreProvider` interface) |
| Wayback Machine | Article archival retrieval | Yes (`IArticleProvider` interface) |
| structlog | Structured JSON logging | Yes (standard logging adapter) |

"""Public interface definitions for all external service providers.

Every external API or service in the raiveFlier pipeline is accessed
exclusively through the abstract base classes defined in this package.
Concrete adapters implement these interfaces and are injected at runtime,
following the adapter pattern (CLAUDE.md Section 6).

ADAPTER PATTERN EXPLAINED (for junior developers):
    The adapter pattern decouples business logic from specific external services.
    Instead of calling ``openai.chat.completions.create(...)`` directly in your
    code, you call ``llm_provider.complete(...)`` where ``llm_provider`` is any
    object implementing ``ILLMProvider``. This means:
        - Swapping OpenAI for Anthropic requires changing ONE line (the provider
          instantiation in main.py) instead of every file that uses LLMs.
        - Unit tests can inject a mock/fake provider without real API calls.
        - Multiple providers can be tried in priority order (fallback chains).

    The concrete providers live in ``src/providers/`` and are registered in
    ``src/main.py`` during application startup via FastAPI's dependency injection.

CONCRETE PROVIDER MAP:
    Interface                  →  Concrete implementations (in src/providers/)
    ─────────────────────────────────────────────────────────────────────
    ILLMProvider               →  OpenAILLMProvider, AnthropicLLMProvider,
                                  OllamaLLMProvider
    IOCRProvider               →  LLMVisionOCRProvider, EasyOCRProvider,
                                  TesseractOCRProvider
    IMusicDatabaseProvider     →  DiscogsAPIProvider, DiscogsScrapeProvider,
                                  MusicBrainzProvider, BandcampProvider,
                                  BeatportProvider
    IWebSearchProvider         →  DuckDuckGoSearchProvider
    IArticleProvider           →  WebScraperProvider, WaybackProvider
    IEmbeddingProvider         →  FastEmbedEmbeddingProvider,
                                  OpenAIEmbeddingProvider,
                                  SentenceTransformerEmbeddingProvider,
                                  NomicEmbeddingProvider
    IVectorStoreProvider       →  ChromaDBProvider
    ICacheProvider             →  MemoryCacheProvider
    IFeedbackProvider          →  SQLiteFeedbackProvider
    IFlierHistoryProvider      →  SQLiteFlierHistoryProvider

Re-exports
----------
IOCRProvider
    OCR text-extraction contract.
ILLMProvider
    LLM completion and vision-analysis contract.
IMusicDatabaseProvider, ArtistSearchResult
    Music-database query contract and helper dataclass.
IWebSearchProvider, SearchResult
    Web-search contract and helper dataclass.
IArticleProvider, ArticleContent
    Article-extraction contract and helper dataclass.
ICacheProvider
    Key-value cache contract.
IVectorStoreProvider
    Vector-store storage and semantic retrieval contract.
IEmbeddingProvider
    Text-embedding generation contract.
IFeedbackProvider
    User-feedback persistence contract (thumbs up/down ratings).
"""

from src.interfaces.article_provider import ArticleContent, IArticleProvider
from src.interfaces.cache_provider import ICacheProvider
from src.interfaces.embedding_provider import IEmbeddingProvider
from src.interfaces.feedback_provider import IFeedbackProvider
from src.interfaces.llm_provider import ILLMProvider
from src.interfaces.music_db_provider import ArtistSearchResult, IMusicDatabaseProvider
from src.interfaces.ocr_provider import IOCRProvider
from src.interfaces.vector_store_provider import IVectorStoreProvider
from src.interfaces.web_search_provider import IWebSearchProvider, SearchResult

__all__ = [
    "ArticleContent",
    "ArtistSearchResult",
    "IArticleProvider",
    "ICacheProvider",
    "IEmbeddingProvider",
    "IFeedbackProvider",
    "ILLMProvider",
    "IMusicDatabaseProvider",
    "IOCRProvider",
    "IVectorStoreProvider",
    "IWebSearchProvider",
    "SearchResult",
]

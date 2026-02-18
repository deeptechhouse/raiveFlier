"""Public interface definitions for all external service providers.

Every external API or service in the raiveFlier pipeline is accessed
exclusively through the abstract base classes defined in this package.
Concrete adapters implement these interfaces and are injected at runtime,
following the adapter pattern (CLAUDE.md Section 6).

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
"""

from src.interfaces.article_provider import ArticleContent, IArticleProvider
from src.interfaces.cache_provider import ICacheProvider
from src.interfaces.llm_provider import ILLMProvider
from src.interfaces.music_db_provider import ArtistSearchResult, IMusicDatabaseProvider
from src.interfaces.ocr_provider import IOCRProvider
from src.interfaces.web_search_provider import IWebSearchProvider, SearchResult

__all__ = [
    "ArticleContent",
    "ArtistSearchResult",
    "IArticleProvider",
    "ICacheProvider",
    "ILLMProvider",
    "IMusicDatabaseProvider",
    "IOCRProvider",
    "IWebSearchProvider",
    "SearchResult",
]

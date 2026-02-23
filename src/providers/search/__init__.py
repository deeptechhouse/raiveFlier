"""Web-search provider implementations.

Currently only DuckDuckGo (free, no API key). A Serper/Google provider
could be added here as a higher-quality paid alternative — the interface
(IWebSearchProvider) is already defined and the research services would
use the new provider transparently via dependency injection.
"""

from src.providers.search.duckduckgo_provider import DuckDuckGoSearchProvider

__all__ = ["DuckDuckGoSearchProvider"]

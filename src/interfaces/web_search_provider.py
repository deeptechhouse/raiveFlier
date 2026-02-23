"""Abstract base class for web-search service providers.

Defines the contract for general-purpose web searches used during entity
research.  Implementations may wrap Google Custom Search, Bing, Brave Search,
SearXNG, or any other search engine API.  The adapter pattern (CLAUDE.md
Section 6) keeps the research layer provider-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date


# frozen=True makes dataclass instances immutable (like Pydantic's frozen=True).
# We use a plain dataclass here instead of a Pydantic model because SearchResult
# is a simple value object that doesn't need Pydantic's validation overhead.
@dataclass(frozen=True)
class SearchResult:
    """A single web-search result.

    Attributes
    ----------
    title:
        The page title as returned by the search engine.
    url:
        The canonical URL of the result page.
    snippet:
        An optional text excerpt/description from the result.
    date:
        An optional publication or last-modified date if the search
        engine provides one.
    """

    title: str
    url: str
    snippet: str | None = None
    date: date | None = None


# Concrete implementation: DuckDuckGoSearchProvider (src/providers/search/)
# DuckDuckGo is free and requires no API key. A paid alternative (Serper/Google)
# could be added as a second implementation for production use.
class IWebSearchProvider(ABC):
    """Contract for web-search services used during entity research.

    Implementations return a list of :class:`SearchResult` items for a
    given query string, optionally filtered by date.
    """

    @abstractmethod
    async def search(
        self,
        query: str,
        num_results: int = 10,
        before_date: date | None = None,
    ) -> list[SearchResult]:
        """Execute a web search and return the top results.

        Parameters
        ----------
        query:
            The search query string.
        num_results:
            Maximum number of results to return.
        before_date:
            If supplied, only return results published before this date.
            Not all providers support date filtering — those that do not
            should silently ignore this parameter and return unfiltered
            results.

        Returns
        -------
        list[SearchResult]
            Zero or more results ordered by relevance.

        Raises
        ------
        src.core.errors.ResearchError
            If the search API call fails.
        """

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return a human-readable identifier for this search provider.

        Example return values: ``"google"``, ``"brave"``, ``"searxng"``.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if the provider is configured and reachable.

        Implementations should verify that API keys are present and any
        required network endpoints are accessible.
        """

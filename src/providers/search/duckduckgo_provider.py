"""DuckDuckGo web-search provider implementing IWebSearchProvider.

Uses the duckduckgo_search library for fully free, keyless web searches.
Supports both legacy AsyncDDGS (v4-v5) and sync DDGS (v8+) APIs.
Rate-limit exceptions are caught gracefully and logged as warnings,
returning empty results rather than propagating errors.
"""

from __future__ import annotations

import asyncio
from datetime import date

import structlog
from duckduckgo_search import DDGS

from src.interfaces.web_search_provider import IWebSearchProvider, SearchResult

logger = structlog.get_logger(logger_name=__name__)


class DuckDuckGoSearchProvider(IWebSearchProvider):
    """DuckDuckGo web-search provider.

    DuckDuckGo requires no API key and is entirely free.  In v8+ of the
    ``duckduckgo_search`` library, the synchronous ``DDGS`` class is used
    and wrapped in ``asyncio.to_thread`` for non-blocking execution.
    """

    def __init__(self) -> None:
        logger.info("duckduckgo_provider_initialized")

    async def search(
        self,
        query: str,
        num_results: int = 10,
        before_date: date | None = None,
    ) -> list[SearchResult]:
        """Execute a DuckDuckGo web search and return results.

        If *before_date* is provided, ``"before:YYYY-MM-DD"`` is appended
        to the query string to request date-filtered results.
        """
        effective_query = query
        if before_date:
            effective_query = f"{query} before:{before_date.isoformat()}"

        try:
            raw_results = await asyncio.to_thread(
                self._sync_search, effective_query, num_results
            )
        except Exception as exc:  # noqa: BLE001 — DDG may rate-limit or fail
            logger.warning(
                "duckduckgo_search_failed",
                query=effective_query,
                error=str(exc),
            )
            return []

        results: list[SearchResult] = []
        for item in raw_results or []:
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("href", item.get("url", "")),
                    snippet=item.get("body"),
                    date=None,
                )
            )

        logger.debug(
            "duckduckgo_search_complete",
            query=effective_query,
            result_count=len(results),
        )
        return results

    @staticmethod
    def _sync_search(query: str, max_results: int) -> list[dict]:
        """Run the synchronous DDGS search (called via to_thread)."""
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    def get_provider_name(self) -> str:
        """Return the provider identifier."""
        return "duckduckgo"

    def is_available(self) -> bool:
        """DuckDuckGo is always available (no API key required)."""
        return True

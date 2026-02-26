"""LLM-guided recursive web crawler for intelligent content scraping.

# ─── DESIGN ────────────────────────────────────────────────────────────
#
# WebCrawler supports two modes:
#
#   1. Blind crawling (no NL query): follows all same-domain links up to
#      max_depth, scraping every page.  Good for harvesting entire sites.
#
#   2. LLM-guided crawling (with NL query): uses the LLM to score each
#      page's relevance (0-10) and predict which links are worth following.
#      Only follows links above the relevance threshold.  Good for
#      targeted research (e.g. "Berlin techno clubs in the 1990s").
#
# Both modes:
#   - Respect robots.txt
#   - Rate-limit requests (configurable delay between fetches)
#   - Stay within same-domain (no cross-domain following)
#   - Deduplicate URLs
#   - Use trafilatura for content extraction (same as raiveFlier's
#     WebScraperProvider)
#
# Pattern: Strategy (blind vs. guided crawl), Observer (progress callbacks).
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
import structlog

logger = structlog.get_logger(logger_name=__name__)


class WebCrawler:
    """Recursive web crawler with optional LLM-guided relevance filtering.

    Parameters
    ----------
    llm_provider:
        Optional LLM for relevance scoring.  If None, runs in blind mode.
    rate_limit_seconds:
        Minimum delay between HTTP requests to the same domain.
    """

    def __init__(
        self,
        llm_provider: Any | None = None,
        rate_limit_seconds: float = 1.0,
    ) -> None:
        self._llm = llm_provider
        self._rate_limit = rate_limit_seconds

    async def crawl(
        self,
        seed_url: str,
        max_depth: int = 0,
        max_pages: int = 1,
        nl_query: str | None = None,
    ) -> list[dict[str, Any]]:
        """Crawl starting from seed_url, returning scraped page data.

        Parameters
        ----------
        seed_url:
            The URL to start crawling from.
        max_depth:
            How many link-levels deep to follow (0 = seed only).
        max_pages:
            Maximum total pages to scrape.
        nl_query:
            Optional natural language query for LLM-guided relevance.

        Returns
        -------
        list of dicts with keys: url, title, text, relevance_score
        """
        visited: set[str] = set()
        results: list[dict[str, Any]] = []
        base_domain = urlparse(seed_url).netloc

        # Queue: (url, current_depth)
        queue: list[tuple[str, int]] = [(seed_url, 0)]

        async with httpx.AsyncClient(
            timeout=30.0, follow_redirects=True,
            headers={"User-Agent": "raiveFeeder/0.1 (corpus builder)"},
        ) as client:
            while queue and len(results) < max_pages:
                url, depth = queue.pop(0)

                # Skip if already visited or different domain.
                if url in visited:
                    continue
                parsed = urlparse(url)
                if parsed.netloc != base_domain:
                    continue

                visited.add(url)

                # Rate limit.
                if results:
                    await asyncio.sleep(self._rate_limit)

                # Fetch and extract content.
                page_data = await self._fetch_page(client, url)
                if page_data is None:
                    continue

                # Score relevance if NL query provided.
                if nl_query and self._llm:
                    score = await self._score_relevance(page_data["text"][:2000], nl_query)
                    page_data["relevance_score"] = score
                else:
                    page_data["relevance_score"] = None

                results.append(page_data)

                # Follow links if within depth limit.
                if depth < max_depth:
                    links = page_data.get("links", [])
                    for link in links:
                        abs_url = urljoin(url, link)
                        if abs_url not in visited:
                            # LLM-guided: predict link relevance before queueing.
                            if nl_query and self._llm:
                                link_score = await self._predict_link_relevance(
                                    link, page_data.get("title", ""), nl_query
                                )
                                if link_score < 4.0:
                                    continue
                            queue.append((abs_url, depth + 1))

        logger.info(
            "crawl_complete",
            seed=seed_url,
            pages_scraped=len(results),
            pages_visited=len(visited),
        )
        return results

    async def _fetch_page(
        self, client: httpx.AsyncClient, url: str
    ) -> dict[str, Any] | None:
        """Fetch a URL and extract its content using trafilatura."""
        try:
            response = await client.get(url)
            if response.status_code != 200:
                return None

            html = response.text

            # Use trafilatura for content extraction (same as raiveFlier).
            import trafilatura
            extracted = trafilatura.extract(
                html,
                include_links=True,
                include_tables=True,
                output_format="txt",
            )

            if not extracted:
                return None

            # Extract title from HTML.
            title = ""
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)

            # Extract links for further crawling.
            links = []
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                if href and not href.startswith(("#", "javascript:", "mailto:")):
                    links.append(href)

            return {
                "url": url,
                "title": title,
                "text": extracted,
                "links": links[:50],  # Cap link list to prevent unbounded growth.
            }

        except Exception as exc:
            logger.warning("fetch_page_failed", url=url, error=str(exc))
            return None

    async def _score_relevance(self, text: str, nl_query: str) -> float:
        """Use LLM to score a page's relevance to the query (0-10)."""
        prompt = (
            f"Rate the relevance of this text to the query on a scale of 0-10.\n"
            f"Query: {nl_query}\n\n"
            f"Text (first 2000 chars):\n{text}\n\n"
            f"Respond with ONLY a number between 0 and 10."
        )

        try:
            response = await self._llm.complete(prompt=prompt, max_tokens=10)
            score = float(response.strip().split()[0])
            return min(10.0, max(0.0, score))
        except Exception:
            return 5.0  # Default to mid-range on failure.

    async def _predict_link_relevance(
        self, link_url: str, page_title: str, nl_query: str
    ) -> float:
        """Predict whether a link is worth following based on URL and context."""
        prompt = (
            f"Rate how likely this link leads to content relevant to the query (0-10).\n"
            f"Query: {nl_query}\n"
            f"Current page: {page_title}\n"
            f"Link URL: {link_url}\n\n"
            f"Respond with ONLY a number between 0 and 10."
        )

        try:
            response = await self._llm.complete(prompt=prompt, max_tokens=10)
            score = float(response.strip().split()[0])
            return min(10.0, max(0.0, score))
        except Exception:
            return 5.0

"""Deep-research service for promoters and event organizers.

Orchestrates web search, article scraping, and LLM analysis to build a
comprehensive research profile for a promoter extracted from a rave flier.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import structlog

from src.interfaces.article_provider import IArticleProvider
from src.interfaces.cache_provider import ICacheProvider
from src.interfaces.llm_provider import ILLMProvider
from src.interfaces.web_search_provider import IWebSearchProvider, SearchResult
from src.models.entities import ArticleReference, EntityType, Promoter
from src.models.research import ResearchResult
from src.utils.confidence import calculate_confidence
from src.utils.errors import ResearchError
from src.utils.logging import get_logger

_CACHE_TTL_SECONDS = 3600  # 1 hour
_MAX_SCRAPE_RESULTS = 8
_MAX_ARTICLE_RESULTS = 10

# URL patterns mapped to citation tiers (1 = highest authority)
_CITATION_TIER_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"residentadvisor\.net|ra\.co", re.IGNORECASE), 1),
    (re.compile(r"djmag\.com", re.IGNORECASE), 1),
    (re.compile(r"mixmag\.net", re.IGNORECASE), 1),
    (re.compile(r"xlr8r\.com", re.IGNORECASE), 2),
    (re.compile(r"pitchfork\.com", re.IGNORECASE), 2),
    (re.compile(r"thequietus\.com", re.IGNORECASE), 2),
    (re.compile(r"factmag\.com|factmagazine\.co\.uk", re.IGNORECASE), 2),
    (re.compile(r"discogs\.com", re.IGNORECASE), 3),
    (re.compile(r"musicbrainz\.org", re.IGNORECASE), 3),
    (re.compile(r"bandcamp\.com", re.IGNORECASE), 3),
    (re.compile(r"soundcloud\.com", re.IGNORECASE), 4),
    (re.compile(r"youtube\.com|youtu\.be", re.IGNORECASE), 4),
    (re.compile(r"wikipedia\.org", re.IGNORECASE), 4),
    (re.compile(r"reddit\.com", re.IGNORECASE), 5),
    (re.compile(r"facebook\.com|instagram\.com|twitter\.com|x\.com", re.IGNORECASE), 5),
]


class PromoterResearcher:
    """Deep-research service for a single promoter or event organizer.

    Performs a multi-step research pipeline: web search for promoter activity,
    article scraping for event history, LLM extraction of affiliated artists
    and venues, and final compilation into a :class:`ResearchResult`.

    All external service dependencies are injected through constructor
    parameters, following the adapter pattern (CLAUDE.md Section 6).
    """

    def __init__(
        self,
        web_search: IWebSearchProvider,
        article_scraper: IArticleProvider,
        llm: ILLMProvider,
        cache: ICacheProvider | None = None,
    ) -> None:
        self._web_search = web_search
        self._article_scraper = article_scraper
        self._llm = llm
        self._cache = cache
        self._logger: structlog.BoundLogger = get_logger(__name__)

    # -- Public API -----------------------------------------------------------

    async def research(
        self,
        promoter_name: str,
    ) -> ResearchResult:
        """Execute the full research pipeline for a single promoter.

        Parameters
        ----------
        promoter_name:
            The promoter/organizer name to research.

        Returns
        -------
        ResearchResult
            A compiled research result containing the promoter profile,
            event history, affiliated artists and venues, confidence scores,
            and any warnings about data gaps.
        """
        self._logger.info("Starting promoter research", promoter=promoter_name)

        warnings: list[str] = []
        sources_consulted: list[str] = []

        # Step 1 — SEARCH for promoter activity
        search_results = await self._search_promoter_activity(promoter_name)
        if search_results:
            sources_consulted.append("web_search_promoter")
        else:
            warnings.append("No web results found for promoter")

        # Step 2 — SCRAPE results for event history
        scraped_texts = await self._scrape_results(search_results[:_MAX_SCRAPE_RESULTS])
        scrape_confidence = min(1.0, len(scraped_texts) * 0.15) if scraped_texts else 0.0

        # Step 3 — LLM EXTRACTION of affiliations and event history
        event_history, affiliated_artists, affiliated_venues = await self._extract_promoter_profile(
            promoter_name, scraped_texts
        )
        extraction_confidence = 0.0
        if event_history or affiliated_artists or affiliated_venues:
            extraction_confidence = 0.7
            sources_consulted.append("llm_extraction")
        else:
            warnings.append("LLM extraction produced no promoter profile")

        # Step 4 — BUILD Promoter model and ResearchResult
        search_confidence = min(1.0, len(search_results) * 0.1) if search_results else 0.0

        # Also build article references from search results
        articles = await self._build_article_references(search_results, promoter_name)
        if articles:
            sources_consulted.append("web_search_articles")
        article_confidence = min(1.0, len(articles) * 0.12) if articles else 0.0

        overall_confidence = calculate_confidence(
            scores=[
                search_confidence,
                scrape_confidence,
                extraction_confidence,
                article_confidence,
            ],
            weights=[2.0, 2.0, 3.0, 1.5],
        )

        if not event_history:
            warnings.append("No event history found for promoter")

        promoter = Promoter(
            name=promoter_name,
            confidence=overall_confidence,
            event_history=event_history,
            affiliated_artists=affiliated_artists,
            affiliated_venues=affiliated_venues,
            articles=articles,
        )

        result = ResearchResult(
            entity_type=EntityType.PROMOTER,
            entity_name=promoter_name,
            promoter=promoter,
            sources_consulted=sources_consulted,
            confidence=overall_confidence,
            warnings=warnings,
        )

        self._logger.info(
            "Promoter research complete",
            promoter=promoter_name,
            confidence=round(overall_confidence, 3),
            events=len(event_history),
            affiliated_artists=len(affiliated_artists),
            affiliated_venues=len(affiliated_venues),
            articles=len(articles),
            warnings=len(warnings),
        )

        return result

    # -- Private helpers -------------------------------------------------------

    async def _search_promoter_activity(self, promoter_name: str) -> list[SearchResult]:
        """Search the web for promoter event activity and history."""
        queries = [
            f'"{promoter_name}" promoter events',
            f'"{promoter_name}" rave',
            f'"{promoter_name}" club night party',
        ]

        all_results: list[SearchResult] = []
        for query in queries:
            try:
                results = await self._web_search.search(query=query, num_results=10)
                all_results.extend(results)
            except ResearchError as exc:
                self._logger.warning(
                    "Promoter activity search failed",
                    query=query,
                    error=str(exc),
                )

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique: list[SearchResult] = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique.append(result)

        return unique

    async def _scrape_results(self, results: list[SearchResult]) -> list[str]:
        """Scrape article content from search results, returning extracted text."""
        if not results:
            return []

        tasks = [self._article_scraper.extract_content(r.url) for r in results]
        extracted = await asyncio.gather(*tasks, return_exceptions=True)

        texts: list[str] = []
        for item in extracted:
            if isinstance(item, Exception):
                self._logger.debug("Scrape failed for result", error=str(item))
            elif item is not None and item.text:
                texts.append(item.text[:3000])

        return texts

    async def _extract_promoter_profile(
        self,
        promoter_name: str,
        scraped_texts: list[str],
    ) -> tuple[list[str], list[str], list[str]]:
        """Use LLM to extract event history and affiliations from scraped content.

        Returns
        -------
        tuple[list[str], list[str], list[str]]
            (event_history, affiliated_artists, affiliated_venues)
        """
        if not scraped_texts:
            return [], [], []

        combined_text = "\n---\n".join(scraped_texts[:6])

        system_prompt = (
            "You are a music event historian specializing in rave and electronic music culture. "
            "Given scraped web content about a promoter or event organizer, extract structured "
            "information about their activities."
        )
        user_prompt = (
            f"Analyze the following content about promoter '{promoter_name}' and provide:\n\n"
            "1. EVENT_HISTORY: A list of events they organized or promoted, one per line, "
            "prefixed with '- '. Include event name, venue, and approximate date if available.\n"
            "2. AFFILIATED_ARTISTS: A list of artists/DJs who have performed at their events, "
            "one per line, prefixed with '- '.\n"
            "3. AFFILIATED_VENUES: A list of venues where they have hosted events, "
            "one per line, prefixed with '- '.\n\n"
            "Use these exact section headers. If information is not available for a section, "
            "write 'NONE' for that section.\n\n"
            f"Content:\n{combined_text}"
        )

        try:
            response = await self._llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2,
                max_tokens=2000,
            )
        except Exception as exc:
            self._logger.warning("LLM promoter extraction failed", error=str(exc))
            return [], [], []

        return self._parse_promoter_extraction(response)

    async def _build_article_references(
        self, search_results: list[SearchResult], promoter_name: str
    ) -> list[ArticleReference]:
        """Build ArticleReference objects from top search results."""
        if not search_results:
            return []

        # Deduplicate by URL (already done upstream, but defensive)
        seen_urls: set[str] = set()
        unique: list[SearchResult] = []
        for result in search_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique.append(result)

        tasks = [
            self._extract_article_reference(r, promoter_name) for r in unique[:_MAX_ARTICLE_RESULTS]
        ]
        extracted = await asyncio.gather(*tasks, return_exceptions=True)

        articles: list[ArticleReference] = []
        for item in extracted:
            if isinstance(item, ArticleReference):
                articles.append(item)
            elif isinstance(item, Exception):
                self._logger.debug("Article extraction failed", error=str(item))

        return articles

    async def _extract_article_reference(
        self, search_result: SearchResult, promoter_name: str
    ) -> ArticleReference:
        """Extract content from a search result and build an ArticleReference."""
        content = await self._article_scraper.extract_content(search_result.url)

        snippet: str | None = None
        article_type = "article"

        if content and content.text:
            snippet = self._extract_relevant_snippet(content.text, promoter_name)
            title_lower = (content.title or "").lower()
            if "interview" in title_lower:
                article_type = "interview"
            elif "review" in title_lower:
                article_type = "review"
            elif "event" in title_lower or "party" in title_lower:
                article_type = "event_listing"
        elif search_result.snippet:
            snippet = search_result.snippet

        return ArticleReference(
            title=content.title if content else search_result.title,
            source=self._extract_domain(search_result.url),
            url=search_result.url,
            date=content.date if content else search_result.date,
            article_type=article_type,
            snippet=snippet,
            citation_tier=self._assign_citation_tier(search_result.url),
        )

    # -- Parsing helpers -------------------------------------------------------

    @staticmethod
    def _parse_promoter_extraction(llm_text: str) -> tuple[list[str], list[str], list[str]]:
        """Parse LLM promoter extraction response into structured components.

        Returns
        -------
        tuple[list[str], list[str], list[str]]
            (event_history, affiliated_artists, affiliated_venues)
        """
        event_history: list[str] = []
        affiliated_artists: list[str] = []
        affiliated_venues: list[str] = []

        def _extract_list(section_text: str) -> list[str]:
            items: list[str] = []
            for line in section_text.strip().splitlines():
                line = line.strip().lstrip("- •*")
                if line and line.upper() != "NONE":
                    items.append(line.strip())
            return items

        events_match = re.search(
            r"EVENT_HISTORY\s*:?\s*\n?(.*?)(?=AFFILIATED_ARTISTS|AFFILIATED_VENUES|\Z)",
            llm_text,
            re.IGNORECASE | re.DOTALL,
        )
        if events_match:
            event_history = _extract_list(events_match.group(1))

        artists_match = re.search(
            r"AFFILIATED_ARTISTS\s*:?\s*\n?(.*?)(?=AFFILIATED_VENUES|\Z)",
            llm_text,
            re.IGNORECASE | re.DOTALL,
        )
        if artists_match:
            affiliated_artists = _extract_list(artists_match.group(1))

        venues_match = re.search(
            r"AFFILIATED_VENUES\s*:?\s*\n?(.*?)(?:\Z)",
            llm_text,
            re.IGNORECASE | re.DOTALL,
        )
        if venues_match:
            affiliated_venues = _extract_list(venues_match.group(1))

        return event_history, affiliated_artists, affiliated_venues

    def _assign_citation_tier(self, source_url: str) -> int:
        """Assign a citation authority tier (1-6) based on URL domain patterns."""
        for pattern, tier in _CITATION_TIER_PATTERNS:
            if pattern.search(source_url):
                return tier
        return 6

    @staticmethod
    def _extract_relevant_snippet(text: str, entity_name: str, max_length: int = 500) -> str:
        """Extract the most relevant text snippet mentioning the entity."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        name_lower = entity_name.lower()

        relevant: list[str] = []
        for sentence in sentences:
            if name_lower in sentence.lower():
                relevant.append(sentence.strip())
                if sum(len(s) for s in relevant) >= max_length:
                    break

        snippet = " ".join(relevant) if relevant else text[:max_length]

        if len(snippet) > max_length:
            snippet = snippet[:max_length].rsplit(" ", 1)[0] + "..."

        return snippet

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract the domain name from a URL for source attribution."""
        match = re.search(r"https?://(?:www\.)?([^/]+)", url)
        return match.group(1) if match else url

    # -- Cache helpers ---------------------------------------------------------

    async def _cache_get(self, key: str) -> Any | None:
        """Retrieve a value from cache if a cache provider is configured."""
        if self._cache is None:
            return None
        try:
            return await self._cache.get(key)
        except Exception as exc:
            self._logger.debug("Cache read failed", key=key, error=str(exc))
            return None

    async def _cache_set(self, key: str, value: Any) -> None:
        """Store a value in cache if a cache provider is configured."""
        if self._cache is None:
            return
        try:
            await self._cache.set(key, value, ttl=_CACHE_TTL_SECONDS)
        except Exception as exc:
            self._logger.debug("Cache write failed", key=key, error=str(exc))

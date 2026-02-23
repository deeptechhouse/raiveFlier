"""Deep-research service for date and era context.

Orchestrates web search, article scraping, and LLM analysis to build a
comprehensive cultural/historical context for a date extracted from a rave
flier — what was happening in the scene, the city, and the broader culture
around that time.
"""

from __future__ import annotations

import asyncio
import re
from datetime import date
from typing import Any

import structlog

from src.interfaces.article_provider import IArticleProvider
from src.interfaces.cache_provider import ICacheProvider
from src.interfaces.llm_provider import ILLMProvider
from src.interfaces.web_search_provider import IWebSearchProvider, SearchResult
from src.models.entities import ArticleReference, EntityType
from src.models.research import DateContext, ResearchResult
from src.utils.concurrency import parallel_search
from src.utils.confidence import calculate_confidence
from src.utils.errors import ResearchError
from src.utils.logging import get_logger

_CACHE_TTL_SECONDS = 7200  # 2 hours (date context changes less frequently)
_MAX_SCRAPE_RESULTS = 10
_MAX_ARTICLE_RESULTS = 12

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
    (re.compile(r"beatport\.com", re.IGNORECASE), 3),
    (re.compile(r"soundcloud\.com", re.IGNORECASE), 4),
    (re.compile(r"youtube\.com|youtu\.be", re.IGNORECASE), 4),
    (re.compile(r"wikipedia\.org", re.IGNORECASE), 4),
    (re.compile(r"reddit\.com", re.IGNORECASE), 5),
    (re.compile(r"facebook\.com|instagram\.com|twitter\.com|x\.com", re.IGNORECASE), 5),
]

# Month names for search queries
_MONTH_NAMES = [
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


class DateContextResearcher:
    """Deep-research service for the cultural context of a date/era.

    Performs a multi-step research pipeline: web search for scene activity
    around the date, cultural/historical context search, article scraping,
    LLM synthesis of scene, city, and cultural context, and compilation
    into a :class:`ResearchResult`.

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
        event_date: date,
        city: str | None = None,
    ) -> ResearchResult:
        """Execute the full research pipeline for a date/era context.

        Parameters
        ----------
        event_date:
            The date to research context for.
        city:
            Optional city name to narrow search results for local scene context.

        Returns
        -------
        ResearchResult
            A compiled research result containing the date context with
            scene, city, and cultural information, article references,
            confidence scores, and any warnings about data gaps.
        """
        date_str = event_date.isoformat()
        year = event_date.year
        month_name = _MONTH_NAMES[event_date.month]

        self._logger.info(
            "Starting date context research",
            event_date=date_str,
            city=city,
        )

        warnings: list[str] = []
        sources_consulted: list[str] = []

        # Step 1 — SEARCH for scene context around the date
        scene_results = await self._search_scene_context(year, month_name, city)
        if scene_results:
            sources_consulted.append("web_search_scene")
        else:
            warnings.append("No web results found for scene context")

        # Step 2 — SEARCH for cultural/historical context
        cultural_results = await self._search_cultural_context(year, month_name)
        if cultural_results:
            sources_consulted.append("web_search_cultural")
        else:
            warnings.append("No web results found for cultural context")

        # Combine and scrape all results
        all_search_results = scene_results + cultural_results

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique_results: list[SearchResult] = []
        for result in all_search_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        scraped_texts = await self._scrape_results(unique_results[:_MAX_SCRAPE_RESULTS])
        scrape_confidence = min(1.0, len(scraped_texts) * 0.15) if scraped_texts else 0.0

        # Step 3 — LLM SYNTHESIS of date context
        scene_context, city_context, cultural_context, nearby_events = (
            await self._synthesize_date_context(event_date, year, month_name, city, scraped_texts)
        )
        synthesis_confidence = 0.0
        if scene_context or city_context or cultural_context:
            synthesis_confidence = 0.7
            sources_consulted.append("llm_synthesis")
        else:
            warnings.append("LLM synthesis produced no date context")

        # Build article references from search results
        articles = await self._build_article_references(unique_results, str(year))
        if articles:
            sources_consulted.append("web_search_articles")
        article_confidence = min(1.0, len(articles) * 0.12) if articles else 0.0

        # Step 4 — BUILD DateContext and ResearchResult
        search_confidence = min(1.0, len(unique_results) * 0.08) if unique_results else 0.0
        overall_confidence = calculate_confidence(
            scores=[search_confidence, scrape_confidence, synthesis_confidence, article_confidence],
            weights=[1.5, 2.0, 3.5, 1.5],
        )

        date_context = DateContext(
            event_date=event_date,
            scene_context=scene_context,
            city_context=city_context,
            cultural_context=cultural_context,
            nearby_events=nearby_events,
            sources=articles,
        )

        entity_name = f"{month_name} {year}"
        if city:
            entity_name = f"{city} — {entity_name}"

        result = ResearchResult(
            entity_type=EntityType.DATE,
            entity_name=entity_name,
            date_context=date_context,
            sources_consulted=sources_consulted,
            confidence=overall_confidence,
            warnings=warnings,
        )

        self._logger.info(
            "Date context research complete",
            event_date=date_str,
            city=city,
            confidence=round(overall_confidence, 3),
            articles=len(articles),
            nearby_events=len(nearby_events),
            warnings=len(warnings),
        )

        return result

    # -- Private helpers -------------------------------------------------------

    async def _search_scene_context(
        self, year: int, month_name: str, city: str | None
    ) -> list[SearchResult]:
        """Search the web for electronic music scene context around the date.

        Uses RA.co as the primary source for scene context.
        """
        queries: list[str] = []

        # RA.co-first queries
        if city:
            queries.append(f'site:ra.co "{city}" {year}')
            queries.append(f"{city} rave scene {year}")
            queries.append(f"{city} electronic music {month_name} {year}")
        else:
            queries.append(f"site:ra.co electronic music {month_name} {year}")
            queries.append(f"electronic music scene {month_name} {year}")

        queries.append(f"site:ra.co events {month_name} {year}")
        queries.append(f"rave culture {year}")

        # Execute all queries in parallel with throttling
        all_results: list[SearchResult] = await parallel_search(
            self._web_search.search,
            [{"query": q, "num_results": 12} for q in queries],
            logger=self._logger,
            error_msg="Scene context search failed",
        )

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique: list[SearchResult] = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique.append(result)

        return unique

    async def _search_cultural_context(self, year: int, month_name: str) -> list[SearchResult]:
        """Search for broader cultural and historical context around the date."""
        queries = [
            f"site:ra.co features {year}",
            f"electronic music history {year}",
            f"rave legislation law {year}",
            f"nightlife culture {month_name} {year}",
        ]

        # Execute all queries in parallel with throttling
        all_results: list[SearchResult] = await parallel_search(
            self._web_search.search,
            [{"query": q, "num_results": 10} for q in queries],
            logger=self._logger,
            error_msg="Cultural context search failed",
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

    async def _synthesize_date_context(
        self,
        event_date: date,
        year: int,
        month_name: str,
        city: str | None,
        scraped_texts: list[str],
    ) -> tuple[str | None, str | None, str | None, list[str]]:
        """Use LLM to synthesize date context from scraped content.

        Returns
        -------
        tuple[str | None, str | None, str | None, list[str]]
            (scene_context, city_context, cultural_context, nearby_events)
        """
        if not scraped_texts:
            return None, None, None, []

        combined_text = "\n---\n".join(scraped_texts[:6])
        city_clause = f" in {city}" if city else ""
        date_str = f"{month_name} {year}"

        system_prompt = (
            "You are a music and cultural historian specializing in electronic music, "
            "rave culture, and nightlife. Given scraped web content about a specific time "
            "period, synthesize comprehensive cultural context."
        )
        user_prompt = (
            f"Analyze the following content about the electronic music scene around "
            f"{date_str}{city_clause} (event date: {event_date.isoformat()}) and provide:\n\n"
            "1. SCENE_CONTEXT: A 2-4 sentence description of what was happening in the "
            "electronic music / rave scene at this time. What genres were popular? What "
            "movements were active? What was the general state of the scene?\n"
            "2. CITY_CONTEXT: A 1-3 sentence description of the local music scene "
            f"{city_clause} at this time. If no city-specific information is available, "
            "write 'NONE'.\n"
            "3. CULTURAL_CONTEXT: A 2-4 sentence description of relevant broader cultural "
            "context — legislation (e.g. Criminal Justice Act), media coverage, social "
            "attitudes toward rave culture, notable events in music generally.\n"
            "4. NEARBY_EVENTS: A list of notable events, releases, or happenings around "
            "this time, one per line, prefixed with '- '.\n\n"
            "Use these exact section headers. If information is not available for a section, "
            "write 'NONE' for that section.\n\n"
            f"Content:\n{combined_text}"
        )

        try:
            response = await self._llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2,
                max_tokens=2500,
            )
        except Exception as exc:
            self._logger.warning("LLM date context synthesis failed", error=str(exc))
            return None, None, None, []

        return self._parse_date_context_synthesis(response)

    async def _build_article_references(
        self, search_results: list[SearchResult], entity_name: str
    ) -> list[ArticleReference]:
        """Build ArticleReference objects from top search results."""
        if not search_results:
            return []

        tasks = [
            self._extract_article_reference(r, entity_name)
            for r in search_results[:_MAX_ARTICLE_RESULTS]
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
        self, search_result: SearchResult, entity_name: str
    ) -> ArticleReference:
        """Extract content from a search result and build an ArticleReference."""
        content = await self._article_scraper.extract_content(search_result.url)

        snippet: str | None = None
        article_type = "article"

        if content and content.text:
            snippet = self._extract_relevant_snippet(content.text, entity_name)
            title_lower = (content.title or "").lower()
            if "history" in title_lower or "story" in title_lower:
                article_type = "history"
            elif "law" in title_lower or "legislation" in title_lower or "act" in title_lower:
                article_type = "legislation"
            elif "review" in title_lower:
                article_type = "review"
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
    def _parse_date_context_synthesis(
        llm_text: str,
    ) -> tuple[str | None, str | None, str | None, list[str]]:
        """Parse LLM date context synthesis response into structured components.

        Returns
        -------
        tuple[str | None, str | None, str | None, list[str]]
            (scene_context, city_context, cultural_context, nearby_events)
        """
        scene_context: str | None = None
        city_context: str | None = None
        cultural_context: str | None = None
        nearby_events: list[str] = []

        scene_match = re.search(
            r"SCENE_CONTEXT\s*:?\s*\n?(.*?)(?=CITY_CONTEXT|CULTURAL_CONTEXT|NEARBY_EVENTS|\Z)",
            llm_text,
            re.IGNORECASE | re.DOTALL,
        )
        if scene_match:
            text = scene_match.group(1).strip()
            if text and text.upper() != "NONE":
                scene_context = text

        city_match = re.search(
            r"CITY_CONTEXT\s*:?\s*\n?(.*?)(?=CULTURAL_CONTEXT|NEARBY_EVENTS|\Z)",
            llm_text,
            re.IGNORECASE | re.DOTALL,
        )
        if city_match:
            text = city_match.group(1).strip()
            if text and text.upper() != "NONE":
                city_context = text

        cultural_match = re.search(
            r"CULTURAL_CONTEXT\s*:?\s*\n?(.*?)(?=NEARBY_EVENTS|\Z)",
            llm_text,
            re.IGNORECASE | re.DOTALL,
        )
        if cultural_match:
            text = cultural_match.group(1).strip()
            if text and text.upper() != "NONE":
                cultural_context = text

        events_match = re.search(
            r"NEARBY_EVENTS\s*:?\s*\n?(.*?)(?:\Z)",
            llm_text,
            re.IGNORECASE | re.DOTALL,
        )
        if events_match:
            text = events_match.group(1).strip()
            if text and text.upper() != "NONE":
                for line in text.splitlines():
                    line = line.strip().lstrip("- •*")
                    if line and line.upper() != "NONE":
                        nearby_events.append(line.strip())

        return scene_context, city_context, cultural_context, nearby_events

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

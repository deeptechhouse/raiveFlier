"""Deep-research service for venues and nightclubs.

Orchestrates web search, article scraping, and LLM analysis to build a
comprehensive research profile for a venue extracted from a rave flier.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

import structlog

from src.interfaces.article_provider import IArticleProvider
from src.interfaces.cache_provider import ICacheProvider
from src.interfaces.llm_provider import ILLMProvider
from src.interfaces.web_search_provider import IWebSearchProvider, SearchResult
from src.models.entities import ArticleReference, EntityType, Venue
from src.models.research import ResearchResult
from src.utils.concurrency import parallel_search
from src.utils.confidence import calculate_confidence
from src.utils.errors import ResearchError
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.interfaces.vector_store_provider import IVectorStoreProvider

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


class VenueResearcher:
    """Deep-research service for a single venue or nightclub.

    Performs a multi-step research pipeline: web search for venue history,
    article scraping for historical context, LLM synthesis of cultural
    significance, and final compilation into a :class:`ResearchResult`.

    All external service dependencies are injected through constructor
    parameters, following the adapter pattern (CLAUDE.md Section 6).
    """

    def __init__(
        self,
        web_search: IWebSearchProvider,
        article_scraper: IArticleProvider,
        llm: ILLMProvider,
        cache: ICacheProvider | None = None,
        vector_store: IVectorStoreProvider | None = None,
    ) -> None:
        self._web_search = web_search
        self._article_scraper = article_scraper
        self._llm = llm
        self._cache = cache
        self._vector_store = vector_store
        self._logger: structlog.BoundLogger = get_logger(__name__)

    # -- Public API -----------------------------------------------------------

    async def research(
        self,
        venue_name: str,
        city: str | None = None,
    ) -> ResearchResult:
        """Execute the full research pipeline for a single venue.

        Parameters
        ----------
        venue_name:
            The venue name to research.
        city:
            Optional city name to narrow search results.

        Returns
        -------
        ResearchResult
            A compiled research result containing the venue profile,
            history, cultural significance, article references, confidence
            scores, and any warnings about data gaps.
        """
        self._logger.info(
            "Starting venue research",
            venue=venue_name,
            city=city,
        )

        warnings: list[str] = []
        sources_consulted: list[str] = []

        # Step 1 — SEARCH for venue history
        history_results = await self._search_venue_history(venue_name, city)
        if history_results:
            sources_consulted.append("web_search_history")
        else:
            warnings.append("No web results found for venue history")

        # Step 2 — SCRAPE top results for historical context
        scraped_texts = await self._scrape_results(history_results[:_MAX_SCRAPE_RESULTS])
        scrape_confidence = min(1.0, len(scraped_texts) * 0.15) if scraped_texts else 0.0

        # Step 3 — LLM SYNTHESIS of venue history and cultural significance
        history, notable_events, cultural_significance = await self._synthesize_venue_profile(
            venue_name, city, scraped_texts
        )
        synthesis_confidence = 0.0
        if history or notable_events or cultural_significance:
            synthesis_confidence = 0.7
            sources_consulted.append("llm_synthesis")
        else:
            warnings.append("LLM synthesis produced no venue profile")

        # Step 4 — ARTICLE search for press mentions
        articles = await self._search_venue_articles(venue_name, city)
        if articles:
            sources_consulted.append("web_search_articles")
        else:
            warnings.append("No press articles found for venue")
        article_confidence = min(1.0, len(articles) * 0.12) if articles else 0.0

        # Step 4.5 — CORPUS RETRIEVAL (RAG)
        corpus_refs = await self._retrieve_from_corpus(venue_name)
        if corpus_refs:
            sources_consulted.append("rag_corpus")
            existing_titles = {a.title.lower().strip() for a in articles if a.title}
            for ref in corpus_refs:
                ref_title_lower = ref.title.lower().strip() if ref.title else ""
                if ref_title_lower and ref_title_lower not in existing_titles:
                    existing_titles.add(ref_title_lower)
                    articles.append(ref)

        # Step 5 — BUILD Venue model and ResearchResult
        search_confidence = min(1.0, len(history_results) * 0.1) if history_results else 0.0
        overall_confidence = calculate_confidence(
            scores=[search_confidence, scrape_confidence, synthesis_confidence, article_confidence],
            weights=[2.0, 2.0, 3.0, 1.5],
        )

        venue = Venue(
            name=venue_name,
            city=city,
            confidence=overall_confidence,
            history=history,
            notable_events=notable_events,
            cultural_significance=cultural_significance,
            articles=articles,
        )

        result = ResearchResult(
            entity_type=EntityType.VENUE,
            entity_name=venue_name,
            venue=venue,
            sources_consulted=sources_consulted,
            confidence=overall_confidence,
            warnings=warnings,
        )

        self._logger.info(
            "Venue research complete",
            venue=venue_name,
            confidence=round(overall_confidence, 3),
            articles=len(articles),
            warnings=len(warnings),
        )

        return result

    # -- Private helpers -------------------------------------------------------

    async def _retrieve_from_corpus(self, venue_name: str) -> list[ArticleReference]:
        """Retrieve relevant passages from the RAG corpus for this venue.

        Returns an empty list if no vector store is configured or available.
        """
        if not self._vector_store or not self._vector_store.is_available():
            return []

        query = f"{venue_name} venue nightclub history"

        try:
            chunks = await self._vector_store.query(query_text=query, top_k=15)
        except Exception as exc:
            self._logger.debug("Corpus retrieval failed", venue=venue_name, error=str(exc))
            return []

        # Deduplicate by source_id — keep top 3 chunks per source (safety net)
        _MAX_PER_SOURCE = 3
        source_chunks: dict[str, list[tuple[float, ArticleReference]]] = {}
        for chunk in chunks:
            if chunk.similarity_score < 0.7:
                continue
            sid = chunk.chunk.source_id
            ref = ArticleReference(
                title=chunk.chunk.source_title,
                source=chunk.chunk.source_type,
                url=None,
                date=chunk.chunk.publication_date,
                article_type="book" if chunk.chunk.source_type == "book" else "article",
                snippet=chunk.chunk.text[:200] + "...",
                citation_tier=chunk.chunk.citation_tier,
            )
            source_chunks.setdefault(sid, []).append((chunk.similarity_score, ref))

        refs: list[ArticleReference] = []
        for entries in source_chunks.values():
            entries.sort(key=lambda t: t[0], reverse=True)
            refs.extend(ref for _, ref in entries[:_MAX_PER_SOURCE])

        if refs:
            self._logger.info(
                "Corpus retrieval complete",
                venue=venue_name,
                chunks_retrieved=len(refs),
            )
        return refs

    async def _search_venue_history(self, venue_name: str, city: str | None) -> list[SearchResult]:
        """Search the web for venue history and background information."""
        city_clause = f" {city}" if city else ""
        queries = [
            f'site:ra.co/clubs "{venue_name}"',
            f'"{venue_name}"{city_clause} venue history',
        ]
        if city:
            queries.append(f'"{venue_name}" {city} nightclub events')
        else:
            queries.append(f'"{venue_name}" nightclub club events')

        # Execute all queries in parallel with throttling
        all_results: list[SearchResult] = await parallel_search(
            self._web_search.search,
            [{"query": q, "num_results": 15} for q in queries],
            logger=self._logger,
            error_msg="Venue history search failed",
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
                # Truncate very long articles to keep LLM context manageable
                texts.append(item.text[:3000])

        return texts

    async def _synthesize_venue_profile(
        self,
        venue_name: str,
        city: str | None,
        scraped_texts: list[str],
    ) -> tuple[str | None, list[str], str | None]:
        """Use LLM to synthesize venue history, notable events, and cultural significance.

        Returns
        -------
        tuple[str | None, list[str], str | None]
            (history_summary, notable_events, cultural_significance)
        """
        if not scraped_texts:
            return None, [], None

        combined_text = "\n---\n".join(scraped_texts[:6])
        city_clause = f" in {city}" if city else ""

        system_prompt = (
            "You are a music venue historian specializing in nightclub and rave culture. "
            "Given scraped web content about a venue, synthesize a comprehensive profile."
        )
        user_prompt = (
            f"Analyze the following content about '{venue_name}'{city_clause} and provide:\n\n"
            "1. HISTORY: A 2-4 sentence summary of the venue's history and background.\n"
            "2. NOTABLE_EVENTS: A list of notable events, one per line, prefixed with '- '.\n"
            "3. CULTURAL_SIGNIFICANCE: A 1-3 sentence description of the venue's cultural "
            "significance in the electronic music / rave scene.\n\n"
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
            self._logger.warning("LLM venue synthesis failed", error=str(exc))
            return None, [], None

        return self._parse_venue_synthesis(response)

    async def _search_venue_articles(
        self, venue_name: str, city: str | None
    ) -> list[ArticleReference]:
        """Search for press articles mentioning the venue."""
        city_clause = f" {city}" if city else ""
        queries = [
            f'site:ra.co "{venue_name}"',
            f'"{venue_name}"{city_clause} "Resident Advisor" OR Mixmag OR "DJ Mag"',
            f'"{venue_name}" nightclub review OR history OR events',
        ]

        # Execute all queries in parallel with throttling
        all_results: list[SearchResult] = await parallel_search(
            self._web_search.search,
            [{"query": q, "num_results": 15} for q in queries],
            logger=self._logger,
            error_msg="Venue article search failed",
        )

        if not all_results:
            return []

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique: list[SearchResult] = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique.append(result)

        # Extract article references from top results
        tasks = [
            self._extract_article_reference(r, venue_name) for r in unique[:_MAX_ARTICLE_RESULTS]
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
        self, search_result: SearchResult, venue_name: str
    ) -> ArticleReference:
        """Extract content from a search result and build an ArticleReference."""
        content = await self._article_scraper.extract_content(search_result.url)

        snippet: str | None = None
        article_type = "article"

        if content and content.text:
            snippet = self._extract_relevant_snippet(content.text, venue_name)
            title_lower = (content.title or "").lower()
            if "review" in title_lower:
                article_type = "review"
            elif "history" in title_lower or "story" in title_lower:
                article_type = "history"
            elif "closing" in title_lower or "closed" in title_lower:
                article_type = "obituary"
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
    def _parse_venue_synthesis(llm_text: str) -> tuple[str | None, list[str], str | None]:
        """Parse LLM venue synthesis response into structured components.

        Returns
        -------
        tuple[str | None, list[str], str | None]
            (history, notable_events, cultural_significance)
        """
        history: str | None = None
        notable_events: list[str] = []
        cultural_significance: str | None = None

        # Match each section by scanning for headers
        history_match = re.search(
            r"HISTORY\s*:?\s*\n?(.*?)(?=NOTABLE_EVENTS|CULTURAL_SIGNIFICANCE|\Z)",
            llm_text,
            re.IGNORECASE | re.DOTALL,
        )
        if history_match:
            text = history_match.group(1).strip()
            if text and text.upper() != "NONE":
                history = text

        events_match = re.search(
            r"NOTABLE_EVENTS\s*:?\s*\n?(.*?)(?=CULTURAL_SIGNIFICANCE|\Z)",
            llm_text,
            re.IGNORECASE | re.DOTALL,
        )
        if events_match:
            text = events_match.group(1).strip()
            if text and text.upper() != "NONE":
                for line in text.splitlines():
                    line = line.strip().lstrip("- •*")
                    if line and line.upper() != "NONE":
                        notable_events.append(line.strip())

        significance_match = re.search(
            r"CULTURAL_SIGNIFICANCE\s*:?\s*\n?(.*?)(?:\Z)",
            llm_text,
            re.IGNORECASE | re.DOTALL,
        )
        if significance_match:
            text = significance_match.group(1).strip()
            if text and text.upper() != "NONE":
                cultural_significance = text

        return history, notable_events, cultural_significance

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

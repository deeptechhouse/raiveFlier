"""Deep-research service for promoters and event organizers.

Orchestrates web search, article scraping, and LLM analysis to build a
comprehensive research profile for a promoter extracted from a rave flier.
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
from src.models.entities import ArticleReference, EntityType, Promoter
from src.models.research import ResearchResult
from src.utils.concurrency import parallel_search
from src.utils.confidence import calculate_confidence
from src.utils.errors import ResearchError
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.interfaces.vector_store_provider import IVectorStoreProvider

_CACHE_TTL_SECONDS = 3600  # 1 hour
_MAX_SCRAPE_RESULTS = 10
_MAX_ARTICLE_RESULTS = 12
_MIN_ADEQUATE_RESULTS = 4

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
        promoter_name: str,
        city: str | None = None,
    ) -> ResearchResult:
        """Execute the full research pipeline for a single promoter.

        Parameters
        ----------
        promoter_name:
            The promoter/organizer name to research.
        city:
            Optional city hint from the venue entity for geographic
            disambiguation and context-aware search queries.

        Returns
        -------
        ResearchResult
            A compiled research result containing the promoter profile,
            event history, affiliated artists and venues, confidence scores,
            and any warnings about data gaps.
        """
        self._logger.info("Starting promoter research", promoter=promoter_name, city=city)

        warnings: list[str] = []
        sources_consulted: list[str] = []

        # Step 1 — SEARCH for promoter activity
        search_results = await self._search_promoter_activity(promoter_name, city=city)
        if search_results:
            sources_consulted.append("web_search_promoter")
        else:
            warnings.append("No web results found for promoter")

        # Step 2 — SCRAPE results for event history
        scraped_texts = await self._scrape_results(search_results[:_MAX_SCRAPE_RESULTS])
        scrape_confidence = min(1.0, len(scraped_texts) * 0.15) if scraped_texts else 0.0

        # Step 3 — LLM EXTRACTION of affiliations and event history
        extraction_result = await self._extract_promoter_profile(
            promoter_name, scraped_texts, city=city
        )
        event_history = extraction_result["event_history"]
        affiliated_artists = extraction_result["affiliated_artists"]
        affiliated_venues = extraction_result["affiliated_venues"]
        based_city = extraction_result.get("based_city")
        based_region = extraction_result.get("based_region")
        based_country = extraction_result.get("based_country")

        extraction_confidence = 0.0
        if event_history or affiliated_artists or affiliated_venues:
            extraction_confidence = 0.7
            sources_consulted.append("llm_extraction")
        else:
            warnings.append("LLM extraction produced no promoter profile")

        # Step 3.5 — CORPUS RETRIEVAL (RAG)
        corpus_refs = await self._retrieve_from_corpus(promoter_name, city=city)
        if corpus_refs:
            sources_consulted.append("rag_corpus")

        # Step 4 — BUILD Promoter model and ResearchResult
        search_confidence = min(1.0, len(search_results) * 0.1) if search_results else 0.0

        # Also build article references from search results
        articles = await self._build_article_references(search_results, promoter_name)
        if articles:
            sources_consulted.append("web_search_articles")

        # Merge corpus refs into articles
        if corpus_refs:
            existing_titles = {a.title.lower().strip() for a in articles if a.title}
            for ref in corpus_refs:
                ref_title_lower = ref.title.lower().strip() if ref.title else ""
                if ref_title_lower and ref_title_lower not in existing_titles:
                    existing_titles.add(ref_title_lower)
                    articles.append(ref)

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
            city=based_city,
            region=based_region,
            country=based_country,
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
            based_city=based_city,
            warnings=len(warnings),
        )

        return result

    # -- Private helpers -------------------------------------------------------

    async def _retrieve_from_corpus(
        self, promoter_name: str, city: str | None = None
    ) -> list[ArticleReference]:
        """Retrieve relevant passages from the RAG corpus for this promoter.

        Returns an empty list if no vector store is configured or available.
        """
        if not self._vector_store or not self._vector_store.is_available():
            return []

        city_part = f" {city}" if city else ""
        query = f"{promoter_name}{city_part} promoter events rave"

        try:
            chunks = await self._vector_store.query(query_text=query, top_k=15)
        except Exception as exc:
            self._logger.debug("Corpus retrieval failed", promoter=promoter_name, error=str(exc))
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
                promoter=promoter_name,
                chunks_retrieved=len(refs),
            )
        return refs

    async def _search_promoter_activity(
        self, promoter_name: str, city: str | None = None
    ) -> list[SearchResult]:
        """Search the web for promoter event activity and history.

        Uses RA.co as the primary source, then general web searches.
        When ``city`` is provided, city-qualified queries are prepended
        to improve disambiguation for common-word promoter names.
        Performs adaptive deepening if initial results are sparse.
        """
        # Phase 0 — City-qualified queries (disambiguation)
        queries: list[str] = []
        if city:
            queries.extend([
                f'site:ra.co/promoters "{promoter_name}" "{city}"',
                f'"{promoter_name}" promoter events "{city}"',
                f'"{promoter_name}" "{city}" rave club night',
            ])

        # Phase 1 — RA.co-first + general queries (always included as fallback)
        queries.extend([
            f'site:ra.co/promoters "{promoter_name}"',
            f'site:ra.co/events "{promoter_name}"',
            f'"{promoter_name}" promoter events',
            f'"{promoter_name}" rave club night party',
        ])

        # Execute all primary queries in parallel with throttling
        all_results: list[SearchResult] = await parallel_search(
            self._web_search.search,
            [{"query": q, "num_results": 12} for q in queries],
            logger=self._logger,
            error_msg="Promoter activity search failed",
        )

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique: list[SearchResult] = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique.append(result)

        # Phase 2 — Adaptive deepening if sparse results
        if len(unique) < _MIN_ADEQUATE_RESULTS:
            self._logger.info(
                "Sparse promoter results, deepening search",
                promoter=promoter_name,
                initial_count=len(unique),
            )
            deepening_queries = [
                f'"{promoter_name}" promoter DJ event lineup',
                f'"{promoter_name}" nightlife organizer',
                f'site:ra.co "{promoter_name}"',
            ]
            deeper_results: list[SearchResult] = await parallel_search(
                self._web_search.search,
                [{"query": q, "num_results": 10} for q in deepening_queries],
                logger=self._logger,
                error_msg="Deepened promoter search failed",
            )
            for r in deeper_results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    unique.append(r)

        # Sort: RA.co results first
        ra_results = [r for r in unique if "ra.co" in r.url]
        other_results = [r for r in unique if "ra.co" not in r.url]
        return ra_results + other_results

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
        city: str | None = None,
    ) -> dict[str, Any]:
        """Use LLM to extract event history, affiliations, and geography.

        Returns
        -------
        dict[str, Any]
            Keys: event_history, affiliated_artists, affiliated_venues,
            based_city, based_region, based_country.
        """
        empty_result: dict[str, Any] = {
            "event_history": [],
            "affiliated_artists": [],
            "affiliated_venues": [],
            "based_city": None,
            "based_region": None,
            "based_country": None,
        }
        if not scraped_texts:
            return empty_result

        combined_text = "\n---\n".join(scraped_texts[:6])

        city_context = f"The event flier is from {city}.\n" if city else ""

        system_prompt = (
            "You are a music event historian specializing in rave and electronic music culture. "
            "Given scraped web content about a promoter or event organizer, extract structured "
            "information about their activities."
        )
        user_prompt = (
            f"{city_context}"
            f"Analyze the following content about promoter '{promoter_name}' and provide:\n\n"
            "1. BASED_IN: The city, region/state, and country where this promoter primarily "
            "operates, on one line, prefixed with '- '. Format: CITY, REGION, COUNTRY. "
            "If unknown, write 'NONE'.\n"
            "2. EVENT_HISTORY: A list of events they organized or promoted, one per line, "
            "prefixed with '- '. Include event name, venue, and approximate date if available.\n"
            "3. AFFILIATED_ARTISTS: A list of artists/DJs who have performed at their events, "
            "one per line, prefixed with '- '.\n"
            "4. AFFILIATED_VENUES: A list of venues where they have hosted events, "
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
            return empty_result

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
    def _parse_promoter_extraction(llm_text: str) -> dict[str, Any]:
        """Parse LLM promoter extraction response into structured components.

        Returns
        -------
        dict[str, Any]
            Keys: event_history, affiliated_artists, affiliated_venues,
            based_city, based_region, based_country.
        """
        event_history: list[str] = []
        affiliated_artists: list[str] = []
        affiliated_venues: list[str] = []
        based_city: str | None = None
        based_region: str | None = None
        based_country: str | None = None

        def _extract_list(section_text: str) -> list[str]:
            items: list[str] = []
            for line in section_text.strip().splitlines():
                line = line.strip().lstrip("- •*")
                if line and line.upper() != "NONE":
                    items.append(line.strip())
            return items

        # Parse BASED_IN section
        based_match = re.search(
            r"BASED_IN\s*:?\s*\n?(.*?)(?=EVENT_HISTORY|AFFILIATED_ARTISTS|AFFILIATED_VENUES|\Z)",
            llm_text,
            re.IGNORECASE | re.DOTALL,
        )
        if based_match:
            line = based_match.group(1).strip().lstrip("- •*").strip()
            if line and line.upper() != "NONE":
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 1 and parts[0]:
                    based_city = parts[0]
                if len(parts) >= 2 and parts[1]:
                    based_region = parts[1]
                if len(parts) >= 3 and parts[2]:
                    based_country = parts[2]

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

        return {
            "event_history": event_history,
            "affiliated_artists": affiliated_artists,
            "affiliated_venues": affiliated_venues,
            "based_city": based_city,
            "based_region": based_region,
            "based_country": based_country,
        }

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

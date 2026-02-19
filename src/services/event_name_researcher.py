"""Deep-research service for event/series names from rave fliers.

Orchestrates web search, article scraping, RAG corpus retrieval, and
LLM analysis to discover historical instances of a named event, group
them by promoter, and detect promoter name changes over time.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING, Any

import structlog

from src.interfaces.article_provider import IArticleProvider
from src.interfaces.cache_provider import ICacheProvider
from src.interfaces.llm_provider import ILLMProvider
from src.interfaces.web_search_provider import IWebSearchProvider, SearchResult
from src.models.entities import (
    ArticleReference,
    EntityType,
    EventInstance,
    EventSeriesHistory,
)
from src.models.research import ResearchResult
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

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


class EventNameResearcher:
    """Deep-research service for a named event or party series.

    Performs a multi-step research pipeline: web search for event instances,
    article scraping, RAG corpus retrieval, LLM extraction of structured
    event records, grouping by promoter, and detection of promoter name
    changes.

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
        event_name: str,
        promoter_name: str | None = None,
        city: str | None = None,
    ) -> ResearchResult:
        """Execute the full research pipeline for an event/series name.

        Parameters
        ----------
        event_name:
            The event or party series name to research.
        promoter_name:
            Optional known promoter name (used for targeted search).
        city:
            Optional city hint for geographic context.

        Returns
        -------
        ResearchResult
            A compiled research result containing event history,
            promoter groupings, and any detected promoter name changes.
        """
        self._logger.info(
            "Starting event name research",
            event_name=event_name,
            promoter=promoter_name,
            city=city,
        )

        warnings: list[str] = []
        sources_consulted: list[str] = []

        # Step 1 — SEARCH for event instances
        search_results = await self._search_event_instances(event_name, promoter_name)
        if search_results:
            sources_consulted.append("web_search_event")
        else:
            warnings.append("No web results found for event name")

        # Step 2 — SCRAPE top results
        scraped_texts = await self._scrape_results(search_results[:_MAX_SCRAPE_RESULTS])
        scrape_confidence = min(1.0, len(scraped_texts) * 0.15) if scraped_texts else 0.0

        # Step 3 — LLM EXTRACTION of event instances
        instances = await self._extract_event_instances(event_name, scraped_texts)
        extraction_confidence = 0.0
        if instances:
            extraction_confidence = 0.7
            sources_consulted.append("llm_extraction")
        else:
            warnings.append("LLM extraction produced no event instances")

        # Step 3.5 — CORPUS RETRIEVAL (RAG)
        corpus_refs = await self._retrieve_from_corpus(event_name)
        if corpus_refs:
            sources_consulted.append("rag_corpus")

        # Step 4 — GROUP by promoter
        promoter_groups = self._group_by_promoter(instances)

        # Step 5 — DETECT promoter name changes (if multiple promoters found)
        promoter_name_changes: list[str] = []
        if len(promoter_groups) > 1:
            promoter_name_changes = await self._detect_promoter_name_changes(
                event_name, promoter_groups
            )
            if promoter_name_changes:
                sources_consulted.append("llm_analysis")

        # Step 6 — BUILD article references from search results
        articles = await self._build_article_references(search_results, event_name)
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

        # Step 7 — CONFIDENCE calculation
        search_confidence = min(1.0, len(search_results) * 0.1) if search_results else 0.0
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

        # Serialize promoter_groups for the model (dict[str, list[EventInstance]])
        serialized_groups: dict[str, list[EventInstance]] = {}
        for pname, insts in promoter_groups.items():
            serialized_groups[pname] = insts

        event_history = EventSeriesHistory(
            event_name=event_name,
            instances=instances,
            promoter_groups=serialized_groups,
            promoter_name_changes=promoter_name_changes,
            total_found=len(instances),
            articles=articles,
        )

        result = ResearchResult(
            entity_type=EntityType.EVENT,
            entity_name=event_name,
            event_history=event_history,
            sources_consulted=sources_consulted,
            confidence=overall_confidence,
            warnings=warnings,
        )

        self._logger.info(
            "Event name research complete",
            event_name=event_name,
            confidence=round(overall_confidence, 3),
            instances=len(instances),
            promoter_groups=len(promoter_groups),
            name_changes=len(promoter_name_changes),
            articles=len(articles),
            warnings=len(warnings),
        )

        return result

    # -- Private helpers -------------------------------------------------------

    async def _retrieve_from_corpus(self, event_name: str) -> list[ArticleReference]:
        """Retrieve relevant passages from the RAG corpus for this event."""
        if not self._vector_store or not self._vector_store.is_available():
            return []

        query = f"{event_name} event party rave club night"

        try:
            chunks = await self._vector_store.query(query_text=query, top_k=15)
        except Exception as exc:
            self._logger.debug("Corpus retrieval failed", event_name=event_name, error=str(exc))
            return []

        refs: list[ArticleReference] = []
        for chunk in chunks:
            if chunk.similarity_score < 0.7:
                continue
            refs.append(
                ArticleReference(
                    title=chunk.chunk.source_title,
                    source=chunk.chunk.source_type,
                    url=None,
                    date=chunk.chunk.publication_date,
                    article_type="book" if chunk.chunk.source_type == "book" else "article",
                    snippet=chunk.chunk.text[:200] + "...",
                    citation_tier=chunk.chunk.citation_tier,
                )
            )

        if refs:
            self._logger.info(
                "Corpus retrieval complete",
                event_name=event_name,
                chunks_retrieved=len(refs),
            )
        return refs

    async def _search_event_instances(
        self, event_name: str, promoter_name: str | None
    ) -> list[SearchResult]:
        """Search the web for instances of the named event."""
        queries = [
            f'"{event_name}" rave OR club OR electronic music event',
            f'"{event_name}" party night lineup',
        ]

        if promoter_name:
            queries.append(f'"{event_name}" "{promoter_name}"')

        all_results: list[SearchResult] = []
        for query in queries:
            try:
                results = await self._web_search.search(query=query, num_results=10)
                all_results.extend(results)
            except ResearchError as exc:
                self._logger.warning(
                    "Event search failed",
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

    async def _extract_event_instances(
        self,
        event_name: str,
        scraped_texts: list[str],
    ) -> list[EventInstance]:
        """Use LLM to extract structured event instances from scraped content."""
        if not scraped_texts:
            return []

        combined_text = "\n---\n".join(scraped_texts[:6])

        system_prompt = (
            "You are a music event historian specializing in rave and electronic music culture. "
            "Given scraped web content about a named event or party series, extract every "
            "identifiable historical instance of this event."
        )
        user_prompt = (
            f"Find all instances of the event/party called '{event_name}' in the following "
            "scraped content. For each instance, extract:\n\n"
            "Return a JSON array of objects with these keys:\n"
            '- "event_name": the exact event name\n'
            '- "promoter": who organized/promoted it (null if unknown)\n'
            '- "venue": where it was held (null if unknown)\n'
            '- "city": the city (null if unknown)\n'
            '- "date": approximate date as a string (null if unknown)\n'
            '- "source_url": the URL this info came from (null if unknown)\n\n'
            "Return ONLY the JSON array, no commentary.\n\n"
            f"Content:\n{combined_text}"
        )

        try:
            response = await self._llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2,
                max_tokens=3000,
            )
        except Exception as exc:
            self._logger.warning("LLM event extraction failed", error=str(exc))
            return []

        return self._parse_event_instances(response)

    @staticmethod
    def _parse_event_instances(llm_text: str) -> list[EventInstance]:
        """Parse LLM response into a list of EventInstance objects."""
        text = llm_text.strip()

        # Try to extract JSON from markdown fences first
        fence_match = _JSON_FENCE_RE.search(text)
        if fence_match:
            text = fence_match.group(1).strip()

        # Fallback: find the first [ ... ] block
        if not text.startswith("["):
            bracket_start = text.find("[")
            bracket_end = text.rfind("]")
            if bracket_start != -1 and bracket_end > bracket_start:
                text = text[bracket_start : bracket_end + 1]

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []

        if not isinstance(parsed, list):
            return []

        instances: list[EventInstance] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            instances.append(
                EventInstance(
                    event_name=str(item.get("event_name", "")).strip() or "Unknown",
                    promoter=item.get("promoter") if item.get("promoter") else None,
                    venue=item.get("venue") if item.get("venue") else None,
                    city=item.get("city") if item.get("city") else None,
                    date=item.get("date") if item.get("date") else None,
                    source_url=item.get("source_url") if item.get("source_url") else None,
                )
            )

        return instances

    @staticmethod
    def _group_by_promoter(instances: list[EventInstance]) -> dict[str, list[EventInstance]]:
        """Group event instances by promoter name."""
        groups: dict[str, list[EventInstance]] = {}
        for inst in instances:
            key = inst.promoter or "Unknown Promoter"
            if key not in groups:
                groups[key] = []
            groups[key].append(inst)
        return groups

    async def _detect_promoter_name_changes(
        self,
        event_name: str,
        promoter_groups: dict[str, list[EventInstance]],
    ) -> list[str]:
        """Use LLM to detect if different promoter names are the same entity."""
        promoter_names = list(promoter_groups.keys())
        if len(promoter_names) < 2:
            return []

        # Build a summary of each promoter's activity
        summary_lines: list[str] = []
        for pname, insts in promoter_groups.items():
            dates = [i.date for i in insts if i.date]
            venues = [i.venue for i in insts if i.venue]
            summary_lines.append(
                f"- {pname}: {len(insts)} events"
                f"{', dates: ' + ', '.join(dates[:5]) if dates else ''}"
                f"{', venues: ' + ', '.join(venues[:5]) if venues else ''}"
            )

        system_prompt = (
            "You are an expert on rave and electronic music event history. "
            "Analyze promoter names to determine if any appear to be the same "
            "entity operating under different names."
        )
        user_prompt = (
            f"The event '{event_name}' has been promoted by these different names:\n\n"
            + "\n".join(summary_lines)
            + "\n\nDo any of these promoter names appear to be the same entity under "
            "different names? Consider:\n"
            "- Similar names (abbreviations, slight variations)\n"
            "- Overlapping timeframes and venues\n"
            "- Known promoter rebrandings in the scene\n\n"
            "Return a JSON array of strings, each describing a detected connection. "
            'For example: ["Promoter A and Promoter B appear to be the same entity '
            '(name change around 2005)", ...]\n'
            "If no connections found, return an empty array: []\n"
            "Return ONLY the JSON array."
        )

        try:
            response = await self._llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=1000,
            )
        except Exception as exc:
            self._logger.warning("LLM promoter analysis failed", error=str(exc))
            return []

        return self._parse_name_changes(response)

    @staticmethod
    def _parse_name_changes(llm_text: str) -> list[str]:
        """Parse LLM response for promoter name change detections."""
        text = llm_text.strip()

        fence_match = _JSON_FENCE_RE.search(text)
        if fence_match:
            text = fence_match.group(1).strip()

        if not text.startswith("["):
            bracket_start = text.find("[")
            bracket_end = text.rfind("]")
            if bracket_start != -1 and bracket_end > bracket_start:
                text = text[bracket_start : bracket_end + 1]

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []

        if not isinstance(parsed, list):
            return []

        return [str(item).strip() for item in parsed if str(item).strip()]

    async def _build_article_references(
        self, search_results: list[SearchResult], event_name: str
    ) -> list[ArticleReference]:
        """Build ArticleReference objects from top search results."""
        if not search_results:
            return []

        seen_urls: set[str] = set()
        unique: list[SearchResult] = []
        for result in search_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique.append(result)

        tasks = [
            self._extract_article_reference(r, event_name) for r in unique[:_MAX_ARTICLE_RESULTS]
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
        self, search_result: SearchResult, event_name: str
    ) -> ArticleReference:
        """Extract content from a search result and build an ArticleReference."""
        content = await self._article_scraper.extract_content(search_result.url)

        snippet: str | None = None
        article_type = "article"

        if content and content.text:
            snippet = self._extract_relevant_snippet(content.text, event_name)
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

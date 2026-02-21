"""Deep-research service for individual artists and DJs.

Orchestrates multiple data sources — music databases (Discogs, MusicBrainz),
web search, article scraping, and LLM analysis — to build a comprehensive
research profile for a single artist extracted from a rave flier.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from datetime import date
from typing import TYPE_CHECKING, Any

import structlog

from src.interfaces.article_provider import IArticleProvider
from src.interfaces.cache_provider import ICacheProvider
from src.interfaces.llm_provider import ILLMProvider
from src.interfaces.music_db_provider import ArtistSearchResult, IMusicDatabaseProvider
from src.interfaces.web_search_provider import IWebSearchProvider, SearchResult
from src.models.entities import (
    ArticleReference,
    Artist,
    EntityType,
    EventAppearance,
    Label,
    Release,
)
from src.models.research import ResearchResult
from src.utils.confidence import calculate_confidence
from src.utils.errors import RateLimitError, ResearchError
from src.utils.logging import get_logger
from src.utils.text_normalizer import fuzzy_match, normalize_artist_name

if TYPE_CHECKING:
    from src.interfaces.vector_store_provider import IVectorStoreProvider

_CACHE_TTL_SECONDS = 3600  # 1 hour
_MAX_PRESS_ARTICLES = 15
_GIG_SEARCH_LIMIT = 30
_PRESS_SEARCH_LIMIT = 20
_MIN_ADEQUATE_APPEARANCES = 3  # trigger deeper search if below this
_MIN_ADEQUATE_ARTICLES = 2

# Domains known to be music-related — results from these always pass relevance check
_MUSIC_DOMAINS = re.compile(
    r"ra\.co|residentadvisor|djmag\.com|mixmag\.net|xlr8r\.com|pitchfork\.com|"
    r"thequietus\.com|factmag\.com|factmagazine|discogs\.com|musicbrainz\.org|"
    r"bandcamp\.com|beatport\.com|soundcloud\.com|youtube\.com|youtu\.be|"
    r"boilerroom\.tv|traxsource\.com|juno\.co\.uk|boomkat\.com|"
    r"electronicbeats\.net|djtechtools\.com|attackmagazine\.com",
    re.IGNORECASE,
)

# Terms that signal music/electronic-music relevance in titles and snippets
_MUSIC_RELEVANCE_TERMS = re.compile(
    r"\b(?:dj|producer|remix|techno|house|drum\s*(?:and|&|n)\s*bass|"
    r"electronic\s*music|rave|club|vinyl|label|release|mix|track|"
    r"bpm|ep\b|lp\b|album|record|boiler\s*room|soundsystem|"
    r"beatport|discogs|bandcamp|resident\s*advisor|soundcloud|"
    r"dance\s*music|edm|jungle|garage|dubstep|acid|trance|"
    r"breakbeat|ambient|industrial|synth|turntable|decks|"
    r"festival|warehouse|nightclub|set\b|lineup|b2b)\b",
    re.IGNORECASE,
)

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


class ArtistResearcher:
    """Deep-research service for a single artist/DJ.

    Performs a multi-step research pipeline: identity resolution via music
    databases, discography collection, gig history search, press/interview
    aggregation, and final compilation into a :class:`ResearchResult`.

    All external service dependencies are injected through constructor
    parameters, following the adapter pattern (CLAUDE.md Section 6).
    """

    def __init__(
        self,
        music_dbs: list[IMusicDatabaseProvider],
        web_search: IWebSearchProvider,
        article_scraper: IArticleProvider,
        llm: ILLMProvider,
        text_normalizer: type | None = None,
        cache: ICacheProvider | None = None,
        vector_store: IVectorStoreProvider | None = None,
    ) -> None:
        self._music_dbs = list(music_dbs)
        self._web_search = web_search
        self._article_scraper = article_scraper
        self._llm = llm
        self._cache = cache
        self._vector_store = vector_store
        self._logger: structlog.BoundLogger = get_logger(__name__)

    # -- Public API -----------------------------------------------------------

    async def research(
        self,
        artist_name: str,
        before_date: date | None = None,
        city: str | None = None,
    ) -> ResearchResult:
        """Execute the full research pipeline for a single artist.

        Parameters
        ----------
        artist_name:
            The raw artist/DJ name to research.
        before_date:
            If supplied, constrain discography and event results to entries
            before this date (useful for era-specific flier analysis).
        city:
            Optional city hint from the venue entity for geographic
            disambiguation and context-aware search queries.

        Returns
        -------
        ResearchResult
            A compiled research result containing the artist profile,
            discography, gig history, press references, confidence scores,
            and any warnings about data gaps.
        """
        normalized_name = normalize_artist_name(artist_name)
        self._logger.info(
            "Starting artist research",
            artist=normalized_name,
            before_date=str(before_date) if before_date else None,
        )

        warnings: list[str] = []
        sources_consulted: list[str] = []

        # Step 1 — IDENTIFY
        discogs_id, musicbrainz_id, id_confidence, provider_ids = (
            await self._search_music_databases(normalized_name)
        )
        if discogs_id is None and musicbrainz_id is None and not provider_ids:
            warnings.append("Artist not found in music databases")
            self._logger.warning("Artist not found in any music database", artist=normalized_name)

        # Step 2 — DISCOGRAPHY
        releases, labels = await self._fetch_discography(
            normalized_name, discogs_id, musicbrainz_id, before_date, provider_ids
        )
        if discogs_id or musicbrainz_id or provider_ids:
            sources_consulted.append("music_databases")
        disco_confidence = min(1.0, len(releases) * 0.1) if releases else 0.0

        # Step 3 — GIG HISTORY
        appearances = await self._search_gig_history(normalized_name, before_date, city=city)
        if appearances:
            sources_consulted.append("web_search_gigs")
        gig_confidence = min(1.0, len(appearances) * 0.15) if appearances else 0.0

        # Step 4 — PRESS
        articles = await self._search_press(normalized_name, before_date, city=city)
        if articles:
            sources_consulted.append("web_search_press")
        press_confidence = min(1.0, len(articles) * 0.12) if articles else 0.0

        # Step 4.5 — CORPUS RETRIEVAL (RAG)
        corpus_refs = await self._retrieve_from_corpus(normalized_name, before_date)
        if corpus_refs:
            sources_consulted.append("rag_corpus")
            # Merge corpus refs, deduplicating by title similarity
            existing_titles = {a.title.lower().strip() for a in articles if a.title}
            for ref in corpus_refs:
                ref_title_lower = ref.title.lower().strip() if ref.title else ""
                if ref_title_lower and ref_title_lower not in existing_titles:
                    existing_titles.add(ref_title_lower)
                    articles.append(ref)

        # Step 4.7 — PROFILE SYNTHESIS (only if at least 2 data sources)
        profile_summary: str | None = None
        data_source_count = sum([
            bool(discogs_id or musicbrainz_id),
            bool(releases),
            bool(appearances),
            bool(articles),
        ])
        if data_source_count >= 2:
            profile_summary = await self._synthesize_profile(
                normalized_name, releases, appearances, articles, labels, city=city
            )

        # Step 4.8 — GEOGRAPHIC EXTRACTION
        artist_city, artist_region, artist_country = self._extract_artist_geography(
            appearances, city
        )

        # Step 5 — COMPILE
        overall_confidence = calculate_confidence(
            scores=[id_confidence, disco_confidence, gig_confidence, press_confidence],
            weights=[3.0, 2.0, 1.5, 1.5],
        )

        if not releases:
            warnings.append("No releases found for artist")
        if not appearances:
            warnings.append("No event appearances found for artist")
        if not articles:
            warnings.append("No press articles found for artist")

        artist = Artist(
            name=normalized_name,
            discogs_id=int(discogs_id) if discogs_id and discogs_id.isdigit() else None,
            musicbrainz_id=musicbrainz_id,
            bandcamp_url=provider_ids.get("bandcamp"),
            beatport_url=(
                f"https://www.beatport.com/artist/{provider_ids['beatport']}"
                if provider_ids.get("beatport")
                else None
            ),
            confidence=overall_confidence,
            releases=releases,
            labels=labels,
            appearances=appearances,
            articles=articles,
            profile_summary=profile_summary,
            city=artist_city,
            region=artist_region,
            country=artist_country,
        )

        result = ResearchResult(
            entity_type=EntityType.ARTIST,
            entity_name=normalized_name,
            artist=artist,
            sources_consulted=sources_consulted,
            confidence=overall_confidence,
            warnings=warnings,
        )

        self._logger.info(
            "Artist research complete",
            artist=normalized_name,
            confidence=round(overall_confidence, 3),
            releases=len(releases),
            appearances=len(appearances),
            articles=len(articles),
            warnings=len(warnings),
        )

        return result

    # -- Private helpers -------------------------------------------------------

    async def _search_music_databases(
        self, name: str
    ) -> tuple[str | None, str | None, float, dict[str, str]]:
        """Search all music database providers for the artist.

        Returns
        -------
        tuple[str | None, str | None, float, dict[str, str]]
            (discogs_id, musicbrainz_id, identification_confidence, provider_ids)
            where ``provider_ids`` maps provider name to the matched artist ID
            for every provider that found a match.
        """
        cached = await self._cache_get(f"artist_search:{name}")
        if cached is not None:
            self._logger.debug("Cache hit for artist search", artist=name)
            provider_ids = cached.get("provider_ids", {})
            return (
                cached["discogs_id"],
                cached["musicbrainz_id"],
                cached["confidence"],
                provider_ids,
            )

        discogs_id: str | None = None
        musicbrainz_id: str | None = None
        best_confidence = 0.0
        provider_ids: dict[str, str] = {}

        for provider in self._music_dbs:
            provider_name = provider.get_provider_name()
            try:
                results: list[ArtistSearchResult] = await provider.search_artist(name)
            except (ResearchError, RateLimitError) as exc:
                self._logger.warning(
                    "Music database search failed",
                    provider=provider_name,
                    error=str(exc),
                )
                continue
            except Exception as exc:
                self._logger.error(
                    "Unexpected error searching music database",
                    provider=provider_name,
                    error=str(exc),
                )
                continue

            if not results:
                continue

            # Build candidate names for fuzzy matching
            candidate_map: dict[str, ArtistSearchResult] = {r.name: r for r in results}
            match_result = fuzzy_match(name, list(candidate_map.keys()), threshold=0.7)

            if match_result is None:
                continue

            matched_name, score = match_result
            matched = candidate_map[matched_name]

            # Use the higher of fuzzy score and provider-reported confidence
            effective_confidence = max(score, matched.confidence)

            # Store the matched ID for this provider
            provider_ids[provider_name] = matched.id

            if "discogs" in provider_name.lower():
                discogs_id = matched.id
                best_confidence = max(best_confidence, effective_confidence)
            elif "musicbrainz" in provider_name.lower():
                musicbrainz_id = matched.id
                best_confidence = max(best_confidence, effective_confidence)
            else:
                best_confidence = max(best_confidence, effective_confidence)

        # Cross-reference bonus: if multiple databases agree, boost confidence
        confirmed_count = sum(1 for _ in provider_ids)
        if confirmed_count >= 2:
            bonus = min(0.2, confirmed_count * 0.05)
            best_confidence = min(1.0, best_confidence + bonus)
            self._logger.info(
                "Cross-reference match found",
                artist=name,
                providers=list(provider_ids.keys()),
                bonus=bonus,
            )

        await self._cache_set(
            f"artist_search:{name}",
            {
                "discogs_id": discogs_id,
                "musicbrainz_id": musicbrainz_id,
                "confidence": best_confidence,
                "provider_ids": provider_ids,
            },
        )

        return discogs_id, musicbrainz_id, best_confidence, provider_ids

    async def _fetch_discography(
        self,
        artist_name: str,
        discogs_id: str | None,
        musicbrainz_id: str | None,
        before_date: date | None,
        provider_ids: dict[str, str] | None = None,
    ) -> tuple[list[Release], list[Label]]:
        """Fetch releases and labels from matched database entries.

        Attempts Discogs API first, falls back to scrape provider, then
        supplements with MusicBrainz data and any additional providers
        (Bandcamp, Beatport, etc.) that found a match.
        """
        releases: list[Release] = []
        labels: list[Label] = []
        seen_titles: set[str] = set()
        fetched_providers: set[str] = set()

        # Try Discogs providers (API first, then scrape fallback)
        discogs_providers = [
            p for p in self._music_dbs if "discogs" in p.get_provider_name().lower()
        ]

        if discogs_id and discogs_providers:
            for provider in discogs_providers:
                try:
                    provider_releases = await provider.get_artist_releases(discogs_id, before_date)
                    for release in provider_releases:
                        title_key = release.title.lower().strip()
                        if title_key not in seen_titles:
                            seen_titles.add(title_key)
                            releases.append(release)

                    provider_labels = await provider.get_artist_labels(discogs_id)
                    labels.extend(provider_labels)
                    fetched_providers.add(provider.get_provider_name())
                    self._logger.info(
                        "Discography fetched",
                        provider=provider.get_provider_name(),
                        releases=len(provider_releases),
                        labels=len(provider_labels),
                    )
                    break  # Success — skip fallback providers
                except RateLimitError:
                    self._logger.warning(
                        "Rate limited, trying next provider",
                        provider=provider.get_provider_name(),
                    )
                    continue
                except ResearchError as exc:
                    self._logger.warning(
                        "Discography fetch failed, trying next provider",
                        provider=provider.get_provider_name(),
                        error=str(exc),
                    )
                    continue

        # Supplement with MusicBrainz data if available
        if musicbrainz_id:
            mb_providers = [
                p for p in self._music_dbs if "musicbrainz" in p.get_provider_name().lower()
            ]
            for provider in mb_providers:
                try:
                    mb_releases = await provider.get_artist_releases(musicbrainz_id, before_date)
                    for release in mb_releases:
                        title_key = release.title.lower().strip()
                        if title_key not in seen_titles:
                            seen_titles.add(title_key)
                            releases.append(release)

                    if not labels:
                        mb_labels = await provider.get_artist_labels(musicbrainz_id)
                        labels.extend(mb_labels)

                    fetched_providers.add(provider.get_provider_name())
                    self._logger.debug(
                        "MusicBrainz supplement complete",
                        new_releases=len(mb_releases),
                    )
                    break
                except (ResearchError, RateLimitError) as exc:
                    self._logger.warning(
                        "MusicBrainz supplement failed",
                        error=str(exc),
                    )

        # Supplement with additional providers (Bandcamp, Beatport, etc.)
        if provider_ids:
            for provider in self._music_dbs:
                pname = provider.get_provider_name()
                if pname in fetched_providers or pname not in provider_ids:
                    continue
                if "discogs" in pname.lower() or "musicbrainz" in pname.lower():
                    continue

                artist_id = provider_ids[pname]
                try:
                    extra_releases = await provider.get_artist_releases(artist_id, before_date)
                    added = 0
                    for release in extra_releases:
                        title_key = release.title.lower().strip()
                        if title_key not in seen_titles:
                            seen_titles.add(title_key)
                            releases.append(release)
                            added += 1

                    extra_labels = await provider.get_artist_labels(artist_id)
                    labels.extend(extra_labels)
                    fetched_providers.add(pname)

                    self._logger.info(
                        "Additional provider supplement complete",
                        provider=pname,
                        new_releases=added,
                        new_labels=len(extra_labels),
                    )
                except Exception as exc:
                    self._logger.warning(
                        "Additional provider fetch failed",
                        provider=pname,
                        error=str(exc),
                    )

        # Deduplicate labels by name
        seen_label_names: set[str] = set()
        unique_labels: list[Label] = []
        for label in labels:
            if label.name.lower() not in seen_label_names:
                seen_label_names.add(label.name.lower())
                unique_labels.append(label)

        return releases, unique_labels

    async def _search_gig_history(
        self, name: str, before_date: date | None, city: str | None = None
    ) -> list[EventAppearance]:
        """Search the web for past event appearances by the artist.

        Uses RA.co (Resident Advisor) as the primary source for event
        listings, supplemented by general web search. When ``city`` is
        provided, city-qualified queries are included to improve
        disambiguation. Performs deeper follow-up searches if initial
        results are sparse.
        """
        # Phase 1: Primary queries — RA.co first, then general
        primary_queries = [
            f'site:ra.co/dj "{name}"',
            f'site:ra.co/events "{name}"',
            f'"{name}" DJ set event lineup',
            f'"{name}" live club night rave',
        ]
        if city:
            primary_queries.insert(2, f'"{name}" DJ "{city}" event')

        all_results: list[SearchResult] = []
        for query in primary_queries:
            try:
                results = await self._web_search.search(
                    query=query,
                    num_results=_GIG_SEARCH_LIMIT,
                    before_date=before_date,
                )
                all_results.extend(results)
            except ResearchError as exc:
                self._logger.warning(
                    "Gig history search failed",
                    query=query,
                    error=str(exc),
                )

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique_results: list[SearchResult] = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        # Phase 2: Deepen search if results are thin
        if len(unique_results) < _MIN_ADEQUATE_APPEARANCES:
            deeper_queries = [
                f'"{name}" resident advisor event',
                f'"{name}" electronic music festival',
                f'"{name}" boiler room OR dekmantel OR fabric OR berghain',
                f'"{name}" DJ biography',
            ]
            for query in deeper_queries:
                try:
                    results = await self._web_search.search(
                        query=query,
                        num_results=_GIG_SEARCH_LIMIT,
                        before_date=before_date,
                    )
                    for r in results:
                        if r.url not in seen_urls:
                            seen_urls.add(r.url)
                            unique_results.append(r)
                except ResearchError:
                    continue

            self._logger.info(
                "Deepened gig search",
                artist=name,
                total_results=len(unique_results),
            )

        if not unique_results:
            return []

        # Prioritize RA.co results first in the snippet list
        ra_results = [r for r in unique_results if "ra.co" in r.url or "residentadvisor" in r.url]
        other_results = [r for r in unique_results if r not in ra_results]
        ordered = ra_results + other_results

        # Use LLM to extract structured event data from search snippets
        snippet_text = "\n".join(
            f"- {r.title}: {r.snippet or '(no snippet)'} [{r.url}]"
            for r in ordered[:_GIG_SEARCH_LIMIT]
        )

        system_prompt = (
            "You are a music event data extractor. Given web search results about "
            "a DJ/artist, extract any identifiable past event appearances. "
            "Pay special attention to Resident Advisor (ra.co) results — these are "
            "the most authoritative source for electronic music event listings. "
            "Return one appearance per line in the format:\n"
            "EVENT_NAME | VENUE | DATE (YYYY-MM-DD or unknown) | SOURCE_URL\n"
            "Only include results that clearly indicate the artist performed at an event. "
            "If no clear events are found, return NONE."
        )
        user_prompt = (
            f"Extract event appearances for artist '{name}' from these search results:\n\n"
            f"{snippet_text}"
        )

        try:
            llm_response = await self._llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=3000,
            )
        except Exception as exc:
            self._logger.warning("LLM gig extraction failed", error=str(exc))
            return []

        return self._parse_event_appearances(llm_response)

    async def _retrieve_from_corpus(
        self, artist_name: str, before_date: date | None
    ) -> list[ArticleReference]:
        """Retrieve relevant passages from the RAG corpus for this artist.

        Returns an empty list if no vector store is configured or available.
        """
        if not self._vector_store or not self._vector_store.is_available():
            return []

        query = f"{artist_name} DJ electronic music history career"
        filters: dict[str, object] = {}
        if before_date:
            filters["date"] = {"$lte": before_date.isoformat()}

        try:
            chunks = await self._vector_store.query(
                query_text=query, top_k=15, filters=filters if filters else None
            )
        except Exception as exc:
            self._logger.debug("Corpus retrieval failed", artist=artist_name, error=str(exc))
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
                artist=artist_name,
                chunks_retrieved=len(refs),
            )
        return refs

    async def _search_press(
        self, name: str, before_date: date | None, city: str | None = None
    ) -> list[ArticleReference]:
        """Search for press articles and interviews mentioning the artist.

        Prioritizes RA.co (Resident Advisor) as the premier source for
        electronic music artist profiles and event coverage. When ``city``
        is provided, a city-qualified query is added for disambiguation.
        Performs adaptive deepening when initial results are insufficient.
        """
        # Phase 1: Primary queries — RA.co artist profiles, then major press
        primary_queries = [
            f'site:ra.co "{name}"',
            f'"{name}" interview OR profile "Resident Advisor" OR "DJ Mag" OR Mixmag',
            f'"{name}" electronic music artist biography',
        ]
        if city:
            primary_queries.append(f'"{name}" "{city}" electronic music DJ')

        all_results: list[SearchResult] = []
        seen_urls: set[str] = set()

        for query in primary_queries:
            try:
                results = await self._web_search.search(
                    query=query,
                    num_results=_PRESS_SEARCH_LIMIT,
                    before_date=before_date,
                )
                for r in results:
                    if r.url not in seen_urls:
                        seen_urls.add(r.url)
                        all_results.append(r)
            except ResearchError as exc:
                self._logger.warning(
                    "Press search failed",
                    query=query,
                    error=str(exc),
                )

        # Phase 2: Deepen if results are thin
        if len(all_results) < _MIN_ADEQUATE_ARTICLES:
            deeper_queries = [
                f'"{name}" XLR8R OR Pitchfork OR "Fact Magazine" OR Mixmag',
                f'"{name}" electronic music producer DJ',
                f'"{name}" record label release announcement',
            ]
            for query in deeper_queries:
                try:
                    results = await self._web_search.search(
                        query=query,
                        num_results=_PRESS_SEARCH_LIMIT,
                        before_date=before_date,
                    )
                    for r in results:
                        if r.url not in seen_urls:
                            seen_urls.add(r.url)
                            all_results.append(r)
                except ResearchError:
                    continue

            self._logger.info(
                "Deepened press search",
                artist=name,
                total_results=len(all_results),
            )

        if not all_results:
            return []

        # Filter out results that are clearly not about music/electronic music
        relevant_results = [r for r in all_results if self._is_music_relevant(r)]

        if len(relevant_results) < len(all_results):
            self._logger.info(
                "press_relevance_filter",
                artist=name,
                before=len(all_results),
                after=len(relevant_results),
                removed=len(all_results) - len(relevant_results),
            )

        if not relevant_results:
            return []

        # Prioritize RA.co and tier-1 sources for scraping
        def _sort_key(r: SearchResult) -> int:
            url = r.url.lower()
            if "ra.co" in url or "residentadvisor" in url:
                return 0
            if any(d in url for d in ("djmag.com", "mixmag.net")):
                return 1
            return 2

        relevant_results.sort(key=_sort_key)

        # Extract article content for the top results
        articles: list[ArticleReference] = []
        extraction_tasks = [
            self._extract_article_reference(result, name)
            for result in relevant_results[:_MAX_PRESS_ARTICLES]
        ]
        extracted = await asyncio.gather(*extraction_tasks, return_exceptions=True)

        for item in extracted:
            if isinstance(item, ArticleReference):
                articles.append(item)
            elif isinstance(item, Exception):
                self._logger.debug("Article extraction failed", error=str(item))

        return articles

    async def _extract_article_reference(
        self, search_result: SearchResult, artist_name: str
    ) -> ArticleReference:
        """Extract content from a single search result and build an ArticleReference."""
        content = await self._article_scraper.extract_content(search_result.url)

        snippet: str | None = None
        article_type = "article"

        if content and content.text:
            # Extract the most relevant snippet mentioning the artist
            snippet = self._extract_relevant_snippet(content.text, artist_name)
            # Determine article type from title/content
            title_lower = (content.title or "").lower()
            if "interview" in title_lower:
                article_type = "interview"
            elif "review" in title_lower:
                article_type = "review"
            elif "mix" in title_lower or "podcast" in title_lower:
                article_type = "mix"
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
        """Assign a citation authority tier (1–6) based on URL domain patterns.

        Tier 1: Major music publications (RA, DJ Mag, Mixmag)
        Tier 2: Specialist music press (XLR8R, Pitchfork, Fact)
        Tier 3: Music databases (Discogs, MusicBrainz, Bandcamp)
        Tier 4: Media platforms (YouTube, SoundCloud, Wikipedia)
        Tier 5: Social media and forums (Reddit, Facebook, etc.)
        Tier 6: Unknown / uncategorized sources
        """
        for pattern, tier in _CITATION_TIER_PATTERNS:
            if pattern.search(source_url):
                return tier
        return 6

    # -- Parsing helpers -------------------------------------------------------

    def _parse_event_appearances(self, llm_text: str) -> list[EventAppearance]:
        """Parse LLM-generated event appearance text into structured objects."""
        appearances: list[EventAppearance] = []

        if "NONE" in llm_text.strip().upper():
            return appearances

        for line in llm_text.strip().splitlines():
            line = line.strip().lstrip("- ")
            if not line or "|" not in line:
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue

            event_name = parts[0] if parts[0] and parts[0].lower() != "unknown" else None
            venue = parts[1] if len(parts) > 1 and parts[1].lower() != "unknown" else None
            date_str = parts[2] if len(parts) > 2 else None
            source_url = parts[3] if len(parts) > 3 else None

            event_date: date | None = None
            if date_str and date_str.lower() != "unknown":
                with contextlib.suppress(ValueError):
                    event_date = date.fromisoformat(date_str)

            appearances.append(
                EventAppearance(
                    event_name=event_name,
                    venue=venue,
                    date=event_date,
                    source="web_search",
                    source_url=source_url,
                )
            )

        return appearances

    @staticmethod
    def _is_music_relevant(result: SearchResult) -> bool:
        """Check whether a search result is plausibly about music/electronic music.

        Results from known music domains always pass. For other domains,
        the title and snippet must contain at least one music-related term.
        """
        # Known music domains always pass
        if _MUSIC_DOMAINS.search(result.url):
            return True

        # Check title and snippet for music relevance signals
        text_to_check = f"{result.title or ''} {result.snippet or ''}"
        return bool(_MUSIC_RELEVANCE_TERMS.search(text_to_check))

    @staticmethod
    def _extract_relevant_snippet(text: str, artist_name: str, max_length: int = 500) -> str:
        """Extract the most relevant text snippet mentioning the artist."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        name_lower = artist_name.lower()

        # Find sentences mentioning the artist
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

    # -- Geography extraction --------------------------------------------------

    @staticmethod
    def _extract_artist_geography(
        appearances: list[EventAppearance],
        flier_city: str | None,
    ) -> tuple[str | None, str | None, str | None]:
        """Determine the artist's primary geographic base from appearance data.

        Uses a heuristic: if >= 50% of appearances with a known city share
        the same city, return that city.  Otherwise returns ``(None, None, None)``.

        Returns
        -------
        tuple[str | None, str | None, str | None]
            (city, region, country)
        """
        city_counts: dict[str, int] = {}
        for app in appearances:
            if app.city:
                key = app.city.strip().lower()
                city_counts[key] = city_counts.get(key, 0) + 1

        if not city_counts:
            return (None, None, None)

        top_city = max(city_counts, key=city_counts.get)  # type: ignore[arg-type]
        total_with_city = sum(city_counts.values())

        if city_counts[top_city] >= total_with_city * 0.5:
            return (top_city.title(), None, None)

        return (None, None, None)

    # -- Profile synthesis -----------------------------------------------------

    async def _synthesize_profile(
        self,
        artist_name: str,
        releases: list[Release],
        appearances: list[EventAppearance],
        articles: list[ArticleReference],
        labels: list[Label],
        city: str | None = None,
    ) -> str | None:
        """Use LLM to synthesize a 2-3 sentence artist profile summary."""
        # Build context from available data
        context_parts: list[str] = []

        if city:
            context_parts.append(f"Flier city: {city}")

        if releases:
            release_titles = [r.title for r in releases[:10]]
            context_parts.append(f"Releases: {', '.join(release_titles)}")

        if labels:
            label_names = [lb.name for lb in labels[:10]]
            context_parts.append(f"Labels: {', '.join(label_names)}")

        if appearances:
            event_names = [a.event_name for a in appearances[:10] if a.event_name]
            if event_names:
                context_parts.append(f"Events: {', '.join(event_names)}")

        if articles:
            article_titles = [a.title for a in articles[:5] if a.title]
            if article_titles:
                context_parts.append(f"Articles: {', '.join(article_titles)}")

        if not context_parts:
            return None

        context_text = "\n".join(context_parts)

        system_prompt = (
            "You are a music journalist specializing in electronic and dance music. "
            "Write concise, factual artist profiles."
        )
        user_prompt = (
            f"Based on the following data about '{artist_name}', write a 2-3 sentence "
            "profile summary placing them in the context of electronic/dance music culture. "
            "Include the city/region they are based in if apparent from the data. "
            "Be specific and factual. Do not speculate.\n\n"
            f"{context_text}"
        )

        try:
            response = await self._llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=300,
            )
            summary = response.strip()
            return summary if summary else None
        except Exception as exc:
            self._logger.debug("Profile synthesis failed", artist=artist_name, error=str(exc))
            return None

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

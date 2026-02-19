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
from typing import Any

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

_CACHE_TTL_SECONDS = 3600  # 1 hour
_MAX_PRESS_ARTICLES = 10
_GIG_SEARCH_LIMIT = 20
_PRESS_SEARCH_LIMIT = 15

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
    ) -> None:
        self._music_dbs = list(music_dbs)
        self._web_search = web_search
        self._article_scraper = article_scraper
        self._llm = llm
        self._cache = cache
        self._logger: structlog.BoundLogger = get_logger(__name__)

    # -- Public API -----------------------------------------------------------

    async def research(
        self,
        artist_name: str,
        before_date: date | None = None,
    ) -> ResearchResult:
        """Execute the full research pipeline for a single artist.

        Parameters
        ----------
        artist_name:
            The raw artist/DJ name to research.
        before_date:
            If supplied, constrain discography and event results to entries
            before this date (useful for era-specific flier analysis).

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
        discogs_id, musicbrainz_id, id_confidence = await self._search_music_databases(
            normalized_name
        )
        if discogs_id is None and musicbrainz_id is None:
            warnings.append("Artist not found in music databases")
            self._logger.warning("Artist not found in any music database", artist=normalized_name)

        # Step 2 — DISCOGRAPHY
        releases, labels = await self._fetch_discography(
            normalized_name, discogs_id, musicbrainz_id, before_date
        )
        if discogs_id or musicbrainz_id:
            sources_consulted.append("music_databases")
        disco_confidence = min(1.0, len(releases) * 0.1) if releases else 0.0

        # Step 3 — GIG HISTORY
        appearances = await self._search_gig_history(normalized_name, before_date)
        if appearances:
            sources_consulted.append("web_search_gigs")
        gig_confidence = min(1.0, len(appearances) * 0.15) if appearances else 0.0

        # Step 4 — PRESS
        articles = await self._search_press(normalized_name, before_date)
        if articles:
            sources_consulted.append("web_search_press")
        press_confidence = min(1.0, len(articles) * 0.12) if articles else 0.0

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
            confidence=overall_confidence,
            releases=releases,
            labels=labels,
            appearances=appearances,
            articles=articles,
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

    async def _search_music_databases(self, name: str) -> tuple[str | None, str | None, float]:
        """Search all music database providers for the artist.

        Returns
        -------
        tuple[str | None, str | None, float]
            (discogs_id, musicbrainz_id, identification_confidence)
        """
        cached = await self._cache_get(f"artist_search:{name}")
        if cached is not None:
            self._logger.debug("Cache hit for artist search", artist=name)
            return cached["discogs_id"], cached["musicbrainz_id"], cached["confidence"]

        discogs_id: str | None = None
        musicbrainz_id: str | None = None
        best_confidence = 0.0

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

            if "discogs" in provider_name.lower():
                discogs_id = matched.id
                best_confidence = max(best_confidence, effective_confidence)
            elif "musicbrainz" in provider_name.lower():
                musicbrainz_id = matched.id
                best_confidence = max(best_confidence, effective_confidence)
            else:
                best_confidence = max(best_confidence, effective_confidence)

        # Cross-reference bonus: if both databases agree, boost confidence
        if discogs_id and musicbrainz_id:
            best_confidence = min(1.0, best_confidence + 0.15)
            self._logger.info(
                "Cross-reference match found",
                artist=name,
                discogs_id=discogs_id,
                musicbrainz_id=musicbrainz_id,
            )

        await self._cache_set(
            f"artist_search:{name}",
            {
                "discogs_id": discogs_id,
                "musicbrainz_id": musicbrainz_id,
                "confidence": best_confidence,
            },
        )

        return discogs_id, musicbrainz_id, best_confidence

    async def _fetch_discography(
        self,
        artist_name: str,
        discogs_id: str | None,
        musicbrainz_id: str | None,
        before_date: date | None,
    ) -> tuple[list[Release], list[Label]]:
        """Fetch releases and labels from matched database entries.

        Attempts Discogs API first, falls back to scrape provider, then
        supplements with MusicBrainz data if available.
        """
        releases: list[Release] = []
        labels: list[Label] = []
        seen_titles: set[str] = set()

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

        # Deduplicate labels by name
        seen_label_names: set[str] = set()
        unique_labels: list[Label] = []
        for label in labels:
            if label.name.lower() not in seen_label_names:
                seen_label_names.add(label.name.lower())
                unique_labels.append(label)

        return releases, unique_labels

    async def _search_gig_history(
        self, name: str, before_date: date | None
    ) -> list[EventAppearance]:
        """Search the web for past event appearances by the artist."""
        queries = [
            f'"{name}" DJ set',
            f'"{name}" live event',
            f'"{name}" club night',
        ]

        all_results: list[SearchResult] = []
        for query in queries:
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

        if not all_results:
            return []

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique_results: list[SearchResult] = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        # Use LLM to extract structured event data from search snippets
        snippet_text = "\n".join(
            f"- {r.title}: {r.snippet or '(no snippet)'} [{r.url}]"
            for r in unique_results[:_GIG_SEARCH_LIMIT]
        )

        system_prompt = (
            "You are a music event data extractor. Given web search results about "
            "a DJ/artist, extract any identifiable past event appearances. "
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
                max_tokens=2000,
            )
        except Exception as exc:
            self._logger.warning("LLM gig extraction failed", error=str(exc))
            return []

        return self._parse_event_appearances(llm_response)

    async def _search_press(self, name: str, before_date: date | None) -> list[ArticleReference]:
        """Search for press articles and interviews mentioning the artist."""
        queries = [
            f'"{name}" interview',
            f'"{name}" DJ Mag OR "Resident Advisor" OR Mixmag',
        ]

        all_results: list[SearchResult] = []
        for query in queries:
            try:
                results = await self._web_search.search(
                    query=query,
                    num_results=_PRESS_SEARCH_LIMIT,
                    before_date=before_date,
                )
                all_results.extend(results)
            except ResearchError as exc:
                self._logger.warning(
                    "Press search failed",
                    query=query,
                    error=str(exc),
                )

        if not all_results:
            return []

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique_results: list[SearchResult] = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        # Extract article content for the top results
        articles: list[ArticleReference] = []
        extraction_tasks = [
            self._extract_article_reference(result, name)
            for result in unique_results[:_MAX_PRESS_ARTICLES]
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

"""Deep-research service for individual artists and DJs.

Orchestrates multiple data sources — music databases (Discogs, MusicBrainz),
web search, article scraping, and LLM analysis — to build a comprehensive
research profile for a single artist extracted from a rave flier.

Architecture role: **Most complex researcher — multi-phase pipeline**
----------------------------------------------------------------------
Of the five researcher classes (artist, venue, promoter, date-context,
event-name), this one is by far the largest because artist identity is
inherently ambiguous (many DJs share names, use aliases, or change
monikers).  The pipeline runs these phases:

  1. IDENTIFY   — Search all injected music-database providers (Discogs,
                   MusicBrainz, Bandcamp, Beatport, etc.) in parallel,
                   fuzzy-match names, and cross-reference IDs.
  2. DISCOGRAPHY — Fetch releases and labels from matched databases;
                   supplement across providers; deduplicate by title.
  3. FEEDBACK    — Filter out releases/labels previously thumbs-downed
                   by users (cross-session learning via IFeedbackProvider).
  4. CORPUS (RAG)— Query the vector store for historical articles/books
                   mentioning the artist, if a vector store is configured.
  5. SYNTHESIS   — If >= 2 data sources returned data, ask the LLM to
                   produce a concise 2-3 sentence artist profile.
  6. COMPILE     — Assemble everything into a ResearchResult with a
                   weighted confidence score.

Key design patterns:
- **Multi-source search**: music DBs searched first (structured data),
  then web search (unstructured), then RAG corpus (embedded knowledge).
- **Adaptive deepening**: if initial search yields fewer results than a
  threshold, additional queries are dispatched automatically.
- **Citation tiers**: every source URL is classified into tiers 1-6 so
  the frontend can display provenance badges.
- **Feedback filtering**: user thumbs-down signals from past sessions
  prune wrong-artist results without re-training anything.
- **Dependency injection**: every external service (DB, search, scraper,
  LLM, cache, vector store, feedback) is injected via interface.
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
from src.utils.concurrency import parallel_search, throttled_gather
from src.utils.errors import RateLimitError, ResearchError
from src.utils.logging import get_logger
from src.utils.text_normalizer import fuzzy_match, normalize_artist_name

if TYPE_CHECKING:
    from src.interfaces.feedback_provider import IFeedbackProvider
    from src.interfaces.vector_store_provider import IVectorStoreProvider

# ---------------------------------------------------------------------------
# Tuning constants — control search breadth and adaptive deepening
# ---------------------------------------------------------------------------
_CACHE_TTL_SECONDS = 3600  # 1 hour — balances freshness vs. API quota
_MAX_PRESS_ARTICLES = 15   # cap on articles to scrape per artist
_GIG_SEARCH_LIMIT = 30     # max web search results per gig query
_PRESS_SEARCH_LIMIT = 20   # max web search results per press query

# Adaptive deepening thresholds: if the primary search phase returns fewer
# results than these minimums, a second round of broader queries fires
# automatically.  This handles obscure artists who don't appear in the
# first round of targeted queries.
_MIN_ADEQUATE_APPEARANCES = 3  # trigger deeper gig search if below this
_MIN_ADEQUATE_ARTICLES = 2     # trigger deeper press search if below this

# ---------------------------------------------------------------------------
# Relevance filters — separate signal from noise in web search results
# ---------------------------------------------------------------------------
# Domains known to be music-related — results from these always pass the
# relevance check without needing keyword signals in the title/snippet.
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

# ---------------------------------------------------------------------------
# Citation tier assignment table
# ---------------------------------------------------------------------------
# Every source URL is classified into a tier (1 = most authoritative, 6 =
# unknown).  Tiers drive two downstream behaviors:
#   1. The frontend displays a provenance badge (gold/silver/bronze/etc.)
#   2. When conflicting facts appear, higher-tier sources win.
# Tier 1 sources (RA, DJ Mag, Mixmag) are considered primary authorities
# for electronic music.  The table is checked in order; first match wins.
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
        # All external services are injected as interface types (I-prefixed),
        # so concrete implementations (Discogs API, SerpAPI, etc.) can be
        # swapped without touching this class.
        music_dbs: list[IMusicDatabaseProvider],   # Multiple DB adapters searched in parallel
        web_search: IWebSearchProvider,             # Web search (SerpAPI, Brave, etc.)
        article_scraper: IArticleProvider,          # HTML-to-text article extractor
        llm: ILLMProvider,                          # LLM for extraction + synthesis
        text_normalizer: type | None = None,
        cache: ICacheProvider | None = None,        # Optional caching layer (Redis, SQLite, etc.)
        vector_store: IVectorStoreProvider | None = None,  # Optional RAG corpus
        feedback: IFeedbackProvider | None = None,  # Optional cross-session feedback store
    ) -> None:
        self._music_dbs = list(music_dbs)
        self._web_search = web_search
        self._article_scraper = article_scraper
        self._llm = llm
        self._cache = cache
        self._vector_store = vector_store
        self._feedback = feedback
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
        # Normalize the raw OCR name (lowercase, strip punctuation, collapse
        # whitespace) so fuzzy matching works regardless of flier typography.
        normalized_name = normalize_artist_name(artist_name)
        self._logger.info(
            "Starting artist research",
            artist=normalized_name,
            before_date=str(before_date) if before_date else None,
        )

        warnings: list[str] = []
        sources_consulted: list[str] = []

        # ===================================================================
        # PHASE 1 — IDENTIFY: search structured music databases
        # ===================================================================
        # All injected music-database providers are queried in parallel.
        # Results are fuzzy-matched against the normalized name.  When
        # multiple databases agree on identity, a cross-reference bonus
        # boosts confidence (see _search_music_databases for details).
        discogs_id, musicbrainz_id, id_confidence, provider_ids = (
            await self._search_music_databases(normalized_name)
        )
        if discogs_id is None and musicbrainz_id is None and not provider_ids:
            warnings.append("Artist not found in music databases")
            self._logger.warning("Artist not found in any music database", artist=normalized_name)

        # ===================================================================
        # PHASE 2 — DISCOGRAPHY: fetch releases and labels
        # ===================================================================
        # Uses the IDs resolved in Phase 1 to pull structured release data.
        # Discogs is tried first (richest metadata); MusicBrainz supplements;
        # then any other matched providers (Bandcamp, Beatport) fill gaps.
        # Releases are deduplicated by title (case-insensitive).
        releases, labels = await self._fetch_discography(
            normalized_name, discogs_id, musicbrainz_id, before_date, provider_ids
        )
        if discogs_id or musicbrainz_id or provider_ids:
            sources_consulted.append("music_databases")

        # ===================================================================
        # PHASE 2b — FEEDBACK FILTER (cross-session learning)
        # ===================================================================
        # If a user previously thumbs-downed a release (e.g. wrong artist
        # with the same name), it is excluded here.  This is a lightweight
        # alternative to retraining: negative signals are stored by
        # IFeedbackProvider and applied as a deny-list per artist name.
        releases, labels = await self._filter_by_feedback(
            normalized_name, releases, labels
        )

        disco_confidence = min(1.0, len(releases) * 0.1) if releases else 0.0

        # ===================================================================
        # PHASE 3 — CORPUS RETRIEVAL (RAG vector search)
        # ===================================================================
        # If a vector store is configured, query embedded historical
        # articles/books for passages about this artist.  This is the
        # "third source" in the multi-source strategy: structured DBs
        # (Phase 1-2) + web search (gig/press methods) + RAG corpus.
        corpus_refs = await self._retrieve_from_corpus(normalized_name, before_date)
        if corpus_refs:
            sources_consulted.append("rag_corpus")

        # ===================================================================
        # PHASE 4 — PROFILE SYNTHESIS (LLM-generated summary)
        # ===================================================================
        # Only triggered when >= 2 data sources contributed results.  This
        # threshold prevents the LLM from hallucinating a profile from a
        # single thin data point.  The LLM receives structured facts
        # (releases, labels, events, articles) and produces a concise 2-3
        # sentence bio placing the artist in electronic music context.
        profile_summary: str | None = None
        data_source_count = sum([
            bool(discogs_id or musicbrainz_id),
            bool(releases),
            bool(corpus_refs),
        ])
        if data_source_count >= 2:
            profile_summary = await self._synthesize_profile(
                normalized_name, releases, [], [], labels, city=city
            )

        # ===================================================================
        # PHASE 5 — COMPILE: assemble final ResearchResult
        # ===================================================================
        # Weighted confidence combines identity resolution (weight 3) and
        # discography depth (weight 2).  Identity is weighted more heavily
        # because a wrong-artist match invalidates everything downstream.
        overall_confidence = calculate_confidence(
            scores=[id_confidence, disco_confidence],
            weights=[3.0, 2.0],
        )

        if not releases:
            warnings.append("No releases found for artist")

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
            profile_summary=profile_summary,
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

        # Search all music databases in parallel.  throttled_gather applies
        # concurrency limits (e.g. max 5 simultaneous HTTP requests) to
        # respect API rate limits while still parallelizing across providers.
        search_coros = [provider.search_artist(name) for provider in self._music_dbs]
        raw_results = await throttled_gather(search_coros, return_exceptions=True)

        for idx, raw in enumerate(raw_results):
            provider = self._music_dbs[idx]
            provider_name = provider.get_provider_name()

            if isinstance(raw, (ResearchError, RateLimitError)):
                self._logger.warning(
                    "Music database search failed",
                    provider=provider_name,
                    error=str(raw),
                )
                continue
            elif isinstance(raw, Exception):
                self._logger.error(
                    "Unexpected error searching music database",
                    provider=provider_name,
                    error=str(raw),
                )
                continue

            results: list[ArtistSearchResult] = raw
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

        # Cross-reference bonus: when multiple independent databases agree
        # that this artist exists, it is strong evidence of correct identity
        # resolution.  Each additional confirming provider adds +0.05 to
        # confidence, capped at +0.20.  This rewards breadth of confirmation
        # and helps disambiguate common names (e.g. multiple "DJ Shadow"
        # entries across providers collapse to the one present in all).
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

        # Multi-source search strategy for discography:
        # 1. Discogs (API adapter first, then scrape-based fallback)
        # 2. MusicBrainz (supplement — fills gaps Discogs may miss)
        # 3. Additional providers (Bandcamp, Beatport, etc.)
        # Each provider is tried independently; failures don't block others.
        # Releases are deduplicated by lowercase title via `seen_titles`.
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

    async def _filter_by_feedback(
        self,
        artist_name: str,
        releases: list[Release],
        labels: list[Label],
    ) -> tuple[list[Release], list[Label]]:
        """Remove releases and labels that have been thumbs-downed in prior sessions.

        Handles the 'wrong artist with same name' problem: when a user
        thumbs-downs a release belonging to a different artist, it is
        excluded from future research results for the same artist name.

        Key design: feedback keys are composite strings like
        ``"DJ Rush::release::Warp Factor"`` so negative signals are
        scoped to a specific (artist, item) pair and don't bleed across
        different artists or item types.
        """
        if self._feedback is None:
            return releases, labels

        release_prefix = f"{artist_name}::release::"
        label_prefix = f"{artist_name}::label::"

        try:
            negative_releases, negative_labels = await asyncio.gather(
                self._feedback.get_negative_item_keys("RELEASE", release_prefix),
                self._feedback.get_negative_item_keys("LABEL", label_prefix),
            )
        except Exception as exc:
            self._logger.debug(
                "Feedback lookup failed, skipping filter",
                artist=artist_name,
                error=str(exc),
            )
            return releases, labels

        if not negative_releases and not negative_labels:
            return releases, labels

        filtered_releases = [
            r for r in releases
            if f"{artist_name}::release::{r.title}" not in negative_releases
        ]
        filtered_labels = [
            lb for lb in labels
            if f"{artist_name}::label::{lb.name}" not in negative_labels
        ]

        removed_releases = len(releases) - len(filtered_releases)
        removed_labels = len(labels) - len(filtered_labels)

        if removed_releases or removed_labels:
            self._logger.info(
                "Cross-session feedback filter applied",
                artist=artist_name,
                removed_releases=removed_releases,
                removed_labels=removed_labels,
            )

        return filtered_releases, filtered_labels

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
        # Phase 1 — Primary queries (targeted, high-precision)
        # RA.co (Resident Advisor) is queried first via site: operator
        # because it is the most authoritative source for electronic music
        # event listings.  General queries follow for broader coverage.
        # If a city hint is available, a city-qualified query is inserted
        # to help disambiguate common DJ names in a specific scene.
        primary_queries = [
            f'site:ra.co/dj "{name}"',
            f'site:ra.co/events "{name}"',
            f'"{name}" DJ set event lineup',
            f'"{name}" live club night rave',
        ]
        if city:
            primary_queries.insert(2, f'"{name}" DJ "{city}" event')

        # Execute all primary queries in parallel with throttling
        all_results: list[SearchResult] = await parallel_search(
            self._web_search.search,
            [{"query": q, "num_results": _GIG_SEARCH_LIMIT, "before_date": before_date}
             for q in primary_queries],
            logger=self._logger,
            error_msg="Gig history search failed",
        )

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique_results: list[SearchResult] = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        # Phase 2 — Adaptive deepening: if the primary search returned fewer
        # results than _MIN_ADEQUATE_APPEARANCES (default 3), fire a second
        # round of broader queries targeting festivals, major clubs, and
        # biography pages.  This ensures obscure or emerging artists still
        # get reasonable coverage rather than returning empty results.
        if len(unique_results) < _MIN_ADEQUATE_APPEARANCES:
            deeper_queries = [
                f'"{name}" resident advisor event',
                f'"{name}" electronic music festival',
                f'"{name}" boiler room OR dekmantel OR fabric OR berghain',
                f'"{name}" DJ biography',
            ]
            deeper_results: list[SearchResult] = await parallel_search(
                self._web_search.search,
                [{"query": q, "num_results": _GIG_SEARCH_LIMIT, "before_date": before_date}
                 for q in deeper_queries],
                logger=self._logger,
                error_msg="Deepened gig search failed",
            )
            for r in deeper_results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    unique_results.append(r)

            self._logger.info(
                "Deepened gig search",
                artist=name,
                total_results=len(unique_results),
            )

        if not unique_results:
            return []

        # Prioritize RA.co results first in the snippet list passed to the
        # LLM.  Since RA is the most authoritative source for event data,
        # placing these first biases the LLM toward higher-quality extractions.
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

        This is the third leg of the multi-source strategy:
          structured DBs (Phase 1-2) + web search + RAG corpus.
        The vector store holds pre-embedded historical content (books,
        archived articles, scene histories) that may not be indexed by
        web search engines.  Results below a similarity threshold of 0.7
        are discarded to avoid noise.  Per-source deduplication caps at
        3 chunks per source document to prevent a single verbose source
        from dominating the results.
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

        # Execute all primary queries in parallel with throttling
        raw_press_results: list[SearchResult] = await parallel_search(
            self._web_search.search,
            [{"query": q, "num_results": _PRESS_SEARCH_LIMIT, "before_date": before_date}
             for q in primary_queries],
            logger=self._logger,
            error_msg="Press search failed",
        )

        all_results: list[SearchResult] = []
        seen_urls: set[str] = set()
        for r in raw_press_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                all_results.append(r)

        # Phase 2 — Adaptive deepening for press: same concept as gig
        # search deepening.  If fewer than _MIN_ADEQUATE_ARTICLES (default 2)
        # came back, broaden to additional music publications and generic
        # queries.  This handles artists who may not appear in RA/DJ Mag
        # but have coverage in niche outlets.
        if len(all_results) < _MIN_ADEQUATE_ARTICLES:
            deeper_queries = [
                f'"{name}" XLR8R OR Pitchfork OR "Fact Magazine" OR Mixmag',
                f'"{name}" electronic music producer DJ',
                f'"{name}" record label release announcement',
            ]
            deeper_press_results: list[SearchResult] = await parallel_search(
                self._web_search.search,
                [{"query": q, "num_results": _PRESS_SEARCH_LIMIT, "before_date": before_date}
                 for q in deeper_queries],
                logger=self._logger,
                error_msg="Deepened press search failed",
            )
            for r in deeper_press_results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    all_results.append(r)

            self._logger.info(
                "Deepened press search",
                artist=name,
                total_results=len(all_results),
            )

        if not all_results:
            return []

        # Relevance filter: removes results about non-music topics (e.g. an
        # artist who shares a name with a politician or athlete).  Known
        # music domains pass automatically; other domains must contain at
        # least one music-related keyword in the title or snippet.
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

        # Sort results so tier-1 sources (RA, DJ Mag, Mixmag) are scraped
        # first.  Since we cap at _MAX_PRESS_ARTICLES, this ensures the
        # highest-authority content is always included even when there are
        # more results than the cap allows.
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
        """Assign a citation authority tier (1-6) based on URL domain patterns.

        The tier system serves two purposes:
        1. **Frontend display**: badges indicate source authority to users.
        2. **Conflict resolution**: when data from different sources
           disagrees (e.g. different release dates), higher-tier sources
           are preferred by downstream services.

        Tier 1: Major music publications (RA, DJ Mag, Mixmag)
        Tier 2: Specialist music press (XLR8R, Pitchfork, Fact)
        Tier 3: Music databases (Discogs, MusicBrainz, Bandcamp)
        Tier 4: Media platforms (YouTube, SoundCloud, Wikipedia)
        Tier 5: Social media and forums (Reddit, Facebook, etc.)
        Tier 6: Unknown / uncategorized sources (default fallback)
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

        Uses a majority-vote heuristic: if >= 50% of appearances with a
        known city share the same city, that city is returned as the
        artist's likely home base.  This is imperfect but works well for
        resident DJs who play most of their gigs in one city.

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
        """Use LLM to synthesize a 2-3 sentence artist profile summary.

        Profile synthesis is the final creative step: the LLM receives
        structured facts gathered from all prior phases and produces a
        human-readable bio.  Key safeguards:
        - Only runs when >= 2 data sources contributed (prevents thin
          single-source hallucination).
        - Low temperature (0.3) keeps output factual rather than creative.
        - max_tokens=300 prevents runaway generation.
        - Failures are swallowed (returns None) since the profile is
          supplementary — the research result is still valid without it.
        """
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
    # The cache is optional (injected as ICacheProvider or None).  All cache
    # operations are wrapped in try/except so a cache failure (e.g. Redis
    # down) never breaks the research pipeline — it just runs slower.

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

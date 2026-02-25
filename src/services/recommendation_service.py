"""LLM-driven recommendation engine for the raiveFlier pipeline.

Generates artist recommendations based on flier analysis data using a
three-tier approach: data-driven sources first (label-mates, shared-flier
artists, shared-lineup artists), then LLM reasoning to fill remaining
slots up to ten total.

All candidates receive an LLM-generated explanation pass that produces
a 2-3 sentence reason describing WHY each artist was recommended.

Architecture overview for junior developers
--------------------------------------------
This module answers "Who else should I listen to?"  It produces up to 10
artist recommendations using a three-tier strategy:

  TIER 1 (data-driven, highest trust):
    a) shared_flier  -- Artists who appeared on OTHER fliers alongside the
                        current flier's artists (from the flier history DB).
                        Priority 0 (highest) because co-billing is the
                        strongest signal of scene proximity.
    b) label_mate    -- Artists on the same record labels as flier artists,
                        discovered via Discogs/MusicBrainz.  Priority 1.
    c) shared_lineup -- Artists found in RAG corpus passages about events
                        involving flier artists.  Priority 2.

  TIER 2 (LLM fill):
    d) llm           -- If Tier 1 produces fewer than 10 candidates, the
                        LLM fills the remaining slots based on genre, style,
                        label, and scene context.  Priority 3 (lowest).

  EXPLANATION PASS (all tiers):
    After candidates are selected, a single LLM call generates a 2-3
    sentence "reason" for each recommendation.  If the LLM call fails,
    a template-based fallback reason is used instead.

Two public methods expose different speed/depth tradeoffs:
  - recommend()       -- Full pipeline (all tiers + LLM explanation).
                         Slower but comprehensive.
  - recommend_quick() -- Label-mates only, no LLM calls.  Returns in
                         1-3 seconds for immediate display while the
                         full pipeline runs in the background.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from src.interfaces.flier_history_provider import IFlierHistoryProvider
from src.interfaces.llm_provider import ILLMProvider
from src.interfaces.music_db_provider import IMusicDatabaseProvider

if TYPE_CHECKING:
    from src.interfaces.vector_store_provider import IVectorStoreProvider

from src.models.analysis import InterconnectionMap
from src.models.flier import ExtractedEntities
from src.models.recommendation import (
    PreloadedTier1,
    RecommendationResult,
    RecommendedArtist,
)
from src.models.research import ResearchResult
from src.utils.errors import LLMError
from src.utils.logging import get_logger
from src.utils.text_normalizer import normalize_artist_name

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)

_MAX_RECOMMENDATIONS = 10

# Cap the number of Discogs label-release queries to keep the quick
# recommendation path under ~4 seconds.  Each query is serialized by
# the Discogs 1-req/sec throttle, so N labels = N seconds minimum.
_MAX_LABEL_QUERIES = 4

# Source tier priority ordering (lower number = higher priority).
# When two sources recommend the same artist, the higher-priority source
# wins.  This ordering reflects trust: co-billing on a real flier is the
# strongest signal; LLM inference is the weakest.
_TIER_PRIORITY = {
    "shared_flier": 0,   # Appeared on another flier with these artists
    "label_mate": 1,     # Released on the same record label
    "shared_lineup": 2,  # Found in RAG corpus event/lineup passages
    "llm": 3,            # LLM-generated based on style/genre reasoning
}


class RecommendationService:
    """Generates artist recommendations based on flier analysis data.

    Uses a three-tier approach: data-driven sources first (label-mates,
    shared-flier artists, shared-lineup artists), then LLM reasoning
    to fill remaining slots.
    """

    def __init__(
        self,
        llm_provider: ILLMProvider,
        music_dbs: list[IMusicDatabaseProvider] | None = None,
        vector_store: IVectorStoreProvider | None = None,
        flier_history: IFlierHistoryProvider | None = None,
    ) -> None:
        """Initialise the service with injected dependencies.

        Parameters
        ----------
        llm_provider:
            The LLM backend used for text completion and explanation.
        music_dbs:
            Optional list of music-database providers for label-mate
            discovery (e.g. Discogs, MusicBrainz).
        vector_store:
            Optional vector store for shared-lineup RAG retrieval.
        flier_history:
            Optional flier-history provider for shared-flier lookups.
        """
        self._llm = llm_provider
        self._music_dbs = music_dbs or []
        self._vector_store = vector_store
        self._flier_history = flier_history
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def recommend(
        self,
        research_results: list[ResearchResult],
        entities: ExtractedEntities,
        interconnection_map: InterconnectionMap | None = None,
        preloaded: PreloadedTier1 | None = None,
    ) -> RecommendationResult:
        """Run the full recommendation pipeline.

        Parameters
        ----------
        research_results:
            Research profiles for every entity on the flier.
        entities:
            The raw extracted entities from the OCR / entity-extraction phase.
        interconnection_map:
            Optional interconnection analysis for additional context.
        preloaded:
            Optional pre-fetched Tier 1 discovery results from the
            background preload task.  When provided, the service skips
            the Discogs, flier-history, and RAG queries entirely,
            cutting latency from 5-15+ seconds to near-zero.

        Returns
        -------
        RecommendationResult
            Up to ten recommended artists with provenance metadata.

        Raises
        ------
        LLMError
            If all tiers fail and zero candidates are found.
        """
        self._logger.info(
            "recommendation_pipeline_start",
            research_count=len(research_results),
            artist_count=len(entities.artists),
            using_preload=preloaded is not None,
        )

        # Step 1 -- build exclusion set.  We never want to recommend an
        # artist who is ALREADY on the flier.  The exclusion set includes
        # all known names plus aliases (e.g., an artist's real name and
        # their DJ alias).
        exclusion_set = self._build_exclusion_set(research_results, entities)
        self._logger.debug(
            "exclusion_set_built",
            excluded_names=len(exclusion_set),
        )

        # Steps 2-4 -- Tier 1 data-driven discovery.
        # If preloaded results are available from the background preload
        # task, use them directly — this avoids re-fetching from Discogs,
        # flier history DB, and the RAG vector store.
        if preloaded is not None:
            self._logger.info("recommend_using_preloaded_tier1")
            label_mates = list(preloaded.label_mates)
            shared_flier = list(preloaded.shared_flier)
            shared_lineup = list(preloaded.shared_lineup)
        else:
            # No preload cache — discover from scratch in parallel.
            artist_names = [e.text for e in entities.artists]
            label_mates_task = self._discover_label_mates(
                research_results, exclusion_set,
            )
            shared_flier_task = self._discover_shared_flier_artists(
                artist_names, exclusion_set,
            )
            shared_lineup_task = self._discover_shared_lineup_artists(
                artist_names, exclusion_set,
            )

            label_mates, shared_flier, shared_lineup = await asyncio.gather(
                label_mates_task,
                shared_flier_task,
                shared_lineup_task,
            )

        self._logger.info(
            "tier1_discovery_complete",
            label_mates=len(label_mates),
            shared_flier=len(shared_flier),
            shared_lineup=len(shared_lineup),
        )

        # Step 5 -- merge and deduplicate Tier 1.  When the same artist
        # appears in multiple sources (e.g., both a label-mate AND a shared-
        # flier artist), the highest-priority source wins per _TIER_PRIORITY.
        tier1_candidates = self._merge_tier1_candidates(
            label_mates, shared_flier, shared_lineup,
        )

        if not tier1_candidates:
            raise LLMError(
                message="Recommendation pipeline produced zero candidates across all tiers",
                provider_name=self._llm.get_provider_name(),
            )

        # Steps 6+7 — Combined LLM fill + explanation pass.
        # A single LLM call handles both: (a) generating LLM-tier
        # recommendations to fill remaining slots (if < 10 Tier 1
        # candidates), and (b) writing 2-3 sentence explanations for
        # ALL candidates.  This halves LLM round-trips vs. the old
        # two-step approach, saving 2-5 seconds of latency.
        recommendations = await self._generate_fill_and_explanations(
            tier1_candidates=tier1_candidates,
            research_results=research_results,
            entities=entities,
            interconnection_map=interconnection_map,
            exclusion_set=exclusion_set,
        )

        # Step 8 — build result
        genres_analyzed = list(entities.genre_tags) if entities.genre_tags else []
        flier_artist_names = [e.text for e in entities.artists]

        result = RecommendationResult(
            recommendations=recommendations,
            flier_artists=flier_artist_names,
            genres_analyzed=genres_analyzed,
            generated_at=datetime.now(tz=timezone.utc),
        )

        self._logger.info(
            "recommendation_pipeline_complete",
            total_recommendations=len(result.recommendations),
            tier1_count=sum(
                1 for r in result.recommendations if r.source_tier != "llm"
            ),
            llm_count=sum(
                1 for r in result.recommendations if r.source_tier == "llm"
            ),
        )
        return result

    async def recommend_quick(
        self,
        research_results: list[ResearchResult],
        entities: ExtractedEntities,
        preloaded: PreloadedTier1 | None = None,
    ) -> RecommendationResult:
        """Run the fast label-mate-only recommendation pass (no LLM calls).

        This is the "quick path" -- it skips shared-flier lookup, RAG
        retrieval, and ALL LLM calls.  When preloaded data is available
        from the background preload task, this returns instantly.
        Otherwise, the only external dependency is the Discogs/MusicBrainz
        label-release query.

        The frontend calls this first for immediate display, then replaces
        results with the full `recommend()` output when it finishes.

        Parameters
        ----------
        research_results:
            Research profiles for every entity on the flier.
        entities:
            The raw extracted entities from the OCR / entity-extraction phase.
        preloaded:
            Optional pre-fetched Tier 1 results.  When provided, label-
            mate discovery is skipped entirely.

        Returns
        -------
        RecommendationResult
            Label-mate recommendations only (may be empty if no Discogs
            data is available).
        """
        self._logger.info(
            "quick_recommendation_start",
            research_count=len(research_results),
            using_preload=preloaded is not None,
        )

        if preloaded is not None:
            self._logger.info("recommend_quick_using_preloaded")
            label_mates = list(preloaded.label_mates)
        else:
            exclusion_set = self._build_exclusion_set(research_results, entities)
            label_mates = await self._discover_label_mates(
                research_results, exclusion_set,
            )
        label_mates = label_mates[:_MAX_RECOMMENDATIONS]

        recommendations: list[RecommendedArtist] = []
        for candidate in label_mates:
            reason = self._build_fallback_reason(candidate)
            connected_to = candidate.get("connected_to", [])
            if isinstance(connected_to, set):
                connected_to = sorted(connected_to)

            recommendations.append(
                RecommendedArtist(
                    artist_name=candidate["artist_name"],
                    genres=candidate.get("genres", []),
                    reason=reason,
                    source_tier=candidate.get("source_tier", "label_mate"),
                    connection_strength=float(
                        candidate.get("connection_strength", 0.5),
                    ),
                    connected_to=connected_to,
                    label_name=candidate.get("label_name"),
                    event_name=candidate.get("event_name"),
                ),
            )

        flier_artist_names = [e.text for e in entities.artists]
        genres_analyzed = list(entities.genre_tags) if entities.genre_tags else []

        result = RecommendationResult(
            recommendations=recommendations,
            flier_artists=flier_artist_names,
            genres_analyzed=genres_analyzed,
            generated_at=datetime.now(tz=timezone.utc),
        )

        self._logger.info(
            "quick_recommendation_complete",
            total=len(result.recommendations),
        )
        return result

    async def preload_tier1(
        self,
        research_results: list[ResearchResult],
        entities: ExtractedEntities,
    ) -> PreloadedTier1:
        """Preload all Tier 1 discovery results for background caching.

        Runs label-mate, shared-flier, and shared-lineup discovery in
        parallel via asyncio.gather.  Called by the background pipeline
        task in routes.py after the main analysis completes, so that
        Tier 1 data is ready before the user opens the discovery panel.

        Parameters
        ----------
        research_results:
            Research profiles from the completed pipeline.
        entities:
            Confirmed or extracted entities from the flier.

        Returns
        -------
        PreloadedTier1
            Cached discovery results for all three Tier 1 sources.
        """
        exclusion_set = self._build_exclusion_set(research_results, entities)
        artist_names = [e.text for e in entities.artists]

        label_mates, shared_flier, shared_lineup = await asyncio.gather(
            self._discover_label_mates(research_results, exclusion_set),
            self._discover_shared_flier_artists(artist_names, exclusion_set),
            self._discover_shared_lineup_artists(artist_names, exclusion_set),
        )

        self._logger.info(
            "tier1_preload_complete",
            label_mates=len(label_mates),
            shared_flier=len(shared_flier),
            shared_lineup=len(shared_lineup),
        )

        return PreloadedTier1(
            label_mates=label_mates,
            shared_flier=shared_flier,
            shared_lineup=shared_lineup,
        )

    # ------------------------------------------------------------------
    # Step 1 — exclusion set
    # ------------------------------------------------------------------

    @staticmethod
    def _build_exclusion_set(
        research_results: list[ResearchResult],
        entities: ExtractedEntities,
    ) -> set[str]:
        """Collect all artist names from the flier for exclusion.

        Normalizes all names to lowercase for case-insensitive matching.

        Parameters
        ----------
        research_results:
            Research profiles that may contain artist aliases.
        entities:
            Extracted entities from the flier.

        Returns
        -------
        set[str]
            Lowercase artist names (including aliases) to exclude.
        """
        excluded: set[str] = set()

        # Artist names from extracted entities
        for artist_entity in entities.artists:
            excluded.add(artist_entity.text.strip().lower())

        # Aliases from research results
        for result in research_results:
            if result.artist:
                excluded.add(result.artist.name.strip().lower())
                for alias in result.artist.aliases:
                    excluded.add(alias.strip().lower())

        return excluded

    # ------------------------------------------------------------------
    # Step 2 — Tier 1a: label-mate discovery (Discogs)
    # ------------------------------------------------------------------

    async def _discover_label_mates(
        self,
        research_results: list[ResearchResult],
        exclusion_set: set[str],
    ) -> list[dict[str, Any]]:
        """Find artists who share a label with flier artists via music DBs.

        For each artist with Discogs label IDs, queries the first available
        music-database provider for label releases and extracts other artist
        names.

        Parameters
        ----------
        research_results:
            Research profiles containing artist label information.
        exclusion_set:
            Lowercase names to exclude from results.

        Returns
        -------
        list[dict]
            Dicts with ``artist_name``, ``label_name``, ``connected_to``,
            ``source_tier``, ``connection_strength``, sorted by shared
            label count descending.
        """
        if not self._music_dbs:
            self._logger.debug("label_mate_discovery_skipped", reason="no music DB providers")
            return []

        # Find the first available provider
        provider: IMusicDatabaseProvider | None = None
        for db in self._music_dbs:
            if db.is_available():
                provider = db
                break
        if provider is None:
            self._logger.debug("label_mate_discovery_skipped", reason="no available provider")
            return []

        # ── Collect all (label, artist) pairs, then batch-fetch concurrently ──
        # Previously each label was fetched sequentially inside a nested loop.
        # Now we collect all label IDs upfront, fire all get_label_releases()
        # calls via asyncio.gather, then merge results.  With N artists * M
        # labels this cuts latency from O(N*M) sequential to O(1) parallel.
        label_tasks: list[tuple[str, str, str]] = []  # (label_discogs_id, label_name, artist_name)
        for result in research_results:
            if not result.artist or not result.artist.labels:
                continue
            for label in result.artist.labels:
                if label.discogs_id:
                    label_tasks.append(
                        (str(label.discogs_id), label.name, result.artist.name)
                    )

        # Cap the number of label queries to keep latency bounded.
        # Each Discogs call is serialized by the 1-req/sec throttle, so
        # N labels = N seconds.  Capping at _MAX_LABEL_QUERIES keeps the
        # quick recommendation path under ~4 seconds.
        if len(label_tasks) > _MAX_LABEL_QUERIES:
            self._logger.debug(
                "label_queries_capped",
                total_labels=len(label_tasks),
                cap=_MAX_LABEL_QUERIES,
            )
            label_tasks = label_tasks[:_MAX_LABEL_QUERIES]

        # Fetch all label releases concurrently
        async def _fetch_label(label_id: str, label_name: str) -> list[Any]:
            try:
                return await provider.get_label_releases(label_id, max_results=30)
            except Exception as exc:
                self._logger.debug(
                    "label_releases_fetch_failed",
                    label=label_name,
                    error=str(exc),
                )
                return []

        fetched = await asyncio.gather(
            *(_fetch_label(lid, lname) for lid, lname, _aname in label_tasks)
        )

        # Merge fetched releases into candidate map
        candidate_map: dict[str, dict[str, Any]] = {}
        for (_, label_name, artist_name), releases in zip(label_tasks, fetched):
            for release in releases:
                release_artist = self._extract_artist_from_release(release.title)
                if not release_artist:
                    continue

                normalized = release_artist.strip().lower()
                if normalized in exclusion_set or not normalized:
                    continue

                if normalized not in candidate_map:
                    candidate_map[normalized] = {
                        "artist_name": release_artist.strip(),
                        "label_names": set(),
                        "connected_to": set(),
                        "shared_label_count": 0,
                    }
                candidate_map[normalized]["label_names"].add(label_name)
                candidate_map[normalized]["connected_to"].add(artist_name)
                candidate_map[normalized]["shared_label_count"] = len(
                    candidate_map[normalized]["label_names"]
                )

        # Convert to list and sort.  Connection strength is a linear formula:
        #   strength = min(1.0, shared_label_count * 0.3 + 0.2)
        # One shared label yields 0.5; two yields 0.8; three caps at 1.0.
        # The 0.2 base ensures even a single shared label has meaningful weight.
        results: list[dict[str, Any]] = []
        for _key, data in candidate_map.items():
            strength = min(1.0, data["shared_label_count"] * 0.3 + 0.2)
            results.append({
                "artist_name": data["artist_name"],
                "label_name": ", ".join(sorted(data["label_names"])),
                "connected_to": sorted(data["connected_to"]),
                "source_tier": "label_mate",
                "connection_strength": strength,
            })

        results.sort(key=lambda x: x["connection_strength"], reverse=True)

        self._logger.debug(
            "label_mate_discovery_complete",
            candidates_found=len(results),
        )
        return results

    @staticmethod
    def _extract_artist_from_release(title: str) -> str | None:
        """Extract the artist name from a release title.

        Release titles from label queries often follow the format
        ``"Artist - Title"`` or ``"Artist - Title (Remix)"``; this
        method extracts the artist portion.

        Parameters
        ----------
        title:
            The release title string.

        Returns
        -------
        str or None
            The extracted artist name, or None if parsing fails.
        """
        if " - " in title:
            artist_part = title.split(" - ", 1)[0].strip()
            if artist_part:
                return artist_part
        return None

    # ------------------------------------------------------------------
    # Step 3 — Tier 1b: shared-flier artists
    # ------------------------------------------------------------------

    async def _discover_shared_flier_artists(
        self,
        artist_names: list[str],
        exclusion_set: set[str],
    ) -> list[dict[str, Any]]:
        """Find artists who appeared on other fliers alongside any queried artists.

        Uses the flier-history provider to query co-appearances.

        Parameters
        ----------
        artist_names:
            Names of artists from the current flier.
        exclusion_set:
            Lowercase names to exclude from results.

        Returns
        -------
        list[dict]
            Dicts with ``artist_name``, ``connected_to``, ``source_tier``,
            ``connection_strength``, ``times_seen``, sorted by
            ``times_seen`` descending.
        """
        if self._flier_history is None:
            self._logger.debug("shared_flier_discovery_skipped", reason="no flier history provider")
            return []

        try:
            co_artists = await self._flier_history.find_co_artists(artist_names)
        except Exception as exc:
            self._logger.debug(
                "shared_flier_discovery_failed",
                error=str(exc),
            )
            return []

        results: list[dict[str, Any]] = []
        for co in co_artists:
            name = str(co.get("artist_name", "")).strip()
            if not name or name.strip().lower() in exclusion_set:
                continue

            # Connection strength for shared-flier candidates:
            #   strength = min(1.0, times_seen * 0.2 + 0.3)
            # Seen once yields 0.5; twice yields 0.7; four times caps at 1.0.
            # The higher base (0.3 vs 0.2 for label-mates) reflects that
            # co-billing is a stronger signal than shared-label association.
            times_seen = int(co.get("times_seen", 1))
            strength = min(1.0, times_seen * 0.2 + 0.3)

            shared_with = co.get("shared_with", "")
            connected_to = [shared_with] if isinstance(shared_with, str) and shared_with else []

            results.append({
                "artist_name": name,
                "connected_to": connected_to,
                "source_tier": "shared_flier",
                "connection_strength": strength,
                "times_seen": times_seen,
                "event_name": co.get("event_names", ""),
            })

        results.sort(key=lambda x: x.get("times_seen", 0), reverse=True)

        self._logger.debug(
            "shared_flier_discovery_complete",
            candidates_found=len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Step 4 — Tier 1c: shared-lineup artists (RAG corpus)
    # ------------------------------------------------------------------

    async def _discover_shared_lineup_artists(
        self,
        artist_names: list[str],
        exclusion_set: set[str],
    ) -> list[dict[str, Any]]:
        """Find artists who appeared on lineups alongside flier artists via RAG.

        Queries the vector store for event/lineup documents mentioning each
        artist, then uses the LLM to extract other artist names from the
        retrieved chunks.

        Parameters
        ----------
        artist_names:
            Names of artists from the current flier.
        exclusion_set:
            Lowercase names to exclude from results.

        Returns
        -------
        list[dict]
            Dicts with ``artist_name``, ``connected_to``, ``source_tier``,
            ``connection_strength``.
        """
        if not self._vector_store or not self._vector_store.is_available():
            self._logger.debug(
                "shared_lineup_discovery_skipped",
                reason="no vector store available",
            )
            return []

        # Two-phase approach: (1) RAG retrieval to find event/lineup passages
        # mentioning each flier artist, then (2) LLM extraction to pull
        # other artist names from those passages.  This is necessary because
        # artist names in free-text lineup descriptions are too varied for
        # regex extraction.
        #
        # All per-artist RAG queries are dispatched concurrently via
        # asyncio.gather instead of the previous sequential loop, reducing
        # latency from O(N) sequential to O(1) parallel for N artists.
        all_chunks: list[str] = []
        chunk_artist_map: dict[int, str] = {}

        async def _query_for_artist(name: str) -> tuple[str, list]:
            query = f'"{name}" event lineup artists performing'
            try:
                chunks = await self._vector_store.query(
                    query_text=query, top_k=5,
                )
                return name, chunks
            except Exception as exc:
                self._logger.debug(
                    "lineup_rag_query_failed",
                    artist=name,
                    error=str(exc),
                )
                return name, []

        rag_results = await asyncio.gather(
            *(_query_for_artist(n) for n in artist_names)
        )

        for name, chunks in rag_results:
            for chunk in chunks:
                idx = len(all_chunks)
                all_chunks.append(chunk.chunk.text)
                chunk_artist_map[idx] = name

        if not all_chunks:
            return []

        # Phase 2: Use LLM to extract artist/DJ names from the retrieved
        # passages.  Cap at 20 chunks to stay within the LLM token budget.
        chunk_text = "\n---\n".join(all_chunks[:20])

        system_prompt = (
            "You are a music data extraction engine. You extract artist and "
            "DJ names from text about events, lineups, and music scenes."
        )
        user_prompt = (
            "The following text passages describe events, lineups, or music "
            "scenes. Extract all artist/DJ names mentioned in these passages.\n\n"
            f"{chunk_text}\n\n"
            "Return a JSON array of objects, each with:\n"
            '- "artist_name": the name of the artist/DJ\n'
            '- "context": a brief note on where/how they were mentioned\n\n'
            "Return ONLY the JSON array, no other text."
        )

        try:
            response = await self._llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=2000,
            )
        except Exception as exc:
            self._logger.debug(
                "lineup_extraction_llm_failed",
                error=str(exc),
            )
            return []

        extracted = self._parse_json_array(response)
        if not extracted:
            return []

        # Build results, filtering exclusions
        candidate_map: dict[str, dict[str, Any]] = {}
        for item in extracted:
            name = str(item.get("artist_name", "")).strip()
            normalized = name.lower()
            if not name or normalized in exclusion_set:
                continue

            if normalized not in candidate_map:
                candidate_map[normalized] = {
                    "artist_name": name,
                    "connected_to": set(),
                    "context": str(item.get("context", "")),
                }
            # Associate with all queried artists (broad connection)
            for queried_name in artist_names:
                candidate_map[normalized]["connected_to"].add(queried_name)

        results: list[dict[str, Any]] = []
        for _key, data in candidate_map.items():
            results.append({
                "artist_name": data["artist_name"],
                "connected_to": sorted(data["connected_to"]),
                "source_tier": "shared_lineup",
                "connection_strength": 0.4,
            })

        self._logger.debug(
            "shared_lineup_discovery_complete",
            candidates_found=len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Step 5 — merge and deduplicate Tier 1
    # ------------------------------------------------------------------

    def _merge_tier1_candidates(
        self,
        label_mates: list[dict[str, Any]],
        shared_flier: list[dict[str, Any]],
        shared_lineup: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge all Tier 1 sources, deduplicating by normalized artist name.

        When duplicates exist, the candidate from the highest-priority
        source is kept (shared_flier > label_mate > shared_lineup).

        Parameters
        ----------
        label_mates:
            Candidates from label-mate discovery.
        shared_flier:
            Candidates from shared-flier history.
        shared_lineup:
            Candidates from RAG lineup extraction.

        Returns
        -------
        list[dict]
            Merged and deduplicated candidates, up to ``_MAX_RECOMMENDATIONS``.
        """
        seen: dict[str, dict[str, Any]] = {}

        # Process in priority order: shared_flier first (priority 0), then
        # label_mate (priority 1), then shared_lineup (priority 2).
        # Because we process highest-priority first and skip duplicates,
        # the first occurrence wins -- giving us automatic priority-based
        # deduplication.
        for candidate_list in [shared_flier, label_mates, shared_lineup]:
            for candidate in candidate_list:
                name = candidate["artist_name"]
                normalized = normalize_artist_name(name).lower()

                if normalized not in seen:
                    seen[normalized] = candidate
                else:
                    # Keep the higher-priority source
                    existing_priority = _TIER_PRIORITY.get(
                        seen[normalized].get("source_tier", "llm"), 99,
                    )
                    new_priority = _TIER_PRIORITY.get(
                        candidate.get("source_tier", "llm"), 99,
                    )
                    if new_priority < existing_priority:
                        seen[normalized] = candidate

        # Sort by composite key: (source priority ASC, strength DESC, times
        # seen DESC).  This puts shared_flier candidates first, then label-
        # mates, then shared-lineup.  Within each tier, stronger connections
        # and more frequent co-appearances rank higher.
        merged = list(seen.values())
        merged.sort(
            key=lambda x: (
                _TIER_PRIORITY.get(x.get("source_tier", "llm"), 99),
                -x.get("connection_strength", 0),
                -x.get("times_seen", 0),
            ),
        )

        self._logger.debug(
            "tier1_merge_complete",
            total_merged=len(merged),
            capped=min(len(merged), _MAX_RECOMMENDATIONS),
        )
        return merged[:_MAX_RECOMMENDATIONS]

    # ------------------------------------------------------------------
    # Step 6 — Tier 2: LLM fill
    # ------------------------------------------------------------------

    async def _generate_llm_recommendations(
        self,
        research_results: list[ResearchResult],
        entities: ExtractedEntities,
        interconnection_map: InterconnectionMap | None,
        exclusion_set: set[str],
        existing_count: int,
    ) -> list[dict[str, Any]]:
        """Generate LLM-based recommendations to fill remaining slots.

        Compiles context from research, entities, and interconnection data
        to prompt the LLM for musically relevant recommendations.

        Parameters
        ----------
        research_results:
            Research profiles containing genre, style, label info.
        entities:
            Extracted entities including genre tags.
        interconnection_map:
            Optional analysis with patterns and narrative.
        exclusion_set:
            Lowercase names to exclude.
        existing_count:
            Number of Tier 1 candidates already found.

        Returns
        -------
        list[dict]
            LLM-generated candidate dicts.
        """
        needed = _MAX_RECOMMENDATIONS - existing_count
        if needed <= 0:
            return []

        # Compile context
        context_parts: list[str] = []

        for result in research_results:
            if not result.artist:
                continue
            artist = result.artist
            parts: list[str] = [f"Artist: {artist.name}"]
            if artist.city:
                parts.append(f"  City: {artist.city}")
            if artist.labels:
                parts.append(f"  Labels: {', '.join(lb.name for lb in artist.labels)}")
            if artist.releases:
                genres = set()
                styles = set()
                for r in artist.releases:
                    genres.update(r.genres)
                    styles.update(r.styles)
                if genres:
                    parts.append(f"  Genres: {', '.join(sorted(genres))}")
                if styles:
                    parts.append(f"  Styles: {', '.join(sorted(styles))}")
            if artist.profile_summary:
                parts.append(f"  Profile: {artist.profile_summary[:300]}")
            context_parts.append("\n".join(parts))

        if entities.genre_tags:
            context_parts.append(f"Flier genre tags: {', '.join(entities.genre_tags)}")

        if interconnection_map:
            if interconnection_map.narrative:
                context_parts.append(
                    f"Scene narrative: {interconnection_map.narrative[:500]}"
                )
            if interconnection_map.patterns:
                pattern_descs = [p.description for p in interconnection_map.patterns[:5]]
                context_parts.append(
                    "Patterns: " + "; ".join(pattern_descs)
                )

        context_text = "\n\n".join(context_parts)
        exclusion_list = ", ".join(sorted(exclusion_set))

        system_prompt = (
            "You are a music recommendation engine specializing in electronic "
            "and dance music. You recommend artists based on genre affinity, "
            "label connections, scene proximity, and stylistic similarity. "
            "You only recommend real, verifiable artists."
        )

        user_prompt = (
            "Based on the following context about artists on a rave flier, "
            f"recommend exactly {needed} additional artists that fans of "
            "these artists would enjoy.\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"DO NOT recommend any of these excluded artists: {exclusion_list}\n\n"
            f"Return exactly {needed} recommendations as a JSON array:\n"
            "[\n"
            '  {{"artist_name": "name", "genres": ["genre1", "genre2"], '
            '"reason": "brief reason for recommendation", '
            '"connected_to": ["flier_artist_name"]}}\n'
            "]\n"
            "Return ONLY the JSON array."
        )

        try:
            response = await self._llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.5,
                max_tokens=3000,
            )
        except Exception as exc:
            self._logger.error(
                "llm_recommendation_failed",
                error=str(exc),
                provider=self._llm.get_provider_name(),
            )
            return []

        parsed = self._parse_json_array(response)
        if not parsed:
            self._logger.warning(
                "llm_recommendation_parse_empty",
                response_preview=response[:200],
            )
            return []

        # Filter and build results
        results: list[dict[str, Any]] = []
        for item in parsed:
            name = str(item.get("artist_name", "")).strip()
            if not name or name.strip().lower() in exclusion_set:
                continue

            results.append({
                "artist_name": name,
                "genres": item.get("genres", []),
                "reason": str(item.get("reason", "")),
                "connected_to": item.get("connected_to", []),
                "source_tier": "llm",
                "connection_strength": 0.3,
            })

        self._logger.debug(
            "llm_recommendations_generated",
            requested=needed,
            returned=len(results),
        )
        return results[:needed]

    # ------------------------------------------------------------------
    # Steps 6+7 — Combined LLM fill + explanation (single call)
    # ------------------------------------------------------------------

    async def _generate_fill_and_explanations(
        self,
        tier1_candidates: list[dict[str, Any]],
        research_results: list[ResearchResult],
        entities: ExtractedEntities,
        interconnection_map: InterconnectionMap | None,
        exclusion_set: set[str],
    ) -> list[RecommendedArtist]:
        """Combined LLM fill + explanation in a single call.

        If Tier 1 produced fewer than ``_MAX_RECOMMENDATIONS`` candidates,
        this method asks the LLM to both suggest additional artists AND
        explain all candidates in one prompt.  If Tier 1 is already full,
        it only asks for explanations.

        This halves the LLM calls compared to the previous two-step
        approach (``_generate_llm_recommendations`` + ``_generate_explanations``),
        saving 2-5 seconds of latency.

        Falls back to template-based reasons if the LLM call fails.

        Parameters
        ----------
        tier1_candidates:
            Merged Tier 1 candidate dicts.
        research_results:
            Full research profiles for context.
        entities:
            Extracted/confirmed entities.
        interconnection_map:
            Optional interconnection analysis.
        exclusion_set:
            Names to exclude from LLM suggestions.

        Returns
        -------
        list[RecommendedArtist]
            Final recommendation models with explanations.
        """
        needed = max(0, _MAX_RECOMMENDATIONS - len(tier1_candidates))

        # Build context from research profiles
        context_parts: list[str] = []
        for result in research_results:
            if not result.artist:
                continue
            artist = result.artist
            parts: list[str] = [f"Artist: {artist.name}"]
            if artist.city:
                parts.append(f"  City: {artist.city}")
            if artist.labels:
                parts.append(
                    f"  Labels: {', '.join(lb.name for lb in artist.labels)}"
                )
            if artist.releases:
                genres: set[str] = set()
                styles: set[str] = set()
                for r in artist.releases:
                    genres.update(r.genres)
                    styles.update(r.styles)
                if genres:
                    parts.append(f"  Genres: {', '.join(sorted(genres))}")
                if styles:
                    parts.append(f"  Styles: {', '.join(sorted(styles))}")
            if artist.profile_summary:
                parts.append(f"  Profile: {artist.profile_summary[:300]}")
            context_parts.append("\n".join(parts))

        if entities.genre_tags:
            context_parts.append(
                f"Flier genre tags: {', '.join(entities.genre_tags)}"
            )
        if interconnection_map and interconnection_map.narrative:
            context_parts.append(
                f"Scene narrative: {interconnection_map.narrative[:500]}"
            )

        context_text = "\n\n".join(context_parts)

        # Build candidate info block for existing Tier 1 recommendations
        candidate_info = "\n".join(
            f"- {c['artist_name']} (source: {c.get('source_tier', 'unknown')}, "
            f"connected to: {', '.join(c.get('connected_to', []))})"
            for c in tier1_candidates
        )

        # Build the combined prompt
        system_prompt = (
            "You are a music recommendation engine and journalist specializing "
            "in electronic and dance music. You recommend real, verifiable artists "
            "and write concise, knowledgeable explanations."
        )

        fill_section = ""
        if needed > 0:
            exclusion_list = ", ".join(sorted(exclusion_set))
            fill_section = (
                f"\n\nTASK 1: Recommend exactly {needed} additional artists "
                f"that fans of these flier artists would enjoy. For each, "
                f"include genres and connected_to (which flier artist they "
                f"relate to). Only recommend real artists. "
                f"DO NOT recommend: {exclusion_list}\n"
            )

        task_label = "TASK 2" if needed > 0 else "TASK 1"
        also_new = " AND any new ones you suggest" if needed > 0 else ""

        user_prompt = (
            f"Research context about artists on a rave flier:\n\n"
            f"{context_text}\n\n"
            f"Existing data-driven recommendations:\n{candidate_info}\n"
            f"{fill_section}\n"
            f"{task_label}: For ALL artists listed above{also_new}, "
            f"write a 2-3 sentence explanation of WHY each artist would "
            f"appeal to fans of the flier artists.\n\n"
            f"Return a JSON object with two keys:\n"
            f'{{"new_recommendations": '
            f'[{{"artist_name": "...", "genres": [...], '
            f'"connected_to": ["flier_artist"]}}], '
            f'"explanations": '
            f'[{{"artist_name": "...", "reason": "2-3 sentences"}}]}}\n\n'
            f"If no new recommendations are needed, return an empty array "
            f"for new_recommendations. Return ONLY the JSON object."
        )

        try:
            response = await self._llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.4,
                max_tokens=5000,
            )

            # Parse the combined JSON response
            parsed = self._parse_json_object(response)

            # Extract new recommendations (LLM fill)
            llm_candidates: list[dict[str, Any]] = []
            if needed > 0:
                for item in parsed.get("new_recommendations", []):
                    name = str(item.get("artist_name", "")).strip()
                    if not name or name.strip().lower() in exclusion_set:
                        continue
                    llm_candidates.append({
                        "artist_name": name,
                        "genres": item.get("genres", []),
                        "connected_to": item.get("connected_to", []),
                        "source_tier": "llm",
                        "connection_strength": 0.3,
                    })
                llm_candidates = llm_candidates[:needed]

            # Build explanation map
            explanations_map: dict[str, str] = {}
            for item in parsed.get("explanations", []):
                exp_name = str(item.get("artist_name", "")).strip()
                exp_reason = str(item.get("reason", "")).strip()
                if exp_name and exp_reason:
                    explanations_map[exp_name.lower()] = exp_reason

            # Merge Tier 1 + LLM fill
            all_candidates = list(tier1_candidates) + llm_candidates
            all_candidates = all_candidates[:_MAX_RECOMMENDATIONS]

            # Build final models with explanations
            recommendations: list[RecommendedArtist] = []
            for candidate in all_candidates:
                name = candidate["artist_name"]
                reason = explanations_map.get(name.lower(), "")
                if not reason:
                    reason = self._build_fallback_reason(candidate)

                connected_to = candidate.get("connected_to", [])
                if isinstance(connected_to, set):
                    connected_to = sorted(connected_to)

                recommendations.append(
                    RecommendedArtist(
                        artist_name=name,
                        genres=candidate.get("genres", []),
                        reason=reason,
                        source_tier=candidate.get("source_tier", "llm"),
                        connection_strength=float(
                            candidate.get("connection_strength", 0.5),
                        ),
                        connected_to=connected_to,
                        label_name=candidate.get("label_name"),
                        event_name=candidate.get("event_name"),
                    ),
                )

            self._logger.info(
                "combined_fill_and_explain_complete",
                llm_fill=len(llm_candidates),
                explanations_matched=sum(
                    1 for r in recommendations
                    if r.artist_name.lower() in explanations_map
                ),
            )
            return recommendations

        except Exception as exc:
            self._logger.warning(
                "combined_fill_and_explain_failed",
                error=str(exc),
                fallback="using template reasons",
            )
            # Fallback: no LLM fill, template reasons for all Tier 1
            all_candidates = tier1_candidates[:_MAX_RECOMMENDATIONS]
            return [
                RecommendedArtist(
                    artist_name=c["artist_name"],
                    genres=c.get("genres", []),
                    reason=self._build_fallback_reason(c),
                    source_tier=c.get("source_tier", "llm"),
                    connection_strength=float(
                        c.get("connection_strength", 0.5),
                    ),
                    connected_to=(
                        sorted(c["connected_to"])
                        if isinstance(c.get("connected_to"), set)
                        else c.get("connected_to", [])
                    ),
                    label_name=c.get("label_name"),
                    event_name=c.get("event_name"),
                )
                for c in all_candidates
            ]

    # ------------------------------------------------------------------
    # DEPRECATED: Step 6 — LLM fill (now handled by _generate_fill_and_explanations)
    # Kept as fallback safety net for one release cycle.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # DEPRECATED: Step 7 — LLM explanation pass (now handled by _generate_fill_and_explanations)
    # Kept as fallback safety net for one release cycle.
    # ------------------------------------------------------------------

    async def _generate_explanations(
        self,
        candidates: list[dict[str, Any]],
        research_context: str,
    ) -> list[RecommendedArtist]:
        """Generate LLM explanations for all candidates.

        Calls the LLM once with the full context and candidate list,
        asking for a 2-3 sentence reason for each recommendation.
        Falls back to source-specific reasons if the LLM call fails.

        Parameters
        ----------
        candidates:
            The merged candidate dicts from all tiers.
        research_context:
            Compiled research summary for LLM context.

        Returns
        -------
        list[RecommendedArtist]
            Final recommendation models with explanations.
        """
        candidate_names = [c["artist_name"] for c in candidates]

        system_prompt = (
            "You are a music journalist explaining artist recommendations "
            "to a fan exploring a rave scene. Write concise, knowledgeable "
            "explanations grounded in facts."
        )

        candidate_info = "\n".join(
            f"- {c['artist_name']} (source: {c.get('source_tier', 'unknown')}, "
            f"connected to: {', '.join(c.get('connected_to', []))})"
            for c in candidates
        )

        user_prompt = (
            "Given the following research context about a rave flier:\n\n"
            f"{research_context}\n\n"
            "Explain WHY each of these recommended artists would appeal to "
            "fans of the flier artists. Write 2-3 sentences per artist.\n\n"
            f"Artists to explain:\n{candidate_info}\n\n"
            "Return a JSON array where each object has:\n"
            '- "artist_name": the name exactly as given above\n'
            '- "reason": 2-3 sentence explanation\n\n'
            "Return ONLY the JSON array."
        )

        explanations_map: dict[str, str] = {}

        try:
            response = await self._llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.4,
                max_tokens=4000,
            )
            parsed = self._parse_json_array(response)
            for item in parsed:
                name = str(item.get("artist_name", "")).strip()
                reason = str(item.get("reason", "")).strip()
                if name and reason:
                    explanations_map[name.lower()] = reason
        except Exception as exc:
            self._logger.warning(
                "explanation_pass_failed",
                error=str(exc),
                fallback="using source-specific reasons",
            )

        # Build final RecommendedArtist models
        recommendations: list[RecommendedArtist] = []
        for candidate in candidates:
            name = candidate["artist_name"]
            source_tier = candidate.get("source_tier", "llm")

            # Use LLM explanation if available, otherwise fallback
            reason = explanations_map.get(name.lower(), "")
            if not reason:
                reason = self._build_fallback_reason(candidate)

            connected_to = candidate.get("connected_to", [])
            if isinstance(connected_to, set):
                connected_to = sorted(connected_to)

            recommendations.append(
                RecommendedArtist(
                    artist_name=name,
                    genres=candidate.get("genres", []),
                    reason=reason,
                    source_tier=source_tier,
                    connection_strength=float(
                        candidate.get("connection_strength", 0.5)
                    ),
                    connected_to=connected_to,
                    label_name=candidate.get("label_name"),
                    event_name=candidate.get("event_name"),
                )
            )

        return recommendations

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_fallback_reason(candidate: dict[str, Any]) -> str:
        """Build a fallback explanation when the LLM explanation pass fails.

        Uses template strings that incorporate the candidate's metadata
        (label name, times seen, connected artist names) to produce a
        meaningful reason without any LLM call.  This is the safety net
        for both recommend_quick() (which never calls the LLM) and the
        full pipeline when the explanation LLM call fails.

        Produces a source-specific reason string based on the candidate's
        tier and connection metadata.

        Parameters
        ----------
        candidate:
            Candidate dict with source_tier and connection metadata.

        Returns
        -------
        str
            A human-readable fallback reason.
        """
        source_tier = candidate.get("source_tier", "llm")
        connected_to = candidate.get("connected_to", [])
        if isinstance(connected_to, set):
            connected_to = sorted(connected_to)
        connected_str = ", ".join(connected_to) if connected_to else "flier artists"

        if source_tier == "label_mate":
            label = candidate.get("label_name", "a shared label")
            return f"Released on {label} alongside {connected_str}."
        if source_tier == "shared_flier":
            times = candidate.get("times_seen", 1)
            return (
                f"Appeared on {times} other flier(s) alongside {connected_str}."
            )
        if source_tier == "shared_lineup":
            return f"Appeared on event lineups alongside {connected_str}."
        # LLM tier — use whatever reason was provided
        return candidate.get("reason", "Recommended based on stylistic similarity.")

    def _compile_research_summary(
        self, research_results: list[ResearchResult],
    ) -> str:
        """Compile a concise research summary for the explanation pass.

        Produces a shorter context than the full interconnection context
        to stay within token budgets for the explanation LLM call.

        Parameters
        ----------
        research_results:
            Research profiles for entities.

        Returns
        -------
        str
            Condensed research summary text.
        """
        parts: list[str] = []
        for result in research_results:
            if not result.artist:
                continue
            artist = result.artist
            summary_parts = [f"{artist.name}"]
            if artist.city:
                summary_parts.append(f"({artist.city})")
            if artist.labels:
                label_names = [lb.name for lb in artist.labels[:5]]
                summary_parts.append(f"labels: {', '.join(label_names)}")
            if artist.releases:
                genres = set()
                styles = set()
                for r in artist.releases[:10]:
                    genres.update(r.genres)
                    styles.update(r.styles)
                if genres:
                    summary_parts.append(f"genres: {', '.join(sorted(genres)[:5])}")
                if styles:
                    summary_parts.append(f"styles: {', '.join(sorted(styles)[:5])}")
            if artist.profile_summary:
                summary_parts.append(f"— {artist.profile_summary[:200]}")
            parts.append(" | ".join(summary_parts))

        return "\n".join(parts)

    def _parse_json_array(self, response: str) -> list[dict[str, Any]]:
        """Extract and parse a JSON array from an LLM response.

        Handles markdown code fences and bare JSON arrays.

        Parameters
        ----------
        response:
            Raw LLM response text.

        Returns
        -------
        list[dict]
            Parsed JSON array, or an empty list if parsing fails.
        """
        text = response.strip()

        # Try markdown fence extraction
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
        except json.JSONDecodeError as exc:
            self._logger.warning(
                "json_array_parse_failed",
                error=str(exc),
                response_preview=response[:200],
            )
            return []

        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            # Single object wrapped — return as one-element list
            return [parsed]

        return []

    def _parse_json_object(self, response: str) -> dict[str, Any]:
        """Extract and parse a JSON object from an LLM response.

        Similar to ``_parse_json_array`` but expects a top-level dict.
        Handles markdown code fences and bare JSON.

        Parameters
        ----------
        response:
            Raw LLM response text.

        Returns
        -------
        dict
            Parsed JSON object, or an empty dict if parsing fails.
        """
        text = response.strip()

        # Try markdown fence extraction
        fence_match = _JSON_FENCE_RE.search(text)
        if fence_match:
            text = fence_match.group(1).strip()

        # Fallback: find the first { ... } block
        if not text.startswith("{"):
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start != -1 and brace_end > brace_start:
                text = text[brace_start : brace_end + 1]

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            self._logger.warning(
                "json_object_parse_not_dict",
                type=type(parsed).__name__,
            )
            return {}
        except json.JSONDecodeError as exc:
            self._logger.warning(
                "json_object_parse_failed",
                error=str(exc),
                response_preview=text[:200],
            )
            return {}

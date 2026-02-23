"""LLM-driven recommendation engine for the raiveFlier pipeline.

Generates artist recommendations based on flier analysis data using a
three-tier approach: data-driven sources first (label-mates, shared-flier
artists, shared-lineup artists), then LLM reasoning to fill remaining
slots up to ten total.

All candidates receive an LLM-generated explanation pass that produces
a 2-3 sentence reason describing WHY each artist was recommended.
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
    RecommendationResult,
    RecommendedArtist,
)
from src.models.research import ResearchResult
from src.utils.errors import LLMError
from src.utils.logging import get_logger
from src.utils.text_normalizer import normalize_artist_name

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)

_MAX_RECOMMENDATIONS = 10

# Source tier priority ordering (lower number = higher priority).
_TIER_PRIORITY = {
    "shared_flier": 0,
    "label_mate": 1,
    "shared_lineup": 2,
    "llm": 3,
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
        )

        # Step 1 — build exclusion set
        exclusion_set = self._build_exclusion_set(research_results, entities)
        self._logger.debug(
            "exclusion_set_built",
            excluded_names=len(exclusion_set),
        )

        # Collect artist names for shared-flier and shared-lineup queries
        artist_names = [e.text for e in entities.artists]

        # Step 2-4 — execute Tier 1 sources in parallel
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

        # Step 5 — merge and deduplicate Tier 1
        tier1_candidates = self._merge_tier1_candidates(
            label_mates, shared_flier, shared_lineup,
        )

        # Step 6 — LLM fill if needed
        if len(tier1_candidates) < _MAX_RECOMMENDATIONS:
            llm_candidates = await self._generate_llm_recommendations(
                research_results,
                entities,
                interconnection_map,
                exclusion_set,
                len(tier1_candidates),
            )
            tier1_candidates.extend(llm_candidates)

        # Cap at max
        tier1_candidates = tier1_candidates[:_MAX_RECOMMENDATIONS]

        if not tier1_candidates:
            raise LLMError(
                message="Recommendation pipeline produced zero candidates across all tiers",
                provider_name=self._llm.get_provider_name(),
            )

        # Step 7 — LLM explanation pass
        research_context = self._compile_research_summary(research_results)
        recommendations = await self._generate_explanations(
            tier1_candidates, research_context,
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

        # Gather: artist_name_normalized -> {label_names, connected_to_set, count}
        candidate_map: dict[str, dict[str, Any]] = {}

        for result in research_results:
            if not result.artist:
                continue
            artist = result.artist
            if not artist.labels:
                continue

            for label in artist.labels:
                if not label.discogs_id:
                    continue

                try:
                    releases = await provider.get_label_releases(
                        str(label.discogs_id), max_results=30,
                    )
                except Exception as exc:
                    self._logger.debug(
                        "label_releases_fetch_failed",
                        label=label.name,
                        error=str(exc),
                    )
                    continue

                for release in releases:
                    # Release titles often follow "Artist - Title" format
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
                    candidate_map[normalized]["label_names"].add(label.name)
                    candidate_map[normalized]["connected_to"].add(artist.name)
                    candidate_map[normalized]["shared_label_count"] = len(
                        candidate_map[normalized]["label_names"]
                    )

        # Convert to list and sort
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

        all_chunks: list[str] = []
        chunk_artist_map: dict[int, str] = {}

        for name in artist_names:
            query = f'"{name}" event lineup artists performing'
            try:
                chunks = await self._vector_store.query(
                    query_text=query, top_k=5,
                )
            except Exception as exc:
                self._logger.debug(
                    "lineup_rag_query_failed",
                    artist=name,
                    error=str(exc),
                )
                continue

            for chunk in chunks:
                idx = len(all_chunks)
                all_chunks.append(chunk.chunk.text)
                chunk_artist_map[idx] = name

        if not all_chunks:
            return []

        # Use LLM to extract artist/DJ names from chunks
        chunk_text = "\n---\n".join(all_chunks[:20])  # Budget cap

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

        # Process in priority order: shared_flier first, then label_mate, then shared_lineup
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

        # Sort by: source priority, then connection_strength
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
    # Step 7 — LLM explanation pass
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

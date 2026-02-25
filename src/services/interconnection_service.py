"""LLM-driven interconnection analysis engine for the raiveFlier pipeline.

Accepts the complete set of research results and extracted entities from a
single flier, compiles them into a structured context block, sends that
context to an LLM with a synthesis prompt, and parses / validates the
response into an :class:`InterconnectionMap`.

Every relationship claim returned by the LLM is verified against the
research data via citation validation — claims without a traceable source
are discarded.

Architecture overview for junior developers
--------------------------------------------
This module is the "relationship discovery" brain of the pipeline. It takes
all the raw research gathered about every entity on a rave flier (artists,
venues, promoters) and asks an LLM to find connections between them.

The five-step pipeline is:
  1. COMPILE  -- Assemble all research into a numbered-source text block.
  1b. RA EVENTS -- Query ChromaDB for RA event listings where 2+ flier
                   artists co-appeared.  Compile as citable [RA-n] context.
  2. PROMPT   -- Send a 13-point synthesis prompt to the LLM asking it to
                 identify specific relationship types (shared labels,
                 lineups, geographic patterns, genre alignment, etc.).
  3. VALIDATE -- Every claim the LLM returns must cite a real source from
                 the research data (including RA event sources).  Claims
                 without valid citations are discarded.
  4. ENRICH   -- Confidence scores are penalised for uncertain language
                 (the LLM flags these with [UNCERTAIN]) and for geographic
                 mismatches (e.g., linking artists from different cities).
                 RA-backed shared_lineup edges get a confidence floor of 0.6.

The output is an InterconnectionMap: a graph of EntityNodes (vertices) and
RelationshipEdges (edges) plus higher-level PatternInsights and a narrative.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from src.interfaces.llm_provider import ILLMProvider

if TYPE_CHECKING:
    from src.interfaces.vector_store_provider import IVectorStoreProvider
from src.models.analysis import (
    Citation,
    EntityNode,
    InterconnectionMap,
    PatternInsight,
    RelationshipEdge,
)
from src.models.entities import EntityType
from src.models.flier import ExtractedEntities
from src.models.research import ResearchResult
from src.services.citation_service import CitationService
from src.utils.errors import LLMError
from src.utils.logging import get_logger

# Regex to extract JSON from an LLM response wrapped in ```json ... ``` fences.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)

# The LLM is instructed to flag weak claims with this tag in its output.
# See the synthesis prompt's "STRICT RULES" section.
_UNCERTAIN_TAG = "[UNCERTAIN]"

# How much to subtract from a relationship's confidence when [UNCERTAIN] is
# present.  A 0.3 penalty on a default 0.5 confidence produces 0.2 -- near
# the 0.15 discard threshold, making uncertain claims barely survive.
_UNCERTAIN_CONFIDENCE_PENALTY = 0.3


class InterconnectionService:
    """Traces all links between entities extracted from a rave flier.

    The service follows a four-step pipeline:

    1. **Build context** — compile research results into a structured text
       summary suitable for LLM consumption.
    2. **LLM analysis** — send the context with a detailed synthesis prompt
       requesting relationships, patterns, and a narrative.
    3. **Parse + validate** — parse the JSON response, verify every citation
       against the original research data, and discard unsupported claims.
    4. **Enrich** — lower confidence on relationships flagged as uncertain
       by the LLM.

    All external dependencies are injected through the constructor,
    following the adapter pattern (CLAUDE.md Section 6).
    """

    def __init__(
        self,
        llm_provider: ILLMProvider,
        citation_service: CitationService,
        vector_store: IVectorStoreProvider | None = None,
    ) -> None:
        """Initialise the service with injected dependencies.

        Parameters
        ----------
        llm_provider:
            The LLM backend used for text completion.
        citation_service:
            Service used to assign citation tiers to sources.
        vector_store:
            Optional vector store for cross-entity RAG context retrieval.
        """
        # All dependencies are injected (adapter pattern).  The LLM and
        # vector store are behind interfaces so the concrete provider (OpenAI,
        # Anthropic, Pinecone, Chroma, etc.) can be swapped without touching
        # this service.
        self._llm = llm_provider
        self._citation_service = citation_service
        self._vector_store = vector_store
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(
        self,
        research_results: list[ResearchResult],
        entities: ExtractedEntities,
    ) -> InterconnectionMap:
        """Run the full interconnection analysis pipeline.

        Parameters
        ----------
        research_results:
            Research profiles for every entity on the flier.
        entities:
            The raw extracted entities from the OCR / entity-extraction phase.

        Returns
        -------
        InterconnectionMap
            Graph of nodes, edges, patterns, narrative, and citations.

        Raises
        ------
        LLMError
            If the LLM call fails or returns unparseable output after retry.
        """
        self._logger.info(
            "interconnection_analysis_start",
            research_count=len(research_results),
            artist_count=len(entities.artists),
        )

        # Step 1 — build context: flatten all research into a single text
        # block with numbered source references ([1], [2], ...) so the LLM
        # can produce inline citations.
        compiled_context = self._compile_research_context(research_results)

        # Step 1.5a — RAG cross-entity context augmentation: query the vector
        # store for passages that mention MULTIPLE entities from the flier.
        # This surfaces connections the per-entity research may have missed.
        rag_context = await self._retrieve_cross_entity_context(entities)
        if rag_context:
            compiled_context += "\n\n=== CORPUS CONTEXT (from indexed books/articles) ===\n"
            compiled_context += rag_context

        # Step 1.5b — RA event shared lineup discovery: query the RA event
        # corpus for events where 2+ flier artists appeared on the same bill.
        # This provides concrete, citable evidence for the "SHARED LINEUPS"
        # dimension (#2) of the synthesis prompt.  Without this data, the LLM
        # has no event evidence and shared lineup claims get discarded by
        # citation validation.
        shared_ra_events = await self._discover_shared_ra_events(entities)
        if shared_ra_events:
            ra_context = self._compile_shared_event_context(shared_ra_events)
            compiled_context += "\n\n" + ra_context

        # Step 2 — LLM analysis.  The system prompt establishes a strict
        # factual persona.  The user prompt (built below) contains the
        # 13-point synthesis framework and all compiled research data.
        # Temperature is kept very low (0.1) to minimise creative invention.
        system_prompt = (
            "You are a factual music research analyst.  You report only "
            "verified facts from the data provided, with inline source "
            "citations.  You never invent, embellish, or speculate.  "
            "Silence is preferable to unsourced claims."
        )
        user_prompt = self._build_synthesis_prompt(compiled_context)

        try:
            response = await self._llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=8000,
            )
        except Exception as exc:
            self._logger.error(
                "llm_analysis_failed",
                error=str(exc),
                provider=self._llm.get_provider_name(),
            )
            raise LLMError(
                message=f"Interconnection LLM call failed: {exc}",
                provider_name=self._llm.get_provider_name(),
            ) from exc

        # Step 3 — parse + validate.  This is the trust gate: every
        # relationship the LLM claims must cite a source that exists in
        # the original research data.  Unsupported claims are discarded.
        parsed = self._parse_analysis_response(response)

        raw_relationships = parsed.get("relationships", [])
        # Citation validation: cross-reference each claim's source_citation
        # field against known source strings from the research results.
        # shared_ra_events is passed so RA event sources (e.g., "Resident
        # Advisor", "RA Events: London (...)") are registered in the known
        # sources set and RA-backed citations pass validation.
        validated_relationships = self._validate_citations(
            raw_relationships, research_results, shared_ra_events=shared_ra_events
        )

        edges = self._build_edges(validated_relationships)
        patterns = self._build_patterns(parsed.get("patterns", []))
        nodes = self._build_nodes(research_results, entities)
        narrative: str | None = parsed.get("narrative")

        # Step 4 — enrich: confidence penalty pipeline.
        # Two independent penalty passes run in sequence:
        #   a) Uncertain language: -0.3 for edges containing [UNCERTAIN]
        #   b) Geographic mismatch: -0.4 if both entities have different
        #      cities, -0.2 if one entity's city differs from the venue city.
        # Penalties stack if both apply.
        edges = self._penalise_uncertain(edges)

        # Step 4.5 — RA-backed confidence boost: shared_lineup edges backed
        # by RA event data are structural evidence (concrete co-appearances,
        # not inferred), so they get a confidence floor of 0.6 to survive
        # downstream penalties.  This runs after uncertain penalty but before
        # geographic mismatch penalty.
        edges = self._boost_ra_backed_edges(edges, shared_ra_events)

        edges = self._penalise_geographic_mismatch(edges, research_results)

        # Final threshold: edges below 0.15 confidence are noise and are
        # dropped.  This catches cases where multiple penalties compound
        # (e.g., uncertain + geographic mismatch = near-zero confidence).
        edges = [e for e in edges if e.confidence >= 0.15]

        all_citations = self._collect_all_citations(edges, patterns)

        imap = InterconnectionMap(
            nodes=nodes,
            edges=edges,
            patterns=patterns,
            narrative=narrative,
            citations=all_citations,
        )

        self._logger.info(
            "interconnection_analysis_complete",
            edges=len(imap.edges),
            patterns=len(imap.patterns),
            has_narrative=imap.narrative is not None,
        )
        return imap

    # ------------------------------------------------------------------
    # Step 1 — context compilation
    # ------------------------------------------------------------------

    @staticmethod
    def _compile_research_context(results: list[ResearchResult]) -> str:
        """Compile all research results into a structured, source-indexed summary.

        Every citable fact is tagged with a numbered source reference ``[n]``
        so the LLM can produce inline citations in its output.

        Parameters
        ----------
        results:
            Research profiles produced by the research phase.

        Returns
        -------
        str
            Multi-section text block with a source index appended.
        """
        sections: list[str] = []
        # Global source index: maps source_key -> (index, display_label, url).
        # Every citable fact gets tagged with [n] so the LLM can produce
        # inline citations.  The index is appended at the end of the
        # compiled context as "=== SOURCE INDEX ===" for reference.
        source_index: dict[str, tuple[int, str, str | None]] = {}

        def _ref(label: str, url: str | None = None) -> str:
            """Return ``[n]`` for *label*, registering the source if new."""
            key = (label.lower().strip(), (url or "").lower().strip())
            if key not in source_index:
                idx = len(source_index) + 1
                source_index[key] = (idx, label, url)
            return f"[{source_index[key][0]}]"

        for result in results:
            header = f"=== {result.entity_type.value}: {result.entity_name} ==="
            parts: list[str] = [header]

            if result.artist:
                artist = result.artist
                if artist.city:
                    geo_str = artist.city
                    if artist.region:
                        geo_str += f", {artist.region}"
                    if artist.country:
                        geo_str += f", {artist.country}"
                    parts.append(f"Based in: {geo_str}")
                if artist.profile_summary:
                    parts.append(f"Profile: {artist.profile_summary}")
                if artist.aliases:
                    parts.append(f"Aliases: {', '.join(artist.aliases)}")
                if artist.labels:
                    label_names = [lb.name for lb in artist.labels]
                    parts.append(f"Labels: {', '.join(label_names)}")
                if artist.releases:
                    rel_lines: list[str] = []
                    for r in artist.releases[:15]:
                        src = r.label or "release"
                        ref = _ref(src, r.discogs_url or r.bandcamp_url or r.beatport_url)
                        line = f"  - {r.title}"
                        if r.label:
                            line += f" ({r.label})"
                        if r.year:
                            line += f" [{r.year}]"
                        if r.format:
                            line += f" format:{r.format}"
                        if r.catalog_number:
                            line += f" cat:{r.catalog_number}"
                        if r.genres:
                            line += f" genres:{','.join(r.genres)}"
                        if r.styles:
                            line += f" styles:{','.join(r.styles)}"
                        line += f" {ref}"
                        rel_lines.append(line)
                    parts.append("Releases:\n" + "\n".join(rel_lines))

                    # Release format summary for pattern analysis
                    formats = [r.format for r in artist.releases if r.format]
                    if formats:
                        format_counts: dict[str, int] = {}
                        for fmt in formats:
                            fmt_key = fmt.strip().lower()
                            format_counts[fmt_key] = format_counts.get(fmt_key, 0) + 1
                        fmt_summary = ", ".join(
                            f"{fmt}: {count}" for fmt, count in
                            sorted(format_counts.items(), key=lambda x: -x[1])
                        )
                        parts.append(f"Release format breakdown: {fmt_summary}")

                    # Release type hints from catalog structure
                    genres_all = [g for r in artist.releases for g in r.genres]
                    styles_all = [s for r in artist.releases for s in r.styles]
                    if genres_all:
                        genre_counts: dict[str, int] = {}
                        for g in genres_all:
                            genre_counts[g] = genre_counts.get(g, 0) + 1
                        top_genres = sorted(genre_counts.items(), key=lambda x: -x[1])[:5]
                        parts.append(
                            "Top genres across releases: "
                            + ", ".join(f"{g} ({c})" for g, c in top_genres)
                        )
                    if styles_all:
                        style_counts: dict[str, int] = {}
                        for s in styles_all:
                            style_counts[s] = style_counts.get(s, 0) + 1
                        top_styles = sorted(style_counts.items(), key=lambda x: -x[1])[:5]
                        parts.append(
                            "Top styles across releases: "
                            + ", ".join(f"{s} ({c})" for s, c in top_styles)
                        )

            if result.venue:
                venue = result.venue
                if venue.location:
                    parts.append(f"Location: {venue.location}")
                if venue.city:
                    city_str = venue.city
                    if venue.country:
                        city_str += f", {venue.country}"
                    parts.append(f"City: {city_str}")
                if venue.history:
                    parts.append(f"History: {venue.history}")
                if venue.cultural_significance:
                    parts.append(f"Cultural significance: {venue.cultural_significance}")
                if venue.notable_events:
                    parts.append("Notable events: " + ", ".join(venue.notable_events[:10]))
                if venue.articles:
                    art_lines = []
                    for a in venue.articles[:10]:
                        ref = _ref(f"{a.title} ({a.source})", a.url)
                        line = f"  - {a.title} ({a.source}) {ref}"
                        art_lines.append(line)
                    parts.append("Articles:\n" + "\n".join(art_lines))

            if result.promoter:
                promoter = result.promoter
                if promoter.city:
                    geo_str = promoter.city
                    if promoter.region:
                        geo_str += f", {promoter.region}"
                    if promoter.country:
                        geo_str += f", {promoter.country}"
                    parts.append(f"Based in: {geo_str}")
                if promoter.event_history:
                    parts.append("Event history: " + ", ".join(promoter.event_history[:10]))
                if promoter.affiliated_artists:
                    parts.append(
                        "Affiliated artists: " + ", ".join(promoter.affiliated_artists[:10])
                    )
                if promoter.affiliated_venues:
                    parts.append("Affiliated venues: " + ", ".join(promoter.affiliated_venues[:10]))
                if promoter.articles:
                    art_lines = []
                    for a in promoter.articles[:10]:
                        ref = _ref(f"{a.title} ({a.source})", a.url)
                        line = f"  - {a.title} ({a.source}) {ref}"
                        art_lines.append(line)
                    parts.append("Articles:\n" + "\n".join(art_lines))

            if result.date_context:
                dc = result.date_context
                parts.append(f"Event date: {dc.event_date.isoformat()}")
                if dc.scene_context:
                    parts.append(f"Scene context: {dc.scene_context}")
                if dc.city_context:
                    parts.append(f"City context: {dc.city_context}")
                if dc.cultural_context:
                    parts.append(f"Cultural context: {dc.cultural_context}")
                if dc.nearby_events:
                    parts.append("Nearby events: " + ", ".join(dc.nearby_events[:10]))
                if dc.sources:
                    for s in dc.sources[:10]:
                        _ref(f"{s.title} ({s.source})", s.url)

            if result.sources_consulted:
                parts.append("Sources consulted: " + ", ".join(result.sources_consulted))

            sections.append("\n".join(parts))

        # Append the numbered source index
        if source_index:
            idx_lines = ["\n=== SOURCE INDEX ==="]
            for _key, (idx, label, url) in sorted(source_index.items(), key=lambda x: x[1][0]):
                entry = f"[{idx}] {label}"
                if url:
                    entry += f"  —  {url}"
                idx_lines.append(entry)
            sections.append("\n".join(idx_lines))

        return "\n\n".join(sections)

    async def _retrieve_cross_entity_context(self, entities: ExtractedEntities) -> str:
        """Retrieve cross-entity passages from the RAG corpus.

        Builds a query from all entity names and retrieves passages
        that mention multiple entities (connection context).  Returns
        formatted text within a token budget, or an empty string if
        RAG is not available.
        """
        if not self._vector_store or not self._vector_store.is_available():
            return ""

        # Build a cross-entity query from all entity names
        entity_names: list[str] = []
        for artist_entity in entities.artists:
            entity_names.append(artist_entity.text)
        if entities.venue:
            entity_names.append(entities.venue.text)
        if entities.promoter:
            entity_names.append(entities.promoter.text)

        if not entity_names:
            return ""

        # The query concatenates ALL entity names plus relationship keywords.
        # This biases the vector search toward passages that mention multiple
        # entities from the same flier -- exactly the kind of cross-entity
        # context the interconnection analysis needs.
        cross_query = " ".join(entity_names) + " connection relationship scene"

        try:
            chunks = await self._vector_store.query(query_text=cross_query, top_k=30)
        except Exception as exc:
            self._logger.debug("Cross-entity corpus retrieval failed", error=str(exc))
            return ""

        if not chunks:
            return ""

        # Token budget guard: the compiled context + RAG context both go into
        # the LLM prompt, so we cap RAG at ~30K tokens (~120K chars) to leave
        # room for the synthesis prompt and the LLM's response.
        max_chars = 30000 * 4  # ~30K tokens
        parts: list[str] = []
        current_chars = 0

        for chunk in chunks:
            if chunk.similarity_score < 0.6:
                continue

            entry = (
                f"[{chunk.chunk.source_title}"
                f"{', ' + chunk.chunk.author if chunk.chunk.author else ''}"
                f" — Tier {chunk.chunk.citation_tier}]\n"
                f"{chunk.chunk.text}\n"
            )

            if current_chars + len(entry) > max_chars:
                break
            parts.append(entry)
            current_chars += len(entry)

        if parts:
            self._logger.info(
                "cross_entity_corpus_retrieval",
                chunks_used=len(parts),
                approx_tokens=current_chars // 4,
            )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Step 1.5b — RA event shared lineup discovery
    # ------------------------------------------------------------------
    #
    # The methods below query the ChromaDB vector store specifically for
    # RA event listing chunks where flier artists appear, then intersect
    # the results to find events where 2+ flier artists shared a lineup.
    # This produces concrete, citable evidence for the LLM's "SHARED
    # LINEUPS" synthesis dimension (point #2 in the 13-point prompt).
    #
    # Data flow:
    #   _discover_shared_ra_events(entities)
    #       → per-artist filtered ChromaDB queries (concurrent)
    #       → _parse_ra_event_chunk_text() for each chunk
    #       → pairwise intersection of artist event sets
    #       → list of shared event dicts
    #   _compile_shared_event_context(shared_events)
    #       → formatted text with [RA-n] citations for LLM context
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_ra_event_chunk_text(
        chunk_text: str,
    ) -> list[dict[str, str | list[str]]]:
        """Parse structured RA event text from a DocumentChunk into individual events.

        RA event chunks are generated by ``RAEventProcessor._build_chunk`` and
        contain ~8 events separated by ``---``.  Each event uses labeled fields
        on separate lines (``Event:``, ``Date:``, ``Venue:``, ``Artists:``,
        ``City:``, ``URL:``).  This parser extracts those fields without any
        LLM call -- the text is machine-generated and structurally reliable.

        Parameters
        ----------
        chunk_text:
            Raw text from a DocumentChunk with ``source_type="event_listing"``.

        Returns
        -------
        list[dict]
            Parsed events with keys: ``title``, ``date``, ``venue``,
            ``artists`` (list[str]), ``city``, ``url``.  Missing optional
            fields default to empty string or empty list.
        """
        if not chunk_text or not chunk_text.strip():
            return []

        # Split on the separator used by RAEventProcessor._build_chunk (line 143).
        blocks = re.split(r"\n\n---\n\n", chunk_text.strip())

        parsed_events: list[dict[str, str | list[str]]] = []
        for block in blocks:
            block = block.strip()
            if not block:
                continue

            event: dict[str, str | list[str]] = {
                "title": "",
                "date": "",
                "venue": "",
                "artists": [],
                "city": "",
                "url": "",
            }

            for line in block.split("\n"):
                line = line.strip()
                if line.startswith("Event:"):
                    event["title"] = line[len("Event:"):].strip()
                elif line.startswith("Date:"):
                    event["date"] = line[len("Date:"):].strip()
                elif line.startswith("Venue:"):
                    event["venue"] = line[len("Venue:"):].strip()
                elif line.startswith("Artists:"):
                    raw_artists = line[len("Artists:"):].strip()
                    event["artists"] = [
                        a.strip() for a in raw_artists.split(",") if a.strip()
                    ]
                elif line.startswith("City:"):
                    event["city"] = line[len("City:"):].strip()
                elif line.startswith("URL:"):
                    event["url"] = line[len("URL:"):].strip()

            # Only include events that have at least a title or artists
            if event["title"] or event["artists"]:
                parsed_events.append(event)

        return parsed_events

    async def _discover_shared_ra_events(
        self,
        entities: ExtractedEntities,
    ) -> list[dict[str, Any]]:
        """Query the RA event corpus for events where 2+ flier artists co-appeared.

        For each artist on the flier, issues a filtered ChromaDB query
        restricted to ``source_type="event_listing"`` chunks whose
        ``entity_tags`` metadata contains the artist's name.  All queries
        run concurrently via ``asyncio.gather``.  Results are parsed and
        intersected pairwise to find events shared by multiple flier artists.

        This method is the key enabler for the "SHARED LINEUPS" dimension
        (#2) of the 13-point synthesis prompt.  Without it, the LLM has
        no concrete event evidence and shared lineup claims are discarded
        by citation validation.

        Parameters
        ----------
        entities:
            Extracted entities from the flier (artists, venue, promoter).

        Returns
        -------
        list[dict]
            Shared event dicts, each containing: ``artist_pair`` (tuple of
            two artist names), ``event_title``, ``event_date``, ``venue``,
            ``city``, ``source_title`` (ChromaDB source_title for citation),
            ``ra_url``, ``full_lineup`` (all artists on that event).
            Capped at 50 results to stay within token budget.
        """
        import asyncio

        # Guard: need a working vector store and at least 2 artists to
        # find pairwise co-appearances.
        if not self._vector_store or not self._vector_store.is_available():
            return []

        artist_names = [a.text for a in entities.artists]
        if len(artist_names) < 2:
            return []

        # Concurrent per-artist queries.  Each query uses the entity_tags
        # metadata filter to restrict results to RA event chunks that
        # explicitly mention the artist (pre-extracted by RAEventProcessor
        # during ingestion — no LLM needed for matching).
        async def _query_for_artist(name: str) -> list[Any]:
            try:
                return await self._vector_store.query(
                    query_text=f'"{name}" event lineup',
                    top_k=15,
                    filters={
                        "entity_tags": {"$contains": name},
                        "source_type": {"$in": ["event_listing"]},
                    },
                    max_per_source=5,
                )
            except Exception as exc:
                self._logger.debug(
                    "ra_event_query_failed", artist=name, error=str(exc)
                )
                return []

        all_results = await asyncio.gather(
            *[_query_for_artist(name) for name in artist_names]
        )

        # Build per-artist event index.  Key = artist name (lower), value =
        # list of parsed events with their chunk source_title for citation.
        artist_events: dict[str, list[dict[str, Any]]] = {}
        for artist_name, chunks in zip(artist_names, all_results):
            events_for_artist: list[dict[str, Any]] = []
            for chunk in chunks:
                parsed = self._parse_ra_event_chunk_text(chunk.chunk.text)
                for event in parsed:
                    # Only include events where this artist actually appears
                    event_artist_names_lower = [
                        a.lower() for a in event.get("artists", [])
                    ]
                    if artist_name.lower() in event_artist_names_lower:
                        event["_source_title"] = chunk.chunk.source_title
                        events_for_artist.append(event)
            artist_events[artist_name] = events_for_artist

        # Pairwise intersection: find events shared by any two flier artists.
        # Deduplicate by (title, date) to avoid counting the same event twice
        # when it appears in overlapping chunks.
        shared: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str, str, str]] = set()  # (a1, a2, title, date)

        for i, artist_a in enumerate(artist_names):
            for artist_b in artist_names[i + 1:]:
                events_a = artist_events.get(artist_a, [])
                events_b = artist_events.get(artist_b, [])

                # Build lookup from events_b keyed on (title_lower, date)
                b_lookup: dict[tuple[str, str], dict[str, Any]] = {}
                for ev in events_b:
                    key = (str(ev.get("title", "")).lower(), str(ev.get("date", "")))
                    if key not in b_lookup:
                        b_lookup[key] = ev

                for ev_a in events_a:
                    key = (str(ev_a.get("title", "")).lower(), str(ev_a.get("date", "")))
                    if key in b_lookup:
                        dedup_key = (artist_a, artist_b, key[0], key[1])
                        if dedup_key in seen_keys:
                            continue
                        seen_keys.add(dedup_key)

                        shared.append({
                            "artist_pair": (artist_a, artist_b),
                            "event_title": str(ev_a.get("title", "")),
                            "event_date": str(ev_a.get("date", "")),
                            "venue": str(ev_a.get("venue", "")),
                            "city": str(ev_a.get("city", "")),
                            "source_title": str(ev_a.get("_source_title", "")),
                            "ra_url": str(ev_a.get("url", "")),
                            "full_lineup": list(ev_a.get("artists", [])),
                        })

        # Cap at 50 to stay within the ~5K token budget for RA context.
        shared = shared[:50]

        if shared:
            self._logger.info(
                "shared_ra_events_discovered",
                count=len(shared),
                pairs=len(seen_keys),
            )

        return shared

    @staticmethod
    def _compile_shared_event_context(
        shared_events: list[dict[str, Any]],
    ) -> str:
        """Format discovered shared RA events as citable context text.

        Uses ``[RA-n]`` reference prefixes to avoid collision with the
        ``[n]`` numbering from ``_compile_research_context``.  Produces
        a self-contained section with an ``RA EVENT SOURCE INDEX`` footer
        that maps each ``[RA-n]`` to a Resident Advisor URL/source.

        This context is appended to the compiled research text before the
        LLM synthesis prompt, giving the LLM concrete evidence to produce
        ``shared_lineup`` relationships with verifiable citations.

        Parameters
        ----------
        shared_events:
            Output from ``_discover_shared_ra_events``.

        Returns
        -------
        str
            Formatted text block with ``[RA-n]`` citations, or empty string
            if no shared events.
        """
        if not shared_events:
            return ""

        # Group shared events by artist pair for readable output.
        from collections import defaultdict
        pair_events: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for event in shared_events:
            pair_key = tuple(event["artist_pair"])
            pair_events[pair_key].append(event)

        lines: list[str] = [
            "=== SHARED EVENT APPEARANCES (from Resident Advisor event listings) ===",
            "",
            "The following shared event appearances were found in the RA event corpus.",
            "Each [RA-n] reference links to a verified Resident Advisor event listing.",
            "",
        ]

        source_index: list[str] = []
        ref_counter = 0

        for pair, events in pair_events.items():
            lines.append(f"{pair[0]} and {pair[1]} appeared together at:")
            for event in events:
                ref_counter += 1
                ref_tag = f"[RA-{ref_counter}]"

                lineup_str = ", ".join(event.get("full_lineup", []))
                venue_str = event.get("venue", "Unknown venue")
                city_str = event.get("city", "")
                date_str = event.get("event_date", "")
                title_str = event.get("event_title", "Untitled event")

                location = f"{venue_str} ({city_str})" if city_str else venue_str

                lines.append(
                    f'  - "{title_str}" at {location}, {date_str}. '
                    f"Lineup: {lineup_str}. {ref_tag}"
                )

                # Build source index entry
                ra_url = event.get("ra_url", "")
                source_title = event.get("source_title", "Resident Advisor")
                url_part = f" — {ra_url}" if ra_url else ""
                source_index.append(f"{ref_tag} {source_title}{url_part}")

            lines.append("")

        lines.append("=== RA EVENT SOURCE INDEX ===")
        lines.extend(source_index)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Step 2 — prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_synthesis_prompt(compiled_context: str) -> str:
        """Build the LLM user prompt for interconnection synthesis.

        The prompt requests a strictly fact-based chronicle with inline
        numbered citations — no generated story or speculation.

        THE 13-POINT SYNTHESIS FRAMEWORK
        ---------------------------------
        The prompt asks the LLM to analyze 13 specific relationship
        dimensions.  Each point targets a different axis of connection:

         1. Shared Labels          -- Record labels where multiple flier
                                      artists released music
         2. Shared Lineups         -- Previous events with overlapping artists
         3. Promoter-Artist Links  -- How the promoter connects to each act
         4. Venue-Scene Connections -- The venue's role in the scene
         5. Geographic Patterns    -- Same city/region alignments
         6. Temporal Patterns      -- Career timeline positioning
         7. Scene Context          -- Movement or subgenre classification
         8. Release Format         -- Vinyl-first vs digital-first strategies
         9. Performance Style      -- DJ vs live act classification
        10. Touring & Venue Overlap -- Shared booking circuits
        11. Label Ecosystem Depth  -- Parent labels, sister imprints, A&R
        12. Career Stage           -- Emerging vs veteran dynamic
        13. Genre & Style Alignment -- Niche cohesion vs deliberate spread

        This breadth ensures the LLM does not fixate on one dimension
        (e.g., only shared labels) and instead maps the full relationship
        space.  The output is structured JSON with relationships, patterns,
        and a narrative.

        Parameters
        ----------
        compiled_context:
            The compiled research context text with numbered source index.

        Returns
        -------
        str
            Full synthesis prompt with the context embedded.
        """
        return (
            "You have been given detailed, source-indexed research on all "
            "entities connected to a single rave/electronic music event "
            "flier.  Every fact in the research data has a numbered source "
            "reference like [1], [2], etc.  A SOURCE INDEX at the end maps "
            "each number to its origin.\n"
            "\n"
            "RESEARCH DATA:\n"
            f"{compiled_context}\n"
            "\n"
            "YOUR TASK:\n"
            "Produce a factual chronicle that traces the path leading up to "
            "this event.  The chronicle must be built ENTIRELY from the "
            "facts provided — do NOT generate, invent, or embellish any "
            "content.  Every factual statement MUST include its source "
            "reference number inline, e.g. 'Artist X released Y on Label Z "
            "[3].'  If a fact has no source number, omit it.\n"
            "\n"
            "ANALYSIS REQUIREMENTS:\n"
            "1. SHARED LABELS: Record labels where multiple artists on "
            "this flier have released music, with source references. Go "
            "deeper — are these labels part of the same family or "
            "distribution network? Does one artist run or co-run a label "
            "that others release on?\n"
            "2. SHARED LINEUPS: Previous events where two or more of "
            "these artists appeared together, with dates and sources.\n"
            "3. PROMOTER-ARTIST LINKS: How the promoter connects to "
            "each artist — past bookings, shared scenes, geographic ties "
            "— citing specific events or articles.\n"
            "4. VENUE-SCENE CONNECTIONS: The venue's role in the broader "
            "scene — what events it is known for, cited from research. "
            "Do any of these artists have recurring residencies or "
            "repeated bookings at this venue or the promoter's other "
            "venues?\n"
            "5. GEOGRAPHIC PATTERNS: Whether artists are from the same "
            "city/region and how that relates to the event.\n"
            "6. TEMPORAL PATTERNS: Where this event falls in each "
            "artist's career timeline.\n"
            "7. SCENE CONTEXT: What movement or subgenre this event "
            "represents, grounded in cited facts.\n"
            "8. RELEASE FORMAT PATTERNS: Based on release data, do these "
            "artists primarily release on vinyl, digital, or both? Do "
            "they share a vinyl-first or digital-first release strategy? "
            "Are releases primarily singles/EPs or full albums? Cite "
            "specific releases and format data.\n"
            "9. PERFORMANCE STYLE: Based on available evidence (articles, "
            "event listings, profiles), are these artists primarily DJs, "
            "live performers, or hybrid live/DJ acts? Do they share a "
            "similar performance approach? Cite sources.\n"
            "10. TOURING & VENUE OVERLAP: Are any of these artists "
            "playing the same club or venue chain across different "
            "cities on their own solo tours? Do they share festival "
            "circuits or the same booking agency ecosystem? Look for "
            "patterns in where these artists regularly perform.\n"
            "11. LABEL ECOSYSTEM DEPTH: Beyond simple shared labels, "
            "are there deeper label connections? Same parent label, "
            "sister imprints, shared A&R, label founders who also "
            "appear on the flier? Do multiple artists have long-term "
            "relationships with the same label vs. one-off releases?\n"
            "12. CAREER STAGE & TRAJECTORY: Are these artists at similar "
            "career stages — emerging, mid-career, established, veteran? "
            "Is there a headliner/support dynamic visible from the data? "
            "How does each artist's output volume and recency compare?\n"
            "13. GENRE & STYLE ALIGNMENT: Based on release genre/style "
            "tags, do these artists operate in the same subgenre niche "
            "or do they represent a deliberate genre spread? Note any "
            "stylistic evolution visible in their catalogs.\n"
            "\n"
            "STRICT RULES:\n"
            "- ONLY state facts that appear in the research data above.\n"
            "- EVERY factual claim MUST include at least one [n] source "
            "reference from the SOURCE INDEX.\n"
            "- Do NOT add connective language that implies causation, "
            "motivation, or intent unless a source explicitly states it.\n"
            "- Do NOT speculate or fill gaps.  If information is missing, "
            "say nothing about it — silence is better than invention.\n"
            "- Flag genuinely uncertain connections with [UNCERTAIN].\n"
            "- Prioritize first-hand sources (tier 1-2) over secondary.\n"
            "\n"
            "Return your analysis as JSON:\n"
            "{\n"
            '  "relationships": [\n'
            '    {"source": "entity1", "target": "entity2", '
            '"type": "relationship_type", "details": "factual explanation '
            'with [n] inline citations", '
            '"source_citation": "the [n] reference(s) used", '
            '"confidence": 0.0-1.0}\n'
            "  ],\n"
            '  "patterns": [\n'
            '    {"type": "pattern_type", "description": "factual pattern '
            'with [n] citations", "entities": ["entity1", "entity2"], '
            '"source_citation": "[n] reference(s)"}\n'
            "  ],\n"
            '  "narrative": "A multi-paragraph factual chronicle of the '
            "path leading to this event, built from cited facts only.  "
            "Each sentence that states a fact includes its [n] source "
            "reference inline.  No embellishment, no invented connective "
            "tissue — just facts arranged in meaningful order.  End with "
            "a brief factual summary of how these entities converge at "
            'this event."\n'
            "}\n"
            "\n"
            "RELATIONSHIP TYPE VALUES — use these in the 'type' field:\n"
            "  shared_label, shared_lineup, promoter_booking, "
            "venue_residency, geographic_link, temporal_link, "
            "release_format_pattern, performance_style_link, "
            "touring_overlap, label_ecosystem, career_stage_alignment, "
            "genre_style_alignment\n"
            "\n"
            "PATTERN TYPE VALUES — use these in the patterns 'type' field:\n"
            "  label_network, geographic_cluster, release_strategy, "
            "format_preference, performance_style, touring_circuit, "
            "career_stage_mix, genre_cohesion, genre_spread, "
            "vinyl_culture, digital_native, label_loyalty"
        )

    # ------------------------------------------------------------------
    # Step 3 — parsing and validation
    # ------------------------------------------------------------------

    def _parse_analysis_response(self, response: str) -> dict[str, Any]:
        """Extract and validate JSON from the LLM response.

        Handles markdown code fences and bare JSON objects.

        Parameters
        ----------
        response:
            Raw LLM response text.

        Returns
        -------
        dict
            Parsed JSON object containing relationships, patterns,
            and narrative.

        Raises
        ------
        LLMError
            If no valid JSON can be extracted.
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
        except json.JSONDecodeError as exc:
            self._logger.error(
                "json_parse_failed",
                error=str(exc),
                response_preview=response[:200],
            )
            raise LLMError(
                message=f"Failed to parse interconnection JSON: {exc}",
                provider_name=self._llm.get_provider_name(),
            ) from exc

        if not isinstance(parsed, dict):
            raise LLMError(
                message="LLM response is not a JSON object",
                provider_name=self._llm.get_provider_name(),
            )

        return parsed

    def _validate_citations(
        self,
        relationships: list[dict[str, Any]],
        research_data: list[ResearchResult],
        shared_ra_events: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Validate that each relationship cites an actual research source.

        A citation is considered valid if the ``source_citation`` text
        appears as a substring in any of the following research data fields:
        source names, article titles, article URLs, event names, or
        sources-consulted entries.  RA event sources are also registered
        when ``shared_ra_events`` is provided, so the LLM's RA-backed
        citations (using ``[RA-n]`` references) pass validation.

        Parameters
        ----------
        relationships:
            Raw relationship dicts from the LLM response.
        research_data:
            The original research results to validate against.
        shared_ra_events:
            Optional list of shared RA event dicts from
            ``_discover_shared_ra_events``.  When provided, RA event
            source titles, event titles, and URLs are added to the
            known sources set so RA-backed citations pass validation.

        Returns
        -------
        list[dict]
            Only relationships whose citations map to real research data.
        """
        # Build a set of known source strings for fast lookup.
        # This is the "ground truth" corpus: every label name, release title,
        # article title, article URL, and sources-consulted entry across all
        # research results.  If a citation from the LLM cannot be matched to
        # any string in this set (using bidirectional substring matching),
        # the relationship is discarded as unverifiable.
        known_sources: set[str] = set()
        for result in research_data:
            for src in result.sources_consulted:
                known_sources.add(src.lower())

            if result.artist:
                for label in result.artist.labels:
                    known_sources.add(label.name.lower())
                for release in result.artist.releases:
                    known_sources.add(release.title.lower())
                    known_sources.add(release.label.lower())

            if result.venue:
                for article in result.venue.articles:
                    known_sources.add(article.title.lower())
                    known_sources.add(article.source.lower())
                    if article.url:
                        known_sources.add(article.url.lower())

            if result.promoter:
                for article in result.promoter.articles:
                    known_sources.add(article.title.lower())
                    known_sources.add(article.source.lower())
                    if article.url:
                        known_sources.add(article.url.lower())

            if result.date_context:
                for source in result.date_context.sources:
                    known_sources.add(source.title.lower())
                    known_sources.add(source.source.lower())
                    if source.url:
                        known_sources.add(source.url.lower())

        # Register RA event sources so the LLM's [RA-n] citations pass
        # the bidirectional substring match.  Generic "resident advisor"
        # and "ra events" strings are added once; per-event source titles,
        # event titles, and URLs provide fine-grained matching.
        if shared_ra_events:
            known_sources.add("resident advisor")
            known_sources.add("ra.co")
            known_sources.add("ra events")
            for event in shared_ra_events:
                if event.get("source_title"):
                    known_sources.add(str(event["source_title"]).lower())
                if event.get("event_title"):
                    known_sources.add(str(event["event_title"]).lower())
                if event.get("ra_url"):
                    known_sources.add(str(event["ra_url"]).lower())

        validated: list[dict[str, Any]] = []
        discarded_count = 0

        for rel in relationships:
            citation_text = str(rel.get("source_citation", "")).lower().strip()
            if not citation_text:
                discarded_count += 1
                continue

            # Bidirectional substring match: the LLM's citation text may be
            # a superset ("Discogs: Artist X") or subset ("Artist X") of
            # the known source string.  Either direction counts as a match.
            is_valid = any(
                citation_text in ks or ks in citation_text
                for ks in known_sources
                if ks  # skip empty strings
            )

            if is_valid:
                validated.append(rel)
            else:
                discarded_count += 1
                self._logger.debug(
                    "citation_discarded",
                    source=rel.get("source"),
                    target=rel.get("target"),
                    citation=citation_text[:100],
                )

        if discarded_count:
            self._logger.info(
                "citations_validated",
                total=len(relationships),
                valid=len(validated),
                discarded=discarded_count,
            )

        return validated

    # ------------------------------------------------------------------
    # Model builders
    # ------------------------------------------------------------------

    def _build_edges(self, relationships: list[dict[str, Any]]) -> list[RelationshipEdge]:
        """Convert validated relationship dicts into RelationshipEdge models.

        Parameters
        ----------
        relationships:
            Validated relationship dicts from the LLM response.

        Returns
        -------
        list[RelationshipEdge]
        """
        edges: list[RelationshipEdge] = []
        for rel in relationships:
            source_name = str(rel.get("source", "")).strip()
            target_name = str(rel.get("target", "")).strip()
            if not source_name or not target_name:
                continue

            citation_text = str(rel.get("source_citation", ""))

            # Detect RA event citations by checking for "ra" indicators
            # in the citation text.  RA-backed citations get source_type
            # "event_listing" (tier 3) instead of generic "research".
            citation_lower = citation_text.lower()
            is_ra_citation = (
                "ra-" in citation_lower
                or "ra events" in citation_lower
                or "resident advisor" in citation_lower
                or "ra.co" in citation_lower
            )

            citation = self._citation_service.build_citation(
                text=citation_text,
                source_name="Resident Advisor Events" if is_ra_citation else citation_text,
                source_type="event_listing" if is_ra_citation else "research",
            )

            confidence = float(rel.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            details = str(rel.get("details", ""))

            edges.append(
                RelationshipEdge(
                    source=source_name,
                    target=target_name,
                    relationship_type=str(rel.get("type", "related")),
                    details=details if details else None,
                    citations=[citation],
                    confidence=confidence,
                )
            )

        return edges

    def _build_patterns(self, raw_patterns: list[dict[str, Any]]) -> list[PatternInsight]:
        """Convert raw pattern dicts into PatternInsight models.

        Parameters
        ----------
        raw_patterns:
            Pattern dicts from the LLM response.

        Returns
        -------
        list[PatternInsight]
        """
        patterns: list[PatternInsight] = []
        for p in raw_patterns:
            pattern_type = str(p.get("type", "general")).strip()
            description = str(p.get("description", "")).strip()
            if not description:
                continue

            involved = [str(e) for e in p.get("entities", []) if str(e).strip()]

            citation_text = str(p.get("source_citation", ""))
            citations: list[Citation] = []
            if citation_text.strip():
                citations.append(
                    self._citation_service.build_citation(
                        text=citation_text,
                        source_name=citation_text,
                        source_type="research",
                    )
                )

            patterns.append(
                PatternInsight(
                    pattern_type=pattern_type,
                    description=description,
                    involved_entities=involved,
                    citations=citations,
                )
            )

        return patterns

    @staticmethod
    def _build_nodes(
        research_results: list[ResearchResult],
        entities: ExtractedEntities,
    ) -> list[EntityNode]:
        """Build graph nodes from research results and extracted entities.

        Parameters
        ----------
        research_results:
            Research profiles for entities.
        entities:
            Raw extracted entities from the flier.

        Returns
        -------
        list[EntityNode]
        """
        # Build graph nodes using a two-pass approach:
        #   Pass 1: Create nodes from research results (rich metadata).
        #   Pass 2: Backfill any extracted entities that the research phase
        #           did not cover (minimal metadata, just name + confidence).
        # The seen_names set prevents duplicates across both passes.
        nodes: list[EntityNode] = []
        seen_names: set[str] = set()

        for result in research_results:
            name = result.entity_name
            if name.lower() in seen_names:
                continue
            seen_names.add(name.lower())

            properties: dict[str, Any] = {
                "confidence": result.confidence,
            }

            if result.artist:
                if result.artist.aliases:
                    properties["aliases"] = result.artist.aliases
                if result.artist.labels:
                    properties["labels"] = [lb.name for lb in result.artist.labels]
                if result.artist.city:
                    properties["city"] = result.artist.city
            if result.venue and result.venue.city:
                properties["city"] = result.venue.city
            if result.promoter and result.promoter.city:
                properties["city"] = result.promoter.city

            nodes.append(
                EntityNode(
                    entity_type=result.entity_type,
                    name=name,
                    properties=properties,
                )
            )

        # Add any extracted entities not covered by research
        for artist_entity in entities.artists:
            if artist_entity.text.lower() not in seen_names:
                seen_names.add(artist_entity.text.lower())
                nodes.append(
                    EntityNode(
                        entity_type=EntityType.ARTIST,
                        name=artist_entity.text,
                        properties={"confidence": artist_entity.confidence},
                    )
                )

        if entities.venue and entities.venue.text.lower() not in seen_names:
            seen_names.add(entities.venue.text.lower())
            nodes.append(
                EntityNode(
                    entity_type=EntityType.VENUE,
                    name=entities.venue.text,
                    properties={"confidence": entities.venue.confidence},
                )
            )

        if entities.promoter and entities.promoter.text.lower() not in seen_names:
            nodes.append(
                EntityNode(
                    entity_type=EntityType.PROMOTER,
                    name=entities.promoter.text,
                    properties={"confidence": entities.promoter.confidence},
                )
            )

        return nodes

    # ------------------------------------------------------------------
    # Step 4 — enrichment
    # ------------------------------------------------------------------

    @staticmethod
    def _penalise_uncertain(edges: list[RelationshipEdge]) -> list[RelationshipEdge]:
        """Lower confidence on edges whose details contain [UNCERTAIN].

        Parameters
        ----------
        edges:
            Relationship edges to check.

        Returns
        -------
        list[RelationshipEdge]
            Edges with confidence adjusted where necessary.
        """
        enriched: list[RelationshipEdge] = []
        for edge in edges:
            details_text = edge.details or ""
            if _UNCERTAIN_TAG in details_text:
                new_confidence = max(0.0, edge.confidence - _UNCERTAIN_CONFIDENCE_PENALTY)
                edge = edge.model_copy(update={"confidence": new_confidence})
            enriched.append(edge)
        return enriched

    @staticmethod
    def _penalise_geographic_mismatch(
        edges: list[RelationshipEdge],
        research_results: list[ResearchResult],
    ) -> list[RelationshipEdge]:
        """Penalise edges linking entities in geographically incompatible locations.

        If both source and target have known cities that do not match,
        reduce confidence by 0.4.  If only one has a city and it does
        not match the venue city, reduce by 0.2.

        Parameters
        ----------
        edges:
            Relationship edges to check.
        research_results:
            Research results containing geographic data for entities.

        Returns
        -------
        list[RelationshipEdge]
            Edges with confidence adjusted for geographic mismatches.
        """
        # Build a city lookup from research results.  The venue's city is
        # stored separately as a reference point for single-entity mismatches.
        city_map: dict[str, str] = {}
        venue_city: str | None = None

        for result in research_results:
            name_lower = result.entity_name.lower()
            if result.artist and result.artist.city:
                city_map[name_lower] = result.artist.city.lower()
            if result.venue and result.venue.city:
                city_map[name_lower] = result.venue.city.lower()
                venue_city = result.venue.city.lower()
            if result.promoter and result.promoter.city:
                city_map[name_lower] = result.promoter.city.lower()

        enriched: list[RelationshipEdge] = []
        for edge in edges:
            source_city = city_map.get(edge.source.lower())
            target_city = city_map.get(edge.target.lower())

            # Penalty logic:
            #   - Both entities have known, different cities: -0.4
            #     (strong signal of a false geographic link)
            #   - Only one entity has a known city, and it differs from the
            #     venue city: -0.2 (weaker signal, benefit of the doubt)
            #   - Cities match OR no city data: no penalty
            penalty = 0.0
            if source_city and target_city:
                if source_city != target_city:
                    penalty = 0.4
            elif source_city or target_city:
                known_city = source_city or target_city
                if venue_city and known_city != venue_city:
                    penalty = 0.2

            if penalty > 0:
                new_confidence = max(0.0, edge.confidence - penalty)
                edge = edge.model_copy(update={"confidence": new_confidence})

            enriched.append(edge)

        return enriched

    @staticmethod
    def _boost_ra_backed_edges(
        edges: list[RelationshipEdge],
        shared_ra_events: list[dict[str, Any]] | None,
    ) -> list[RelationshipEdge]:
        """Ensure RA-backed shared_lineup edges have a minimum confidence of 0.6.

        RA event co-appearances are structural evidence — two artists literally
        appeared on the same lineup — not LLM inference.  This is stronger than
        discography-based connections, so edges backed by RA event citations
        should not fall below the database evidence threshold even if the LLM
        assigned conservative confidence or the uncertain penalty was applied.

        Only applies to edges with ``relationship_type == "shared_lineup"``
        whose citation text references an RA source.

        Parameters
        ----------
        edges:
            Relationship edges (post-uncertain penalty, pre-geographic penalty).
        shared_ra_events:
            Shared RA event dicts from ``_discover_shared_ra_events``.
            If None or empty, no boost is applied.

        Returns
        -------
        list[RelationshipEdge]
            Edges with RA-backed shared_lineup confidence boosted to minimum 0.6.
        """
        if not shared_ra_events:
            return edges

        _RA_MIN_CONFIDENCE = 0.6

        boosted: list[RelationshipEdge] = []
        for edge in edges:
            if edge.relationship_type == "shared_lineup" and edge.confidence < _RA_MIN_CONFIDENCE:
                # Check if any citation references RA
                is_ra_backed = any(
                    "ra" in (c.source_name or "").lower()
                    or "resident advisor" in (c.source_name or "").lower()
                    or "event_listing" in (c.source_type or "")
                    for c in edge.citations
                )
                if is_ra_backed:
                    edge = edge.model_copy(update={"confidence": _RA_MIN_CONFIDENCE})
            boosted.append(edge)

        return boosted

    @staticmethod
    def _collect_all_citations(
        edges: list[RelationshipEdge],
        patterns: list[PatternInsight],
    ) -> list[Citation]:
        """Deduplicate and collect all citations from edges and patterns.

        Parameters
        ----------
        edges:
            Relationship edges with citations.
        patterns:
            Pattern insights with citations.

        Returns
        -------
        list[Citation]
            Deduplicated list of all citations.
        """
        seen: set[str] = set()
        citations: list[Citation] = []

        for edge in edges:
            for c in edge.citations:
                key = f"{c.source_name}|{c.text}"
                if key not in seen:
                    seen.add(key)
                    citations.append(c)

        for pattern in patterns:
            for c in pattern.citations:
                key = f"{c.source_name}|{c.text}"
                if key not in seen:
                    seen.add(key)
                    citations.append(c)

        return citations

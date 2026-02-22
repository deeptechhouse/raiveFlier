"""LLM-driven interconnection analysis engine for the raiveFlier pipeline.

Accepts the complete set of research results and extracted entities from a
single flier, compiles them into a structured context block, sends that
context to an LLM with a synthesis prompt, and parses / validates the
response into an :class:`InterconnectionMap`.

Every relationship claim returned by the LLM is verified against the
research data via citation validation — claims without a traceable source
are discarded.
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

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
_UNCERTAIN_TAG = "[UNCERTAIN]"
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

        # Step 1 — build context
        compiled_context = self._compile_research_context(research_results)

        # Step 1.5 — RAG cross-entity context augmentation
        rag_context = await self._retrieve_cross_entity_context(entities)
        if rag_context:
            compiled_context += "\n\n=== CORPUS CONTEXT (from indexed books/articles) ===\n"
            compiled_context += rag_context

        # Step 2 — LLM analysis
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
                max_tokens=5000,
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

        # Step 3 — parse + validate
        parsed = self._parse_analysis_response(response)

        raw_relationships = parsed.get("relationships", [])
        validated_relationships = self._validate_citations(raw_relationships, research_results)

        edges = self._build_edges(validated_relationships)
        patterns = self._build_patterns(parsed.get("patterns", []))
        nodes = self._build_nodes(research_results, entities)
        narrative: str | None = parsed.get("narrative")

        # Step 4 — enrich: penalise uncertain relationships
        edges = self._penalise_uncertain(edges)

        # Step 4.5 — penalise geographic mismatches
        edges = self._penalise_geographic_mismatch(edges, research_results)

        # Step 4.6 — discard edges with very low confidence after all penalties
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
        # Global source index: maps source_key → (index, display_label, url)
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
                        line += f" {ref}"
                        rel_lines.append(line)
                    parts.append("Releases:\n" + "\n".join(rel_lines))

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

        cross_query = " ".join(entity_names) + " connection relationship scene"

        try:
            chunks = await self._vector_store.query(query_text=cross_query, top_k=30)
        except Exception as exc:
            self._logger.debug("Cross-entity corpus retrieval failed", error=str(exc))
            return ""

        if not chunks:
            return ""

        # Budget: approximate token count (chars / 4), stay within limit
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
    # Step 2 — prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_synthesis_prompt(compiled_context: str) -> str:
        """Build the LLM user prompt for interconnection synthesis.

        The prompt requests a strictly fact-based chronicle with inline
        numbered citations — no generated story or speculation.

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
            "this flier have released music, with source references.\n"
            "2. SHARED LINEUPS: Previous events where two or more of "
            "these artists appeared together, with dates and sources.\n"
            "3. PROMOTER-ARTIST LINKS: How the promoter connects to "
            "each artist — past bookings, shared scenes, geographic ties "
            "— citing specific events or articles.\n"
            "4. VENUE-SCENE CONNECTIONS: The venue's role in the broader "
            "scene — what events it is known for, cited from research.\n"
            "5. GEOGRAPHIC PATTERNS: Whether artists are from the same "
            "city/region and how that relates to the event.\n"
            "6. TEMPORAL PATTERNS: Where this event falls in each "
            "artist's career timeline.\n"
            "7. SCENE CONTEXT: What movement or subgenre this event "
            "represents, grounded in cited facts.\n"
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
            "}"
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
    ) -> list[dict[str, Any]]:
        """Validate that each relationship cites an actual research source.

        A citation is considered valid if the ``source_citation`` text
        appears as a substring in any of the following research data fields:
        source names, article titles, article URLs, event names, or
        sources-consulted entries.

        Parameters
        ----------
        relationships:
            Raw relationship dicts from the LLM response.
        research_data:
            The original research results to validate against.

        Returns
        -------
        list[dict]
            Only relationships whose citations map to real research data.
        """
        # Build a set of known source strings for fast lookup
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

        validated: list[dict[str, Any]] = []
        discarded_count = 0

        for rel in relationships:
            citation_text = str(rel.get("source_citation", "")).lower().strip()
            if not citation_text:
                discarded_count += 1
                continue

            # Check if the citation text matches any known source
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
            citation = self._citation_service.build_citation(
                text=citation_text,
                source_name=citation_text,
                source_type="research",
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
        # Build a city lookup from research results
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

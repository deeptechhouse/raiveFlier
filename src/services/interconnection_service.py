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
            "You are an expert music historian specialising in underground "
            "electronic music culture."
        )
        user_prompt = self._build_synthesis_prompt(compiled_context)

        try:
            response = await self._llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=4000,
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
        """Compile all research results into a structured text summary.

        For each entity the summary includes: name, type, key data points,
        and the sources consulted.

        Parameters
        ----------
        results:
            Research profiles produced by the research phase.

        Returns
        -------
        str
            Multi-section text block ready for LLM consumption.
        """
        sections: list[str] = []

        for result in results:
            header = f"=== {result.entity_type.value}: {result.entity_name} ==="
            parts: list[str] = [header]

            if result.artist:
                artist = result.artist
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
                        line = f"  - {r.title}"
                        if r.label:
                            line += f" ({r.label})"
                        if r.year:
                            line += f" [{r.year}]"
                        rel_lines.append(line)
                    parts.append("Releases:\n" + "\n".join(rel_lines))
                if artist.appearances:
                    app_lines: list[str] = []
                    for a in artist.appearances[:10]:
                        line = f"  - {a.event_name or 'Event'}"
                        if a.venue:
                            line += f" at {a.venue}"
                        if a.date:
                            line += f" ({a.date.isoformat()})"
                        if a.source:
                            line += f" [source: {a.source}]"
                        app_lines.append(line)
                    parts.append("Appearances:\n" + "\n".join(app_lines))
                if artist.articles:
                    art_lines: list[str] = []
                    for a in artist.articles[:10]:
                        line = f"  - {a.title} ({a.source})"
                        if a.url:
                            line += f" [{a.url}]"
                        art_lines.append(line)
                    parts.append("Articles:\n" + "\n".join(art_lines))

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
                        line = f"  - {a.title} ({a.source})"
                        if a.url:
                            line += f" [{a.url}]"
                        art_lines.append(line)
                    parts.append("Articles:\n" + "\n".join(art_lines))

            if result.promoter:
                promoter = result.promoter
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
                        line = f"  - {a.title} ({a.source})"
                        if a.url:
                            line += f" [{a.url}]"
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

            if result.sources_consulted:
                parts.append("Sources consulted: " + ", ".join(result.sources_consulted))

            sections.append("\n".join(parts))

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

        Parameters
        ----------
        compiled_context:
            The compiled research context text.

        Returns
        -------
        str
            Full synthesis prompt with the context embedded.
        """
        return (
            "You are an expert music historian specializing in underground "
            "electronic music culture.\n"
            "\n"
            "You have been given detailed research on all entities connected "
            "to a single rave/electronic music event flier. Your task is to "
            "trace ALL interconnections, relationships, and historical threads "
            "linking these entities.\n"
            "\n"
            "RESEARCH DATA:\n"
            f"{compiled_context}\n"
            "\n"
            "ANALYSIS REQUIREMENTS:\n"
            "1. SHARED LABELS: Identify any record labels that multiple "
            "artists on this flier have released on.\n"
            "2. SHARED LINEUPS: Identify previous events where two or more "
            "of these artists appeared together.\n"
            "3. PROMOTER-ARTIST LINKS: Trace how the promoter connects to "
            "each artist (past bookings, shared scenes, geographic ties).\n"
            "4. VENUE-SCENE CONNECTIONS: How does this venue fit into the "
            "broader scene? What kind of events is it known for?\n"
            "5. GEOGRAPHIC PATTERNS: Are these artists from the same "
            "city/region? Is there a geographic cluster?\n"
            "6. TEMPORAL PATTERNS: How does this event fit into the timeline "
            "of these artists' careers?\n"
            "7. SCENE CONTEXT: What movement or subgenre does this event "
            "represent? Who are the key figures?\n"
            "\n"
            "CRITICAL RULES:\n"
            "- EVERY claim must cite a specific source from the research "
            "data provided\n"
            "- If you cannot find a source for a claim, DO NOT include it\n"
            "- Distinguish between confirmed facts and reasonable inferences\n"
            "- Flag uncertain connections with [UNCERTAIN]\n"
            "- Prioritize first-hand sources (books, contemporary press) "
            "over later retrospectives\n"
            "\n"
            "Return your analysis as JSON:\n"
            "{\n"
            '  "relationships": [\n'
            '    {"source": "entity1", "target": "entity2", '
            '"type": "relationship_type", "details": "explanation", '
            '"source_citation": "where this fact comes from", '
            '"confidence": 0.0-1.0}\n'
            "  ],\n"
            '  "patterns": [\n'
            '    {"type": "pattern_type", "description": "what the pattern '
            'is", "entities": ["entity1", "entity2"], '
            '"source_citation": "source"}\n'
            "  ],\n"
            '  "narrative": "A 2-3 paragraph narrative summary of how all '
            "these entities connect, written in an engaging style suitable "
            'for a music history reader."\n'
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
                for article in result.artist.articles:
                    known_sources.add(article.title.lower())
                    known_sources.add(article.source.lower())
                    if article.url:
                        known_sources.add(article.url.lower())
                for appearance in result.artist.appearances:
                    if appearance.source:
                        known_sources.add(appearance.source.lower())
                    if appearance.source_url:
                        known_sources.add(appearance.source_url.lower())
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
            if result.venue and result.venue.city:
                properties["city"] = result.venue.city

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

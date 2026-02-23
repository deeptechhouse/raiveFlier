"""RAG-powered Q&A service for interactive analysis exploration.

Accepts a user question together with session analysis context, queries
the RAG vector store for relevant passages, sends everything to the LLM,
and returns an answer plus related contextual facts.

Design decisions
----------------
- Follows the existing OOP adapter pattern: inject *interfaces*, not
  concrete classes (CLAUDE.md Section 5 / Section 6).
- When ``vector_store`` is ``None`` (RAG disabled), falls back to
  LLM-only answers using just the session context.
- Uses the cache provider to avoid re-asking identical questions.
- Generates 3-4 related facts in a single LLM call
  (not a separate call) to minimise latency.

Architecture overview for junior developers
--------------------------------------------
This module powers the interactive Q&A feature on the frontend.  After
the pipeline finishes analyzing a flier (extraction -> research ->
interconnection), the user can ask follow-up questions like "What label
did Artist X release on?" or "How are these two DJs connected?"

The data flow follows a classic RAG (Retrieval-Augmented Generation)
pattern:
  1. CACHE CHECK  -- Hash the question + session to see if we already
                     answered it.  Avoids duplicate LLM calls.
  2. CONTEXT BUILD -- Summarize the session's analysis (artists, venue,
                      research results, interconnection narrative) into
                      compact text the LLM can consume.
  3. RAG RETRIEVE -- Query the vector store for book/article passages
                     relevant to the question.  Filter by similarity
                     threshold (0.6) and deduplicate by source.
  4. LLM SYNTHESIS -- Send context + retrieved passages + question to
                      the LLM.  The LLM returns structured JSON with
                      an answer, citations, and 3-4 related facts.
  5. CACHE STORE  -- Cache the result for future identical questions.

When RAG is unavailable (no vector store injected), the service falls
back to LLM-only mode using only the session context -- degraded but
still functional.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import structlog

from src.interfaces.cache_provider import ICacheProvider
from src.interfaces.llm_provider import ILLMProvider
from src.interfaces.vector_store_provider import IVectorStoreProvider
from src.utils.logging import get_logger

logger: structlog.BoundLogger = get_logger(__name__)


class QAResponse:
    """Structured response from the Q&A service.

    Attributes
    ----------
    answer:
        The generated answer text.
    citations:
        List of citation dicts with ``text``, ``source``, ``tier``, and
        optional ``similarity`` keys.
    related_facts:
        List of contextual fact dicts with ``text``, optional
        ``category``, and optional ``entity_name`` keys.
    """

    def __init__(
        self,
        answer: str,
        citations: list[dict[str, Any]],
        related_facts: list[dict[str, Any]],
    ) -> None:
        # Internal state is stored in private fields; access is through
        # read-only @property methods below.  The lists are copied on
        # input AND on output (defensive copies) to enforce encapsulation
        # -- callers cannot mutate our internal state.
        self._answer = answer
        self._citations = list(citations)
        self._related_facts = list(related_facts)

    @property
    def answer(self) -> str:
        """Return the generated answer text."""
        return self._answer

    @property
    def citations(self) -> list[dict[str, Any]]:
        """Return a defensive copy of the citations list."""
        return list(self._citations)

    @property
    def related_facts(self) -> list[dict[str, Any]]:
        """Return a defensive copy of the related facts list."""
        return list(self._related_facts)


class QAService:
    """Answers user questions about analysis results using RAG retrieval + LLM.

    Parameters
    ----------
    llm:
        LLM provider used for generating answers.
    vector_store:
        Optional vector store for RAG retrieval.  When ``None``, the
        service falls back to LLM-only answers using session context.
    cache:
        Optional cache provider to avoid re-asking identical questions.
    """

    _SYSTEM_PROMPT = (
        "You are a knowledgeable assistant specializing in electronic music, rave culture, "
        "and the history of DJs, venues, labels, and promoters. You are helping a user explore "
        "the results of an automated analysis of a rave flier.\n\n"
        "You have access to:\n"
        "1. The analysis results from the flier (artists, venue, date, research findings)\n"
        "2. Relevant passages from a curated knowledge base of books, articles, and prior analyses\n\n"
        "Guidelines:\n"
        "- Answer concisely but thoroughly (2-4 paragraphs max)\n"
        "- Cite specific sources when the retrieved passages support your answer\n"
        "- If the retrieved passages don't contain relevant info, say so and answer from general knowledge\n"
        "- Be honest about uncertainty\n"
        "- Stay strictly on-topic: electronic music, rave culture, DJs, labels, venues, promoters, "
        "and the specific entities found on this flier. Do NOT generate content about unrelated topics.\n\n"
        "IMPORTANT: Your response MUST be valid JSON with this exact structure:\n"
        '{"answer": "your answer text here", "citations": [{"text": "citation text", '
        '"source": "source name", "tier": 1}], '
        '"related_facts": [{"text": "A concise, interesting fact", '
        '"category": "LABEL or HISTORY or SCENE or VENUE or ARTIST or CONNECTION", '
        '"entity_name": "name or null"}]}\n\n'
        "Include 3-4 related facts. Each fact MUST be:\n"
        "- A specific, true, interesting tidbit about one of the entities on this flier\n"
        "- Written as a short declarative statement (NOT a question)\n"
        "- Categories: LABEL (record labels artists released on), HISTORY (historical events "
        "around the flier's date/city), SCENE (rave/club scene context), VENUE (venue history), "
        "ARTIST (career facts, collaborations, discography), CONNECTION (links between flier entities)\n"
        "- Grounded in the entities listed in the flier analysis context below. "
        "Never invent facts about unrelated people, places, or topics."
    )

    # Minimum cosine similarity score for a RAG passage to be included in
    # the LLM prompt.  Below this threshold, passages are likely noise and
    # would dilute the answer quality.  Tuned empirically.
    _RAG_SIMILARITY_THRESHOLD = 0.6

    def __init__(
        self,
        llm: ILLMProvider,
        vector_store: IVectorStoreProvider | None = None,
        cache: ICacheProvider | None = None,
    ) -> None:
        self._llm = llm
        self._vector_store = vector_store
        self._cache = cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ask(
        self,
        question: str,
        session_context: dict[str, Any],
        entity_type: str | None = None,
        entity_name: str | None = None,
    ) -> QAResponse:
        """Answer a user question using RAG retrieval + session context + LLM.

        Parameters
        ----------
        question:
            The user's natural-language question.
        session_context:
            Dict containing ``session_id``, ``extracted_entities``,
            ``research_results``, and ``interconnection_map``.
        entity_type:
            Optional entity type filter (e.g. ``"ARTIST"``, ``"VENUE"``).
        entity_name:
            Optional entity name to focus the answer on.

        Returns
        -------
        QAResponse
            The generated answer, citations, and related facts.
        """
        # CACHE CHECK -- deterministic SHA-256 hash of (question, entity_type,
        # entity_name, session_id).  The session_id scoping ensures answers
        # from one flier analysis do not leak into another.
        cache_key = self._cache_key(
            question, entity_type, entity_name, session_context.get("session_id", "")
        )
        if self._cache:
            cached = await self._cache.get(cache_key)
            if cached:
                logger.debug("qa_cache_hit", question=question[:50])
                data = json.loads(cached)
                return QAResponse(
                    answer=data["answer"],
                    citations=data.get("citations", []),
                    related_facts=data.get("related_facts", data.get("suggested_questions", [])),
                )

        # BUILD CONTEXT -- summarize the completed analysis (entities, research
        # results, interconnection narrative) into a compact text block.
        context_summary = self._build_context_summary(
            session_context, entity_type, entity_name
        )

        # RAG RETRIEVE -- query the vector store for relevant passages from
        # the indexed corpus (books, articles, prior analyses).  Returns
        # formatted text plus citation metadata for the LLM to reference.
        rag_context = ""
        rag_citations: list[dict[str, Any]] = []
        if self._vector_store is not None:
            rag_context, rag_citations = await self._retrieve_passages(
                question, entity_name
            )

        if self._vector_store is not None and not rag_context:
            logger.warning(
                "qa_rag_empty",
                question=question[:80],
                entity_name=entity_name,
                msg="RAG enabled but no relevant passages found above similarity threshold.",
            )

        # Build user prompt
        user_prompt = self._build_user_prompt(
            question, context_summary, rag_context, entity_type, entity_name,
            session_context,
        )

        # LLM SYNTHESIS -- send context + RAG passages + question to the LLM.
        # Temperature 0.3 allows slight creativity for natural language while
        # staying factual.  The response is structured JSON with answer,
        # citations, and related facts.
        try:
            raw_response = await self._llm.complete(
                system_prompt=self._SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=2000,
            )

            response = self._parse_response(raw_response, rag_citations)

            # CACHE STORE -- persist the answer so repeated identical
            # questions skip the LLM entirely.
            if self._cache:
                cache_data = json.dumps({
                    "answer": response.answer,
                    "citations": response.citations,
                    "related_facts": response.related_facts,
                })
                await self._cache.set(cache_key, cache_data)

            logger.info(
                "qa_answered",
                question=question[:80],
                citations=len(response.citations),
                facts=len(response.related_facts),
                rag_used=self._vector_store is not None,
            )
            return response

        except Exception as exc:
            logger.error("qa_llm_failed", error=str(exc), question=question[:80])
            return QAResponse(
                answer="I wasn't able to answer that question right now. Please try again.",
                citations=[],
                related_facts=[],
            )

    # ------------------------------------------------------------------
    # Private helpers — context building
    # ------------------------------------------------------------------

    def _build_context_summary(
        self,
        session_context: dict[str, Any],
        entity_type: str | None = None,
        entity_name: str | None = None,
    ) -> str:
        """Build a compact text summary of the session analysis for LLM context."""
        parts: list[str] = []

        # Entities
        entities = session_context.get("extracted_entities") or {}
        if hasattr(entities, "model_dump"):
            entities = entities.model_dump()

        artists = entities.get("artists", [])
        if artists:
            names = [a.get("text", a.get("name", "")) for a in artists]
            parts.append(f"Artists on flier: {', '.join(names)}")

        venue = entities.get("venue")
        if venue:
            parts.append(f"Venue: {venue.get('text', venue.get('name', ''))}")

        date_ent = entities.get("date")
        if date_ent:
            parts.append(f"Date: {date_ent.get('text', '')}")

        promoter = entities.get("promoter")
        if promoter:
            parts.append(
                f"Promoter: {promoter.get('text', promoter.get('name', ''))}"
            )

        genre_tags = entities.get("genre_tags", [])
        if genre_tags:
            parts.append(f"Genres: {', '.join(genre_tags)}")

        # Research results summary
        research = session_context.get("research_results", [])
        if research:
            for result in research:
                if hasattr(result, "model_dump"):
                    result = result.model_dump()
                name = result.get("entity_name", "")

                # If user is asking about a specific entity, include detailed context
                if entity_name and name and entity_name.lower() in name.lower():
                    artist_data = result.get("artist", {})
                    if artist_data:
                        if artist_data.get("profile_summary"):
                            parts.append(f"\n--- Detailed context for {name} ---")
                            parts.append(
                                f"Profile: {artist_data['profile_summary']}"
                            )
                        releases = artist_data.get("releases", [])
                        if releases:
                            rel_strs = [
                                f"{r.get('title', '')} ({r.get('year', '')})"
                                for r in releases[:5]
                            ]
                            parts.append(f"Key releases: {', '.join(rel_strs)}")

                    venue_data = result.get("venue", {})
                    if venue_data and venue_data.get("history"):
                        parts.append(f"\n--- Detailed context for {name} ---")
                        parts.append(f"History: {venue_data['history']}")

        # Interconnections
        imap = session_context.get("interconnection_map")
        if imap:
            if hasattr(imap, "model_dump"):
                imap = imap.model_dump()
            narrative = imap.get("narrative")
            if narrative:
                parts.append(f"\nInterconnection narrative: {narrative}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Private helpers — RAG retrieval
    # ------------------------------------------------------------------

    async def _retrieve_passages(
        self, question: str, entity_name: str | None
    ) -> tuple[str, list[dict[str, Any]]]:
        """Query the vector store for relevant passages.

        Returns a ``(rag_text, citations)`` tuple.
        """
        if not self._vector_store:
            return "", []

        query = question
        if entity_name:
            query = f"{entity_name}: {question}"

        filters: dict[str, Any] = {}
        if entity_name:
            filters["entity_tags"] = {"$contains": entity_name}

        try:
            chunks = await self._vector_store.query(
                query_text=query,
                top_k=10,
                filters=filters if filters else None,
            )

            # Deduplicate by source_id -- keep top 3 chunks per source.
            # Without this cap, a single long book could dominate the entire
            # RAG context, crowding out other sources.  3 chunks per source
            # is a balance between depth and breadth.
            _MAX_PER_SOURCE = 3
            source_chunks: dict[str, list[tuple[float, str, dict[str, Any]]]] = {}

            for chunk in chunks:
                if chunk.similarity_score < self._RAG_SIMILARITY_THRESHOLD:
                    continue
                sid = chunk.chunk.source_id
                passage = (
                    f"[Source: {chunk.chunk.source_title}, "
                    f"Tier {chunk.chunk.citation_tier}]\n{chunk.chunk.text}"
                )
                citation = {
                    "text": chunk.formatted_citation,
                    "source": chunk.chunk.source_title,
                    "tier": chunk.chunk.citation_tier,
                    "similarity": round(chunk.similarity_score, 3),
                }
                source_chunks.setdefault(sid, []).append(
                    (chunk.similarity_score, passage, citation)
                )

            # Keep top N per source, sort globally
            deduped: list[tuple[float, str, dict[str, Any]]] = []
            for entries in source_chunks.values():
                entries.sort(key=lambda t: t[0], reverse=True)
                deduped.extend(entries[:_MAX_PER_SOURCE])

            ordered = sorted(deduped, key=lambda t: t[0], reverse=True)
            passages = [p for _, p, _ in ordered]
            citations = [c for _, _, c in ordered]

            rag_text = "\n\n---\n\n".join(passages) if passages else ""
            return rag_text, citations

        except Exception as exc:
            logger.warning("qa_rag_retrieval_failed", error=str(exc))
            return "", []

    # ------------------------------------------------------------------
    # Private helpers — prompt assembly
    # ------------------------------------------------------------------

    def _build_user_prompt(
        self,
        question: str,
        context_summary: str,
        rag_context: str,
        entity_type: str | None,
        entity_name: str | None,
        session_context: dict[str, Any] | None = None,
    ) -> str:
        """Assemble the full user prompt for the LLM."""
        parts: list[str] = []

        parts.append("## Flier Analysis Context")
        parts.append(context_summary)

        if rag_context:
            # Token budget guard: at ~4 chars/token, 80K chars is ~20K tokens.
            # We reserve the remaining context window for the session summary,
            # question, system prompt, and response generation.
            max_rag_chars = 80_000
            if len(rag_context) > max_rag_chars:
                logger.info(
                    "qa_rag_context_truncated",
                    original_chars=len(rag_context),
                    max_chars=max_rag_chars,
                )
                rag_context = rag_context[:max_rag_chars]
            parts.append("\n## Retrieved Knowledge Base Passages")
            parts.append(rag_context)

        if entity_type and entity_name:
            parts.append(
                f"\n## Focus Entity\nType: {entity_type}\nName: {entity_name}"
            )

        # Provide an explicit list of entity names so the LLM anchors its
        # "related facts" to actual flier content.  Without this constraint,
        # the LLM tends to invent facts about unrelated artists/venues.
        entity_names = self._extract_entity_names(session_context)
        if entity_names:
            parts.append(
                "\n## Entities on this flier (use ONLY these for related facts)\n"
                + ", ".join(entity_names)
            )

        parts.append(f"\n## User Question\n{question}")
        parts.append(
            "\nRespond with valid JSON containing 'answer', 'citations', "
            "and 'related_facts' fields. "
            "Each related fact MUST reference one of the entities listed above."
        )

        return "\n".join(parts)

    @staticmethod
    def _extract_entity_names(session_context: dict[str, Any] | None) -> list[str]:
        """Extract all entity names from the session context for prompt anchoring."""
        if not session_context:
            return []

        names: list[str] = []
        entities = session_context.get("extracted_entities") or {}
        if hasattr(entities, "model_dump"):
            entities = entities.model_dump()

        for artist in entities.get("artists", []):
            name = artist.get("text", artist.get("name", ""))
            if name:
                names.append(name)

        venue = entities.get("venue")
        if venue:
            name = venue.get("text", venue.get("name", ""))
            if name:
                names.append(name)

        promoter = entities.get("promoter")
        if promoter:
            name = promoter.get("text", promoter.get("name", ""))
            if name:
                names.append(name)

        return names

    # ------------------------------------------------------------------
    # Private helpers — response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self, raw: str, rag_citations: list[dict[str, Any]]
    ) -> QAResponse:
        """Parse the LLM's JSON response into a ``QAResponse``."""
        text = raw.strip()

        # Handle markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # Handle LLM responses that wrap JSON in prose (e.g. "Here is my response: {...}")
        if not text.startswith("{"):
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                text = json_match.group(0)

        try:
            data = json.loads(text)
            answer = data.get("answer", text)
            llm_citations = data.get("citations", [])
            # Support both new key and legacy key for transitional compatibility
            facts = data.get("related_facts", data.get("suggested_questions", []))

            # Merge RAG citations (from the vector store) with any citations
            # the LLM itself mentioned in its answer.  Deduplicate by text
            # to avoid showing the same citation twice in the UI.
            all_citations = list(rag_citations)
            for citation in llm_citations:
                if isinstance(citation, dict) and citation.get("text"):
                    # Avoid duplicates
                    if not any(
                        existing.get("text") == citation["text"]
                        for existing in all_citations
                    ):
                        all_citations.append(citation)

            return QAResponse(
                answer=answer,
                citations=all_citations,
                related_facts=facts,
            )

        except (json.JSONDecodeError, KeyError):
            # If JSON parsing fails, use raw text as answer
            logger.warning(
                "qa_json_parse_failed", response_preview=text[:200]
            )
            return QAResponse(
                answer=text,
                citations=list(rag_citations),
                related_facts=[],
            )

    # ------------------------------------------------------------------
    # Private helpers — caching
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(
        question: str,
        entity_type: str | None,
        entity_name: str | None,
        session_id: str,
    ) -> str:
        """Generate a deterministic cache key for a Q&A query.

        The key is scoped by session_id so answers from one flier analysis
        never collide with another.  The question is lowercased to treat
        "Who is DJ X?" and "who is dj x?" as identical queries.  SHA-256
        truncated to 32 hex chars keeps keys short while avoiding collisions.
        """
        raw = f"qa:{session_id}:{entity_type}:{entity_name}:{question.lower().strip()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

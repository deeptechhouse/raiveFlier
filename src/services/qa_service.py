"""RAG-powered Q&A service for interactive analysis exploration.

Accepts a user question together with session analysis context, queries
the RAG vector store for relevant passages, sends everything to the LLM,
and returns an answer plus suggested follow-up questions.

Design decisions
----------------
- Follows the existing OOP adapter pattern: inject *interfaces*, not
  concrete classes (CLAUDE.md Section 5 / Section 6).
- When ``vector_store`` is ``None`` (RAG disabled), falls back to
  LLM-only answers using just the session context.
- Uses the cache provider to avoid re-asking identical questions.
- Generates 3-4 suggested follow-up questions in a single LLM call
  (not a separate call) to minimise latency.
"""

from __future__ import annotations

import hashlib
import json
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
    suggested_questions:
        List of follow-up question dicts with ``text``, optional
        ``entity_type``, and optional ``entity_name`` keys.
    """

    def __init__(
        self,
        answer: str,
        citations: list[dict[str, Any]],
        suggested_questions: list[dict[str, Any]],
    ) -> None:
        self._answer = answer
        self._citations = list(citations)
        self._suggested_questions = list(suggested_questions)

    @property
    def answer(self) -> str:
        """Return the generated answer text."""
        return self._answer

    @property
    def citations(self) -> list[dict[str, Any]]:
        """Return a defensive copy of the citations list."""
        return list(self._citations)

    @property
    def suggested_questions(self) -> list[dict[str, Any]]:
        """Return a defensive copy of the suggested questions list."""
        return list(self._suggested_questions)


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
        "- Be honest about uncertainty\n\n"
        "IMPORTANT: Your response MUST be valid JSON with this exact structure:\n"
        '{"answer": "your answer text here", "citations": [{"text": "citation text", '
        '"source": "source name", "tier": 1}], '
        '"suggested_questions": [{"text": "follow-up question?", "entity_type": "ARTIST or '
        'VENUE or null", "entity_name": "name or null"}]}\n'
        "Include 3-4 suggested follow-up questions that the user would naturally want to ask next, "
        "related to the topic. Make them specific and interesting."
    )

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
            The generated answer, citations, and suggested follow-ups.
        """
        # Check cache first
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
                    suggested_questions=data.get("suggested_questions", []),
                )

        # Build context summary from session analysis
        context_summary = self._build_context_summary(
            session_context, entity_type, entity_name
        )

        # Retrieve from RAG corpus if available
        rag_context = ""
        rag_citations: list[dict[str, Any]] = []
        if self._vector_store is not None:
            rag_context, rag_citations = await self._retrieve_passages(
                question, entity_name
            )

        # Build user prompt
        user_prompt = self._build_user_prompt(
            question, context_summary, rag_context, entity_type, entity_name
        )

        # Call LLM
        try:
            raw_response = await self._llm.complete(
                system_prompt=self._SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=2000,
            )

            response = self._parse_response(raw_response, rag_citations)

            # Cache the result
            if self._cache:
                cache_data = json.dumps({
                    "answer": response.answer,
                    "citations": response.citations,
                    "suggested_questions": response.suggested_questions,
                })
                await self._cache.set(cache_key, cache_data)

            logger.info(
                "qa_answered",
                question=question[:80],
                citations=len(response.citations),
                suggestions=len(response.suggested_questions),
                rag_used=self._vector_store is not None,
            )
            return response

        except Exception as exc:
            logger.error("qa_llm_failed", error=str(exc), question=question[:80])
            return QAResponse(
                answer="I wasn't able to answer that question right now. Please try again.",
                citations=[],
                suggested_questions=[],
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
                        appearances = artist_data.get("appearances", [])
                        if appearances:
                            app_strs = [
                                f"{a.get('event_name', '')} @ {a.get('venue', '')}"
                                for a in appearances[:5]
                            ]
                            parts.append(
                                f"Past appearances: {', '.join(app_strs)}"
                            )

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

            passages: list[str] = []
            citations: list[dict[str, Any]] = []

            for chunk in chunks:
                if chunk.similarity_score < self._RAG_SIMILARITY_THRESHOLD:
                    continue
                passages.append(
                    f"[Source: {chunk.chunk.source_title}, "
                    f"Tier {chunk.chunk.citation_tier}]\n{chunk.chunk.text}"
                )
                citations.append({
                    "text": chunk.formatted_citation,
                    "source": chunk.chunk.source_title,
                    "tier": chunk.chunk.citation_tier,
                    "similarity": round(chunk.similarity_score, 3),
                })

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
    ) -> str:
        """Assemble the full user prompt for the LLM."""
        parts: list[str] = []

        parts.append("## Flier Analysis Context")
        parts.append(context_summary)

        if rag_context:
            parts.append("\n## Retrieved Knowledge Base Passages")
            parts.append(rag_context)

        if entity_type and entity_name:
            parts.append(
                f"\n## Focus Entity\nType: {entity_type}\nName: {entity_name}"
            )

        parts.append(f"\n## User Question\n{question}")
        parts.append(
            "\nRespond with valid JSON containing 'answer', 'citations', "
            "and 'suggested_questions' fields."
        )

        return "\n".join(parts)

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

        try:
            data = json.loads(text)
            answer = data.get("answer", text)
            llm_citations = data.get("citations", [])
            suggestions = data.get("suggested_questions", [])

            # Merge RAG citations with LLM-mentioned citations
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
                suggested_questions=suggestions,
            )

        except (json.JSONDecodeError, KeyError):
            # If JSON parsing fails, use raw text as answer
            logger.warning(
                "qa_json_parse_failed", response_preview=text[:200]
            )
            return QAResponse(
                answer=text,
                citations=list(rag_citations),
                suggested_questions=[],
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
        """Generate a deterministic cache key for a Q&A query."""
        raw = f"qa:{session_id}:{entity_type}:{entity_name}:{question.lower().strip()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

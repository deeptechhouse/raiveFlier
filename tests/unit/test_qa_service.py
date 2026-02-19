"""Unit tests for QAService — RAG-powered interactive Q&A."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# These imports will work once the service is created
from src.services.qa_service import QAResponse, QAService


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture()
def mock_llm() -> AsyncMock:
    """Create a mock LLM provider."""
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=json.dumps({
        "answer": "Test answer about the artist.",
        "citations": [{"text": "Energy Flash, p.142", "source": "Energy Flash", "tier": 1}],
        "suggested_questions": [
            {"text": "What labels has this artist released on?", "entity_type": "ARTIST", "entity_name": "Test DJ"},
            {"text": "Who else played at this venue?", "entity_type": "VENUE", "entity_name": None},
            {"text": "What was the scene like at this time?", "entity_type": None, "entity_name": None},
        ],
    }))
    return llm


@pytest.fixture()
def mock_vector_store() -> AsyncMock:
    """Create a mock vector store provider."""
    store = AsyncMock()

    # Create mock RetrievedChunk objects
    mock_chunk = MagicMock()
    mock_chunk.similarity_score = 0.85
    mock_chunk.chunk.text = "Test passage about the artist from Energy Flash."
    mock_chunk.chunk.source_title = "Energy Flash"
    mock_chunk.chunk.citation_tier = 1
    mock_chunk.formatted_citation = "Energy Flash, Simon Reynolds, p.142, 1998 [Tier 1]"

    store.query = AsyncMock(return_value=[mock_chunk])
    return store


@pytest.fixture()
def mock_cache() -> AsyncMock:
    """Create a mock cache provider."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)  # Cache miss by default
    cache.set = AsyncMock()
    return cache


@pytest.fixture()
def sample_session_context() -> dict[str, Any]:
    """Create a sample session context dict."""
    return {
        "session_id": "test-session-123",
        "extracted_entities": {
            "artists": [
                {"text": "Test DJ", "name": "Test DJ", "confidence": 0.9},
                {"text": "Another Artist", "name": "Another Artist", "confidence": 0.8},
            ],
            "venue": {"text": "Underground Club", "name": "Underground Club"},
            "date": {"text": "March 15, 1997"},
            "promoter": {"text": "Night Crew", "name": "Night Crew"},
            "genre_tags": ["techno", "house"],
        },
        "research_results": [],
        "interconnection_map": None,
    }


# ── Basic Functionality ───────────────────────────────────


class TestQAServiceBasic:
    """Test basic Q&A functionality with all dependencies."""

    @pytest.mark.asyncio()
    async def test_ask_returns_qa_response(
        self, mock_llm: AsyncMock, mock_vector_store: AsyncMock, mock_cache: AsyncMock, sample_session_context: dict
    ) -> None:
        service = QAService(llm=mock_llm, vector_store=mock_vector_store, cache=mock_cache)
        result = await service.ask("Tell me about this artist", sample_session_context, "ARTIST", "Test DJ")

        assert isinstance(result, QAResponse)
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    @pytest.mark.asyncio()
    async def test_ask_returns_citations(
        self, mock_llm: AsyncMock, mock_vector_store: AsyncMock, mock_cache: AsyncMock, sample_session_context: dict
    ) -> None:
        service = QAService(llm=mock_llm, vector_store=mock_vector_store, cache=mock_cache)
        result = await service.ask("Tell me about this artist", sample_session_context, "ARTIST", "Test DJ")

        assert isinstance(result.citations, list)

    @pytest.mark.asyncio()
    async def test_ask_returns_suggested_questions(
        self, mock_llm: AsyncMock, mock_vector_store: AsyncMock, mock_cache: AsyncMock, sample_session_context: dict
    ) -> None:
        service = QAService(llm=mock_llm, vector_store=mock_vector_store, cache=mock_cache)
        result = await service.ask("Tell me about this artist", sample_session_context, "ARTIST", "Test DJ")

        assert isinstance(result.suggested_questions, list)
        assert len(result.suggested_questions) > 0

    @pytest.mark.asyncio()
    async def test_ask_calls_vector_store_query(
        self, mock_llm: AsyncMock, mock_vector_store: AsyncMock, mock_cache: AsyncMock, sample_session_context: dict
    ) -> None:
        service = QAService(llm=mock_llm, vector_store=mock_vector_store, cache=mock_cache)
        await service.ask("Tell me about this artist", sample_session_context, "ARTIST", "Test DJ")

        mock_vector_store.query.assert_called_once()

    @pytest.mark.asyncio()
    async def test_ask_calls_llm_complete(
        self, mock_llm: AsyncMock, mock_vector_store: AsyncMock, mock_cache: AsyncMock, sample_session_context: dict
    ) -> None:
        service = QAService(llm=mock_llm, vector_store=mock_vector_store, cache=mock_cache)
        await service.ask("What genre is this?", sample_session_context)

        mock_llm.complete.assert_called_once()


# ── RAG Disabled (No Vector Store) ────────────────────────


class TestQAServiceNoRAG:
    """Test Q&A when RAG is disabled (vector_store=None)."""

    @pytest.mark.asyncio()
    async def test_works_without_vector_store(
        self, mock_llm: AsyncMock, mock_cache: AsyncMock, sample_session_context: dict
    ) -> None:
        service = QAService(llm=mock_llm, vector_store=None, cache=mock_cache)
        result = await service.ask("Tell me about this artist", sample_session_context, "ARTIST", "Test DJ")

        assert isinstance(result, QAResponse)
        assert len(result.answer) > 0

    @pytest.mark.asyncio()
    async def test_no_vector_store_still_calls_llm(
        self, mock_llm: AsyncMock, sample_session_context: dict
    ) -> None:
        service = QAService(llm=mock_llm, vector_store=None, cache=None)
        await service.ask("What genre is this?", sample_session_context)

        mock_llm.complete.assert_called_once()


# ── Caching ───────────────────────────────────────────────


class TestQAServiceCaching:
    """Test caching behavior."""

    @pytest.mark.asyncio()
    async def test_caches_result_after_llm_call(
        self, mock_llm: AsyncMock, mock_vector_store: AsyncMock, mock_cache: AsyncMock, sample_session_context: dict
    ) -> None:
        service = QAService(llm=mock_llm, vector_store=mock_vector_store, cache=mock_cache)
        await service.ask("Tell me about this artist", sample_session_context, "ARTIST", "Test DJ")

        mock_cache.set.assert_called_once()

    @pytest.mark.asyncio()
    async def test_returns_cached_result_on_hit(
        self, mock_llm: AsyncMock, mock_vector_store: AsyncMock, mock_cache: AsyncMock, sample_session_context: dict
    ) -> None:
        cached_data = json.dumps({
            "answer": "Cached answer",
            "citations": [],
            "suggested_questions": [{"text": "cached question?"}],
        })
        mock_cache.get = AsyncMock(return_value=cached_data)

        service = QAService(llm=mock_llm, vector_store=mock_vector_store, cache=mock_cache)
        result = await service.ask("Tell me about this artist", sample_session_context, "ARTIST", "Test DJ")

        assert result.answer == "Cached answer"
        mock_llm.complete.assert_not_called()
        mock_vector_store.query.assert_not_called()

    @pytest.mark.asyncio()
    async def test_works_without_cache(
        self, mock_llm: AsyncMock, mock_vector_store: AsyncMock, sample_session_context: dict
    ) -> None:
        service = QAService(llm=mock_llm, vector_store=mock_vector_store, cache=None)
        result = await service.ask("Tell me about this artist", sample_session_context)

        assert isinstance(result, QAResponse)


# ── Error Handling ────────────────────────────────────────


class TestQAServiceErrors:
    """Test graceful error handling."""

    @pytest.mark.asyncio()
    async def test_llm_failure_returns_error_message(
        self, mock_vector_store: AsyncMock, mock_cache: AsyncMock, sample_session_context: dict
    ) -> None:
        failing_llm = AsyncMock()
        failing_llm.complete = AsyncMock(side_effect=Exception("LLM API error"))

        service = QAService(llm=failing_llm, vector_store=mock_vector_store, cache=mock_cache)
        result = await service.ask("Tell me about this artist", sample_session_context)

        assert isinstance(result, QAResponse)
        assert "couldn't" in result.answer.lower() or "unable" in result.answer.lower() or "wasn't" in result.answer.lower()

    @pytest.mark.asyncio()
    async def test_vector_store_failure_still_answers(
        self, mock_llm: AsyncMock, mock_cache: AsyncMock, sample_session_context: dict
    ) -> None:
        failing_store = AsyncMock()
        failing_store.query = AsyncMock(side_effect=Exception("ChromaDB error"))

        service = QAService(llm=mock_llm, vector_store=failing_store, cache=mock_cache)
        result = await service.ask("Tell me about this artist", sample_session_context)

        # Should still return an answer (LLM-only fallback)
        assert isinstance(result, QAResponse)
        mock_llm.complete.assert_called_once()

    @pytest.mark.asyncio()
    async def test_malformed_llm_json_uses_raw_text(
        self, mock_vector_store: AsyncMock, sample_session_context: dict
    ) -> None:
        bad_json_llm = AsyncMock()
        bad_json_llm.complete = AsyncMock(return_value="This is a plain text response, not JSON.")

        service = QAService(llm=bad_json_llm, vector_store=mock_vector_store, cache=None)
        result = await service.ask("Tell me about this", sample_session_context)

        assert isinstance(result, QAResponse)
        assert "plain text response" in result.answer


# ── Context Building ──────────────────────────────────────


class TestQAServiceContext:
    """Test context summary building."""

    @pytest.mark.asyncio()
    async def test_includes_entity_context_in_llm_prompt(
        self, mock_llm: AsyncMock, mock_vector_store: AsyncMock, sample_session_context: dict
    ) -> None:
        service = QAService(llm=mock_llm, vector_store=mock_vector_store, cache=None)
        await service.ask("What about this artist?", sample_session_context, "ARTIST", "Test DJ")

        call_args = mock_llm.complete.call_args
        user_prompt = call_args.kwargs.get("user_prompt", call_args[1] if len(call_args.args) > 1 else "")
        # The user prompt should contain the entity name
        assert "Test DJ" in str(user_prompt) or "Test DJ" in str(call_args)

    @pytest.mark.asyncio()
    async def test_includes_session_entities_in_context(
        self, mock_llm: AsyncMock, sample_session_context: dict
    ) -> None:
        service = QAService(llm=mock_llm, vector_store=None, cache=None)
        await service.ask("What's happening here?", sample_session_context)

        call_args = mock_llm.complete.call_args
        prompt_text = str(call_args)
        # Should include data from the session context
        assert "Underground Club" in prompt_text or "Test DJ" in prompt_text

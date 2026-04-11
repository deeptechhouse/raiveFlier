"""Unit tests for Phase 4 LLM streaming and interconnection streaming.

# ─── MODULE OVERVIEW ───
# Tests the streaming extensions to the raiveFlier pipeline:
#
# Part 1: ILLMProvider.stream_complete() — the default implementation on the
#   abstract base class that wraps complete() in a single-chunk async generator.
#   Concrete providers (OpenAI, Anthropic) will override this with real
#   token-level streaming; the default ensures every provider supports the
#   streaming interface even without native streaming support.
#
# Part 2: InterconnectionService.analyze_streaming() — the async generator
#   that yields {"type": "narrative_chunk", ...} dicts as LLM tokens arrive,
#   followed by a final {"type": "analysis_complete", ...} with the parsed
#   InterconnectionMap.  This enables real-time WebSocket delivery to the
#   frontend while reusing the same parse/validate/enrich pipeline as the
#   non-streaming analyze() method.
#
# Architecture: Tests layer (tests/unit/) — depends on interfaces and
#   services layers.  All external dependencies are mocked.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.interfaces.llm_provider import ILLMProvider
from src.models.entities import EntityType
from src.models.flier import ExtractedEntities, ExtractedEntity, OCRResult
from src.models.research import ResearchResult
from src.services.citation_service import CitationService
from src.services.interconnection_service import InterconnectionService
from src.utils.errors import LLMError

# ======================================================================
# Part 1: ILLMProvider.stream_complete() default implementation
# ======================================================================


class _MinimalLLMProvider(ILLMProvider):
    """Concrete subclass that only implements the required abstract methods.

    Used to verify that the default stream_complete() on the ABC works
    correctly — it should call complete() and yield the full result as a
    single chunk without requiring the subclass to override anything.
    """

    def __init__(self, complete_return: str = "full response") -> None:
        self._complete_return = complete_return
        self._last_call_kwargs: dict[str, Any] = {}

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 4000,
    ) -> str:
        # Record the call params so tests can verify pass-through
        self._last_call_kwargs = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        return self._complete_return

    async def vision_extract(self, image_bytes: bytes, prompt: str) -> str:
        raise NotImplementedError

    def supports_vision(self) -> bool:
        return False

    def get_provider_name(self) -> str:
        return "minimal-test"

    def is_available(self) -> bool:
        return True

    async def validate_credentials(self) -> bool:
        return True


class TestDefaultStreamComplete:
    """Tests for the default stream_complete() on ILLMProvider."""

    @pytest.mark.asyncio()
    async def test_default_stream_complete_yields_full_result(self) -> None:
        """Default stream_complete calls complete() and yields the full result
        as a single chunk."""
        provider = _MinimalLLMProvider(complete_return="full response")

        chunks: list[str] = []
        async for chunk in provider.stream_complete("sys", "usr"):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0] == "full response"

    @pytest.mark.asyncio()
    async def test_default_stream_complete_passes_params(self) -> None:
        """Verify temperature and max_tokens are forwarded to complete()."""
        provider = _MinimalLLMProvider()

        # Consume the generator to trigger the complete() call
        async for _ in provider.stream_complete(
            "system", "user", temperature=0.7, max_tokens=2000
        ):
            pass

        assert provider._last_call_kwargs["system_prompt"] == "system"
        assert provider._last_call_kwargs["user_prompt"] == "user"
        assert provider._last_call_kwargs["temperature"] == 0.7
        assert provider._last_call_kwargs["max_tokens"] == 2000


# ======================================================================
# Part 2: InterconnectionService.analyze_streaming()
# ======================================================================

# Valid JSON that _parse_analysis_response can parse — split across
# three chunks to simulate token-level streaming from an LLM provider.
_VALID_JSON_CHUNKS = [
    '{"relationships": [], ',
    '"patterns": [], ',
    '"narrative": "Test narrative."}',
]

# The assembled JSON string for reference in assertions.
_FULL_JSON = "".join(_VALID_JSON_CHUNKS)


@pytest.fixture()
def _mock_entities() -> ExtractedEntities:
    """Minimal ExtractedEntities with one artist for streaming tests."""
    return ExtractedEntities(
        artists=[
            ExtractedEntity(
                text="DJ Test",
                entity_type=EntityType.ARTIST,
                confidence=0.9,
            ),
        ],
        raw_ocr=OCRResult(
            raw_text="DJ Test",
            confidence=0.9,
            provider_used="mock",
            processing_time=0.1,
            bounding_boxes=[],
        ),
    )


@pytest.fixture()
def _mock_research_results() -> list[ResearchResult]:
    """Minimal research results list with one ARTIST entry."""
    return [
        ResearchResult(
            entity_type=EntityType.ARTIST,
            entity_name="DJ Test",
            sources_consulted=[],
            confidence=0.8,
            warnings=[],
        ),
    ]


async def _make_mock_stream(
    chunks: list[str] | None = None,
) -> AsyncGenerator[str, None]:
    """Helper: async generator that yields predetermined chunks."""
    for chunk in chunks or _VALID_JSON_CHUNKS:
        yield chunk


def _build_streaming_service() -> (
    tuple[InterconnectionService, MagicMock, MagicMock]
):
    """Build an InterconnectionService wired with mock LLM and citation service.

    Returns (service, mock_llm, mock_citation_service).
    The mock LLM's stream_complete is wired to yield _VALID_JSON_CHUNKS.
    """
    mock_llm = MagicMock(spec=ILLMProvider)
    mock_llm.get_provider_name.return_value = "mock"

    # Wire stream_complete to the async generator helper
    mock_llm.stream_complete = lambda **kwargs: _make_mock_stream()

    mock_citation = MagicMock(spec=CitationService)
    mock_citation.rank_citations = MagicMock(return_value=[])

    service = InterconnectionService(
        llm_provider=mock_llm,
        citation_service=mock_citation,
        vector_store=None,
    )
    return service, mock_llm, mock_citation


class TestAnalyzeStreaming:
    """Tests for InterconnectionService.analyze_streaming()."""

    @pytest.mark.asyncio()
    async def test_streaming_yields_narrative_chunks(
        self,
        _mock_entities: ExtractedEntities,
        _mock_research_results: list[ResearchResult],
    ) -> None:
        """At least one narrative_chunk dict is yielded during streaming."""
        service, _, _ = _build_streaming_service()

        results: list[dict[str, Any]] = []
        async for item in service.analyze_streaming(
            _mock_research_results, _mock_entities
        ):
            results.append(item)

        narrative_chunks = [r for r in results if r["type"] == "narrative_chunk"]
        assert len(narrative_chunks) >= 1

    @pytest.mark.asyncio()
    async def test_streaming_yields_analysis_complete(
        self,
        _mock_entities: ExtractedEntities,
        _mock_research_results: list[ResearchResult],
    ) -> None:
        """The last yielded dict has type='analysis_complete'."""
        service, _, _ = _build_streaming_service()

        results: list[dict[str, Any]] = []
        async for item in service.analyze_streaming(
            _mock_research_results, _mock_entities
        ):
            results.append(item)

        assert results[-1]["type"] == "analysis_complete"

    @pytest.mark.asyncio()
    async def test_streaming_complete_has_interconnection_map(
        self,
        _mock_entities: ExtractedEntities,
        _mock_research_results: list[ResearchResult],
    ) -> None:
        """The analysis_complete dict contains an 'interconnection_map' key."""
        service, _, _ = _build_streaming_service()

        results: list[dict[str, Any]] = []
        async for item in service.analyze_streaming(
            _mock_research_results, _mock_entities
        ):
            results.append(item)

        final = results[-1]
        assert "interconnection_map" in final

    @pytest.mark.asyncio()
    async def test_streaming_accumulates_full_response(
        self,
        _mock_entities: ExtractedEntities,
        _mock_research_results: list[ResearchResult],
    ) -> None:
        """The interconnection_map is properly parsed from accumulated chunks."""
        service, _, _ = _build_streaming_service()

        results: list[dict[str, Any]] = []
        async for item in service.analyze_streaming(
            _mock_research_results, _mock_entities
        ):
            results.append(item)

        imap = results[-1]["interconnection_map"]

        # The mock JSON has an empty relationships list, empty patterns,
        # and a narrative string — verify they survived parsing.
        assert imap["narrative"] == "Test narrative."
        assert isinstance(imap["edges"], list)
        assert isinstance(imap["patterns"], list)

    @pytest.mark.asyncio()
    async def test_streaming_error_raises_llm_error(
        self,
        _mock_entities: ExtractedEntities,
        _mock_research_results: list[ResearchResult],
    ) -> None:
        """When stream_complete raises, analyze_streaming wraps it in LLMError."""

        async def _exploding_stream(**kwargs: Any) -> AsyncGenerator[str, None]:
            raise RuntimeError("LLM connection lost")
            # The yield makes this a generator function even though it never
            # reaches this line — required for Python to treat it as an
            # async generator rather than a plain coroutine.
            yield ""  # pragma: no cover

        mock_llm = MagicMock(spec=ILLMProvider)
        mock_llm.get_provider_name.return_value = "mock"
        mock_llm.stream_complete = _exploding_stream

        mock_citation = MagicMock(spec=CitationService)
        mock_citation.rank_citations = MagicMock(return_value=[])

        service = InterconnectionService(
            llm_provider=mock_llm,
            citation_service=mock_citation,
            vector_store=None,
        )

        with pytest.raises(LLMError, match="streaming .* failed"):
            async for _ in service.analyze_streaming(
                _mock_research_results, _mock_entities
            ):
                pass

"""Unit tests for the EntityExtractor service with mocked LLM provider."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.interfaces.llm_provider import ILLMProvider
from src.models.entities import EntityType
from src.models.flier import ExtractedEntities, OCRResult
from src.services.entity_extractor import EntityExtractor
from src.utils.errors import EntityExtractionError


# ======================================================================
# Helpers
# ======================================================================


def _make_mock_llm(response: str) -> ILLMProvider:
    """Create a mock LLM provider that returns *response* from complete()."""
    mock = MagicMock(spec=ILLMProvider)
    mock.complete = AsyncMock(return_value=response)
    mock.get_provider_name.return_value = "mock-llm"
    return mock


def _make_ocr(raw_text: str = "CARL COX\nTresor Berlin\n15 March 1997") -> OCRResult:
    """Create a simple OCRResult fixture."""
    return OCRResult(
        raw_text=raw_text,
        confidence=0.9,
        provider_used="tesseract",
        processing_time=0.5,
    )


def _valid_json_response(**overrides: Any) -> str:
    """Build a valid JSON response string with optional overrides."""
    data: dict[str, Any] = {
        "artists": [
            {"name": "Carl Cox", "confidence": 0.95},
        ],
        "venue": {"name": "Tresor Berlin", "confidence": 0.9},
        "date": {"text": "15 March 1997", "confidence": 0.85},
        "promoter": {"name": "Tresor Records", "confidence": 0.7},
        "genre_tags": ["techno"],
        "ticket_price": "10 DM",
    }
    data.update(overrides)
    return json.dumps(data)


# ======================================================================
# Tests
# ======================================================================


class TestEntityExtractor:
    """Tests for EntityExtractor.extract()."""

    @pytest.mark.asyncio()
    async def test_successful_extraction(self) -> None:
        mock_llm = _make_mock_llm(_valid_json_response())
        extractor = EntityExtractor(llm_provider=mock_llm)
        ocr = _make_ocr()

        result = await extractor.extract(ocr)

        assert isinstance(result, ExtractedEntities)
        assert len(result.artists) == 1
        assert result.artists[0].text == "Carl Cox"
        assert result.artists[0].entity_type == EntityType.ARTIST
        assert result.venue is not None
        assert result.venue.text == "Tresor Berlin"
        assert result.date is not None
        assert result.date.text == "15 March 1997"
        assert result.promoter is not None
        assert result.promoter.text == "Tresor Records"
        assert result.genre_tags == ["techno"]
        assert result.ticket_price == "10 DM"

    @pytest.mark.asyncio()
    async def test_json_fenced_response(self) -> None:
        fenced = f"```json\n{_valid_json_response()}\n```"
        mock_llm = _make_mock_llm(fenced)
        extractor = EntityExtractor(llm_provider=mock_llm)

        result = await extractor.extract(_make_ocr())

        assert len(result.artists) == 1
        assert result.artists[0].text == "Carl Cox"

    @pytest.mark.asyncio()
    async def test_plain_fence_response(self) -> None:
        fenced = f"```\n{_valid_json_response()}\n```"
        mock_llm = _make_mock_llm(fenced)
        extractor = EntityExtractor(llm_provider=mock_llm)

        result = await extractor.extract(_make_ocr())

        assert len(result.artists) >= 1

    @pytest.mark.asyncio()
    async def test_invalid_json_retries_then_fails(self) -> None:
        mock_llm = _make_mock_llm("this is not json at all")
        extractor = EntityExtractor(llm_provider=mock_llm)

        with pytest.raises(EntityExtractionError):
            await extractor.extract(_make_ocr())

        # complete() should have been called twice (primary + retry)
        assert mock_llm.complete.call_count == 2

    @pytest.mark.asyncio()
    async def test_retry_succeeds_after_first_failure(self) -> None:
        valid_response = _valid_json_response()
        mock_llm = MagicMock(spec=ILLMProvider)
        mock_llm.get_provider_name.return_value = "mock-llm"
        mock_llm.complete = AsyncMock(
            side_effect=["not json {{{{", valid_response],
        )
        extractor = EntityExtractor(llm_provider=mock_llm)

        result = await extractor.extract(_make_ocr())

        assert len(result.artists) == 1
        assert mock_llm.complete.call_count == 2

    @pytest.mark.asyncio()
    async def test_b2b_split_into_separate_entities(self) -> None:
        response = json.dumps(
            {
                "artists": [
                    {"name": "Carl Cox b2b Adam Beyer", "confidence": 0.9},
                ],
                "venue": None,
                "date": None,
                "promoter": None,
                "genre_tags": [],
                "ticket_price": None,
            }
        )
        mock_llm = _make_mock_llm(response)
        extractor = EntityExtractor(llm_provider=mock_llm)

        result = await extractor.extract(_make_ocr())

        assert len(result.artists) == 2
        names = {a.text for a in result.artists}
        assert "Carl Cox" in names
        assert "Adam Beyer" in names

    @pytest.mark.asyncio()
    async def test_empty_ocr_text_returns_empty_entities(self) -> None:
        mock_llm = _make_mock_llm("")
        extractor = EntityExtractor(llm_provider=mock_llm)
        ocr = _make_ocr(raw_text="")

        result = await extractor.extract(ocr)

        assert len(result.artists) == 0
        assert result.venue is None
        assert result.date is None
        mock_llm.complete.assert_not_called()

    @pytest.mark.asyncio()
    async def test_whitespace_only_ocr_text_returns_empty(self) -> None:
        mock_llm = _make_mock_llm("")
        extractor = EntityExtractor(llm_provider=mock_llm)
        ocr = _make_ocr(raw_text="   \n  \t  ")

        result = await extractor.extract(ocr)

        assert len(result.artists) == 0
        mock_llm.complete.assert_not_called()

    @pytest.mark.asyncio()
    async def test_empty_artist_name_skipped(self) -> None:
        response = json.dumps(
            {
                "artists": [
                    {"name": "", "confidence": 0.9},
                    {"name": "Carl Cox", "confidence": 0.8},
                ],
                "venue": None,
                "date": None,
                "promoter": None,
                "genre_tags": [],
                "ticket_price": None,
            }
        )
        mock_llm = _make_mock_llm(response)
        extractor = EntityExtractor(llm_provider=mock_llm)

        result = await extractor.extract(_make_ocr())

        assert len(result.artists) == 1
        assert result.artists[0].text == "Carl Cox"

    @pytest.mark.asyncio()
    async def test_artist_name_normalization(self) -> None:
        response = json.dumps(
            {
                "artists": [
                    {"name": "DJ Shadow", "confidence": 0.8},
                ],
                "venue": None,
                "date": None,
                "promoter": None,
                "genre_tags": [],
                "ticket_price": None,
            }
        )
        mock_llm = _make_mock_llm(response)
        extractor = EntityExtractor(llm_provider=mock_llm)

        result = await extractor.extract(_make_ocr())

        assert len(result.artists) == 1
        assert result.artists[0].text == "Shadow"

    @pytest.mark.asyncio()
    async def test_confidence_preserved_from_llm_response(self) -> None:
        response = json.dumps(
            {
                "artists": [
                    {"name": "Carl Cox", "confidence": 0.42},
                ],
                "venue": None,
                "date": None,
                "promoter": None,
                "genre_tags": [],
                "ticket_price": None,
            }
        )
        mock_llm = _make_mock_llm(response)
        extractor = EntityExtractor(llm_provider=mock_llm)

        result = await extractor.extract(_make_ocr())

        assert result.artists[0].confidence == pytest.approx(0.42)

    @pytest.mark.asyncio()
    async def test_missing_artists_key_triggers_retry(self) -> None:
        bad_response = json.dumps({"venue": {"name": "X", "confidence": 0.5}})
        good_response = _valid_json_response()
        mock_llm = MagicMock(spec=ILLMProvider)
        mock_llm.get_provider_name.return_value = "mock-llm"
        mock_llm.complete = AsyncMock(side_effect=[bad_response, good_response])
        extractor = EntityExtractor(llm_provider=mock_llm)

        result = await extractor.extract(_make_ocr())

        assert len(result.artists) == 1
        assert mock_llm.complete.call_count == 2

    @pytest.mark.asyncio()
    async def test_response_with_extra_text_around_json(self) -> None:
        wrapped = f"Here is the JSON:\n{_valid_json_response()}\nHope this helps!"
        mock_llm = _make_mock_llm(wrapped)
        extractor = EntityExtractor(llm_provider=mock_llm)

        result = await extractor.extract(_make_ocr())

        assert len(result.artists) == 1

    @pytest.mark.asyncio()
    async def test_null_venue_date_promoter(self) -> None:
        response = json.dumps(
            {
                "artists": [{"name": "A", "confidence": 0.5}],
                "venue": None,
                "date": None,
                "promoter": None,
                "genre_tags": [],
                "ticket_price": None,
            }
        )
        mock_llm = _make_mock_llm(response)
        extractor = EntityExtractor(llm_provider=mock_llm)

        result = await extractor.extract(_make_ocr())

        assert result.venue is None
        assert result.date is None
        assert result.promoter is None
        assert result.ticket_price is None

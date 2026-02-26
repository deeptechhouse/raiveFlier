"""Unit tests for ImageIngester service.

Tests single flier and multi-page scan modes with mocked OCR providers.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from tools.raive_feeder.services.image_ingester import ImageIngester


def _make_mock_ocr(text: str = "Extracted text"):
    """Create a mock OCR provider that returns the given text."""
    provider = AsyncMock()
    result = MagicMock()
    result.text = text
    provider.extract_text = AsyncMock(return_value=result)
    provider.get_provider_name = MagicMock(return_value="mock_ocr")
    return provider


@pytest.fixture
def ingester_with_ocr():
    """ImageIngester with a working mock OCR provider."""
    return ImageIngester(ocr_providers=[_make_mock_ocr("Sample OCR text")])


@pytest.fixture
def ingester_no_ocr():
    """ImageIngester with no OCR providers."""
    return ImageIngester(ocr_providers=[])


@pytest.fixture
def dummy_image():
    """Create a temporary dummy image file."""
    # Minimal 1x1 PNG (43 bytes).
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00"
        b"\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx"
        b"\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00"
        b"\x00IEND\xaeB`\x82"
    )
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(png_bytes)
        return tmp.name


class TestSingleFlierMode:
    """Tests for single image OCR ingestion."""

    @pytest.mark.asyncio
    async def test_ocr_returns_text(self, ingester_with_ocr, dummy_image):
        """Single flier OCR should return extracted text."""
        result = await ingester_with_ocr.ingest_single(
            image_path=dummy_image,
            title="Test Flier",
        )
        assert result["ocr_text"] == "Sample OCR text"

    @pytest.mark.asyncio
    async def test_no_ocr_providers(self, ingester_no_ocr, dummy_image):
        """No OCR providers should return empty text."""
        result = await ingester_no_ocr.ingest_single(
            image_path=dummy_image,
            title="Test",
        )
        assert result["ocr_text"] == ""
        assert result["chunks_created"] == 0


class TestMultiPageMode:
    """Tests for multi-page scan OCR ingestion."""

    @pytest.mark.asyncio
    async def test_multi_page_concatenation(self, dummy_image):
        """Multi-page OCR should concatenate text from all images."""
        call_count = 0

        async def _mock_ocr(image_bytes):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            result.text = f"Page {call_count} text"
            return result

        provider = AsyncMock()
        provider.extract_text = _mock_ocr
        provider.get_provider_name = MagicMock(return_value="mock")

        ingester = ImageIngester(ocr_providers=[provider])
        result = await ingester.ingest_multi_page(
            image_paths=[dummy_image, dummy_image, dummy_image],
            title="Test Multi-Page",
        )

        assert "Page 1 text" in result["ocr_text"]
        assert "Page 2 text" in result["ocr_text"]
        assert "Page 3 text" in result["ocr_text"]


class TestOCRFallback:
    """Tests for OCR provider fallback chain."""

    @pytest.mark.asyncio
    async def test_fallback_to_second_provider(self, dummy_image):
        """If first OCR provider fails, should fall back to second."""
        failing_provider = AsyncMock()
        failing_provider.extract_text = AsyncMock(side_effect=RuntimeError("OCR failed"))
        failing_provider.get_provider_name = MagicMock(return_value="failing")

        working_provider = _make_mock_ocr("Fallback text")

        ingester = ImageIngester(ocr_providers=[failing_provider, working_provider])
        result = await ingester.ingest_single(image_path=dummy_image, title="Test")

        assert result["ocr_text"] == "Fallback text"

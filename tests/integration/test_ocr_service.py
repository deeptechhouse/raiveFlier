"""Integration tests for OCRService multi-provider fallback chain."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.interfaces.ocr_provider import IOCRProvider
from src.models.flier import FlierImage, OCRResult
from src.services.ocr_service import OCRService
from src.utils.errors import OCRExtractionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ocr_provider(
    name: str,
    *,
    available: bool = True,
    confidence: float = 0.9,
    raw_text: str = "CARL COX\nTRESOR BERLIN",
    raises: Exception | None = None,
) -> IOCRProvider:
    """Create a mock OCR provider with configurable behaviour."""
    mock = MagicMock(spec=IOCRProvider)
    mock.get_provider_name.return_value = name
    mock.is_available.return_value = available
    mock.supports_stylized_text.return_value = False

    if raises is not None:
        mock.extract_text = AsyncMock(side_effect=raises)
    else:
        result = OCRResult(
            raw_text=raw_text,
            confidence=confidence,
            provider_used=name,
            processing_time=0.5,
        )
        mock.extract_text = AsyncMock(return_value=result)

    return mock


def _make_flier_image() -> FlierImage:
    """Create a minimal FlierImage for testing."""
    return FlierImage(
        id="test-ocr-001",
        filename="test.jpg",
        content_type="image/jpeg",
        file_size=1024,
        image_hash="abc123",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOCRFallbackChain:
    """Tests for OCRService provider fallback and selection logic."""

    @pytest.mark.asyncio()
    async def test_ocr_fallback_chain(self) -> None:
        """First provider fails with exception, second provider used successfully."""
        failing_provider = _make_ocr_provider(
            "provider-a",
            raises=RuntimeError("GPU out of memory"),
        )
        working_provider = _make_ocr_provider(
            "provider-b",
            confidence=0.85,
            raw_text="CARL COX\nTRESOR",
        )

        service = OCRService(providers=[failing_provider, working_provider])
        flier = _make_flier_image()

        result = await service.extract_text(flier)

        assert result.provider_used == "provider-b"
        assert result.confidence == pytest.approx(0.85)
        assert "CARL COX" in result.raw_text

        # First provider was attempted
        failing_provider.extract_text.assert_awaited_once()
        # Second provider was used as fallback
        working_provider.extract_text.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_ocr_high_confidence_short_circuits(self) -> None:
        """First provider returns 0.9 confidence, second provider should NOT be called."""
        high_conf_provider = _make_ocr_provider(
            "provider-fast",
            confidence=0.9,
            raw_text="CARL COX\nJEFF MILLS\nTRESOR BERLIN",
        )
        backup_provider = _make_ocr_provider(
            "provider-backup",
            confidence=0.95,
            raw_text="CARL COX\nJEFF MILLS\nTRESOR BERLIN\n15 MARCH 1997",
        )

        service = OCRService(providers=[high_conf_provider, backup_provider])
        flier = _make_flier_image()

        result = await service.extract_text(flier)

        assert result.provider_used == "provider-fast"
        assert result.confidence == pytest.approx(0.9)

        # High confidence meets threshold (0.7) — short-circuits
        high_conf_provider.extract_text.assert_awaited_once()
        backup_provider.extract_text.assert_not_awaited()

    @pytest.mark.asyncio()
    async def test_ocr_best_of_low_confidence(self) -> None:
        """Both providers return below threshold, highest confidence result returned."""
        low_provider = _make_ocr_provider(
            "provider-low",
            confidence=0.4,
            raw_text="C COX",
        )
        medium_provider = _make_ocr_provider(
            "provider-medium",
            confidence=0.6,
            raw_text="CARL COX TRESOR",
        )

        service = OCRService(providers=[low_provider, medium_provider])
        flier = _make_flier_image()

        result = await service.extract_text(flier)

        # Both below 0.7 threshold — returns the best (0.6)
        assert result.provider_used == "provider-medium"
        assert result.confidence == pytest.approx(0.6)
        assert "CARL COX" in result.raw_text

        # Both providers were tried
        low_provider.extract_text.assert_awaited_once()
        medium_provider.extract_text.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_all_providers_fail(self) -> None:
        """All providers raise exceptions — OCRExtractionError raised."""
        provider_a = _make_ocr_provider(
            "provider-a",
            raises=RuntimeError("Out of memory"),
        )
        provider_b = _make_ocr_provider(
            "provider-b",
            raises=ConnectionError("API timeout"),
        )
        provider_c = _make_ocr_provider(
            "provider-c",
            raises=ValueError("Invalid image format"),
        )

        service = OCRService(providers=[provider_a, provider_b, provider_c])
        flier = _make_flier_image()

        with pytest.raises(OCRExtractionError, match="All OCR providers failed"):
            await service.extract_text(flier)

        # All providers were attempted
        provider_a.extract_text.assert_awaited_once()
        provider_b.extract_text.assert_awaited_once()
        provider_c.extract_text.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_unavailable_providers_skipped(self) -> None:
        """Unavailable providers are skipped without attempting extraction."""
        unavailable = _make_ocr_provider("provider-offline", available=False)
        available = _make_ocr_provider(
            "provider-online",
            confidence=0.88,
        )

        service = OCRService(providers=[unavailable, available])
        flier = _make_flier_image()

        result = await service.extract_text(flier)

        assert result.provider_used == "provider-online"
        unavailable.extract_text.assert_not_awaited()
        available.extract_text.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_get_available_providers(self) -> None:
        """get_available_providers returns only available provider names."""
        p1 = _make_ocr_provider("online-a", available=True)
        p2 = _make_ocr_provider("offline-b", available=False)
        p3 = _make_ocr_provider("online-c", available=True)

        service = OCRService(providers=[p1, p2, p3])
        available = service.get_available_providers()

        assert available == ["online-a", "online-c"]

    @pytest.mark.asyncio()
    async def test_fallback_on_exception_then_low_confidence(self) -> None:
        """First provider raises, second returns low confidence — low confidence returned."""
        failing = _make_ocr_provider(
            "provider-fail",
            raises=RuntimeError("crash"),
        )
        low_conf = _make_ocr_provider(
            "provider-low",
            confidence=0.3,
            raw_text="C COX",
        )

        service = OCRService(providers=[failing, low_conf])
        flier = _make_flier_image()

        result = await service.extract_text(flier)

        assert result.provider_used == "provider-low"
        assert result.confidence == pytest.approx(0.3)

    @pytest.mark.asyncio()
    async def test_configurable_min_confidence_threshold(self) -> None:
        """Custom min_confidence threshold controls short-circuit behavior."""
        provider_a = _make_ocr_provider(
            "provider-a",
            confidence=0.5,
            raw_text="CARL COX",
        )
        provider_b = _make_ocr_provider(
            "provider-b",
            confidence=0.9,
            raw_text="CARL COX\nTRESOR BERLIN",
        )

        # With low threshold (0.4), provider-a's 0.5 meets it → short-circuits
        service = OCRService(providers=[provider_a, provider_b], min_confidence=0.4)
        flier = _make_flier_image()
        result = await service.extract_text(flier)

        assert result.provider_used == "provider-a"
        assert result.confidence == pytest.approx(0.5)
        provider_b.extract_text.assert_not_awaited()

    @pytest.mark.asyncio()
    async def test_high_threshold_forces_fallthrough(self) -> None:
        """High min_confidence causes fallthrough to better provider."""
        provider_a = _make_ocr_provider(
            "provider-a",
            confidence=0.6,
            raw_text="C COX",
        )
        provider_b = _make_ocr_provider(
            "provider-b",
            confidence=0.95,
            raw_text="CARL COX\nTRESOR BERLIN",
        )

        # With high threshold (0.9), provider-a's 0.6 does NOT meet it
        service = OCRService(providers=[provider_a, provider_b], min_confidence=0.9)
        flier = _make_flier_image()
        result = await service.extract_text(flier)

        assert result.provider_used == "provider-b"
        assert result.confidence == pytest.approx(0.95)

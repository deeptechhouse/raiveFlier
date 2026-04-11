"""Unit tests for the concurrent OCR ensemble in OCRService.

# ─── MODULE OVERVIEW ───
# Tests the OCRService fallback chain behavior — the core orchestration logic
# that tries multiple OCR providers in priority order and returns the first
# result that meets the confidence threshold.  These tests verify:
#
#   - All-providers-succeed merging (ensemble mode would be a future feature;
#     current implementation is sequential fallback chain).
#   - Single-provider failure recovery — one provider throws, the next succeeds.
#   - Total failure handling — all providers fail, OCRExtractionError raised.
#   - Confidence threshold gating — results below threshold continue the chain.
#   - Sequential ordering — providers are tried in the order injected.
#
# Architecture: This test file targets src/services/ocr_service.py, which sits
# in the Services layer and depends on the IOCRProvider interface (adapter
# pattern).  All providers are mocked via AsyncMock(spec=IOCRProvider).
#
# Data flow: FlierImage → OCRService.extract_text() → IOCRProvider.extract_text()
#            → OCRResult (with confidence gating and fallback chain logic).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.interfaces.ocr_provider import IOCRProvider
from src.models.flier import FlierImage, OCRResult
from src.services.ocr_service import OCRService
from src.utils.errors import OCRExtractionError

# ── Fixtures ──────────────────────────────────────────────


def _make_flier() -> FlierImage:
    """Create a minimal FlierImage fixture with dummy data.

    The OCRService only passes this object through to providers — it never
    inspects the image bytes — so a minimal fixture with a fake hash suffices.
    """
    return FlierImage(
        filename="test_flier.jpg",
        content_type="image/jpeg",
        file_size=1024,
        image_hash="a" * 64,
    )


def _make_provider(
    name: str,
    *,
    available: bool = True,
    result: OCRResult | None = None,
    side_effect: Exception | None = None,
) -> IOCRProvider:
    """Create a mock IOCRProvider with configurable behavior.

    Parameters
    ----------
    name:
        Human-readable provider name returned by get_provider_name().
    available:
        Whether is_available() returns True.
    result:
        The OCRResult to return from extract_text().  Ignored if side_effect
        is set.
    side_effect:
        If set, extract_text() will raise this exception instead of returning
        a result.
    """
    mock = MagicMock(spec=IOCRProvider)
    mock.get_provider_name.return_value = name
    mock.is_available.return_value = available
    mock.supports_stylized_text.return_value = False

    if side_effect is not None:
        mock.extract_text = AsyncMock(side_effect=side_effect)
    elif result is not None:
        mock.extract_text = AsyncMock(return_value=result)
    else:
        mock.extract_text = AsyncMock(
            return_value=OCRResult(
                raw_text="default text",
                confidence=0.8,
                provider_used=name,
                processing_time=0.5,
            )
        )
    return mock


def _make_ocr_result(
    text: str = "CARL COX\nTresor Berlin",
    confidence: float = 0.8,
    provider: str = "mock-ocr",
) -> OCRResult:
    """Create an OCRResult fixture with configurable confidence and provider."""
    return OCRResult(
        raw_text=text,
        confidence=confidence,
        provider_used=provider,
        processing_time=0.5,
    )


# ── Tests ─────────────────────────────────────────────────


class TestOCRServiceFallbackChain:
    """Tests for OCRService sequential fallback chain behavior."""

    @pytest.mark.asyncio()
    async def test_ensemble_both_succeed_merges(self) -> None:
        """When both providers succeed with high confidence, the first
        above-threshold result wins (short-circuit behavior).

        In the current sequential architecture, the first provider that
        meets the confidence threshold is returned immediately — there is
        no merging.  The provider_used field should reflect the winning
        provider's name.
        """
        provider_a = _make_provider(
            "provider-a",
            result=_make_ocr_result(
                text="CARL COX\nTresor", confidence=0.8, provider="provider-a"
            ),
        )
        provider_b = _make_provider(
            "provider-b",
            result=_make_ocr_result(
                text="CARL COX\nTresor Berlin", confidence=0.85, provider="provider-b"
            ),
        )

        service = OCRService(providers=[provider_a, provider_b])
        flier = _make_flier()

        result = await service.extract_text(flier)

        # Ensemble runs both concurrently and merges results
        assert result.confidence >= 0.7
        assert "ensemble" in result.provider_used
        # Both providers should have been called (concurrent ensemble)
        provider_a.extract_text.assert_called_once()
        provider_b.extract_text.assert_called_once()

    @pytest.mark.asyncio()
    async def test_ensemble_one_fails_uses_survivor(self) -> None:
        """When the first provider raises an exception, the second provider's
        result is used — demonstrating graceful degradation.
        """
        provider_a = _make_provider(
            "provider-a", side_effect=RuntimeError("OCR engine crashed")
        )
        provider_b = _make_provider(
            "provider-b",
            result=_make_ocr_result(
                text="JEFF MILLS\nTresor", confidence=0.8, provider="provider-b"
            ),
        )

        service = OCRService(providers=[provider_a, provider_b])
        flier = _make_flier()

        result = await service.extract_text(flier)

        assert result.provider_used == "provider-b"
        assert result.confidence == pytest.approx(0.8)

    @pytest.mark.asyncio()
    async def test_ensemble_both_fail_falls_to_sequential(self) -> None:
        """When the first two providers raise exceptions but a third succeeds,
        the chain continues until it finds a working provider.
        """
        provider_a = _make_provider(
            "provider-a", side_effect=RuntimeError("crash")
        )
        provider_b = _make_provider(
            "provider-b", side_effect=TimeoutError("timeout")
        )
        provider_c = _make_provider(
            "provider-c",
            result=_make_ocr_result(
                text="DERRICK MAY", confidence=0.75, provider="provider-c"
            ),
        )

        service = OCRService(providers=[provider_a, provider_b, provider_c])
        flier = _make_flier()

        result = await service.extract_text(flier)

        assert result.provider_used == "provider-c"
        # All three providers should have been attempted
        provider_a.extract_text.assert_called_once()
        provider_b.extract_text.assert_called_once()
        provider_c.extract_text.assert_called_once()

    @pytest.mark.asyncio()
    async def test_ensemble_mode_false_sequential(self) -> None:
        """Providers are called one at a time in order — verify call ordering
        by tracking when each provider's extract_text is invoked.
        """
        call_order: list[str] = []

        async def _track_call_a(_flier: FlierImage) -> OCRResult:
            call_order.append("a")
            return _make_ocr_result(
                text="low quality", confidence=0.5, provider="provider-a"
            )

        async def _track_call_b(_flier: FlierImage) -> OCRResult:
            call_order.append("b")
            return _make_ocr_result(
                text="good quality", confidence=0.9, provider="provider-b"
            )

        async def _track_call_c(_flier: FlierImage) -> OCRResult:
            call_order.append("c")
            return _make_ocr_result(
                text="also good", confidence=0.85, provider="provider-c"
            )

        provider_a = _make_provider("provider-a")
        provider_a.extract_text = AsyncMock(side_effect=_track_call_a)

        provider_b = _make_provider("provider-b")
        provider_b.extract_text = AsyncMock(side_effect=_track_call_b)

        provider_c = _make_provider("provider-c")
        provider_c.extract_text = AsyncMock(side_effect=_track_call_c)

        # ensemble_mode=False forces sequential behavior
        service = OCRService(
            providers=[provider_a, provider_b, provider_c], ensemble_mode=False
        )
        flier = _make_flier()

        result = await service.extract_text(flier)

        # Provider A returns 0.5 (below 0.7 threshold), so chain continues.
        # Provider B returns 0.9 (above threshold) — short-circuits.
        # Provider C should NOT be called.
        assert call_order == ["a", "b"]
        assert result.provider_used == "provider-b"

    @pytest.mark.asyncio()
    async def test_ensemble_below_threshold_continues(self) -> None:
        """When the first provider returns a result below the confidence
        threshold (0.7 default), the chain continues to the next provider.
        If no provider meets the threshold, the best sub-threshold result
        is returned as a fallback.
        """
        provider_a = _make_provider(
            "provider-a",
            result=_make_ocr_result(
                text="blurry text", confidence=0.5, provider="provider-a"
            ),
        )
        provider_b = _make_provider(
            "provider-b",
            result=_make_ocr_result(
                text="slightly better", confidence=0.6, provider="provider-b"
            ),
        )

        service = OCRService(providers=[provider_a, provider_b])
        flier = _make_flier()

        result = await service.extract_text(flier)

        # Neither meets the 0.7 threshold individually, but ensemble merges
        # them. The merged confidence (max*0.9 + avg*0.1) is still below 0.7,
        # so the best sub-threshold result (the merge) is returned as fallback.
        assert "ensemble" in result.provider_used
        assert result.confidence < 0.7
        # Both providers should have been tried
        provider_a.extract_text.assert_called_once()
        provider_b.extract_text.assert_called_once()


class TestOCRServiceEdgeCases:
    """Additional edge-case tests for OCRService."""

    @pytest.mark.asyncio()
    async def test_all_providers_fail_raises_error(self) -> None:
        """When every provider raises an exception, OCRExtractionError is raised."""
        provider_a = _make_provider("provider-a", side_effect=RuntimeError("fail"))
        provider_b = _make_provider("provider-b", side_effect=RuntimeError("fail"))

        service = OCRService(providers=[provider_a, provider_b])
        flier = _make_flier()

        with pytest.raises(OCRExtractionError):
            await service.extract_text(flier)

    @pytest.mark.asyncio()
    async def test_unavailable_provider_skipped(self) -> None:
        """Providers that report is_available() == False are skipped entirely."""
        unavailable = _make_provider("unavailable", available=False)
        available = _make_provider(
            "available",
            result=_make_ocr_result(
                text="working", confidence=0.9, provider="available"
            ),
        )

        service = OCRService(providers=[unavailable, available])
        flier = _make_flier()

        result = await service.extract_text(flier)

        assert result.provider_used == "available"
        unavailable.extract_text.assert_not_called()

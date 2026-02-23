"""OCR orchestration service with multi-provider fallback chain.

Manages a priority-ordered list of OCR providers and tries each in turn
until one returns a result with acceptable confidence.  The default
provider order is: ``llm_vision`` → ``easyocr`` → ``tesseract``.

Architecture: Fallback Chain Pattern
-------------------------------------
This service implements a **Chain of Responsibility** variant tailored
for quality-scored results.  Rather than the first handler that *can*
respond winning outright, each provider produces a result with an
associated confidence score.  The chain short-circuits as soon as a
provider meets the confidence threshold, but tracks the best
sub-threshold result as insurance.  This means:

    1. Fast, high-quality providers are tried first (LLM vision).
    2. Cheaper/local providers are tried next (EasyOCR, Tesseract).
    3. The caller always gets *something* unless every provider hard-fails.

All providers implement ``IOCRProvider``, so new backends (e.g. a future
Google Document AI adapter) can be injected without modifying this file.
"""

from __future__ import annotations

from src.interfaces.ocr_provider import IOCRProvider
from src.models.flier import FlierImage, OCRResult
from src.utils.errors import OCRExtractionError
from src.utils.logging import get_logger

# Confidence threshold below which a result is treated as "low quality" and
# the chain continues trying the next provider.  0.7 was tuned against a
# test corpus of ~200 rave fliers with varying image quality.
_DEFAULT_CONFIDENCE_THRESHOLD = 0.7


class OCRService:
    """Orchestrates OCR extraction across multiple providers.

    Providers are tried in the order supplied at construction time.  The
    first result that meets the confidence threshold is returned immediately.
    If no provider reaches the threshold, the best result seen so far is
    returned.  If every provider fails with an exception, an
    :class:`OCRExtractionError` is raised.
    """

    def __init__(
        self,
        providers: list[IOCRProvider],
        min_confidence: float = _DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        # Providers are stored in priority order.  The caller (typically the
        # DI container in app startup) controls the ordering, so this class
        # stays agnostic about which backends exist.
        self._providers = providers
        self._min_confidence = min_confidence
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def extract_text(self, flier: FlierImage) -> OCRResult:
        """Run OCR on *flier* using the provider fallback chain.

        Parameters
        ----------
        flier:
            The flier image to process.

        Returns
        -------
        OCRResult
            The best extraction result obtained from any provider.

        Raises
        ------
        OCRExtractionError
            If every provider either is unavailable or raises an exception.
        """
        # Tracks the highest-confidence result seen across providers that
        # did NOT meet the threshold.  Acts as the "safety net" -- we would
        # rather return a low-confidence extraction than nothing at all,
        # because downstream stages (entity extraction, confirmation gate)
        # can still recover or ask the user to verify.
        best_result: OCRResult | None = None

        # --- Provider iteration: try each in priority order ---
        for provider in self._providers:
            name = provider.get_provider_name()

            # Availability check lets providers self-report runtime issues
            # (missing API key, binary not installed, etc.) without raising.
            if not provider.is_available():
                self._logger.warning("ocr_provider_unavailable", provider=name)
                continue

            try:
                self._logger.info("ocr_provider_attempting", provider=name)
                result = await provider.extract_text(flier)

                # --- Confidence gate: early-return on first "good enough" result ---
                # This short-circuits the chain, avoiding unnecessary (and
                # potentially slow/expensive) calls to lower-priority providers.
                if result.confidence >= self._min_confidence:
                    self._logger.info(
                        "ocr_provider_accepted",
                        provider=name,
                        confidence=round(result.confidence, 4),
                    )
                    return result

                # Below threshold -- hold onto this result if it is the best
                # sub-threshold result we have seen so far.
                if best_result is None or result.confidence > best_result.confidence:
                    best_result = result
                    self._logger.info(
                        "ocr_provider_below_threshold",
                        provider=name,
                        confidence=round(result.confidence, 4),
                    )

            except Exception as exc:
                # Individual provider failures are non-fatal.  The chain
                # continues with the next provider.  Only if *all* providers
                # fail (no best_result at all) do we raise.
                self._logger.warning(
                    "ocr_provider_failed",
                    provider=name,
                    error=str(exc),
                )

        # --- Graceful degradation ---
        # If we reach this point, no provider met the confidence threshold.
        # Returning the best sub-threshold result lets downstream stages
        # (entity extraction, user confirmation gate) decide whether the
        # quality is acceptable rather than hard-failing here.
        if best_result is not None:
            self._logger.info(
                "ocr_returning_best_fallback",
                provider=best_result.provider_used,
                confidence=round(best_result.confidence, 4),
            )
            return best_result

        # Total failure: every provider was unavailable or threw an exception.
        raise OCRExtractionError("All OCR providers failed")

    def get_available_providers(self) -> list[str]:
        """Return the names of providers that are currently available."""
        return [p.get_provider_name() for p in self._providers if p.is_available()]

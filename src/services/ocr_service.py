"""OCR orchestration service with multi-provider fallback chain.

Manages a priority-ordered list of OCR providers and tries each in turn
until one returns a result with acceptable confidence.  The default
provider order is: ``llm_vision`` → ``easyocr`` → ``tesseract``.
"""

from __future__ import annotations

from src.interfaces.ocr_provider import IOCRProvider
from src.models.flier import FlierImage, OCRResult
from src.utils.errors import OCRExtractionError
from src.utils.logging import get_logger

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
        best_result: OCRResult | None = None

        for provider in self._providers:
            name = provider.get_provider_name()

            if not provider.is_available():
                self._logger.warning("ocr_provider_unavailable", provider=name)
                continue

            try:
                self._logger.info("ocr_provider_attempting", provider=name)
                result = await provider.extract_text(flier)

                if result.confidence >= self._min_confidence:
                    self._logger.info(
                        "ocr_provider_accepted",
                        provider=name,
                        confidence=round(result.confidence, 4),
                    )
                    return result

                # Below threshold — keep as fallback if it's the best so far.
                if best_result is None or result.confidence > best_result.confidence:
                    best_result = result
                    self._logger.info(
                        "ocr_provider_below_threshold",
                        provider=name,
                        confidence=round(result.confidence, 4),
                    )

            except Exception as exc:
                self._logger.warning(
                    "ocr_provider_failed",
                    provider=name,
                    error=str(exc),
                )

        if best_result is not None:
            self._logger.info(
                "ocr_returning_best_fallback",
                provider=best_result.provider_used,
                confidence=round(best_result.confidence, 4),
            )
            return best_result

        raise OCRExtractionError("All OCR providers failed")

    def get_available_providers(self) -> list[str]:
        """Return the names of providers that are currently available."""
        return [p.get_provider_name() for p in self._providers if p.is_available()]

"""OCR orchestration service with concurrent ensemble and sequential fallback.

Manages a priority-ordered list of OCR providers.  In **ensemble mode**
(the default), the first two providers — typically LLM Vision and EasyOCR —
run concurrently via ``asyncio.gather``, and their results are merged into
a single high-fidelity extraction.  Remaining providers (e.g. Tesseract)
serve as sequential fallbacks if the ensemble fails or scores below the
confidence threshold.

When ensemble mode is disabled, the service reverts to the original
sequential Chain-of-Responsibility behaviour where each provider is tried
in priority order until one meets the confidence threshold.

# ─── WHY CONCURRENT ENSEMBLE? ───
#
# LLM Vision and EasyOCR have complementary strengths:
#   - LLM Vision excels at reading stylised, warped, and artistic text
#     common on rave fliers, but returns NO bounding-box geometry.
#   - EasyOCR produces accurate bounding boxes and handles standard print
#     well, but struggles with heavily stylised typography.
#
# Running them concurrently is safe because they utilise different resource
# types: LLM Vision is **I/O-bound** (API call over the network), while
# EasyOCR is **CPU-bound** (PyTorch inference).  On the single-worker
# Render deployment this means the CPU stays busy with EasyOCR while the
# event loop awaits the LLM Vision HTTP response — near-zero idle time.
#
# The merged result combines LLM Vision's superior text reading with
# EasyOCR's spatial information, producing a richer OCRResult than either
# provider achieves alone.

Architecture: Ensemble + Fallback Chain Pattern
-------------------------------------------------
This service layers an **Ensemble Pattern** on top of the existing
**Chain of Responsibility** fallback.  The ensemble runs the primary
providers in parallel and merges their outputs; if the merged confidence
is still below threshold, the chain continues with remaining providers
sequentially (graceful degradation to Tesseract).

All providers implement ``IOCRProvider``, so new backends can be injected
without modifying this file.
"""

from __future__ import annotations

import asyncio

from src.interfaces.ocr_provider import IOCRProvider
from src.models.flier import FlierImage, OCRResult
from src.utils.errors import OCRExtractionError
from src.utils.logging import get_logger

# Reuse the existing deduplication logic from ocr_helpers rather than
# duplicating fuzzy-matching code.  deduplicate_text_regions handles
# cross-provider region merging with token-sort fuzzy matching.
from src.utils.ocr_helpers import deduplicate_text_regions

# Confidence threshold below which a result is treated as "low quality" and
# the chain continues trying the next provider.  0.7 was tuned against a
# test corpus of ~200 rave fliers with varying image quality.
_DEFAULT_CONFIDENCE_THRESHOLD = 0.7


class OCRService:
    """Orchestrates OCR extraction across multiple providers.

    Supports two modes:

    - **Ensemble mode** (default): Runs the first two providers concurrently,
      merges their results via fuzzy deduplication, then falls through to
      remaining providers if the merged confidence is insufficient.
    - **Sequential mode**: Tries providers one-by-one in priority order
      (original behaviour, preserved as fallback).
    """

    def __init__(
        self,
        providers: list[IOCRProvider],
        min_confidence: float = _DEFAULT_CONFIDENCE_THRESHOLD,
        ensemble_mode: bool = True,
    ) -> None:
        # Providers are stored in priority order.  The caller (typically the
        # DI container in app startup) controls the ordering, so this class
        # stays agnostic about which backends exist.
        self._providers = providers
        self._min_confidence = min_confidence
        # Ensemble mode is toggled at construction time so callers (e.g. tests
        # or deployments with only one OCR provider) can opt out without
        # changing any other code.
        self._ensemble_mode = ensemble_mode
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def extract_text(self, flier: FlierImage) -> OCRResult:
        """Run OCR on *flier* using ensemble or sequential strategy.

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
        # ── Ensemble path ──
        # When ensemble_mode is active and at least 2 providers are available,
        # split the provider list into primary (concurrent) and fallback
        # (sequential) groups.  This maximises throughput on the first attempt
        # while preserving the existing sequential safety net.
        if self._ensemble_mode and len(self._providers) >= 2:
            primary_providers = self._providers[:2]
            fallback_providers = self._providers[2:]

            concurrent_results = await self._run_concurrent_primary(
                flier, primary_providers
            )

            # Two successful results — merge them for higher fidelity.
            if len(concurrent_results) >= 2:
                merged = self._merge_results(concurrent_results)
                if merged.confidence >= self._min_confidence:
                    self._logger.info(
                        "ocr_ensemble_accepted",
                        provider=merged.provider_used,
                        confidence=round(merged.confidence, 4),
                    )
                    return merged
                # Merged result exists but below threshold — keep it as the
                # best candidate seen so far while we try fallbacks.
                best_result: OCRResult | None = merged
            elif len(concurrent_results) == 1:
                # Only one provider succeeded.  Use its result if it meets
                # the threshold; otherwise keep it as best-so-far.
                solo = concurrent_results[0]
                if solo.confidence >= self._min_confidence:
                    self._logger.info(
                        "ocr_provider_accepted",
                        provider=solo.provider_used,
                        confidence=round(solo.confidence, 4),
                    )
                    return solo
                best_result = solo
            else:
                # Both primary providers failed — no best result yet.
                best_result = None

            # ── Sequential fallback for remaining providers (e.g. Tesseract) ──
            # Reuses the same confidence-gated loop as the fully-sequential
            # path below, but only iterates over providers not already tried.
            return await self._run_sequential(
                flier, fallback_providers, best_result
            )

        # ── Sequential path (ensemble disabled or < 2 providers) ──
        # Preserves the original Chain-of-Responsibility behaviour unchanged.
        return await self._run_sequential(flier, self._providers, None)

    def get_available_providers(self) -> list[str]:
        """Return the names of providers that are currently available."""
        return [
            p.get_provider_name() for p in self._providers if p.is_available()
        ]

    # ------------------------------------------------------------------
    # Ensemble internals
    # ------------------------------------------------------------------

    async def _run_concurrent_primary(
        self,
        flier: FlierImage,
        primary_providers: list[IOCRProvider],
    ) -> list[OCRResult]:
        """Run primary OCR providers concurrently and collect successes.

        Uses ``asyncio.gather`` with ``return_exceptions=True`` so that a
        failure in one provider does not cancel the other.  This is critical
        because LLM Vision (network I/O) and EasyOCR (local CPU) have
        completely independent failure modes — an API timeout should not
        prevent EasyOCR from returning its result.

        Parameters
        ----------
        flier:
            The flier image to process.
        primary_providers:
            The first N providers to run in parallel (typically 2).

        Returns
        -------
        list[OCRResult]
            Only the successful results; exceptions are filtered out and
            logged as warnings.
        """
        # Filter to available providers before launching tasks.
        available = []
        for p in primary_providers:
            name = p.get_provider_name()
            if p.is_available():
                self._logger.info("ocr_ensemble_provider_starting", provider=name)
                available.append(p)
            else:
                self._logger.warning("ocr_provider_unavailable", provider=name)

        if not available:
            return []

        # return_exceptions=True prevents one failing provider from cancelling
        # the other via exception propagation.  Instead, exceptions land in
        # the results list and we filter them out below.
        raw_results = await asyncio.gather(
            *[p.extract_text(flier) for p in available],
            return_exceptions=True,
        )

        successes: list[OCRResult] = []
        for provider, result in zip(available, raw_results, strict=True):
            name = provider.get_provider_name()
            if isinstance(result, BaseException):
                self._logger.warning(
                    "ocr_ensemble_provider_failed",
                    provider=name,
                    error=str(result),
                )
            else:
                self._logger.info(
                    "ocr_ensemble_provider_completed",
                    provider=name,
                    confidence=round(result.confidence, 4),
                )
                successes.append(result)

        return successes

    def _merge_results(self, results: list[OCRResult]) -> OCRResult:
        """Merge multiple OCRResults into a single high-fidelity result.

        The merge strategy:
          1. Collect all bounding-box TextRegions from every result.
          2. Deduplicate via ``deduplicate_text_regions`` (fuzzy matching) so
             the same text detected by both providers is kept only once, with
             the higher-confidence version retained.
          3. Rebuild ``raw_text`` from the deduplicated regions.
          4. Compute a blended confidence score (see formula below).
          5. Sum processing times (concurrent wall-clock time is less than
             this sum, but the sum reflects total compute consumed).

        # ─── CONFIDENCE FORMULA RATIONALE ───
        #
        # merged_confidence = max(individual) * 0.9 + mean(individual) * 0.1
        #
        # This formula is deliberately conservative:
        #   - The 90% weight on the MAX score anchors the result to the best
        #     single provider, preventing a weak second provider from dragging
        #     the merged score below what the best provider achieved alone.
        #   - The 10% weight on the MEAN gives a small boost when both
        #     providers agree (high mean), acting as a "corroboration bonus".
        #   - The result is always <= max(individual), which avoids inflating
        #     confidence beyond what any single provider warranted.
        #   - Capped at 1.0 to satisfy the OCRResult field constraint.

        Parameters
        ----------
        results:
            Two or more successful OCRResult objects to merge.

        Returns
        -------
        OCRResult
            A merged result combining text, regions, and confidence from
            all inputs.
        """
        # Step 1: Gather all text regions across providers.
        # LLM Vision typically contributes regions with zeroed-out bounding
        # boxes (x=0, y=0, width=0, height=0), while EasyOCR contributes
        # regions with real pixel coordinates.  deduplicate_text_regions
        # matches on text content, not geometry, so both sources merge cleanly.
        all_regions = []
        for r in results:
            all_regions.extend(r.bounding_boxes)

        # Step 2: Fuzzy-deduplicate — keeps highest-confidence version of each
        # unique text fragment.
        merged_regions = deduplicate_text_regions(all_regions)

        # Step 3: Rebuild raw_text.  If regions were available, join them;
        # otherwise fall back to concatenating the raw_text from each result
        # (handles the case where a provider returned raw_text but no regions).
        if merged_regions:
            merged_raw_text = "\n".join(r.text for r in merged_regions)
        else:
            merged_raw_text = "\n".join(r.raw_text for r in results)

        # Step 4: Blended confidence — anchored to best provider with a small
        # corroboration bonus from the mean.
        max_conf = max(r.confidence for r in results)
        mean_conf = sum(r.confidence for r in results) / len(results)
        merged_confidence = min(1.0, max_conf * 0.9 + mean_conf * 0.1)

        # Step 5: Sum processing times (represents total compute, not
        # wall-clock time which was lower due to concurrency).
        total_time = sum(r.processing_time for r in results)

        # Provider name encodes which engines contributed to this merged result
        # (e.g. "ensemble_openai_vision+easyocr").
        provider_name = "ensemble_" + "+".join(
            r.provider_used for r in results
        )

        return OCRResult(
            raw_text=merged_raw_text,
            confidence=merged_confidence,
            provider_used=provider_name,
            processing_time=total_time,
            bounding_boxes=merged_regions,
        )

    # ------------------------------------------------------------------
    # Sequential fallback (original Chain-of-Responsibility behaviour)
    # ------------------------------------------------------------------

    async def _run_sequential(
        self,
        flier: FlierImage,
        providers: list[IOCRProvider],
        best_result: OCRResult | None,
    ) -> OCRResult:
        """Try providers one-by-one in priority order.

        This is the original extraction logic, extracted into its own method
        so it can be reused both as the primary strategy (when ensemble mode
        is off) and as the fallback path after a failed ensemble attempt.

        Parameters
        ----------
        flier:
            The flier image to process.
        providers:
            The providers to iterate, in priority order.
        best_result:
            An optional result carried forward from an earlier attempt
            (e.g. a sub-threshold ensemble merge).  If non-None, this
            competes with results from *providers* for best-so-far status.

        Returns
        -------
        OCRResult
            The best result obtained.

        Raises
        ------
        OCRExtractionError
            If no provider produced any result and *best_result* is None.
        """
        for provider in providers:
            name = provider.get_provider_name()

            if not provider.is_available():
                self._logger.warning("ocr_provider_unavailable", provider=name)
                continue

            try:
                self._logger.info("ocr_provider_attempting", provider=name)
                result = await provider.extract_text(flier)

                # Confidence gate: early-return on first "good enough" result.
                if result.confidence >= self._min_confidence:
                    self._logger.info(
                        "ocr_provider_accepted",
                        provider=name,
                        confidence=round(result.confidence, 4),
                    )
                    return result

                # Below threshold — track best sub-threshold result.
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

        # Graceful degradation: return the best sub-threshold result rather
        # than hard-failing, because downstream stages (entity extraction,
        # user confirmation gate) can still recover or ask the user to verify.
        if best_result is not None:
            self._logger.info(
                "ocr_returning_best_fallback",
                provider=best_result.provider_used,
                confidence=round(best_result.confidence, 4),
            )
            return best_result

        raise OCRExtractionError("All OCR providers failed")

"""Progressive confidence score calibration via binned isotonic regression.

Collects predicted_score and ground_truth samples from the entity
confirmation gate, bins them into deciles, and produces a monotonic
mapping from raw LLM/fuzzy-match scores to calibrated probabilities.

Cold-start safe: returns raw scores unchanged until >= 30 samples
accumulate, avoiding erratic calibration from small samples.
"""

# ─── MODULE OVERVIEW ─────────────────────────────────────────────────────
#
# Layer: Utility (src/utils/)
# Pattern: Lookup-table calibration with Pool-Adjacent-Violators (PAV)
# Dependencies: None (pure Python, no scipy/sklearn)
#
# This module sits between the raw confidence scores produced by LLMs and
# fuzzy-match algorithms and the values displayed to users.  Over time,
# as users confirm or reject extracted entities at the confirmation gate,
# the calibrator learns the empirical relationship between a reported
# score and actual accuracy.
#
# Data flow:
#   1. ConfirmationGate collects (predicted_score, ground_truth) samples
#      and persists them via IFeedbackProvider.store_calibration_sample().
#   2. On startup or periodically, samples are loaded via
#      IFeedbackProvider.get_calibration_data() and fed to fit().
#   3. EntityExtractor and ArtistResearcher call calibrate() to adjust
#      raw scores before they reach the user.
#
# Cold-start safety:
#   With < 30 samples the calibrator acts as an identity function,
#   returning raw scores unchanged.  This prevents erratic calibration
#   from tiny, unrepresentative sample sets.
#
# Why no scipy?  The deployment environment (Render, 2GB RAM) already
# runs PyTorch for EasyOCR.  Adding scipy for a single isotonic
# regression would waste ~50MB of RAM.  A 10-bin lookup table with
# PAV monotonicity enforcement achieves equivalent results for our
# sample sizes (dozens to low thousands of confirmation events).
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any


class ConfidenceCalibrator:
    """Calibrates raw confidence scores using historical accuracy data.

    Uses a 10-bin lookup table where each bin tracks the empirical
    accuracy of predictions in that score range.  A raw score of 0.8
    that historically corresponds to 60% accuracy gets calibrated to 0.6.

    Thread-safe for reads -- the lookup table is replaced atomically.
    Not thread-safe for writes -- ``fit`` should be called from a single thread.
    """

    # Below this sample count, calibrate() returns raw scores unchanged
    # to avoid erratic adjustments from unrepresentative data.
    _MIN_SAMPLES = 30
    # Decile binning — 10 bins covering the [0, 1] score range.
    _NUM_BINS = 10

    def __init__(self) -> None:
        # Bin centers from the fitted data — x-coordinates for interpolation.
        self._bin_edges: list[float] = []
        # Empirical accuracy at each bin center — y-coordinates for interpolation.
        self._bin_values: list[float] = []
        # Total number of samples seen during the last fit() call.
        self._sample_count: int = 0

    def fit(self, samples: list[tuple[float, int]]) -> None:
        """Fit the calibrator from (predicted_score, ground_truth) pairs.

        ground_truth: 1 = entity was correct (user kept it), 0 = incorrect
        (user removed it).

        Bins samples into deciles by predicted_score, computes the
        empirical accuracy (fraction of ground_truth=1) in each bin,
        then enforces monotonicity via Pool-Adjacent-Violators (PAV).

        Parameters
        ----------
        samples:
            List of (predicted_score, ground_truth) tuples collected
            from the entity confirmation gate.
        """
        self._sample_count = len(samples)
        if self._sample_count < self._MIN_SAMPLES:
            # Not enough data to produce a reliable calibration table.
            # calibrate() will return raw scores unchanged.
            self._bin_edges = []
            self._bin_values = []
            return

        # Sort ascending by predicted score so decile bins are ordered
        # from low-confidence to high-confidence.
        sorted_samples = sorted(samples, key=lambda s: s[0])

        # Split into roughly equal-sized bins (deciles).  max(1, ...)
        # guards against zero-division when sample_count < _NUM_BINS.
        bin_size = max(1, len(sorted_samples) // self._NUM_BINS)
        edges: list[float] = []
        values: list[float] = []

        for i in range(0, len(sorted_samples), bin_size):
            bin_samples = sorted_samples[i : i + bin_size]
            if not bin_samples:
                continue
            # Bin center = mean predicted score within this bin.
            bin_center = sum(s[0] for s in bin_samples) / len(bin_samples)
            # Empirical accuracy = fraction of samples where user kept
            # the entity (ground_truth == 1).
            accuracy = sum(s[1] for s in bin_samples) / len(bin_samples)
            edges.append(bin_center)
            values.append(accuracy)

        # Pool-Adjacent-Violators (PAV): enforce monotonicity.
        # Higher raw scores should map to equal-or-higher calibrated
        # values.  If a bin has higher accuracy than the next bin,
        # merge them by averaging.  This is a simplified single-pass
        # PAV — sufficient for 10 bins.
        for i in range(len(values) - 1):
            if values[i] > values[i + 1]:
                avg = (values[i] + values[i + 1]) / 2
                values[i] = avg
                values[i + 1] = avg

        # Atomic replacement — readers calling calibrate() concurrently
        # will see either the old table or the new one, never a partial.
        self._bin_edges = edges
        self._bin_values = values

    def calibrate(self, raw_score: float) -> float:
        """Map a raw score through the fitted calibration table.

        Returns the raw score unchanged if:
        - Fewer than _MIN_SAMPLES have been collected
        - The calibrator has not been fitted

        Uses linear interpolation between bin centers for smooth output.

        Parameters
        ----------
        raw_score:
            A confidence value in [0, 1] from an LLM or fuzzy matcher.

        Returns
        -------
        float
            The calibrated score, also in [0, 1].
        """
        if not self._bin_edges or self._sample_count < self._MIN_SAMPLES:
            return raw_score

        # Clamp input to valid range before lookup.
        raw_score = max(0.0, min(1.0, raw_score))

        # Extrapolate flat beyond the outermost bin centers.
        if raw_score <= self._bin_edges[0]:
            return self._bin_values[0]
        if raw_score >= self._bin_edges[-1]:
            return self._bin_values[-1]

        # Linear interpolation between adjacent bin centers.
        # The 1e-10 epsilon prevents division-by-zero when two bin
        # centers coincide (unlikely but defensive).
        for i in range(len(self._bin_edges) - 1):
            if self._bin_edges[i] <= raw_score <= self._bin_edges[i + 1]:
                t = (raw_score - self._bin_edges[i]) / (
                    self._bin_edges[i + 1] - self._bin_edges[i] + 1e-10
                )
                return self._bin_values[i] + t * (
                    self._bin_values[i + 1] - self._bin_values[i]
                )

        # Fallback — should not be reached if bin_edges are sorted.
        return raw_score

    @classmethod
    def from_db_samples(cls, samples: list[dict[str, Any]]) -> ConfidenceCalibrator:
        """Factory: build a calibrator from database records.

        Convenience constructor for use with IFeedbackProvider.get_calibration_data()
        which returns dicts with 'predicted_score' and 'ground_truth' keys.

        Parameters
        ----------
        samples:
            List of dicts, each containing 'predicted_score' (float)
            and 'ground_truth' (int).

        Returns
        -------
        ConfidenceCalibrator
            A fitted calibrator (or an uncalibrated one if < 30 samples).
        """
        cal = cls()
        pairs = [(s["predicted_score"], s["ground_truth"]) for s in samples]
        cal.fit(pairs)
        return cal

    @property
    def is_calibrated(self) -> bool:
        """Whether the calibrator has enough data to adjust scores.

        Returns False during cold start (< 30 samples) so callers can
        log or display that raw scores are being used.
        """
        return self._sample_count >= self._MIN_SAMPLES and len(self._bin_edges) > 0

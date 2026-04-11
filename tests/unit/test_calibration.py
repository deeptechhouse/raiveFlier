"""Unit tests for the ConfidenceCalibrator (post-hoc score calibration).

# ─── MODULE OVERVIEW ────────────────────────────────────────────────
# Tests the ConfidenceCalibrator class that maps raw confidence/similarity
# scores to calibrated probabilities using isotonic regression.
#
# Key behaviors tested:
# - Cold start identity (no fit → raw score returned unchanged)
# - Minimum sample threshold (< 30 samples → identity)
# - Well-calibrated models remain approximately unchanged
# - Overconfident models are corrected downward
# - Monotonicity of calibrated output
# - Factory construction from database records
# - Edge case handling (out-of-range scores)
#
# The calibrator uses a binned lookup table fitted via pool adjacent
# violators (PAV) isotonic regression to ensure monotonicity.
# ────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import random

import pytest

from src.utils.calibration import ConfidenceCalibrator

# ─── Tests ─────────────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_cold_start_identity() -> None:
    """Before fit() is called, calibrate() returns the raw score unchanged."""
    cal = ConfidenceCalibrator()
    assert cal.calibrate(0.8) == 0.8
    assert cal.calibrate(0.3) == 0.3
    assert cal.calibrate(0.0) == 0.0
    assert cal.calibrate(1.0) == 1.0


@pytest.mark.asyncio()
async def test_under_30_samples_identity() -> None:
    """With fewer than 30 samples, calibrate() still returns raw scores.

    The minimum sample threshold prevents overfitting to a tiny,
    unrepresentative feedback dataset.
    """
    cal = ConfidenceCalibrator()
    # 20 samples — below the 30-sample threshold
    samples = [(0.9, 1)] * 10 + [(0.9, 0)] * 10
    cal.fit(samples)

    # Should still act as identity because sample count < 30
    assert cal.calibrate(0.8) == 0.8


@pytest.mark.asyncio()
async def test_well_calibrated_model() -> None:
    """When P(correct | score=x) approximately equals x, calibration is near-identity.

    We generate 100 samples where a score of ~x has roughly x probability
    of being correct.  After calibration, the output should be close to
    the input for mid-range scores.
    """
    random.seed(42)
    samples: list[tuple[float, int]] = []
    for _ in range(100):
        score = random.random()
        # Ground truth matches score probability — well-calibrated by construction
        truth = 1 if random.random() < score else 0
        samples.append((score, truth))

    cal = ConfidenceCalibrator()
    cal.fit(samples)

    # For a well-calibrated model, calibrate(0.5) should be roughly 0.5
    # Allow generous tolerance since we have only 100 random samples
    calibrated = cal.calibrate(0.5)
    assert 0.15 <= calibrated <= 0.85, (
        f"Well-calibrated model: calibrate(0.5) = {calibrated}, expected near 0.5"
    )


@pytest.mark.asyncio()
async def test_overconfident_model_corrected() -> None:
    """When score=0.9 but only ~50% are correct, calibration lowers the score.

    This simulates an overconfident model where high raw scores don't
    correspond to high accuracy.  The calibrator should learn to map
    0.9 to something closer to 0.5.
    """
    # All samples have score ~0.9 but only half are correct
    samples: list[tuple[float, int]] = []
    for i in range(100):
        # Spread scores slightly around 0.9 to avoid all landing in one bin
        score = 0.85 + (i % 10) * 0.015
        truth = 1 if i % 2 == 0 else 0  # 50% accuracy
        samples.append((score, truth))

    cal = ConfidenceCalibrator()
    cal.fit(samples)

    calibrated = cal.calibrate(0.9)
    # Should be pulled down from 0.9 toward ~0.5
    assert calibrated < 0.9, (
        f"Overconfident model: calibrate(0.9) = {calibrated}, should be < 0.9"
    )


@pytest.mark.asyncio()
async def test_monotonicity() -> None:
    """After fitting, calibrate(x) should be non-decreasing in x.

    The isotonic regression (PAV algorithm) enforces this invariant.
    We verify it empirically across a range of input scores.
    """
    random.seed(123)
    # Generate samples that might create non-monotonic raw bins
    samples: list[tuple[float, int]] = []
    for _ in range(200):
        score = random.random()
        # Slightly noisy calibration curve
        truth = 1 if random.random() < (score * 0.8 + 0.1) else 0
        samples.append((score, truth))

    cal = ConfidenceCalibrator()
    cal.fit(samples)

    # Check monotonicity: calibrate(0.3) <= calibrate(0.5) <= calibrate(0.7)
    c3 = cal.calibrate(0.3)
    c5 = cal.calibrate(0.5)
    c7 = cal.calibrate(0.7)
    assert c3 <= c5, f"calibrate(0.3)={c3} > calibrate(0.5)={c5}"
    assert c5 <= c7, f"calibrate(0.5)={c5} > calibrate(0.7)={c7}"


@pytest.mark.asyncio()
async def test_from_db_samples_factory() -> None:
    """from_db_samples() creates a working calibrator from dict records."""
    db_records = [
        {"predicted_score": 0.9, "ground_truth": 1},
        {"predicted_score": 0.8, "ground_truth": 1},
        {"predicted_score": 0.3, "ground_truth": 0},
    ] * 15  # 45 records — above the 30-sample minimum

    cal = ConfidenceCalibrator.from_db_samples(db_records)

    assert cal.is_calibrated is True
    # Should return a valid float
    result = cal.calibrate(0.7)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


@pytest.mark.asyncio()
async def test_is_calibrated_false_initially() -> None:
    """A freshly created calibrator reports is_calibrated = False."""
    cal = ConfidenceCalibrator()
    assert cal.is_calibrated is False


@pytest.mark.asyncio()
async def test_is_calibrated_true_after_fit() -> None:
    """After fitting with >= 30 samples, is_calibrated = True."""
    cal = ConfidenceCalibrator()
    samples = [(0.5, 1)] * 25 + [(0.5, 0)] * 25  # 50 samples
    cal.fit(samples)
    assert cal.is_calibrated is True


@pytest.mark.asyncio()
async def test_edge_scores_clamped() -> None:
    """Scores outside [0, 1] are clamped and don't cause errors."""
    cal = ConfidenceCalibrator()

    # Without calibration (identity mode) — returns raw scores unchanged
    assert cal.calibrate(-0.1) == -0.1
    assert cal.calibrate(1.5) == 1.5

    # With calibration — should also handle gracefully
    samples = [(0.5, 1)] * 20 + [(0.5, 0)] * 20
    cal.fit(samples)

    result_low = cal.calibrate(-0.1)
    assert 0.0 <= result_low <= 1.0

    result_high = cal.calibrate(1.5)
    assert 0.0 <= result_high <= 1.0

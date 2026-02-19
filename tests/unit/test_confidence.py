"""Unit tests for confidence scoring utilities."""

from __future__ import annotations

import pytest

from src.utils.confidence import (
    ConfidenceLevel,
    calculate_confidence,
    confidence_to_level,
    merge_confidence,
)


# ======================================================================
# calculate_confidence
# ======================================================================


class TestCalculateConfidence:
    """Tests for the calculate_confidence function."""

    def test_equal_weights(self) -> None:
        result = calculate_confidence([0.8, 0.6, 0.4])
        assert result == pytest.approx(0.6, abs=1e-9)

    def test_custom_weights(self) -> None:
        result = calculate_confidence([1.0, 0.0], weights=[3.0, 1.0])
        assert result == pytest.approx(0.75, abs=1e-9)

    def test_single_score(self) -> None:
        result = calculate_confidence([0.7])
        assert result == pytest.approx(0.7, abs=1e-9)

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="scores must not be empty"):
            calculate_confidence([])

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            calculate_confidence([0.5, 0.5], weights=[1.0])

    def test_all_zero_weights_returns_zero(self) -> None:
        result = calculate_confidence([0.8, 0.6], weights=[0.0, 0.0])
        assert result == 0.0

    def test_clamped_to_one(self) -> None:
        # Scores already in [0,1], but weighted average with normal weights
        # won't exceed 1.0; test with scores at max
        result = calculate_confidence([1.0, 1.0])
        assert result <= 1.0

    def test_clamped_to_zero(self) -> None:
        result = calculate_confidence([0.0, 0.0])
        assert result >= 0.0


# ======================================================================
# confidence_to_level
# ======================================================================


class TestConfidenceToLevel:
    """Tests for the confidence_to_level function."""

    def test_very_high_at_0_8(self) -> None:
        assert confidence_to_level(0.8) == ConfidenceLevel.VERY_HIGH

    def test_very_high_at_1_0(self) -> None:
        assert confidence_to_level(1.0) == ConfidenceLevel.VERY_HIGH

    def test_high_at_0_6(self) -> None:
        assert confidence_to_level(0.6) == ConfidenceLevel.HIGH

    def test_high_at_0_79(self) -> None:
        assert confidence_to_level(0.79) == ConfidenceLevel.HIGH

    def test_medium_at_0_5(self) -> None:
        assert confidence_to_level(0.5) == ConfidenceLevel.MEDIUM

    def test_medium_at_0_4(self) -> None:
        assert confidence_to_level(0.4) == ConfidenceLevel.MEDIUM

    def test_low_at_0_3(self) -> None:
        assert confidence_to_level(0.3) == ConfidenceLevel.LOW

    def test_low_at_0_2(self) -> None:
        assert confidence_to_level(0.2) == ConfidenceLevel.LOW

    def test_very_low_at_0_0(self) -> None:
        assert confidence_to_level(0.0) == ConfidenceLevel.VERY_LOW

    def test_very_low_at_0_19(self) -> None:
        assert confidence_to_level(0.19) == ConfidenceLevel.VERY_LOW


# ======================================================================
# merge_confidence
# ======================================================================


class TestMergeConfidence:
    """Tests for the merge_confidence function."""

    def test_default_weight_fifty_fifty(self) -> None:
        result = merge_confidence(0.8, 0.4)
        assert result == pytest.approx(0.6, abs=1e-9)

    def test_full_new_weight(self) -> None:
        result = merge_confidence(0.8, 0.2, new_weight=1.0)
        assert result == pytest.approx(0.2, abs=1e-9)

    def test_zero_new_weight(self) -> None:
        result = merge_confidence(0.8, 0.2, new_weight=0.0)
        assert result == pytest.approx(0.8, abs=1e-9)

    def test_custom_weight(self) -> None:
        result = merge_confidence(0.6, 1.0, new_weight=0.3)
        # 0.6 * 0.7 + 1.0 * 0.3 = 0.42 + 0.30 = 0.72
        assert result == pytest.approx(0.72, abs=1e-9)

    def test_clamped_result(self) -> None:
        result = merge_confidence(1.0, 1.0, new_weight=0.5)
        assert result <= 1.0
        assert result >= 0.0

    def test_weight_clamped_above_one(self) -> None:
        # new_weight > 1.0 should be clamped to 1.0
        result = merge_confidence(0.5, 0.9, new_weight=2.0)
        assert result == pytest.approx(0.9, abs=1e-9)

    def test_weight_clamped_below_zero(self) -> None:
        # new_weight < 0.0 should be clamped to 0.0
        result = merge_confidence(0.5, 0.9, new_weight=-1.0)
        assert result == pytest.approx(0.5, abs=1e-9)

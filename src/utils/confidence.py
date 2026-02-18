"""Confidence scoring utilities for entity extraction and research results."""

from enum import Enum


class ConfidenceLevel(Enum):
    """Human-readable confidence tiers."""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


def calculate_confidence(
    scores: list[float],
    weights: list[float] | None = None,
) -> float:
    """Compute a weighted average confidence score.

    Args:
        scores: Individual confidence scores, each in [0.0, 1.0].
        weights: Optional weights for each score. Defaults to equal weighting.

    Returns:
        Weighted average clamped to [0.0, 1.0].

    Raises:
        ValueError: If scores is empty or lengths of scores and weights differ.
    """
    if not scores:
        raise ValueError("scores must not be empty")

    if weights is None:
        weights = [1.0] * len(scores)

    if len(scores) != len(weights):
        raise ValueError("scores and weights must have the same length")

    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0

    weighted_sum = sum(s * w for s, w in zip(scores, weights, strict=True))
    return max(0.0, min(1.0, weighted_sum / total_weight))


def confidence_to_level(score: float) -> ConfidenceLevel:
    """Map a numeric confidence score to a human-readable level.

    Args:
        score: Confidence score in [0.0, 1.0].

    Returns:
        Corresponding ConfidenceLevel enum member.
    """
    if score < 0.2:
        return ConfidenceLevel.VERY_LOW
    if score < 0.4:
        return ConfidenceLevel.LOW
    if score < 0.6:
        return ConfidenceLevel.MEDIUM
    if score < 0.8:
        return ConfidenceLevel.HIGH
    return ConfidenceLevel.VERY_HIGH


def merge_confidence(existing: float, new: float, new_weight: float = 0.5) -> float:
    """Bayesian-style update of a confidence score with new evidence.

    Args:
        existing: Current confidence score in [0.0, 1.0].
        new: New evidence confidence score in [0.0, 1.0].
        new_weight: Weight given to the new evidence (0.0–1.0). Defaults to 0.5.

    Returns:
        Updated confidence score clamped to [0.0, 1.0].
    """
    new_weight = max(0.0, min(1.0, new_weight))
    merged = existing * (1.0 - new_weight) + new * new_weight
    return max(0.0, min(1.0, merged))

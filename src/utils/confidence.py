"""Confidence scoring utilities for entity extraction and research results.

The pipeline assigns numeric confidence scores (0.0--1.0) at many stages:
OCR text extraction, entity parsing, web research, artist identification, etc.
This module provides three core operations:

1. **calculate_confidence** -- Weighted average of multiple score signals.
   Used when combining OCR confidence, LLM confidence, and fuzzy-match
   similarity into a single per-entity score.
2. **confidence_to_level** -- Maps a numeric score to a human-readable
   tier (VERY_LOW through VERY_HIGH) for frontend display and logging.
3. **merge_confidence** -- Bayesian-style update that blends an existing
   score with new evidence, allowing the pipeline to incrementally
   improve confidence as more data sources are consulted.
"""

from enum import Enum


class ConfidenceLevel(Enum):
    """Human-readable confidence tiers.

    Used in API responses and frontend badges to give users an intuitive
    sense of how reliable an extracted entity or research result is.
    """

    VERY_LOW = "very_low"    # < 0.2 -- nearly no corroboration
    LOW = "low"              # 0.2 - 0.4 -- weak signal
    MEDIUM = "medium"        # 0.4 - 0.6 -- moderate evidence
    HIGH = "high"            # 0.6 - 0.8 -- strong corroboration
    VERY_HIGH = "very_high"  # >= 0.8 -- multiple independent confirmations


def calculate_confidence(
    scores: list[float],
    weights: list[float] | None = None,
) -> float:
    """Compute a weighted average confidence score.

    This is the core scoring function used across the pipeline. Example usage:
    combine OCR confidence (weight=1.0), LLM entity confidence (weight=2.0),
    and fuzzy-match similarity (weight=1.5) into a single artist score.

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

    # Default to equal weighting when no weights are provided.
    if weights is None:
        weights = [1.0] * len(scores)

    if len(scores) != len(weights):
        raise ValueError("scores and weights must have the same length")

    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0

    # Standard weighted average formula: sum(score_i * weight_i) / sum(weight_i)
    weighted_sum = sum(s * w for s, w in zip(scores, weights, strict=True))
    # Clamp to [0.0, 1.0] to guard against floating-point drift.
    return max(0.0, min(1.0, weighted_sum / total_weight))


def confidence_to_level(score: float) -> ConfidenceLevel:
    """Map a numeric confidence score to a human-readable level.

    The thresholds are evenly spaced at 0.2 intervals. These were chosen
    empirically: scores below 0.4 typically come from single-source
    extraction with no corroboration; scores above 0.8 usually mean
    multiple independent sources confirmed the entity.

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

    Called each time the pipeline obtains new evidence about an entity
    (e.g. a web search confirms an OCR-extracted artist name). The
    ``new_weight`` parameter controls how much the new evidence shifts
    the existing score -- higher values make the update more aggressive.

    Args:
        existing: Current confidence score in [0.0, 1.0].
        new: New evidence confidence score in [0.0, 1.0].
        new_weight: Weight given to the new evidence (0.0--1.0). Defaults to 0.5.

    Returns:
        Updated confidence score clamped to [0.0, 1.0].
    """
    # Clamp new_weight to valid range before applying.
    new_weight = max(0.0, min(1.0, new_weight))
    # Linear interpolation: existing * (1 - w) + new * w
    merged = existing * (1.0 - new_weight) + new * new_weight
    return max(0.0, min(1.0, merged))

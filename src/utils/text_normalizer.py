"""Text normalization utilities for DJ and artist names."""

import re

from rapidfuzz import fuzz, process


def normalize_artist_name(name: str) -> str:
    """Normalize a DJ/artist name for consistent matching.

    Strips common prefixes like "DJ", normalizes case, and removes
    excess whitespace.

    Args:
        name: Raw artist name string.

    Returns:
        Normalized artist name.
    """
    normalized = name.strip()

    # Strip leading "DJ " / "Dj " / "dj " prefix (case-insensitive)
    normalized = re.sub(r"^[Dd][Jj]\s+", "", normalized)

    # Collapse multiple spaces
    normalized = re.sub(r"\s+", " ", normalized)

    # Title-case normalize
    normalized = normalized.title()

    return normalized


def fuzzy_match(
    query: str,
    candidates: list[str],
    threshold: float = 0.8,
) -> tuple[str, float] | None:
    """Find the best fuzzy match for a query among candidates.

    Uses rapidfuzz token_sort_ratio for robust matching across word-order
    differences.

    Args:
        query: The string to match.
        candidates: List of candidate strings to match against.
        threshold: Minimum similarity score (0.0–1.0) to accept a match.

    Returns:
        A (best_match, score) tuple if a match meets the threshold, else None.
    """
    if not candidates:
        return None

    result = process.extractOne(
        query,
        candidates,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=threshold * 100,  # rapidfuzz uses 0–100 scale
    )

    if result is None:
        return None

    match_str, score, _ = result
    return (match_str, score / 100.0)


_SEPARATOR_PATTERN = re.compile(
    r"\s+[Bb]2[Bb]\s+|\s+[Vv][Ss]\.?\s+|\s+&\s+|\s+feat\.?\s+" r"|\s+ft\.?\s+|\s+featuring\s+|,\s*",
    re.IGNORECASE,
)


def split_artist_names(raw: str) -> list[str]:
    """Split a raw artist string into individual names.

    Handles common separators found on rave fliers: b2b, vs, &, feat., ft.,
    featuring, and commas.

    Args:
        raw: Raw artist name string potentially containing multiple names.

    Returns:
        List of individual artist name strings, stripped of whitespace.
    """
    parts = _SEPARATOR_PATTERN.split(raw)
    return [part.strip() for part in parts if part.strip()]

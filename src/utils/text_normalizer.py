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


# ------------------------------------------------------------------
# OCR error correction
# ------------------------------------------------------------------

# Common character misreads produced by traditional OCR engines on
# stylised rave-flier text.  Each tuple is (pattern, replacement).
# Applied only to EasyOCR/Tesseract output — NOT to LLM Vision output,
# which does not make these systematic errors.
_OCR_CORRECTIONS: list[tuple[re.Pattern[str], str]] = [
    # "rn" misread as two characters instead of "m"
    (re.compile(r"(?<=[a-zA-Z])rn(?=[a-zA-Z])"), "m"),
    # Leading "0" before 2+ letters → "O" (e.g. "0PEN" → "OPEN")
    (re.compile(r"\b0(?=[a-zA-Z]{2,})"), "O"),
    # Trailing "0" after 2+ letters → "O" (e.g. "CARL0" → "CARLO")
    (re.compile(r"(?<=[a-zA-Z][a-zA-Z])0\b"), "O"),
    # Leading "1" before 2+ letters → "l" (e.g. "1ive" → "live")
    (re.compile(r"\b1(?=[a-zA-Z]{2,})"), "l"),
    # Pipe character → "l"
    (re.compile(r"\|(?=[a-zA-Z])"), "l"),
    # "VV" → "W" (common with wide fonts)
    (re.compile(r"\bVV"), "W"),
]


def correct_ocr_errors(text: str) -> str:
    """Apply common OCR character-substitution corrections.

    Fixes systematic misreads that traditional OCR engines (EasyOCR,
    Tesseract) make on stylised rave-flier typography.  Should NOT
    be applied to LLM Vision output.

    Args:
        text: Raw OCR text.

    Returns:
        Corrected text.
    """
    corrected = text
    for pattern, replacement in _OCR_CORRECTIONS:
        corrected = pattern.sub(replacement, corrected)
    return corrected


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

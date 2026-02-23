"""Text normalization utilities for DJ and artist names.

This module handles three distinct normalization concerns:

1. **Artist name normalization** -- Strips "DJ" prefixes, normalizes case,
   and provides fuzzy matching via rapidfuzz so that OCR-mangled names
   (e.g. "CARL C0X") can be matched to known artists ("Carl Cox").

2. **OCR error correction** -- Fixes systematic character substitutions
   that traditional OCR engines make on stylised rave-flier typography
   (e.g. "rn" -> "m", "0" -> "O", "VV" -> "W").  Only applied to
   EasyOCR/Tesseract output, NOT to LLM Vision output.

3. **Transcript preprocessing** -- Strips timestamps, noise markers, and
   speaker labels from interview/podcast transcripts before they enter
   the RAG ingestion pipeline, so embeddings capture content not formatting.
"""

import re

from rapidfuzz import fuzz, process


def normalize_artist_name(name: str) -> str:
    """Normalize a DJ/artist name for consistent matching.

    Strips common prefixes like "DJ", normalizes case, and removes
    excess whitespace.  This ensures that "DJ Rush", "dj rush", and
    "  DJ  Rush " all normalize to "Rush" for deduplication.

    Args:
        name: Raw artist name string.

    Returns:
        Normalized artist name.
    """
    normalized = name.strip()

    # Strip leading "DJ " / "Dj " / "dj " prefix -- DJs are often listed
    # both with and without the prefix on the same flier.
    normalized = re.sub(r"^[Dd][Jj]\s+", "", normalized)

    # Collapse multiple spaces (common in OCR output from spaced-out fonts)
    normalized = re.sub(r"\s+", " ", normalized)

    # Title-case for display consistency ("carl cox" -> "Carl Cox")
    normalized = normalized.title()

    return normalized


def fuzzy_match(
    query: str,
    candidates: list[str],
    threshold: float = 0.8,
) -> tuple[str, float] | None:
    """Find the best fuzzy match for a query among candidates.

    Uses rapidfuzz ``token_sort_ratio`` which sorts tokens alphabetically
    before comparing -- this handles word-order differences common in
    OCR output (e.g. "Cox Carl" matches "Carl Cox").

    Args:
        query: The string to match.
        candidates: List of candidate strings to match against.
        threshold: Minimum similarity score (0.0--1.0) to accept a match.

    Returns:
        A (best_match, score) tuple if a match meets the threshold, else None.
    """
    if not candidates:
        return None

    result = process.extractOne(
        query,
        candidates,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=threshold * 100,  # rapidfuzz uses 0-100 scale internally
    )

    if result is None:
        return None

    match_str, score, _ = result
    return (match_str, score / 100.0)  # Normalize back to 0.0-1.0 range


# ------------------------------------------------------------------
# OCR error correction
# ------------------------------------------------------------------

# Common character misreads produced by traditional OCR engines on
# stylised rave-flier text.  Each tuple is (compiled_regex, replacement).
# Applied only to EasyOCR/Tesseract output -- NOT to LLM Vision output,
# which uses visual understanding rather than character recognition and
# does not make these systematic substitution errors.
_OCR_CORRECTIONS: list[tuple[re.Pattern[str], str]] = [
    # "rn" misread as two chars instead of "m" (e.g. "Arrning" -> "Arming")
    (re.compile(r"(?<=[a-zA-Z])rn(?=[a-zA-Z])"), "m"),
    # Leading "0" before 2+ letters -> "O" (e.g. "0PEN" -> "OPEN")
    (re.compile(r"\b0(?=[a-zA-Z]{2,})"), "O"),
    # Trailing "0" after 2+ letters -> "O" (e.g. "CARL0" -> "CARLO")
    (re.compile(r"(?<=[a-zA-Z][a-zA-Z])0\b"), "O"),
    # Leading "1" before 2+ letters -> "l" (e.g. "1ive" -> "live")
    (re.compile(r"\b1(?=[a-zA-Z]{2,})"), "l"),
    # Pipe "|" before letters -> "l" (vertical bar confused with lowercase L)
    (re.compile(r"\|(?=[a-zA-Z])"), "l"),
    # "VV" at word start -> "W" (common with wide/stylized fonts)
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


# Regex that splits on all common multi-artist separators found on rave fliers.
# Handles: "b2b", "vs", "&", "feat.", "ft.", "featuring", and commas.
_SEPARATOR_PATTERN = re.compile(
    r"\s+[Bb]2[Bb]\s+|\s+[Vv][Ss]\.?\s+|\s+&\s+|\s+feat\.?\s+" r"|\s+ft\.?\s+|\s+featuring\s+|,\s*",
    re.IGNORECASE,
)


def split_artist_names(raw: str) -> list[str]:
    """Split a raw artist string into individual names.

    Handles common separators found on rave fliers: b2b, vs, &, feat., ft.,
    featuring, and commas.  For example:
    "Carl Cox b2b Adam Beyer, Nina Kraviz" -> ["Carl Cox", "Adam Beyer", "Nina Kraviz"]

    Args:
        raw: Raw artist name string potentially containing multiple names.

    Returns:
        List of individual artist name strings, stripped of whitespace.
    """
    parts = _SEPARATOR_PATTERN.split(raw)
    return [part.strip() for part in parts if part.strip()]


# ------------------------------------------------------------------
# Transcript text preprocessing
# ------------------------------------------------------------------
# These patterns strip formatting artifacts from interview/podcast
# transcripts so the text is clean for RAG embedding.  Timestamps and
# speaker labels would otherwise pollute semantic similarity.

# Bracketed timestamps: [00:15:22], [1:05:30], [15:22]
_BRACKET_TIMESTAMP = re.compile(r"\[\d{1,2}:\d{2}(?::\d{2})?\]")

# Bare timestamps at line start: 00:15:22 or 00:15:22 -
_BARE_TIMESTAMP = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?\s*[-\u2013\u2014]?\s*", re.MULTILINE)

# Parenthesized timestamps: (15:22), (1:05:30)
_PAREN_TIMESTAMP = re.compile(r"\(\d{1,2}:\d{2}(?::\d{2})?\)")

# Noise markers: [INAUDIBLE], [CROSSTALK], [LAUGHTER], [MUSIC], etc.
_NOISE_MARKER = re.compile(
    r"\[(?:INAUDIBLE|CROSSTALK|LAUGHTER|MUSIC|APPLAUSE|SILENCE|PAUSE"
    r"|BACKGROUND NOISE|OVERLAPPING|UNINTELLIGIBLE|FOREIGN LANGUAGE"
    r"|inaudible|crosstalk|laughter|music|applause)\]",
)

# Speaker labels at line start: "SPEAKER NAME:", "DJ Rush:", "Interviewer:"
# Matches 1-4 capitalized/mixed words followed by a colon.
_SPEAKER_LABEL = re.compile(r"^[ \t]*[A-Z][A-Za-z'\-.]+(?:\s+[A-Za-z'\-.]+){0,3}\s*:\s*", re.MULTILINE)

# Collapse 3+ newlines to double-newline (preserves paragraph breaks)
_MULTI_NEWLINE = re.compile(r"\n{3,}")

# Collapse 2+ spaces/tabs to single space
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def preprocess_transcript(text: str) -> str:
    """Clean transcript text before chunking and embedding.

    Removes timestamps, noise markers, and speaker labels that would
    pollute semantic embeddings.  Speaker label removal inserts paragraph
    breaks so the chunker creates natural boundaries at speaker turns.

    Args:
        text: Raw transcript text.

    Returns:
        Cleaned text suitable for chunking.
    """
    if not text:
        return text

    cleaned = text

    # 1. Strip timestamps
    cleaned = _BRACKET_TIMESTAMP.sub("", cleaned)
    cleaned = _BARE_TIMESTAMP.sub("", cleaned)
    cleaned = _PAREN_TIMESTAMP.sub("", cleaned)

    # 2. Remove noise markers
    cleaned = _NOISE_MARKER.sub("", cleaned)

    # 3. Normalize speaker labels → paragraph break (preserves turn boundaries)
    cleaned = _SPEAKER_LABEL.sub("\n\n", cleaned)

    # 4. Collapse excessive whitespace
    cleaned = _MULTI_NEWLINE.sub("\n\n", cleaned)
    cleaned = _MULTI_SPACE.sub(" ", cleaned)

    return cleaned.strip()

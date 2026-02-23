"""Shared OCR utilities for cross-pass text deduplication and result merging.

Used by both the EasyOCR and Tesseract providers to combine unique text
findings from multiple preprocessing passes into a single merged result.

The core problem: each of the 7 preprocessing passes (see ``image_preprocessor``)
produces its own OCR output, so the same text fragment (e.g. "CARL COX") may
appear in 3-4 passes with slightly different spellings and confidence scores.
This module solves that with two functions:

1. **deduplicate_text_regions** -- Fuzzy-matches all TextRegion objects and
   keeps only the highest-confidence version of each unique text fragment.
   Uses a length pre-filter + rapidfuzz token-sort ratio for efficiency.

2. **merge_pass_results** -- Collects regions from all passes, filters by
   a minimum confidence floor, deduplicates, and assembles a single merged
   result with combined raw text and an average confidence score.
"""

from __future__ import annotations

from rapidfuzz import fuzz

from src.models.flier import TextRegion


def deduplicate_text_regions(
    regions: list[TextRegion],
    similarity_threshold: float = 0.80,
) -> list[TextRegion]:
    """Remove near-duplicate text regions, keeping the highest-confidence version.

    Uses token-sort fuzzy matching so word-order differences (e.g. "CARL COX"
    vs "COX CARL") are tolerated.  The algorithm is O(n*k) where n is the
    input count and k is the deduplicated count -- acceptable because OCR
    rarely produces more than a few hundred regions per image.

    Args:
        regions: List of TextRegion objects, possibly from different OCR passes.
        similarity_threshold: Minimum similarity ratio (0.0--1.0) to consider
            two regions as duplicates.

    Returns:
        Deduplicated list of TextRegion objects.
    """
    if not regions:
        return []

    kept: list[TextRegion] = []
    kept_lower: list[str] = []  # Pre-computed lowercase text for O(1) comparisons

    for region in regions:
        text_lower = region.text.strip().lower()
        if not text_lower:
            continue
        text_len = len(text_lower)

        is_duplicate = False
        for idx, existing_lower in enumerate(kept_lower):
            # Length pre-filter: if strings differ by more than 2x in length,
            # they can't possibly be 80%+ similar -- skip the expensive
            # fuzzy comparison entirely.
            existing_len = len(existing_lower)
            if existing_len == 0:
                continue
            len_ratio = min(text_len, existing_len) / max(text_len, existing_len)
            if len_ratio < 0.5:
                continue

            # token_sort_ratio sorts tokens alphabetically before comparing,
            # making it robust to OCR word-order scrambling.
            ratio = fuzz.token_sort_ratio(text_lower, existing_lower) / 100.0
            if ratio >= similarity_threshold:
                # Duplicate found -- keep whichever version has higher confidence.
                if region.confidence > kept[idx].confidence:
                    kept[idx] = region
                    kept_lower[idx] = text_lower
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(region)
            kept_lower.append(text_lower)

    return kept


def merge_pass_results(
    all_results: list[dict | None],
    min_region_confidence: float = 0.15,
    similarity_threshold: float = 0.80,
) -> dict | None:
    """Merge OCR results from multiple preprocessing passes into one result.

    This is the final assembly step after all passes have run.  The flow is:
    1. Collect all TextRegion objects from every pass result
    2. Filter out regions below ``min_region_confidence`` (noise floor)
    3. Deduplicate via fuzzy matching (``deduplicate_text_regions``)
    4. Assemble merged raw_text and compute average confidence

    Args:
        all_results: List of result dicts from individual passes. Each dict
            must contain ``raw_text`` (str), ``confidence`` (float), and
            ``bounding_boxes`` (list[TextRegion]).  ``None`` entries are
            skipped.
        min_region_confidence: Minimum per-region confidence to include a
            text region in the merged output.  The 0.15 default is
            deliberately low to avoid discarding valid text from
            challenging passes.
        similarity_threshold: Fuzzy-match threshold for deduplication.

    Returns:
        A merged result dict with ``raw_text``, ``confidence``, and
        ``bounding_boxes``, or ``None`` if no usable text was found.
    """
    # Step 1-2: Collect all regions that meet the confidence floor.
    all_regions: list[TextRegion] = []

    for result in all_results:
        if result is None:
            continue
        for region in result.get("bounding_boxes", []):
            if region.confidence >= min_region_confidence:
                all_regions.append(region)

    if not all_regions:
        return None

    # Step 3: Deduplicate -- same text from different passes gets merged.
    merged_regions = deduplicate_text_regions(all_regions, similarity_threshold)

    if not merged_regions:
        return None

    # Step 4: Assemble final output with combined text and average confidence.
    raw_text = "\n".join(r.text for r in merged_regions)
    total_conf = sum(r.confidence for r in merged_regions)
    avg_confidence = total_conf / len(merged_regions)

    return {
        "raw_text": raw_text,
        "confidence": min(1.0, avg_confidence),
        "bounding_boxes": merged_regions,
    }

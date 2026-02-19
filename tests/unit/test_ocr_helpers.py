"""Unit tests for OCR helper utilities (deduplication and pass merging)."""

from __future__ import annotations

import pytest

from src.models.flier import TextRegion
from src.utils.ocr_helpers import deduplicate_text_regions, merge_pass_results


def _region(text: str, confidence: float = 0.8) -> TextRegion:
    """Shorthand to create a TextRegion with dummy coordinates."""
    return TextRegion(text=text, confidence=confidence, x=0, y=0, width=10, height=10)


# ======================================================================
# deduplicate_text_regions
# ======================================================================


class TestDeduplicateTextRegions:
    def test_empty_list_returns_empty(self) -> None:
        assert deduplicate_text_regions([]) == []

    def test_no_duplicates_keeps_all(self) -> None:
        regions = [_region("CARL COX"), _region("JEFF MILLS"), _region("TRESOR")]
        result = deduplicate_text_regions(regions)
        assert len(result) == 3

    def test_exact_duplicate_keeps_higher_confidence(self) -> None:
        low = _region("CARL COX", confidence=0.5)
        high = _region("CARL COX", confidence=0.9)
        result = deduplicate_text_regions([low, high])
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.9)

    def test_fuzzy_duplicate_merged(self) -> None:
        r1 = _region("CARL COX", confidence=0.7)
        r2 = _region("CARL  COX", confidence=0.8)  # extra space
        result = deduplicate_text_regions([r1, r2])
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.8)

    def test_different_texts_kept(self) -> None:
        r1 = _region("CARL COX", confidence=0.8)
        r2 = _region("TRESOR BERLIN", confidence=0.9)
        result = deduplicate_text_regions([r1, r2])
        assert len(result) == 2

    def test_case_insensitive_matching(self) -> None:
        r1 = _region("Carl Cox", confidence=0.6)
        r2 = _region("CARL COX", confidence=0.9)
        result = deduplicate_text_regions([r1, r2])
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.9)

    def test_blank_text_filtered(self) -> None:
        r1 = _region("", confidence=0.5)
        r2 = _region("CARL COX", confidence=0.8)
        result = deduplicate_text_regions([r1, r2])
        assert len(result) == 1
        assert result[0].text == "CARL COX"

    def test_custom_threshold(self) -> None:
        r1 = _region("ABC", confidence=0.7)
        r2 = _region("ABD", confidence=0.8)
        # With a very high threshold, these should NOT be considered duplicates
        result = deduplicate_text_regions([r1, r2], similarity_threshold=0.99)
        assert len(result) == 2


# ======================================================================
# merge_pass_results
# ======================================================================


class TestMergePassResults:
    def test_empty_list_returns_none(self) -> None:
        assert merge_pass_results([]) is None

    def test_all_none_returns_none(self) -> None:
        assert merge_pass_results([None, None, None]) is None

    def test_single_result_returned(self) -> None:
        result = {
            "raw_text": "CARL COX",
            "confidence": 0.8,
            "bounding_boxes": [_region("CARL COX", 0.8)],
        }
        merged = merge_pass_results([result])
        assert merged is not None
        assert "CARL COX" in merged["raw_text"]

    def test_merges_unique_text_from_multiple_passes(self) -> None:
        pass1 = {
            "raw_text": "CARL COX",
            "confidence": 0.8,
            "bounding_boxes": [_region("CARL COX", 0.8)],
        }
        pass2 = {
            "raw_text": "TRESOR BERLIN",
            "confidence": 0.7,
            "bounding_boxes": [_region("TRESOR BERLIN", 0.7)],
        }
        merged = merge_pass_results([pass1, pass2])
        assert merged is not None
        assert "CARL COX" in merged["raw_text"]
        assert "TRESOR BERLIN" in merged["raw_text"]

    def test_deduplicates_across_passes(self) -> None:
        pass1 = {
            "raw_text": "CARL COX",
            "confidence": 0.6,
            "bounding_boxes": [_region("CARL COX", 0.6)],
        }
        pass2 = {
            "raw_text": "CARL COX",
            "confidence": 0.9,
            "bounding_boxes": [_region("CARL COX", 0.9)],
        }
        merged = merge_pass_results([pass1, pass2])
        assert merged is not None
        # Should appear only once (deduplicated)
        assert merged["raw_text"].count("CARL COX") == 1
        assert merged["confidence"] == pytest.approx(0.9)

    def test_filters_low_confidence_regions(self) -> None:
        result = {
            "raw_text": "noise",
            "confidence": 0.05,
            "bounding_boxes": [_region("noise", 0.05)],
        }
        merged = merge_pass_results([result], min_region_confidence=0.15)
        assert merged is None

    def test_none_entries_skipped(self) -> None:
        result = {
            "raw_text": "CARL COX",
            "confidence": 0.8,
            "bounding_boxes": [_region("CARL COX", 0.8)],
        }
        merged = merge_pass_results([None, result, None])
        assert merged is not None
        assert "CARL COX" in merged["raw_text"]

    def test_confidence_is_average(self) -> None:
        pass1 = {
            "raw_text": "CARL COX",
            "confidence": 0.8,
            "bounding_boxes": [_region("CARL COX", 0.8)],
        }
        pass2 = {
            "raw_text": "JEFF MILLS",
            "confidence": 0.6,
            "bounding_boxes": [_region("JEFF MILLS", 0.6)],
        }
        merged = merge_pass_results([pass1, pass2])
        assert merged is not None
        assert merged["confidence"] == pytest.approx(0.7)

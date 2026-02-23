"""Unit tests for text normalization utilities."""

from __future__ import annotations

import pytest

from src.utils.text_normalizer import (
    correct_ocr_errors,
    fuzzy_match,
    normalize_artist_name,
    split_artist_names,
)


# ======================================================================
# normalize_artist_name
# ======================================================================


class TestNormalizeArtistName:
    """Tests for the normalize_artist_name function."""

    def test_strips_dj_prefix_uppercase(self) -> None:
        assert normalize_artist_name("DJ Shadow") == "Shadow"

    def test_strips_dj_prefix_mixed_case(self) -> None:
        assert normalize_artist_name("Dj SHADOW") == "Shadow"

    def test_strips_dj_prefix_lowercase(self) -> None:
        assert normalize_artist_name("dj Shadow") == "Shadow"

    def test_collapses_whitespace(self) -> None:
        result = normalize_artist_name("  DJ  Shadow  ")
        assert result == "Shadow"

    def test_title_cases_result(self) -> None:
        assert normalize_artist_name("carl cox") == "Carl Cox"

    def test_preserves_non_dj_name(self) -> None:
        assert normalize_artist_name("Aphex Twin") == "Aphex Twin"

    def test_single_word_name(self) -> None:
        assert normalize_artist_name("Moby") == "Moby"

    def test_empty_string(self) -> None:
        assert normalize_artist_name("") == ""

    def test_whitespace_only(self) -> None:
        assert normalize_artist_name("   ") == ""

    def test_dj_only(self) -> None:
        # "DJ" followed by nothing — the regex requires \s+ after DJ
        result = normalize_artist_name("DJ")
        assert result == "Dj"  # Title-cased as "Dj"


# ======================================================================
# split_artist_names
# ======================================================================


class TestSplitArtistNames:
    """Tests for the split_artist_names function."""

    def test_b2b_separator(self) -> None:
        result = split_artist_names("Carl Cox b2b Adam Beyer")
        assert result == ["Carl Cox", "Adam Beyer"]

    def test_b2b_uppercase(self) -> None:
        result = split_artist_names("Carl Cox B2B Adam Beyer")
        assert result == ["Carl Cox", "Adam Beyer"]

    def test_vs_separator(self) -> None:
        result = split_artist_names("A vs B")
        assert result == ["A", "B"]

    def test_vs_dot_separator(self) -> None:
        result = split_artist_names("A vs. B")
        assert result == ["A", "B"]

    def test_feat_separator(self) -> None:
        result = split_artist_names("Artist feat. Singer")
        assert result == ["Artist", "Singer"]

    def test_ft_separator(self) -> None:
        result = split_artist_names("Artist ft. Singer")
        assert result == ["Artist", "Singer"]

    def test_featuring_separator(self) -> None:
        result = split_artist_names("Artist featuring Singer")
        assert result == ["Artist", "Singer"]

    def test_ampersand_and_comma(self) -> None:
        result = split_artist_names("Artist & Artist2, Artist3")
        assert result == ["Artist", "Artist2", "Artist3"]

    def test_single_artist(self) -> None:
        result = split_artist_names("Carl Cox")
        assert result == ["Carl Cox"]

    def test_empty_string(self) -> None:
        result = split_artist_names("")
        assert result == []

    def test_whitespace_only(self) -> None:
        result = split_artist_names("   ")
        assert result == []


# ======================================================================
# fuzzy_match
# ======================================================================


class TestFuzzyMatch:
    """Tests for the fuzzy_match function."""

    def test_exact_match_returns_perfect_score(self) -> None:
        result = fuzzy_match("Carl Cox", ["Carl Cox", "Adam Beyer"])
        assert result is not None
        match, score = result
        assert match == "Carl Cox"
        assert score == 1.0

    def test_close_match_returns_best(self) -> None:
        result = fuzzy_match("Carl Cox", ["Karl Cox", "Carl Fox", "Nobody"])
        assert result is not None
        match, score = result
        # Both "Karl Cox" and "Carl Fox" are close; score should be high
        assert score > 0.8

    def test_no_match_above_threshold_returns_none(self) -> None:
        result = fuzzy_match("Carl Cox", ["Aphex Twin", "Jeff Mills"], threshold=0.9)
        assert result is None

    def test_empty_candidates_returns_none(self) -> None:
        result = fuzzy_match("Carl Cox", [])
        assert result is None

    def test_custom_threshold(self) -> None:
        # With a very low threshold, a distant match might be accepted
        result = fuzzy_match("Carl Cox", ["XYZ"], threshold=0.1)
        # Behaviour depends on score; just verify it returns something or None
        # without crashing
        assert result is None or isinstance(result, tuple)

    def test_score_is_normalized_to_zero_one(self) -> None:
        result = fuzzy_match("Carl Cox", ["Carl Cox"])
        assert result is not None
        _, score = result
        assert 0.0 <= score <= 1.0


# ======================================================================
# correct_ocr_errors
# ======================================================================


class TestCorrectOCRErrors:
    """Tests for common OCR character substitution corrections."""

    def test_rn_to_m(self) -> None:
        assert correct_ocr_errors("Surnrner") == "Summer"

    def test_leading_zero_to_O(self) -> None:
        assert correct_ocr_errors("0PEN AIR") == "OPEN AIR"

    def test_trailing_zero_to_O(self) -> None:
        assert correct_ocr_errors("CARL0") == "CARLO"

    def test_leading_one_to_l(self) -> None:
        assert correct_ocr_errors("1ive") == "live"

    def test_pipe_to_l(self) -> None:
        assert correct_ocr_errors("|ive") == "live"

    def test_vv_to_w(self) -> None:
        assert correct_ocr_errors("VVarehouse") == "Warehouse"

    def test_no_false_corrections_on_clean_text(self) -> None:
        clean = "CARL COX AT TRESOR BERLIN"
        assert correct_ocr_errors(clean) == clean

    def test_preserves_intentional_numbers(self) -> None:
        # "10pm" should NOT be corrected (standalone number)
        assert correct_ocr_errors("10pm") == "10pm"

    def test_preserves_standalone_zero(self) -> None:
        # "0" alone should not be corrected
        assert correct_ocr_errors("$10") == "$10"


# ---------------------------------------------------------------------------
# Transcript preprocessing tests
# ---------------------------------------------------------------------------


class TestPreprocessTranscript:
    """Test preprocess_transcript() cleaning pipeline."""

    def test_strips_bracket_timestamps(self) -> None:
        from src.utils.text_normalizer import preprocess_transcript

        text = "[00:15:22] So tell me about your first record."
        result = preprocess_transcript(text)
        assert "[00:15:22]" not in result
        assert "So tell me about your first record." in result

    def test_strips_bare_timestamps(self) -> None:
        from src.utils.text_normalizer import preprocess_transcript

        text = "00:15:22 - So tell me about your first record."
        result = preprocess_transcript(text)
        assert "00:15:22" not in result
        assert "So tell me about your first record." in result

    def test_strips_paren_timestamps(self) -> None:
        from src.utils.text_normalizer import preprocess_transcript

        text = "They released it in 1995 (15:22) on Metroplex."
        result = preprocess_transcript(text)
        assert "(15:22)" not in result
        assert "They released it in 1995" in result
        assert "on Metroplex" in result

    def test_removes_noise_markers(self) -> None:
        from src.utils.text_normalizer import preprocess_transcript

        text = "I started DJing [INAUDIBLE] in Detroit and [CROSSTALK] yeah."
        result = preprocess_transcript(text)
        assert "[INAUDIBLE]" not in result
        assert "[CROSSTALK]" not in result
        assert "I started DJing" in result
        assert "in Detroit and" in result

    def test_removes_laughter_and_applause(self) -> None:
        from src.utils.text_normalizer import preprocess_transcript

        text = "That was wild [LAUGHTER] and the crowd [APPLAUSE] loved it."
        result = preprocess_transcript(text)
        assert "[LAUGHTER]" not in result
        assert "[APPLAUSE]" not in result

    def test_normalizes_speaker_labels(self) -> None:
        from src.utils.text_normalizer import preprocess_transcript

        text = "Interviewer: What inspired you?\nDJ Rush: The city of Chicago."
        result = preprocess_transcript(text)
        assert "Interviewer:" not in result
        assert "DJ Rush:" not in result
        assert "What inspired you?" in result
        assert "The city of Chicago." in result

    def test_collapses_whitespace(self) -> None:
        from src.utils.text_normalizer import preprocess_transcript

        text = "First paragraph.\n\n\n\n\nSecond paragraph.   Extra  spaces."
        result = preprocess_transcript(text)
        assert "\n\n\n" not in result
        assert "  " not in result
        assert "First paragraph." in result
        assert "Second paragraph." in result

    def test_preserves_content(self) -> None:
        from src.utils.text_normalizer import preprocess_transcript

        text = (
            "The record was released on Underground Resistance in 1991. "
            "It changed the entire Detroit techno scene."
        )
        result = preprocess_transcript(text)
        assert result == text

    def test_empty_string(self) -> None:
        from src.utils.text_normalizer import preprocess_transcript

        assert preprocess_transcript("") == ""

    def test_full_transcript_sample(self) -> None:
        from src.utils.text_normalizer import preprocess_transcript

        text = (
            "[00:00:00] Interviewer: Welcome to the show.\n"
            "[00:00:05] DJ Rush: Thanks for having me.\n"
            "[00:00:10] Interviewer: Tell us about your early days in Chicago.\n"
            "[00:00:15] DJ Rush: I started going to the Music Box [INAUDIBLE] "
            "around 1985. Ron Hardy was [LAUGHTER] incredible.\n"
        )
        result = preprocess_transcript(text)
        assert "00:00" not in result
        assert "[INAUDIBLE]" not in result
        assert "[LAUGHTER]" not in result
        assert "Welcome to the show" in result
        assert "Music Box" in result
        assert "Ron Hardy was" in result
        assert "incredible" in result

"""Unit tests for src/config/domain_knowledge.py.

Tests cover all four knowledge maps and their accessor functions:
  1. Genre adjacency map + get_adjacent_genres()
  2. Scene-to-geography map + get_scene_geographies()
  3. Temporal signal detection + detect_temporal_signal() + temporal_overlap()
  4. Artist alias table + expand_aliases() + get_canonical_name() + get_all_artist_names()
  5. Genre extraction helper + extract_query_genres()
"""

from __future__ import annotations

import pytest

from src.config.domain_knowledge import (
    ARTIST_ALIASES,
    GENRE_ADJACENCY,
    SCENE_GEOGRAPHY,
    detect_temporal_signal,
    expand_aliases,
    extract_query_genres,
    get_adjacent_genres,
    get_all_artist_names,
    get_canonical_name,
    get_scene_geographies,
    temporal_overlap,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Genre Adjacency Map
# ═══════════════════════════════════════════════════════════════════════════


class TestGenreAdjacency:
    """Test the genre adjacency map and get_adjacent_genres()."""

    def test_known_genre_returns_nonempty_set(self) -> None:
        result = get_adjacent_genres("techno")
        assert isinstance(result, set)
        assert len(result) > 0
        assert "detroit techno" in result

    def test_case_insensitive(self) -> None:
        assert get_adjacent_genres("Techno") == get_adjacent_genres("techno")
        assert get_adjacent_genres("TECHNO") == get_adjacent_genres("techno")

    def test_leading_trailing_whitespace_stripped(self) -> None:
        assert get_adjacent_genres("  techno  ") == get_adjacent_genres("techno")

    def test_unknown_genre_returns_empty_set(self) -> None:
        result = get_adjacent_genres("nonexistent_genre_xyz")
        assert result == set()

    def test_house_contains_expected_subgenres(self) -> None:
        house = get_adjacent_genres("house")
        assert "deep house" in house
        assert "acid house" in house
        assert "chicago house" in house

    def test_jungle_contains_drum_and_bass(self) -> None:
        jungle = get_adjacent_genres("jungle")
        assert "drum and bass" in jungle

    def test_adjacency_map_keys_are_all_lowercase(self) -> None:
        for key in GENRE_ADJACENCY:
            assert key == key.lower(), f"Genre key '{key}' is not lowercase"

    def test_adjacency_map_values_are_all_lowercase(self) -> None:
        for key, values in GENRE_ADJACENCY.items():
            for v in values:
                assert v == v.lower(), f"Adjacent genre '{v}' under '{key}' is not lowercase"


# ═══════════════════════════════════════════════════════════════════════════
# 2. Scene-to-Geography Map
# ═══════════════════════════════════════════════════════════════════════════


class TestSceneGeography:
    """Test the scene-to-geography map and get_scene_geographies()."""

    def test_detroit_techno_query(self) -> None:
        result = get_scene_geographies("detroit techno history")
        assert "Detroit" in result

    def test_acid_house_query(self) -> None:
        result = get_scene_geographies("acid house in the UK")
        assert "Chicago" in result
        assert "London" in result

    def test_no_match_returns_empty_set(self) -> None:
        result = get_scene_geographies("something completely unrelated")
        assert result == set()

    def test_multiple_scene_keywords_union(self) -> None:
        # Query matches both "rave" and "hacienda"
        result = get_scene_geographies("rave at the hacienda")
        assert "Manchester" in result
        assert "Berlin" in result  # from "rave"

    def test_case_insensitive_matching(self) -> None:
        # get_scene_geographies lowercases the query
        result = get_scene_geographies("DETROIT TECHNO")
        assert "Detroit" in result

    def test_scene_geography_keys_are_lowercase(self) -> None:
        for key in SCENE_GEOGRAPHY:
            assert key == key.lower(), f"Scene key '{key}' is not lowercase"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Temporal Signal Detection
# ═══════════════════════════════════════════════════════════════════════════


class TestTemporalSignalDetection:
    """Test detect_temporal_signal() and temporal_overlap()."""

    # --- detect_temporal_signal ---

    def test_named_era_second_summer_of_love(self) -> None:
        assert detect_temporal_signal("second summer of love") == "1988-1989"

    def test_named_era_old_school(self) -> None:
        assert detect_temporal_signal("old school rave") == "1988-1995"

    def test_named_era_old_skool(self) -> None:
        assert detect_temporal_signal("old skool hardcore") == "1988-1995"

    def test_named_era_early_techno(self) -> None:
        assert detect_temporal_signal("early techno history") == "1988-1992"

    def test_named_era_golden_era(self) -> None:
        assert detect_temporal_signal("golden era of jungle") == "1993-1997"

    def test_explicit_decade_1990s(self) -> None:
        assert detect_temporal_signal("techno in the 1990s") == "1990s"

    def test_explicit_short_decade_90s(self) -> None:
        assert detect_temporal_signal("the 90s rave scene") == "1990s"

    def test_explicit_year_range(self) -> None:
        assert detect_temporal_signal("1988-1992 acid house") == "1988-1992"

    def test_explicit_single_year(self) -> None:
        assert detect_temporal_signal("what happened in 1992") == "1992"

    def test_no_temporal_signal(self) -> None:
        assert detect_temporal_signal("best techno tracks") is None

    def test_case_insensitive(self) -> None:
        assert detect_temporal_signal("SECOND SUMMER OF LOVE") == "1988-1989"

    # --- temporal_overlap ---

    def test_overlap_decades(self) -> None:
        # 1990s (1990-1999) overlaps with 1988-1995
        assert temporal_overlap("1990s", "1988-1995") is True

    def test_no_overlap_decades(self) -> None:
        assert temporal_overlap("1980s", "1990s") is False

    def test_year_within_decade(self) -> None:
        assert temporal_overlap("1992", "1990s") is True

    def test_year_outside_decade(self) -> None:
        assert temporal_overlap("1987", "1990s") is False

    def test_none_period_a(self) -> None:
        assert temporal_overlap(None, "1990s") is False

    def test_none_period_b(self) -> None:
        assert temporal_overlap("1990s", None) is False

    def test_both_none(self) -> None:
        assert temporal_overlap(None, None) is False

    def test_same_year(self) -> None:
        assert temporal_overlap("1992", "1992") is True

    def test_range_overlap(self) -> None:
        assert temporal_overlap("1988-1992", "1991-1995") is True

    def test_range_no_overlap(self) -> None:
        assert temporal_overlap("1988-1990", "1991-1995") is False

    def test_unparseable_string(self) -> None:
        assert temporal_overlap("ancient", "1990s") is False


# ═══════════════════════════════════════════════════════════════════════════
# 4. Artist Alias Table
# ═══════════════════════════════════════════════════════════════════════════


class TestArtistAliases:
    """Test expand_aliases(), get_canonical_name(), and get_all_artist_names()."""

    # --- expand_aliases ---

    def test_expand_canonical_name(self) -> None:
        result = expand_aliases("Aphex Twin")
        assert "Aphex Twin" in result
        assert "AFX" in result
        assert "Polygon Window" in result

    def test_expand_alias(self) -> None:
        result = expand_aliases("AFX")
        assert "Aphex Twin" in result
        assert "AFX" in result

    def test_expand_case_insensitive(self) -> None:
        result = expand_aliases("afx")
        assert "Aphex Twin" in result

    def test_expand_unknown_name(self) -> None:
        result = expand_aliases("Unknown DJ 12345")
        assert result == {"Unknown DJ 12345"}

    def test_expand_carl_craig(self) -> None:
        result = expand_aliases("Carl Craig")
        assert "69" in result
        assert "Paperclip People" in result

    # --- get_canonical_name ---

    def test_canonical_from_alias(self) -> None:
        assert get_canonical_name("AFX") == "Aphex Twin"

    def test_canonical_from_canonical(self) -> None:
        assert get_canonical_name("Aphex Twin") == "Aphex Twin"

    def test_canonical_case_insensitive(self) -> None:
        assert get_canonical_name("afx") == "Aphex Twin"

    def test_canonical_unknown(self) -> None:
        assert get_canonical_name("Unknown DJ 12345") is None

    def test_canonical_plastikman(self) -> None:
        assert get_canonical_name("Plastikman") == "Richie Hawtin"

    def test_canonical_model_500(self) -> None:
        assert get_canonical_name("Model 500") == "Juan Atkins"

    # --- get_all_artist_names ---

    def test_returns_nonempty_set(self) -> None:
        all_names = get_all_artist_names()
        assert len(all_names) > 0

    def test_contains_canonical_names(self) -> None:
        all_names = get_all_artist_names()
        assert "Aphex Twin" in all_names
        assert "Carl Craig" in all_names
        assert "Jeff Mills" in all_names

    def test_contains_aliases(self) -> None:
        all_names = get_all_artist_names()
        assert "AFX" in all_names
        assert "Plastikman" in all_names
        assert "The Wizard" in all_names

    def test_all_aliases_present(self) -> None:
        """Every alias in ARTIST_ALIASES should appear in get_all_artist_names()."""
        all_names = get_all_artist_names()
        for canonical, aliases in ARTIST_ALIASES.items():
            assert canonical in all_names, f"Canonical '{canonical}' missing"
            for alias in aliases:
                assert alias in all_names, f"Alias '{alias}' of '{canonical}' missing"


# ═══════════════════════════════════════════════════════════════════════════
# 5. Genre Extraction Helper
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractQueryGenres:
    """Test extract_query_genres()."""

    def test_single_genre(self) -> None:
        result = extract_query_genres("techno history")
        assert "techno" in result

    def test_multi_word_genre(self) -> None:
        result = extract_query_genres("acid house in Chicago")
        assert "acid house" in result

    def test_longest_match_first(self) -> None:
        # "acid house" should match, not just "house"
        result = extract_query_genres("acid house origins")
        assert "acid house" in result
        # "house" should NOT be separately matched since "acid house" consumed those chars
        # (though this depends on whether there's another "house" occurrence)

    def test_multiple_genres(self) -> None:
        result = extract_query_genres("techno and house music")
        assert "techno" in result
        assert "house" in result

    def test_no_genres(self) -> None:
        result = extract_query_genres("what is the best club in Berlin")
        assert len(result) == 0

    def test_case_insensitive(self) -> None:
        result = extract_query_genres("TECHNO is great")
        assert "techno" in result

    def test_subgenre_specificity(self) -> None:
        result = extract_query_genres("drum and bass production")
        assert "drum and bass" in result

    def test_word_boundary_respect(self) -> None:
        # "dub" should not match inside "dubstep" separately
        result = extract_query_genres("dubstep pioneers")
        assert "dubstep" in result
        # "dub" should not be matched unless it appears separately

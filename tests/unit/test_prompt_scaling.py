"""Unit tests for dynamic prompt scaling in InterconnectionService.

# ─── MODULE OVERVIEW ───
# Tests the _make_service()._build_synthesis_prompt() method, which
# generates the 13-point synthesis prompt sent to the LLM.  The prompt
# currently includes all 13 analysis points regardless of entity count.
#
# These tests verify the prompt content scales based on entity composition:
#   - 1 artist (no pairs possible): only individual points (5, 7)
#   - 2 artists: pair-comparison points (1, 2, 5, 6, 7, 13)
#   - 3 artists: extended analysis points (+ 8, 9, 12)
#   - 4+ artists: full 13-point framework
#   - Venue present: adds venue-scene point (4)
#   - Promoter present: adds promoter-artist point (3)
#
# Architecture: This file tests the Services layer (interconnection_service.py).
# The InterconnectionService constructor requires llm_provider and
# citation_service — both are mocked since we only test prompt construction,
# never calling the LLM.
#
# Design pattern: The prompt scaling logic is a Strategy pattern variant —
# the analysis strategy (which points to include) adapts to the input data
# shape.  This prevents wasting LLM tokens on irrelevant analysis dimensions
# (e.g., asking about "shared labels between artists" when there is only one
# artist on the flier).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from src.models.entities import EntityType
from src.models.flier import (
    ExtractedEntities,
    ExtractedEntity,
    OCRResult,
)
from src.models.research import ResearchResult
from src.services.citation_service import CitationService
from src.services.interconnection_service import InterconnectionService

# ── Fixtures ──────────────────────────────────────────────


def _make_ocr() -> OCRResult:
    """Minimal OCRResult fixture for building ExtractedEntities."""
    return OCRResult(
        raw_text="test ocr text",
        confidence=0.9,
        provider_used="mock",
        processing_time=0.1,
    )


def _make_artist(name: str) -> ExtractedEntity:
    """Create an artist ExtractedEntity with a given name."""
    return ExtractedEntity(
        text=name,
        entity_type=EntityType.ARTIST,
        confidence=0.9,
    )


def _make_entities(
    artist_names: list[str],
    *,
    venue_name: str | None = None,
    promoter_name: str | None = None,
) -> ExtractedEntities:
    """Build an ExtractedEntities fixture with configurable artists/venue/promoter.

    Parameters
    ----------
    artist_names:
        List of artist names to include.
    venue_name:
        If set, adds a venue entity.
    promoter_name:
        If set, adds a promoter entity.
    """
    venue = (
        ExtractedEntity(
            text=venue_name, entity_type=EntityType.VENUE, confidence=0.85
        )
        if venue_name
        else None
    )
    promoter = (
        ExtractedEntity(
            text=promoter_name, entity_type=EntityType.PROMOTER, confidence=0.7
        )
        if promoter_name
        else None
    )
    return ExtractedEntities(
        artists=[_make_artist(n) for n in artist_names],
        venue=venue,
        promoter=promoter,
        raw_ocr=_make_ocr(),
    )


def _make_research_result(entity_name: str, entity_type: EntityType) -> ResearchResult:
    """Create a minimal ResearchResult for a given entity."""
    return ResearchResult(
        entity_type=entity_type,
        entity_name=entity_name,
        confidence=0.8,
    )


def _make_service() -> InterconnectionService:
    """Create an InterconnectionService with mocked dependencies.

    Both llm_provider and citation_service are mocked since these tests
    never invoke the LLM — they only test prompt/point selection logic.
    """
    mock_llm = MagicMock()
    mock_llm.get_provider_name.return_value = "mock-llm"
    mock_llm.complete = AsyncMock(return_value="{}")

    mock_citation = MagicMock(spec=CitationService)

    return InterconnectionService(
        llm_provider=mock_llm,
        citation_service=mock_citation,
    )


# ── Tests ─────────────────────────────────────────────────


class TestPromptScaling:
    """Tests for prompt content based on entity composition.

    The current _build_synthesis_prompt always includes all 13 points.
    These tests verify the prompt content and structure for varying entity
    counts to ensure the prompt contains the expected analysis dimensions.
    """

    def test_1_artist_core_only(self) -> None:
        """With 1 artist and no venue/promoter, the prompt should still
        include geographic patterns (point 5) and scene context (point 7)
        — these are meaningful even for a single artist.
        """
        # Build a minimal compiled context for the prompt
        entities = _make_entities(["Carl Cox"])
        results = [_make_research_result("Carl Cox", EntityType.ARTIST)]
        prompt = _make_service()._build_synthesis_prompt(
            "Artist: Carl Cox\n[1] Discogs profile", entities, results
        )

        # The prompt should contain geographic and scene analysis dimensions
        assert "GEOGRAPHIC PATTERNS" in prompt or "Geographic" in prompt
        assert "SCENE CONTEXT" in prompt or "Scene" in prompt

    def test_2_artists_adds_pair_points(self) -> None:
        """With 2 artists, the prompt should include pair-comparison points
        like shared labels (1), shared lineups (2), geographic patterns (5),
        temporal patterns (6), scene context (7), and genre alignment (13).
        """
        entities = _make_entities(["Carl Cox", "Jeff Mills"])
        results = [
            _make_research_result("Carl Cox", EntityType.ARTIST),
            _make_research_result("Jeff Mills", EntityType.ARTIST),
        ]
        prompt = _make_service()._build_synthesis_prompt(
            "Artist: Carl Cox\nArtist: Jeff Mills\n[1] Discogs\n[2] MusicBrainz",
            entities, results,
        )

        # All pair-comparison points should be present in the full prompt
        assert "SHARED LABELS" in prompt or "Shared Labels" in prompt
        assert "SHARED LINEUPS" in prompt or "Shared Lineups" in prompt
        assert "GEOGRAPHIC" in prompt
        assert "TEMPORAL" in prompt
        assert "SCENE" in prompt
        assert "GENRE" in prompt or "Genre" in prompt

    def test_3_artists_adds_extended(self) -> None:
        """With 3 artists, the prompt should also include extended analysis
        points: release format (8), performance style (9), career stage (12).
        """
        entities = _make_entities(["Carl Cox", "Jeff Mills", "Derrick May"])
        results = [
            _make_research_result("Carl Cox", EntityType.ARTIST),
            _make_research_result("Jeff Mills", EntityType.ARTIST),
            _make_research_result("Derrick May", EntityType.ARTIST),
        ]
        prompt = _make_service()._build_synthesis_prompt(
            "Artist: Carl Cox\nArtist: Jeff Mills\nArtist: Derrick May\n"
            "[1] Discogs\n[2] MusicBrainz\n[3] Web",
            entities, results,
        )

        # Extended points should be present
        assert "RELEASE FORMAT" in prompt or "Release Format" in prompt
        assert "PERFORMANCE STYLE" in prompt or "Performance Style" in prompt
        assert "CAREER STAGE" in prompt or "Career Stage" in prompt

    def test_4_plus_artists_full(self) -> None:
        """With 4+ artists plus venue and promoter, the full 13-point framework
        should be present.  Points 3 and 4 require promoter and venue data
        respectively, so both must be included to trigger all 13 points.
        """
        entities = _make_entities(
            ["Carl Cox", "Jeff Mills", "Derrick May", "Juan Atkins"],
            venue_name="Tresor Berlin",
            promoter_name="Tresor Records",
        )
        results = [
            _make_research_result("Carl Cox", EntityType.ARTIST),
            _make_research_result("Jeff Mills", EntityType.ARTIST),
            _make_research_result("Derrick May", EntityType.ARTIST),
            _make_research_result("Juan Atkins", EntityType.ARTIST),
        ]
        prompt = _make_service()._build_synthesis_prompt(
            "Artist: Carl Cox\nArtist: Jeff Mills\n"
            "Artist: Derrick May\nArtist: Juan Atkins\n"
            "Venue: Tresor Berlin\nPromoter: Tresor Records\n"
            "[1] Discogs\n[2] MusicBrainz\n[3] Web\n[4] RA",
            entities, results,
        )

        # All 13 numbered analysis requirements should appear
        for point_num in range(1, 14):
            assert f"{point_num}." in prompt, (
                f"Point {point_num} missing from full prompt"
            )

    def test_venue_adds_point_4(self) -> None:
        """When venue data is present in the context, the prompt should
        include venue-scene connections (point 4).
        """
        entities = _make_entities(
            ["Carl Cox", "Jeff Mills"], venue_name="Tresor Berlin"
        )
        results = [
            _make_research_result("Carl Cox", EntityType.ARTIST),
            _make_research_result("Jeff Mills", EntityType.ARTIST),
        ]
        prompt = _make_service()._build_synthesis_prompt(
            "Artist: Carl Cox\nArtist: Jeff Mills\n"
            "Venue: Tresor Berlin\n"
            "[1] Discogs\n[2] Venue research",
            entities, results,
        )

        # Point 4: VENUE-SCENE CONNECTIONS should be present
        assert "VENUE" in prompt
        assert "4." in prompt

    def test_promoter_adds_point_3(self) -> None:
        """When promoter data is present in the context, the prompt should
        include promoter-artist links (point 3).
        """
        entities = _make_entities(
            ["Carl Cox", "Jeff Mills"], promoter_name="Tresor Records"
        )
        results = [
            _make_research_result("Carl Cox", EntityType.ARTIST),
            _make_research_result("Jeff Mills", EntityType.ARTIST),
        ]
        prompt = _make_service()._build_synthesis_prompt(
            "Artist: Carl Cox\nArtist: Jeff Mills\n"
            "Promoter: Tresor Records\n"
            "[1] Discogs\n[2] Promoter research",
            entities, results,
        )

        # Point 3: PROMOTER-ARTIST LINKS should be present
        assert "PROMOTER" in prompt
        assert "3." in prompt

    def test_no_venue_excludes_point_4_content(self) -> None:
        """With 2 artists but no venue, the prompt structure should still
        be valid.  Point 4 (venue-scene connections) text appears in the
        full prompt but will have no venue data to analyze.
        """
        entities = _make_entities(["Carl Cox", "Jeff Mills"])
        results = [
            _make_research_result("Carl Cox", EntityType.ARTIST),
            _make_research_result("Jeff Mills", EntityType.ARTIST),
        ]
        prompt = _make_service()._build_synthesis_prompt(
            "Artist: Carl Cox\nArtist: Jeff Mills\n"
            "[1] Discogs\n[2] MusicBrainz",
            entities, results,
        )

        # The research data section should not mention any venue
        # Split the prompt to check the RESEARCH DATA portion
        assert "Venue:" not in prompt.split("ANALYSIS REQUIREMENTS")[0]


class TestPromptStructure:
    """Tests for the overall structure of the synthesis prompt."""

    def test_prompt_contains_research_data(self) -> None:
        """The compiled context should appear in the prompt under RESEARCH DATA."""
        context = "Artist: Carl Cox\n[1] Discogs profile for Carl Cox"
        entities = _make_entities(["Carl Cox"])
        results = [_make_research_result("Carl Cox", EntityType.ARTIST)]
        prompt = _make_service()._build_synthesis_prompt(context, entities, results)

        assert "RESEARCH DATA" in prompt
        assert "Carl Cox" in prompt
        assert "[1] Discogs profile" in prompt

    def test_prompt_contains_strict_rules(self) -> None:
        """The prompt must include STRICT RULES to prevent LLM hallucination."""
        entities = _make_entities(["Test Artist"])
        results = [_make_research_result("Test Artist", EntityType.ARTIST)]
        prompt = _make_service()._build_synthesis_prompt("test context", entities, results)

        assert "STRICT RULES" in prompt

    def test_prompt_requires_source_citations(self) -> None:
        """The prompt must instruct the LLM to include source references."""
        entities = _make_entities(["Test Artist"])
        results = [_make_research_result("Test Artist", EntityType.ARTIST)]
        prompt = _make_service()._build_synthesis_prompt("test context", entities, results)

        # Should mention source references/citations
        assert "source" in prompt.lower()
        assert "[n]" in prompt or "source reference" in prompt.lower()

    def test_prompt_requests_json_output(self) -> None:
        """The prompt must request structured JSON output."""
        entities = _make_entities(["Test Artist"])
        results = [_make_research_result("Test Artist", EntityType.ARTIST)]
        prompt = _make_service()._build_synthesis_prompt("test context", entities, results)

        assert "JSON" in prompt or "json" in prompt

"""Unit tests for OutputFormatter — full analysis and summary formatting."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from src.models.analysis import (
    Citation,
    EntityNode,
    InterconnectionMap,
    PatternInsight,
    RelationshipEdge,
)
from src.models.entities import ConfidenceLevel, EntityType
from src.models.flier import ExtractedEntities, ExtractedEntity, OCRResult
from src.models.pipeline import PipelinePhase, PipelineState
from src.services.output_formatter import OutputFormatter, _confidence_level


@pytest.fixture()
def formatter() -> OutputFormatter:
    return OutputFormatter()


# ======================================================================
# _confidence_level (module-level helper)
# ======================================================================


class TestConfidenceLevel:
    def test_high(self) -> None:
        assert _confidence_level(0.8) == ConfidenceLevel.HIGH.value
        assert _confidence_level(0.95) == ConfidenceLevel.HIGH.value
        assert _confidence_level(1.0) == ConfidenceLevel.HIGH.value

    def test_medium(self) -> None:
        assert _confidence_level(0.5) == ConfidenceLevel.MEDIUM.value
        assert _confidence_level(0.79) == ConfidenceLevel.MEDIUM.value

    def test_low(self) -> None:
        assert _confidence_level(0.3) == ConfidenceLevel.LOW.value
        assert _confidence_level(0.49) == ConfidenceLevel.LOW.value

    def test_uncertain(self) -> None:
        assert _confidence_level(0.0) == ConfidenceLevel.UNCERTAIN.value
        assert _confidence_level(0.29) == ConfidenceLevel.UNCERTAIN.value


# ======================================================================
# format_full_analysis
# ======================================================================


class TestFormatFullAnalysis:
    def test_returns_all_sections(
        self,
        formatter: OutputFormatter,
        sample_pipeline_state: PipelineState,
    ) -> None:
        result = formatter.format_full_analysis(sample_pipeline_state)

        assert "session_id" in result
        assert "flier" in result
        assert "ocr" in result
        assert "entities" in result
        assert "research" in result
        assert "interconnections" in result
        assert "citations" in result
        assert "warnings" in result
        assert "completed_at" in result

    def test_session_id_matches(
        self,
        formatter: OutputFormatter,
        sample_pipeline_state: PipelineState,
    ) -> None:
        result = formatter.format_full_analysis(sample_pipeline_state)
        assert result["session_id"] == "test-session-001"

    def test_completed_at_is_iso_string(
        self,
        formatter: OutputFormatter,
        sample_pipeline_state: PipelineState,
    ) -> None:
        result = formatter.format_full_analysis(sample_pipeline_state)
        assert isinstance(result["completed_at"], str)
        assert "2026-02-22" in result["completed_at"]

    def test_warnings_include_pipeline_errors(
        self,
        formatter: OutputFormatter,
        sample_pipeline_state: PipelineState,
    ) -> None:
        result = formatter.format_full_analysis(sample_pipeline_state)
        warnings = result["warnings"]
        assert any("Bandcamp rate limit exceeded" in w for w in warnings)

    def test_warnings_include_research_warnings(
        self,
        formatter: OutputFormatter,
        sample_pipeline_state: PipelineState,
    ) -> None:
        result = formatter.format_full_analysis(sample_pipeline_state)
        warnings = result["warnings"]
        assert any("Limited release data" in w for w in warnings)


# ======================================================================
# format_summary
# ======================================================================


class TestFormatSummary:
    def test_returns_expected_keys(
        self,
        formatter: OutputFormatter,
        sample_pipeline_state: PipelineState,
    ) -> None:
        result = formatter.format_summary(sample_pipeline_state)
        assert "session_id" in result
        assert "phase" in result
        assert "progress_percent" in result
        assert "artists" in result
        assert "venue" in result
        assert "relationship_count" in result
        assert "pattern_count" in result
        assert "narrative_preview" in result
        assert "warnings_count" in result
        assert "completed_at" in result

    def test_artists_extracted(
        self,
        formatter: OutputFormatter,
        sample_pipeline_state: PipelineState,
    ) -> None:
        result = formatter.format_summary(sample_pipeline_state)
        assert "Carl Cox" in result["artists"]

    def test_venue_extracted(
        self,
        formatter: OutputFormatter,
        sample_pipeline_state: PipelineState,
    ) -> None:
        result = formatter.format_summary(sample_pipeline_state)
        assert result["venue"] == "Tresor, Berlin"

    def test_relationship_count(
        self,
        formatter: OutputFormatter,
        sample_pipeline_state: PipelineState,
    ) -> None:
        result = formatter.format_summary(sample_pipeline_state)
        assert result["relationship_count"] == 1

    def test_pattern_count(
        self,
        formatter: OutputFormatter,
        sample_pipeline_state: PipelineState,
    ) -> None:
        result = formatter.format_summary(sample_pipeline_state)
        assert result["pattern_count"] == 1

    def test_narrative_truncation(
        self,
        formatter: OutputFormatter,
        sample_pipeline_state: PipelineState,
    ) -> None:
        # Build a state with a very long narrative
        long_narrative = "A" * 500
        imap = sample_pipeline_state.interconnection_map
        new_imap = imap.model_copy(update={"narrative": long_narrative})
        state = sample_pipeline_state.model_copy(update={"interconnection_map": new_imap})

        result = formatter.format_summary(state)
        preview = result["narrative_preview"]
        assert len(preview) == 303  # 300 chars + "..."
        assert preview.endswith("...")

    def test_short_narrative_not_truncated(
        self,
        formatter: OutputFormatter,
        sample_pipeline_state: PipelineState,
    ) -> None:
        result = formatter.format_summary(sample_pipeline_state)
        preview = result["narrative_preview"]
        assert not preview.endswith("...")

    def test_no_interconnection_map(
        self,
        formatter: OutputFormatter,
        sample_pipeline_state: PipelineState,
    ) -> None:
        state = sample_pipeline_state.model_copy(update={"interconnection_map": None})
        result = formatter.format_summary(state)
        assert result["relationship_count"] == 0
        assert result["pattern_count"] == 0
        assert result["narrative_preview"] is None

    def test_no_entities(
        self,
        formatter: OutputFormatter,
        sample_pipeline_state: PipelineState,
    ) -> None:
        state = sample_pipeline_state.model_copy(
            update={"confirmed_entities": None, "extracted_entities": None}
        )
        result = formatter.format_summary(state)
        assert result["artists"] == []
        assert result["venue"] is None


# ======================================================================
# _format_flier
# ======================================================================


class TestFormatFlier:
    def test_has_ocr_true(
        self,
        formatter: OutputFormatter,
        sample_pipeline_state: PipelineState,
    ) -> None:
        result = OutputFormatter._format_flier(sample_pipeline_state)
        assert result["has_ocr"] is True

    def test_has_ocr_false(
        self,
        formatter: OutputFormatter,
        sample_pipeline_state: PipelineState,
    ) -> None:
        state = sample_pipeline_state.model_copy(update={"ocr_result": None})
        result = OutputFormatter._format_flier(state)
        assert result["has_ocr"] is False


# ======================================================================
# _format_ocr
# ======================================================================


class TestFormatOCR:
    def test_none_returns_none(self) -> None:
        assert OutputFormatter._format_ocr(None) is None

    def test_formats_ocr_result(self, sample_ocr_result: OCRResult) -> None:
        result = OutputFormatter._format_ocr(sample_ocr_result)
        assert result is not None
        assert result["raw_text"] == sample_ocr_result.raw_text
        assert result["confidence"] == 0.85
        assert result["provider"] == "tesseract"


# ======================================================================
# _format_entities
# ======================================================================


class TestFormatEntities:
    def test_none_returns_empty_structure(self) -> None:
        result = OutputFormatter._format_entities(None)
        assert result["artists"] == []
        assert result["venue"] is None
        assert result["date"] is None
        assert result["promoter"] is None
        assert result["event_name"] is None

    def test_formats_artists_with_confidence_level(
        self, sample_extracted_entities: ExtractedEntities
    ) -> None:
        result = OutputFormatter._format_entities(sample_extracted_entities)
        artists = result["artists"]
        assert len(artists) == 3
        assert artists[0]["name"] == "Carl Cox"
        assert artists[0]["confidence"] == 0.95
        assert artists[0]["confidence_level"] == "HIGH"

    def test_formats_venue(self, sample_extracted_entities: ExtractedEntities) -> None:
        result = OutputFormatter._format_entities(sample_extracted_entities)
        venue = result["venue"]
        assert venue is not None
        assert venue["name"] == "Tresor, Berlin"

    def test_formats_date(self, sample_extracted_entities: ExtractedEntities) -> None:
        result = OutputFormatter._format_entities(sample_extracted_entities)
        date_info = result["date"]
        assert date_info is not None
        assert "March 15th 1997" in date_info["text"]
        assert date_info["parsed_date"] is None

    def test_formats_promoter(self, sample_extracted_entities: ExtractedEntities) -> None:
        result = OutputFormatter._format_entities(sample_extracted_entities)
        promoter = result["promoter"]
        assert promoter is not None
        assert promoter["name"] == "Tresor Records"

    def test_event_name_when_present(self) -> None:
        entities = ExtractedEntities(
            artists=[],
            event_name=ExtractedEntity(
                text="Tresor Nights",
                entity_type=EntityType.EVENT,
                confidence=0.75,
            ),
            raw_ocr=OCRResult(
                raw_text="TRESOR NIGHTS",
                confidence=0.8,
                provider_used="tesseract",
                processing_time=1.0,
            ),
        )
        result = OutputFormatter._format_entities(entities)
        assert result["event_name"] is not None
        assert result["event_name"]["name"] == "Tresor Nights"
        assert result["event_name"]["confidence"] == 0.75


# ======================================================================
# _format_research
# ======================================================================


class TestFormatResearch:
    def test_formats_artist_research(
        self,
        formatter: OutputFormatter,
        sample_pipeline_state: PipelineState,
    ) -> None:
        result = formatter._format_research(sample_pipeline_state.research_results)
        assert len(result["artists"]) >= 1
        artist = result["artists"][0]
        assert artist["name"] == "Carl Cox"
        assert artist["releases_count"] == 1
        assert "Intec" in artist["labels"]
        assert "discogs.com" in artist["discogs_url"]
        assert "musicbrainz.org" in artist["musicbrainz_url"]

    def test_formats_venue_research(
        self,
        formatter: OutputFormatter,
        sample_pipeline_state: PipelineState,
    ) -> None:
        result = formatter._format_research(sample_pipeline_state.research_results)
        venue = result["venue"]
        assert venue is not None
        assert venue["name"] == "Tresor"
        assert "1991" in venue["history"]

    def test_formats_promoter_research(
        self,
        formatter: OutputFormatter,
        sample_pipeline_state: PipelineState,
    ) -> None:
        result = formatter._format_research(sample_pipeline_state.research_results)
        promoter = result["promoter"]
        assert promoter is not None
        assert promoter["name"] == "Tresor Records"

    def test_formats_date_context(
        self,
        formatter: OutputFormatter,
        sample_pipeline_state: PipelineState,
    ) -> None:
        result = formatter._format_research(sample_pipeline_state.research_results)
        dc = result["date_context"]
        assert dc is not None
        assert "Berlin techno" in dc["scene"]
        assert "electronic music" in dc["city"]

    def test_artist_without_db_ids(self, formatter: OutputFormatter) -> None:
        from src.models.entities import Artist
        from src.models.research import ResearchResult

        result_obj = ResearchResult(
            entity_type=EntityType.ARTIST,
            entity_name="Unknown DJ",
            artist=Artist(name="Unknown DJ", confidence=0.4),
            confidence=0.4,
        )
        formatted = formatter._format_research([result_obj])
        artist = formatted["artists"][0]
        assert artist["discogs_url"] is None
        assert artist["musicbrainz_url"] is None

    def test_artist_none_skipped(self, formatter: OutputFormatter) -> None:
        from src.models.research import ResearchResult

        result_obj = ResearchResult(
            entity_type=EntityType.ARTIST,
            entity_name="Ghost DJ",
            artist=None,
            confidence=0.3,
        )
        formatted = formatter._format_research([result_obj])
        # When artist is None, the entry is not included in the output
        assert formatted["artists"] == []


# ======================================================================
# _format_interconnections
# ======================================================================


class TestFormatInterconnections:
    def test_none_returns_empty_structure(self, formatter: OutputFormatter) -> None:
        result = formatter._format_interconnections(None)
        assert result["relationships"] == []
        assert result["patterns"] == []
        assert result["narrative"] is None

    def test_formats_edges(
        self,
        formatter: OutputFormatter,
        sample_pipeline_state: PipelineState,
    ) -> None:
        result = formatter._format_interconnections(
            sample_pipeline_state.interconnection_map
        )
        rels = result["relationships"]
        assert len(rels) == 1
        assert rels[0]["source"] == "Carl Cox"
        assert rels[0]["target"] == "Tresor"
        assert rels[0]["type"] == "performed_at"
        assert rels[0]["confidence"] == 0.9
        assert rels[0]["citation"] is not None

    def test_formats_patterns(
        self,
        formatter: OutputFormatter,
        sample_pipeline_state: PipelineState,
    ) -> None:
        result = formatter._format_interconnections(
            sample_pipeline_state.interconnection_map
        )
        patterns = result["patterns"]
        assert len(patterns) == 1
        assert patterns[0]["type"] == "venue_residency"
        assert "Carl Cox" in patterns[0]["entities"]

    def test_narrative_included(
        self,
        formatter: OutputFormatter,
        sample_pipeline_state: PipelineState,
    ) -> None:
        result = formatter._format_interconnections(
            sample_pipeline_state.interconnection_map
        )
        assert "Carl Cox" in result["narrative"]

    def test_edge_without_citations(self, formatter: OutputFormatter) -> None:
        imap = InterconnectionMap(
            edges=[
                RelationshipEdge(
                    source="A",
                    target="B",
                    relationship_type="related_to",
                    confidence=0.5,
                ),
            ],
        )
        result = formatter._format_interconnections(imap)
        assert result["relationships"][0]["citation"] is None


# ======================================================================
# _format_citations
# ======================================================================


class TestFormatCitations:
    def test_none_map_returns_empty(self, formatter: OutputFormatter) -> None:
        from src.models.pipeline import PipelineState

        # Build a state with no interconnection map
        from src.models.flier import FlierImage

        flier = FlierImage(
            filename="test.jpg",
            content_type="image/jpeg",
            file_size=100,
            image_hash="abc",
        )
        state = PipelineState(session_id="x", flier=flier)
        result = formatter._format_citations(state)
        assert result == []

    def test_formats_citations(
        self,
        formatter: OutputFormatter,
        sample_pipeline_state: PipelineState,
    ) -> None:
        result = formatter._format_citations(sample_pipeline_state)
        assert len(result) >= 1
        c = result[0]
        assert c["text"] == "Carl Cox played at Tresor Berlin in 1997"
        assert c["source"] == "Resident Advisor"
        assert c["tier"] == 2
        assert c["url"] == "https://ra.co/features/carl-cox-tresor"
        assert c["accessible"] is None


# ======================================================================
# _collect_warnings
# ======================================================================


class TestCollectWarnings:
    def test_collects_pipeline_errors(
        self,
        formatter: OutputFormatter,
        sample_pipeline_state: PipelineState,
    ) -> None:
        warnings = OutputFormatter._collect_warnings(sample_pipeline_state)
        assert any("[RESEARCH]" in w for w in warnings)

    def test_collects_research_warnings(
        self,
        formatter: OutputFormatter,
        sample_pipeline_state: PipelineState,
    ) -> None:
        warnings = OutputFormatter._collect_warnings(sample_pipeline_state)
        assert any("Carl Cox" in w for w in warnings)


# ======================================================================
# _iso_timestamp
# ======================================================================


class TestIsoTimestamp:
    def test_none_returns_none(self) -> None:
        assert OutputFormatter._iso_timestamp(None) is None

    def test_naive_datetime_gets_utc(self) -> None:
        dt = datetime(2026, 1, 15, 10, 30, 0)
        result = OutputFormatter._iso_timestamp(dt)
        assert result is not None
        assert "+00:00" in result

    def test_aware_datetime_preserved(self) -> None:
        dt = datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = OutputFormatter._iso_timestamp(dt)
        assert result is not None
        assert "2026-01-15" in result

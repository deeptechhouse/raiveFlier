"""Comprehensive unit tests for all Pydantic domain models."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from src.models.analysis import (
    Citation,
    EntityNode,
    InterconnectionMap,
    PatternInsight,
    RelationshipEdge,
)
from src.models.entities import (
    ArticleReference,
    Artist,
    ConfidenceLevel,
    EntityType,
    EventAppearance,
    Label,
    Promoter,
    Release,
    Venue,
)
from src.models.flier import (
    ExtractedEntities,
    ExtractedEntity,
    FlierImage,
    OCRResult,
    TextRegion,
)
from src.models.pipeline import (
    PipelineError,
    PipelinePhase,
    PipelineState,
)
from src.models.research import DateContext, ResearchResult


# ======================================================================
# Artist
# ======================================================================


class TestArtist:
    """Tests for the Artist model and its confidence_level property."""

    def test_create_with_valid_data(self) -> None:
        artist = Artist(name="Carl Cox", confidence=0.95)
        assert artist.name == "Carl Cox"
        assert artist.confidence == 0.95

    def test_confidence_level_high(self) -> None:
        artist = Artist(name="A", confidence=0.9)
        assert artist.confidence_level == ConfidenceLevel.HIGH

    def test_confidence_level_high_boundary(self) -> None:
        artist = Artist(name="A", confidence=0.8)
        assert artist.confidence_level == ConfidenceLevel.HIGH

    def test_confidence_level_medium(self) -> None:
        artist = Artist(name="A", confidence=0.6)
        assert artist.confidence_level == ConfidenceLevel.MEDIUM

    def test_confidence_level_medium_boundary(self) -> None:
        artist = Artist(name="A", confidence=0.5)
        assert artist.confidence_level == ConfidenceLevel.MEDIUM

    def test_confidence_level_low(self) -> None:
        artist = Artist(name="A", confidence=0.4)
        assert artist.confidence_level == ConfidenceLevel.LOW

    def test_confidence_level_low_boundary(self) -> None:
        artist = Artist(name="A", confidence=0.3)
        assert artist.confidence_level == ConfidenceLevel.LOW

    def test_confidence_level_uncertain(self) -> None:
        artist = Artist(name="A", confidence=0.1)
        assert artist.confidence_level == ConfidenceLevel.UNCERTAIN

    def test_confidence_level_zero(self) -> None:
        artist = Artist(name="A", confidence=0.0)
        assert artist.confidence_level == ConfidenceLevel.UNCERTAIN

    def test_confidence_rejects_above_one(self) -> None:
        with pytest.raises(ValidationError):
            Artist(name="A", confidence=1.1)

    def test_confidence_rejects_below_zero(self) -> None:
        with pytest.raises(ValidationError):
            Artist(name="A", confidence=-0.1)

    def test_default_factory_lists_not_shared(self) -> None:
        a1 = Artist(name="A")
        a2 = Artist(name="B")
        assert a1.releases is not a2.releases
        assert a1.labels is not a2.labels

    def test_default_factory_lists_are_empty(self) -> None:
        artist = Artist(name="A")
        assert artist.releases == []
        assert artist.labels == []
        assert artist.aliases == []

    def test_model_dump_json_valid(self) -> None:
        artist = Artist(name="Jeff Mills", confidence=0.85)
        raw = artist.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["name"] == "Jeff Mills"
        assert parsed["confidence"] == 0.85

    def test_frozen_immutability(self) -> None:
        artist = Artist(name="A", confidence=0.5)
        with pytest.raises(ValidationError):
            artist.name = "B"  # type: ignore[misc]


# ======================================================================
# Release
# ======================================================================


class TestRelease:
    """Tests for the Release model."""

    def test_create_with_valid_data(self) -> None:
        release = Release(title="Strings of Life", label="Transmat")
        assert release.title == "Strings of Life"
        assert release.label == "Transmat"

    def test_optional_fields_default_none(self) -> None:
        release = Release(title="T", label="L")
        assert release.catalog_number is None
        assert release.year is None
        assert release.format is None
        assert release.discogs_url is None

    def test_default_factory_lists(self) -> None:
        r1 = Release(title="T", label="L")
        r2 = Release(title="T2", label="L2")
        assert r1.genres == []
        assert r1.styles == []
        assert r1.genres is not r2.genres

    def test_model_dump_json_valid(self) -> None:
        release = Release(title="T", label="L", year=1988)
        parsed = json.loads(release.model_dump_json())
        assert parsed["year"] == 1988


# ======================================================================
# Label
# ======================================================================


class TestLabel:
    """Tests for the Label model."""

    def test_create_with_valid_data(self) -> None:
        label = Label(name="Warp Records", discogs_id=123)
        assert label.name == "Warp Records"
        assert label.discogs_id == 123

    def test_optional_fields(self) -> None:
        label = Label(name="L")
        assert label.discogs_id is None
        assert label.discogs_url is None

    def test_model_dump_json_valid(self) -> None:
        label = Label(name="L")
        parsed = json.loads(label.model_dump_json())
        assert parsed["name"] == "L"


# ======================================================================
# EventAppearance
# ======================================================================


class TestEventAppearance:
    """Tests for the EventAppearance model."""

    def test_create_with_valid_data(self) -> None:
        ea = EventAppearance(
            event_name="Tresor Night",
            venue="Tresor Berlin",
            date=date(1997, 6, 15),
            source="RA",
            source_url="https://ra.co/events/123",
        )
        assert ea.event_name == "Tresor Night"
        assert ea.date == date(1997, 6, 15)

    def test_all_optional(self) -> None:
        ea = EventAppearance()
        assert ea.event_name is None
        assert ea.venue is None
        assert ea.date is None
        assert ea.source is None
        assert ea.source_url is None

    def test_model_dump_json_valid(self) -> None:
        ea = EventAppearance(event_name="E")
        parsed = json.loads(ea.model_dump_json())
        assert parsed["event_name"] == "E"


# ======================================================================
# ArticleReference
# ======================================================================


class TestArticleReference:
    """Tests for the ArticleReference model with citation tier validation."""

    def test_create_with_valid_data(self) -> None:
        ar = ArticleReference(title="Interview", source="DJ Mag", citation_tier=2)
        assert ar.title == "Interview"
        assert ar.citation_tier == 2

    @pytest.mark.parametrize("tier", [1, 2, 3, 4, 5, 6])
    def test_valid_citation_tiers(self, tier: int) -> None:
        ar = ArticleReference(title="T", source="S", citation_tier=tier)
        assert ar.citation_tier == tier

    def test_rejects_tier_zero(self) -> None:
        with pytest.raises(ValidationError):
            ArticleReference(title="T", source="S", citation_tier=0)

    def test_rejects_tier_seven(self) -> None:
        with pytest.raises(ValidationError):
            ArticleReference(title="T", source="S", citation_tier=7)

    def test_rejects_negative_tier(self) -> None:
        with pytest.raises(ValidationError):
            ArticleReference(title="T", source="S", citation_tier=-1)

    def test_default_tier_is_six(self) -> None:
        ar = ArticleReference(title="T", source="S")
        assert ar.citation_tier == 6

    def test_model_dump_json_valid(self) -> None:
        ar = ArticleReference(title="T", source="S", citation_tier=1)
        parsed = json.loads(ar.model_dump_json())
        assert parsed["citation_tier"] == 1


# ======================================================================
# FlierImage
# ======================================================================


class TestFlierImage:
    """Tests for the FlierImage model."""

    def test_create_with_sha256_hash(self) -> None:
        sha = "a" * 64
        img = FlierImage(
            filename="flier.jpg",
            content_type="image/jpeg",
            file_size=1024,
            image_hash=sha,
        )
        assert img.image_hash == sha
        assert img.filename == "flier.jpg"
        assert img.file_size == 1024
        assert img.content_type == "image/jpeg"

    def test_auto_generated_id(self) -> None:
        img = FlierImage(
            filename="f.png",
            content_type="image/png",
            file_size=100,
            image_hash="b" * 64,
        )
        assert img.id is not None
        assert len(img.id) > 0

    def test_unique_ids(self) -> None:
        kwargs = dict(
            filename="f.png",
            content_type="image/png",
            file_size=100,
            image_hash="c" * 64,
        )
        img1 = FlierImage(**kwargs)
        img2 = FlierImage(**kwargs)
        assert img1.id != img2.id

    def test_upload_timestamp_auto(self) -> None:
        img = FlierImage(
            filename="f.png",
            content_type="image/png",
            file_size=100,
            image_hash="d" * 64,
        )
        assert img.upload_timestamp is not None
        assert img.upload_timestamp.tzinfo is not None

    def test_image_data_private_attr(self) -> None:
        img = FlierImage(
            filename="f.png",
            content_type="image/png",
            file_size=100,
            image_hash="e" * 64,
        )
        assert img.image_data is None

    def test_model_dump_json_valid(self) -> None:
        img = FlierImage(
            filename="f.png",
            content_type="image/png",
            file_size=100,
            image_hash="f" * 64,
        )
        parsed = json.loads(img.model_dump_json())
        assert parsed["filename"] == "f.png"
        assert "_image_data" not in parsed


# ======================================================================
# TextRegion
# ======================================================================


class TestTextRegion:
    """Tests for the TextRegion model."""

    def test_create_with_valid_data(self) -> None:
        tr = TextRegion(text="TECHNO", confidence=0.95, x=10, y=20, width=100, height=30)
        assert tr.text == "TECHNO"
        assert tr.confidence == 0.95

    def test_confidence_validation(self) -> None:
        with pytest.raises(ValidationError):
            TextRegion(text="T", confidence=1.5, x=0, y=0, width=1, height=1)

    def test_model_dump_json_valid(self) -> None:
        tr = TextRegion(text="T", confidence=0.5, x=0, y=0, width=1, height=1)
        parsed = json.loads(tr.model_dump_json())
        assert parsed["text"] == "T"


# ======================================================================
# OCRResult
# ======================================================================


class TestOCRResult:
    """Tests for the OCRResult model."""

    def test_create_with_valid_data(self) -> None:
        ocr = OCRResult(
            raw_text="CARL COX",
            confidence=0.85,
            provider_used="tesseract",
            processing_time=1.23,
        )
        assert ocr.raw_text == "CARL COX"
        assert ocr.provider_used == "tesseract"

    def test_default_bounding_boxes_empty(self) -> None:
        ocr = OCRResult(raw_text="", confidence=0.0, provider_used="t", processing_time=0.0)
        assert ocr.bounding_boxes == []

    def test_bounding_boxes_not_shared(self) -> None:
        o1 = OCRResult(raw_text="", confidence=0.0, provider_used="t", processing_time=0.0)
        o2 = OCRResult(raw_text="", confidence=0.0, provider_used="t", processing_time=0.0)
        assert o1.bounding_boxes is not o2.bounding_boxes

    def test_model_dump_json_valid(self) -> None:
        ocr = OCRResult(raw_text="X", confidence=0.5, provider_used="t", processing_time=0.1)
        parsed = json.loads(ocr.model_dump_json())
        assert parsed["raw_text"] == "X"


# ======================================================================
# ExtractedEntity & ExtractedEntities
# ======================================================================


class TestExtractedEntities:
    """Tests for ExtractedEntity and ExtractedEntities models."""

    def test_extracted_entity_creation(self) -> None:
        ee = ExtractedEntity(
            text="Carl Cox",
            entity_type=EntityType.ARTIST,
            confidence=0.9,
        )
        assert ee.text == "Carl Cox"
        assert ee.entity_type == EntityType.ARTIST

    def test_extracted_entities_with_all_fields(self) -> None:
        ocr = OCRResult(raw_text="text", confidence=0.9, provider_used="t", processing_time=0.1)
        artist = ExtractedEntity(text="A", entity_type=EntityType.ARTIST, confidence=0.9)
        venue = ExtractedEntity(text="V", entity_type=EntityType.VENUE, confidence=0.8)
        entities = ExtractedEntities(
            artists=[artist],
            venue=venue,
            genre_tags=["techno"],
            ticket_price="$10",
            raw_ocr=ocr,
        )
        assert len(entities.artists) == 1
        assert entities.venue is not None
        assert entities.genre_tags == ["techno"]
        assert entities.ticket_price == "$10"

    def test_default_factory_lists(self) -> None:
        ocr = OCRResult(raw_text="", confidence=0.0, provider_used="t", processing_time=0.0)
        e1 = ExtractedEntities(raw_ocr=ocr)
        e2 = ExtractedEntities(raw_ocr=ocr)
        assert e1.artists == []
        assert e1.genre_tags == []
        assert e1.supporting_text == []
        assert e1.artists is not e2.artists

    def test_model_dump_json_valid(self) -> None:
        ocr = OCRResult(raw_text="", confidence=0.0, provider_used="t", processing_time=0.0)
        entities = ExtractedEntities(raw_ocr=ocr)
        parsed = json.loads(entities.model_dump_json())
        assert "artists" in parsed
        assert "raw_ocr" in parsed


# ======================================================================
# Citation
# ======================================================================


class TestCitation:
    """Tests for the Citation model with tier validation."""

    def test_create_with_valid_data(self) -> None:
        c = Citation(
            text="Carl Cox played Tresor",
            source_type="press",
            source_name="Resident Advisor",
            tier=2,
        )
        assert c.tier == 2
        assert c.text == "Carl Cox played Tresor"

    @pytest.mark.parametrize("tier", [1, 2, 3, 4, 5, 6])
    def test_valid_tiers(self, tier: int) -> None:
        c = Citation(text="T", source_type="s", source_name="n", tier=tier)
        assert c.tier == tier

    def test_rejects_tier_zero(self) -> None:
        with pytest.raises(ValidationError):
            Citation(text="T", source_type="s", source_name="n", tier=0)

    def test_rejects_tier_seven(self) -> None:
        with pytest.raises(ValidationError):
            Citation(text="T", source_type="s", source_name="n", tier=7)

    def test_default_tier_is_six(self) -> None:
        c = Citation(text="T", source_type="s", source_name="n")
        assert c.tier == 6

    def test_with_source_date(self) -> None:
        c = Citation(
            text="T",
            source_type="s",
            source_name="n",
            source_date=date(2024, 1, 15),
            tier=1,
        )
        assert c.source_date == date(2024, 1, 15)

    def test_model_dump_json_valid(self) -> None:
        c = Citation(text="T", source_type="s", source_name="n", tier=3)
        parsed = json.loads(c.model_dump_json())
        assert parsed["tier"] == 3


# ======================================================================
# InterconnectionMap, EntityNode, RelationshipEdge, PatternInsight
# ======================================================================


class TestInterconnectionMap:
    """Tests for the interconnection analysis models."""

    def test_entity_node_creation(self) -> None:
        node = EntityNode(entity_type=EntityType.ARTIST, name="Jeff Mills")
        assert node.name == "Jeff Mills"
        assert node.properties == {}

    def test_relationship_edge_creation(self) -> None:
        edge = RelationshipEdge(
            source="Carl Cox",
            target="Space Ibiza",
            relationship_type="residency",
            confidence=0.9,
        )
        assert edge.source == "Carl Cox"
        assert edge.confidence == 0.9

    def test_relationship_edge_confidence_validation(self) -> None:
        with pytest.raises(ValidationError):
            RelationshipEdge(
                source="A",
                target="B",
                relationship_type="r",
                confidence=1.5,
            )

    def test_pattern_insight_creation(self) -> None:
        pi = PatternInsight(
            pattern_type="label_cluster",
            description="Artists share label",
            involved_entities=["A", "B"],
        )
        assert pi.pattern_type == "label_cluster"
        assert len(pi.involved_entities) == 2

    def test_interconnection_map_with_nodes_edges_patterns(self) -> None:
        node = EntityNode(entity_type=EntityType.ARTIST, name="A")
        edge = RelationshipEdge(source="A", target="B", relationship_type="collab", confidence=0.8)
        pattern = PatternInsight(pattern_type="p", description="d")
        citation = Citation(text="c", source_type="s", source_name="n", tier=2)
        imap = InterconnectionMap(
            nodes=[node],
            edges=[edge],
            patterns=[pattern],
            narrative="Test narrative",
            citations=[citation],
        )
        assert len(imap.nodes) == 1
        assert len(imap.edges) == 1
        assert len(imap.patterns) == 1
        assert imap.narrative == "Test narrative"
        assert len(imap.citations) == 1

    def test_default_factory_lists_not_shared(self) -> None:
        m1 = InterconnectionMap()
        m2 = InterconnectionMap()
        assert m1.nodes is not m2.nodes
        assert m1.edges is not m2.edges
        assert m1.patterns is not m2.patterns
        assert m1.citations is not m2.citations

    def test_model_dump_json_valid(self) -> None:
        imap = InterconnectionMap(narrative="N")
        parsed = json.loads(imap.model_dump_json())
        assert parsed["narrative"] == "N"
        assert parsed["nodes"] == []


# ======================================================================
# PipelineState & PipelinePhase
# ======================================================================


class TestPipelineState:
    """Tests for PipelineState phase transitions and validation."""

    @pytest.fixture()
    def base_flier(self) -> FlierImage:
        return FlierImage(
            filename="test.jpg",
            content_type="image/jpeg",
            file_size=500,
            image_hash="a" * 64,
        )

    def test_create_initial_state(self, base_flier: FlierImage) -> None:
        state = PipelineState(session_id="s1", flier=base_flier)
        assert state.current_phase == PipelinePhase.UPLOAD
        assert state.progress_percent == 0.0
        assert state.errors == []
        assert state.research_results == []

    def test_phase_transition_via_model_copy(self, base_flier: FlierImage) -> None:
        state = PipelineState(session_id="s1", flier=base_flier)
        new_state = state.model_copy(update={"current_phase": PipelinePhase.OCR})
        assert new_state.current_phase == PipelinePhase.OCR
        assert state.current_phase == PipelinePhase.UPLOAD  # original unchanged

    def test_all_pipeline_phases(self) -> None:
        expected = {"UPLOAD", "OCR", "ENTITY_EXTRACTION", "USER_CONFIRMATION", "RESEARCH", "INTERCONNECTION", "OUTPUT"}
        assert {p.value for p in PipelinePhase} == expected

    def test_progress_percent_validation(self, base_flier: FlierImage) -> None:
        with pytest.raises(ValidationError):
            PipelineState(session_id="s1", flier=base_flier, progress_percent=101.0)

    def test_progress_percent_lower_bound(self, base_flier: FlierImage) -> None:
        with pytest.raises(ValidationError):
            PipelineState(session_id="s1", flier=base_flier, progress_percent=-1.0)

    def test_pipeline_error_model(self, base_flier: FlierImage) -> None:
        error = PipelineError(phase=PipelinePhase.OCR, message="OCR failed")
        assert error.phase == PipelinePhase.OCR
        assert error.recoverable is True
        assert error.timestamp.tzinfo is not None

    def test_default_factory_lists_not_shared(self, base_flier: FlierImage) -> None:
        s1 = PipelineState(session_id="s1", flier=base_flier)
        s2 = PipelineState(session_id="s2", flier=base_flier)
        assert s1.research_results is not s2.research_results
        assert s1.errors is not s2.errors

    def test_model_dump_json_valid(self, base_flier: FlierImage) -> None:
        state = PipelineState(session_id="s1", flier=base_flier)
        parsed = json.loads(state.model_dump_json())
        assert parsed["session_id"] == "s1"
        assert parsed["current_phase"] == "UPLOAD"


# ======================================================================
# Venue & Promoter
# ======================================================================


class TestVenueAndPromoter:
    """Tests for the Venue and Promoter models."""

    def test_venue_creation(self) -> None:
        v = Venue(name="Tresor", city="Berlin", country="Germany", confidence=0.9)
        assert v.name == "Tresor"
        assert v.city == "Berlin"

    def test_venue_confidence_validation(self) -> None:
        with pytest.raises(ValidationError):
            Venue(name="V", confidence=1.5)

    def test_venue_default_factory_lists(self) -> None:
        v1 = Venue(name="V1")
        v2 = Venue(name="V2")
        assert v1.notable_events is not v2.notable_events
        assert v1.articles is not v2.articles

    def test_promoter_creation(self) -> None:
        p = Promoter(name="Tresor Records", confidence=0.8)
        assert p.name == "Tresor Records"

    def test_promoter_default_factory_lists(self) -> None:
        p1 = Promoter(name="P1")
        p2 = Promoter(name="P2")
        assert p1.event_history is not p2.event_history
        assert p1.affiliated_artists is not p2.affiliated_artists
        assert p1.affiliated_venues is not p2.affiliated_venues
        assert p1.articles is not p2.articles


# ======================================================================
# ResearchResult & DateContext
# ======================================================================


class TestResearchModels:
    """Tests for the ResearchResult and DateContext models."""

    def test_date_context_creation(self) -> None:
        dc = DateContext(
            event_date=date(1997, 6, 15),
            scene_context="Detroit techno peak",
        )
        assert dc.event_date == date(1997, 6, 15)
        assert dc.scene_context == "Detroit techno peak"

    def test_date_context_default_factory_lists(self) -> None:
        dc1 = DateContext(event_date=date(2000, 1, 1))
        dc2 = DateContext(event_date=date(2000, 1, 1))
        assert dc1.nearby_events is not dc2.nearby_events
        assert dc1.sources is not dc2.sources

    def test_research_result_creation(self) -> None:
        rr = ResearchResult(
            entity_type=EntityType.ARTIST,
            entity_name="Carl Cox",
            confidence=0.85,
        )
        assert rr.entity_name == "Carl Cox"
        assert rr.confidence == 0.85

    def test_research_result_confidence_validation(self) -> None:
        with pytest.raises(ValidationError):
            ResearchResult(entity_type=EntityType.ARTIST, entity_name="A", confidence=1.5)

    def test_research_result_default_factory_lists(self) -> None:
        r1 = ResearchResult(entity_type=EntityType.ARTIST, entity_name="A")
        r2 = ResearchResult(entity_type=EntityType.ARTIST, entity_name="B")
        assert r1.sources_consulted is not r2.sources_consulted
        assert r1.warnings is not r2.warnings

    def test_research_result_model_dump_json_valid(self) -> None:
        rr = ResearchResult(entity_type=EntityType.ARTIST, entity_name="A")
        parsed = json.loads(rr.model_dump_json())
        assert parsed["entity_name"] == "A"

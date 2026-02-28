"""Unit tests for RA event integration in the InterconnectionService.

Tests cover the three new methods added to wire Resident Advisor event
data into the interconnection pipeline:
  - _parse_ra_event_chunk_text: parse structured RA event text from chunks
  - _discover_shared_ra_events: query vector store for shared lineups
  - _compile_shared_event_context: format shared events as citable context

Also tests modifications to existing methods:
  - _validate_citations: RA source recognition
  - _build_edges: RA citation tier assignment
  - _boost_ra_backed_edges: confidence floor for RA-backed edges
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import date
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.analysis import Citation, RelationshipEdge
from src.models.entities import EntityType
from src.models.flier import ExtractedEntities, ExtractedEntity, OCRResult
from src.models.rag import DocumentChunk, RetrievedChunk
from src.models.research import ResearchResult
from src.services.citation_service import CitationService
from src.services.interconnection_service import InterconnectionService


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def citation_service() -> CitationService:
    """Create a real CitationService for tier assignment tests."""
    return CitationService()


@pytest.fixture()
def mock_llm() -> MagicMock:
    """Create a mock LLM provider."""
    llm = MagicMock()
    llm.get_provider_name.return_value = "mock-llm"
    return llm


@pytest.fixture()
def service(mock_llm: MagicMock, citation_service: CitationService) -> InterconnectionService:
    """Create an InterconnectionService with a mock LLM and no vector store."""
    return InterconnectionService(
        llm_provider=mock_llm,
        citation_service=citation_service,
        vector_store=None,
    )


def _make_ra_chunk_text(events: list[dict[str, Any]]) -> str:
    """Build RA event chunk text matching RAEventProcessor.event_to_text format.

    Each event dict should have keys: title, date, venue, artists (list),
    city, url (optional), attending (optional).
    """
    blocks: list[str] = []
    for ev in events:
        lines: list[str] = []
        lines.append(f"Event: {ev['title']}")
        lines.append(f"Date: {ev['date']}")
        if ev.get("venue"):
            lines.append(f"Venue: {ev['venue']}")
        if ev.get("artists"):
            lines.append(f"Artists: {', '.join(ev['artists'])}")
        if ev.get("city"):
            lines.append(f"City: {ev['city']}")
        if ev.get("url"):
            lines.append(f"URL: {ev['url']}")
        if ev.get("attending"):
            lines.append(f"Attending: {ev['attending']:,}")
        blocks.append("\n".join(lines))
    return "\n\n---\n\n".join(blocks)


def _make_document_chunk(
    text: str,
    source_title: str = "RA Events: London (2019-06-01 to 2019-06-30)",
    entity_tags: list[str] | None = None,
    source_type: str = "event",
) -> DocumentChunk:
    """Build a DocumentChunk with RA event metadata."""
    return DocumentChunk(
        chunk_id=str(uuid.uuid4()),
        text=text,
        source_id=hashlib.sha256(source_title.encode()).hexdigest()[:16],
        source_title=source_title,
        source_type=source_type,
        citation_tier=3,
        entity_tags=entity_tags or [],
        geographic_tags=["London"],
    )


def _make_retrieved_chunk(
    chunk: DocumentChunk,
    score: float = 0.85,
) -> RetrievedChunk:
    """Wrap a DocumentChunk in a RetrievedChunk."""
    return RetrievedChunk(
        chunk=chunk,
        similarity_score=score,
        formatted_citation=f"{chunk.source_title} [Tier {chunk.citation_tier}]",
    )


def _make_entities(*artist_names: str) -> ExtractedEntities:
    """Build ExtractedEntities with the given artist names."""
    artists = [
        ExtractedEntity(text=name, entity_type=EntityType.ARTIST)
        for name in artist_names
    ]
    return ExtractedEntities(
        artists=artists,
        raw_ocr=OCRResult(
            raw_text="test ocr text",
            provider_used="mock",
            confidence=1.0,
            processing_time=0.1,
        ),
    )


# ======================================================================
# _parse_ra_event_chunk_text
# ======================================================================


class TestParseRaEventChunkText:
    """Tests for the RA event chunk text parser."""

    def test_standard_three_events(self, service: InterconnectionService) -> None:
        """Parse 3 events in exact RAEventProcessor format."""
        text = _make_ra_chunk_text([
            {
                "title": "Boiler Room London",
                "date": "2019-06-15",
                "venue": "Corsica Studios (London)",
                "artists": ["DJ A", "DJ B", "DJ C"],
                "city": "London",
                "url": "https://ra.co/events/1234567",
                "attending": 500,
            },
            {
                "title": "Fabric Presents",
                "date": "2020-02-28",
                "venue": "Fabric",
                "artists": ["DJ A", "DJ D"],
                "city": "London",
                "url": "https://ra.co/events/2345678",
            },
            {
                "title": "Dekmantel Festival",
                "date": "2018-11-10",
                "venue": "Gashouder (Amsterdam)",
                "artists": ["DJ B", "DJ E"],
                "city": "Amsterdam",
                "url": "https://ra.co/events/3456789",
                "attending": 2000,
            },
        ])

        parsed = InterconnectionService._parse_ra_event_chunk_text(text)

        assert len(parsed) == 3

        assert parsed[0]["title"] == "Boiler Room London"
        assert parsed[0]["date"] == "2019-06-15"
        assert parsed[0]["venue"] == "Corsica Studios (London)"
        assert parsed[0]["artists"] == ["DJ A", "DJ B", "DJ C"]
        assert parsed[0]["city"] == "London"
        assert parsed[0]["url"] == "https://ra.co/events/1234567"

        assert parsed[1]["title"] == "Fabric Presents"
        assert parsed[1]["artists"] == ["DJ A", "DJ D"]

        assert parsed[2]["city"] == "Amsterdam"

    def test_missing_optional_fields(self, service: InterconnectionService) -> None:
        """Events with missing optional fields parse without error."""
        text = "Event: Minimal Event\nDate: 2020-01-01\nArtists: DJ X"

        parsed = InterconnectionService._parse_ra_event_chunk_text(text)

        assert len(parsed) == 1
        assert parsed[0]["title"] == "Minimal Event"
        assert parsed[0]["artists"] == ["DJ X"]
        assert parsed[0]["venue"] == ""
        assert parsed[0]["city"] == ""
        assert parsed[0]["url"] == ""

    def test_empty_input(self, service: InterconnectionService) -> None:
        """Empty string returns empty list."""
        assert InterconnectionService._parse_ra_event_chunk_text("") == []
        assert InterconnectionService._parse_ra_event_chunk_text("   ") == []


# ======================================================================
# _discover_shared_ra_events
# ======================================================================


class TestDiscoverSharedRaEvents:
    """Tests for the RA event shared lineup discovery."""

    @pytest.mark.asyncio()
    async def test_two_artists_one_shared_event(
        self, mock_llm: MagicMock, citation_service: CitationService
    ) -> None:
        """Two artists who share one event produce one shared event result."""
        # Build chunk text with an event featuring both DJ A and DJ B
        chunk_text = _make_ra_chunk_text([
            {
                "title": "Shared Event",
                "date": "2019-06-15",
                "venue": "Club X",
                "artists": ["DJ A", "DJ B", "DJ C"],
                "city": "London",
                "url": "https://ra.co/events/111",
            },
        ])
        chunk = _make_document_chunk(
            text=chunk_text,
            entity_tags=["DJ A", "DJ B", "DJ C"],
        )
        retrieved = _make_retrieved_chunk(chunk)

        # Mock vector store returns the same chunk for both artist queries
        mock_vs = AsyncMock()
        mock_vs.is_available = MagicMock(return_value=True)
        mock_vs.query.return_value = [retrieved]

        svc = InterconnectionService(
            llm_provider=mock_llm,
            citation_service=citation_service,
            vector_store=mock_vs,
        )

        entities = _make_entities("DJ A", "DJ B")
        result = await svc._discover_shared_ra_events(entities)

        assert len(result) == 1
        assert set(result[0]["artist_pair"]) == {"DJ A", "DJ B"}
        assert result[0]["event_title"] == "Shared Event"
        assert result[0]["event_date"] == "2019-06-15"
        assert result[0]["venue"] == "Club X"
        assert result[0]["city"] == "London"
        assert "DJ A" in result[0]["full_lineup"]
        assert "DJ B" in result[0]["full_lineup"]

    @pytest.mark.asyncio()
    async def test_no_overlap(
        self, mock_llm: MagicMock, citation_service: CitationService
    ) -> None:
        """Two artists with no shared events return empty list."""
        chunk_a_text = _make_ra_chunk_text([
            {"title": "Event A Only", "date": "2019-01-01", "venue": "V1",
             "artists": ["DJ A", "DJ X"], "city": "London"},
        ])
        chunk_b_text = _make_ra_chunk_text([
            {"title": "Event B Only", "date": "2019-02-01", "venue": "V2",
             "artists": ["DJ B", "DJ Y"], "city": "Berlin"},
        ])

        chunk_a = _make_document_chunk(chunk_a_text, entity_tags=["DJ A", "DJ X"])
        chunk_b = _make_document_chunk(chunk_b_text, entity_tags=["DJ B", "DJ Y"])

        mock_vs = AsyncMock()
        mock_vs.is_available = MagicMock(return_value=True)

        # Return different chunks per query based on artist name
        async def side_effect(query_text: str, **kwargs: Any) -> list[RetrievedChunk]:
            if "DJ A" in query_text:
                return [_make_retrieved_chunk(chunk_a)]
            if "DJ B" in query_text:
                return [_make_retrieved_chunk(chunk_b)]
            return []

        mock_vs.query.side_effect = side_effect

        svc = InterconnectionService(
            llm_provider=mock_llm,
            citation_service=citation_service,
            vector_store=mock_vs,
        )

        entities = _make_entities("DJ A", "DJ B")
        result = await svc._discover_shared_ra_events(entities)

        assert result == []

    @pytest.mark.asyncio()
    async def test_no_vector_store(self, service: InterconnectionService) -> None:
        """Returns empty list when vector store is None."""
        entities = _make_entities("DJ A", "DJ B")
        result = await service._discover_shared_ra_events(entities)
        assert result == []

    @pytest.mark.asyncio()
    async def test_vector_store_unavailable(
        self, mock_llm: MagicMock, citation_service: CitationService
    ) -> None:
        """Returns empty list when vector store reports unavailable."""
        mock_vs = AsyncMock()
        mock_vs.is_available = MagicMock(return_value=False)

        svc = InterconnectionService(
            llm_provider=mock_llm,
            citation_service=citation_service,
            vector_store=mock_vs,
        )

        entities = _make_entities("DJ A", "DJ B")
        result = await svc._discover_shared_ra_events(entities)
        assert result == []

    @pytest.mark.asyncio()
    async def test_fewer_than_two_artists(
        self, mock_llm: MagicMock, citation_service: CitationService
    ) -> None:
        """Returns empty list with only one artist (no pairs to compare)."""
        mock_vs = AsyncMock()
        mock_vs.is_available = MagicMock(return_value=True)

        svc = InterconnectionService(
            llm_provider=mock_llm,
            citation_service=citation_service,
            vector_store=mock_vs,
        )

        entities = _make_entities("DJ A")
        result = await svc._discover_shared_ra_events(entities)
        assert result == []
        # Should not have queried the vector store
        mock_vs.query.assert_not_called()


# ======================================================================
# _compile_shared_event_context
# ======================================================================


class TestCompileSharedEventContext:
    """Tests for the shared event context compiler."""

    def test_format_with_events(self, service: InterconnectionService) -> None:
        """Shared events produce context with [RA-n] refs and source index."""
        shared_events = [
            {
                "artist_pair": ("DJ A", "DJ B"),
                "event_title": "Boiler Room London",
                "event_date": "2019-06-15",
                "venue": "Corsica Studios",
                "city": "London",
                "source_title": "RA Events: London (2019-06-01 to 2019-06-30)",
                "ra_url": "https://ra.co/events/1234567",
                "full_lineup": ["DJ A", "DJ B", "DJ C"],
            },
            {
                "artist_pair": ("DJ A", "DJ B"),
                "event_title": "Fabric Presents",
                "event_date": "2020-02-28",
                "venue": "Fabric",
                "city": "London",
                "source_title": "RA Events: London (2020-02-01 to 2020-02-29)",
                "ra_url": "https://ra.co/events/2345678",
                "full_lineup": ["DJ A", "DJ B"],
            },
        ]

        result = InterconnectionService._compile_shared_event_context(shared_events)

        assert "=== SHARED EVENT APPEARANCES" in result
        assert "[RA-1]" in result
        assert "[RA-2]" in result
        assert "Boiler Room London" in result
        assert "Fabric Presents" in result
        assert "DJ A and DJ B" in result
        assert "=== RA EVENT SOURCE INDEX ===" in result
        assert "https://ra.co/events/1234567" in result
        assert "https://ra.co/events/2345678" in result

    def test_empty_input(self, service: InterconnectionService) -> None:
        """Empty shared events list returns empty string."""
        result = InterconnectionService._compile_shared_event_context([])
        assert result == ""


# ======================================================================
# _validate_citations — RA source recognition
# ======================================================================


class TestValidateCitationsRaSources:
    """Tests that _validate_citations recognizes RA event sources."""

    def test_recognizes_ra_source(self, service: InterconnectionService) -> None:
        """Citation referencing 'RA Events' passes when shared_ra_events provided."""
        relationships = [
            {
                "source": "DJ A",
                "target": "DJ B",
                "type": "shared_lineup",
                "details": "Appeared together at Boiler Room London",
                "source_citation": "RA Events: London (2019-06-01 to 2019-06-30)",
                "confidence": 0.8,
            },
        ]
        # Empty research data — no regular sources registered
        research_data: list[ResearchResult] = []

        shared_ra_events = [
            {
                "source_title": "RA Events: London (2019-06-01 to 2019-06-30)",
                "event_title": "Boiler Room London",
                "ra_url": "https://ra.co/events/1234567",
            },
        ]

        result = service._validate_citations(
            relationships, research_data, shared_ra_events=shared_ra_events
        )
        assert len(result) == 1

    def test_rejects_unknown_ra_source(self, service: InterconnectionService) -> None:
        """Citation referencing RA without matching entry is rejected."""
        relationships = [
            {
                "source": "DJ A",
                "target": "DJ B",
                "type": "shared_lineup",
                "details": "Appeared together",
                "source_citation": "Some made up source that doesn't exist",
                "confidence": 0.8,
            },
        ]
        research_data: list[ResearchResult] = []

        # No shared_ra_events — RA sources not registered
        result = service._validate_citations(
            relationships, research_data, shared_ra_events=None
        )
        assert len(result) == 0


# ======================================================================
# _boost_ra_backed_edges
# ======================================================================


class TestBoostRaBackedEdges:
    """Tests for the RA-backed confidence boost."""

    def test_boost_shared_lineup_with_ra_citation(
        self, service: InterconnectionService
    ) -> None:
        """shared_lineup edge with RA citation gets boosted to min 0.6."""
        edge = RelationshipEdge(
            source="DJ A",
            target="DJ B",
            relationship_type="shared_lineup",
            details="Appeared together at Boiler Room",
            citations=[
                Citation(
                    text="RA Events: London",
                    source_type="event",
                    source_name="Resident Advisor Events",
                )
            ],
            confidence=0.2,  # Low after uncertain penalty
        )

        shared_ra_events = [{"event_title": "Boiler Room"}]

        result = InterconnectionService._boost_ra_backed_edges([edge], shared_ra_events)

        assert len(result) == 1
        assert result[0].confidence == 0.6

    def test_no_boost_without_shared_events(
        self, service: InterconnectionService
    ) -> None:
        """No boost applied when shared_ra_events is None."""
        edge = RelationshipEdge(
            source="DJ A",
            target="DJ B",
            relationship_type="shared_lineup",
            citations=[
                Citation(
                    text="some citation",
                    source_type="research",
                    source_name="generic",
                )
            ],
            confidence=0.3,
        )

        result = InterconnectionService._boost_ra_backed_edges([edge], None)
        assert result[0].confidence == 0.3

    def test_no_boost_for_non_shared_lineup(
        self, service: InterconnectionService
    ) -> None:
        """Non-shared_lineup edges are not boosted even with RA citations."""
        edge = RelationshipEdge(
            source="DJ A",
            target="DJ B",
            relationship_type="shared_label",
            citations=[
                Citation(
                    text="RA Events",
                    source_type="event",
                    source_name="Resident Advisor Events",
                )
            ],
            confidence=0.2,
        )

        result = InterconnectionService._boost_ra_backed_edges(
            [edge], [{"event_title": "test"}]
        )
        assert result[0].confidence == 0.2


# ======================================================================
# _build_edges — RA citation tier
# ======================================================================


class TestBuildEdgesRaCitationTier:
    """Tests that _build_edges assigns proper tier to RA citations."""

    def test_ra_citation_gets_event_type(
        self, mock_llm: MagicMock, citation_service: CitationService
    ) -> None:
        """Edge citing RA events gets source_type='event' (tier 3)."""
        svc = InterconnectionService(
            llm_provider=mock_llm,
            citation_service=citation_service,
        )

        relationships = [
            {
                "source": "DJ A",
                "target": "DJ B",
                "type": "shared_lineup",
                "details": "Shared lineup",
                "source_citation": "[RA-1] RA Events: London — https://ra.co/events/123",
                "confidence": 0.8,
            },
        ]

        edges = svc._build_edges(relationships)

        assert len(edges) == 1
        assert edges[0].citations[0].source_name == "Resident Advisor Events"
        assert edges[0].citations[0].tier == 3

    def test_non_ra_citation_gets_research_type(
        self, mock_llm: MagicMock, citation_service: CitationService
    ) -> None:
        """Non-RA citation keeps default 'research' source type."""
        svc = InterconnectionService(
            llm_provider=mock_llm,
            citation_service=citation_service,
        )

        relationships = [
            {
                "source": "DJ A",
                "target": "DJ B",
                "type": "shared_label",
                "details": "Both released on Label X",
                "source_citation": "Discogs: Label X",
                "confidence": 0.7,
            },
        ]

        edges = svc._build_edges(relationships)

        assert len(edges) == 1
        assert edges[0].citations[0].source_name != "Resident Advisor Events"

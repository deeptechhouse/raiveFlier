"""Integration tests for ResearchService parallel research orchestration."""

from __future__ import annotations

import asyncio
import time
from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.entities import EntityType
from src.models.flier import ExtractedEntities, ExtractedEntity, OCRResult
from src.models.research import ResearchResult
from src.services.artist_researcher import ArtistResearcher
from src.services.date_context_researcher import DateContextResearcher
from src.services.promoter_researcher import PromoterResearcher
from src.services.research_service import ResearchService
from src.services.venue_researcher import VenueResearcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ocr() -> OCRResult:
    """Create a minimal OCRResult for building ExtractedEntities."""
    return OCRResult(
        raw_text="test text",
        confidence=0.9,
        provider_used="test",
        processing_time=0.1,
    )


def _make_research_result(
    entity_type: EntityType,
    entity_name: str,
    confidence: float = 0.8,
    warnings: list[str] | None = None,
) -> ResearchResult:
    """Build a ResearchResult for a given entity."""
    return ResearchResult(
        entity_type=entity_type,
        entity_name=entity_name,
        sources_consulted=["mock-source"],
        confidence=confidence,
        warnings=warnings or [],
    )


def _make_research_service(
    artist_side_effect: list | None = None,
    venue_result: ResearchResult | None = None,
    promoter_result: ResearchResult | None = None,
    date_result: ResearchResult | None = None,
    artist_delay: float = 0.0,
) -> ResearchService:
    """Create a ResearchService with mocked researchers."""
    artist_researcher = MagicMock(spec=ArtistResearcher)
    venue_researcher = MagicMock(spec=VenueResearcher)
    promoter_researcher = MagicMock(spec=PromoterResearcher)
    date_researcher = MagicMock(spec=DateContextResearcher)

    if artist_side_effect is not None:

        async def _delayed_artist(*args, **kwargs):
            if artist_delay > 0:
                await asyncio.sleep(artist_delay)
            result = artist_side_effect.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        artist_researcher.research = _delayed_artist
    else:
        artist_researcher.research = AsyncMock(
            return_value=_make_research_result(EntityType.ARTIST, "Unknown")
        )

    if venue_result is not None:

        async def _venue_research(*args, **kwargs):
            if artist_delay > 0:
                await asyncio.sleep(artist_delay)
            return venue_result

        venue_researcher.research = _venue_research
    else:
        venue_researcher.research = AsyncMock(
            return_value=_make_research_result(EntityType.VENUE, "Unknown Venue")
        )

    if promoter_result is not None:

        async def _promoter_research(*args, **kwargs):
            if artist_delay > 0:
                await asyncio.sleep(artist_delay)
            return promoter_result

        promoter_researcher.research = _promoter_research
    else:
        promoter_researcher.research = AsyncMock(
            return_value=_make_research_result(EntityType.PROMOTER, "Unknown Promoter")
        )

    if date_result is not None:

        async def _date_research(*args, **kwargs):
            if artist_delay > 0:
                await asyncio.sleep(artist_delay)
            return date_result

        date_researcher.research = _date_research
    else:
        date_researcher.research = AsyncMock(
            return_value=_make_research_result(EntityType.DATE, "Unknown Date")
        )

    return ResearchService(
        artist_researcher=artist_researcher,
        venue_researcher=venue_researcher,
        promoter_researcher=promoter_researcher,
        date_context_researcher=date_researcher,
    )


def _make_entities(
    artists: list[str] | None = None,
    venue: str | None = None,
    promoter: str | None = None,
    date_text: str | None = None,
) -> ExtractedEntities:
    """Build ExtractedEntities with specified entities."""
    artist_entities = [
        ExtractedEntity(text=name, entity_type=EntityType.ARTIST, confidence=0.9)
        for name in (artists or [])
    ]
    return ExtractedEntities(
        artists=artist_entities,
        venue=(
            ExtractedEntity(text=venue, entity_type=EntityType.VENUE, confidence=0.85)
            if venue
            else None
        ),
        date=(
            ExtractedEntity(text=date_text, entity_type=EntityType.DATE, confidence=0.8)
            if date_text
            else None
        ),
        promoter=(
            ExtractedEntity(text=promoter, entity_type=EntityType.PROMOTER, confidence=0.7)
            if promoter
            else None
        ),
        raw_ocr=_make_ocr(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResearchServiceParallel:
    """Tests for ResearchService.research_all() concurrent execution."""

    @pytest.mark.asyncio()
    async def test_parallel_research(self) -> None:
        """All entities researched concurrently — total time ~= single task time."""
        delay = 0.05  # 50ms per task

        artist_results = [
            _make_research_result(EntityType.ARTIST, "Carl Cox"),
            _make_research_result(EntityType.ARTIST, "Jeff Mills"),
            _make_research_result(EntityType.ARTIST, "Derrick May"),
        ]

        service = _make_research_service(
            artist_side_effect=artist_results,
            venue_result=_make_research_result(EntityType.VENUE, "Tresor"),
            promoter_result=_make_research_result(EntityType.PROMOTER, "Tresor Records"),
            date_result=_make_research_result(EntityType.DATE, "1997-03-15"),
            artist_delay=delay,
        )

        entities = _make_entities(
            artists=["Carl Cox", "Jeff Mills", "Derrick May"],
            venue="Tresor, Berlin",
            promoter="Tresor Records",
            date_text="March 15, 1997",
        )

        start = time.monotonic()
        results = await service.research_all(entities, event_date=date(1997, 3, 15))
        elapsed = time.monotonic() - start

        # 6 tasks (3 artists + venue + promoter + date) but running in parallel
        # Should take ~1x delay, not ~6x delay
        assert elapsed < delay * 4  # generous margin for CI
        assert len(results) == 6

        entity_names = {r.entity_name for r in results}
        assert "Carl Cox" in entity_names
        assert "Jeff Mills" in entity_names
        assert "Derrick May" in entity_names
        assert "Tresor" in entity_names
        assert "Tresor Records" in entity_names

    @pytest.mark.asyncio()
    async def test_partial_failure(self) -> None:
        """One artist research fails, others succeed; failure captured as warning."""
        artist_results = [
            _make_research_result(EntityType.ARTIST, "Carl Cox"),
            RuntimeError("Database connection lost"),
            _make_research_result(EntityType.ARTIST, "Derrick May"),
        ]

        service = _make_research_service(
            artist_side_effect=artist_results,
            venue_result=_make_research_result(EntityType.VENUE, "Tresor"),
        )

        entities = _make_entities(
            artists=["Carl Cox", "Jeff Mills", "Derrick May"],
            venue="Tresor, Berlin",
        )

        results = await service.research_all(entities, event_date=date(1997, 3, 15))

        # 5 results: 3 artists + 1 venue + 1 date-context
        assert len(results) == 5

        # Successful artist results
        cox_result = next(r for r in results if r.entity_name == "Carl Cox")
        assert cox_result.confidence > 0

        # Failed artist result — captured as warning with confidence=0
        mills_result = next(r for r in results if r.entity_name == "Jeff Mills")
        assert mills_result.confidence == 0.0
        assert len(mills_result.warnings) > 0
        assert "RuntimeError" in mills_result.warnings[0]

        may_result = next(r for r in results if r.entity_name == "Derrick May")
        assert may_result.confidence > 0

        # Other entities still succeed
        venue_result = next(r for r in results if r.entity_type == EntityType.VENUE)
        assert venue_result.confidence > 0


class TestResearchServiceDateParsing:
    """Tests for date parsing within the research service."""

    @pytest.mark.asyncio()
    async def test_date_format_month_day_year(self) -> None:
        """'March 15, 1997' parsed correctly."""
        service = _make_research_service(artist_side_effect=[])
        entities = _make_entities(date_text="March 15, 1997")

        results = await service.research_all(entities)

        # Date task was dispatched (empty artist list, no venue/promoter, just date)
        assert len(results) == 1
        assert results[0].entity_type == EntityType.DATE

    @pytest.mark.asyncio()
    async def test_date_format_slash_short_year(self) -> None:
        """'03/15/97' parsed correctly via dateutil or manual fallback."""
        service = _make_research_service(artist_side_effect=[])
        entities = _make_entities(date_text="03/15/97")

        results = await service.research_all(entities)

        assert len(results) == 1
        assert results[0].entity_type == EntityType.DATE

    @pytest.mark.asyncio()
    async def test_date_format_dot_separated(self) -> None:
        """'03.15.1997' parsed correctly."""
        service = _make_research_service(artist_side_effect=[])
        entities = _make_entities(date_text="03.15.1997")

        results = await service.research_all(entities)

        assert len(results) == 1
        assert results[0].entity_type == EntityType.DATE

    @pytest.mark.asyncio()
    async def test_date_with_day_name_and_ordinal(self) -> None:
        """'Saturday March 15th 1997' parsed correctly (ordinal stripped)."""
        service = _make_research_service(artist_side_effect=[])
        entities = _make_entities(date_text="Saturday March 15th 1997")

        results = await service.research_all(entities)

        assert len(results) == 1
        assert results[0].entity_type == EntityType.DATE

    @pytest.mark.asyncio()
    async def test_no_entities_returns_empty(self) -> None:
        """research_all with no entities returns empty list."""
        service = _make_research_service(artist_side_effect=[])
        entities = _make_entities()

        results = await service.research_all(entities)

        assert results == []


class TestResearchServiceCityExtraction:
    """Tests for city hint extraction from venue text."""

    @pytest.mark.asyncio()
    async def test_city_extracted_from_venue_comma(self) -> None:
        """Venue 'Tresor, Berlin' extracts 'Berlin' as city hint."""
        service = _make_research_service(
            artist_side_effect=[_make_research_result(EntityType.ARTIST, "Carl Cox")],
            venue_result=_make_research_result(EntityType.VENUE, "Tresor"),
        )

        entities = _make_entities(
            artists=["Carl Cox"],
            venue="Tresor, Berlin",
        )

        results = await service.research_all(entities, event_date=date(1997, 3, 15))

        # 3 results: 1 artist + 1 venue + 1 date-context
        assert len(results) == 3
        venue_result = next(r for r in results if r.entity_type == EntityType.VENUE)
        assert venue_result.entity_name == "Tresor"

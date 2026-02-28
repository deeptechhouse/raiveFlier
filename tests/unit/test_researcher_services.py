"""Unit tests for researcher services — Artist, Venue, DateContext, EventName, Promoter, Interconnection."""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.interfaces.article_provider import ArticleContent
from src.interfaces.web_search_provider import SearchResult
from src.interfaces.music_db_provider import ArtistSearchResult
from src.models.entities import (
    ArticleReference,
    Artist,
    EntityType,
    EventAppearance,
    EventInstance,
    EventSeriesHistory,
    Label,
    Promoter,
    Release,
    Venue,
)
from src.models.research import DateContext, ResearchResult
from src.models.analysis import (
    Citation,
    EntityNode,
    InterconnectionMap,
    PatternInsight,
    RelationshipEdge,
)
from src.models.flier import ExtractedEntities, ExtractedEntity, OCRResult
from src.utils.errors import LLMError


# ======================================================================
# Shared fixtures and helpers
# ======================================================================


def _mock_web_search(results=None):
    mock = AsyncMock()
    mock.search = AsyncMock(return_value=results or [])
    return mock


def _mock_article_scraper(content=None):
    mock = AsyncMock()
    mock.extract_content = AsyncMock(return_value=content)
    return mock


def _mock_llm(response=""):
    mock = MagicMock()
    mock.complete = AsyncMock(return_value=response)
    mock.get_provider_name = MagicMock(return_value="mock_llm")
    mock.is_available = MagicMock(return_value=True)
    mock.supports_vision = MagicMock(return_value=False)
    return mock


def _mock_cache():
    mock = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock()
    return mock


def _mock_vector_store(results=None):
    mock = MagicMock()
    mock.is_available = MagicMock(return_value=bool(results))
    mock.query = AsyncMock(return_value=results or [])
    return mock


def _mock_music_db(provider_name="discogs_api", search_results=None, releases=None, labels=None):
    mock = MagicMock()
    mock.get_provider_name = MagicMock(return_value=provider_name)
    mock.is_available = MagicMock(return_value=True)
    mock.search_artist = AsyncMock(return_value=search_results or [])
    mock.get_artist_releases = AsyncMock(return_value=releases or [])
    mock.get_artist_labels = AsyncMock(return_value=labels or [])
    return mock


def _search_result(title="Test Article", url="https://ra.co/test", snippet="test snippet"):
    return SearchResult(title=title, url=url, snippet=snippet)


def _article_content(title="Article Title", text="Article about Carl Cox at Tresor Berlin."):
    return ArticleContent(title=title, text=text, date=None)


def _make_ocr_result():
    """Build a minimal OCRResult for use in ExtractedEntities."""
    return OCRResult(
        raw_text="Carl Cox at Tresor",
        confidence=0.9,
        provider_used="mock_ocr",
        processing_time=0.1,
    )


# ======================================================================
# TestArtistResearcher
# ======================================================================


class TestArtistResearcher:
    """Tests for the ArtistResearcher service."""

    @pytest.mark.asyncio
    async def test_research_happy_path(self):
        """Music DB returns a match, releases, and labels -- full pipeline."""
        from src.services.artist_researcher import ArtistResearcher

        db = _mock_music_db(
            provider_name="discogs_api",
            search_results=[
                ArtistSearchResult(id="123", name="Carl Cox", confidence=0.95),
            ],
            releases=[Release(title="Phat Trax", label="Intec", year=1995)],
            labels=[Label(name="Intec")],
        )

        researcher = ArtistResearcher(
            music_dbs=[db],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(response="Carl Cox is a British techno DJ based in Melbourne."),
            cache=_mock_cache(),
            vector_store=None,
        )

        result = await researcher.research("Carl Cox")

        assert result.entity_type == EntityType.ARTIST
        assert result.artist is not None
        # normalize_artist_name uses .title(), so "Carl Cox" stays "Carl Cox"
        assert "carl cox" in result.artist.name.lower()
        assert result.confidence > 0.0

    @pytest.mark.asyncio
    async def test_research_no_db_match(self):
        """All music DBs return empty -- warning should be present."""
        from src.services.artist_researcher import ArtistResearcher

        db = _mock_music_db(provider_name="discogs_api", search_results=[])

        researcher = ArtistResearcher(
            music_dbs=[db],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=_mock_cache(),
            vector_store=None,
        )

        result = await researcher.research("UnknownArtist9999")

        assert result.artist is not None
        assert any("not found" in w.lower() for w in result.warnings)

    def test_is_music_relevant_known_domain(self):
        """Result from ra.co always passes relevance check."""
        from src.services.artist_researcher import ArtistResearcher

        result = SearchResult(
            title="Some random title",
            url="https://ra.co/features/1234",
            snippet="unrelated snippet",
        )
        assert ArtistResearcher._is_music_relevant(result) is True

    def test_is_music_relevant_unknown_domain_no_terms(self):
        """Non-music domain with no music-related terms fails."""
        from src.services.artist_researcher import ArtistResearcher

        result = SearchResult(
            title="John Smith biography",
            url="https://example.com/bio",
            snippet="A life story.",
        )
        assert ArtistResearcher._is_music_relevant(result) is False

    def test_is_music_relevant_unknown_domain_with_dj(self):
        """Non-music domain but title contains 'DJ' passes."""
        from src.services.artist_researcher import ArtistResearcher

        result = SearchResult(
            title="DJ John Smith interview",
            url="https://example.com/interview",
            snippet="An interview with DJ John Smith.",
        )
        assert ArtistResearcher._is_music_relevant(result) is True

    def test_extract_relevant_snippet(self):
        """Snippet extraction finds sentences mentioning the artist."""
        from src.services.artist_researcher import ArtistResearcher

        text = (
            "The crowd gathered early. Carl Cox played a memorable set. "
            "The night ended at dawn. Carl Cox closed with acid techno."
        )
        snippet = ArtistResearcher._extract_relevant_snippet(text, "carl cox")

        assert "Carl Cox" in snippet
        assert "crowd gathered" not in snippet  # unrelated sentence excluded

    def test_extract_domain(self):
        """Domain extraction strips protocol and www prefix."""
        from src.services.artist_researcher import ArtistResearcher

        assert ArtistResearcher._extract_domain("https://www.ra.co/features") == "ra.co"

    def test_assign_citation_tier_various_urls(self):
        """Citation tier assignment for known URL patterns."""
        from src.services.artist_researcher import ArtistResearcher

        researcher = ArtistResearcher(
            music_dbs=[],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
        )

        assert researcher._assign_citation_tier("https://ra.co/dj/123") == 1
        assert researcher._assign_citation_tier("https://xlr8r.com/reviews") == 2
        assert researcher._assign_citation_tier("https://discogs.com/artist/123") == 3
        assert researcher._assign_citation_tier("https://youtube.com/watch?v=x") == 4
        assert researcher._assign_citation_tier("https://reddit.com/r/techno") == 5
        assert researcher._assign_citation_tier("https://unknown.com/page") == 6


# ======================================================================
# TestVenueResearcher
# ======================================================================


class TestVenueResearcher:
    """Tests for the VenueResearcher service."""

    @pytest.mark.asyncio
    async def test_research_happy_path(self):
        """Web search finds results, LLM synthesizes a venue profile."""
        from src.services.venue_researcher import VenueResearcher

        web_search = _mock_web_search(results=[
            _search_result(title="Tresor Berlin", url="https://ra.co/clubs/tresor"),
            _search_result(title="Tresor history", url="https://example.com/tresor"),
        ])
        article_scraper = _mock_article_scraper(
            content=_article_content(
                title="Tresor: A Berlin Institution",
                text="Tresor opened in 1991 in a former department store vault.",
            )
        )
        llm = _mock_llm(
            response=(
                "HISTORY:\n"
                "Tresor opened in 1991 in a vault beneath Leipziger Strasse.\n\n"
                "NOTABLE_EVENTS:\n"
                "- Tresor Records showcase 1992\n"
                "- Jeff Mills residency 1993\n\n"
                "CULTURAL_SIGNIFICANCE:\n"
                "Tresor was the birthplace of Berlin techno."
            )
        )

        researcher = VenueResearcher(
            web_search=web_search,
            article_scraper=article_scraper,
            llm=llm,
        )

        result = await researcher.research("Tresor", city="Berlin")

        assert result.entity_type == EntityType.VENUE
        assert result.venue is not None
        assert result.venue.name == "Tresor"

    @pytest.mark.asyncio
    async def test_research_no_results(self):
        """All sources return empty -- warnings produced."""
        from src.services.venue_researcher import VenueResearcher

        researcher = VenueResearcher(
            web_search=_mock_web_search(results=[]),
            article_scraper=_mock_article_scraper(content=None),
            llm=_mock_llm(response="HISTORY:\nNONE\nNOTABLE_EVENTS:\nNONE\nCULTURAL_SIGNIFICANCE:\nNONE"),
        )

        result = await researcher.research("FakeVenue123")

        assert any("no web results" in w.lower() for w in result.warnings)

    def test_parse_venue_synthesis_full(self):
        """Full synthesis text with all 3 sections parses correctly."""
        from src.services.venue_researcher import VenueResearcher

        text = (
            "HISTORY:\n"
            "Opened in 1991 in a vault.\n\n"
            "NOTABLE_EVENTS:\n"
            "- Tresor Records showcase 1992\n"
            "- Jeff Mills residency\n\n"
            "CULTURAL_SIGNIFICANCE:\n"
            "Birthplace of Berlin techno."
        )

        history, events, significance = VenueResearcher._parse_venue_synthesis(text)

        assert history is not None
        assert "1991" in history
        assert len(events) == 2
        assert "Jeff Mills" in events[1]
        assert significance is not None
        assert "Berlin techno" in significance

    def test_parse_venue_synthesis_all_none(self):
        """All sections set to NONE returns (None, [], None)."""
        from src.services.venue_researcher import VenueResearcher

        text = (
            "HISTORY:\nNONE\n\n"
            "NOTABLE_EVENTS:\nNONE\n\n"
            "CULTURAL_SIGNIFICANCE:\nNONE"
        )

        history, events, significance = VenueResearcher._parse_venue_synthesis(text)

        assert history is None
        assert events == []
        assert significance is None

    def test_parse_venue_synthesis_partial(self):
        """Only HISTORY has content -- other sections return defaults."""
        from src.services.venue_researcher import VenueResearcher

        text = (
            "HISTORY:\n"
            "A legendary venue in the East End.\n\n"
            "NOTABLE_EVENTS:\nNONE\n\n"
            "CULTURAL_SIGNIFICANCE:\nNONE"
        )

        history, events, significance = VenueResearcher._parse_venue_synthesis(text)

        assert history is not None
        assert "legendary" in history
        assert events == []
        assert significance is None


# ======================================================================
# TestDateContextResearcher
# ======================================================================


class TestDateContextResearcher:
    """Tests for the DateContextResearcher service."""

    @pytest.mark.asyncio
    async def test_research_happy_path(self):
        """Web search and LLM produce a date context result."""
        from src.services.date_context_researcher import DateContextResearcher

        web_search = _mock_web_search(results=[
            _search_result(title="Berlin rave scene 1997", url="https://ra.co/features/1997"),
        ])
        llm = _mock_llm(
            response=(
                "SCENE_CONTEXT:\n"
                "Techno was at its peak in Berlin in 1997.\n\n"
                "CITY_CONTEXT:\n"
                "Berlin's club scene was flourishing after reunification.\n\n"
                "CULTURAL_CONTEXT:\n"
                "The Love Parade attracted over 1 million people.\n\n"
                "NEARBY_EVENTS:\n"
                "- Love Parade 1997\n"
                "- Tresor 5th anniversary"
            )
        )
        article_scraper = _mock_article_scraper(
            content=_article_content(title="1997 Rave Scene", text="1997 was a pivotal year.")
        )

        researcher = DateContextResearcher(
            web_search=web_search,
            article_scraper=article_scraper,
            llm=llm,
        )

        result = await researcher.research(date(1997, 3, 15), city="Berlin")

        assert result.entity_type == EntityType.DATE
        assert result.date_context is not None
        assert result.date_context.event_date == date(1997, 3, 15)

    def test_parse_date_context_synthesis_full(self):
        """Full text with all 4 sections parses correctly."""
        from src.services.date_context_researcher import DateContextResearcher

        text = (
            "SCENE_CONTEXT:\n"
            "Minimal techno was rising.\n\n"
            "CITY_CONTEXT:\n"
            "Berlin was Europe's club capital.\n\n"
            "CULTURAL_CONTEXT:\n"
            "The Criminal Justice Act had pushed raves underground in the UK.\n\n"
            "NEARBY_EVENTS:\n"
            "- Tresor opening night\n"
            "- Love Parade\n"
            "- Sven Vath at Omen"
        )

        scene, city, cultural, events = DateContextResearcher._parse_date_context_synthesis(text)

        assert scene is not None
        assert "Minimal techno" in scene
        assert city is not None
        assert "Berlin" in city
        assert cultural is not None
        assert "Criminal Justice Act" in cultural
        assert len(events) == 3
        assert "Love Parade" in events[1]

    def test_parse_date_context_synthesis_all_none(self):
        """All sections set to NONE returns (None, None, None, [])."""
        from src.services.date_context_researcher import DateContextResearcher

        text = (
            "SCENE_CONTEXT:\nNONE\n\n"
            "CITY_CONTEXT:\nNONE\n\n"
            "CULTURAL_CONTEXT:\nNONE\n\n"
            "NEARBY_EVENTS:\nNONE"
        )

        scene, city, cultural, events = DateContextResearcher._parse_date_context_synthesis(text)

        assert scene is None
        assert city is None
        assert cultural is None
        assert events == []

    def test_parse_date_context_synthesis_bullet_list(self):
        """NEARBY_EVENTS with '- ' prefix lines parsed correctly."""
        from src.services.date_context_researcher import DateContextResearcher

        text = (
            "SCENE_CONTEXT:\nNONE\n\n"
            "CITY_CONTEXT:\nNONE\n\n"
            "CULTURAL_CONTEXT:\nNONE\n\n"
            "NEARBY_EVENTS:\n"
            "- Event A\n"
            "- Event B\n"
            "- Event C"
        )

        _, _, _, events = DateContextResearcher._parse_date_context_synthesis(text)

        assert len(events) == 3
        assert events[0] == "Event A"
        assert events[2] == "Event C"


# ======================================================================
# TestEventNameResearcher
# ======================================================================


class TestEventNameResearcher:
    """Tests for the EventNameResearcher service."""

    @pytest.mark.asyncio
    async def test_research_happy_path(self):
        """Web search and LLM produce event instances and a history."""
        from src.services.event_name_researcher import EventNameResearcher

        instances_json = json.dumps([
            {
                "event_name": "Bugged Out!",
                "promoter": "Bugged Out Promotions",
                "venue": "Fabric",
                "city": "London",
                "date": "2001-05-12",
                "source_url": "https://ra.co/events/123",
            },
        ])

        web_search = _mock_web_search(results=[
            _search_result(title="Bugged Out! at Fabric", url="https://ra.co/events/123"),
        ])
        llm = _mock_llm(response=instances_json)
        article_scraper = _mock_article_scraper(
            content=_article_content(
                title="Bugged Out! History", text="Bugged Out! is a legendary club night."
            )
        )

        researcher = EventNameResearcher(
            web_search=web_search,
            article_scraper=article_scraper,
            llm=llm,
        )

        result = await researcher.research("Bugged Out!")

        assert result.entity_type == EntityType.EVENT
        assert result.event_history is not None
        assert len(result.event_history.instances) >= 1

    def test_parse_event_instances_json_fence(self):
        """JSON wrapped in markdown ```json ... ``` fence parses correctly."""
        from src.services.event_name_researcher import EventNameResearcher

        llm_text = '```json\n[{"event_name": "BLOC", "promoter": null, "venue": "Butlins", "city": "Minehead", "date": "2009-03-13", "source_url": null}]\n```'
        instances = EventNameResearcher._parse_event_instances(llm_text)

        assert len(instances) == 1
        assert instances[0].event_name == "BLOC"
        assert instances[0].venue == "Butlins"
        assert instances[0].city == "Minehead"
        assert instances[0].promoter is None

    def test_parse_event_instances_bare_json(self):
        """Bare JSON array (no fence) parses correctly."""
        from src.services.event_name_researcher import EventNameResearcher

        llm_text = '[{"event_name": "Fabric Live", "promoter": "Fabric", "venue": "Fabric", "city": "London", "date": null, "source_url": null}]'
        instances = EventNameResearcher._parse_event_instances(llm_text)

        assert len(instances) == 1
        assert instances[0].event_name == "Fabric Live"
        assert instances[0].date is None

    def test_parse_event_instances_invalid_json(self):
        """Invalid/non-JSON text returns empty list without crashing."""
        from src.services.event_name_researcher import EventNameResearcher

        instances = EventNameResearcher._parse_event_instances(
            "I could not find any event instances for this query."
        )
        assert instances == []

    def test_parse_event_instances_missing_fields(self):
        """JSON with null/missing fields: str(None) -> 'None'; missing key -> 'Unknown'."""
        from src.services.event_name_researcher import EventNameResearcher

        llm_text = json.dumps([
            {"event_name": None, "promoter": None, "venue": None, "city": None, "date": None},
            {"venue": "Tresor"},  # missing event_name key entirely
            {"event_name": "", "venue": "Berghain"},  # empty string -> "Unknown"
        ])

        instances = EventNameResearcher._parse_event_instances(llm_text)

        assert len(instances) == 3
        # str(None) -> "None" which is truthy, so stays as "None"
        assert instances[0].event_name == "None"
        assert instances[0].promoter is None
        assert instances[0].venue is None
        # Missing key: str(item.get("event_name", "")).strip() -> "" -> "Unknown"
        assert instances[1].event_name == "Unknown"
        assert instances[1].venue == "Tresor"
        # Empty string: stripped -> "" -> "Unknown"
        assert instances[2].event_name == "Unknown"
        assert instances[2].venue == "Berghain"

    def test_group_by_promoter(self):
        """3 instances with 2 different promoters group correctly."""
        from src.services.event_name_researcher import EventNameResearcher

        instances = [
            EventInstance(event_name="Night A", promoter="Promoter X"),
            EventInstance(event_name="Night B", promoter="Promoter Y"),
            EventInstance(event_name="Night C", promoter="Promoter X"),
        ]

        groups = EventNameResearcher._group_by_promoter(instances)

        assert len(groups) == 2
        assert len(groups["Promoter X"]) == 2
        assert len(groups["Promoter Y"]) == 1

    def test_group_by_promoter_unknown(self):
        """Instances with None promoter grouped under 'Unknown Promoter'."""
        from src.services.event_name_researcher import EventNameResearcher

        instances = [
            EventInstance(event_name="Night A", promoter=None),
            EventInstance(event_name="Night B", promoter=None),
            EventInstance(event_name="Night C", promoter="Known Promoter"),
        ]

        groups = EventNameResearcher._group_by_promoter(instances)

        assert "Unknown Promoter" in groups
        assert len(groups["Unknown Promoter"]) == 2
        assert "Known Promoter" in groups

    def test_parse_name_changes_valid_json(self):
        """Valid JSON array of strings parses correctly."""
        from src.services.event_name_researcher import EventNameResearcher

        text = json.dumps([
            "Promoter A and Promoter B appear to be the same entity (rebrand 2005)",
        ])
        changes = EventNameResearcher._parse_name_changes(text)

        assert len(changes) == 1
        assert "rebrand 2005" in changes[0]

    def test_parse_name_changes_invalid_json(self):
        """Invalid text returns empty list."""
        from src.services.event_name_researcher import EventNameResearcher

        changes = EventNameResearcher._parse_name_changes(
            "No connections were found between these promoters."
        )
        assert changes == []


# ======================================================================
# TestPromoterResearcher
# ======================================================================


class TestPromoterResearcher:
    """Tests for the PromoterResearcher service."""

    @pytest.mark.asyncio
    async def test_research_happy_path(self):
        """Web search and LLM produce a promoter profile."""
        from src.services.promoter_researcher import PromoterResearcher

        web_search = _mock_web_search(results=[
            _search_result(
                title="Bugged Out Promotions",
                url="https://ra.co/promoters/buggedout",
            ),
        ])
        llm = _mock_llm(
            response=(
                "BASED_IN:\n"
                "- London, England, UK\n\n"
                "EVENT_HISTORY:\n"
                "- Bugged Out! at Fabric, 2001\n"
                "- Bugged Out Weekender at Butlins, 2008\n\n"
                "AFFILIATED_ARTISTS:\n"
                "- Erol Alkan\n"
                "- James Murphy\n\n"
                "AFFILIATED_VENUES:\n"
                "- Fabric\n"
                "- Sankeys"
            )
        )
        article_scraper = _mock_article_scraper(
            content=_article_content(
                title="Bugged Out! Profile",
                text="Bugged Out is a legendary London promoter.",
            )
        )

        researcher = PromoterResearcher(
            web_search=web_search,
            article_scraper=article_scraper,
            llm=llm,
        )

        result = await researcher.research("Bugged Out", city="London")

        assert result.entity_type == EntityType.PROMOTER
        assert result.promoter is not None
        assert result.promoter.name == "Bugged Out"

    @pytest.mark.asyncio
    async def test_research_no_results(self):
        """All sources return empty -- warnings produced."""
        from src.services.promoter_researcher import PromoterResearcher

        researcher = PromoterResearcher(
            web_search=_mock_web_search(results=[]),
            article_scraper=_mock_article_scraper(content=None),
            llm=_mock_llm(
                response=(
                    "BASED_IN:\nNONE\n\n"
                    "EVENT_HISTORY:\nNONE\n\n"
                    "AFFILIATED_ARTISTS:\nNONE\n\n"
                    "AFFILIATED_VENUES:\nNONE"
                )
            ),
        )

        result = await researcher.research("UnknownPromoter999")

        assert any("no web results" in w.lower() for w in result.warnings)

    def test_parse_promoter_extraction_full(self):
        """Full text with all 4 sections parses correctly."""
        from src.services.promoter_researcher import PromoterResearcher

        text = (
            "BASED_IN:\n"
            "- Berlin, Brandenburg, Germany\n\n"
            "EVENT_HISTORY:\n"
            "- Tresor Opening Night 1991\n"
            "- Tresor Records Showcase 1993\n\n"
            "AFFILIATED_ARTISTS:\n"
            "- Jeff Mills\n"
            "- Dimitri Hegemann\n\n"
            "AFFILIATED_VENUES:\n"
            "- Tresor\n"
            "- Globus"
        )

        parsed = PromoterResearcher._parse_promoter_extraction(text)

        assert parsed["based_city"] == "Berlin"
        assert parsed["based_region"] == "Brandenburg"
        assert parsed["based_country"] == "Germany"
        assert len(parsed["event_history"]) == 2
        assert "Tresor Opening Night" in parsed["event_history"][0]
        assert len(parsed["affiliated_artists"]) == 2
        assert "Jeff Mills" in parsed["affiliated_artists"][0]
        assert len(parsed["affiliated_venues"]) == 2

    def test_parse_promoter_extraction_all_none(self):
        """All sections set to NONE returns empty defaults."""
        from src.services.promoter_researcher import PromoterResearcher

        text = (
            "BASED_IN:\nNONE\n\n"
            "EVENT_HISTORY:\nNONE\n\n"
            "AFFILIATED_ARTISTS:\nNONE\n\n"
            "AFFILIATED_VENUES:\nNONE"
        )

        parsed = PromoterResearcher._parse_promoter_extraction(text)

        assert parsed["based_city"] is None
        assert parsed["based_region"] is None
        assert parsed["based_country"] is None
        assert parsed["event_history"] == []
        assert parsed["affiliated_artists"] == []
        assert parsed["affiliated_venues"] == []

    def test_parse_promoter_extraction_based_in_parsing(self):
        """BASED_IN with 3 comma-separated parts extracts city/region/country."""
        from src.services.promoter_researcher import PromoterResearcher

        text = (
            "BASED_IN:\n"
            "- Berlin, Brandenburg, Germany\n\n"
            "EVENT_HISTORY:\nNONE\n\n"
            "AFFILIATED_ARTISTS:\nNONE\n\n"
            "AFFILIATED_VENUES:\nNONE"
        )

        parsed = PromoterResearcher._parse_promoter_extraction(text)

        assert parsed["based_city"] == "Berlin"
        assert parsed["based_region"] == "Brandenburg"
        assert parsed["based_country"] == "Germany"


# ======================================================================
# TestInterconnectionService
# ======================================================================


class TestInterconnectionService:
    """Tests for the InterconnectionService."""

    def _make_citation_service(self):
        """Build a mock CitationService that returns real Citation objects."""
        service = MagicMock()
        service.build_citation = MagicMock(
            side_effect=lambda text, source_name, source_type, **kw: Citation(
                text=text,
                source_type=source_type,
                source_name=source_name,
                source_url=kw.get("source_url"),
                source_date=None,
                tier=6,
            )
        )
        return service

    def _make_research_results(self):
        """Build a small list of ResearchResult fixtures."""
        artist_result = ResearchResult(
            entity_type=EntityType.ARTIST,
            entity_name="carl cox",
            artist=Artist(
                name="carl cox",
                confidence=0.9,
                releases=[Release(title="Phat Trax", label="Intec", year=1995)],
                labels=[Label(name="Intec")],
                city="Melbourne",
            ),
            sources_consulted=["music_databases"],
            confidence=0.9,
        )
        venue_result = ResearchResult(
            entity_type=EntityType.VENUE,
            entity_name="tresor",
            venue=Venue(
                name="Tresor",
                city="Berlin",
                confidence=0.85,
                history="Opened in 1991.",
                articles=[
                    ArticleReference(
                        title="Tresor History",
                        source="ra.co",
                        url="https://ra.co/clubs/tresor",
                        citation_tier=1,
                    )
                ],
            ),
            sources_consulted=["web_search_history"],
            confidence=0.85,
        )
        return [artist_result, venue_result]

    def _make_extracted_entities(self):
        """Build an ExtractedEntities fixture."""
        return ExtractedEntities(
            artists=[
                ExtractedEntity(text="Carl Cox", entity_type=EntityType.ARTIST, confidence=0.95),
            ],
            venue=ExtractedEntity(text="Tresor", entity_type=EntityType.VENUE, confidence=0.9),
            promoter=ExtractedEntity(
                text="Tresor Berlin", entity_type=EntityType.PROMOTER, confidence=0.7
            ),
            raw_ocr=_make_ocr_result(),
        )

    @pytest.mark.asyncio
    async def test_analyze_happy_path(self):
        """LLM returns valid JSON with relationships, patterns, and narrative."""
        from src.services.interconnection_service import InterconnectionService

        llm_response = json.dumps({
            "relationships": [
                {
                    "source": "carl cox",
                    "target": "tresor",
                    "type": "performed_at",
                    "details": "Carl Cox played at Tresor in 1993 [1]",
                    "source_citation": "intec",
                    "confidence": 0.8,
                },
            ],
            "patterns": [
                {
                    "type": "shared_scene",
                    "description": "Both entities are connected through the Berlin techno scene [1]",
                    "entities": ["carl cox", "tresor"],
                    "source_citation": "intec",
                },
            ],
            "narrative": "Carl Cox performed at Tresor during the early 90s Berlin scene [1].",
        })

        llm = _mock_llm(response=llm_response)
        citation_service = self._make_citation_service()

        service = InterconnectionService(
            llm_provider=llm,
            citation_service=citation_service,
        )

        result = await service.analyze(
            research_results=self._make_research_results(),
            entities=self._make_extracted_entities(),
        )

        assert isinstance(result, InterconnectionMap)
        assert len(result.edges) >= 1
        assert len(result.patterns) >= 1
        assert result.narrative is not None

    @pytest.mark.asyncio
    async def test_analyze_llm_failure(self):
        """LLM.complete raises Exception -- service raises LLMError."""
        from src.services.interconnection_service import InterconnectionService

        llm = _mock_llm()
        llm.complete = AsyncMock(side_effect=RuntimeError("API timeout"))
        citation_service = self._make_citation_service()

        service = InterconnectionService(
            llm_provider=llm,
            citation_service=citation_service,
        )

        with pytest.raises(LLMError):
            await service.analyze(
                research_results=self._make_research_results(),
                entities=self._make_extracted_entities(),
            )

    def test_parse_analysis_response_json_fence(self):
        """JSON in markdown fence parses correctly."""
        from src.services.interconnection_service import InterconnectionService

        service = InterconnectionService(
            llm_provider=_mock_llm(),
            citation_service=self._make_citation_service(),
        )

        response = '```json\n{"relationships": [], "patterns": [], "narrative": "test"}\n```'
        parsed = service._parse_analysis_response(response)

        assert parsed["narrative"] == "test"
        assert parsed["relationships"] == []

    def test_parse_analysis_response_bare_json(self):
        """Bare JSON object parses correctly."""
        from src.services.interconnection_service import InterconnectionService

        service = InterconnectionService(
            llm_provider=_mock_llm(),
            citation_service=self._make_citation_service(),
        )

        response = '{"relationships": [{"source": "a", "target": "b"}], "patterns": [], "narrative": null}'
        parsed = service._parse_analysis_response(response)

        assert len(parsed["relationships"]) == 1
        assert parsed["narrative"] is None

    def test_parse_analysis_response_invalid(self):
        """Non-JSON text raises LLMError."""
        from src.services.interconnection_service import InterconnectionService

        service = InterconnectionService(
            llm_provider=_mock_llm(),
            citation_service=self._make_citation_service(),
        )

        with pytest.raises(LLMError):
            service._parse_analysis_response("This is not JSON at all and has no braces")

    def test_validate_citations(self):
        """Relationships with matching citations pass; non-matching are discarded."""
        from src.services.interconnection_service import InterconnectionService

        service = InterconnectionService(
            llm_provider=_mock_llm(),
            citation_service=self._make_citation_service(),
        )

        research_results = self._make_research_results()

        relationships = [
            {
                "source": "carl cox",
                "target": "tresor",
                "type": "released_on",
                "details": "Released on Intec",
                "source_citation": "intec",  # matches Label name "Intec"
                "confidence": 0.8,
            },
            {
                "source": "carl cox",
                "target": "tresor",
                "type": "imagined",
                "details": "Imagined connection",
                "source_citation": "totally_fabricated_source_xyz_99",
                "confidence": 0.5,
            },
        ]

        validated = service._validate_citations(relationships, research_results)

        # First should pass (intec matches the label "Intec")
        # Second should be discarded (no matching source)
        assert len(validated) == 1
        assert validated[0]["source_citation"] == "intec"

    def test_build_nodes(self):
        """Nodes built from research results + extracted entities with no duplicates."""
        from src.services.interconnection_service import InterconnectionService

        research_results = self._make_research_results()
        entities = self._make_extracted_entities()

        nodes = InterconnectionService._build_nodes(research_results, entities)

        # Research results contribute: "carl cox" (ARTIST) and "tresor" (VENUE)
        # Extracted entities: "Carl Cox" (ARTIST, duplicate), "Tresor" (VENUE, duplicate),
        # "Tresor Berlin" (PROMOTER, new)
        node_names = [n.name.lower() for n in nodes]

        assert "carl cox" in node_names
        assert "tresor" in node_names
        assert "tresor berlin" in node_names
        # No duplicates -- carl cox and tresor appear only once each
        assert node_names.count("carl cox") == 1
        assert node_names.count("tresor") == 1

    def test_penalise_uncertain(self):
        """Edge with [UNCERTAIN] in details has confidence reduced by 0.3."""
        from src.services.interconnection_service import InterconnectionService

        edges = [
            RelationshipEdge(
                source="A",
                target="B",
                relationship_type="test",
                confidence=0.8,
                details="Some [UNCERTAIN] connection",
                citations=[],
            ),
            RelationshipEdge(
                source="C",
                target="D",
                relationship_type="test",
                confidence=0.9,
                details="A certain connection",
                citations=[],
            ),
        ]

        result = InterconnectionService._penalise_uncertain(edges)

        assert len(result) == 2
        # First edge: 0.8 - 0.3 = 0.5
        assert abs(result[0].confidence - 0.5) < 0.001
        # Second edge: unchanged
        assert abs(result[1].confidence - 0.9) < 0.001

    def test_penalise_geographic_mismatch(self):
        """Two entities in different cities get a confidence penalty."""
        from src.services.interconnection_service import InterconnectionService

        # Research results: carl cox in Melbourne, tresor in Berlin
        research_results = self._make_research_results()

        edges = [
            RelationshipEdge(
                source="carl cox",
                target="tresor",
                relationship_type="performed_at",
                confidence=0.8,
                details="Performed at venue",
                citations=[],
            ),
        ]

        result = InterconnectionService._penalise_geographic_mismatch(edges, research_results)

        # Melbourne != Berlin, so penalty of 0.4 -> 0.8 - 0.4 = 0.4
        assert len(result) == 1
        assert abs(result[0].confidence - 0.4) < 0.001

    def test_penalise_geographic_same_city(self):
        """Two entities in the same city get no penalty."""
        from src.services.interconnection_service import InterconnectionService

        # Both in Berlin
        results = [
            ResearchResult(
                entity_type=EntityType.ARTIST,
                entity_name="artist_a",
                artist=Artist(name="artist_a", confidence=0.9, city="Berlin"),
                confidence=0.9,
            ),
            ResearchResult(
                entity_type=EntityType.VENUE,
                entity_name="venue_b",
                venue=Venue(name="venue_b", city="Berlin", confidence=0.85),
                confidence=0.85,
            ),
        ]

        edges = [
            RelationshipEdge(
                source="artist_a",
                target="venue_b",
                relationship_type="local",
                confidence=0.8,
                details="Local connection",
                citations=[],
            ),
        ]

        result = InterconnectionService._penalise_geographic_mismatch(edges, results)

        # Same city -- no penalty
        assert abs(result[0].confidence - 0.8) < 0.001

    def test_collect_all_citations(self):
        """Citations from edges and patterns are deduplicated."""
        from src.services.interconnection_service import InterconnectionService

        c1 = Citation(
            text="Fact 1",
            source_type="research",
            source_name="Source A",
            tier=3,
        )
        c2 = Citation(
            text="Fact 2",
            source_type="research",
            source_name="Source B",
            tier=2,
        )
        # c3 is a duplicate of c1 (same source_name + text)
        c3 = Citation(
            text="Fact 1",
            source_type="research",
            source_name="Source A",
            tier=3,
        )

        edges = [
            RelationshipEdge(
                source="A",
                target="B",
                relationship_type="test",
                confidence=0.8,
                citations=[c1, c2],
            ),
        ]
        patterns = [
            PatternInsight(
                pattern_type="test",
                description="A pattern",
                involved_entities=["A", "B"],
                citations=[c3],  # duplicate of c1
            ),
        ]

        result = InterconnectionService._collect_all_citations(edges, patterns)

        # c1 and c2 from edges, c3 is a duplicate of c1 so deduplicated
        assert len(result) == 2
        source_names = [c.source_name for c in result]
        assert "Source A" in source_names
        assert "Source B" in source_names

    def test_compile_research_context(self):
        """Compiled context text contains entity names and source index."""
        from src.services.interconnection_service import InterconnectionService

        research_results = self._make_research_results()

        context = InterconnectionService._compile_research_context(research_results)

        # Should contain entity names
        assert "carl cox" in context.lower()
        assert "tresor" in context.lower()
        # Should contain at least one source index reference
        assert "[1]" in context
        # Should contain release/label information
        assert "Phat Trax" in context
        assert "Intec" in context


# ======================================================================
# Extended ArtistResearcher tests — coverage additions
# ======================================================================


class TestArtistResearcherExtended:
    """Additional tests for ArtistResearcher targeting uncovered lines."""

    @pytest.mark.asyncio
    async def test_search_music_databases_cache_hit(self):
        """Cache hit for artist search skips provider calls and returns cached tuple."""
        from src.services.artist_researcher import ArtistResearcher

        cached_value = {
            "discogs_id": "999",
            "musicbrainz_id": "mb-uuid-123",
            "confidence": 0.88,
            "provider_ids": {"discogs_api": "999", "musicbrainz_api": "mb-uuid-123"},
        }
        cache = _mock_cache()
        cache.get = AsyncMock(return_value=cached_value)

        db = _mock_music_db(provider_name="discogs_api")

        researcher = ArtistResearcher(
            music_dbs=[db],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=cache,
        )

        discogs_id, mb_id, confidence, provider_ids = await researcher._search_music_databases(
            "Carl Cox"
        )

        assert discogs_id == "999"
        assert mb_id == "mb-uuid-123"
        assert confidence == 0.88
        assert provider_ids == {"discogs_api": "999", "musicbrainz_api": "mb-uuid-123"}
        # Provider was never called because cache hit
        db.search_artist.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_search_music_databases_multiple_providers_cross_reference(self):
        """Two providers match the same artist — cross-reference bonus applied."""
        from src.services.artist_researcher import ArtistResearcher

        db_discogs = _mock_music_db(
            provider_name="discogs_api",
            search_results=[
                ArtistSearchResult(id="123", name="Carl Cox", confidence=0.85),
            ],
        )
        db_musicbrainz = _mock_music_db(
            provider_name="musicbrainz_api",
            search_results=[
                ArtistSearchResult(id="mb-uuid-123", name="Carl Cox", confidence=0.80),
            ],
        )

        researcher = ArtistResearcher(
            music_dbs=[db_discogs, db_musicbrainz],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=_mock_cache(),
        )

        discogs_id, mb_id, confidence, provider_ids = await researcher._search_music_databases(
            "Carl Cox"
        )

        assert discogs_id == "123"
        assert mb_id == "mb-uuid-123"
        # Cross-reference bonus: 2 providers * 0.05 = 0.1 bonus
        # Base was max(0.85, 0.80) ~ 0.85 with fuzzy, +0.1 = 0.95
        assert confidence > 0.85
        assert len(provider_ids) == 2

    @pytest.mark.asyncio
    async def test_search_music_databases_provider_raises_research_error(self):
        """Provider raises ResearchError — that provider is skipped gracefully."""
        from src.services.artist_researcher import ArtistResearcher
        from src.utils.errors import ResearchError

        db = _mock_music_db(provider_name="discogs_api")
        db.search_artist = AsyncMock(side_effect=ResearchError("API down"))

        researcher = ArtistResearcher(
            music_dbs=[db],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=_mock_cache(),
        )

        discogs_id, mb_id, confidence, provider_ids = await researcher._search_music_databases(
            "Carl Cox"
        )

        assert discogs_id is None
        assert mb_id is None
        assert confidence == 0.0
        assert provider_ids == {}

    @pytest.mark.asyncio
    async def test_search_music_databases_provider_raises_unexpected_error(self):
        """Provider raises an unexpected Exception — that provider is skipped."""
        from src.services.artist_researcher import ArtistResearcher

        db = _mock_music_db(provider_name="discogs_api")
        db.search_artist = AsyncMock(side_effect=TypeError("bad arg"))

        researcher = ArtistResearcher(
            music_dbs=[db],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=_mock_cache(),
        )

        discogs_id, mb_id, confidence, provider_ids = await researcher._search_music_databases(
            "Carl Cox"
        )

        assert discogs_id is None
        assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_search_music_databases_no_fuzzy_match(self):
        """Provider returns results but none fuzzy-match the query name."""
        from src.services.artist_researcher import ArtistResearcher

        db = _mock_music_db(
            provider_name="discogs_api",
            search_results=[
                ArtistSearchResult(id="999", name="Totally Different Artist", confidence=0.95),
            ],
        )

        researcher = ArtistResearcher(
            music_dbs=[db],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=_mock_cache(),
        )

        discogs_id, mb_id, confidence, provider_ids = await researcher._search_music_databases(
            "Carl Cox"
        )

        # No fuzzy match found, so IDs remain None
        assert discogs_id is None
        assert provider_ids == {}

    @pytest.mark.asyncio
    async def test_search_music_databases_non_discogs_non_musicbrainz_provider(self):
        """A provider that is not Discogs or MusicBrainz stores its ID correctly."""
        from src.services.artist_researcher import ArtistResearcher

        db = _mock_music_db(
            provider_name="beatport_api",
            search_results=[
                ArtistSearchResult(id="bp-123", name="Carl Cox", confidence=0.9),
            ],
        )

        researcher = ArtistResearcher(
            music_dbs=[db],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=_mock_cache(),
        )

        discogs_id, mb_id, confidence, provider_ids = await researcher._search_music_databases(
            "Carl Cox"
        )

        # Not discogs or musicbrainz, so those stay None
        assert discogs_id is None
        assert mb_id is None
        # But the provider_ids dict should have the beatport entry
        assert "beatport_api" in provider_ids
        assert provider_ids["beatport_api"] == "bp-123"
        assert confidence > 0.0

    @pytest.mark.asyncio
    async def test_fetch_discography_dedup_and_before_date(self):
        """Discography fetch deduplicates releases and respects before_date."""
        from src.services.artist_researcher import ArtistResearcher

        db = _mock_music_db(
            provider_name="discogs_api",
            releases=[
                Release(title="Track A", label="Intec", year=1995),
                Release(title="Track A", label="Intec", year=1995),  # duplicate
                Release(title="Track B", label="Intec", year=1996),
            ],
            labels=[Label(name="Intec"), Label(name="Intec")],  # duplicate label
        )

        researcher = ArtistResearcher(
            music_dbs=[db],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=_mock_cache(),
        )

        releases, labels = await researcher._fetch_discography(
            "Carl Cox",
            discogs_id="123",
            musicbrainz_id=None,
            before_date=date(2000, 1, 1),
            provider_ids={},
        )

        # Releases should be deduplicated
        assert len(releases) == 2
        # Labels should be deduplicated
        assert len(labels) == 1
        assert labels[0].name == "Intec"

    @pytest.mark.asyncio
    async def test_fetch_discography_rate_limited_falls_to_next_provider(self):
        """Rate-limited Discogs provider falls through to next Discogs provider."""
        from src.services.artist_researcher import ArtistResearcher
        from src.utils.errors import RateLimitError

        db_primary = _mock_music_db(provider_name="discogs_api")
        db_primary.get_artist_releases = AsyncMock(side_effect=RateLimitError("rate limited"))

        db_fallback = _mock_music_db(
            provider_name="discogs_scrape",
            releases=[Release(title="Fallback Track", label="Intec", year=1999)],
            labels=[Label(name="Intec")],
        )

        researcher = ArtistResearcher(
            music_dbs=[db_primary, db_fallback],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=_mock_cache(),
        )

        releases, labels = await researcher._fetch_discography(
            "Carl Cox",
            discogs_id="123",
            musicbrainz_id=None,
            before_date=None,
            provider_ids={},
        )

        assert len(releases) == 1
        assert releases[0].title == "Fallback Track"

    @pytest.mark.asyncio
    async def test_fetch_discography_musicbrainz_supplement(self):
        """MusicBrainz supplement adds non-duplicate releases."""
        from src.services.artist_researcher import ArtistResearcher

        db_discogs = _mock_music_db(
            provider_name="discogs_api",
            releases=[Release(title="Track A", label="Intec", year=1995)],
            labels=[Label(name="Intec")],
        )

        db_mb = _mock_music_db(
            provider_name="musicbrainz_api",
            releases=[
                Release(title="Track A", label="Intec", year=1995),  # duplicate
                Release(title="Track C", label="Warp", year=1997),  # new
            ],
            labels=[Label(name="Warp")],
        )

        researcher = ArtistResearcher(
            music_dbs=[db_discogs, db_mb],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=_mock_cache(),
        )

        releases, labels = await researcher._fetch_discography(
            "Carl Cox",
            discogs_id="123",
            musicbrainz_id="mb-uuid",
            before_date=None,
            provider_ids={},
        )

        # Track A (from Discogs) + Track C (from MB, non-duplicate)
        release_titles = [r.title for r in releases]
        assert "Track A" in release_titles
        assert "Track C" in release_titles

    @pytest.mark.asyncio
    async def test_fetch_discography_additional_providers(self):
        """Additional providers (Bandcamp, etc.) are used when in provider_ids."""
        from src.services.artist_researcher import ArtistResearcher

        db_discogs = _mock_music_db(
            provider_name="discogs_api",
            releases=[Release(title="Track A", label="Intec", year=1995)],
            labels=[Label(name="Intec")],
        )

        db_bandcamp = _mock_music_db(
            provider_name="bandcamp_api",
            releases=[Release(title="Track D", label="Self-released", year=2010)],
            labels=[],
        )

        researcher = ArtistResearcher(
            music_dbs=[db_discogs, db_bandcamp],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=_mock_cache(),
        )

        releases, labels = await researcher._fetch_discography(
            "Carl Cox",
            discogs_id="123",
            musicbrainz_id=None,
            before_date=None,
            provider_ids={"bandcamp_api": "bc-123"},
        )

        release_titles = [r.title for r in releases]
        assert "Track A" in release_titles
        assert "Track D" in release_titles

    @pytest.mark.asyncio
    async def test_fetch_discography_additional_provider_failure(self):
        """Additional provider fails gracefully, primary results still returned."""
        from src.services.artist_researcher import ArtistResearcher

        db_discogs = _mock_music_db(
            provider_name="discogs_api",
            releases=[Release(title="Track A", label="Intec", year=1995)],
            labels=[Label(name="Intec")],
        )

        db_bandcamp = _mock_music_db(provider_name="bandcamp_api")
        db_bandcamp.get_artist_releases = AsyncMock(side_effect=RuntimeError("timeout"))

        researcher = ArtistResearcher(
            music_dbs=[db_discogs, db_bandcamp],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=_mock_cache(),
        )

        releases, labels = await researcher._fetch_discography(
            "Carl Cox",
            discogs_id="123",
            musicbrainz_id=None,
            before_date=None,
            provider_ids={"bandcamp_api": "bc-123"},
        )

        assert len(releases) == 1
        assert releases[0].title == "Track A"

    @pytest.mark.asyncio
    async def test_retrieve_from_corpus_no_vector_store(self):
        """No vector store configured — returns empty list."""
        from src.services.artist_researcher import ArtistResearcher

        researcher = ArtistResearcher(
            music_dbs=[],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            vector_store=None,
        )

        refs = await researcher._retrieve_from_corpus("Carl Cox", before_date=None)

        assert refs == []

    @pytest.mark.asyncio
    async def test_retrieve_from_corpus_vector_store_unavailable(self):
        """Vector store exists but is_available returns False — returns empty list."""
        from src.services.artist_researcher import ArtistResearcher

        vector_store = _mock_vector_store(results=None)
        # is_available returns False when results=None in helper

        researcher = ArtistResearcher(
            music_dbs=[],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            vector_store=vector_store,
        )

        refs = await researcher._retrieve_from_corpus("Carl Cox", before_date=None)

        assert refs == []

    @pytest.mark.asyncio
    async def test_retrieve_from_corpus_with_results(self):
        """Vector store returns matching chunks — converted to ArticleReferences."""
        from src.models.rag import DocumentChunk, RetrievedChunk
        from src.services.artist_researcher import ArtistResearcher

        chunk = DocumentChunk(
            chunk_id="c1",
            text="Carl Cox is a legendary British DJ known for techno." * 5,
            source_id="src1",
            source_title="Energy Flash",
            source_type="book",
            publication_date=date(1998, 1, 1),
            citation_tier=1,
        )
        retrieved = RetrievedChunk(
            chunk=chunk,
            similarity_score=0.85,
            formatted_citation="Energy Flash, p.142",
        )

        vector_store = _mock_vector_store(results=[retrieved])

        researcher = ArtistResearcher(
            music_dbs=[],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            vector_store=vector_store,
        )

        refs = await researcher._retrieve_from_corpus("Carl Cox", before_date=None)

        assert len(refs) == 1
        assert refs[0].title == "Energy Flash"
        assert refs[0].article_type == "book"
        assert refs[0].citation_tier == 1

    @pytest.mark.asyncio
    async def test_retrieve_from_corpus_low_similarity_filtered(self):
        """Chunks with similarity < 0.7 are filtered out."""
        from src.models.rag import DocumentChunk, RetrievedChunk
        from src.services.artist_researcher import ArtistResearcher

        chunk = DocumentChunk(
            chunk_id="c1",
            text="Some vaguely related text" * 10,
            source_id="src1",
            source_title="Random Article",
            source_type="article",
            citation_tier=5,
        )
        retrieved = RetrievedChunk(
            chunk=chunk,
            similarity_score=0.5,  # Below 0.7 threshold
        )

        vector_store = _mock_vector_store(results=[retrieved])

        researcher = ArtistResearcher(
            music_dbs=[],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            vector_store=vector_store,
        )

        refs = await researcher._retrieve_from_corpus("Carl Cox", before_date=None)

        assert refs == []

    @pytest.mark.asyncio
    async def test_retrieve_from_corpus_exception_handled(self):
        """Vector store query exception returns empty list."""
        from src.services.artist_researcher import ArtistResearcher

        vector_store = _mock_vector_store(results=[])
        vector_store.is_available = MagicMock(return_value=True)
        vector_store.query = AsyncMock(side_effect=RuntimeError("ChromaDB down"))

        researcher = ArtistResearcher(
            music_dbs=[],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            vector_store=vector_store,
        )

        refs = await researcher._retrieve_from_corpus("Carl Cox", before_date=None)

        assert refs == []

    @pytest.mark.asyncio
    async def test_retrieve_from_corpus_with_before_date_filter(self):
        """before_date is passed as a filter to the vector store."""
        from src.models.rag import DocumentChunk, RetrievedChunk
        from src.services.artist_researcher import ArtistResearcher

        chunk = DocumentChunk(
            chunk_id="c1",
            text="Carl Cox is a DJ" * 20,
            source_id="src1",
            source_title="Energy Flash",
            source_type="book",
            citation_tier=1,
        )
        retrieved = RetrievedChunk(chunk=chunk, similarity_score=0.9)
        vector_store = _mock_vector_store(results=[retrieved])

        researcher = ArtistResearcher(
            music_dbs=[],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            vector_store=vector_store,
        )

        refs = await researcher._retrieve_from_corpus("Carl Cox", before_date=date(1998, 1, 1))

        assert len(refs) == 1
        # Verify the query was called with filters
        vector_store.query.assert_awaited_once()
        call_kwargs = vector_store.query.call_args
        assert call_kwargs.kwargs.get("filters") is not None

    @pytest.mark.asyncio
    async def test_synthesize_profile_with_data(self):
        """Profile synthesis returns LLM-generated summary when data available."""
        from src.services.artist_researcher import ArtistResearcher

        llm = _mock_llm(response="Carl Cox is a British techno DJ based in Melbourne.")

        researcher = ArtistResearcher(
            music_dbs=[],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=llm,
        )

        summary = await researcher._synthesize_profile(
            "Carl Cox",
            releases=[Release(title="Phat Trax", label="Intec", year=1995)],
            labels=[Label(name="Intec")],
            city="Melbourne",
        )

        assert summary is not None
        assert "Carl Cox" in summary

    @pytest.mark.asyncio
    async def test_synthesize_profile_no_data(self):
        """Profile synthesis returns None when no context data available."""
        from src.services.artist_researcher import ArtistResearcher

        researcher = ArtistResearcher(
            music_dbs=[],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
        )

        summary = await researcher._synthesize_profile(
            "Unknown DJ",
            releases=[],
            labels=[],
            city=None,
        )

        assert summary is None

    @pytest.mark.asyncio
    async def test_synthesize_profile_llm_failure(self):
        """LLM failure during synthesis returns None."""
        from src.services.artist_researcher import ArtistResearcher

        llm = _mock_llm()
        llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))

        researcher = ArtistResearcher(
            music_dbs=[],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=llm,
        )

        summary = await researcher._synthesize_profile(
            "Carl Cox",
            releases=[Release(title="Track", label="Intec", year=1995)],
            labels=[],
            city=None,
        )

        assert summary is None

    @pytest.mark.asyncio
    async def test_synthesize_profile_empty_response(self):
        """LLM returns empty string — synthesis returns None."""
        from src.services.artist_researcher import ArtistResearcher

        llm = _mock_llm(response="   ")

        researcher = ArtistResearcher(
            music_dbs=[],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=llm,
        )

        summary = await researcher._synthesize_profile(
            "Carl Cox",
            releases=[Release(title="Track", label="Intec", year=1995)],
            labels=[],
            city=None,
        )

        assert summary is None

    @pytest.mark.asyncio
    async def test_research_with_before_date_and_profile_synthesis(self):
        """Full pipeline with before_date and enough data for profile synthesis."""
        from src.models.rag import DocumentChunk, RetrievedChunk
        from src.services.artist_researcher import ArtistResearcher

        db = _mock_music_db(
            provider_name="discogs_api",
            search_results=[
                ArtistSearchResult(id="123", name="Carl Cox", confidence=0.95),
            ],
            releases=[Release(title="Phat Trax", label="Intec", year=1995)],
            labels=[Label(name="Intec")],
        )

        chunk = DocumentChunk(
            chunk_id="c1",
            text="Carl Cox biography text" * 20,
            source_id="src1",
            source_title="Energy Flash",
            source_type="book",
            citation_tier=1,
        )
        retrieved = RetrievedChunk(chunk=chunk, similarity_score=0.9)
        vector_store = _mock_vector_store(results=[retrieved])

        llm = _mock_llm(response="Carl Cox is a British techno DJ.")

        researcher = ArtistResearcher(
            music_dbs=[db],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=llm,
            cache=_mock_cache(),
            vector_store=vector_store,
        )

        result = await researcher.research(
            "Carl Cox",
            before_date=date(2000, 1, 1),
            city="Berlin",
        )

        assert result.entity_type == EntityType.ARTIST
        assert result.artist is not None
        assert "rag_corpus" in result.sources_consulted
        assert "music_databases" in result.sources_consulted
        # Profile synthesis should have run because 2+ data sources
        assert result.artist.profile_summary is not None

    @pytest.mark.asyncio
    async def test_research_with_beatport_provider_id(self):
        """Artist with beatport provider ID gets beatport_url populated."""
        from src.services.artist_researcher import ArtistResearcher

        db_discogs = _mock_music_db(
            provider_name="discogs_api",
            search_results=[
                ArtistSearchResult(id="123", name="Carl Cox", confidence=0.95),
            ],
            releases=[Release(title="Track", label="Intec", year=1995)],
            labels=[Label(name="Intec")],
        )

        db_beatport = _mock_music_db(
            provider_name="beatport",
            search_results=[
                ArtistSearchResult(id="bp-456", name="Carl Cox", confidence=0.90),
            ],
            releases=[],
            labels=[],
        )

        researcher = ArtistResearcher(
            music_dbs=[db_discogs, db_beatport],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=_mock_cache(),
            vector_store=None,
        )

        result = await researcher.research("Carl Cox")

        assert result.artist is not None
        assert result.artist.discogs_id == 123
        assert result.artist.beatport_url is not None
        assert "bp-456" in result.artist.beatport_url

    @pytest.mark.asyncio
    async def test_cache_get_failure_returns_none(self):
        """Cache get failure returns None silently."""
        from src.services.artist_researcher import ArtistResearcher

        cache = _mock_cache()
        cache.get = AsyncMock(side_effect=RuntimeError("Redis down"))

        researcher = ArtistResearcher(
            music_dbs=[],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=cache,
        )

        result = await researcher._cache_get("some_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_set_failure_handled_silently(self):
        """Cache set failure is caught and does not raise."""
        from src.services.artist_researcher import ArtistResearcher

        cache = _mock_cache()
        cache.set = AsyncMock(side_effect=RuntimeError("Redis down"))

        researcher = ArtistResearcher(
            music_dbs=[],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=cache,
        )

        # Should not raise
        await researcher._cache_set("some_key", {"data": True})

    @pytest.mark.asyncio
    async def test_cache_get_no_cache_configured(self):
        """No cache configured — returns None."""
        from src.services.artist_researcher import ArtistResearcher

        researcher = ArtistResearcher(
            music_dbs=[],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=None,
        )

        result = await researcher._cache_get("some_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_set_no_cache_configured(self):
        """No cache configured — set is a no-op."""
        from src.services.artist_researcher import ArtistResearcher

        researcher = ArtistResearcher(
            music_dbs=[],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=None,
        )

        # Should not raise
        await researcher._cache_set("some_key", {"data": True})

    def test_extract_artist_geography_majority_city(self):
        """Majority of appearances in one city — returns that city."""
        from src.services.artist_researcher import ArtistResearcher

        appearances = [
            EventAppearance(event_name="A", city="Berlin"),
            EventAppearance(event_name="B", city="Berlin"),
            EventAppearance(event_name="C", city="London"),
        ]

        city, region, country = ArtistResearcher._extract_artist_geography(appearances, None)

        assert city == "Berlin"
        assert region is None
        assert country is None

    def test_extract_artist_geography_no_cities(self):
        """No city info in appearances — returns (None, None, None)."""
        from src.services.artist_researcher import ArtistResearcher

        appearances = [
            EventAppearance(event_name="A"),
            EventAppearance(event_name="B"),
        ]

        city, region, country = ArtistResearcher._extract_artist_geography(appearances, None)

        assert city is None

    def test_extract_artist_geography_no_majority(self):
        """No city has >= 50% — returns (None, None, None)."""
        from src.services.artist_researcher import ArtistResearcher

        appearances = [
            EventAppearance(event_name="A", city="Berlin"),
            EventAppearance(event_name="B", city="London"),
            EventAppearance(event_name="C", city="Tokyo"),
            EventAppearance(event_name="D", city="Paris"),
        ]

        city, region, country = ArtistResearcher._extract_artist_geography(appearances, None)

        # Each city has 25%, none >= 50%
        assert city is None

    def test_extract_relevant_snippet_truncation(self):
        """Snippet exceeding max_length is truncated with ellipsis."""
        from src.services.artist_researcher import ArtistResearcher

        # Build a text where the relevant sentence is very long
        long_sentence = "Carl Cox " + "played techno music " * 30 + "at the festival."
        text = long_sentence

        snippet = ArtistResearcher._extract_relevant_snippet(text, "carl cox", max_length=100)

        assert len(snippet) <= 103  # 100 + "..."
        assert snippet.endswith("...")


# ======================================================================
# Extended EventNameResearcher tests — coverage additions
# ======================================================================


class TestEventNameResearcherExtended:
    """Additional tests for EventNameResearcher targeting uncovered lines."""

    @pytest.mark.asyncio
    async def test_research_no_search_results(self):
        """When web search returns no results, warnings are produced."""
        from src.services.event_name_researcher import EventNameResearcher

        researcher = EventNameResearcher(
            web_search=_mock_web_search(results=[]),
            article_scraper=_mock_article_scraper(content=None),
            llm=_mock_llm(response="[]"),
        )

        result = await researcher.research("NonexistentEvent999")

        assert result.entity_type == EntityType.EVENT
        assert any("no web results" in w.lower() for w in result.warnings)

    @pytest.mark.asyncio
    async def test_research_with_promoter_name(self):
        """Research with promoter_name adds promoter-qualified search queries."""
        from src.services.event_name_researcher import EventNameResearcher

        instances_json = json.dumps([{
            "event_name": "Bugged Out!",
            "promoter": "Bugged Out Promotions",
            "venue": "Fabric",
            "city": "London",
            "date": "2001-05-12",
            "source_url": "https://ra.co/events/123",
        }])

        web_search = _mock_web_search(results=[
            _search_result(title="Bugged Out!", url="https://ra.co/events/123"),
        ])

        researcher = EventNameResearcher(
            web_search=web_search,
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(response=instances_json),
        )

        result = await researcher.research("Bugged Out!", promoter_name="Bugged Out Promotions")

        assert result.event_history is not None
        # Extra search queries should have been made for promoter
        assert web_search.search.await_count > 3  # base 3 + promoter queries

    @pytest.mark.asyncio
    async def test_research_with_multiple_promoters_detects_name_changes(self):
        """Multiple promoter groups triggers name-change detection."""
        from src.services.event_name_researcher import EventNameResearcher

        instances_json = json.dumps([
            {
                "event_name": "TestEvent",
                "promoter": "Promoter Alpha",
                "venue": "Fabric",
                "city": "London",
                "date": "2001-05-12",
                "source_url": None,
            },
            {
                "event_name": "TestEvent",
                "promoter": "Promoter Beta",
                "venue": "Fabric",
                "city": "London",
                "date": "2005-10-20",
                "source_url": None,
            },
        ])

        name_changes_json = json.dumps([
            "Promoter Alpha and Promoter Beta appear to be the same entity (rebrand 2004)"
        ])

        # LLM: first call returns instances, second call returns name changes
        llm = _mock_llm()
        llm.complete = AsyncMock(side_effect=[instances_json, name_changes_json, "[]"])

        web_search = _mock_web_search(results=[
            _search_result(title="TestEvent", url="https://ra.co/events/100"),
        ])

        researcher = EventNameResearcher(
            web_search=web_search,
            article_scraper=_mock_article_scraper(
                content=_article_content(text="TestEvent is a legendary party.")
            ),
            llm=llm,
        )

        result = await researcher.research("TestEvent")

        assert result.event_history is not None
        assert len(result.event_history.promoter_groups) == 2
        assert len(result.event_history.promoter_name_changes) >= 1

    @pytest.mark.asyncio
    async def test_extract_event_instances_llm_failure(self):
        """LLM failure returns empty instances list."""
        from src.services.event_name_researcher import EventNameResearcher

        llm = _mock_llm()
        llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))

        researcher = EventNameResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=llm,
        )

        instances = await researcher._extract_event_instances("TestEvent", ["some scraped text"])

        assert instances == []

    @pytest.mark.asyncio
    async def test_extract_event_instances_no_scraped_texts(self):
        """No scraped texts means no LLM call, returns empty list."""
        from src.services.event_name_researcher import EventNameResearcher

        llm = _mock_llm()

        researcher = EventNameResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=llm,
        )

        instances = await researcher._extract_event_instances("TestEvent", [])

        assert instances == []
        llm.complete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_scrape_results_handles_exceptions(self):
        """Scrape failures are caught; successful ones still returned."""
        from src.services.event_name_researcher import EventNameResearcher

        article_scraper = _mock_article_scraper()
        article_scraper.extract_content = AsyncMock(
            side_effect=[
                _article_content(text="Good content"),
                RuntimeError("Scrape failed"),
            ]
        )

        researcher = EventNameResearcher(
            web_search=_mock_web_search(),
            article_scraper=article_scraper,
            llm=_mock_llm(),
        )

        results = [
            _search_result(url="https://example.com/1"),
            _search_result(url="https://example.com/2"),
        ]
        texts = await researcher._scrape_results(results)

        assert len(texts) == 1
        assert "Good content" in texts[0]

    @pytest.mark.asyncio
    async def test_scrape_results_empty_list(self):
        """Empty results list returns empty texts."""
        from src.services.event_name_researcher import EventNameResearcher

        researcher = EventNameResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
        )

        texts = await researcher._scrape_results([])

        assert texts == []

    @pytest.mark.asyncio
    async def test_search_event_instances_deepens_when_thin(self):
        """Fewer than 3 search results triggers deeper queries."""
        from src.services.event_name_researcher import EventNameResearcher

        call_count = 0

        async def search_side_effect(query, num_results=15):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return [_search_result(url=f"https://ra.co/{call_count}")]
            return [_search_result(url=f"https://ra.co/deep/{call_count}")]

        web_search = _mock_web_search()
        web_search.search = AsyncMock(side_effect=search_side_effect)

        researcher = EventNameResearcher(
            web_search=web_search,
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
        )

        results = await researcher._search_event_instances("TestEvent", promoter_name=None)

        # Should have called at least the base search queries
        assert call_count >= 3

    @pytest.mark.asyncio
    async def test_detect_promoter_name_changes_single_promoter(self):
        """Single promoter group — no name change detection needed."""
        from src.services.event_name_researcher import EventNameResearcher

        researcher = EventNameResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
        )

        changes = await researcher._detect_promoter_name_changes(
            "TestEvent",
            {"Only Promoter": [EventInstance(event_name="TestEvent")]},
        )

        assert changes == []

    @pytest.mark.asyncio
    async def test_detect_promoter_name_changes_llm_failure(self):
        """LLM failure during promoter analysis returns empty list."""
        from src.services.event_name_researcher import EventNameResearcher

        llm = _mock_llm()
        llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))

        researcher = EventNameResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=llm,
        )

        changes = await researcher._detect_promoter_name_changes(
            "TestEvent",
            {
                "Promoter A": [EventInstance(event_name="TestEvent", promoter="Promoter A")],
                "Promoter B": [EventInstance(event_name="TestEvent", promoter="Promoter B")],
            },
        )

        assert changes == []

    def test_parse_event_instances_not_list(self):
        """JSON that parses to a dict (not list) returns empty list."""
        from src.services.event_name_researcher import EventNameResearcher

        instances = EventNameResearcher._parse_event_instances('{"key": "value"}')

        assert instances == []

    def test_parse_event_instances_non_dict_items(self):
        """JSON array with non-dict items — those items are skipped."""
        from src.services.event_name_researcher import EventNameResearcher

        text = json.dumps(["string item", 42, {"event_name": "Real Event", "venue": "Fabric"}])
        instances = EventNameResearcher._parse_event_instances(text)

        assert len(instances) == 1
        assert instances[0].event_name == "Real Event"

    def test_parse_event_instances_bracket_fallback(self):
        """JSON with leading text before the array is still parsed."""
        from src.services.event_name_researcher import EventNameResearcher

        text = 'Here are the results: [{"event_name": "BLOC", "venue": "Butlins"}]'
        instances = EventNameResearcher._parse_event_instances(text)

        assert len(instances) == 1
        assert instances[0].event_name == "BLOC"

    def test_parse_name_changes_non_list_json(self):
        """JSON that parses to a dict returns empty list."""
        from src.services.event_name_researcher import EventNameResearcher

        changes = EventNameResearcher._parse_name_changes('{"key": "value"}')

        assert changes == []

    def test_parse_name_changes_bracket_fallback(self):
        """Text with leading prose before JSON array still parses."""
        from src.services.event_name_researcher import EventNameResearcher

        text = 'Analysis: ["Promoter A and B are the same entity"]'
        changes = EventNameResearcher._parse_name_changes(text)

        assert len(changes) == 1

    @pytest.mark.asyncio
    async def test_retrieve_from_corpus_no_vector_store(self):
        """No vector store returns empty list."""
        from src.services.event_name_researcher import EventNameResearcher

        researcher = EventNameResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            vector_store=None,
        )

        refs = await researcher._retrieve_from_corpus("TestEvent")

        assert refs == []

    @pytest.mark.asyncio
    async def test_retrieve_from_corpus_exception(self):
        """Vector store exception returns empty list."""
        from src.services.event_name_researcher import EventNameResearcher

        vector_store = _mock_vector_store()
        vector_store.is_available = MagicMock(return_value=True)
        vector_store.query = AsyncMock(side_effect=RuntimeError("DB error"))

        researcher = EventNameResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            vector_store=vector_store,
        )

        refs = await researcher._retrieve_from_corpus("TestEvent")

        assert refs == []


# ======================================================================
# Extended InterconnectionService tests — coverage additions
# ======================================================================


class TestInterconnectionServiceExtended:
    """Additional tests for InterconnectionService targeting uncovered lines."""

    def _make_citation_service(self):
        """Build a mock CitationService that returns real Citation objects."""
        service = MagicMock()
        service.build_citation = MagicMock(
            side_effect=lambda text, source_name, source_type, **kw: Citation(
                text=text,
                source_type=source_type,
                source_name=source_name,
                source_url=kw.get("source_url"),
                source_date=None,
                tier=6,
            )
        )
        return service

    def test_build_edges_missing_source_or_target_skipped(self):
        """Relationships with empty source or target are skipped."""
        from src.services.interconnection_service import InterconnectionService

        service = InterconnectionService(
            llm_provider=_mock_llm(),
            citation_service=self._make_citation_service(),
        )

        relationships = [
            {"source": "", "target": "B", "type": "test"},
            {"source": "A", "target": "", "type": "test"},
            {"source": "A", "target": "B", "type": "test", "confidence": 0.8},
        ]

        edges = service._build_edges(relationships)

        assert len(edges) == 1
        assert edges[0].source == "A"
        assert edges[0].target == "B"

    def test_build_edges_confidence_clamping(self):
        """Confidence values outside [0, 1] are clamped."""
        from src.services.interconnection_service import InterconnectionService

        service = InterconnectionService(
            llm_provider=_mock_llm(),
            citation_service=self._make_citation_service(),
        )

        relationships = [
            {
                "source": "A",
                "target": "B",
                "type": "test",
                "confidence": 1.5,
                "source_citation": "src",
            },
            {
                "source": "C",
                "target": "D",
                "type": "test",
                "confidence": -0.5,
                "source_citation": "src",
            },
        ]

        edges = service._build_edges(relationships)

        assert edges[0].confidence == 1.0
        assert edges[1].confidence == 0.0

    def test_build_edges_default_type(self):
        """Missing 'type' key defaults to 'related'."""
        from src.services.interconnection_service import InterconnectionService

        service = InterconnectionService(
            llm_provider=_mock_llm(),
            citation_service=self._make_citation_service(),
        )

        relationships = [
            {"source": "A", "target": "B", "source_citation": "src"},
        ]

        edges = service._build_edges(relationships)

        assert len(edges) == 1
        assert edges[0].relationship_type == "related"

    def test_build_patterns_empty_description_skipped(self):
        """Patterns with empty description are skipped."""
        from src.services.interconnection_service import InterconnectionService

        service = InterconnectionService(
            llm_provider=_mock_llm(),
            citation_service=self._make_citation_service(),
        )

        raw_patterns = [
            {"type": "test", "description": "", "entities": ["A"]},
            {
                "type": "test",
                "description": "Valid pattern [1]",
                "entities": ["A", "B"],
                "source_citation": "ref1",
            },
        ]

        patterns = service._build_patterns(raw_patterns)

        assert len(patterns) == 1
        assert patterns[0].description == "Valid pattern [1]"
        assert len(patterns[0].citations) == 1

    def test_build_patterns_no_citation(self):
        """Pattern without source_citation gets no citations list."""
        from src.services.interconnection_service import InterconnectionService

        service = InterconnectionService(
            llm_provider=_mock_llm(),
            citation_service=self._make_citation_service(),
        )

        raw_patterns = [
            {"type": "test", "description": "A pattern", "entities": ["A"]},
        ]

        patterns = service._build_patterns(raw_patterns)

        assert len(patterns) == 1
        assert len(patterns[0].citations) == 0

    def test_parse_analysis_response_brace_fallback(self):
        """Response with leading text before JSON object is still parsed."""
        from src.services.interconnection_service import InterconnectionService

        service = InterconnectionService(
            llm_provider=_mock_llm(),
            citation_service=self._make_citation_service(),
        )

        response = (
            'Here is the analysis:\n{"relationships": [], "patterns": [], "narrative": "test"}'
        )
        parsed = service._parse_analysis_response(response)

        assert parsed["narrative"] == "test"

    def test_parse_analysis_response_non_dict_json_raises(self):
        """JSON that parses to a list (not dict) raises LLMError."""
        from src.services.interconnection_service import InterconnectionService

        service = InterconnectionService(
            llm_provider=_mock_llm(),
            citation_service=self._make_citation_service(),
        )

        with pytest.raises(LLMError):
            service._parse_analysis_response("[1, 2, 3]")

    def test_validate_citations_empty_citation_discarded(self):
        """Relationship with empty source_citation is discarded."""
        from src.services.interconnection_service import InterconnectionService

        service = InterconnectionService(
            llm_provider=_mock_llm(),
            citation_service=self._make_citation_service(),
        )

        relationships = [
            {"source": "A", "target": "B", "type": "test", "source_citation": ""},
            {"source": "A", "target": "B", "type": "test"},  # missing key
        ]

        research_results = [
            ResearchResult(
                entity_type=EntityType.ARTIST,
                entity_name="A",
                confidence=0.9,
            ),
        ]

        validated = service._validate_citations(relationships, research_results)

        assert len(validated) == 0

    def test_validate_citations_checks_promoter_articles(self):
        """Citations are validated against promoter article titles."""
        from src.services.interconnection_service import InterconnectionService

        service = InterconnectionService(
            llm_provider=_mock_llm(),
            citation_service=self._make_citation_service(),
        )

        research_results = [
            ResearchResult(
                entity_type=EntityType.PROMOTER,
                entity_name="promoter_x",
                promoter=Promoter(
                    name="promoter_x",
                    confidence=0.9,
                    articles=[
                        ArticleReference(
                            title="Promoter X History",
                            source="ra.co",
                            url="https://ra.co/promoters/x",
                            citation_tier=1,
                        )
                    ],
                ),
                confidence=0.9,
            ),
        ]

        relationships = [
            {
                "source": "A",
                "target": "promoter_x",
                "type": "promoted_by",
                "source_citation": "promoter x history",
                "confidence": 0.8,
            },
        ]

        validated = service._validate_citations(relationships, research_results)

        assert len(validated) == 1

    def test_validate_citations_checks_date_context_sources(self):
        """Citations are validated against date_context.sources."""
        from src.services.interconnection_service import InterconnectionService

        service = InterconnectionService(
            llm_provider=_mock_llm(),
            citation_service=self._make_citation_service(),
        )

        research_results = [
            ResearchResult(
                entity_type=EntityType.DATE,
                entity_name="March 1997",
                date_context=DateContext(
                    event_date=date(1997, 3, 15),
                    sources=[
                        ArticleReference(
                            title="Berlin Scene 1997",
                            source="ra.co",
                            url="https://ra.co/features/1997",
                            citation_tier=1,
                        )
                    ],
                ),
                confidence=0.7,
            ),
        ]

        relationships = [
            {
                "source": "A",
                "target": "B",
                "type": "shared_scene",
                "source_citation": "berlin scene 1997",
                "confidence": 0.8,
            },
        ]

        validated = service._validate_citations(relationships, research_results)

        assert len(validated) == 1

    def test_penalise_geographic_partial_city_mismatch(self):
        """One entity has a city, venue has different city — 0.2 penalty."""
        from src.services.interconnection_service import InterconnectionService

        research_results = [
            ResearchResult(
                entity_type=EntityType.ARTIST,
                entity_name="artist_a",
                artist=Artist(name="artist_a", confidence=0.9, city="Tokyo"),
                confidence=0.9,
            ),
            ResearchResult(
                entity_type=EntityType.VENUE,
                entity_name="venue_b",
                venue=Venue(name="venue_b", city="Berlin", confidence=0.85),
                confidence=0.85,
            ),
        ]

        edges = [
            RelationshipEdge(
                source="artist_a",
                target="unknown_entity",  # not in city_map
                relationship_type="test",
                confidence=0.8,
                details="Some connection",
                citations=[],
            ),
        ]

        result = InterconnectionService._penalise_geographic_mismatch(edges, research_results)

        # artist_a has city=Tokyo, unknown_entity has no city,
        # venue_city is Berlin, Tokyo != Berlin -> penalty 0.2
        assert abs(result[0].confidence - 0.6) < 0.001

    @pytest.mark.asyncio
    async def test_analyze_edge_confidence_below_threshold_discarded(self):
        """Edges with confidence < 0.15 after all penalties are discarded."""
        from src.services.interconnection_service import InterconnectionService

        llm_response = json.dumps({
            "relationships": [
                {
                    "source": "carl cox",
                    "target": "tresor",
                    "type": "performed_at",
                    "details": "An [UNCERTAIN] connection [1]",
                    "source_citation": "intec",
                    "confidence": 0.1,
                },
            ],
            "patterns": [],
            "narrative": "Minimal narrative.",
        })

        llm = _mock_llm(response=llm_response)
        citation_service = self._make_citation_service()

        # Research results with cities in different locations for geo penalty
        research_results = [
            ResearchResult(
                entity_type=EntityType.ARTIST,
                entity_name="carl cox",
                artist=Artist(
                    name="carl cox",
                    confidence=0.9,
                    releases=[Release(title="Phat Trax", label="Intec", year=1995)],
                    labels=[Label(name="Intec")],
                    city="Melbourne",
                ),
                sources_consulted=["music_databases"],
                confidence=0.9,
            ),
            ResearchResult(
                entity_type=EntityType.VENUE,
                entity_name="tresor",
                venue=Venue(name="Tresor", city="Berlin", confidence=0.85),
                sources_consulted=["web_search_history"],
                confidence=0.85,
            ),
        ]

        entities = ExtractedEntities(
            artists=[
                ExtractedEntity(text="Carl Cox", entity_type=EntityType.ARTIST, confidence=0.95),
            ],
            venue=ExtractedEntity(text="Tresor", entity_type=EntityType.VENUE, confidence=0.9),
            raw_ocr=_make_ocr_result(),
        )

        service = InterconnectionService(
            llm_provider=llm,
            citation_service=citation_service,
        )

        result = await service.analyze(
            research_results=research_results,
            entities=entities,
        )

        # confidence=0.1, uncertain penalty=-0.3 -> max(0, -0.2)=0.0
        # geo mismatch penalty=-0.4 (already 0, stays 0)
        # 0.0 < 0.15 -> edge discarded
        assert len(result.edges) == 0

    @pytest.mark.asyncio
    async def test_retrieve_cross_entity_context_no_vector_store(self):
        """No vector store returns empty string."""
        from src.services.interconnection_service import InterconnectionService

        service = InterconnectionService(
            llm_provider=_mock_llm(),
            citation_service=self._make_citation_service(),
            vector_store=None,
        )

        entities = ExtractedEntities(
            artists=[
                ExtractedEntity(text="Carl Cox", entity_type=EntityType.ARTIST, confidence=0.95),
            ],
            raw_ocr=_make_ocr_result(),
        )

        result = await service._retrieve_cross_entity_context(entities)

        assert result == ""

    @pytest.mark.asyncio
    async def test_retrieve_cross_entity_context_no_entities(self):
        """No entities returns empty string."""
        from src.services.interconnection_service import InterconnectionService

        vector_store = _mock_vector_store(results=[])
        vector_store.is_available = MagicMock(return_value=True)

        service = InterconnectionService(
            llm_provider=_mock_llm(),
            citation_service=self._make_citation_service(),
            vector_store=vector_store,
        )

        entities = ExtractedEntities(
            artists=[],
            raw_ocr=_make_ocr_result(),
        )

        result = await service._retrieve_cross_entity_context(entities)

        assert result == ""

    @pytest.mark.asyncio
    async def test_retrieve_cross_entity_context_with_chunks(self):
        """Cross-entity retrieval returns formatted text from vector store."""
        from src.models.rag import DocumentChunk, RetrievedChunk
        from src.services.interconnection_service import InterconnectionService

        chunk = DocumentChunk(
            chunk_id="c1",
            text="Carl Cox played at Tresor in 1993 during the Berlin techno explosion.",
            source_id="src1",
            source_title="Energy Flash",
            source_type="book",
            author="Simon Reynolds",
            citation_tier=1,
        )
        retrieved = RetrievedChunk(chunk=chunk, similarity_score=0.85)

        vector_store = _mock_vector_store(results=[retrieved])

        service = InterconnectionService(
            llm_provider=_mock_llm(),
            citation_service=self._make_citation_service(),
            vector_store=vector_store,
        )

        entities = ExtractedEntities(
            artists=[
                ExtractedEntity(text="Carl Cox", entity_type=EntityType.ARTIST, confidence=0.95),
            ],
            venue=ExtractedEntity(text="Tresor", entity_type=EntityType.VENUE, confidence=0.9),
            raw_ocr=_make_ocr_result(),
        )

        result = await service._retrieve_cross_entity_context(entities)

        assert "Energy Flash" in result
        assert "Simon Reynolds" in result
        assert "Tier 1" in result

    @pytest.mark.asyncio
    async def test_retrieve_cross_entity_context_exception(self):
        """Vector store exception returns empty string."""
        from src.services.interconnection_service import InterconnectionService

        vector_store = _mock_vector_store()
        vector_store.is_available = MagicMock(return_value=True)
        vector_store.query = AsyncMock(side_effect=RuntimeError("DB error"))

        service = InterconnectionService(
            llm_provider=_mock_llm(),
            citation_service=self._make_citation_service(),
            vector_store=vector_store,
        )

        entities = ExtractedEntities(
            artists=[
                ExtractedEntity(text="Carl Cox", entity_type=EntityType.ARTIST, confidence=0.95),
            ],
            raw_ocr=_make_ocr_result(),
        )

        result = await service._retrieve_cross_entity_context(entities)

        assert result == ""

    def test_compile_research_context_with_promoter_result(self):
        """Context compilation includes promoter data."""
        from src.services.interconnection_service import InterconnectionService

        research_results = [
            ResearchResult(
                entity_type=EntityType.PROMOTER,
                entity_name="tresor berlin",
                promoter=Promoter(
                    name="Tresor Berlin",
                    confidence=0.9,
                    city="Berlin",
                    region="Brandenburg",
                    country="Germany",
                    event_history=["Tresor Opening 1991"],
                    affiliated_artists=["Jeff Mills"],
                    affiliated_venues=["Tresor"],
                    articles=[
                        ArticleReference(
                            title="Tresor Profile",
                            source="ra.co",
                            url="https://ra.co/promoters/tresor",
                            citation_tier=1,
                        )
                    ],
                ),
                sources_consulted=["web_search_promoter"],
                confidence=0.9,
            ),
        ]

        context = InterconnectionService._compile_research_context(research_results)

        assert "tresor berlin" in context.lower()
        assert "Berlin" in context
        assert "Jeff Mills" in context
        assert "Tresor Opening 1991" in context

    def test_compile_research_context_with_date_context_result(self):
        """Context compilation includes date context data."""
        from src.services.interconnection_service import InterconnectionService

        research_results = [
            ResearchResult(
                entity_type=EntityType.DATE,
                entity_name="March 1997",
                date_context=DateContext(
                    event_date=date(1997, 3, 15),
                    scene_context="Techno was booming.",
                    city_context="Berlin was the capital of techno.",
                    cultural_context="Love Parade attracted millions.",
                    nearby_events=["Love Parade 1997", "Tresor anniversary"],
                    sources=[
                        ArticleReference(
                            title="1997 Scene",
                            source="ra.co",
                            url="https://ra.co/1997",
                            citation_tier=1,
                        )
                    ],
                ),
                sources_consulted=["web_search_scene"],
                confidence=0.7,
            ),
        ]

        context = InterconnectionService._compile_research_context(research_results)

        assert "1997-03-15" in context
        assert "Techno was booming" in context
        assert "Love Parade 1997" in context


# ======================================================================
# TestResearchService
# ======================================================================


class TestResearchService:
    """Tests for the ResearchService orchestrator."""

    def _make_mock_researchers(self):
        """Build mock researcher instances."""
        artist_researcher = AsyncMock()
        artist_researcher.research = AsyncMock(
            return_value=ResearchResult(
                entity_type=EntityType.ARTIST,
                entity_name="Carl Cox",
                artist=Artist(name="Carl Cox", confidence=0.9),
                confidence=0.9,
            )
        )

        venue_researcher = AsyncMock()
        venue_researcher.research = AsyncMock(
            return_value=ResearchResult(
                entity_type=EntityType.VENUE,
                entity_name="Tresor",
                venue=Venue(name="Tresor", city="Berlin", confidence=0.85),
                confidence=0.85,
            )
        )

        promoter_researcher = AsyncMock()
        promoter_researcher.research = AsyncMock(
            return_value=ResearchResult(
                entity_type=EntityType.PROMOTER,
                entity_name="Tresor Berlin",
                promoter=Promoter(name="Tresor Berlin", confidence=0.7),
                confidence=0.7,
            )
        )

        date_researcher = AsyncMock()
        date_researcher.research = AsyncMock(
            return_value=ResearchResult(
                entity_type=EntityType.DATE,
                entity_name="March 1997",
                date_context=DateContext(event_date=date(1997, 3, 15)),
                confidence=0.6,
            )
        )

        event_researcher = AsyncMock()
        event_researcher.research = AsyncMock(
            return_value=ResearchResult(
                entity_type=EntityType.EVENT,
                entity_name="Techno Night",
                event_history=EventSeriesHistory(event_name="Techno Night"),
                confidence=0.5,
            )
        )

        return artist_researcher, venue_researcher, promoter_researcher, date_researcher, event_researcher

    @pytest.mark.asyncio
    async def test_research_all_happy_path(self):
        """Full orchestration with all entity types present."""
        from src.services.research_service import ResearchService

        (
            artist_researcher,
            venue_researcher,
            promoter_researcher,
            date_researcher,
            event_researcher,
        ) = self._make_mock_researchers()

        service = ResearchService(
            artist_researcher=artist_researcher,
            venue_researcher=venue_researcher,
            promoter_researcher=promoter_researcher,
            date_context_researcher=date_researcher,
            event_name_researcher=event_researcher,
        )

        entities = ExtractedEntities(
            artists=[
                ExtractedEntity(text="Carl Cox", entity_type=EntityType.ARTIST, confidence=0.95),
            ],
            venue=ExtractedEntity(text="Tresor", entity_type=EntityType.VENUE, confidence=0.9),
            promoter=ExtractedEntity(
                text="Tresor Berlin", entity_type=EntityType.PROMOTER, confidence=0.7
            ),
            event_name=ExtractedEntity(
                text="Techno Night", entity_type=EntityType.EVENT, confidence=0.8
            ),
            raw_ocr=_make_ocr_result(),
        )

        results = await service.research_all(entities, event_date=date(1997, 3, 15))

        # 1 artist + 1 venue + 1 promoter + 1 date + 1 event = 5 results
        assert len(results) == 5
        entity_types = [r.entity_type for r in results]
        assert EntityType.ARTIST in entity_types
        assert EntityType.VENUE in entity_types
        assert EntityType.PROMOTER in entity_types
        assert EntityType.DATE in entity_types
        assert EntityType.EVENT in entity_types

    @pytest.mark.asyncio
    async def test_research_all_no_entities(self):
        """No researchable entities returns empty list."""
        from src.services.research_service import ResearchService

        (
            artist_researcher,
            venue_researcher,
            promoter_researcher,
            date_researcher,
            event_researcher,
        ) = self._make_mock_researchers()

        service = ResearchService(
            artist_researcher=artist_researcher,
            venue_researcher=venue_researcher,
            promoter_researcher=promoter_researcher,
            date_context_researcher=date_researcher,
        )

        entities = ExtractedEntities(
            artists=[],
            raw_ocr=_make_ocr_result(),
        )

        results = await service.research_all(entities)

        assert results == []

    @pytest.mark.asyncio
    async def test_research_all_artist_failure_produces_fallback(self):
        """Artist researcher raises exception — fallback result is produced."""
        from src.services.research_service import ResearchService

        artist_researcher = AsyncMock()
        artist_researcher.research = AsyncMock(side_effect=RuntimeError("API timeout"))

        venue_researcher = AsyncMock()
        venue_researcher.research = AsyncMock(
            return_value=ResearchResult(
                entity_type=EntityType.VENUE,
                entity_name="Tresor",
                venue=Venue(name="Tresor", confidence=0.85),
                confidence=0.85,
            )
        )

        promoter_researcher = AsyncMock()
        date_researcher = AsyncMock()

        service = ResearchService(
            artist_researcher=artist_researcher,
            venue_researcher=venue_researcher,
            promoter_researcher=promoter_researcher,
            date_context_researcher=date_researcher,
        )

        entities = ExtractedEntities(
            artists=[
                ExtractedEntity(text="Carl Cox", entity_type=EntityType.ARTIST, confidence=0.95),
            ],
            venue=ExtractedEntity(text="Tresor", entity_type=EntityType.VENUE, confidence=0.9),
            raw_ocr=_make_ocr_result(),
        )

        results = await service.research_all(entities, event_date=None)

        # 1 failed artist + 1 successful venue = 2 results
        assert len(results) == 2
        artist_result = [r for r in results if r.entity_type == EntityType.ARTIST][0]
        assert artist_result.confidence == 0.0
        assert any("Research failed" in w for w in artist_result.warnings)

    @pytest.mark.asyncio
    async def test_research_all_multiple_artists(self):
        """Multiple artists produce one result per artist."""
        from src.services.research_service import ResearchService

        artist_researcher = AsyncMock()
        artist_researcher.research = AsyncMock(
            return_value=ResearchResult(
                entity_type=EntityType.ARTIST,
                entity_name="DJ Test",
                artist=Artist(name="DJ Test", confidence=0.9),
                confidence=0.9,
            )
        )

        venue_researcher = AsyncMock()
        promoter_researcher = AsyncMock()
        date_researcher = AsyncMock()

        service = ResearchService(
            artist_researcher=artist_researcher,
            venue_researcher=venue_researcher,
            promoter_researcher=promoter_researcher,
            date_context_researcher=date_researcher,
        )

        entities = ExtractedEntities(
            artists=[
                ExtractedEntity(text="Carl Cox", entity_type=EntityType.ARTIST, confidence=0.95),
                ExtractedEntity(text="Jeff Mills", entity_type=EntityType.ARTIST, confidence=0.90),
                ExtractedEntity(text="Derrick May", entity_type=EntityType.ARTIST, confidence=0.85),
            ],
            raw_ocr=_make_ocr_result(),
        )

        results = await service.research_all(entities)

        assert len(results) == 3
        assert artist_researcher.research.await_count == 3

    @pytest.mark.asyncio
    async def test_research_all_parses_date_from_entity(self):
        """When event_date is None, parses date from entities.date."""
        from src.services.research_service import ResearchService

        (
            artist_researcher,
            venue_researcher,
            promoter_researcher,
            date_researcher,
            _,
        ) = self._make_mock_researchers()

        service = ResearchService(
            artist_researcher=artist_researcher,
            venue_researcher=venue_researcher,
            promoter_researcher=promoter_researcher,
            date_context_researcher=date_researcher,
        )

        entities = ExtractedEntities(
            artists=[
                ExtractedEntity(text="Carl Cox", entity_type=EntityType.ARTIST, confidence=0.95),
            ],
            date=ExtractedEntity(
                text="Saturday March 15th 1997", entity_type=EntityType.DATE, confidence=0.8
            ),
            raw_ocr=_make_ocr_result(),
        )

        results = await service.research_all(entities, event_date=None)

        # Should have artist + date results
        assert len(results) == 2
        # Date researcher should have been called
        date_researcher.research.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_research_all_city_hint_from_venue(self):
        """City hint is extracted from venue text with comma separator."""
        from src.services.research_service import ResearchService

        (
            artist_researcher,
            venue_researcher,
            promoter_researcher,
            date_researcher,
            _,
        ) = self._make_mock_researchers()

        service = ResearchService(
            artist_researcher=artist_researcher,
            venue_researcher=venue_researcher,
            promoter_researcher=promoter_researcher,
            date_context_researcher=date_researcher,
        )

        entities = ExtractedEntities(
            artists=[
                ExtractedEntity(text="Carl Cox", entity_type=EntityType.ARTIST, confidence=0.95),
            ],
            venue=ExtractedEntity(
                text="The Warehouse, Chicago",
                entity_type=EntityType.VENUE,
                confidence=0.9,
            ),
            raw_ocr=_make_ocr_result(),
        )

        results = await service.research_all(entities)

        # Venue researcher should have been called with city hint
        venue_researcher.research.assert_awaited_once()

    def test_extract_city_hint_comma_separator(self):
        """City extracted from 'Venue, City' format."""
        from src.services.research_service import ResearchService

        city = ResearchService._extract_city_hint("The Warehouse, Chicago")

        assert city == "Chicago"

    def test_extract_city_hint_dash_separator(self):
        """City extracted from 'Venue - City' format."""
        from src.services.research_service import ResearchService

        city = ResearchService._extract_city_hint("Tresor - Berlin")

        assert city == "Berlin"

    def test_extract_city_hint_no_separator(self):
        """No separator returns None."""
        from src.services.research_service import ResearchService

        city = ResearchService._extract_city_hint("Tresor")

        assert city is None

    def test_extract_city_hint_short_candidate(self):
        """Candidate <= 2 characters returns None."""
        from src.services.research_service import ResearchService

        city = ResearchService._extract_city_hint("Tresor, UK")

        assert city is None

    def test_manual_date_parse_month_day_year(self):
        """'March 15 1997' format parses correctly."""
        from src.services.research_service import ResearchService

        result = ResearchService._manual_date_parse("March 15 1997")

        assert result == date(1997, 3, 15)

    def test_manual_date_parse_day_month_year(self):
        """'15 March 1997' format parses correctly."""
        from src.services.research_service import ResearchService

        result = ResearchService._manual_date_parse("15 March 1997")

        assert result == date(1997, 3, 15)

    def test_manual_date_parse_mm_dd_yy(self):
        """'03/15/97' format parses correctly (2-digit year > 50 -> 1900s)."""
        from src.services.research_service import ResearchService

        result = ResearchService._manual_date_parse("03/15/97")

        assert result == date(1997, 3, 15)

    def test_manual_date_parse_mm_dd_yy_2000s(self):
        """'03/15/05' format parses correctly (2-digit year <= 50 -> 2000s)."""
        from src.services.research_service import ResearchService

        result = ResearchService._manual_date_parse("03/15/05")

        assert result == date(2005, 3, 15)

    def test_manual_date_parse_dd_mm_yyyy(self):
        """'15.03.1997' format parses correctly."""
        from src.services.research_service import ResearchService

        result = ResearchService._manual_date_parse("15.03.1997")

        assert result == date(1997, 3, 15)

    def test_manual_date_parse_iso(self):
        """'1997-03-15' ISO format parses correctly."""
        from src.services.research_service import ResearchService

        result = ResearchService._manual_date_parse("1997-03-15")

        assert result == date(1997, 3, 15)

    def test_manual_date_parse_unrecognized(self):
        """Unrecognized format returns None."""
        from src.services.research_service import ResearchService

        result = ResearchService._manual_date_parse("not a date at all")

        assert result is None

    def test_manual_date_parse_abbreviated_month(self):
        """'Mar 15 1997' abbreviated month format parses correctly."""
        from src.services.research_service import ResearchService

        result = ResearchService._manual_date_parse("Mar 15 1997")

        assert result == date(1997, 3, 15)

    def test_manual_date_parse_month_day_comma_year(self):
        """'March 15, 1997' format with comma parses correctly."""
        from src.services.research_service import ResearchService

        result = ResearchService._manual_date_parse("March 15, 1997")

        assert result == date(1997, 3, 15)

    @pytest.mark.asyncio
    async def test_parse_event_date_none_entity(self):
        """None date entity returns None."""
        from src.services.research_service import ResearchService

        (
            artist_researcher,
            venue_researcher,
            promoter_researcher,
            date_researcher,
            _,
        ) = self._make_mock_researchers()

        service = ResearchService(
            artist_researcher=artist_researcher,
            venue_researcher=venue_researcher,
            promoter_researcher=promoter_researcher,
            date_context_researcher=date_researcher,
        )

        result = await service._parse_event_date(None)

        assert result is None

    @pytest.mark.asyncio
    async def test_parse_event_date_empty_text(self):
        """Empty text in date entity returns None."""
        from src.services.research_service import ResearchService

        (
            artist_researcher,
            venue_researcher,
            promoter_researcher,
            date_researcher,
            _,
        ) = self._make_mock_researchers()

        service = ResearchService(
            artist_researcher=artist_researcher,
            venue_researcher=venue_researcher,
            promoter_researcher=promoter_researcher,
            date_context_researcher=date_researcher,
        )

        entity = ExtractedEntity(text="   ", entity_type=EntityType.DATE, confidence=0.8)
        result = await service._parse_event_date(entity)

        assert result is None

    @pytest.mark.asyncio
    async def test_parse_event_date_with_day_name(self):
        """Date string with day-of-week prefix is cleaned and parsed."""
        from src.services.research_service import ResearchService

        (
            artist_researcher,
            venue_researcher,
            promoter_researcher,
            date_researcher,
            _,
        ) = self._make_mock_researchers()

        service = ResearchService(
            artist_researcher=artist_researcher,
            venue_researcher=venue_researcher,
            promoter_researcher=promoter_researcher,
            date_context_researcher=date_researcher,
        )

        entity = ExtractedEntity(
            text="Saturday March 15th 1997",
            entity_type=EntityType.DATE,
            confidence=0.8,
        )
        result = await service._parse_event_date(entity)

        assert result == date(1997, 3, 15)

    @pytest.mark.asyncio
    async def test_research_all_without_event_name_researcher(self):
        """Event name entity present but no event researcher configured — skipped."""
        from src.services.research_service import ResearchService

        (
            artist_researcher,
            venue_researcher,
            promoter_researcher,
            date_researcher,
            _,
        ) = self._make_mock_researchers()

        service = ResearchService(
            artist_researcher=artist_researcher,
            venue_researcher=venue_researcher,
            promoter_researcher=promoter_researcher,
            date_context_researcher=date_researcher,
            event_name_researcher=None,  # Not configured
        )

        entities = ExtractedEntities(
            artists=[
                ExtractedEntity(text="Carl Cox", entity_type=EntityType.ARTIST, confidence=0.95),
            ],
            event_name=ExtractedEntity(
                text="Techno Night", entity_type=EntityType.EVENT, confidence=0.8
            ),
            raw_ocr=_make_ocr_result(),
        )

        results = await service.research_all(entities)

        # Only artist result — event skipped because no researcher
        assert len(results) == 1
        assert results[0].entity_type == EntityType.ARTIST


# ======================================================================
# Extended VenueResearcher tests — coverage additions
# ======================================================================


class TestVenueResearcherExtended:
    """Additional tests for VenueResearcher targeting uncovered lines."""

    @pytest.mark.asyncio
    async def test_search_venue_history_with_city(self):
        """City qualifier added to search queries."""
        from src.services.venue_researcher import VenueResearcher

        web_search = _mock_web_search(results=[
            _search_result(url="https://ra.co/clubs/tresor"),
        ])

        researcher = VenueResearcher(
            web_search=web_search,
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(
                response="HISTORY:\nNONE\nNOTABLE_EVENTS:\nNONE\nCULTURAL_SIGNIFICANCE:\nNONE"
            ),
        )

        result = await researcher.research("Tresor", city="Berlin")

        # web_search should have been called with city-qualified queries
        assert web_search.search.await_count >= 3  # at least 3 queries

    @pytest.mark.asyncio
    async def test_search_venue_history_without_city(self):
        """No city — uses generic queries."""
        from src.services.venue_researcher import VenueResearcher

        web_search = _mock_web_search(results=[
            _search_result(url="https://ra.co/clubs/tresor"),
        ])

        researcher = VenueResearcher(
            web_search=web_search,
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(
                response="HISTORY:\nNONE\nNOTABLE_EVENTS:\nNONE\nCULTURAL_SIGNIFICANCE:\nNONE"
            ),
        )

        result = await researcher.research("Tresor", city=None)

        assert result.venue is not None
        assert result.venue.city is None

    @pytest.mark.asyncio
    async def test_synthesize_venue_profile_no_scraped_texts(self):
        """No scraped texts — returns (None, [], None)."""
        from src.services.venue_researcher import VenueResearcher

        researcher = VenueResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
        )

        history, events, sig = await researcher._synthesize_venue_profile("Tresor", "Berlin", [])

        assert history is None
        assert events == []
        assert sig is None

    @pytest.mark.asyncio
    async def test_synthesize_venue_profile_llm_failure(self):
        """LLM failure returns (None, [], None)."""
        from src.services.venue_researcher import VenueResearcher

        llm = _mock_llm()
        llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))

        researcher = VenueResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=llm,
        )

        history, events, sig = await researcher._synthesize_venue_profile(
            "Tresor", "Berlin", ["some text"]
        )

        assert history is None
        assert events == []
        assert sig is None

    @pytest.mark.asyncio
    async def test_retrieve_from_corpus_no_vector_store(self):
        """No vector store returns empty list."""
        from src.services.venue_researcher import VenueResearcher

        researcher = VenueResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            vector_store=None,
        )

        refs = await researcher._retrieve_from_corpus("Tresor")

        assert refs == []

    @pytest.mark.asyncio
    async def test_scrape_results_mixed_success_and_failure(self):
        """Scrape mix of success and failure — only successful returned."""
        from src.services.venue_researcher import VenueResearcher

        article_scraper = _mock_article_scraper()
        article_scraper.extract_content = AsyncMock(
            side_effect=[
                _article_content(text="Good content about Tresor."),
                RuntimeError("timeout"),
                _article_content(text="Another good article."),
            ]
        )

        researcher = VenueResearcher(
            web_search=_mock_web_search(),
            article_scraper=article_scraper,
            llm=_mock_llm(),
        )

        results = [
            _search_result(url="https://example.com/1"),
            _search_result(url="https://example.com/2"),
            _search_result(url="https://example.com/3"),
        ]
        texts = await researcher._scrape_results(results)

        assert len(texts) == 2

    @pytest.mark.asyncio
    async def test_search_venue_articles_no_results(self):
        """No article search results returns empty list."""
        from src.services.venue_researcher import VenueResearcher

        researcher = VenueResearcher(
            web_search=_mock_web_search(results=[]),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
        )

        articles = await researcher._search_venue_articles("Tresor", city=None)

        assert articles == []


# ======================================================================
# Extended PromoterResearcher tests — coverage additions
# ======================================================================


class TestPromoterResearcherExtended:
    """Additional tests for PromoterResearcher targeting uncovered lines."""

    @pytest.mark.asyncio
    async def test_search_promoter_activity_with_city(self):
        """City hint adds city-qualified queries."""
        from src.services.promoter_researcher import PromoterResearcher

        web_search = _mock_web_search(results=[
            _search_result(url="https://ra.co/promoters/test"),
        ])

        researcher = PromoterResearcher(
            web_search=web_search,
            article_scraper=_mock_article_scraper(content=None),
            llm=_mock_llm(
                response=(
                    "BASED_IN:\nNONE\n\n"
                    "EVENT_HISTORY:\nNONE\n\n"
                    "AFFILIATED_ARTISTS:\nNONE\n\n"
                    "AFFILIATED_VENUES:\nNONE"
                )
            ),
        )

        result = await researcher.research("Test Promoter", city="London")

        # City-qualified queries should have been added
        assert web_search.search.await_count >= 4  # 3 city + 4 base

    @pytest.mark.asyncio
    async def test_search_promoter_activity_sparse_deepens(self):
        """Sparse results trigger deepening queries."""
        from src.services.promoter_researcher import PromoterResearcher

        call_count = 0

        async def search_side_effect(query, num_results=12):
            nonlocal call_count
            call_count += 1
            if call_count <= 4:
                return []  # Sparse results
            return [_search_result(url=f"https://ra.co/deep/{call_count}")]

        web_search = _mock_web_search()
        web_search.search = AsyncMock(side_effect=search_side_effect)

        researcher = PromoterResearcher(
            web_search=web_search,
            article_scraper=_mock_article_scraper(content=None),
            llm=_mock_llm(
                response=(
                    "BASED_IN:\nNONE\n\n"
                    "EVENT_HISTORY:\nNONE\n\n"
                    "AFFILIATED_ARTISTS:\nNONE\n\n"
                    "AFFILIATED_VENUES:\nNONE"
                )
            ),
        )

        result = await researcher.research("Test Promoter", city=None)

        # Deepening queries should have fired
        assert call_count > 4

    @pytest.mark.asyncio
    async def test_extract_promoter_profile_no_scraped_texts(self):
        """No scraped texts returns empty defaults."""
        from src.services.promoter_researcher import PromoterResearcher

        researcher = PromoterResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
        )

        result = await researcher._extract_promoter_profile("Test", [], city=None)

        assert result["event_history"] == []
        assert result["affiliated_artists"] == []

    @pytest.mark.asyncio
    async def test_extract_promoter_profile_llm_failure(self):
        """LLM failure returns empty defaults."""
        from src.services.promoter_researcher import PromoterResearcher

        llm = _mock_llm()
        llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))

        researcher = PromoterResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=llm,
        )

        result = await researcher._extract_promoter_profile("Test", ["some text"], city=None)

        assert result["event_history"] == []

    @pytest.mark.asyncio
    async def test_retrieve_from_corpus_no_vector_store(self):
        """No vector store returns empty list."""
        from src.services.promoter_researcher import PromoterResearcher

        researcher = PromoterResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            vector_store=None,
        )

        refs = await researcher._retrieve_from_corpus("Test", city=None)

        assert refs == []

    def test_parse_promoter_extraction_based_in_city_only(self):
        """BASED_IN with only city (no comma) — region and country are None."""
        from src.services.promoter_researcher import PromoterResearcher

        text = (
            "BASED_IN:\n"
            "- Berlin\n\n"
            "EVENT_HISTORY:\nNONE\n\n"
            "AFFILIATED_ARTISTS:\nNONE\n\n"
            "AFFILIATED_VENUES:\nNONE"
        )

        parsed = PromoterResearcher._parse_promoter_extraction(text)

        assert parsed["based_city"] == "Berlin"
        assert parsed["based_region"] is None
        assert parsed["based_country"] is None


# ======================================================================
# Extended DateContextResearcher tests — coverage additions
# ======================================================================


class TestDateContextResearcherExtended:
    """Additional tests for DateContextResearcher targeting uncovered lines."""

    @pytest.mark.asyncio
    async def test_research_no_scene_results(self):
        """No scene results produces warning."""
        from src.services.date_context_researcher import DateContextResearcher

        researcher = DateContextResearcher(
            web_search=_mock_web_search(results=[]),
            article_scraper=_mock_article_scraper(content=None),
            llm=_mock_llm(
                response=(
                    "SCENE_CONTEXT:\nNONE\n\n"
                    "CITY_CONTEXT:\nNONE\n\n"
                    "CULTURAL_CONTEXT:\nNONE\n\n"
                    "NEARBY_EVENTS:\nNONE"
                )
            ),
        )

        result = await researcher.research(date(1997, 3, 15), city=None)

        assert any("no web results" in w.lower() for w in result.warnings)

    @pytest.mark.asyncio
    async def test_research_no_city(self):
        """Research without city — entity_name does not include city."""
        from src.services.date_context_researcher import DateContextResearcher

        researcher = DateContextResearcher(
            web_search=_mock_web_search(results=[]),
            article_scraper=_mock_article_scraper(content=None),
            llm=_mock_llm(
                response=(
                    "SCENE_CONTEXT:\nNONE\n\n"
                    "CITY_CONTEXT:\nNONE\n\n"
                    "CULTURAL_CONTEXT:\nNONE\n\n"
                    "NEARBY_EVENTS:\nNONE"
                )
            ),
        )

        result = await researcher.research(date(1997, 3, 15), city=None)

        assert result.entity_name == "March 1997"

    @pytest.mark.asyncio
    async def test_research_with_city_in_entity_name(self):
        """Research with city — entity_name includes city prefix."""
        from src.services.date_context_researcher import DateContextResearcher

        researcher = DateContextResearcher(
            web_search=_mock_web_search(results=[]),
            article_scraper=_mock_article_scraper(content=None),
            llm=_mock_llm(
                response=(
                    "SCENE_CONTEXT:\nNONE\n\n"
                    "CITY_CONTEXT:\nNONE\n\n"
                    "CULTURAL_CONTEXT:\nNONE\n\n"
                    "NEARBY_EVENTS:\nNONE"
                )
            ),
        )

        result = await researcher.research(date(1997, 3, 15), city="Berlin")

        assert "Berlin" in result.entity_name

    @pytest.mark.asyncio
    async def test_synthesize_date_context_no_scraped_texts(self):
        """No scraped texts returns all None/empty."""
        from src.services.date_context_researcher import DateContextResearcher

        researcher = DateContextResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
        )

        scene, city, cultural, events = await researcher._synthesize_date_context(
            date(1997, 3, 15), 1997, "March", "Berlin", []
        )

        assert scene is None
        assert city is None
        assert cultural is None
        assert events == []

    @pytest.mark.asyncio
    async def test_synthesize_date_context_llm_failure(self):
        """LLM failure returns all None/empty."""
        from src.services.date_context_researcher import DateContextResearcher

        llm = _mock_llm()
        llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))

        researcher = DateContextResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=llm,
        )

        scene, city, cultural, events = await researcher._synthesize_date_context(
            date(1997, 3, 15), 1997, "March", "Berlin", ["some text"]
        )

        assert scene is None
        assert events == []

    @pytest.mark.asyncio
    async def test_search_scene_context_without_city(self):
        """Scene search without city uses generic electronic music queries."""
        from src.services.date_context_researcher import DateContextResearcher

        web_search = _mock_web_search(results=[
            _search_result(url="https://ra.co/features/1997"),
        ])

        researcher = DateContextResearcher(
            web_search=web_search,
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
        )

        results = await researcher._search_scene_context(1997, "March", city=None)

        assert len(results) >= 1
        assert web_search.search.await_count >= 3

    @pytest.mark.asyncio
    async def test_build_article_references_empty_results(self):
        """No search results returns empty article list."""
        from src.services.date_context_researcher import DateContextResearcher

        researcher = DateContextResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
        )

        articles = await researcher._build_article_references([], "1997")

        assert articles == []

    @pytest.mark.asyncio
    async def test_extract_article_reference_no_content(self):
        """Scraper returns None — uses search result snippet/title."""
        from src.services.date_context_researcher import DateContextResearcher

        researcher = DateContextResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(content=None),
            llm=_mock_llm(),
        )

        result = _search_result(
            title="1997 Rave Scene", url="https://ra.co/features/1997", snippet="A great year."
        )
        ref = await researcher._extract_article_reference(result, "1997")

        assert ref.title == "1997 Rave Scene"
        assert ref.snippet == "A great year."
        assert ref.citation_tier == 1  # ra.co is tier 1

    @pytest.mark.asyncio
    async def test_extract_article_reference_with_history_type(self):
        """Article with 'history' in title classified as history type."""
        from src.services.date_context_researcher import DateContextResearcher

        researcher = DateContextResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(
                content=_article_content(
                    title="History of Berlin Techno", text="1997 was a pivotal year."
                )
            ),
            llm=_mock_llm(),
        )

        result = _search_result(url="https://example.com/history")
        ref = await researcher._extract_article_reference(result, "1997")

        assert ref.article_type == "history"

    @pytest.mark.asyncio
    async def test_extract_article_reference_with_legislation_type(self):
        """Article with 'legislation' in title classified as legislation type."""
        from src.services.date_context_researcher import DateContextResearcher

        researcher = DateContextResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(
                content=_article_content(
                    title="Rave Legislation in the UK", text="The Criminal Justice Act."
                )
            ),
            llm=_mock_llm(),
        )

        result = _search_result(url="https://example.com/law")
        ref = await researcher._extract_article_reference(result, "1997")

        assert ref.article_type == "legislation"


# ======================================================================
# Coverage-targeted: ArtistResearcher — lines 385-391, 408-418, 430,
#   532-533, 643-742 (_search_press full flow), 748-767
#   (_extract_article_reference), 804/808, 935-942
# ======================================================================


class TestArtistResearcherCoverageFill:
    """Tests targeting remaining uncovered lines in artist_researcher.py."""

    # --- lines 385-391: _fetch_discography Discogs ResearchError fallthrough ---
    @pytest.mark.asyncio
    async def test_fetch_discography_discogs_research_error_continues(self):
        """Discogs provider raises ResearchError — skips to next provider."""
        from src.services.artist_researcher import ArtistResearcher
        from src.utils.errors import ResearchError

        db_primary = _mock_music_db(provider_name="discogs_api")
        db_primary.get_artist_releases = AsyncMock(
            side_effect=ResearchError("Discogs API error")
        )

        db_fallback = _mock_music_db(
            provider_name="discogs_scrape",
            releases=[Release(title="Fallback", label="Intec", year=1997)],
            labels=[Label(name="Intec")],
        )

        researcher = ArtistResearcher(
            music_dbs=[db_primary, db_fallback],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=_mock_cache(),
        )

        releases, labels = await researcher._fetch_discography(
            "Carl Cox",
            discogs_id="123",
            musicbrainz_id=None,
            before_date=None,
            provider_ids={},
        )

        assert len(releases) == 1
        assert releases[0].title == "Fallback"

    # --- lines 408-418: MusicBrainz supplement with labels fallback ---
    @pytest.mark.asyncio
    async def test_fetch_discography_musicbrainz_labels_when_discogs_has_none(self):
        """MusicBrainz provides labels when Discogs returned no labels."""
        from src.services.artist_researcher import ArtistResearcher

        db_discogs = _mock_music_db(
            provider_name="discogs_api",
            releases=[Release(title="Track A", label="Intec", year=1995)],
            labels=[],  # No labels from Discogs
        )

        db_mb = _mock_music_db(
            provider_name="musicbrainz_api",
            releases=[Release(title="Track B", label="Warp", year=1997)],
            labels=[Label(name="Warp"), Label(name="Rephlex")],
        )

        researcher = ArtistResearcher(
            music_dbs=[db_discogs, db_mb],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=_mock_cache(),
        )

        releases, labels = await researcher._fetch_discography(
            "Carl Cox",
            discogs_id="123",
            musicbrainz_id="mb-uuid",
            before_date=None,
            provider_ids={},
        )

        # MusicBrainz labels should have been added since Discogs had none
        label_names = [lb.name for lb in labels]
        assert "Warp" in label_names
        assert "Rephlex" in label_names

    # --- lines 408-418: MusicBrainz supplement raises ResearchError ---
    @pytest.mark.asyncio
    async def test_fetch_discography_musicbrainz_error_handled(self):
        """MusicBrainz supplement failure is handled gracefully."""
        from src.services.artist_researcher import ArtistResearcher
        from src.utils.errors import ResearchError

        db_discogs = _mock_music_db(
            provider_name="discogs_api",
            releases=[Release(title="Track A", label="Intec", year=1995)],
            labels=[Label(name="Intec")],
        )

        db_mb = _mock_music_db(provider_name="musicbrainz_api")
        db_mb.get_artist_releases = AsyncMock(
            side_effect=ResearchError("MB API down")
        )

        researcher = ArtistResearcher(
            music_dbs=[db_discogs, db_mb],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=_mock_cache(),
        )

        releases, labels = await researcher._fetch_discography(
            "Carl Cox",
            discogs_id="123",
            musicbrainz_id="mb-uuid",
            before_date=None,
            provider_ids={},
        )

        # Discogs results should still be present despite MB failure
        assert len(releases) == 1
        assert releases[0].title == "Track A"

    # --- line 430: additional provider skip for discogs/musicbrainz ---
    @pytest.mark.asyncio
    async def test_fetch_discography_skips_discogs_in_additional_providers(self):
        """Discogs providers are skipped in the additional-provider loop."""
        from src.services.artist_researcher import ArtistResearcher

        db_discogs = _mock_music_db(
            provider_name="discogs_api",
            releases=[Release(title="Track A", label="Intec", year=1995)],
            labels=[Label(name="Intec")],
        )

        researcher = ArtistResearcher(
            music_dbs=[db_discogs],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=_mock_cache(),
        )

        releases, labels = await researcher._fetch_discography(
            "Carl Cox",
            discogs_id="123",
            musicbrainz_id=None,
            before_date=None,
            # discogs_api in provider_ids but already fetched as primary
            provider_ids={"discogs_api": "123"},
        )

        # get_artist_releases should only have been called once (primary fetch),
        # not again in the additional provider loop
        assert db_discogs.get_artist_releases.await_count == 1

    # --- lines 748-767: _extract_article_reference with various content types ---
    @pytest.mark.asyncio
    async def test_extract_article_reference_interview(self):
        """Article with 'interview' in title classified as interview."""
        from src.services.artist_researcher import ArtistResearcher

        article_scraper = _mock_article_scraper(
            content=_article_content(
                title="Interview with Carl Cox",
                text="Carl Cox discusses his career and techno roots.",
            )
        )

        researcher = ArtistResearcher(
            music_dbs=[],
            web_search=_mock_web_search(),
            article_scraper=article_scraper,
            llm=_mock_llm(),
        )

        ref = await researcher._extract_article_reference(
            _search_result(url="https://ra.co/features/interview"),
            "Carl Cox",
        )

        assert ref.article_type == "interview"
        assert ref.snippet is not None

    @pytest.mark.asyncio
    async def test_extract_article_reference_review(self):
        """Article with 'review' in title classified as review."""
        from src.services.artist_researcher import ArtistResearcher

        article_scraper = _mock_article_scraper(
            content=_article_content(
                title="Album Review: Carl Cox - Electronic Generations",
                text="Carl Cox delivers another masterpiece.",
            )
        )

        researcher = ArtistResearcher(
            music_dbs=[],
            web_search=_mock_web_search(),
            article_scraper=article_scraper,
            llm=_mock_llm(),
        )

        ref = await researcher._extract_article_reference(
            _search_result(url="https://djmag.com/review"),
            "Carl Cox",
        )

        assert ref.article_type == "review"

    @pytest.mark.asyncio
    async def test_extract_article_reference_mix(self):
        """Article with 'mix' in title classified as mix."""
        from src.services.artist_researcher import ArtistResearcher

        article_scraper = _mock_article_scraper(
            content=_article_content(
                title="Essential Mix: Carl Cox",
                text="Carl Cox essential mix from 1999.",
            )
        )

        researcher = ArtistResearcher(
            music_dbs=[],
            web_search=_mock_web_search(),
            article_scraper=article_scraper,
            llm=_mock_llm(),
        )

        ref = await researcher._extract_article_reference(
            _search_result(url="https://ra.co/mix"),
            "Carl Cox",
        )

        assert ref.article_type == "mix"

    @pytest.mark.asyncio
    async def test_extract_article_reference_no_content_uses_snippet(self):
        """Scraper returns None — falls back to search result snippet."""
        from src.services.artist_researcher import ArtistResearcher

        researcher = ArtistResearcher(
            music_dbs=[],
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(content=None),
            llm=_mock_llm(),
        )

        sr = _search_result(
            title="Carl Cox RA feature",
            url="https://ra.co/features/999",
            snippet="Carl Cox RA feature snippet text",
        )
        ref = await researcher._extract_article_reference(sr, "Carl Cox")

        assert ref.title == "Carl Cox RA feature"
        assert ref.snippet == "Carl Cox RA feature snippet text"
        assert ref.article_type == "article"

# ======================================================================
# Coverage-targeted: VenueResearcher — lines 143-149, 197-227
#   (_synthesize_venue_profile with scraped texts), 246-247, 344-345,
#   372-373, 390-394, 477-511
# ======================================================================


class TestVenueResearcherCoverageFill:
    """Tests targeting remaining uncovered lines in venue_researcher.py."""

    # --- lines 143-149: corpus refs merged into articles ---
    @pytest.mark.asyncio
    async def test_research_merges_corpus_refs_deduplication(self):
        """Corpus refs are merged into articles with deduplication."""
        from src.models.rag import DocumentChunk, RetrievedChunk
        from src.services.venue_researcher import VenueResearcher

        chunk = DocumentChunk(
            chunk_id="c1",
            text="Tresor is a legendary nightclub" * 10,
            source_id="src1",
            source_title="Energy Flash",
            source_type="book",
            citation_tier=1,
        )
        retrieved = RetrievedChunk(chunk=chunk, similarity_score=0.85)
        vector_store = _mock_vector_store(results=[retrieved])

        web_search = _mock_web_search(results=[
            _search_result(title="Tresor Berlin", url="https://ra.co/clubs/tresor"),
        ])

        article_scraper = _mock_article_scraper(
            content=_article_content(
                title="Energy Flash",  # Same title as corpus ref
                text="Tresor opened in 1991 in a vault.",
            )
        )

        llm = _mock_llm(
            response=(
                "HISTORY:\nTresor opened in 1991.\n\n"
                "NOTABLE_EVENTS:\n- Opening night\n\n"
                "CULTURAL_SIGNIFICANCE:\nBirthplace of Berlin techno."
            )
        )

        researcher = VenueResearcher(
            web_search=web_search,
            article_scraper=article_scraper,
            llm=llm,
            vector_store=vector_store,
        )

        result = await researcher.research("Tresor", city="Berlin")

        assert "rag_corpus" in result.sources_consulted
        # The corpus ref "Energy Flash" should not be duplicated with the article
        # of the same title
        titles = [a.title for a in result.venue.articles if a.title]
        assert titles.count("Energy Flash") == 1

    # --- lines 197-227: _retrieve_from_corpus with results ---
    @pytest.mark.asyncio
    async def test_retrieve_from_corpus_with_results(self):
        """Vector store returns matching chunks converted to ArticleReferences."""
        from src.models.rag import DocumentChunk, RetrievedChunk
        from src.services.venue_researcher import VenueResearcher

        chunk = DocumentChunk(
            chunk_id="c1",
            text="Tresor is a legendary nightclub in Berlin." * 5,
            source_id="src1",
            source_title="Last Night a DJ Saved My Life",
            source_type="book",
            publication_date=date(1999, 1, 1),
            citation_tier=2,
        )
        retrieved = RetrievedChunk(
            chunk=chunk,
            similarity_score=0.88,
            formatted_citation="Last Night a DJ Saved My Life, p.200",
        )
        vector_store = _mock_vector_store(results=[retrieved])

        researcher = VenueResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            vector_store=vector_store,
        )

        refs = await researcher._retrieve_from_corpus("Tresor")

        assert len(refs) == 1
        assert refs[0].title == "Last Night a DJ Saved My Life"
        assert refs[0].article_type == "book"
        assert refs[0].citation_tier == 2
        assert refs[0].date == date(1999, 1, 1)

    @pytest.mark.asyncio
    async def test_retrieve_from_corpus_exception(self):
        """Vector store exception returns empty list."""
        from src.services.venue_researcher import VenueResearcher

        vector_store = _mock_vector_store()
        vector_store.is_available = MagicMock(return_value=True)
        vector_store.query = AsyncMock(side_effect=RuntimeError("ChromaDB error"))

        researcher = VenueResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            vector_store=vector_store,
        )

        refs = await researcher._retrieve_from_corpus("Tresor")

        assert refs == []

    @pytest.mark.asyncio
    async def test_retrieve_from_corpus_low_similarity_filtered(self):
        """Chunks with similarity < 0.7 are filtered out."""
        from src.models.rag import DocumentChunk, RetrievedChunk
        from src.services.venue_researcher import VenueResearcher

        chunk = DocumentChunk(
            chunk_id="c1",
            text="Unrelated text" * 10,
            source_id="src1",
            source_title="Random Book",
            source_type="article",
            citation_tier=5,
        )
        retrieved = RetrievedChunk(chunk=chunk, similarity_score=0.5)
        vector_store = _mock_vector_store(results=[retrieved])

        researcher = VenueResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            vector_store=vector_store,
        )

        refs = await researcher._retrieve_from_corpus("Tresor")

        assert refs == []

    # --- lines 344-345, 372-373: _search_venue_articles with ResearchError ---
    @pytest.mark.asyncio
    async def test_search_venue_articles_research_error_handled(self):
        """ResearchError in article search is caught, other queries proceed."""
        from src.services.venue_researcher import VenueResearcher

        call_count = 0

        async def search_side_effect(query, num_results=15):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                from src.utils.errors import ResearchError
                raise ResearchError("Search API down")
            return [_search_result(url=f"https://ra.co/articles/{call_count}")]

        web_search = _mock_web_search()
        web_search.search = AsyncMock(side_effect=search_side_effect)

        researcher = VenueResearcher(
            web_search=web_search,
            article_scraper=_mock_article_scraper(
                content=_article_content(text="Tresor article content.")
            ),
            llm=_mock_llm(),
        )

        articles = await researcher._search_venue_articles("Tresor", city="Berlin")

        # Should have gotten results from non-failing queries
        assert len(articles) >= 1

    # --- lines 372-373: _search_venue_articles extraction failure ---
    @pytest.mark.asyncio
    async def test_search_venue_articles_extraction_failure_handled(self):
        """Article extraction failure is caught; other articles still returned."""
        from src.services.venue_researcher import VenueResearcher

        web_search = _mock_web_search(results=[
            _search_result(url="https://ra.co/1"),
            _search_result(url="https://ra.co/2"),
        ])

        article_scraper = _mock_article_scraper()
        article_scraper.extract_content = AsyncMock(
            side_effect=[
                _article_content(title="Good Article", text="Tresor club article."),
                RuntimeError("Scrape timeout"),
            ]
        )

        researcher = VenueResearcher(
            web_search=web_search,
            article_scraper=article_scraper,
            llm=_mock_llm(),
        )

        articles = await researcher._search_venue_articles("Tresor", city=None)

        # One succeeded, one failed
        assert len(articles) == 1
        assert articles[0].title == "Good Article"

    # --- lines 390-394: _extract_article_reference with history/obituary types ---
    @pytest.mark.asyncio
    async def test_extract_article_reference_review_type(self):
        """Venue article with 'review' in title classified as review."""
        from src.services.venue_researcher import VenueResearcher

        researcher = VenueResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(
                content=_article_content(
                    title="Club Review: Tresor Berlin",
                    text="Tresor is a world-class venue.",
                )
            ),
            llm=_mock_llm(),
        )

        ref = await researcher._extract_article_reference(
            _search_result(url="https://djmag.com/review"),
            "Tresor",
        )

        assert ref.article_type == "review"

    @pytest.mark.asyncio
    async def test_extract_article_reference_history_type(self):
        """Venue article with 'history' in title classified as history."""
        from src.services.venue_researcher import VenueResearcher

        researcher = VenueResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(
                content=_article_content(
                    title="The History of Tresor",
                    text="Tresor opened in 1991 in the vaults.",
                )
            ),
            llm=_mock_llm(),
        )

        ref = await researcher._extract_article_reference(
            _search_result(url="https://example.com/history"),
            "Tresor",
        )

        assert ref.article_type == "history"

    @pytest.mark.asyncio
    async def test_extract_article_reference_obituary_type(self):
        """Venue article with 'closing' in title classified as obituary."""
        from src.services.venue_researcher import VenueResearcher

        researcher = VenueResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(
                content=_article_content(
                    title="Tresor Closing: End of an Era",
                    text="Tresor closed its doors for the last time.",
                )
            ),
            llm=_mock_llm(),
        )

        ref = await researcher._extract_article_reference(
            _search_result(url="https://example.com/closing"),
            "Tresor",
        )

        assert ref.article_type == "obituary"

    @pytest.mark.asyncio
    async def test_extract_article_reference_no_content_fallback(self):
        """No scraped content — uses search result snippet and title."""
        from src.services.venue_researcher import VenueResearcher

        researcher = VenueResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(content=None),
            llm=_mock_llm(),
        )

        sr = _search_result(
            title="Tresor Berlin Venue",
            url="https://ra.co/clubs/tresor",
            snippet="Tresor is a Berlin techno club.",
        )
        ref = await researcher._extract_article_reference(sr, "Tresor")

        assert ref.title == "Tresor Berlin Venue"
        assert ref.snippet == "Tresor is a Berlin techno club."
        assert ref.article_type == "article"

    # --- lines 477-511: _synthesize_venue_profile with scraped texts ---
    @pytest.mark.asyncio
    async def test_synthesize_venue_profile_with_scraped_texts(self):
        """Venue profile synthesis with scraped content produces structured result."""
        from src.services.venue_researcher import VenueResearcher

        llm = _mock_llm(
            response=(
                "HISTORY:\n"
                "Tresor opened in 1991 in a former department store vault in Berlin.\n\n"
                "NOTABLE_EVENTS:\n"
                "- Jeff Mills residency 1993\n"
                "- Tresor Records showcase 1992\n\n"
                "CULTURAL_SIGNIFICANCE:\n"
                "Tresor is considered the birthplace of Berlin techno and a symbol of "
                "German reunification culture."
            )
        )

        researcher = VenueResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=llm,
        )

        history, events, significance = await researcher._synthesize_venue_profile(
            "Tresor",
            "Berlin",
            ["Tresor opened in 1991 in a vault.", "Jeff Mills played at Tresor many times."],
        )

        assert history is not None
        assert "1991" in history
        assert len(events) == 2
        assert significance is not None
        assert "Berlin techno" in significance

    # --- cache helpers ---
    @pytest.mark.asyncio
    async def test_cache_get_failure_returns_none(self):
        """Cache get failure returns None silently."""
        from src.services.venue_researcher import VenueResearcher

        cache = _mock_cache()
        cache.get = AsyncMock(side_effect=RuntimeError("Redis down"))

        researcher = VenueResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=cache,
        )

        result = await researcher._cache_get("some_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_set_failure_handled_silently(self):
        """Cache set failure is caught and does not raise."""
        from src.services.venue_researcher import VenueResearcher

        cache = _mock_cache()
        cache.set = AsyncMock(side_effect=RuntimeError("Redis down"))

        researcher = VenueResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=cache,
        )

        # Should not raise
        await researcher._cache_set("some_key", {"data": True})


# ======================================================================
# Coverage-targeted: PromoterResearcher — lines 141, 153-158, 222-253,
#   287-288, 321-322, 340, 434-435, 452-458, 552-599
# ======================================================================


class TestPromoterResearcherCoverageFill:
    """Tests targeting remaining uncovered lines in promoter_researcher.py."""

    # --- line 141: corpus refs source added + lines 153-158 dedup merge ---
    @pytest.mark.asyncio
    async def test_research_merges_corpus_refs_dedup(self):
        """Corpus refs merged into articles with deduplication."""
        from src.models.rag import DocumentChunk, RetrievedChunk
        from src.services.promoter_researcher import PromoterResearcher

        chunk = DocumentChunk(
            chunk_id="c1",
            text="Bugged Out is a legendary promoter" * 10,
            source_id="src1",
            source_title="Energy Flash",
            source_type="book",
            citation_tier=1,
        )
        retrieved = RetrievedChunk(chunk=chunk, similarity_score=0.85)
        vector_store = _mock_vector_store(results=[retrieved])

        web_search = _mock_web_search(results=[
            _search_result(
                title="Bugged Out RA",
                url="https://ra.co/promoters/buggedout",
                snippet="Bugged Out promoter events",
            ),
        ])

        article_scraper = _mock_article_scraper(
            content=_article_content(
                title="Bugged Out Profile",
                text="Bugged Out is a London promoter.",
            )
        )

        llm = _mock_llm(
            response=(
                "BASED_IN:\n- London, England, UK\n\n"
                "EVENT_HISTORY:\n- Bugged Out! at Fabric 2001\n\n"
                "AFFILIATED_ARTISTS:\n- Erol Alkan\n\n"
                "AFFILIATED_VENUES:\n- Fabric"
            )
        )

        researcher = PromoterResearcher(
            web_search=web_search,
            article_scraper=article_scraper,
            llm=llm,
            vector_store=vector_store,
        )

        result = await researcher.research("Bugged Out", city="London")

        assert "rag_corpus" in result.sources_consulted
        assert result.promoter is not None
        # Corpus ref title should appear in articles
        article_titles = [a.title for a in result.promoter.articles if a.title]
        assert "Energy Flash" in article_titles

    # --- lines 222-253: _extract_promoter_profile with city context ---
    @pytest.mark.asyncio
    async def test_extract_promoter_profile_with_city_context(self):
        """Promoter profile extraction includes city context in LLM prompt."""
        from src.services.promoter_researcher import PromoterResearcher

        llm = _mock_llm(
            response=(
                "BASED_IN:\n- London, England, UK\n\n"
                "EVENT_HISTORY:\n- Bugged Out! at Fabric, 2001\n\n"
                "AFFILIATED_ARTISTS:\n- Erol Alkan\n- James Murphy\n\n"
                "AFFILIATED_VENUES:\n- Fabric\n- Sankeys"
            )
        )

        researcher = PromoterResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=llm,
        )

        result = await researcher._extract_promoter_profile(
            "Bugged Out",
            ["Bugged Out is a London promoter who throws events at Fabric."],
            city="London",
        )

        assert result["based_city"] == "London"
        assert result["based_region"] == "England"
        assert result["based_country"] == "UK"
        assert len(result["event_history"]) >= 1
        assert len(result["affiliated_artists"]) >= 1
        assert len(result["affiliated_venues"]) >= 1

        # Verify city context was included in the prompt
        call_args = llm.complete.call_args
        user_prompt = call_args.kwargs.get("user_prompt", call_args[1].get("user_prompt", ""))
        assert "London" in user_prompt

    # --- lines 287-288: _search_promoter_activity ResearchError ---
    @pytest.mark.asyncio
    async def test_search_promoter_activity_research_error_handled(self):
        """ResearchError in search is caught, other queries proceed."""
        from src.services.promoter_researcher import PromoterResearcher

        call_count = 0

        async def search_side_effect(query, num_results=12):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                from src.utils.errors import ResearchError
                raise ResearchError("Search API down")
            return [_search_result(url=f"https://ra.co/promo/{call_count}")]

        web_search = _mock_web_search()
        web_search.search = AsyncMock(side_effect=search_side_effect)

        researcher = PromoterResearcher(
            web_search=web_search,
            article_scraper=_mock_article_scraper(content=None),
            llm=_mock_llm(
                response="BASED_IN:\nNONE\nEVENT_HISTORY:\nNONE\n"
                "AFFILIATED_ARTISTS:\nNONE\nAFFILIATED_VENUES:\nNONE"
            ),
        )

        result = await researcher.research("Test Promoter")

        assert call_count > 1

    # --- lines 321-322: deepening queries in _search_promoter_activity ---
    @pytest.mark.asyncio
    async def test_search_promoter_activity_deepening_research_error(self):
        """ResearchError in deepening queries is silently caught."""
        from src.services.promoter_researcher import PromoterResearcher
        from src.utils.errors import ResearchError

        call_count = 0

        async def search_side_effect(query, num_results=12):
            nonlocal call_count
            call_count += 1
            if call_count <= 4:
                return []  # Sparse initial results
            raise ResearchError("Deep search failed too")

        web_search = _mock_web_search()
        web_search.search = AsyncMock(side_effect=search_side_effect)

        researcher = PromoterResearcher(
            web_search=web_search,
            article_scraper=_mock_article_scraper(content=None),
            llm=_mock_llm(
                response="BASED_IN:\nNONE\nEVENT_HISTORY:\nNONE\n"
                "AFFILIATED_ARTISTS:\nNONE\nAFFILIATED_VENUES:\nNONE"
            ),
        )

        # Should not raise
        results = await researcher._search_promoter_activity("Test Promoter")

        assert call_count > 4

    # --- line 340: _scrape_results with exception items ---
    @pytest.mark.asyncio
    async def test_scrape_results_handles_mixed_results(self):
        """Scrape with mixed success and failures returns only successes."""
        from src.services.promoter_researcher import PromoterResearcher

        article_scraper = _mock_article_scraper()
        article_scraper.extract_content = AsyncMock(
            side_effect=[
                _article_content(text="Good promoter article."),
                RuntimeError("Timeout"),
                _article_content(text="Another good article."),
            ]
        )

        researcher = PromoterResearcher(
            web_search=_mock_web_search(),
            article_scraper=article_scraper,
            llm=_mock_llm(),
        )

        results = [
            _search_result(url="https://example.com/1"),
            _search_result(url="https://example.com/2"),
            _search_result(url="https://example.com/3"),
        ]
        texts = await researcher._scrape_results(results)

        assert len(texts) == 2

    # --- lines 434-435: _build_article_references extraction failure ---
    @pytest.mark.asyncio
    async def test_build_article_references_extraction_failure(self):
        """Article extraction failure is caught; other articles still returned."""
        from src.services.promoter_researcher import PromoterResearcher

        article_scraper = _mock_article_scraper()
        article_scraper.extract_content = AsyncMock(
            side_effect=[
                _article_content(title="Good Article", text="Promoter article."),
                RuntimeError("Scrape timeout"),
            ]
        )

        researcher = PromoterResearcher(
            web_search=_mock_web_search(),
            article_scraper=article_scraper,
            llm=_mock_llm(),
        )

        results = [
            _search_result(url="https://ra.co/1"),
            _search_result(url="https://ra.co/2"),
        ]
        articles = await researcher._build_article_references(results, "Test Promoter")

        assert len(articles) == 1
        assert articles[0].title == "Good Article"

    @pytest.mark.asyncio
    async def test_build_article_references_empty_results(self):
        """Empty search results returns empty article list."""
        from src.services.promoter_researcher import PromoterResearcher

        researcher = PromoterResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
        )

        articles = await researcher._build_article_references([], "Test Promoter")

        assert articles == []

    # --- lines 452-458: _extract_article_reference with various types ---
    @pytest.mark.asyncio
    async def test_extract_article_reference_interview_type(self):
        """Promoter article with 'interview' in title classified as interview."""
        from src.services.promoter_researcher import PromoterResearcher

        researcher = PromoterResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(
                content=_article_content(
                    title="Interview with Test Promoter",
                    text="Test Promoter discusses club nights.",
                )
            ),
            llm=_mock_llm(),
        )

        ref = await researcher._extract_article_reference(
            _search_result(url="https://ra.co/features/interview"),
            "Test Promoter",
        )

        assert ref.article_type == "interview"

    @pytest.mark.asyncio
    async def test_extract_article_reference_event_type(self):
        """Promoter article with 'event' in title classified as event."""
        from src.services.promoter_researcher import PromoterResearcher

        researcher = PromoterResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(
                content=_article_content(
                    title="Event: Bugged Out at Fabric",
                    text="Bugged Out event listing for Fabric London.",
                )
            ),
            llm=_mock_llm(),
        )

        ref = await researcher._extract_article_reference(
            _search_result(url="https://ra.co/events/123"),
            "Bugged Out",
        )

        assert ref.article_type == "event"

    @pytest.mark.asyncio
    async def test_extract_article_reference_no_content_fallback(self):
        """No scraped content — falls back to search result snippet."""
        from src.services.promoter_researcher import PromoterResearcher

        researcher = PromoterResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(content=None),
            llm=_mock_llm(),
        )

        sr = _search_result(
            title="Bugged Out RA Profile",
            url="https://ra.co/promoters/buggedout",
            snippet="Bugged Out promoter page",
        )
        ref = await researcher._extract_article_reference(sr, "Bugged Out")

        assert ref.title == "Bugged Out RA Profile"
        assert ref.snippet == "Bugged Out promoter page"
        assert ref.article_type == "article"

    # --- lines 552-599: cache helpers ---
    @pytest.mark.asyncio
    async def test_cache_get_failure_returns_none(self):
        """Cache get failure returns None silently."""
        from src.services.promoter_researcher import PromoterResearcher

        cache = _mock_cache()
        cache.get = AsyncMock(side_effect=RuntimeError("Redis down"))

        researcher = PromoterResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=cache,
        )

        result = await researcher._cache_get("some_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_set_failure_handled_silently(self):
        """Cache set failure is caught and does not raise."""
        from src.services.promoter_researcher import PromoterResearcher

        cache = _mock_cache()
        cache.set = AsyncMock(side_effect=RuntimeError("Redis down"))

        researcher = PromoterResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=cache,
        )

        await researcher._cache_set("some_key", {"data": True})

    @pytest.mark.asyncio
    async def test_retrieve_from_corpus_with_results(self):
        """Vector store returns matching chunks converted to ArticleReferences."""
        from src.models.rag import DocumentChunk, RetrievedChunk
        from src.services.promoter_researcher import PromoterResearcher

        chunk = DocumentChunk(
            chunk_id="c1",
            text="Bugged Out is a legendary London promoter." * 5,
            source_id="src1",
            source_title="Energy Flash",
            source_type="book",
            publication_date=date(1998, 1, 1),
            citation_tier=1,
        )
        retrieved = RetrievedChunk(chunk=chunk, similarity_score=0.88)
        vector_store = _mock_vector_store(results=[retrieved])

        researcher = PromoterResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            vector_store=vector_store,
        )

        refs = await researcher._retrieve_from_corpus("Bugged Out", city="London")

        assert len(refs) == 1
        assert refs[0].title == "Energy Flash"
        assert refs[0].article_type == "book"

    @pytest.mark.asyncio
    async def test_retrieve_from_corpus_exception(self):
        """Vector store exception returns empty list."""
        from src.services.promoter_researcher import PromoterResearcher

        vector_store = _mock_vector_store()
        vector_store.is_available = MagicMock(return_value=True)
        vector_store.query = AsyncMock(side_effect=RuntimeError("ChromaDB error"))

        researcher = PromoterResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            vector_store=vector_store,
        )

        refs = await researcher._retrieve_from_corpus("Bugged Out")

        assert refs == []


# ======================================================================
# Coverage-targeted: EventNameResearcher — lines 146, 167-172,
#   239-261 (_extract_event_instances with scraped texts),
#   290-291, 308-311, 489, 530-554, 571-618
# ======================================================================


class TestEventNameResearcherCoverageFill:
    """Tests targeting remaining uncovered lines in event_name_researcher.py."""

    # --- lines 146, 167-172: corpus refs merged into articles ---
    @pytest.mark.asyncio
    async def test_research_merges_corpus_refs(self):
        """Corpus refs merged into articles with deduplication."""
        from src.models.rag import DocumentChunk, RetrievedChunk
        from src.services.event_name_researcher import EventNameResearcher

        chunk = DocumentChunk(
            chunk_id="c1",
            text="Bugged Out is a legendary club night" * 10,
            source_id="src1",
            source_title="Last Night a DJ Saved My Life",
            source_type="book",
            citation_tier=2,
        )
        retrieved = RetrievedChunk(chunk=chunk, similarity_score=0.85)
        vector_store = _mock_vector_store(results=[retrieved])

        instances_json = json.dumps([{
            "event_name": "Bugged Out!",
            "promoter": "Bugged Out Promotions",
            "venue": "Fabric",
            "city": "London",
            "date": "2001-05-12",
            "source_url": None,
        }])

        web_search = _mock_web_search(results=[
            _search_result(title="Bugged Out!", url="https://ra.co/events/123"),
        ])

        researcher = EventNameResearcher(
            web_search=web_search,
            article_scraper=_mock_article_scraper(
                content=_article_content(
                    title="Bugged Out! History",
                    text="Bugged Out! is a legendary club night.",
                )
            ),
            llm=_mock_llm(response=instances_json),
            vector_store=vector_store,
        )

        result = await researcher.research("Bugged Out!")

        assert "rag_corpus" in result.sources_consulted

    # --- lines 239-261: _extract_event_instances with scraped texts ---
    @pytest.mark.asyncio
    async def test_extract_event_instances_with_scraped_texts(self):
        """Event extraction with scraped texts calls LLM and parses response."""
        from src.services.event_name_researcher import EventNameResearcher

        instances_json = json.dumps([
            {
                "event_name": "BLOC",
                "promoter": "BLOC Promotions",
                "venue": "Butlins",
                "city": "Minehead",
                "date": "2009-03-13",
                "source_url": "https://ra.co/events/bloc",
            },
            {
                "event_name": "BLOC",
                "promoter": "BLOC Promotions",
                "venue": "London Pleasure Gardens",
                "city": "London",
                "date": "2012-07-06",
                "source_url": None,
            },
        ])

        llm = _mock_llm(response=instances_json)

        researcher = EventNameResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=llm,
        )

        instances = await researcher._extract_event_instances(
            "BLOC",
            ["BLOC is a legendary electronic music festival.", "BLOC 2012 at London Pleasure Gardens."],
        )

        assert len(instances) == 2
        assert instances[0].event_name == "BLOC"
        assert instances[0].venue == "Butlins"
        assert instances[1].city == "London"

    # --- lines 290-291: _search_event_instances ResearchError ---
    @pytest.mark.asyncio
    async def test_search_event_instances_research_error_handled(self):
        """ResearchError in search is caught, other queries proceed."""
        from src.services.event_name_researcher import EventNameResearcher

        call_count = 0

        async def search_side_effect(query, num_results=15):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                from src.utils.errors import ResearchError
                raise ResearchError("Search API error")
            return [_search_result(url=f"https://ra.co/events/{call_count}")]

        web_search = _mock_web_search()
        web_search.search = AsyncMock(side_effect=search_side_effect)

        researcher = EventNameResearcher(
            web_search=web_search,
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(response="[]"),
        )

        results = await researcher._search_event_instances("TestEvent", promoter_name=None)

        assert call_count > 1
        assert len(results) >= 1

    # --- lines 308-311: _search_event_instances deepening ResearchError ---
    @pytest.mark.asyncio
    async def test_search_event_instances_deepening_error_handled(self):
        """ResearchError in deepening search is caught silently."""
        from src.services.event_name_researcher import EventNameResearcher
        from src.utils.errors import ResearchError

        call_count = 0

        async def search_side_effect(query, num_results=15):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return []  # Sparse initial results
            raise ResearchError("Deeper search also failed")

        web_search = _mock_web_search()
        web_search.search = AsyncMock(side_effect=search_side_effect)

        researcher = EventNameResearcher(
            web_search=web_search,
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(response="[]"),
        )

        results = await researcher._search_event_instances("TestEvent", promoter_name=None)

        # Should not raise, deepening queries caught
        assert call_count > 3

    # --- line 489: _parse_name_changes with fence ---
    def test_parse_name_changes_fence_wrapped(self):
        """JSON wrapped in ```json ... ``` fence parses correctly."""
        from src.services.event_name_researcher import EventNameResearcher

        text = '```json\n["Promoter A and B are the same (rebrand 2010)"]\n```'
        changes = EventNameResearcher._parse_name_changes(text)

        assert len(changes) == 1
        assert "rebrand 2010" in changes[0]

    # --- lines 530-554: _build_article_references with extraction failures ---
    @pytest.mark.asyncio
    async def test_build_article_references_extraction_failure(self):
        """Article extraction failure is caught; other articles still returned."""
        from src.services.event_name_researcher import EventNameResearcher

        article_scraper = _mock_article_scraper()
        article_scraper.extract_content = AsyncMock(
            side_effect=[
                _article_content(title="Good Event Article", text="TestEvent details."),
                RuntimeError("Scrape timeout"),
            ]
        )

        researcher = EventNameResearcher(
            web_search=_mock_web_search(),
            article_scraper=article_scraper,
            llm=_mock_llm(),
        )

        results = [
            _search_result(url="https://ra.co/1"),
            _search_result(url="https://ra.co/2"),
        ]
        articles = await researcher._build_article_references(results, "TestEvent")

        assert len(articles) == 1
        assert articles[0].title == "Good Event Article"

    @pytest.mark.asyncio
    async def test_build_article_references_empty(self):
        """Empty search results returns empty article list."""
        from src.services.event_name_researcher import EventNameResearcher

        researcher = EventNameResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
        )

        articles = await researcher._build_article_references([], "TestEvent")

        assert articles == []

    # --- lines 547-554: _extract_article_reference type classification ---
    @pytest.mark.asyncio
    async def test_extract_article_reference_interview_type(self):
        """Event article with 'interview' in title classified as interview."""
        from src.services.event_name_researcher import EventNameResearcher

        researcher = EventNameResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(
                content=_article_content(
                    title="Interview: BLOC Festival Organizers",
                    text="BLOC festival organizers discuss plans.",
                )
            ),
            llm=_mock_llm(),
        )

        ref = await researcher._extract_article_reference(
            _search_result(url="https://ra.co/features/interview"),
            "BLOC",
        )

        assert ref.article_type == "interview"

    @pytest.mark.asyncio
    async def test_extract_article_reference_event_type(self):
        """Event article with 'event' in title classified as event."""
        from src.services.event_name_researcher import EventNameResearcher

        researcher = EventNameResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(
                content=_article_content(
                    title="Event: BLOC Festival 2012",
                    text="BLOC festival event listing.",
                )
            ),
            llm=_mock_llm(),
        )

        ref = await researcher._extract_article_reference(
            _search_result(url="https://ra.co/events/bloc"),
            "BLOC",
        )

        assert ref.article_type == "event"

    @pytest.mark.asyncio
    async def test_extract_article_reference_no_content_fallback(self):
        """No scraped content — falls back to search result snippet."""
        from src.services.event_name_researcher import EventNameResearcher

        researcher = EventNameResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(content=None),
            llm=_mock_llm(),
        )

        sr = _search_result(
            title="BLOC RA listing",
            url="https://ra.co/events/bloc",
            snippet="BLOC festival event",
        )
        ref = await researcher._extract_article_reference(sr, "BLOC")

        assert ref.title == "BLOC RA listing"
        assert ref.snippet == "BLOC festival event"
        assert ref.article_type == "article"

    # --- lines 571-618: cache helpers ---
    @pytest.mark.asyncio
    async def test_cache_get_failure_returns_none(self):
        """Cache get failure returns None silently."""
        from src.services.event_name_researcher import EventNameResearcher

        cache = _mock_cache()
        cache.get = AsyncMock(side_effect=RuntimeError("Redis down"))

        researcher = EventNameResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=cache,
        )

        result = await researcher._cache_get("some_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_set_failure_handled_silently(self):
        """Cache set failure is caught and does not raise."""
        from src.services.event_name_researcher import EventNameResearcher

        cache = _mock_cache()
        cache.set = AsyncMock(side_effect=RuntimeError("Redis down"))

        researcher = EventNameResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=cache,
        )

        await researcher._cache_set("some_key", {"data": True})

    @pytest.mark.asyncio
    async def test_cache_get_no_cache_configured(self):
        """No cache configured — returns None."""
        from src.services.event_name_researcher import EventNameResearcher

        researcher = EventNameResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=None,
        )

        result = await researcher._cache_get("some_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_set_no_cache_configured(self):
        """No cache configured — set is a no-op."""
        from src.services.event_name_researcher import EventNameResearcher

        researcher = EventNameResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            cache=None,
        )

        await researcher._cache_set("some_key", {"data": True})

    # --- _retrieve_from_corpus with article type (non-book) ---
    @pytest.mark.asyncio
    async def test_retrieve_from_corpus_article_type(self):
        """Corpus chunk with non-book source_type produces article type."""
        from src.models.rag import DocumentChunk, RetrievedChunk
        from src.services.event_name_researcher import EventNameResearcher

        chunk = DocumentChunk(
            chunk_id="c1",
            text="BLOC festival article content" * 10,
            source_id="src1",
            source_title="RA Feature on BLOC",
            source_type="article",
            citation_tier=1,
        )
        retrieved = RetrievedChunk(chunk=chunk, similarity_score=0.9)
        vector_store = _mock_vector_store(results=[retrieved])

        researcher = EventNameResearcher(
            web_search=_mock_web_search(),
            article_scraper=_mock_article_scraper(),
            llm=_mock_llm(),
            vector_store=vector_store,
        )

        refs = await researcher._retrieve_from_corpus("BLOC")

        assert len(refs) == 1
        assert refs[0].article_type == "article"
        assert refs[0].title == "RA Feature on BLOC"


# ======================================================================
# Coverage-targeted: QAService — lines 248, 276-301, 306-310, 326,
#   348, 417, 422, 455-457
# ======================================================================


class TestQAService:
    """Tests for the QAService targeting uncovered lines."""

    def _make_session_context(self):
        """Build a realistic session context dict."""
        return {
            "session_id": "test-session-123",
            "extracted_entities": {
                "artists": [
                    {"text": "Carl Cox", "name": "Carl Cox"},
                    {"text": "Jeff Mills", "name": "Jeff Mills"},
                ],
                "venue": {"text": "Tresor", "name": "Tresor"},
                "promoter": {"text": "Tresor Berlin", "name": "Tresor Berlin"},
                "date": {"text": "March 1997"},
                "genre_tags": ["techno", "acid"],
            },
            "research_results": [
                {
                    "entity_name": "Carl Cox",
                    "artist": {
                        "profile_summary": "Carl Cox is a British techno DJ.",
                        "releases": [
                            {"title": "Phat Trax", "year": "1995"},
                            {"title": "F.A.C.T.", "year": "1995"},
                        ],
                    },
                },
                {
                    "entity_name": "Tresor",
                    "venue": {
                        "history": "Tresor opened in 1991 in a vault.",
                    },
                },
            ],
            "interconnection_map": {
                "narrative": "Carl Cox performed at Tresor during the Berlin techno golden era.",
            },
        }

    # --- ask() success path ---
    @pytest.mark.asyncio
    async def test_ask_success(self):
        """Successful ask returns QAResponse with answer, citations, and facts."""
        from src.services.qa_service import QAResponse, QAService

        llm_response = json.dumps({
            "answer": "Carl Cox is a legendary British techno DJ.",
            "citations": [
                {"text": "Energy Flash, p.142", "source": "Energy Flash", "tier": 1},
            ],
            "related_facts": [
                {"text": "Carl Cox played at Tresor in 1993.", "category": "HISTORY", "entity_name": "Carl Cox"},
            ],
        })

        llm = _mock_llm(response=llm_response)

        service = QAService(llm=llm, vector_store=None, cache=None)

        response = await service.ask(
            question="Tell me about Carl Cox",
            session_context=self._make_session_context(),
            entity_type="ARTIST",
            entity_name="Carl Cox",
        )

        assert isinstance(response, QAResponse)
        assert "Carl Cox" in response.answer
        assert len(response.citations) >= 1
        assert len(response.related_facts) >= 1

    # --- ask() LLM failure ---
    @pytest.mark.asyncio
    async def test_ask_llm_failure(self):
        """LLM failure returns fallback QAResponse."""
        from src.services.qa_service import QAResponse, QAService

        llm = _mock_llm()
        llm.complete = AsyncMock(side_effect=RuntimeError("LLM API timeout"))

        service = QAService(llm=llm, vector_store=None, cache=None)

        response = await service.ask(
            question="Tell me about Carl Cox",
            session_context=self._make_session_context(),
        )

        assert isinstance(response, QAResponse)
        assert "wasn't able to answer" in response.answer.lower()
        assert response.citations == []
        assert response.related_facts == []

    # --- ask() with cache hit ---
    @pytest.mark.asyncio
    async def test_ask_cache_hit(self):
        """Cache hit returns cached QAResponse without LLM call."""
        from src.services.qa_service import QAResponse, QAService

        cached_data = json.dumps({
            "answer": "Cached answer about Carl Cox.",
            "citations": [{"text": "Cached citation", "source": "Cache", "tier": 3}],
            "related_facts": [{"text": "Cached fact", "category": "ARTIST"}],
        })

        cache = _mock_cache()
        cache.get = AsyncMock(return_value=cached_data)

        llm = _mock_llm()

        service = QAService(llm=llm, vector_store=None, cache=cache)

        response = await service.ask(
            question="Tell me about Carl Cox",
            session_context=self._make_session_context(),
        )

        assert isinstance(response, QAResponse)
        assert "Cached answer" in response.answer
        # LLM should not have been called
        llm.complete.assert_not_awaited()

    # --- ask() with cache + caches the result ---
    @pytest.mark.asyncio
    async def test_ask_caches_result(self):
        """Successful ask stores result in cache."""
        from src.services.qa_service import QAService

        llm_response = json.dumps({
            "answer": "Carl Cox is a DJ.",
            "citations": [],
            "related_facts": [],
        })

        cache = _mock_cache()
        llm = _mock_llm(response=llm_response)

        service = QAService(llm=llm, vector_store=None, cache=cache)

        await service.ask(
            question="Tell me about Carl Cox",
            session_context=self._make_session_context(),
        )

        # cache.set should have been called
        cache.set.assert_awaited_once()

    # --- _build_context_summary ---
    def test_build_context_summary(self):
        """Context summary includes entities, research, and interconnections."""
        from src.services.qa_service import QAService

        service = QAService(llm=_mock_llm(), vector_store=None, cache=None)
        context = self._make_session_context()

        summary = service._build_context_summary(context)

        assert "Carl Cox" in summary
        assert "Jeff Mills" in summary
        assert "Tresor" in summary
        assert "March 1997" in summary
        assert "techno" in summary

    def test_build_context_summary_with_entity_filter(self):
        """Context summary with entity_name includes detailed context."""
        from src.services.qa_service import QAService

        service = QAService(llm=_mock_llm(), vector_store=None, cache=None)
        context = self._make_session_context()

        summary = service._build_context_summary(
            context, entity_type="ARTIST", entity_name="Carl Cox"
        )

        assert "Carl Cox" in summary
        # Should include detailed artist profile summary
        assert "Profile:" in summary or "British techno DJ" in summary

    def test_build_context_summary_venue_detail(self):
        """Context summary for venue entity includes venue history detail."""
        from src.services.qa_service import QAService

        service = QAService(llm=_mock_llm(), vector_store=None, cache=None)
        context = self._make_session_context()

        summary = service._build_context_summary(
            context, entity_type="VENUE", entity_name="Tresor"
        )

        assert "Tresor" in summary
        assert "1991" in summary or "History:" in summary

    def test_build_context_summary_empty_entities(self):
        """Empty entities produces minimal summary."""
        from src.services.qa_service import QAService

        service = QAService(llm=_mock_llm(), vector_store=None, cache=None)

        summary = service._build_context_summary(
            {"session_id": "test", "extracted_entities": {}},
        )

        # Should not crash, returns whatever is there
        assert isinstance(summary, str)

    def test_build_context_summary_with_interconnection_narrative(self):
        """Interconnection narrative is included in the context summary."""
        from src.services.qa_service import QAService

        service = QAService(llm=_mock_llm(), vector_store=None, cache=None)
        context = self._make_session_context()

        summary = service._build_context_summary(context)

        assert "Interconnection narrative:" in summary or "Berlin techno" in summary

    # --- _build_user_prompt ---
    def test_build_user_prompt(self):
        """User prompt assembles all sections correctly."""
        from src.services.qa_service import QAService

        service = QAService(llm=_mock_llm(), vector_store=None, cache=None)
        context = self._make_session_context()

        prompt = service._build_user_prompt(
            question="Tell me about Carl Cox",
            context_summary="Artists on flier: Carl Cox, Jeff Mills",
            rag_context="[Source: Energy Flash, Tier 1]\nCarl Cox is a DJ.",
            entity_type="ARTIST",
            entity_name="Carl Cox",
            session_context=context,
        )

        assert "Flier Analysis Context" in prompt
        assert "Retrieved Knowledge Base Passages" in prompt
        assert "Focus Entity" in prompt
        assert "ARTIST" in prompt
        assert "Carl Cox" in prompt
        assert "User Question" in prompt

    def test_build_user_prompt_without_rag(self):
        """User prompt without RAG context omits that section."""
        from src.services.qa_service import QAService

        service = QAService(llm=_mock_llm(), vector_store=None, cache=None)

        prompt = service._build_user_prompt(
            question="Tell me about Carl Cox",
            context_summary="Artists on flier: Carl Cox",
            rag_context="",
            entity_type=None,
            entity_name=None,
            session_context=self._make_session_context(),
        )

        assert "Retrieved Knowledge Base" not in prompt
        assert "Focus Entity" not in prompt

    # --- _retrieve_passages ---
    @pytest.mark.asyncio
    async def test_retrieve_passages_with_vector_store(self):
        """Passages retrieved from vector store with entity filter."""
        from src.models.rag import DocumentChunk, RetrievedChunk
        from src.services.qa_service import QAService

        chunk = DocumentChunk(
            chunk_id="c1",
            text="Carl Cox is a legendary techno DJ." * 5,
            source_id="src1",
            source_title="Energy Flash",
            source_type="book",
            citation_tier=1,
        )
        retrieved = RetrievedChunk(
            chunk=chunk,
            similarity_score=0.85,
            formatted_citation="Energy Flash, p.142",
        )
        vector_store = _mock_vector_store(results=[retrieved])

        service = QAService(llm=_mock_llm(), vector_store=vector_store, cache=None)

        rag_text, citations = await service._retrieve_passages(
            "Who is Carl Cox?", entity_name="Carl Cox"
        )

        assert "Energy Flash" in rag_text
        assert len(citations) >= 1
        assert citations[0]["source"] == "Energy Flash"
        assert citations[0]["tier"] == 1

    @pytest.mark.asyncio
    async def test_retrieve_passages_no_vector_store(self):
        """No vector store — returns empty text and empty citations."""
        from src.services.qa_service import QAService

        service = QAService(llm=_mock_llm(), vector_store=None, cache=None)

        rag_text, citations = await service._retrieve_passages(
            "Who is Carl Cox?", entity_name=None
        )

        assert rag_text == ""
        assert citations == []

    @pytest.mark.asyncio
    async def test_retrieve_passages_low_similarity_filtered(self):
        """Chunks below similarity threshold are filtered out."""
        from src.models.rag import DocumentChunk, RetrievedChunk
        from src.services.qa_service import QAService

        chunk = DocumentChunk(
            chunk_id="c1",
            text="Irrelevant text" * 10,
            source_id="src1",
            source_title="Random Book",
            source_type="article",
            citation_tier=5,
        )
        retrieved = RetrievedChunk(chunk=chunk, similarity_score=0.3)
        vector_store = _mock_vector_store(results=[retrieved])

        service = QAService(llm=_mock_llm(), vector_store=vector_store, cache=None)

        rag_text, citations = await service._retrieve_passages(
            "Who is Carl Cox?", entity_name=None
        )

        assert rag_text == ""
        assert citations == []

    @pytest.mark.asyncio
    async def test_retrieve_passages_exception_handled(self):
        """Vector store exception returns empty text and citations."""
        from src.services.qa_service import QAService

        vector_store = _mock_vector_store()
        vector_store.is_available = MagicMock(return_value=True)
        vector_store.query = AsyncMock(side_effect=RuntimeError("ChromaDB down"))

        service = QAService(llm=_mock_llm(), vector_store=vector_store, cache=None)

        rag_text, citations = await service._retrieve_passages(
            "Who is Carl Cox?", entity_name=None
        )

        assert rag_text == ""
        assert citations == []

    # --- _parse_response ---
    def test_parse_response_valid_json(self):
        """Valid JSON response is parsed correctly."""
        from src.services.qa_service import QAService

        service = QAService(llm=_mock_llm(), vector_store=None, cache=None)

        raw = json.dumps({
            "answer": "Carl Cox is a DJ.",
            "citations": [{"text": "Citation A", "source": "Book A", "tier": 1}],
            "related_facts": [{"text": "Fact about Carl Cox", "category": "ARTIST"}],
        })

        rag_citations = [{"text": "RAG citation", "source": "Energy Flash", "tier": 1}]

        response = service._parse_response(raw, rag_citations)

        assert "Carl Cox" in response.answer
        # RAG citations + LLM citations
        assert len(response.citations) >= 2
        assert len(response.related_facts) >= 1

    def test_parse_response_markdown_fence(self):
        """JSON wrapped in markdown code blocks is parsed."""
        from src.services.qa_service import QAService

        service = QAService(llm=_mock_llm(), vector_store=None, cache=None)

        raw = '```json\n{"answer": "Fenced answer.", "citations": [], "related_facts": []}\n```'

        response = service._parse_response(raw, [])

        assert response.answer == "Fenced answer."

    def test_parse_response_invalid_json_fallback(self):
        """Invalid JSON falls back to raw text as answer."""
        from src.services.qa_service import QAService

        service = QAService(llm=_mock_llm(), vector_store=None, cache=None)

        raw = "This is not JSON at all, just plain text."
        rag_citations = [{"text": "RAG citation", "source": "Book", "tier": 2}]

        response = service._parse_response(raw, rag_citations)

        assert response.answer == raw
        assert response.citations == rag_citations
        assert response.related_facts == []

    def test_parse_response_merges_deduplicates_citations(self):
        """LLM citations are merged with RAG citations, duplicates skipped."""
        from src.services.qa_service import QAService

        service = QAService(llm=_mock_llm(), vector_store=None, cache=None)

        raw = json.dumps({
            "answer": "Answer text.",
            "citations": [
                {"text": "Same citation", "source": "Book A", "tier": 1},
                {"text": "Unique LLM citation", "source": "Book B", "tier": 2},
            ],
            "related_facts": [],
        })

        rag_citations = [{"text": "Same citation", "source": "Book A", "tier": 1}]

        response = service._parse_response(raw, rag_citations)

        # "Same citation" should appear only once
        citation_texts = [c["text"] for c in response.citations]
        assert citation_texts.count("Same citation") == 1
        assert "Unique LLM citation" in citation_texts

    def test_parse_response_legacy_suggested_questions_key(self):
        """Legacy 'suggested_questions' key is used as fallback for related_facts."""
        from src.services.qa_service import QAService

        service = QAService(llm=_mock_llm(), vector_store=None, cache=None)

        raw = json.dumps({
            "answer": "Answer text.",
            "citations": [],
            "suggested_questions": [
                {"text": "Legacy fact", "category": "HISTORY"},
            ],
        })

        response = service._parse_response(raw, [])

        assert len(response.related_facts) == 1
        assert response.related_facts[0]["text"] == "Legacy fact"

    # --- _extract_entity_names ---
    def test_extract_entity_names(self):
        """Entity names extracted from session context."""
        from src.services.qa_service import QAService

        names = QAService._extract_entity_names(self._make_session_context())

        assert "Carl Cox" in names
        assert "Jeff Mills" in names
        assert "Tresor" in names
        assert "Tresor Berlin" in names

    def test_extract_entity_names_none_context(self):
        """None session context returns empty list."""
        from src.services.qa_service import QAService

        names = QAService._extract_entity_names(None)

        assert names == []

    def test_extract_entity_names_empty_entities(self):
        """Empty entities dict returns empty list."""
        from src.services.qa_service import QAService

        names = QAService._extract_entity_names(
            {"session_id": "test", "extracted_entities": {}}
        )

        assert names == []

    # --- _cache_key ---
    def test_cache_key_deterministic(self):
        """Same inputs produce same cache key."""
        from src.services.qa_service import QAService

        key1 = QAService._cache_key("question", "ARTIST", "Carl Cox", "session1")
        key2 = QAService._cache_key("question", "ARTIST", "Carl Cox", "session1")

        assert key1 == key2

    def test_cache_key_varies_by_input(self):
        """Different inputs produce different cache keys."""
        from src.services.qa_service import QAService

        key1 = QAService._cache_key("question A", "ARTIST", "Carl Cox", "session1")
        key2 = QAService._cache_key("question B", "ARTIST", "Carl Cox", "session1")

        assert key1 != key2

    # --- QAResponse data hiding ---
    def test_qa_response_properties(self):
        """QAResponse properties return correct data via defensive copies."""
        from src.services.qa_service import QAResponse

        citations = [{"text": "Citation", "source": "Book", "tier": 1}]
        facts = [{"text": "Fact", "category": "ARTIST"}]

        response = QAResponse(answer="Answer text.", citations=citations, related_facts=facts)

        assert response.answer == "Answer text."
        assert response.citations == citations
        assert response.related_facts == facts
        # Verify defensive copies (modifying returned list does not affect internal)
        response.citations.append({"text": "New"})
        assert len(response.citations) == 1  # Still 1, not 2

    # --- ask() with RAG vector store ---
    @pytest.mark.asyncio
    async def test_ask_with_rag_vector_store(self):
        """Ask with vector store uses RAG retrieval in the answer."""
        from src.models.rag import DocumentChunk, RetrievedChunk
        from src.services.qa_service import QAService

        chunk = DocumentChunk(
            chunk_id="c1",
            text="Carl Cox is one of the most important techno DJs in history." * 3,
            source_id="src1",
            source_title="Energy Flash",
            source_type="book",
            citation_tier=1,
        )
        retrieved = RetrievedChunk(
            chunk=chunk,
            similarity_score=0.9,
            formatted_citation="Energy Flash, p.142",
        )
        vector_store = _mock_vector_store(results=[retrieved])

        llm_response = json.dumps({
            "answer": "Carl Cox is a pioneering techno DJ.",
            "citations": [{"text": "Energy Flash, p.142", "source": "Energy Flash", "tier": 1}],
            "related_facts": [{"text": "Carl Cox started DJing at age 15.", "category": "ARTIST"}],
        })

        service = QAService(
            llm=_mock_llm(response=llm_response),
            vector_store=vector_store,
            cache=None,
        )

        response = await service.ask(
            question="Tell me about Carl Cox",
            session_context=self._make_session_context(),
            entity_name="Carl Cox",
        )

        assert "Carl Cox" in response.answer
        assert len(response.citations) >= 1

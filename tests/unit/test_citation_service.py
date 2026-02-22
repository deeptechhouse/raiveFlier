"""Unit tests for the CitationService."""

from __future__ import annotations

from datetime import date

import pytest

from src.models.analysis import Citation
from src.services.citation_service import CitationService


@pytest.fixture()
def service() -> CitationService:
    """Create a CitationService instance for testing."""
    return CitationService()


# ======================================================================
# assign_tier_from_url
# ======================================================================


class TestAssignTierFromUrl:
    """Tests for URL-based tier assignment."""

    def test_residentadvisor_tier_2(self, service: CitationService) -> None:
        assert service.assign_tier_from_url("https://www.residentadvisor.net/reviews/123") == 2

    def test_djmag_tier_2(self, service: CitationService) -> None:
        assert service.assign_tier_from_url("https://djmag.com/features/article") == 2

    def test_mixmag_tier_2(self, service: CitationService) -> None:
        assert service.assign_tier_from_url("https://mixmag.net/feature/x") == 2

    def test_pitchfork_tier_2(self, service: CitationService) -> None:
        assert service.assign_tier_from_url("https://pitchfork.com/reviews/albums/x") == 2

    def test_factmag_tier_2(self, service: CitationService) -> None:
        assert service.assign_tier_from_url("https://factmag.com/x") == 2

    def test_xlr8r_tier_2(self, service: CitationService) -> None:
        assert service.assign_tier_from_url("https://xlr8r.com/features/x") == 2

    def test_19hz_tier_3(self, service: CitationService) -> None:
        assert service.assign_tier_from_url("https://19hz.info/eventlisting") == 3

    def test_discogs_tier_4(self, service: CitationService) -> None:
        assert service.assign_tier_from_url("https://www.discogs.com/artist/123") == 4

    def test_discogs_forum_tier_6(self, service: CitationService) -> None:
        assert service.assign_tier_from_url("https://www.discogs.com/forum/thread/123") == 6

    def test_musicbrainz_tier_4(self, service: CitationService) -> None:
        assert service.assign_tier_from_url("https://musicbrainz.org/artist/xyz") == 4

    def test_wikipedia_tier_5(self, service: CitationService) -> None:
        assert service.assign_tier_from_url("https://en.wikipedia.org/wiki/Carl_Cox") == 5

    def test_reddit_tier_6(self, service: CitationService) -> None:
        assert service.assign_tier_from_url("https://www.reddit.com/r/techno/comments/abc") == 6

    def test_unknown_domain_defaults_to_5(self, service: CitationService) -> None:
        assert service.assign_tier_from_url("https://unknown-blog.com/post") == 5


# ======================================================================
# assign_tier (combined URL + source_type)
# ======================================================================


class TestAssignTier:
    """Tests for the combined assign_tier method."""

    def test_url_takes_precedence(self, service: CitationService) -> None:
        tier = service.assign_tier(
            source_url="https://djmag.com/article",
            source_type="forum",
        )
        assert tier == 2

    def test_source_type_fallback_book(self, service: CitationService) -> None:
        tier = service.assign_tier(source_type="book")
        assert tier == 1

    def test_source_type_fallback_interview(self, service: CitationService) -> None:
        tier = service.assign_tier(source_type="interview")
        assert tier == 2

    def test_source_type_fallback_forum(self, service: CitationService) -> None:
        tier = service.assign_tier(source_type="forum")
        assert tier == 5

    def test_no_url_no_type_defaults_to_6(self, service: CitationService) -> None:
        tier = service.assign_tier()
        assert tier == 6

    def test_unknown_url_with_source_type(self, service: CitationService) -> None:
        # Unknown URL returns 5, but source_type might override if URL default is 5
        tier = service.assign_tier(
            source_url="https://random.com/page",
            source_type="book",
        )
        # URL matches nothing special → 5, which is the default — check logic:
        # assign_tier_from_url returns 5, assign_tier sees tier == 5,
        # then falls through to source_type
        assert tier == 1


# ======================================================================
# rank_citations
# ======================================================================


class TestRankCitations:
    """Tests for citation ranking."""

    def test_tier_order(self, service: CitationService) -> None:
        c_tier6 = Citation(text="forum", source_type="forum", source_name="reddit", tier=6)
        c_tier1 = Citation(text="book", source_type="book", source_name="pub", tier=1)
        c_tier3 = Citation(text="flier", source_type="flier", source_name="archive", tier=3)

        ranked = service.rank_citations([c_tier6, c_tier1, c_tier3])

        assert ranked[0].tier == 1
        assert ranked[1].tier == 3
        assert ranked[2].tier == 6

    def test_same_tier_sorted_by_date_newest_first(self, service: CitationService) -> None:
        c_old = Citation(
            text="old",
            source_type="press",
            source_name="a",
            tier=2,
            source_date=date(2020, 1, 1),
        )
        c_new = Citation(
            text="new",
            source_type="press",
            source_name="b",
            tier=2,
            source_date=date(2024, 6, 15),
        )

        ranked = service.rank_citations([c_old, c_new])

        assert ranked[0].text == "new"
        assert ranked[1].text == "old"

    def test_citations_without_date_sort_after_dated(self, service: CitationService) -> None:
        c_dated = Citation(
            text="dated",
            source_type="press",
            source_name="a",
            tier=2,
            source_date=date(2023, 1, 1),
        )
        c_undated = Citation(
            text="undated",
            source_type="press",
            source_name="b",
            tier=2,
        )

        ranked = service.rank_citations([c_undated, c_dated])

        assert ranked[0].text == "dated"
        assert ranked[1].text == "undated"

    def test_empty_list(self, service: CitationService) -> None:
        ranked = service.rank_citations([])
        assert ranked == []


# ======================================================================
# format_citation
# ======================================================================


class TestFormatCitation:
    """Tests for citation formatting across tiers."""

    def test_tier_1_book_format(self, service: CitationService) -> None:
        c = Citation(
            text="Energy Flash chapter 5",
            source_type="book",
            source_name="Simon Reynolds",
            tier=1,
            source_date=date(1998, 1, 1),
            page_number="142",
        )
        formatted = service.format_citation(c)
        assert "Simon Reynolds" in formatted
        assert "Energy Flash chapter 5" in formatted
        assert "p.142" in formatted
        assert "1998-01-01" in formatted

    def test_tier_2_press_format(self, service: CitationService) -> None:
        c = Citation(
            text="Carl Cox interview",
            source_type="press",
            source_name="Resident Advisor",
            tier=2,
            source_url="https://ra.co/features/123",
            source_date=date(2024, 3, 1),
        )
        formatted = service.format_citation(c)
        assert "Carl Cox interview" in formatted
        assert "Resident Advisor" in formatted
        assert "https://ra.co/features/123" in formatted
        assert "2024-03-01" in formatted
        assert " — " in formatted

    def test_tier_4_database_format(self, service: CitationService) -> None:
        c = Citation(
            text="Carl Cox — Phat Trax",
            source_type="database",
            source_name="Discogs",
            tier=4,
            source_url="https://discogs.com/release/123",
        )
        formatted = service.format_citation(c)
        assert "Discogs:" in formatted
        assert "Carl Cox — Phat Trax" in formatted
        assert "https://discogs.com/release/123" in formatted

    def test_tier_5_web_format(self, service: CitationService) -> None:
        c = Citation(
            text="Carl Cox Wikipedia bio",
            source_type="web",
            source_name="Wikipedia",
            tier=5,
            source_url="https://en.wikipedia.org/wiki/Carl_Cox",
        )
        formatted = service.format_citation(c)
        assert "Carl Cox Wikipedia bio" in formatted
        assert "Wikipedia" in formatted
        assert "https://en.wikipedia.org/wiki/Carl_Cox" in formatted

    def test_tier_6_forum_format(self, service: CitationService) -> None:
        c = Citation(
            text="Thread about Tresor history",
            source_type="forum",
            source_name="Reddit r/techno",
            tier=6,
            source_url="https://reddit.com/r/techno/comments/abc",
        )
        formatted = service.format_citation(c)
        assert "Thread about Tresor history" in formatted
        assert "Reddit r/techno" in formatted

    def test_tier_3_flier_archive_format(self, service: CitationService) -> None:
        c = Citation(
            text="Tresor 1997 flier",
            source_type="flier",
            source_name="Flyer Archive",
            tier=3,
            source_url="https://flyerarchive.com/tresor1997",
        )
        formatted = service.format_citation(c)
        assert "Tresor 1997 flier" in formatted
        assert "Flyer Archive" in formatted

    def test_tier_1_no_date_no_page(self, service: CitationService) -> None:
        c = Citation(
            text="General claim",
            source_type="book",
            source_name="Author Name",
            tier=1,
        )
        formatted = service.format_citation(c)
        assert "Author Name" in formatted
        assert "General claim" in formatted

    def test_tier_2_no_url(self, service: CitationService) -> None:
        c = Citation(
            text="Article title",
            source_type="press",
            source_name="DJ Mag",
            tier=2,
        )
        formatted = service.format_citation(c)
        assert "Article title — DJ Mag" in formatted


# ======================================================================
# build_citation
# ======================================================================


class TestBuildCitation:
    """Tests for the build_citation convenience method."""

    def test_builds_with_auto_tier(self, service: CitationService) -> None:
        c = service.build_citation(
            text="Carl Cox interview",
            source_name="Resident Advisor",
            source_url="https://residentadvisor.net/features/123",
        )
        assert c.tier == 2
        assert c.text == "Carl Cox interview"
        assert c.source_name == "Resident Advisor"

    def test_builds_with_date_parsing(self, service: CitationService) -> None:
        c = service.build_citation(
            text="Fact",
            source_name="Source",
            source_date="2024-06-15",
        )
        assert c.source_date == date(2024, 6, 15)

    def test_invalid_date_sets_none(self, service: CitationService) -> None:
        c = service.build_citation(
            text="Fact",
            source_name="Source",
            source_date="not-a-date",
        )
        assert c.source_date is None

    def test_builds_with_page_number(self, service: CitationService) -> None:
        c = service.build_citation(
            text="Claim",
            source_name="Book",
            source_type="book",
            page_number="42",
        )
        assert c.page_number == "42"
        assert c.tier == 1


# ======================================================================
# verify_citation (async — mock httpx)
# ======================================================================


class TestVerifyCitation:
    """Tests for citation URL verification with mocked HTTP calls."""

    @pytest.mark.asyncio
    async def test_no_url_returns_false(self, service: CitationService) -> None:
        c = Citation(text="claim", source_type="book", source_name="author", tier=1)
        result_citation, accessible = await service.verify_citation(c)
        assert accessible is False

    @pytest.mark.asyncio
    async def test_url_accessible_returns_true(self, service: CitationService) -> None:
        from unittest.mock import AsyncMock, patch

        c = Citation(
            text="claim",
            source_type="press",
            source_name="RA",
            source_url="https://ra.co/features/123",
            tier=2,
        )

        with patch.object(service, "_head_check", new_callable=AsyncMock, return_value=True):
            _, accessible = await service.verify_citation(c)
            assert accessible is True

    @pytest.mark.asyncio
    async def test_url_down_wayback_available(self, service: CitationService) -> None:
        from unittest.mock import AsyncMock, patch

        c = Citation(
            text="claim",
            source_type="press",
            source_name="Old Blog",
            source_url="https://oldblog.com/post",
            tier=5,
        )

        with (
            patch.object(service, "_head_check", new_callable=AsyncMock, return_value=False),
            patch.object(service, "_check_wayback", new_callable=AsyncMock, return_value=True),
        ):
            _, accessible = await service.verify_citation(c)
            assert accessible is True

    @pytest.mark.asyncio
    async def test_url_down_no_wayback(self, service: CitationService) -> None:
        from unittest.mock import AsyncMock, patch

        c = Citation(
            text="claim",
            source_type="web",
            source_name="Dead Site",
            source_url="https://dead-site.com",
            tier=5,
        )

        with (
            patch.object(service, "_head_check", new_callable=AsyncMock, return_value=False),
            patch.object(service, "_check_wayback", new_callable=AsyncMock, return_value=False),
        ):
            _, accessible = await service.verify_citation(c)
            assert accessible is False


# ======================================================================
# verify_batch (async — mock httpx)
# ======================================================================


class TestVerifyBatch:
    """Tests for batch citation verification."""

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self, service: CitationService) -> None:
        result = await service.verify_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_mixed_results(self, service: CitationService) -> None:
        from unittest.mock import AsyncMock, patch

        c1 = Citation(
            text="accessible",
            source_type="press",
            source_name="RA",
            source_url="https://ra.co/1",
            tier=2,
        )
        c2 = Citation(
            text="dead link",
            source_type="web",
            source_name="Dead",
            source_url="https://dead.com",
            tier=5,
        )
        c3 = Citation(
            text="no url",
            source_type="book",
            source_name="Author",
            tier=1,
        )

        async def mock_verify(c: Citation) -> tuple[Citation, bool]:
            if c.source_url and "ra.co" in c.source_url:
                return (c, True)
            if c.source_url is None:
                return (c, False)
            return (c, False)

        with patch.object(service, "verify_citation", side_effect=mock_verify):
            results = await service.verify_batch([c1, c2, c3])

        assert len(results) == 3
        assert results[0][1] is True   # ra.co accessible
        assert results[1][1] is False  # dead link
        assert results[2][1] is False  # no url

    @pytest.mark.asyncio
    async def test_exception_handling_in_batch(self, service: CitationService) -> None:
        from unittest.mock import AsyncMock, patch

        c1 = Citation(
            text="boom",
            source_type="web",
            source_name="Err",
            source_url="https://err.com",
            tier=5,
        )

        async def mock_verify(c: Citation) -> tuple[Citation, bool]:
            raise ConnectionError("Network error")

        with patch.object(service, "verify_citation", side_effect=mock_verify):
            results = await service.verify_batch([c1])

        assert len(results) == 1
        assert results[0][1] is False  # exception → inaccessible


# ======================================================================
# _head_check and _check_wayback (low-level httpx mocking)
# ======================================================================


class TestHeadCheck:
    @pytest.mark.asyncio
    async def test_successful_head_check(self, service: CitationService) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.head = AsyncMock(return_value=mock_response)

        with patch("src.services.citation_service.httpx.AsyncClient", return_value=mock_client):
            result = await service._head_check("https://example.com")
            assert result is True

    @pytest.mark.asyncio
    async def test_failed_head_check(self, service: CitationService) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.head = AsyncMock(return_value=mock_response)

        with patch("src.services.citation_service.httpx.AsyncClient", return_value=mock_client):
            result = await service._head_check("https://example.com")
            assert result is False

    @pytest.mark.asyncio
    async def test_head_check_exception(self, service: CitationService) -> None:
        import httpx
        from unittest.mock import AsyncMock, patch

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.head = AsyncMock(side_effect=httpx.ConnectTimeout("timeout"))

        with patch("src.services.citation_service.httpx.AsyncClient", return_value=mock_client):
            result = await service._head_check("https://example.com")
            assert result is False


class TestCheckWayback:
    @pytest.mark.asyncio
    async def test_wayback_available(self, service: CitationService) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "archived_snapshots": {
                "closest": {
                    "available": True,
                    "url": "https://web.archive.org/web/20240101/https://example.com",
                }
            }
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("src.services.citation_service.httpx.AsyncClient", return_value=mock_client):
            result = await service._check_wayback("https://example.com")
            assert result is True

    @pytest.mark.asyncio
    async def test_wayback_not_available(self, service: CitationService) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"archived_snapshots": {}}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("src.services.citation_service.httpx.AsyncClient", return_value=mock_client):
            result = await service._check_wayback("https://example.com")
            assert result is False

    @pytest.mark.asyncio
    async def test_wayback_api_error(self, service: CitationService) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("src.services.citation_service.httpx.AsyncClient", return_value=mock_client):
            result = await service._check_wayback("https://example.com")
            assert result is False

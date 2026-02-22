"""Unit tests for search and article provider adapters."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


# ======================================================================
# DuckDuckGo Search Provider
# ======================================================================


class TestDuckDuckGoSearchProvider:
    def test_get_provider_name(self) -> None:
        from src.providers.search.duckduckgo_provider import DuckDuckGoSearchProvider
        provider = DuckDuckGoSearchProvider()
        assert provider.get_provider_name() == "duckduckgo"

    def test_is_available(self) -> None:
        from src.providers.search.duckduckgo_provider import DuckDuckGoSearchProvider
        provider = DuckDuckGoSearchProvider()
        assert provider.is_available() is True

    @pytest.mark.asyncio
    async def test_search_success(self) -> None:
        from src.providers.search.duckduckgo_provider import DuckDuckGoSearchProvider

        mock_results = [
            {
                "title": "Carl Cox Wikipedia",
                "href": "https://en.wikipedia.org/wiki/Carl_Cox",
                "body": "Carl Cox is a British DJ...",
            },
            {
                "title": "Carl Cox at RA",
                "href": "https://ra.co/dj/carlcox",
                "body": "Carl Cox profile on Resident Advisor",
            },
        ]

        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text = MagicMock(return_value=mock_results)

        with patch("src.providers.search.duckduckgo_provider.DDGS", return_value=mock_ddgs):
            provider = DuckDuckGoSearchProvider()
            results = await provider.search("Carl Cox DJ techno")

        assert len(results) >= 1
        assert results[0].title == "Carl Cox Wikipedia"

    @pytest.mark.asyncio
    async def test_search_empty_results(self) -> None:
        from src.providers.search.duckduckgo_provider import DuckDuckGoSearchProvider

        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text = MagicMock(return_value=[])

        with patch("src.providers.search.duckduckgo_provider.DDGS", return_value=mock_ddgs):
            provider = DuckDuckGoSearchProvider()
            results = await provider.search("xyznonexistent")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_exception(self) -> None:
        from src.providers.search.duckduckgo_provider import DuckDuckGoSearchProvider

        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text = MagicMock(side_effect=Exception("Rate limited"))

        with patch("src.providers.search.duckduckgo_provider.DDGS", return_value=mock_ddgs):
            provider = DuckDuckGoSearchProvider()
            results = await provider.search("test query")

        assert results == []


# ======================================================================
# Web Scraper Provider
# ======================================================================


class TestWebScraperProvider:
    def test_get_provider_name(self) -> None:
        from src.providers.article.web_scraper_provider import WebScraperProvider
        mock_client = AsyncMock()
        provider = WebScraperProvider(http_client=mock_client)
        assert provider.get_provider_name() == "web_scraper"

    def test_is_available(self) -> None:
        from src.providers.article.web_scraper_provider import WebScraperProvider
        mock_client = AsyncMock()
        provider = WebScraperProvider(http_client=mock_client)
        assert provider.is_available() is True

    @pytest.mark.asyncio
    async def test_extract_content_success(self) -> None:
        from src.providers.article.web_scraper_provider import WebScraperProvider

        mock_response = MagicMock()
        mock_response.text = "<html><body><p>Article about Carl Cox</p></body></html>"
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch(
            "src.providers.article.web_scraper_provider.trafilatura.extract",
            side_effect=[
                "Article about Carl Cox at Tresor Berlin in 1997.",
                '{"title": "Carl Cox at Tresor", "author": "RA Staff", "date": "1997-05-15"}',
            ],
        ):
            provider = WebScraperProvider(http_client=mock_client)
            result = await provider.extract_content("https://ra.co/features/123")

        assert result is not None
        assert "Carl Cox" in result.text

    @pytest.mark.asyncio
    async def test_extract_content_no_text(self) -> None:
        from src.providers.article.web_scraper_provider import WebScraperProvider

        mock_response = MagicMock()
        mock_response.text = "<html></html>"
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch(
            "src.providers.article.web_scraper_provider.trafilatura.extract",
            return_value=None,
        ):
            provider = WebScraperProvider(http_client=mock_client)
            result = await provider.extract_content("https://example.com/empty")

        assert result is None

    @pytest.mark.asyncio
    async def test_extract_content_http_error(self) -> None:
        from src.providers.article.web_scraper_provider import WebScraperProvider
        from src.utils.errors import ResearchError
        import httpx

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timed out"))

        provider = WebScraperProvider(http_client=mock_client)
        with pytest.raises(ResearchError):
            await provider.extract_content("https://slow-site.com")

    @pytest.mark.asyncio
    async def test_check_availability_accessible(self) -> None:
        from src.providers.article.web_scraper_provider import WebScraperProvider

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.head = AsyncMock(return_value=mock_response)

        provider = WebScraperProvider(http_client=mock_client)
        result = await provider.check_availability("https://ra.co")

        assert result is True

    @pytest.mark.asyncio
    async def test_check_availability_unreachable(self) -> None:
        from src.providers.article.web_scraper_provider import WebScraperProvider
        import httpx

        mock_client = AsyncMock()
        mock_client.head = AsyncMock(side_effect=httpx.ConnectError("refused"))

        provider = WebScraperProvider(http_client=mock_client)
        result = await provider.check_availability("https://dead-site.com")

        assert result is False


# ======================================================================
# Wayback Machine Provider
# ======================================================================


class TestWaybackProvider:
    """Tests for the WaybackProvider (Internet Archive Wayback Machine)."""

    def test_get_provider_name(self) -> None:
        from src.providers.article.wayback_provider import WaybackProvider

        mock_client = AsyncMock()
        provider = WaybackProvider(http_client=mock_client)
        assert provider.get_provider_name() == "wayback"

    def test_is_available(self) -> None:
        from src.providers.article.wayback_provider import WaybackProvider

        mock_client = AsyncMock()
        provider = WaybackProvider(http_client=mock_client)
        assert provider.is_available() is True

    @pytest.mark.asyncio
    async def test_extract_content_success(self) -> None:
        """Test successful extraction: API returns snapshot, trafilatura extracts text."""
        from src.providers.article.wayback_provider import WaybackProvider

        # Mock the Wayback availability API response
        api_response = MagicMock()
        api_response.status_code = 200
        api_response.raise_for_status = MagicMock()
        api_response.json.return_value = {
            "archived_snapshots": {
                "closest": {
                    "url": "https://web.archive.org/web/20000101/https://example.com/article",
                    "available": True,
                    "status": "200",
                    "timestamp": "20000101120000",
                }
            }
        }

        # Mock the snapshot page response
        snapshot_response = MagicMock()
        snapshot_response.status_code = 200
        snapshot_response.raise_for_status = MagicMock()
        snapshot_response.text = "<html><body><p>Archived article about Tresor Berlin</p></body></html>"

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[api_response, snapshot_response])

        with patch(
            "src.providers.article.wayback_provider.trafilatura.extract",
            side_effect=[
                "Archived article about Tresor Berlin and the techno scene in 1997.",
                '{"title": "Tresor Berlin History", "author": "Archive Staff", "date": "1998-06-15"}',
            ],
        ):
            provider = WaybackProvider(http_client=mock_client)
            result = await provider.extract_content("https://example.com/article")

        assert result is not None
        assert "Tresor Berlin" in result.text
        assert result.title == "Tresor Berlin History"
        assert result.author == "Archive Staff"
        assert result.date == date(1998, 6, 15)
        assert "web.archive.org" in result.url

    @pytest.mark.asyncio
    async def test_extract_content_no_snapshot(self) -> None:
        """Test when Wayback API has no snapshot — returns None."""
        from src.providers.article.wayback_provider import WaybackProvider

        api_response = MagicMock()
        api_response.status_code = 200
        api_response.raise_for_status = MagicMock()
        api_response.json.return_value = {"archived_snapshots": {}}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=api_response)

        provider = WaybackProvider(http_client=mock_client)
        result = await provider.extract_content("https://example.com/gone")

        assert result is None

    @pytest.mark.asyncio
    async def test_extract_content_trafilatura_empty(self) -> None:
        """Test when trafilatura fails to extract text — returns None."""
        from src.providers.article.wayback_provider import WaybackProvider

        api_response = MagicMock()
        api_response.status_code = 200
        api_response.raise_for_status = MagicMock()
        api_response.json.return_value = {
            "archived_snapshots": {
                "closest": {
                    "url": "https://web.archive.org/web/20000101/https://example.com",
                    "available": True,
                }
            }
        }

        snapshot_response = MagicMock()
        snapshot_response.status_code = 200
        snapshot_response.raise_for_status = MagicMock()
        snapshot_response.text = "<html><body></body></html>"

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[api_response, snapshot_response])

        with patch(
            "src.providers.article.wayback_provider.trafilatura.extract",
            return_value=None,
        ):
            provider = WaybackProvider(http_client=mock_client)
            result = await provider.extract_content("https://example.com/empty")

        assert result is None

    @pytest.mark.asyncio
    async def test_extract_content_http_timeout(self) -> None:
        """Test that a timeout fetching the snapshot raises ResearchError."""
        from src.providers.article.wayback_provider import WaybackProvider
        from src.utils.errors import ResearchError

        api_response = MagicMock()
        api_response.status_code = 200
        api_response.raise_for_status = MagicMock()
        api_response.json.return_value = {
            "archived_snapshots": {
                "closest": {
                    "url": "https://web.archive.org/web/20000101/https://slow.com",
                    "available": True,
                }
            }
        }

        # First call: availability API succeeds; second call: snapshot times out
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=[api_response, httpx.TimeoutException("Timed out")]
        )

        provider = WaybackProvider(http_client=mock_client)
        with pytest.raises(ResearchError) as exc_info:
            await provider.extract_content("https://slow.com")

        assert "Timeout" in str(exc_info.value)
        assert exc_info.value.provider_name == "wayback"

    @pytest.mark.asyncio
    async def test_extract_content_http_status_error(self) -> None:
        """Test that a 404 from the snapshot raises ResearchError."""
        from src.providers.article.wayback_provider import WaybackProvider
        from src.utils.errors import ResearchError

        api_response = MagicMock()
        api_response.status_code = 200
        api_response.raise_for_status = MagicMock()
        api_response.json.return_value = {
            "archived_snapshots": {
                "closest": {
                    "url": "https://web.archive.org/web/20000101/https://gone.com",
                    "available": True,
                }
            }
        }

        error_response = MagicMock()
        error_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=[
                api_response,
                httpx.HTTPStatusError(
                    "404 Not Found",
                    request=MagicMock(),
                    response=error_response,
                ),
            ]
        )

        provider = WaybackProvider(http_client=mock_client)
        with pytest.raises(ResearchError) as exc_info:
            await provider.extract_content("https://gone.com")

        assert "HTTP 404" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_extract_content_generic_http_error(self) -> None:
        """Test that a generic HTTP error from the snapshot raises ResearchError."""
        from src.providers.article.wayback_provider import WaybackProvider
        from src.utils.errors import ResearchError

        api_response = MagicMock()
        api_response.status_code = 200
        api_response.raise_for_status = MagicMock()
        api_response.json.return_value = {
            "archived_snapshots": {
                "closest": {
                    "url": "https://web.archive.org/web/20000101/https://broken.com",
                    "available": True,
                }
            }
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=[api_response, httpx.HTTPError("Connection reset")]
        )

        provider = WaybackProvider(http_client=mock_client)
        with pytest.raises(ResearchError) as exc_info:
            await provider.extract_content("https://broken.com")

        assert "HTTP error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_extract_content_metadata_without_date(self) -> None:
        """Test extraction when metadata JSON has no date field."""
        from src.providers.article.wayback_provider import WaybackProvider

        api_response = MagicMock()
        api_response.status_code = 200
        api_response.raise_for_status = MagicMock()
        api_response.json.return_value = {
            "archived_snapshots": {
                "closest": {
                    "url": "https://web.archive.org/web/20050101/https://example.com",
                    "available": True,
                }
            }
        }

        snapshot_response = MagicMock()
        snapshot_response.status_code = 200
        snapshot_response.raise_for_status = MagicMock()
        snapshot_response.text = "<html><body>Some content</body></html>"

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[api_response, snapshot_response])

        with patch(
            "src.providers.article.wayback_provider.trafilatura.extract",
            side_effect=[
                "Some content about underground techno.",
                '{"title": "No Date Article", "author": null}',
            ],
        ):
            provider = WaybackProvider(http_client=mock_client)
            result = await provider.extract_content("https://example.com")

        assert result is not None
        assert result.title == "No Date Article"
        assert result.author is None
        assert result.date is None

    @pytest.mark.asyncio
    async def test_check_availability_true(self) -> None:
        """Test check_availability returns True when a snapshot exists."""
        from src.providers.article.wayback_provider import WaybackProvider

        api_response = MagicMock()
        api_response.status_code = 200
        api_response.raise_for_status = MagicMock()
        api_response.json.return_value = {
            "archived_snapshots": {
                "closest": {
                    "url": "https://web.archive.org/web/20000101/https://example.com",
                    "available": True,
                }
            }
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=api_response)

        provider = WaybackProvider(http_client=mock_client)
        result = await provider.check_availability("https://example.com")

        assert result is True

    @pytest.mark.asyncio
    async def test_check_availability_false(self) -> None:
        """Test check_availability returns False when no snapshot exists."""
        from src.providers.article.wayback_provider import WaybackProvider

        api_response = MagicMock()
        api_response.status_code = 200
        api_response.raise_for_status = MagicMock()
        api_response.json.return_value = {"archived_snapshots": {}}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=api_response)

        provider = WaybackProvider(http_client=mock_client)
        result = await provider.check_availability("https://example.com/gone")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_snapshot_url_api_error(self) -> None:
        """Test that _get_snapshot_url returns None on API error."""
        from src.providers.article.wayback_provider import WaybackProvider

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.HTTPError("API down"))

        provider = WaybackProvider(http_client=mock_client)
        result = await provider.check_availability("https://example.com")

        assert result is False

    @pytest.mark.asyncio
    async def test_extract_content_metadata_parse_failure(self) -> None:
        """Test extraction continues when metadata JSON is malformed."""
        from src.providers.article.wayback_provider import WaybackProvider

        api_response = MagicMock()
        api_response.status_code = 200
        api_response.raise_for_status = MagicMock()
        api_response.json.return_value = {
            "archived_snapshots": {
                "closest": {
                    "url": "https://web.archive.org/web/20050101/https://example.com",
                    "available": True,
                }
            }
        }

        snapshot_response = MagicMock()
        snapshot_response.status_code = 200
        snapshot_response.raise_for_status = MagicMock()
        snapshot_response.text = "<html><body>Content here</body></html>"

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[api_response, snapshot_response])

        with patch(
            "src.providers.article.wayback_provider.trafilatura.extract",
            side_effect=[
                "Valid article text.",
                "NOT VALID JSON {{{",  # Malformed metadata
            ],
        ):
            provider = WaybackProvider(http_client=mock_client)
            result = await provider.extract_content("https://example.com")

        assert result is not None
        assert result.text == "Valid article text."
        # Title defaults to empty, date/author to None when parse fails
        assert result.title == ""
        assert result.author is None
        assert result.date is None

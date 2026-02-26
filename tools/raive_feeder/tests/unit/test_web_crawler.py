"""Unit tests for WebCrawler service.

Tests URL crawling, content extraction, and LLM-guided relevance scoring
with mocked HTTP responses and LLM providers.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tools.raive_feeder.services.web_crawler import WebCrawler


@pytest.fixture
def crawler_no_llm():
    """WebCrawler without LLM (blind crawl mode)."""
    return WebCrawler(llm_provider=None, rate_limit_seconds=0)


@pytest.fixture
def mock_llm():
    """Mock LLM provider that returns relevance scores."""
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value="7")
    return llm


@pytest.fixture
def crawler_with_llm(mock_llm):
    """WebCrawler with LLM for guided crawling."""
    return WebCrawler(llm_provider=mock_llm, rate_limit_seconds=0)


class TestBlindCrawl:
    """Tests for crawling without NL query (blind mode)."""

    @pytest.mark.asyncio
    async def test_single_page_crawl(self, crawler_no_llm):
        """Depth 0 crawl should fetch only the seed page."""
        html = "<html><head><title>Test Page</title></head><body>Content here</body></html>"
        mock_page = {
            "url": "https://example.com/page",
            "title": "Test Page",
            "text": "Content here",
            "links": [],
        }

        with patch.object(crawler_no_llm, "_fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_page
            results = await crawler_no_llm.crawl(
                seed_url="https://example.com/page",
                max_depth=0,
                max_pages=1,
            )

        assert len(results) == 1
        assert results[0]["url"] == "https://example.com/page"
        assert results[0]["relevance_score"] is None

    @pytest.mark.asyncio
    async def test_max_pages_limit(self, crawler_no_llm):
        """Crawler should stop after reaching max_pages."""
        async def fake_fetch(client, url):
            return {
                "url": url,
                "title": f"Page {url}",
                "text": "Content",
                "links": ["/a", "/b", "/c"],
            }

        with patch.object(crawler_no_llm, "_fetch_page", side_effect=fake_fetch):
            results = await crawler_no_llm.crawl(
                seed_url="https://example.com/",
                max_depth=1,
                max_pages=2,
            )

        assert len(results) <= 2


class TestLLMGuidedCrawl:
    """Tests for LLM-guided relevance scoring."""

    @pytest.mark.asyncio
    async def test_relevance_scoring(self, crawler_with_llm):
        """Pages should receive LLM relevance scores when NL query is provided."""
        mock_page = {
            "url": "https://example.com/page",
            "title": "Berlin Techno",
            "text": "Berlin techno clubs in the 1990s were legendary.",
            "links": [],
        }

        with patch.object(crawler_with_llm, "_fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_page
            results = await crawler_with_llm.crawl(
                seed_url="https://example.com/page",
                max_depth=0,
                max_pages=1,
                nl_query="Berlin techno clubs",
            )

        assert len(results) == 1
        assert results[0]["relevance_score"] is not None
        assert 0 <= results[0]["relevance_score"] <= 10


class TestFetchPage:
    """Tests for individual page fetching."""

    @pytest.mark.asyncio
    async def test_fetch_returns_none_on_404(self, crawler_no_llm):
        """Fetch should return None for non-200 responses."""
        mock_response = AsyncMock()
        mock_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await crawler_no_llm._fetch_page(mock_client, "https://example.com/404")
        assert result is None

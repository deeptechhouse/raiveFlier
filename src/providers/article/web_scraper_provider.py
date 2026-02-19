"""Web scraper article provider using httpx and trafilatura.

Extracts clean article text from web pages by fetching HTML via httpx
and parsing with trafilatura's content extraction engine.
"""

from __future__ import annotations

from datetime import date, datetime

import httpx
import structlog
import trafilatura

from src.interfaces.article_provider import ArticleContent, IArticleProvider
from src.utils.errors import ResearchError

logger = structlog.get_logger(logger_name=__name__)

_DEFAULT_TIMEOUT = 10.0
_DEFAULT_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (compatible; raiveFlier/0.1; " "+https://github.com/raiveFlier)"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


class WebScraperProvider(IArticleProvider):
    """Article extraction backed by httpx + trafilatura.

    Fetches raw HTML from a URL and uses trafilatura to extract the
    main article content, stripping navigation, ads, and boilerplate.
    """

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(_DEFAULT_TIMEOUT),
            headers=_DEFAULT_HEADERS,
            follow_redirects=True,
        )

    # ------------------------------------------------------------------
    # IArticleProvider implementation
    # ------------------------------------------------------------------

    async def extract_content(self, url: str) -> ArticleContent | None:
        """Fetch *url* and extract readable article text via trafilatura."""
        try:
            response = await self._client.get(url)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ResearchError(
                message=f"Timeout fetching {url}: {exc}",
                provider_name=self.get_provider_name(),
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ResearchError(
                message=f"HTTP {exc.response.status_code} for {url}",
                provider_name=self.get_provider_name(),
            ) from exc
        except httpx.HTTPError as exc:
            raise ResearchError(
                message=f"HTTP error fetching {url}: {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

        html = response.text
        text = trafilatura.extract(html, include_comments=False, include_tables=True)
        if not text:
            logger.warning("trafilatura_extraction_empty", url=url)
            return None

        metadata = trafilatura.extract(
            html,
            include_comments=False,
            output_format="json",
            with_metadata=True,
        )
        title = ""
        author: str | None = None
        pub_date: date | None = None

        if metadata:
            import json

            try:
                meta_dict = json.loads(metadata)
                title = meta_dict.get("title", "")
                author = meta_dict.get("author") or None
                raw_date = meta_dict.get("date")
                if raw_date:
                    pub_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
            except (json.JSONDecodeError, ValueError, KeyError):
                logger.debug("metadata_parse_failed", url=url)

        logger.info(
            "article_extracted",
            url=url,
            title=title,
            text_length=len(text),
        )

        return ArticleContent(
            title=title,
            text=text,
            author=author,
            date=pub_date,
            url=url,
        )

    async def check_availability(self, url: str) -> bool:
        """HEAD request to *url*; return ``True`` if status is 200."""
        try:
            response = await self._client.head(url)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def is_available(self) -> bool:
        """Always available — no external credentials required."""
        return True

    def get_provider_name(self) -> str:
        return "web_scraper"

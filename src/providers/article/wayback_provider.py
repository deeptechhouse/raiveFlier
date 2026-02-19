"""Wayback Machine article provider.

Queries the Internet Archive's Wayback Machine availability API to find
archived snapshots of URLs, then extracts content from the archived version.
Critical for recovering articles about underground music history that may
have been taken down from their original locations.
"""

from __future__ import annotations

from datetime import date, datetime

import httpx
import structlog
import trafilatura

from src.interfaces.article_provider import ArticleContent, IArticleProvider
from src.utils.errors import ResearchError

logger = structlog.get_logger(logger_name=__name__)

_WAYBACK_API_URL = "https://archive.org/wayback/available"
_DEFAULT_TIMEOUT = 15.0
_DEFAULT_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (compatible; raiveFlier/0.1; " "+https://github.com/raiveFlier)"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


class WaybackProvider(IArticleProvider):
    """Article extraction from Internet Archive Wayback Machine snapshots.

    First queries the Wayback availability API to locate an archived
    snapshot.  If one exists, fetches the archived HTML and extracts
    readable content via trafilatura.
    """

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(_DEFAULT_TIMEOUT),
            headers=_DEFAULT_HEADERS,
            follow_redirects=True,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_snapshot_url(self, url: str) -> str | None:
        """Query the Wayback availability API for a snapshot of *url*."""
        try:
            response = await self._client.get(
                _WAYBACK_API_URL,
                params={"url": url},
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("wayback_api_error", url=url, error=str(exc))
            return None

        snapshot = data.get("archived_snapshots", {}).get("closest", {})
        if snapshot and snapshot.get("available"):
            return snapshot.get("url")
        return None

    # ------------------------------------------------------------------
    # IArticleProvider implementation
    # ------------------------------------------------------------------

    async def extract_content(self, url: str) -> ArticleContent | None:
        """Find an archived snapshot and extract its article text."""
        snapshot_url = await self._get_snapshot_url(url)
        if not snapshot_url:
            logger.info("wayback_no_snapshot", url=url)
            return None

        try:
            response = await self._client.get(snapshot_url)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ResearchError(
                message=f"Timeout fetching Wayback snapshot for {url}: {exc}",
                provider_name=self.get_provider_name(),
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ResearchError(
                message=(
                    f"HTTP {exc.response.status_code} fetching " f"Wayback snapshot for {url}"
                ),
                provider_name=self.get_provider_name(),
            ) from exc
        except httpx.HTTPError as exc:
            raise ResearchError(
                message=f"HTTP error fetching Wayback snapshot for {url}: {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

        html = response.text
        text = trafilatura.extract(html, include_comments=False, include_tables=True)
        if not text:
            logger.warning("wayback_extraction_empty", url=url, snapshot=snapshot_url)
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
                logger.debug("wayback_metadata_parse_failed", url=url)

        logger.info(
            "wayback_article_extracted",
            original_url=url,
            snapshot_url=snapshot_url,
            title=title,
            text_length=len(text),
        )

        return ArticleContent(
            title=title,
            text=text,
            author=author,
            date=pub_date,
            url=snapshot_url,
        )

    async def check_availability(self, url: str) -> bool:
        """Return ``True`` if a Wayback Machine snapshot exists for *url*."""
        return (await self._get_snapshot_url(url)) is not None

    def is_available(self) -> bool:
        """Always available — the Wayback Machine API is free and public."""
        return True

    def get_provider_name(self) -> str:
        return "wayback"

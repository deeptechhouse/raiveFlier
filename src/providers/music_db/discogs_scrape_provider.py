"""Discogs web-scraping fallback provider.

Implements IMusicDatabaseProvider by scraping the Discogs website directly.
Used as a fallback when API consumer keys are not available.  Includes
respectful rate limiting (2-second delays) and a proper User-Agent header.
HTTP errors (403, 429, 5xx) are handled gracefully — the provider returns
empty results rather than raising.
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import date
from typing import Any
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup
from rapidfuzz import fuzz

from src.interfaces.music_db_provider import ArtistSearchResult, IMusicDatabaseProvider
from src.models.entities import Label, Release
from src.utils.logging import get_logger
from src.utils.text_normalizer import normalize_artist_name

_BASE_URL = "https://www.discogs.com"
_USER_AGENT = "raiveFlier/0.1.0 (+https://github.com/raiveFlier)"
_SCRAPE_DELAY = 2.0  # seconds between requests
_ARTIST_URL_RE = re.compile(r"/artist/(\d+)")
_RELEASE_URL_RE = re.compile(r"/release/(\d+)")
_MASTER_URL_RE = re.compile(r"/master/(\d+)")
_MAX_SEARCH_RESULTS = 10


class DiscogsScrapeProvider(IMusicDatabaseProvider):
    """Fallback music-database provider that scrapes Discogs web pages.

    No API keys are required.  Requests are throttled to 2-second intervals
    to be respectful of Discogs servers.  The ``httpx.AsyncClient`` is
    injected via the constructor for testability.
    """

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._http = http_client
        self._last_request_time: float = 0.0
        self._logger = get_logger(__name__)

    # -- Private helpers -------------------------------------------------------

    async def _throttle(self) -> None:
        """Enforce minimum delay between scrape requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if self._last_request_time > 0 and elapsed < _SCRAPE_DELAY:
            await asyncio.sleep(_SCRAPE_DELAY - elapsed)
        self._last_request_time = time.monotonic()

    async def _fetch_page(self, url: str) -> BeautifulSoup | None:
        """Fetch *url* and return parsed HTML, or ``None`` on error."""
        await self._throttle()
        headers = {"User-Agent": _USER_AGENT}
        try:
            response = await self._http.get(url, headers=headers, follow_redirects=True)
            if response.status_code in (403, 429, 500, 502, 503):
                self._logger.warning(
                    "discogs_scrape_http_error",
                    url=url,
                    status=response.status_code,
                )
                return None
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except httpx.HTTPError as exc:
            self._logger.warning("discogs_scrape_request_failed", url=url, error=str(exc))
            return None

    @staticmethod
    def _compute_confidence(query: str, result_name: str) -> float:
        """Compute fuzzy-match confidence between *query* and *result_name*."""
        normalized_query = normalize_artist_name(query)
        normalized_result = normalize_artist_name(result_name)
        return fuzz.token_sort_ratio(normalized_query, normalized_result) / 100.0

    @staticmethod
    def _extract_artist_id(href: str) -> str | None:
        """Extract numeric artist ID from a Discogs artist URL path."""
        match = _ARTIST_URL_RE.search(href)
        return match.group(1) if match else None

    @staticmethod
    def _extract_release_id(href: str) -> str | None:
        """Extract numeric release ID from a Discogs release URL path."""
        match = _RELEASE_URL_RE.search(href)
        return match.group(1) if match else None

    @staticmethod
    def _safe_int(value: str) -> int | None:
        """Parse an integer from *value*, returning ``None`` on failure."""
        try:
            return int(value.strip())
        except (ValueError, AttributeError):
            return None

    def _parse_search_results(self, soup: BeautifulSoup) -> list[dict[str, str]]:
        """Parse artist search results from a Discogs search page."""
        results: list[dict[str, str]] = []
        seen_ids: set[str] = set()

        for link in soup.find_all("a", href=_ARTIST_URL_RE):
            href = link.get("href", "")
            artist_id = self._extract_artist_id(href)
            if not artist_id or artist_id in seen_ids:
                continue

            name = link.get_text(strip=True)
            if not name or len(name) < 2:
                continue

            seen_ids.add(artist_id)
            results.append(
                {
                    "id": artist_id,
                    "name": name,
                    "url": f"{_BASE_URL}{href}" if href.startswith("/") else href,
                }
            )
            if len(results) >= _MAX_SEARCH_RESULTS:
                break

        return results

    def _parse_releases_table(self, soup: BeautifulSoup) -> list[dict[str, Any]]:
        """Parse releases from an artist discography page."""
        releases: list[dict[str, Any]] = []

        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue

            release_data: dict[str, Any] = {}

            # Find the first release or master link in the row
            for link in row.find_all("a", href=True):
                href = link["href"]
                if _RELEASE_URL_RE.search(href) or _MASTER_URL_RE.search(href):
                    release_data["title"] = link.get_text(strip=True)
                    release_data["url"] = f"{_BASE_URL}{href}" if href.startswith("/") else href
                    release_id = self._extract_release_id(href)
                    if release_id:
                        release_data["id"] = release_id
                    break

            if not release_data.get("title"):
                continue

            # Extract label, catalog#, year, format from cells by CSS class
            for cell in cells:
                cell_class = " ".join(cell.get("class", []))
                text = cell.get_text(strip=True)
                if "label" in cell_class:
                    release_data.setdefault("label", text)
                elif "catno" in cell_class:
                    release_data.setdefault("catalog_number", text)
                elif "year" in cell_class:
                    release_data.setdefault("year", text)
                elif "format" in cell_class:
                    release_data.setdefault("format", text)

            releases.append(release_data)

        return releases

    # -- IMusicDatabaseProvider implementation ---------------------------------

    async def search_artist(self, name: str) -> list[ArtistSearchResult]:
        """Scrape the Discogs search page for artists matching *name*."""
        encoded_name = quote_plus(name)
        url = f"{_BASE_URL}/search/?q={encoded_name}&type=artist"
        soup = await self._fetch_page(url)

        if soup is None:
            self._logger.warning("discogs_scrape_search_no_results", artist=name)
            return []

        raw_results = self._parse_search_results(soup)
        search_results: list[ArtistSearchResult] = []

        for item in raw_results:
            confidence = self._compute_confidence(name, item["name"])
            search_results.append(
                ArtistSearchResult(
                    id=item["id"],
                    name=item["name"],
                    confidence=confidence,
                )
            )

        search_results.sort(key=lambda r: r.confidence, reverse=True)
        self._logger.info(
            "discogs_scrape_search_complete",
            artist=name,
            result_count=len(search_results),
        )
        return search_results

    async def get_artist_releases(
        self, artist_id: str, before_date: date | None = None
    ) -> list[Release]:
        """Scrape the artist's discography page for releases."""
        url = f"{_BASE_URL}/artist/{artist_id}"
        soup = await self._fetch_page(url)

        if soup is None:
            self._logger.warning("discogs_scrape_releases_empty", artist_id=artist_id)
            return []

        raw_releases = self._parse_releases_table(soup)
        releases: list[Release] = []

        for data in raw_releases:
            year = self._safe_int(str(data.get("year", "")))
            if before_date and year and year > before_date.year:
                continue

            release_id = data.get("id", "")
            discogs_url = f"{_BASE_URL}/release/{release_id}" if release_id else data.get("url", "")

            releases.append(
                Release(
                    title=data.get("title", "Unknown"),
                    label=data.get("label", "Unknown"),
                    catalog_number=data.get("catalog_number"),
                    year=year,
                    format=data.get("format"),
                    discogs_url=discogs_url,
                )
            )

        self._logger.info(
            "discogs_scrape_releases_complete",
            artist_id=artist_id,
            release_count=len(releases),
        )
        return releases

    async def get_artist_labels(self, artist_id: str) -> list[Label]:
        """Extract unique labels from the artist's scraped releases."""
        releases = await self.get_artist_releases(artist_id)
        seen: dict[str, Label] = {}
        for release in releases:
            label_name = release.label
            if label_name and label_name != "Unknown" and label_name not in seen:
                seen[label_name] = Label(name=label_name)
        return list(seen.values())

    async def get_release_details(self, release_id: str) -> Release | None:
        """Scrape an individual release page for full details."""
        url = f"{_BASE_URL}/release/{release_id}"
        soup = await self._fetch_page(url)

        if soup is None:
            return None

        # Title from <h1>
        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else "Unknown"

        label_name = "Unknown"
        catalog_number: str | None = None
        year: int | None = None
        format_str: str | None = None
        genres: list[str] = []
        styles: list[str] = []

        # Parse label info from elements with label-related classes
        for div in soup.find_all(attrs={"class": re.compile(r"label", re.IGNORECASE)}):
            label_link = div.find("a")
            if label_link:
                label_name = label_link.get_text(strip=True)
                break

        # Parse catalog number
        catno_el = soup.find(attrs={"class": re.compile(r"catno", re.IGNORECASE)})
        if catno_el:
            catalog_number = catno_el.get_text(strip=True) or None

        # Parse year
        year_el = soup.find(attrs={"class": re.compile(r"year", re.IGNORECASE)})
        if year_el:
            year = self._safe_int(year_el.get_text(strip=True))

        # Parse format
        format_el = soup.find(attrs={"class": re.compile(r"format", re.IGNORECASE)})
        if format_el:
            format_str = format_el.get_text(strip=True) or None

        # Parse genres and styles from their respective sections
        for section_name, target_list in [("genre", genres), ("style", styles)]:
            section = soup.find(attrs={"class": re.compile(section_name, re.IGNORECASE)})
            if section:
                for link in section.find_all("a"):
                    text = link.get_text(strip=True)
                    if text:
                        target_list.append(text)

        return Release(
            title=title,
            label=label_name,
            catalog_number=catalog_number,
            year=year,
            format=format_str,
            discogs_url=url,
            genres=genres,
            styles=styles,
        )

    def get_provider_name(self) -> str:
        """Return ``'discogs_scrape'``."""
        return "discogs_scrape"

    def is_available(self) -> bool:
        """Always ``True`` — no API key required for web scraping."""
        return True

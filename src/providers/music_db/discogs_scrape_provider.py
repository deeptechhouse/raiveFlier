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
_LABEL_URL_RE = re.compile(r"/label/(\d+)")
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
                    # Try to extract label ID from link
                    label_link = cell.find("a", href=_LABEL_URL_RE)
                    if label_link:
                        label_match = _LABEL_URL_RE.search(label_link["href"])
                        if label_match:
                            release_data.setdefault("label_id", label_match.group(1))
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
        """Extract unique labels from the artist's discography page with Discogs IDs."""
        url = f"{_BASE_URL}/artist/{artist_id}"
        soup = await self._fetch_page(url)

        if soup is None:
            return []

        raw_releases = self._parse_releases_table(soup)
        seen: dict[str, Label] = {}
        for data in raw_releases:
            label_name = data.get("label")
            if not label_name or label_name == "Unknown" or label_name in seen:
                continue
            label_id_str = data.get("label_id")
            label_id = int(label_id_str) if label_id_str else None
            discogs_url = f"https://www.discogs.com/label/{label_id}" if label_id else None
            seen[label_name] = Label(
                name=label_name,
                discogs_id=label_id,
                discogs_url=discogs_url,
            )

        self._logger.info(
            "discogs_scrape_labels_complete",
            artist_id=artist_id,
            label_count=len(seen),
            resolved=sum(1 for lb in seen.values() if lb.discogs_id),
        )
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

    async def get_label_releases(
        self, label_id: str, max_results: int = 50
    ) -> list[Release]:
        """Scrape releases from a Discogs label page."""
        url = f"{_BASE_URL}/label/{label_id}"
        soup = await self._fetch_page(url)

        if soup is None:
            self._logger.warning(
                "discogs_scrape_label_releases_failed",
                label_id=label_id,
            )
            return []

        # Extract label name from the page heading
        label_name_el = soup.select_one("h1")
        label_name = label_name_el.get_text(strip=True) if label_name_el else ""

        releases: list[Release] = []
        for row in soup.find_all("tr"):
            if len(releases) >= max_results:
                break

            # Find a release or master link in the row
            title_el = row.find("a", href=lambda h: h and (_RELEASE_URL_RE.search(h) or _MASTER_URL_RE.search(h)))
            if not title_el:
                continue

            title = title_el.get_text(strip=True)
            href = title_el.get("href", "")
            release_id = self._extract_release_id(href)
            discogs_url = f"{_BASE_URL}{href}" if href.startswith("/") else href

            # Extract artist name from an artist link in the same row
            artist_el = row.find("a", href=_ARTIST_URL_RE)
            artist_name = artist_el.get_text(strip=True) if artist_el else ""

            # Extract year from the row
            year: int | None = None
            for cell in row.find_all("td"):
                cell_text = cell.get_text(strip=True)
                year_match = re.search(r"(\d{4})", cell_text)
                if year_match:
                    candidate = int(year_match.group(1))
                    if 1900 <= candidate <= 2100:
                        year = candidate
                        break

            # Extract format from the row
            format_str: str | None = None
            format_cell = row.find(attrs={"class": re.compile(r"format", re.IGNORECASE)})
            if format_cell:
                format_str = format_cell.get_text(strip=True) or None

            display_title = f"{artist_name} - {title}" if artist_name else title

            releases.append(
                Release(
                    title=display_title,
                    artist=artist_name or None,
                    label=label_name,
                    year=year,
                    format=format_str,
                    discogs_url=discogs_url,
                )
            )

        self._logger.info(
            "discogs_scrape_label_releases_complete",
            label_id=label_id,
            release_count=len(releases),
        )
        return releases

    def get_provider_name(self) -> str:
        """Return ``'discogs_scrape'``."""
        return "discogs_scrape"

    def is_available(self) -> bool:
        """Always ``True`` — no API key required for web scraping."""
        return True

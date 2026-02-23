"""Bandcamp web-scraping provider implementing IMusicDatabaseProvider.

Scrapes bandcamp.com for artist search, releases, and label information.
No API key required. Includes respectful rate limiting (2-second delays)
and a proper User-Agent header. HTTP errors are handled gracefully.
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import date
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup
from rapidfuzz import fuzz

from src.interfaces.music_db_provider import ArtistSearchResult, IMusicDatabaseProvider
from src.models.entities import Label, Release
from src.utils.logging import get_logger
from src.utils.text_normalizer import normalize_artist_name

_SEARCH_URL = "https://bandcamp.com/search"
_USER_AGENT = "raiveFlier/0.1.0 (+https://github.com/raiveFlier)"
_SCRAPE_DELAY = 2.0
_MAX_SEARCH_RESULTS = 10
_MAX_RELEASES = 50


class BandcampProvider(IMusicDatabaseProvider):
    """Music database provider that scrapes Bandcamp for artist data.

    No API keys required. Requests are throttled to 2-second intervals.
    The ``httpx.AsyncClient`` is injected for testability.
    """

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._http = http_client
        self._last_request_time: float = 0.0
        self._logger = get_logger(__name__)

    async def _throttle(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if self._last_request_time > 0 and elapsed < _SCRAPE_DELAY:
            await asyncio.sleep(_SCRAPE_DELAY - elapsed)
        self._last_request_time = time.monotonic()

    async def _fetch_page(self, url: str) -> BeautifulSoup | None:
        await self._throttle()
        headers = {"User-Agent": _USER_AGENT}
        try:
            response = await self._http.get(url, headers=headers, follow_redirects=True)
            if response.status_code in (403, 429, 500, 502, 503):
                self._logger.warning("bandcamp_http_error", url=url, status=response.status_code)
                return None
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except httpx.HTTPError as exc:
            self._logger.warning("bandcamp_request_failed", url=url, error=str(exc))
            return None

    @staticmethod
    def _compute_confidence(query: str, result_name: str) -> float:
        normalized_query = normalize_artist_name(query)
        normalized_result = normalize_artist_name(result_name)
        return fuzz.token_sort_ratio(normalized_query, normalized_result) / 100.0

    # -- IMusicDatabaseProvider implementation ---------------------------------

    async def search_artist(self, name: str) -> list[ArtistSearchResult]:
        encoded = quote_plus(name)
        url = f"{_SEARCH_URL}?q={encoded}&item_type=b"
        soup = await self._fetch_page(url)

        if soup is None:
            return []

        results: list[ArtistSearchResult] = []
        seen_urls: set[str] = set()

        for item in soup.select(".searchresult.band"):
            heading = item.select_one(".heading a")
            if not heading:
                continue

            artist_name = heading.get_text(strip=True)
            artist_url = heading.get("href", "").split("?")[0].rstrip("/")

            if not artist_name or not artist_url or artist_url in seen_urls:
                continue

            seen_urls.add(artist_url)
            confidence = self._compute_confidence(name, artist_name)
            results.append(
                ArtistSearchResult(
                    id=artist_url,
                    name=artist_name,
                    confidence=confidence,
                )
            )
            if len(results) >= _MAX_SEARCH_RESULTS:
                break

        results.sort(key=lambda r: r.confidence, reverse=True)
        self._logger.info("bandcamp_search_complete", artist=name, results=len(results))
        return results

    async def get_artist_releases(
        self, artist_id: str, before_date: date | None = None
    ) -> list[Release]:
        music_url = f"{artist_id}/music"
        soup = await self._fetch_page(music_url)

        if soup is None:
            soup = await self._fetch_page(artist_id)
        if soup is None:
            return []

        releases: list[Release] = []

        for item in soup.select("[data-item-id]"):
            link_el = item.select_one("a")
            title_el = item.select_one(".title")

            if not link_el or not title_el:
                continue

            title = title_el.get_text(strip=True)
            release_url = link_el.get("href", "")
            if release_url and not release_url.startswith("http"):
                release_url = f"{artist_id.rstrip('/')}{release_url}"

            year: int | None = None
            date_el = item.select_one(".released")
            if date_el:
                date_text = date_el.get_text(strip=True)
                year_match = re.search(r"(\d{4})", date_text)
                if year_match:
                    year = int(year_match.group(1))

            if before_date and year and year > before_date.year:
                continue

            label_name = self._extract_label_from_page(soup)

            releases.append(
                Release(
                    title=title,
                    label=label_name or "Self-released",
                    year=year,
                    format="Digital",
                    bandcamp_url=release_url if release_url else None,
                )
            )

            if len(releases) >= _MAX_RELEASES:
                break

        # If no structured discography grid, try the music-grid or ol list
        if not releases:
            releases = self._parse_music_grid(soup, artist_id, before_date)

        self._logger.info(
            "bandcamp_releases_complete", artist_id=artist_id, count=len(releases)
        )
        return releases

    def _parse_music_grid(
        self, soup: BeautifulSoup, artist_id: str, before_date: date | None
    ) -> list[Release]:
        releases: list[Release] = []

        for item in soup.select(".music-grid-item, li.music-grid-item"):
            link_el = item.select_one("a")
            title_el = item.select_one(".title")

            if not link_el:
                continue

            title = title_el.get_text(strip=True) if title_el else link_el.get_text(strip=True)
            if not title:
                continue

            release_url = link_el.get("href", "")
            if release_url and not release_url.startswith("http"):
                release_url = f"{artist_id.rstrip('/')}{release_url}"

            releases.append(
                Release(
                    title=title,
                    label="Self-released",
                    format="Digital",
                    bandcamp_url=release_url if release_url else None,
                )
            )

            if len(releases) >= _MAX_RELEASES:
                break

        return releases

    @staticmethod
    def _extract_label_from_page(soup: BeautifulSoup) -> str | None:
        label_el = soup.select_one("#band-name-location .title")
        if label_el:
            parent_link = label_el.find_parent("a")
            if parent_link and "/label/" in (parent_link.get("href", "") or ""):
                return label_el.get_text(strip=True)
        return None

    async def get_artist_labels(self, artist_id: str) -> list[Label]:
        releases = await self.get_artist_releases(artist_id)
        seen: set[str] = set()
        labels: list[Label] = []
        for release in releases:
            label_name = release.label
            if label_name and label_name != "Self-released" and label_name not in seen:
                seen.add(label_name)
                labels.append(Label(name=label_name))
        return labels

    async def get_release_details(self, release_id: str) -> Release | None:
        soup = await self._fetch_page(release_id)
        if soup is None:
            return None

        title_el = soup.select_one("#name-section h2.trackTitle")
        title = title_el.get_text(strip=True) if title_el else "Unknown"

        label_name = "Self-released"
        label_el = soup.select_one("#band-name-location .title")
        if label_el:
            label_name = label_el.get_text(strip=True)

        year: int | None = None
        meta_el = soup.select_one(".tralbumData.tralbum-credits")
        if meta_el:
            year_match = re.search(r"(\d{4})", meta_el.get_text())
            if year_match:
                year = int(year_match.group(1))

        tags: list[str] = []
        for tag_el in soup.select(".tralbumData.tralbum-tags a.tag"):
            tags.append(tag_el.get_text(strip=True))

        return Release(
            title=title,
            label=label_name,
            year=year,
            format="Digital",
            bandcamp_url=release_id,
            genres=tags[:5],
        )

    async def get_label_releases(
        self, label_id: str, max_results: int = 50
    ) -> list[Release]:
        """Not supported by Bandcamp provider — returns empty list."""
        return []

    def get_provider_name(self) -> str:
        return "bandcamp"

    def is_available(self) -> bool:
        return True

"""Beatport provider implementing IMusicDatabaseProvider.

Uses Beatport's internal JSON API (api.beatport.com/v4/) as the primary
data source, with HTML scraping as a fallback. No API key required.
Includes respectful rate limiting (3-second delays).
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

_API_BASE = "https://api.beatport.com/v4"
_WEB_BASE = "https://www.beatport.com"
_USER_AGENT = "raiveFlier/0.1.0 (+https://github.com/raiveFlier)"
_SCRAPE_DELAY = 3.0
_MAX_SEARCH_RESULTS = 10
_MAX_RELEASES = 50


class BeatportProvider(IMusicDatabaseProvider):
    """Music database provider backed by Beatport's internal JSON API.

    Beatport's frontend is a React SPA that calls public JSON endpoints.
    This provider uses those endpoints directly for structured data.
    Falls back to HTML scraping when the JSON API is unavailable.
    No API key required. Requests are throttled to 3-second intervals.
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

    async def _fetch_json(self, url: str) -> dict[str, Any] | None:
        await self._throttle()
        headers = {
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
        }
        try:
            response = await self._http.get(url, headers=headers, follow_redirects=True)
            if response.status_code in (403, 429, 500, 502, 503):
                self._logger.debug("beatport_api_http_error", url=url, status=response.status_code)
                return None
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            self._logger.debug("beatport_api_request_failed", url=url, error=str(exc))
            return None

    async def _fetch_page(self, url: str) -> BeautifulSoup | None:
        await self._throttle()
        headers = {"User-Agent": _USER_AGENT}
        try:
            response = await self._http.get(url, headers=headers, follow_redirects=True)
            if response.status_code in (403, 429, 500, 502, 503):
                self._logger.debug("beatport_scrape_http_error", url=url, status=response.status_code)
                return None
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except httpx.HTTPError as exc:
            self._logger.debug("beatport_scrape_request_failed", url=url, error=str(exc))
            return None

    @staticmethod
    def _compute_confidence(query: str, result_name: str) -> float:
        normalized_query = normalize_artist_name(query)
        normalized_result = normalize_artist_name(result_name)
        return fuzz.token_sort_ratio(normalized_query, normalized_result) / 100.0

    @staticmethod
    def _parse_artist_id(artist_id: str) -> tuple[str, str]:
        """Parse artist_id in 'slug/numeric_id' format."""
        parts = artist_id.rsplit("/", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return artist_id, artist_id

    # -- IMusicDatabaseProvider implementation ---------------------------------

    async def search_artist(self, name: str) -> list[ArtistSearchResult]:
        encoded = quote_plus(name)

        # Try JSON API first
        data = await self._fetch_json(
            f"{_API_BASE}/catalog/search/?q={encoded}&type=artists&per_page=10"
        )

        if data and data.get("results"):
            return self._parse_json_search(name, data["results"])

        # Also try the newer search endpoint format
        data = await self._fetch_json(
            f"{_API_BASE}/catalog/search/default/?q={encoded}&type=artists&per_page=10"
        )
        if data and isinstance(data, dict):
            artists_data = data.get("artists", data.get("results", []))
            if artists_data:
                return self._parse_json_search(name, artists_data)

        # Fallback: scrape HTML search page
        return await self._scrape_search(name)

    def _parse_json_search(
        self, query: str, results: list[dict[str, Any]]
    ) -> list[ArtistSearchResult]:
        search_results: list[ArtistSearchResult] = []

        for item in results[:_MAX_SEARCH_RESULTS]:
            artist_name = item.get("name", "")
            artist_id = item.get("id", "")
            slug = item.get("slug", "")

            if not artist_name or not artist_id:
                continue

            confidence = self._compute_confidence(query, artist_name)
            composite_id = f"{slug}/{artist_id}" if slug else str(artist_id)

            search_results.append(
                ArtistSearchResult(
                    id=composite_id,
                    name=artist_name,
                    confidence=confidence,
                )
            )

        search_results.sort(key=lambda r: r.confidence, reverse=True)
        self._logger.info("beatport_search_complete", artist=query, results=len(search_results))
        return search_results

    async def _scrape_search(self, name: str) -> list[ArtistSearchResult]:
        encoded = quote_plus(name)
        url = f"{_WEB_BASE}/search?q={encoded}&type=artists"
        soup = await self._fetch_page(url)

        if soup is None:
            return []

        results: list[ArtistSearchResult] = []
        artist_re = re.compile(r"/artist/([^/]+)/(\d+)")

        for link in soup.find_all("a", href=artist_re):
            href = link.get("href", "")
            match = artist_re.search(href)
            if not match:
                continue

            slug, numeric_id = match.groups()
            artist_name = link.get_text(strip=True)
            if not artist_name:
                continue

            confidence = self._compute_confidence(name, artist_name)
            results.append(
                ArtistSearchResult(
                    id=f"{slug}/{numeric_id}",
                    name=artist_name,
                    confidence=confidence,
                )
            )
            if len(results) >= _MAX_SEARCH_RESULTS:
                break

        results.sort(key=lambda r: r.confidence, reverse=True)
        # Deduplicate by ID
        seen: set[str] = set()
        unique: list[ArtistSearchResult] = []
        for r in results:
            if r.id not in seen:
                seen.add(r.id)
                unique.append(r)

        self._logger.info("beatport_scrape_search_complete", artist=name, results=len(unique))
        return unique

    async def get_artist_releases(
        self, artist_id: str, before_date: date | None = None
    ) -> list[Release]:
        slug, numeric_id = self._parse_artist_id(artist_id)

        # Try JSON API
        data = await self._fetch_json(
            f"{_API_BASE}/catalog/releases/?artist_id={numeric_id}&per_page={_MAX_RELEASES}"
        )

        if data and data.get("results"):
            return self._parse_json_releases(data["results"], before_date)

        # Fallback: scrape HTML
        return await self._scrape_releases(slug, numeric_id, before_date)

    def _parse_json_releases(
        self, results: list[dict[str, Any]], before_date: date | None
    ) -> list[Release]:
        releases: list[Release] = []

        for item in results[:_MAX_RELEASES]:
            title = item.get("name", "")
            if not title:
                continue

            label_data = item.get("label", {})
            label_name = label_data.get("name", "Unknown") if isinstance(label_data, dict) else "Unknown"

            year: int | None = None
            date_str = item.get("new_release_date") or item.get("publish_date", "")
            if date_str and len(str(date_str)) >= 4:
                try:
                    year = int(str(date_str)[:4])
                except ValueError:
                    pass

            if before_date and year and year > before_date.year:
                continue

            release_id = item.get("id", "")
            slug = item.get("slug", "")
            beatport_url = f"{_WEB_BASE}/release/{slug}/{release_id}" if release_id else None

            release_type = item.get("type", {})
            format_str = release_type.get("name", "Release") if isinstance(release_type, dict) else "Release"

            genres: list[str] = []
            for genre in item.get("genre", []) if isinstance(item.get("genre"), list) else []:
                if isinstance(genre, dict) and genre.get("name"):
                    genres.append(genre["name"])

            releases.append(
                Release(
                    title=title,
                    label=label_name,
                    year=year,
                    format=format_str,
                    beatport_url=beatport_url,
                    genres=genres,
                )
            )

        self._logger.info("beatport_json_releases_complete", count=len(releases))
        return releases

    async def _scrape_releases(
        self, slug: str, numeric_id: str, before_date: date | None
    ) -> list[Release]:
        url = f"{_WEB_BASE}/artist/{slug}/{numeric_id}/releases"
        soup = await self._fetch_page(url)

        if soup is None:
            return []

        releases: list[Release] = []
        release_re = re.compile(r"/release/([^/]+)/(\d+)")

        for link in soup.find_all("a", href=release_re):
            href = link.get("href", "")
            title = link.get_text(strip=True)
            if not title:
                continue

            match = release_re.search(href)
            if not match:
                continue

            r_slug, r_id = match.groups()
            beatport_url = f"{_WEB_BASE}/release/{r_slug}/{r_id}"

            releases.append(
                Release(
                    title=title,
                    label="Unknown",
                    format="Release",
                    beatport_url=beatport_url,
                )
            )

            if len(releases) >= _MAX_RELEASES:
                break

        # Deduplicate by title
        seen_titles: set[str] = set()
        unique: list[Release] = []
        for r in releases:
            key = r.title.lower().strip()
            if key not in seen_titles:
                seen_titles.add(key)
                unique.append(r)

        self._logger.info(
            "beatport_scrape_releases_complete", slug=slug, count=len(unique)
        )
        return unique

    async def get_artist_labels(self, artist_id: str) -> list[Label]:
        releases = await self.get_artist_releases(artist_id)
        seen: set[str] = set()
        labels: list[Label] = []
        for release in releases:
            label_name = release.label
            if label_name and label_name != "Unknown" and label_name not in seen:
                seen.add(label_name)
                labels.append(Label(name=label_name))
        return labels

    async def get_release_details(self, release_id: str) -> Release | None:
        # release_id may be "slug/numeric_id" or just a numeric id
        _, numeric = self._parse_artist_id(release_id)

        data = await self._fetch_json(f"{_API_BASE}/catalog/releases/{numeric}/")
        if data:
            results = self._parse_json_releases([data], before_date=None)
            return results[0] if results else None

        return None

    async def get_label_releases(
        self, label_id: str, max_results: int = 50
    ) -> list[Release]:
        """Not supported by Beatport provider."""
        return []

    def get_provider_name(self) -> str:
        return "beatport"

    def is_available(self) -> bool:
        return True

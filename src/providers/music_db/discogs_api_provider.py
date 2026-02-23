"""Discogs REST API provider using python3-discogs-client.

Implements IMusicDatabaseProvider for querying artist metadata, releases,
and label information via the authenticated Discogs API.  The client is
initialized lazily and rate-limited to respect Discogs' 60 req/min cap.
"""

from __future__ import annotations

import asyncio
import time
from datetime import date
from typing import Any

import discogs_client
from rapidfuzz import fuzz

from src.config.settings import Settings
from src.interfaces.music_db_provider import ArtistSearchResult, IMusicDatabaseProvider
from src.models.entities import Label, Release
from src.utils.errors import ResearchError
from src.utils.logging import get_logger
from src.utils.text_normalizer import normalize_artist_name

_USER_AGENT = "raiveFlier/0.1.0"
_MIN_REQUEST_INTERVAL = 1.0  # seconds — 60 requests per minute
_MAX_SEARCH_RESULTS = 10
_MAX_RELEASES_PER_FETCH = 100
_MAX_LABEL_LOOKUPS = 10


class DiscogsAPIProvider(IMusicDatabaseProvider):
    """Music database provider backed by the Discogs REST API.

    Uses ``python3-discogs-client`` for authenticated access.  The client
    is initialized lazily on first use.  Rate limiting enforces a minimum
    of 1 second between consecutive API calls so the 60 req/min ceiling
    is never exceeded.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: discogs_client.Client | None = None
        self._last_request_time: float = 0.0
        self._logger = get_logger(__name__)

    # -- Private helpers -------------------------------------------------------

    def _get_client(self) -> discogs_client.Client:
        """Lazily initialize and return the Discogs API client."""
        if self._client is None:
            self._client = discogs_client.Client(
                _USER_AGENT,
                consumer_key=self._settings.discogs_consumer_key,
                consumer_secret=self._settings.discogs_consumer_secret,
            )
        return self._client

    async def _throttle(self) -> None:
        """Enforce minimum interval between API requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if self._last_request_time > 0 and elapsed < _MIN_REQUEST_INTERVAL:
            await asyncio.sleep(_MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()

    @staticmethod
    def _compute_confidence(query: str, result_name: str) -> float:
        """Compute fuzzy-match confidence between *query* and *result_name*."""
        normalized_query = normalize_artist_name(query)
        normalized_result = normalize_artist_name(result_name)
        return fuzz.token_sort_ratio(normalized_query, normalized_result) / 100.0

    def _release_from_list_data(self, data: dict[str, Any]) -> Release:
        """Build a ``Release`` model from an artist-releases list item."""
        release_id = data.get("id", "")
        discogs_url = f"https://www.discogs.com/release/{release_id}" if release_id else ""
        year_raw = data.get("year", 0) or 0
        return Release(
            title=data.get("title", "Unknown"),
            label=data.get("label", "Unknown"),
            year=year_raw if year_raw else None,
            format=data.get("format") or None,
            discogs_url=discogs_url,
        )

    # -- Sync helpers (executed via asyncio.to_thread) -------------------------

    def _search_sync(self, name: str) -> list[dict[str, Any]]:
        """Run a synchronous artist search against the Discogs API."""
        client = self._get_client()
        results = client.search(name, type="artist")
        items: list[dict[str, Any]] = []
        for i, item in enumerate(results):
            if i >= _MAX_SEARCH_RESULTS:
                break
            items.append({"id": item.id, "title": item.data.get("title", "")})
        return items

    def _get_releases_sync(self, artist_id: int) -> list[dict[str, Any]]:
        """Fetch an artist's releases synchronously from the Discogs API."""
        client = self._get_client()
        artist = client.artist(artist_id)
        raw: list[dict[str, Any]] = []
        for i, item in enumerate(artist.releases):
            if i >= _MAX_RELEASES_PER_FETCH:
                break
            raw.append(dict(item.data))
        return raw

    def _get_release_detail_sync(self, release_id: int) -> dict[str, Any]:
        """Fetch full release details synchronously from the Discogs API."""
        client = self._get_client()
        release = client.release(release_id)
        # Accessing .data triggers the HTTP fetch for the full release
        return dict(release.data)

    def _search_label_sync(self, label_name: str) -> dict[str, Any] | None:
        """Search for a label by name and return its ID + name, or ``None``."""
        client = self._get_client()
        results = client.search(label_name, type="label")
        for item in results:
            return {"id": item.id, "name": item.data.get("title", label_name)}
        return None

    def _get_label_releases_sync(self, label_id: int, max_results: int) -> list[dict[str, Any]]:
        """Synchronous fetch of label releases for thread offloading."""
        client = self._get_client()
        label = client.label(label_id)
        results: list[dict[str, Any]] = []
        for release in label.releases:
            if len(results) >= max_results:
                break
            artist_name = getattr(release, "artist", "") or ""
            if not artist_name and hasattr(release, "artists") and release.artists:
                artist_name = release.artists[0].name
            results.append({
                "id": release.id,
                "title": getattr(release, "title", ""),
                "year": getattr(release, "year", 0),
                "format": (
                    ", ".join(getattr(release, "formats", []) or [])
                    if hasattr(release, "formats")
                    else ""
                ),
                "label": label.name if hasattr(label, "name") else "",
                "artist": artist_name,
            })
        return results

    # -- IMusicDatabaseProvider implementation ---------------------------------

    async def search_artist(self, name: str) -> list[ArtistSearchResult]:
        """Search the Discogs API for artists matching *name*."""
        await self._throttle()
        try:
            raw_results = await asyncio.to_thread(self._search_sync, name)
        except Exception as exc:
            self._logger.error("discogs_api_search_failed", artist=name, error=str(exc))
            raise ResearchError(
                message=f"Discogs API search failed for '{name}': {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

        search_results: list[ArtistSearchResult] = []
        for item in raw_results:
            item_name = item.get("title", "")
            confidence = self._compute_confidence(name, item_name)
            search_results.append(
                ArtistSearchResult(
                    id=str(item["id"]),
                    name=item_name,
                    confidence=confidence,
                )
            )

        search_results.sort(key=lambda r: r.confidence, reverse=True)
        self._logger.info(
            "discogs_api_search_complete",
            artist=name,
            result_count=len(search_results),
        )
        return search_results

    async def get_artist_releases(
        self, artist_id: str, before_date: date | None = None
    ) -> list[Release]:
        """Retrieve releases for *artist_id*, optionally filtered by *before_date*."""
        await self._throttle()
        try:
            raw_releases = await asyncio.to_thread(self._get_releases_sync, int(artist_id))
        except Exception as exc:
            self._logger.error(
                "discogs_api_releases_failed",
                artist_id=artist_id,
                error=str(exc),
            )
            raise ResearchError(
                message=(f"Discogs API releases fetch failed for artist {artist_id}: {exc}"),
                provider_name=self.get_provider_name(),
            ) from exc

        releases: list[Release] = []
        for data in raw_releases:
            year = data.get("year", 0) or 0
            if before_date and year and year > before_date.year:
                continue
            releases.append(self._release_from_list_data(data))

        self._logger.info(
            "discogs_api_releases_complete",
            artist_id=artist_id,
            release_count=len(releases),
        )
        return releases

    async def get_artist_labels(self, artist_id: str) -> list[Label]:
        """Extract unique labels from the artist's releases and resolve Discogs IDs."""
        releases = await self.get_artist_releases(artist_id)
        seen: set[str] = set()
        label_names: list[str] = []
        for release in releases:
            label_name = release.label
            if label_name and label_name != "Unknown" and label_name not in seen:
                seen.add(label_name)
                label_names.append(label_name)

        labels: list[Label] = []
        for name in label_names[:_MAX_LABEL_LOOKUPS]:
            await self._throttle()
            try:
                result = await asyncio.to_thread(self._search_label_sync, name)
            except Exception:
                result = None

            if result and result.get("id"):
                label_id = result["id"]
                labels.append(
                    Label(
                        name=name,
                        discogs_id=label_id,
                        discogs_url=f"https://www.discogs.com/label/{label_id}",
                    )
                )
            else:
                labels.append(Label(name=name))

        # Append remaining labels (beyond cap) without IDs
        for name in label_names[_MAX_LABEL_LOOKUPS:]:
            labels.append(Label(name=name))

        self._logger.info(
            "discogs_api_labels_complete",
            artist_id=artist_id,
            label_count=len(labels),
            resolved=sum(1 for lb in labels if lb.discogs_id),
        )
        return labels

    async def get_release_details(self, release_id: str) -> Release | None:
        """Fetch full details for a single release by *release_id*."""
        await self._throttle()
        try:
            data = await asyncio.to_thread(self._get_release_detail_sync, int(release_id))
        except Exception as exc:
            self._logger.error(
                "discogs_api_release_detail_failed",
                release_id=release_id,
                error=str(exc),
            )
            return None

        labels_raw = data.get("labels", [])
        label_name = labels_raw[0]["name"] if labels_raw else "Unknown"
        catalog_number = labels_raw[0].get("catno") if labels_raw else None

        formats_raw = data.get("formats", [])
        format_str: str | None = None
        if formats_raw:
            fmt = formats_raw[0]
            descriptions = fmt.get("descriptions", [])
            format_str = fmt.get("name", "")
            if descriptions:
                format_str = f"{format_str} ({', '.join(descriptions)})"

        return Release(
            title=data.get("title", "Unknown"),
            label=label_name,
            catalog_number=catalog_number,
            year=data.get("year"),
            format=format_str,
            discogs_url=data.get("uri", ""),
            genres=data.get("genres", []),
            styles=data.get("styles", []),
        )

    async def get_label_releases(
        self, label_id: str, max_results: int = 50
    ) -> list[Release]:
        """Fetch releases from a Discogs label to discover label-mates."""
        await self._throttle()
        try:
            releases_data = await asyncio.to_thread(
                self._get_label_releases_sync, int(label_id), max_results
            )
        except Exception as exc:
            self._logger.warning(
                "discogs_api_label_releases_failed",
                label_id=label_id,
                error=str(exc),
            )
            return []

        releases: list[Release] = []
        for data in releases_data:
            releases.append(self._release_from_list_data(data))

        self._logger.info(
            "discogs_api_label_releases_complete",
            label_id=label_id,
            release_count=len(releases),
        )
        return releases

    def get_provider_name(self) -> str:
        """Return ``'discogs_api'``."""
        return "discogs_api"

    def is_available(self) -> bool:
        """Return ``True`` if consumer key and secret are configured."""
        return bool(self._settings.discogs_consumer_key and self._settings.discogs_consumer_secret)

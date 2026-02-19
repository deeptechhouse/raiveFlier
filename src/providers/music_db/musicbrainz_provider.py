"""MusicBrainz provider implementing IMusicDatabaseProvider.

Uses the musicbrainzngs library to query the MusicBrainz open database for
artist metadata, releases, and label information.  Enforces the MusicBrainz
rate limit of 1 request per second via asyncio-based throttling.
"""

from __future__ import annotations

import asyncio
import time
from datetime import date

import musicbrainzngs
import structlog

from src.config.settings import Settings
from src.interfaces.music_db_provider import ArtistSearchResult, IMusicDatabaseProvider
from src.models.entities import Label, Release
from src.utils.errors import ResearchError

logger = structlog.get_logger(logger_name=__name__)


class MusicBrainzProvider(IMusicDatabaseProvider):
    """MusicBrainz music-database provider with built-in rate limiting.

    MusicBrainz is a free, open-source music encyclopedia that provides
    artist, release, and label data.  No API key is required, but clients
    must identify themselves via a user-agent string and respect the
    1 request/second rate limit.

    Attributes
    ----------
    _settings : Settings
        Application settings containing MusicBrainz user-agent details.
    _last_request_time : float
        Monotonic timestamp of the most recent API call, used for throttling.
    """

    _MIN_REQUEST_INTERVAL: float = 1.0  # seconds between requests

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._last_request_time: float = 0.0

        musicbrainzngs.set_useragent(
            settings.musicbrainz_app_name,
            settings.musicbrainz_app_version,
            settings.musicbrainz_contact or None,
        )
        logger.info(
            "musicbrainz_provider_initialized",
            app_name=settings.musicbrainz_app_name,
            app_version=settings.musicbrainz_app_version,
        )

    # ------------------------------------------------------------------
    # Rate-limiting helper
    # ------------------------------------------------------------------

    async def _throttle(self) -> None:
        """Enforce the MusicBrainz 1 req/sec rate limit."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._MIN_REQUEST_INTERVAL:
            await asyncio.sleep(self._MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()

    # ------------------------------------------------------------------
    # IMusicDatabaseProvider implementation
    # ------------------------------------------------------------------

    async def search_artist(self, name: str) -> list[ArtistSearchResult]:
        """Search MusicBrainz for artists matching *name*."""
        await self._throttle()
        try:
            response = musicbrainzngs.search_artists(query=name)
        except musicbrainzngs.WebServiceError as exc:
            raise ResearchError(
                message=f"MusicBrainz artist search failed for '{name}': {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

        results: list[ArtistSearchResult] = []
        for artist in response.get("artist-list", []):
            score = int(artist.get("ext:score", 0))
            results.append(
                ArtistSearchResult(
                    id=artist["id"],
                    name=artist.get("name", ""),
                    disambiguation=artist.get("disambiguation"),
                    confidence=score / 100.0,
                )
            )

        logger.debug(
            "musicbrainz_artist_search",
            query=name,
            result_count=len(results),
        )
        return results

    async def get_artist_releases(
        self, artist_id: str, before_date: date | None = None
    ) -> list[Release]:
        """Retrieve releases for *artist_id*, paginating through all results."""
        all_releases: list[Release] = []
        offset = 0
        limit = 100

        while True:
            await self._throttle()
            try:
                response = musicbrainzngs.browse_releases(
                    artist=artist_id,
                    includes=["labels"],
                    release_type=["album", "ep", "single"],
                    limit=limit,
                    offset=offset,
                )
            except musicbrainzngs.WebServiceError as exc:
                raise ResearchError(
                    message=(
                        f"MusicBrainz browse releases failed for " f"artist '{artist_id}': {exc}"
                    ),
                    provider_name=self.get_provider_name(),
                ) from exc

            release_list = response.get("release-list", [])
            if not release_list:
                break

            for rel in release_list:
                release = self._map_release(rel)
                if before_date and release.year:
                    release_date = date(release.year, 1, 1)
                    if release_date >= before_date:
                        continue
                all_releases.append(release)

            total = int(response.get("release-count", 0))
            offset += limit
            if offset >= total:
                break

        all_releases.sort(key=lambda r: r.year or 0, reverse=True)

        logger.debug(
            "musicbrainz_artist_releases",
            artist_id=artist_id,
            release_count=len(all_releases),
        )
        return all_releases

    async def get_artist_labels(self, artist_id: str) -> list[Label]:
        """Extract unique labels from the artist's releases."""
        releases = await self.get_artist_releases(artist_id)
        seen_names: set[str] = set()
        labels: list[Label] = []

        for release in releases:
            label_name = release.label
            if label_name and label_name not in seen_names:
                seen_names.add(label_name)
                labels.append(Label(name=label_name))

        logger.debug(
            "musicbrainz_artist_labels",
            artist_id=artist_id,
            label_count=len(labels),
        )
        return labels

    async def get_release_details(self, release_id: str) -> Release | None:
        """Fetch full details for a single release by its MusicBrainz ID."""
        await self._throttle()
        try:
            response = musicbrainzngs.get_release_by_id(
                release_id,
                includes=["labels", "recordings"],
            )
        except musicbrainzngs.WebServiceError as exc:
            logger.warning(
                "musicbrainz_release_details_failed",
                release_id=release_id,
                error=str(exc),
            )
            return None

        rel = response.get("release")
        if not rel:
            return None

        return self._map_release(rel)

    def get_provider_name(self) -> str:
        """Return the provider identifier."""
        return "musicbrainz"

    def is_available(self) -> bool:
        """MusicBrainz is always available (no API key required)."""
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _map_release(rel: dict) -> Release:
        """Map a MusicBrainz release dict to a :class:`Release` model."""
        # Extract label name from label-info list
        label_name = ""
        label_info_list = rel.get("label-info-list", [])
        if label_info_list:
            first_label = label_info_list[0]
            label_obj = first_label.get("label", {})
            label_name = label_obj.get("name", "") if label_obj else ""

        # Extract catalog number
        catalog_number = None
        if label_info_list:
            catalog_number = label_info_list[0].get("catalog-number")

        # Extract year from date string (YYYY or YYYY-MM-DD)
        year: int | None = None
        date_str = rel.get("date", "")
        if date_str and len(date_str) >= 4:
            try:
                year = int(date_str[:4])
            except ValueError:
                year = None

        return Release(
            title=rel.get("title", ""),
            label=label_name,
            catalog_number=catalog_number,
            year=year,
            format=rel.get("packaging", rel.get("release-group", {}).get("type")),
        )

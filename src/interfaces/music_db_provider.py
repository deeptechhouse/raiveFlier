"""Abstract base class for music-database service providers.

Defines the contract for querying external music databases (e.g. Discogs,
MusicBrainz) for artist metadata, releases, and label information.  The
adapter pattern (CLAUDE.md Section 6) ensures providers are interchangeable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

from src.models.entities import Label, Release


@dataclass(frozen=True)
class ArtistSearchResult:
    """A single result returned by an artist-name search.

    Attributes
    ----------
    id:
        Provider-specific unique identifier for the artist.
    name:
        The artist's canonical name as listed by the provider.
    disambiguation:
        Optional extra text to distinguish identically-named artists
        (e.g. ``"UK techno DJ"``).
    confidence:
        Estimated match confidence between 0.0 and 1.0.
    """

    id: str
    name: str
    disambiguation: str | None = None
    confidence: float = 0.0


class IMusicDatabaseProvider(ABC):
    """Contract for music-database services used for artist research.

    Implementations query external databases to resolve artist identities,
    retrieve discographies, and look up associated record labels.
    """

    @abstractmethod
    async def search_artist(self, name: str) -> list[ArtistSearchResult]:
        """Search the database for artists matching *name*.

        Parameters
        ----------
        name:
            The artist or DJ name to search for.

        Returns
        -------
        list[ArtistSearchResult]
            Zero or more results ranked by relevance / confidence.

        Raises
        ------
        src.core.errors.ResearchError
            If the API call fails.
        """

    @abstractmethod
    async def get_artist_releases(
        self, artist_id: str, before_date: date | None = None
    ) -> list[Release]:
        """Retrieve releases for the artist identified by *artist_id*.

        Parameters
        ----------
        artist_id:
            Provider-specific artist identifier (from :class:`ArtistSearchResult`).
        before_date:
            If supplied, only return releases published before this date.
            Useful for constraining results to the era of a specific flier.

        Returns
        -------
        list[Release]
            The artist's releases, newest first.
        """

    @abstractmethod
    async def get_artist_labels(self, artist_id: str) -> list[Label]:
        """Retrieve record labels the artist has released on.

        Parameters
        ----------
        artist_id:
            Provider-specific artist identifier.

        Returns
        -------
        list[Label]
            Labels associated with the artist's catalogue.
        """

    @abstractmethod
    async def get_release_details(self, release_id: str) -> Release | None:
        """Fetch full details for a single release.

        Parameters
        ----------
        release_id:
            Provider-specific release identifier.

        Returns
        -------
        Release or None
            The release if found; ``None`` otherwise.
        """

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return a human-readable identifier for this music-database provider.

        Example return values: ``"discogs"``, ``"musicbrainz"``.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if the provider is configured and reachable.

        Implementations should check for required API keys or network
        connectivity without performing a full query.
        """

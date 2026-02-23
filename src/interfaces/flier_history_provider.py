"""Abstract base class for flier history persistence providers.

Defines the contract for logging completed flier analyses and querying
co-artist relationships across historical fliers.  Implementations may
use SQLite (local), PostgreSQL, or any other storage backend.  The
adapter pattern (CLAUDE.md Section 6) allows the persistence backend to
be swapped without touching business logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


# Concrete implementation: SQLiteFlierHistoryProvider (src/providers/flier_history/)
# Stores flier metadata and artist co-appearances in flier_history.db.
# Used by the recommendation service to discover co-appearing artists and
# by the upload endpoint to detect duplicate fliers via perceptual hashing.
class IFlierHistoryProvider(ABC):
    """Contract for flier-history persistence services.

    All operations are async to support network-backed stores.
    """

    @abstractmethod
    async def log_flier(
        self,
        session_id: str,
        artists: list[str],
        venue: str | None,
        promoter: str | None,
        event_name: str | None,
        event_date: str | None,
        genre_tags: list[str],
    ) -> dict[str, Any]:
        """Store a completed flier analysis.

        Parameters
        ----------
        session_id:
            Pipeline session UUID uniquely identifying this flier analysis.
        artists:
            List of artist names extracted from the flier.
        venue:
            Venue name, or ``None`` if not identified.
        promoter:
            Promoter name, or ``None`` if not identified.
        event_name:
            Event/party name, or ``None`` if not identified.
        event_date:
            Event date as a string, or ``None`` if not identified.
        genre_tags:
            List of genre tags associated with the flier.

        Returns
        -------
        dict
            The inserted flier record including ``id``, ``session_id``,
            ``venue``, ``promoter``, ``event_name``, ``event_date``,
            ``genre_tags``, ``created_at``, and ``artists``.
        """

    @abstractmethod
    async def find_co_artists(
        self,
        artist_names: list[str],
    ) -> list[dict[str, Any]]:
        """Find artists who appeared on other fliers alongside any of the given artists.

        Parameters
        ----------
        artist_names:
            List of artist names to search for co-appearances.

        Returns
        -------
        list[dict]
            Each dict contains ``artist_name``, ``shared_with``,
            ``event_names``, ``venues``, ``times_seen``.
        """

    @abstractmethod
    async def get_flier_count(self) -> int:
        """Return the total number of logged fliers.

        Returns
        -------
        int
            Count of all flier records in the store.
        """

    @abstractmethod
    async def register_image_hash(
        self,
        session_id: str,
        image_phash: str,
    ) -> None:
        """Store a perceptual hash immediately on upload for duplicate detection.

        Parameters
        ----------
        session_id:
            Pipeline session UUID for this upload.
        image_phash:
            Hex-encoded perceptual hash of the uploaded image.
        """

    @abstractmethod
    async def find_duplicate_by_phash(
        self,
        image_phash: str,
        threshold: int = 10,
    ) -> dict[str, Any] | None:
        """Check if a visually similar flier has been analyzed before.

        Compares the given perceptual hash against all stored hashes using
        Hamming distance.  Returns metadata about the closest match if
        the distance is below *threshold*, otherwise ``None``.

        Parameters
        ----------
        image_phash:
            Hex-encoded perceptual hash of the new image.
        threshold:
            Maximum Hamming distance to consider a match (default 10).

        Returns
        -------
        dict | None
            Match info with keys ``session_id``, ``similarity``,
            ``analyzed_at``, ``artists``, ``venue``, ``event_name``,
            ``event_date``, ``hamming_distance``; or ``None`` if no match.
        """

    @abstractmethod
    async def initialize(self) -> None:
        """Create tables/indices if they don't exist.  Called at startup."""

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return a human-readable identifier for this provider."""

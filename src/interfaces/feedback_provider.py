"""Abstract base class for feedback/rating service providers.

Defines the contract for persisting user feedback (thumbs up/down) on
analysis results.  Implementations may use SQLite (local), PostgreSQL,
or any other storage backend.  The adapter pattern (CLAUDE.md Section 6)
allows the feedback backend to be swapped without touching business logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class IFeedbackProvider(ABC):
    """Contract for user-feedback persistence services.

    All operations are async to support network-backed stores.
    """

    @abstractmethod
    async def submit_rating(
        self,
        session_id: str,
        item_type: str,
        item_key: str,
        rating: int,
    ) -> dict[str, Any]:
        """Store or update a user rating for a specific result item.

        Parameters
        ----------
        session_id:
            Pipeline session UUID, or ``"global"`` for session-independent
            items (e.g. corpus search results).
        item_type:
            The category of result being rated (ARTIST, VENUE, etc.).
        item_key:
            Natural key identifying the item within its type and session.
        rating:
            ``+1`` for thumbs up, ``-1`` for thumbs down.

        Returns
        -------
        dict
            Contains ``id``, ``session_id``, ``item_type``, ``item_key``,
            ``rating``, ``created_at``, ``updated_at``.
        """

    @abstractmethod
    async def get_ratings(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve all ratings for a given session.

        Parameters
        ----------
        session_id:
            Pipeline session UUID.

        Returns
        -------
        list[dict]
            All ratings stored for this session.
        """

    @abstractmethod
    async def get_rating_summary(
        self,
        item_type: str | None = None,
    ) -> dict[str, Any]:
        """Return aggregate rating statistics across all sessions.

        Used to inform future accuracy adjustments.

        Parameters
        ----------
        item_type:
            Optional filter to restrict summary to a specific item type.

        Returns
        -------
        dict
            Contains ``total_ratings``, ``positive``, ``negative``,
            ``by_type`` breakdown.
        """

    @abstractmethod
    async def initialize(self) -> None:
        """Create tables/indices if they don't exist.  Called at startup."""

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return a human-readable identifier for this provider."""

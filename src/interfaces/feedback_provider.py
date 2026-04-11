"""Abstract base class for feedback/rating service providers.

Defines the contract for persisting user feedback (thumbs up/down) on
analysis results.  Implementations may use SQLite (local), PostgreSQL,
or any other storage backend.  The adapter pattern (CLAUDE.md Section 6)
allows the feedback backend to be swapped without touching business logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


# Concrete implementation: SQLiteFeedbackProvider (src/providers/feedback/)
# Stores ratings in feedback.db on the persistent /data disk (Render).
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
    async def get_negative_item_keys(
        self,
        item_type: str,
        item_key_prefix: str,
    ) -> set[str]:
        """Return item_keys with net-negative ratings across ALL sessions.

        Used for cross-session filtering: when prior sessions have
        thumbs-downed a release or label for a given artist, future
        sessions should exclude or flag that item.

        Parameters
        ----------
        item_type:
            The category to filter (e.g. ``"RELEASE"`` or ``"LABEL"``).
        item_key_prefix:
            A prefix to match against ``item_key`` using SQL ``LIKE``.
            For releases: ``"Henry Brooks::release::"`` matches all
            releases for artist "Henry Brooks".

        Returns
        -------
        set[str]
            Set of full ``item_key`` strings that have net-negative ratings.
        """

    # -- Calibration data collection -------------------------------------------
    # These methods support progressive confidence calibration (Optimization I).
    # The confirmation gate records how often LLM-reported confidence scores
    # correspond to correct entity extractions (user kept vs removed).  Over
    # time this data trains a lookup-table calibrator that maps raw scores to
    # empirical accuracy — e.g. "an LLM score of 0.85 actually means 0.62
    # probability of being correct."

    @abstractmethod
    async def store_calibration_sample(
        self,
        score_type: str,
        predicted_score: float,
        ground_truth: int,
        entity_type: str,
        session_id: str,
    ) -> None:
        """Record a calibration observation (predicted vs actual).

        Called by ConfirmationGate after the user confirms/edits entities.

        Parameters
        ----------
        score_type:
            Category of score being calibrated (e.g. ``"entity_confidence"``
            for LLM extraction scores, ``"fuzzy_match"`` for database match
            scores).
        predicted_score:
            The raw confidence score the model/algorithm reported.
        ground_truth:
            ``1`` if the entity was correct (user kept it), ``0`` if the
            user removed or corrected it.
        entity_type:
            The kind of entity (``"ARTIST"``, ``"VENUE"``, etc.) for
            per-type calibration analysis.
        session_id:
            Pipeline session UUID for traceability.
        """

    @abstractmethod
    async def get_calibration_data(self, score_type: str) -> list[dict]:
        """Return all (predicted_score, ground_truth) pairs for a score type.

        Used to fit a ConfidenceCalibrator on startup or periodically.

        Parameters
        ----------
        score_type:
            The category of calibration data to retrieve.

        Returns
        -------
        list[dict]
            Each dict contains ``predicted_score`` (float) and
            ``ground_truth`` (int).
        """

    @abstractmethod
    async def initialize(self) -> None:
        """Create tables/indices if they don't exist.  Called at startup."""

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return a human-readable identifier for this provider."""

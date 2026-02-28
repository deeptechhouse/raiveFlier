"""Abstract base class for rave story persistence providers.

# ─── ADAPTER PATTERN ─────────────────────────────────────────────────
#
# IStoryProvider follows the same adapter pattern as IFeedbackProvider
# (src/interfaces/feedback_provider.py).  The concrete implementation is
# SQLiteStoryProvider (src/providers/story/sqlite_story_provider.py).
#
# All operations are async to support network-backed stores if the
# persistence layer is swapped from SQLite to PostgreSQL or another
# backend in the future (CLAUDE.md Section 6 — service abstraction).
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.models.story import RaveStory, StoryMetadata


# Concrete implementation: SQLiteStoryProvider (src/providers/story/)
# Stores stories in stories.db on the persistent /data disk (Render).
class IStoryProvider(ABC):
    """Contract for rave story persistence services.

    Handles CRUD operations for anonymous rave stories, event collections,
    entity tags, and collective narratives.  All storage operations are async.
    """

    # ── Lifecycle ──────────────────────────────────────────────────────

    @abstractmethod
    async def initialize(self) -> None:
        """Create tables/indices if they don't exist.  Called at startup."""

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return a human-readable identifier for this provider."""

    # ── Story CRUD ─────────────────────────────────────────────────────

    @abstractmethod
    async def submit_story(
        self,
        story: RaveStory,
    ) -> dict[str, Any]:
        """Persist a new story with its metadata and tags.

        Parameters
        ----------
        story:
            The fully-moderated RaveStory to store.

        Returns
        -------
        dict
            Contains ``story_id``, ``status``, ``created_at``.
        """

    @abstractmethod
    async def get_story(self, story_id: str) -> RaveStory | None:
        """Retrieve a single story by its UUID.

        Returns None if the story doesn't exist or is not APPROVED.
        """

    @abstractmethod
    async def list_stories(
        self,
        *,
        status: str | None = None,
        tag_type: str | None = None,
        tag_value: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[RaveStory]:
        """List stories with optional filtering and pagination.

        Parameters
        ----------
        status:
            Filter by StoryStatus value (e.g. 'APPROVED').
        tag_type:
            Filter by tag type ('entity', 'genre', 'geographic').
        tag_value:
            Filter by tag value (requires tag_type).
        limit:
            Max results per page.
        offset:
            Number of results to skip (for pagination).

        Returns
        -------
        list[RaveStory]
            Stories matching the filter criteria, newest first.
        """

    @abstractmethod
    async def update_story_status(
        self,
        story_id: str,
        status: str,
        moderation_flags: list[str] | None = None,
    ) -> bool:
        """Update a story's moderation status.

        Returns True if the story was found and updated.
        """

    # ── Event collections ──────────────────────────────────────────────

    @abstractmethod
    async def list_events(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List distinct events with approved story counts.

        Returns
        -------
        list[dict]
            Each dict: ``event_name``, ``event_year``, ``city``, ``story_count``.
        """

    @abstractmethod
    async def get_event_stories(
        self,
        event_name: str,
        event_year: int | None = None,
    ) -> list[RaveStory]:
        """Get all approved stories for a specific event."""

    # ── Narrative cache ────────────────────────────────────────────────

    @abstractmethod
    async def get_narrative(
        self,
        event_name: str,
        event_year: int | None = None,
    ) -> dict[str, Any] | None:
        """Retrieve the cached collective narrative for an event.

        Returns None if no narrative has been generated yet.
        """

    @abstractmethod
    async def save_narrative(
        self,
        event_name: str,
        event_year: int | None,
        narrative: str,
        themes: list[str],
        story_count: int,
    ) -> None:
        """Cache a generated collective narrative for an event."""

    # ── Tags ───────────────────────────────────────────────────────────

    @abstractmethod
    async def get_tags(self, tag_type: str) -> list[str]:
        """List all distinct tag values for a given tag type.

        Parameters
        ----------
        tag_type:
            One of 'entity', 'genre', 'geographic'.

        Returns
        -------
        list[str]
            Sorted list of distinct tag values.
        """

    # ── Statistics ─────────────────────────────────────────────────────

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]:
        """Return aggregate statistics about the story collection.

        Returns
        -------
        dict
            Contains ``total_stories``, ``approved_stories``, ``total_events``,
            ``total_entity_tags``, ``total_genre_tags``, ``total_geographic_tags``,
            ``stories_by_status``, ``stories_by_input_mode``.
        """

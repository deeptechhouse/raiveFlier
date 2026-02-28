"""SQLite-backed rave story persistence provider.

# ─── ARCHITECTURE ROLE ───────────────────────────────────────────────
#
# Layer: Providers (concrete adapter implementing IStoryProvider).
# Pattern: Adapter pattern — wraps SQLite behind the IStoryProvider ABC
#          so the persistence backend can be swapped without touching
#          business logic (CLAUDE.md Section 6).
#
# Database: ``data/stories.db`` — stores anonymous rave experience accounts.
#
# Anonymity by design:
#   - No IP address, user ID, session ID, or User-Agent columns.
#   - ``created_at`` stores only the date (YYYY-MM-DD), never a timestamp.
#   - This is enforced at the schema level — the columns don't exist.
#
# Follows the same SQLite provider pattern as:
#   - src/providers/feedback/sqlite_feedback_provider.py
#   - src/providers/flier_history/sqlite_flier_history_provider.py
#
# Uses ``aiosqlite`` for async I/O and ``PRAGMA journal_mode=WAL`` for
# concurrent read safety.
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiosqlite
import structlog

from src.interfaces.story_provider import IStoryProvider
from src.models.story import RaveStory, StoryMetadata, StoryStatus

logger = structlog.get_logger(logger_name=__name__)

_DEFAULT_DB_PATH = Path("data/stories.db")

# ── Schema DDL ────────────────────────────────────────────────────────
# All SQL defined as module-level constants for clarity and testability.

_CREATE_STORIES_TABLE = """\
CREATE TABLE IF NOT EXISTS stories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    story_id        TEXT    NOT NULL UNIQUE,
    text            TEXT    NOT NULL,
    word_count      INTEGER NOT NULL DEFAULT 0,
    input_mode      TEXT    NOT NULL DEFAULT 'text',
    audio_duration  REAL,
    status          TEXT    NOT NULL DEFAULT 'PENDING_MODERATION',
    moderation_flags TEXT,
    created_at      TEXT    NOT NULL,
    moderated_at    TEXT
);
"""

_CREATE_METADATA_TABLE = """\
CREATE TABLE IF NOT EXISTS story_metadata (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    story_id    TEXT NOT NULL UNIQUE REFERENCES stories(story_id),
    event_name  TEXT,
    event_year  INTEGER,
    city        TEXT,
    genre       TEXT,
    promoter    TEXT,
    artist      TEXT,
    other       TEXT
);
"""

_CREATE_TAGS_TABLE = """\
CREATE TABLE IF NOT EXISTS story_entity_tags (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    story_id  TEXT NOT NULL REFERENCES stories(story_id),
    tag_type  TEXT NOT NULL,
    tag_value TEXT NOT NULL,
    UNIQUE(story_id, tag_type, tag_value)
);
"""

_CREATE_NARRATIVES_TABLE = """\
CREATE TABLE IF NOT EXISTS event_narratives (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    event_name    TEXT NOT NULL,
    event_year    INTEGER,
    narrative     TEXT NOT NULL,
    themes        TEXT,
    story_count   INTEGER NOT NULL,
    generated_at  TEXT NOT NULL,
    UNIQUE(event_name, event_year)
);
"""

_CREATE_INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_stories_status ON stories(status);",
    "CREATE INDEX IF NOT EXISTS idx_stories_created ON stories(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_metadata_event ON story_metadata(event_name);",
    "CREATE INDEX IF NOT EXISTS idx_metadata_city ON story_metadata(city);",
    "CREATE INDEX IF NOT EXISTS idx_metadata_genre ON story_metadata(genre);",
    "CREATE INDEX IF NOT EXISTS idx_tags_type_value ON story_entity_tags(tag_type, tag_value);",
    "CREATE INDEX IF NOT EXISTS idx_tags_story ON story_entity_tags(story_id);",
]

# ── DML ───────────────────────────────────────────────────────────────

_INSERT_STORY = """\
INSERT INTO stories (story_id, text, word_count, input_mode, audio_duration, status, moderation_flags, created_at, moderated_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_INSERT_METADATA = """\
INSERT INTO story_metadata (story_id, event_name, event_year, city, genre, promoter, artist, other)
VALUES (?, ?, ?, ?, ?, ?, ?, ?);
"""

_INSERT_TAG = """\
INSERT OR IGNORE INTO story_entity_tags (story_id, tag_type, tag_value)
VALUES (?, ?, ?);
"""

_SELECT_STORY = """\
SELECT s.story_id, s.text, s.word_count, s.input_mode, s.audio_duration,
       s.status, s.moderation_flags, s.created_at, s.moderated_at,
       m.event_name, m.event_year, m.city, m.genre, m.promoter, m.artist, m.other
FROM stories s
LEFT JOIN story_metadata m ON s.story_id = m.story_id
WHERE s.story_id = ?;
"""

_UPDATE_STATUS = """\
UPDATE stories SET status = ?, moderation_flags = ?, moderated_at = date('now')
WHERE story_id = ?;
"""

_UPSERT_NARRATIVE = """\
INSERT INTO event_narratives (event_name, event_year, narrative, themes, story_count, generated_at)
VALUES (?, ?, ?, ?, ?, date('now'))
ON CONFLICT(event_name, event_year)
DO UPDATE SET narrative = excluded.narrative,
              themes = excluded.themes,
              story_count = excluded.story_count,
              generated_at = excluded.generated_at;
"""


class SQLiteStoryProvider(IStoryProvider):
    """SQLite-backed rave story persistence.

    Stores anonymous stories, metadata, entity tags, and cached event
    narratives in ``data/stories.db``.  No user-identifying information
    is stored at the schema level — anonymity is enforced by design.
    """

    def __init__(self, db_path: str | Path = _DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)

    async def initialize(self) -> None:
        """Create all story tables and indices if they don't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(self._db_path)) as db:
            # WAL mode enables concurrent readers while a writer is active.
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.execute(_CREATE_STORIES_TABLE)
            await db.execute(_CREATE_METADATA_TABLE)
            await db.execute(_CREATE_TAGS_TABLE)
            await db.execute(_CREATE_NARRATIVES_TABLE)
            for idx_sql in _CREATE_INDICES:
                await db.execute(idx_sql)
            await db.commit()
        logger.info("story_db_initialized", path=str(self._db_path))

    def get_provider_name(self) -> str:
        return "sqlite_story"

    # ── Story CRUD ─────────────────────────────────────────────────────

    async def submit_story(self, story: RaveStory) -> dict[str, Any]:
        """Persist a new story along with its metadata and entity tags."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            # Insert the story row.
            await db.execute(_INSERT_STORY, (
                story.story_id,
                story.text,
                story.word_count,
                story.input_mode,
                story.audio_duration,
                story.status.value,
                json.dumps(story.moderation_flags) if story.moderation_flags else None,
                story.created_at,
                story.moderated_at,
            ))

            # Insert metadata (one row per story).
            meta = story.metadata
            await db.execute(_INSERT_METADATA, (
                story.story_id,
                meta.event_name,
                meta.event_year,
                meta.city,
                meta.genre,
                meta.promoter,
                meta.artist,
                meta.other,
            ))

            # Insert entity tags (many rows per story).
            for tag in story.entity_tags:
                await db.execute(_INSERT_TAG, (story.story_id, "entity", tag))
            for tag in story.genre_tags:
                await db.execute(_INSERT_TAG, (story.story_id, "genre", tag))
            for tag in story.geographic_tags:
                await db.execute(_INSERT_TAG, (story.story_id, "geographic", tag))

            await db.commit()

        logger.info(
            "story_submitted",
            story_id=story.story_id,
            status=story.status.value,
            word_count=story.word_count,
            input_mode=story.input_mode,
        )
        return {
            "story_id": story.story_id,
            "status": story.status.value,
            "created_at": story.created_at,
        }

    async def get_story(self, story_id: str) -> RaveStory | None:
        """Retrieve a single story by UUID, including metadata and tags."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(_SELECT_STORY, (story_id,))
            row = await cursor.fetchone()
            if row is None:
                return None

            tags = await self._get_tags_for_story(db, story_id)
            return self._row_to_story(dict(row), tags)

    async def list_stories(
        self,
        *,
        status: str | None = None,
        tag_type: str | None = None,
        tag_value: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[RaveStory]:
        """List stories with optional filtering and pagination."""
        # Build dynamic query based on filters.
        conditions: list[str] = []
        params: list[Any] = []

        if status:
            conditions.append("s.status = ?")
            params.append(status)

        # Tag filtering requires a join to story_entity_tags.
        join_tags = ""
        if tag_type and tag_value:
            join_tags = "JOIN story_entity_tags t ON s.story_id = t.story_id"
            conditions.append("t.tag_type = ?")
            params.append(tag_type)
            conditions.append("t.tag_value = ?")
            params.append(tag_value)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([limit, offset])

        query = f"""\
            SELECT DISTINCT s.story_id, s.text, s.word_count, s.input_mode, s.audio_duration,
                   s.status, s.moderation_flags, s.created_at, s.moderated_at,
                   m.event_name, m.event_year, m.city, m.genre, m.promoter, m.artist, m.other
            FROM stories s
            LEFT JOIN story_metadata m ON s.story_id = m.story_id
            {join_tags}
            {where_clause}
            ORDER BY s.created_at DESC, s.id DESC
            LIMIT ? OFFSET ?;
        """

        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

            stories: list[RaveStory] = []
            for row in rows:
                row_dict = dict(row)
                tags = await self._get_tags_for_story(db, row_dict["story_id"])
                stories.append(self._row_to_story(row_dict, tags))

        return stories

    async def update_story_status(
        self,
        story_id: str,
        status: str,
        moderation_flags: list[str] | None = None,
    ) -> bool:
        """Update a story's moderation status and flags."""
        flags_json = json.dumps(moderation_flags) if moderation_flags else None
        async with aiosqlite.connect(str(self._db_path)) as db:
            cursor = await db.execute(_UPDATE_STATUS, (status, flags_json, story_id))
            await db.commit()
            return cursor.rowcount > 0

    # ── Event collections ──────────────────────────────────────────────

    async def list_events(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List distinct events that have approved stories, with counts."""
        query = """\
            SELECT m.event_name, m.event_year, m.city, COUNT(*) as story_count
            FROM story_metadata m
            JOIN stories s ON m.story_id = s.story_id
            WHERE s.status = 'APPROVED' AND m.event_name IS NOT NULL
            GROUP BY m.event_name, m.event_year
            ORDER BY story_count DESC, m.event_name ASC
            LIMIT ? OFFSET ?;
        """
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, (limit, offset))
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_event_stories(
        self,
        event_name: str,
        event_year: int | None = None,
    ) -> list[RaveStory]:
        """Get all approved stories for a specific event."""
        conditions = ["s.status = 'APPROVED'", "m.event_name = ?"]
        params: list[Any] = [event_name]

        if event_year is not None:
            conditions.append("m.event_year = ?")
            params.append(event_year)

        where_clause = " AND ".join(conditions)
        query = f"""\
            SELECT s.story_id, s.text, s.word_count, s.input_mode, s.audio_duration,
                   s.status, s.moderation_flags, s.created_at, s.moderated_at,
                   m.event_name, m.event_year, m.city, m.genre, m.promoter, m.artist, m.other
            FROM stories s
            JOIN story_metadata m ON s.story_id = m.story_id
            WHERE {where_clause}
            ORDER BY s.created_at ASC;
        """

        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

            stories: list[RaveStory] = []
            for row in rows:
                row_dict = dict(row)
                tags = await self._get_tags_for_story(db, row_dict["story_id"])
                stories.append(self._row_to_story(row_dict, tags))

        return stories

    # ── Narrative cache ────────────────────────────────────────────────

    async def get_narrative(
        self,
        event_name: str,
        event_year: int | None = None,
    ) -> dict[str, Any] | None:
        """Retrieve cached collective narrative for an event."""
        if event_year is not None:
            query = "SELECT * FROM event_narratives WHERE event_name = ? AND event_year = ?;"
            params: tuple[Any, ...] = (event_name, event_year)
        else:
            query = "SELECT * FROM event_narratives WHERE event_name = ? AND event_year IS NULL;"
            params = (event_name,)

        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            row = await cursor.fetchone()

        if row is None:
            return None
        result = dict(row)
        # Parse themes from JSON string.
        if result.get("themes"):
            result["themes"] = json.loads(result["themes"])
        else:
            result["themes"] = []
        return result

    async def save_narrative(
        self,
        event_name: str,
        event_year: int | None,
        narrative: str,
        themes: list[str],
        story_count: int,
    ) -> None:
        """Cache a generated collective narrative for an event."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(_UPSERT_NARRATIVE, (
                event_name,
                event_year,
                narrative,
                json.dumps(themes),
                story_count,
            ))
            await db.commit()
        logger.info(
            "narrative_saved",
            event_name=event_name,
            event_year=event_year,
            story_count=story_count,
        )

    # ── Tags ───────────────────────────────────────────────────────────

    async def get_tags(self, tag_type: str) -> list[str]:
        """List all distinct tag values for a given tag type."""
        # Only return tags from approved stories.
        query = """\
            SELECT DISTINCT t.tag_value
            FROM story_entity_tags t
            JOIN stories s ON t.story_id = s.story_id
            WHERE t.tag_type = ? AND s.status = 'APPROVED'
            ORDER BY t.tag_value ASC;
        """
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, (tag_type,))
            rows = await cursor.fetchall()
        return [row["tag_value"] for row in rows]

    # ── Statistics ─────────────────────────────────────────────────────

    async def get_stats(self) -> dict[str, Any]:
        """Return aggregate statistics about the story collection."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row

            # Total stories by status.
            cursor = await db.execute(
                "SELECT status, COUNT(*) as cnt FROM stories GROUP BY status;"
            )
            status_rows = await cursor.fetchall()
            by_status: dict[str, int] = {}
            total = 0
            approved = 0
            for row in status_rows:
                by_status[row["status"]] = row["cnt"]
                total += row["cnt"]
                if row["status"] == "APPROVED":
                    approved = row["cnt"]

            # Stories by input mode.
            cursor = await db.execute(
                "SELECT input_mode, COUNT(*) as cnt FROM stories GROUP BY input_mode;"
            )
            mode_rows = await cursor.fetchall()
            by_mode: dict[str, int] = {row["input_mode"]: row["cnt"] for row in mode_rows}

            # Distinct events.
            cursor = await db.execute(
                "SELECT COUNT(DISTINCT event_name) as cnt FROM story_metadata "
                "WHERE event_name IS NOT NULL;"
            )
            event_count_row = await cursor.fetchone()
            total_events = event_count_row["cnt"] if event_count_row else 0

            # Distinct tag counts.
            tag_counts: dict[str, int] = {}
            for tag_type in ("entity", "genre", "geographic"):
                cursor = await db.execute(
                    "SELECT COUNT(DISTINCT tag_value) as cnt FROM story_entity_tags WHERE tag_type = ?;",
                    (tag_type,),
                )
                tag_row = await cursor.fetchone()
                tag_counts[tag_type] = tag_row["cnt"] if tag_row else 0

        return {
            "total_stories": total,
            "approved_stories": approved,
            "total_events": total_events,
            "total_entity_tags": tag_counts.get("entity", 0),
            "total_genre_tags": tag_counts.get("genre", 0),
            "total_geographic_tags": tag_counts.get("geographic", 0),
            "stories_by_status": by_status,
            "stories_by_input_mode": by_mode,
        }

    # ── Private helpers ────────────────────────────────────────────────

    async def _get_tags_for_story(
        self,
        db: aiosqlite.Connection,
        story_id: str,
    ) -> dict[str, list[str]]:
        """Fetch all entity tags for a story, grouped by tag_type."""
        cursor = await db.execute(
            "SELECT tag_type, tag_value FROM story_entity_tags WHERE story_id = ?;",
            (story_id,),
        )
        rows = await cursor.fetchall()
        tags: dict[str, list[str]] = {"entity": [], "genre": [], "geographic": []}
        for row in rows:
            tag_type = row[0] if isinstance(row, tuple) else row["tag_type"]
            tag_value = row[1] if isinstance(row, tuple) else row["tag_value"]
            if tag_type in tags:
                tags[tag_type].append(tag_value)
        return tags

    @staticmethod
    def _row_to_story(
        row: dict[str, Any],
        tags: dict[str, list[str]],
    ) -> RaveStory:
        """Convert a joined stories+metadata row into a RaveStory model."""
        # Parse moderation_flags from JSON string.
        flags_raw = row.get("moderation_flags")
        moderation_flags = json.loads(flags_raw) if flags_raw else []

        metadata = StoryMetadata(
            event_name=row.get("event_name"),
            event_year=row.get("event_year"),
            city=row.get("city"),
            genre=row.get("genre"),
            promoter=row.get("promoter"),
            artist=row.get("artist"),
            other=row.get("other"),
        )

        return RaveStory(
            story_id=row["story_id"],
            text=row["text"],
            word_count=row["word_count"],
            input_mode=row["input_mode"],
            audio_duration=row.get("audio_duration"),
            status=StoryStatus(row["status"]),
            moderation_flags=moderation_flags,
            created_at=row["created_at"],
            moderated_at=row.get("moderated_at"),
            metadata=metadata,
            entity_tags=tags.get("entity", []),
            genre_tags=tags.get("genre", []),
            geographic_tags=tags.get("geographic", []),
        )

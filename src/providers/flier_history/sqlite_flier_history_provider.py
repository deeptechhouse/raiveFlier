"""SQLite-backed flier history provider.

Persists completed flier analyses and artist co-appearance data to a
local SQLite database at ``data/flier_history.db``.  Uses ``aiosqlite``
for async I/O.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiosqlite
import structlog

from src.interfaces.flier_history_provider import IFlierHistoryProvider

logger = structlog.get_logger(logger_name=__name__)

_DEFAULT_DB_PATH = Path("data/flier_history.db")

_CREATE_FLIERS_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS fliers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL UNIQUE,
    venue       TEXT,
    promoter    TEXT,
    event_name  TEXT,
    event_date  TEXT,
    genre_tags  TEXT,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""

_CREATE_FLIER_ARTISTS_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS flier_artists (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    flier_id               INTEGER NOT NULL REFERENCES fliers(id),
    artist_name            TEXT    NOT NULL,
    artist_name_normalized TEXT    NOT NULL
);
"""

_CREATE_INDICES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_flier_artists_normalized ON flier_artists(artist_name_normalized);",
    "CREATE INDEX IF NOT EXISTS idx_flier_artists_flier ON flier_artists(flier_id);",
    "CREATE INDEX IF NOT EXISTS idx_fliers_session ON fliers(session_id);",
]

_INSERT_FLIER_SQL = """\
INSERT INTO fliers (session_id, venue, promoter, event_name, event_date, genre_tags)
VALUES (?, ?, ?, ?, ?, ?);
"""

_INSERT_FLIER_ARTIST_SQL = """\
INSERT INTO flier_artists (flier_id, artist_name, artist_name_normalized)
VALUES (?, ?, ?);
"""

_SELECT_FLIER_SQL = """\
SELECT id, session_id, venue, promoter, event_name, event_date, genre_tags, created_at
FROM fliers
WHERE id = ?;
"""

_SELECT_FLIER_ARTISTS_SQL = """\
SELECT artist_name
FROM flier_artists
WHERE flier_id = ?;
"""

_FIND_CO_ARTISTS_SQL = """\
SELECT
    fa2.artist_name,
    GROUP_CONCAT(DISTINCT fa1.artist_name) as shared_with,
    GROUP_CONCAT(DISTINCT f.event_name) as event_names,
    GROUP_CONCAT(DISTINCT f.venue) as venues,
    COUNT(DISTINCT f.id) as times_seen
FROM flier_artists fa1
JOIN flier_artists fa2 ON fa1.flier_id = fa2.flier_id
    AND fa2.artist_name_normalized NOT IN ({placeholders})
JOIN fliers f ON fa1.flier_id = f.id
WHERE fa1.artist_name_normalized IN ({placeholders})
GROUP BY fa2.artist_name_normalized
ORDER BY times_seen DESC;
"""

_COUNT_FLIERS_SQL = "SELECT COUNT(*) FROM fliers;"


class SQLiteFlierHistoryProvider(IFlierHistoryProvider):
    """SQLite-backed flier history persistence."""

    def __init__(self, db_path: str | Path = _DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)

    async def initialize(self) -> None:
        """Create the fliers and flier_artists tables and indices if they don't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(_CREATE_FLIERS_TABLE_SQL)
            await db.execute(_CREATE_FLIER_ARTISTS_TABLE_SQL)
            for idx_sql in _CREATE_INDICES_SQL:
                await db.execute(idx_sql)
            await db.commit()
        logger.info("flier_history_db_initialized", path=str(self._db_path))

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
        """Store a completed flier analysis.  Returns the inserted record."""
        genre_tags_json = json.dumps(genre_tags)

        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute(
                _INSERT_FLIER_SQL,
                (session_id, venue, promoter, event_name, event_date, genre_tags_json),
            )
            flier_id = cursor.lastrowid

            for artist_name in artists:
                normalized = artist_name.strip().lower()
                await db.execute(
                    _INSERT_FLIER_ARTIST_SQL,
                    (flier_id, artist_name, normalized),
                )

            await db.commit()

            cursor = await db.execute(_SELECT_FLIER_SQL, (flier_id,))
            row = await cursor.fetchone()

        result = dict(row)
        result["genre_tags"] = json.loads(result["genre_tags"]) if result["genre_tags"] else []
        result["artists"] = list(artists)

        logger.info(
            "flier_logged",
            session_id=session_id,
            artist_count=len(artists),
            venue=venue,
            event_name=event_name,
        )
        return result

    async def find_co_artists(
        self,
        artist_names: list[str],
    ) -> list[dict[str, Any]]:
        """Find artists who appeared on other fliers alongside any of the given artists."""
        if not artist_names:
            return []

        normalized_names = [name.strip().lower() for name in artist_names]
        placeholders = ", ".join("?" for _ in normalized_names)
        query = _FIND_CO_ARTISTS_SQL.format(placeholders=placeholders)
        params = normalized_names + normalized_names

        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            results.append({
                "artist_name": row["artist_name"],
                "shared_with": row["shared_with"],
                "event_names": row["event_names"],
                "venues": row["venues"],
                "times_seen": row["times_seen"],
            })
        return results

    async def get_flier_count(self) -> int:
        """Return the total number of logged fliers."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            cursor = await db.execute(_COUNT_FLIERS_SQL)
            row = await cursor.fetchone()
        return row[0] if row else 0

    def get_provider_name(self) -> str:
        """Return a human-readable identifier for this provider."""
        return "sqlite_flier_history"

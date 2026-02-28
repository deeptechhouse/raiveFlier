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

_CREATE_IMAGE_HASHES_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS image_hashes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL UNIQUE,
    image_phash TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""

_CREATE_ANALYSIS_SNAPSHOTS_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS analysis_snapshots (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id               TEXT    NOT NULL,
    flier_id                 INTEGER NOT NULL REFERENCES fliers(id),
    interconnection_map_json TEXT    NOT NULL,
    research_results_json    TEXT,
    schema_version           INTEGER NOT NULL DEFAULT 1,
    revision                 INTEGER NOT NULL DEFAULT 1,
    is_active                INTEGER NOT NULL DEFAULT 1,
    created_at               TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""

_CREATE_EDGE_DISMISSALS_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS edge_dismissals (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    flier_id          INTEGER NOT NULL REFERENCES fliers(id),
    source_entity     TEXT    NOT NULL,
    target_entity     TEXT    NOT NULL,
    relationship_type TEXT    NOT NULL,
    reason            TEXT,
    dismissed_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""

_CREATE_ANALYSIS_ANNOTATIONS_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS analysis_annotations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    flier_id    INTEGER NOT NULL REFERENCES fliers(id),
    target_type TEXT    NOT NULL DEFAULT 'analysis',
    target_key  TEXT,
    note        TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""

_CREATE_INDICES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_flier_artists_normalized ON flier_artists(artist_name_normalized);",
    "CREATE INDEX IF NOT EXISTS idx_flier_artists_flier ON flier_artists(flier_id);",
    "CREATE INDEX IF NOT EXISTS idx_fliers_session ON fliers(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_image_hashes_phash ON image_hashes(image_phash);",
    "CREATE INDEX IF NOT EXISTS idx_analysis_snapshots_session ON analysis_snapshots(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_analysis_snapshots_flier ON analysis_snapshots(flier_id);",
    "CREATE INDEX IF NOT EXISTS idx_analysis_snapshots_active ON analysis_snapshots(is_active);",
    "CREATE INDEX IF NOT EXISTS idx_edge_dismissals_flier ON edge_dismissals(flier_id);",
    "CREATE INDEX IF NOT EXISTS idx_analysis_annotations_flier ON analysis_annotations(flier_id);",
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
        """Create all tables and indices if they don't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(_CREATE_FLIERS_TABLE_SQL)
            await db.execute(_CREATE_FLIER_ARTISTS_TABLE_SQL)
            await db.execute(_CREATE_IMAGE_HASHES_TABLE_SQL)
            await db.execute(_CREATE_ANALYSIS_SNAPSHOTS_TABLE_SQL)
            await db.execute(_CREATE_EDGE_DISMISSALS_TABLE_SQL)
            await db.execute(_CREATE_ANALYSIS_ANNOTATIONS_TABLE_SQL)
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

    async def register_image_hash(
        self,
        session_id: str,
        image_phash: str,
    ) -> None:
        """Store a perceptual hash immediately on upload for duplicate detection."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(
                "INSERT OR IGNORE INTO image_hashes (session_id, image_phash) VALUES (?, ?);",
                (session_id, image_phash),
            )
            await db.commit()
        logger.debug("image_hash_registered", session_id=session_id, phash=image_phash[:8])

    @staticmethod
    def _hamming_distance(hash1: str, hash2: str) -> int:
        """Compute Hamming distance between two hex-encoded perceptual hashes."""
        n1 = int(hash1, 16)
        n2 = int(hash2, 16)
        return bin(n1 ^ n2).count("1")

    async def find_duplicate_by_phash(
        self,
        image_phash: str,
        threshold: int = 10,
    ) -> dict[str, Any] | None:
        """Check if a visually similar flier has been analyzed before.

        Fetches all stored perceptual hashes, computes Hamming distance against
        each, and returns metadata about the closest match if within threshold.
        """
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row

            # Fetch all stored hashes
            cursor = await db.execute(
                "SELECT session_id, image_phash, created_at FROM image_hashes;"
            )
            hash_rows = await cursor.fetchall()

        if not hash_rows:
            return None

        # Find the closest match by Hamming distance and count all
        # within-threshold matches (= how many times this flier was analyzed)
        best_match = None
        best_distance = threshold + 1
        times_analyzed = 0

        for row in hash_rows:
            stored_phash = row["image_phash"]
            if len(stored_phash) != len(image_phash):
                continue
            distance = self._hamming_distance(image_phash, stored_phash)
            if distance <= threshold:
                times_analyzed += 1
            if distance < best_distance:
                best_distance = distance
                best_match = dict(row)

        if best_match is None or best_distance > threshold:
            return None

        # Enrich with flier metadata (artists, venue, etc.) if available
        match_session_id = best_match["session_id"]
        artists: list[str] = []
        venue = None
        event_name = None
        event_date = None

        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute(
                "SELECT id, venue, event_name, event_date FROM fliers WHERE session_id = ?;",
                (match_session_id,),
            )
            flier_row = await cursor.fetchone()

            if flier_row is not None:
                venue = flier_row["venue"]
                event_name = flier_row["event_name"]
                event_date = flier_row["event_date"]

                cursor = await db.execute(
                    "SELECT artist_name FROM flier_artists WHERE flier_id = ?;",
                    (flier_row["id"],),
                )
                artist_rows = await cursor.fetchall()
                artists = [r["artist_name"] for r in artist_rows]

        # Similarity as 0.0–1.0 (64-bit hash: distance 0 = 1.0, distance 64 = 0.0)
        similarity = 1.0 - (best_distance / 64.0)

        logger.info(
            "duplicate_flier_detected",
            match_session_id=match_session_id,
            hamming_distance=best_distance,
            similarity=round(similarity, 3),
        )

        return {
            "session_id": match_session_id,
            "similarity": round(similarity, 3),
            "analyzed_at": best_match["created_at"],
            "artists": artists,
            "venue": venue,
            "event_name": event_name,
            "event_date": event_date,
            "hamming_distance": best_distance,
            "times_analyzed": times_analyzed,
        }

    async def get_flier_count(self) -> int:
        """Return the total number of logged fliers."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            cursor = await db.execute(_COUNT_FLIERS_SQL)
            row = await cursor.fetchone()
        return row[0] if row else 0

    # --- Helper: resolve session_id → flier_id ---

    async def _resolve_flier_id(self, session_id: str) -> int | None:
        """Look up the flier_id for a session_id.  Returns None if not found."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            cursor = await db.execute(
                "SELECT id FROM fliers WHERE session_id = ?;", (session_id,)
            )
            row = await cursor.fetchone()
        return row[0] if row else None

    # --- Persistent analysis storage ---

    async def store_analysis(
        self,
        session_id: str,
        interconnection_map: dict,
        research_results: list | None = None,
    ) -> dict:
        """Store a full InterconnectionMap snapshot for permanent retention."""
        flier_id = await self._resolve_flier_id(session_id)
        if flier_id is None:
            raise ValueError(f"No flier found for session: {session_id}")

        map_json = json.dumps(interconnection_map)
        research_json = json.dumps(research_results) if research_results else None

        async with aiosqlite.connect(str(self._db_path)) as db:
            # Compute next revision number.
            cursor = await db.execute(
                "SELECT COALESCE(MAX(revision), 0) FROM analysis_snapshots WHERE flier_id = ?;",
                (flier_id,),
            )
            max_rev_row = await cursor.fetchone()
            next_revision = (max_rev_row[0] if max_rev_row else 0) + 1

            # Deactivate previous snapshots.
            await db.execute(
                "UPDATE analysis_snapshots SET is_active = 0 WHERE flier_id = ?;",
                (flier_id,),
            )

            # Insert new snapshot.
            cursor = await db.execute(
                """INSERT INTO analysis_snapshots
                   (session_id, flier_id, interconnection_map_json, research_results_json, revision)
                   VALUES (?, ?, ?, ?, ?);""",
                (session_id, flier_id, map_json, research_json, next_revision),
            )
            snapshot_id = cursor.lastrowid
            await db.commit()

        logger.info("analysis_stored", session_id=session_id, flier_id=flier_id, revision=next_revision)
        return {
            "id": snapshot_id,
            "session_id": session_id,
            "flier_id": flier_id,
            "revision": next_revision,
        }

    async def get_analysis(
        self,
        session_id: str,
        include_research: bool = False,
    ) -> dict | None:
        """Retrieve the active analysis snapshot for a session."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT id, session_id, flier_id, interconnection_map_json,
                          research_results_json, schema_version, revision, created_at
                   FROM analysis_snapshots
                   WHERE session_id = ? AND is_active = 1
                   ORDER BY revision DESC LIMIT 1;""",
                (session_id,),
            )
            row = await cursor.fetchone()

        if row is None:
            return None

        result = dict(row)
        result["interconnection_map"] = json.loads(result.pop("interconnection_map_json"))
        rr_json = result.pop("research_results_json")
        if include_research and rr_json:
            result["research_results"] = json.loads(rr_json)
        else:
            result.pop("research_results_json", None)
        return result

    async def get_analysis_by_flier_id(
        self,
        flier_id: int,
        include_research: bool = False,
    ) -> dict | None:
        """Retrieve the active analysis snapshot by flier ID."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT id, session_id, flier_id, interconnection_map_json,
                          research_results_json, schema_version, revision, created_at
                   FROM analysis_snapshots
                   WHERE flier_id = ? AND is_active = 1
                   ORDER BY revision DESC LIMIT 1;""",
                (flier_id,),
            )
            row = await cursor.fetchone()

        if row is None:
            return None

        result = dict(row)
        result["interconnection_map"] = json.loads(result.pop("interconnection_map_json"))
        rr_json = result.pop("research_results_json")
        if include_research and rr_json:
            result["research_results"] = json.loads(rr_json)
        else:
            result.pop("research_results_json", None)
        return result

    async def list_analyses(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """List stored analyses with pagination (most recent first)."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT a.id, a.session_id, a.flier_id, a.revision, a.created_at,
                          f.venue, f.event_name, f.event_date
                   FROM analysis_snapshots a
                   JOIN fliers f ON a.flier_id = f.id
                   WHERE a.is_active = 1
                   ORDER BY a.created_at DESC
                   LIMIT ? OFFSET ?;""",
                (limit, offset),
            )
            rows = await cursor.fetchall()

        results = []
        for row in rows:
            r = dict(row)
            results.append(r)
        return results

    async def persist_edge_dismissal(
        self,
        session_id: str,
        source: str,
        target: str,
        relationship_type: str,
        reason: str | None = None,
    ) -> bool:
        """Permanently record an edge dismissal."""
        flier_id = await self._resolve_flier_id(session_id)
        if flier_id is None:
            return False

        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(
                """INSERT INTO edge_dismissals
                   (flier_id, source_entity, target_entity, relationship_type, reason)
                   VALUES (?, ?, ?, ?, ?);""",
                (flier_id, source, target, relationship_type, reason),
            )
            await db.commit()

        logger.info(
            "edge_dismissal_persisted",
            session_id=session_id,
            source=source,
            target=target,
        )
        return True

    async def get_edge_dismissals(
        self,
        session_id: str,
    ) -> list[dict]:
        """Get all edge dismissals for a session's flier."""
        flier_id = await self._resolve_flier_id(session_id)
        if flier_id is None:
            return []

        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT id, flier_id, source_entity, target_entity,
                          relationship_type, reason, dismissed_at
                   FROM edge_dismissals
                   WHERE flier_id = ?
                   ORDER BY dismissed_at DESC;""",
                (flier_id,),
            )
            rows = await cursor.fetchall()

        return [dict(row) for row in rows]

    async def add_annotation(
        self,
        session_id: str,
        note: str,
        target_type: str = "analysis",
        target_key: str | None = None,
    ) -> dict:
        """Add a user annotation to a stored analysis."""
        flier_id = await self._resolve_flier_id(session_id)
        if flier_id is None:
            raise ValueError(f"No flier found for session: {session_id}")

        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """INSERT INTO analysis_annotations
                   (flier_id, target_type, target_key, note)
                   VALUES (?, ?, ?, ?);""",
                (flier_id, target_type, target_key, note),
            )
            ann_id = cursor.lastrowid
            await db.commit()

            cursor = await db.execute(
                "SELECT * FROM analysis_annotations WHERE id = ?;", (ann_id,)
            )
            row = await cursor.fetchone()

        return dict(row)

    async def get_annotations(
        self,
        session_id: str,
    ) -> list[dict]:
        """Get all annotations for a session's analysis."""
        flier_id = await self._resolve_flier_id(session_id)
        if flier_id is None:
            return []

        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT id, flier_id, target_type, target_key, note, created_at, updated_at
                   FROM analysis_annotations
                   WHERE flier_id = ?
                   ORDER BY created_at DESC;""",
                (flier_id,),
            )
            rows = await cursor.fetchall()

        return [dict(row) for row in rows]

    async def get_all_active_analyses(self) -> list[dict]:
        """Get all active analysis snapshots for aggregation into a combined map.

        Each result includes a ``dismissals`` list containing any edge
        dismissals stored for that flier, so the aggregation layer can
        filter them out without a separate query.
        """
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT a.id, a.session_id, a.flier_id, a.interconnection_map_json,
                          a.revision, a.created_at,
                          f.venue, f.event_name, f.event_date
                   FROM analysis_snapshots a
                   JOIN fliers f ON a.flier_id = f.id
                   WHERE a.is_active = 1
                   ORDER BY a.created_at DESC;"""
            )
            rows = await cursor.fetchall()

            results = []
            for row in rows:
                r = dict(row)
                r["interconnection_map"] = json.loads(r.pop("interconnection_map_json"))

                # Fetch dismissals for this flier.
                d_cursor = await db.execute(
                    """SELECT id, flier_id, source_entity, target_entity,
                              relationship_type, reason, dismissed_at
                       FROM edge_dismissals
                       WHERE flier_id = ?;""",
                    (r["flier_id"],),
                )
                d_rows = await d_cursor.fetchall()
                r["dismissals"] = [dict(d) for d in d_rows]

                results.append(r)

        return results

    def get_provider_name(self) -> str:
        """Return a human-readable identifier for this provider."""
        return "sqlite_flier_history"

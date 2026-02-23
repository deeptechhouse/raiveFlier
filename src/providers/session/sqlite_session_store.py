"""SQLite-backed persistent session store.

Drop-in replacement for ``dict[str, PipelineState]`` that persists
pipeline session state to a SQLite database on disk.  Uses sync
``sqlite3`` for dict-like interface compatibility — each operation
touches a tiny JSON blob (<100 KB) so event-loop blocking is negligible.

An in-memory cache avoids repeated deserialisation for hot sessions.
Stale sessions (older than ``max_age_hours``) are pruned on
:meth:`initialize`.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import TypeVar

import structlog

from src.models.pipeline import PipelineState
from src.utils.logging import get_logger

_V = TypeVar("_V", bound=PipelineState)

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS {table} (
    session_id TEXT PRIMARY KEY,
    state_json TEXT    NOT NULL,
    created_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""

_CREATE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_{table}_updated ON {table}(updated_at);"
)

_UPSERT_SQL = """\
INSERT INTO {table} (session_id, state_json)
VALUES (?, ?)
ON CONFLICT(session_id)
DO UPDATE SET state_json = excluded.state_json,
              updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now');
"""

_SELECT_SQL = "SELECT state_json FROM {table} WHERE session_id = ?;"

_DELETE_SQL = "DELETE FROM {table} WHERE session_id = ?;"

_EXISTS_SQL = "SELECT 1 FROM {table} WHERE session_id = ? LIMIT 1;"

_ALL_IDS_SQL = "SELECT session_id FROM {table};"

_COUNT_SQL = "SELECT COUNT(*) FROM {table};"

_PRUNE_SQL = """\
DELETE FROM {table}
WHERE updated_at < strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-{hours} hours');
"""


class PersistentSessionStore(MutableMapping[str, PipelineState]):
    """Dict-like session store backed by SQLite.

    Implements :class:`~collections.abc.MutableMapping` so it can be used
    as a drop-in replacement for ``dict[str, PipelineState]`` throughout
    the application.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.
    table_name:
        Table name to use.  Allows multiple stores (e.g. ``sessions``
        and ``pending_sessions``) within the same database.
    max_age_hours:
        Sessions older than this are pruned on :meth:`initialize`.
        Set to ``0`` to disable pruning.
    """

    def __init__(
        self,
        db_path: str | Path,
        table_name: str = "sessions",
        max_age_hours: int = 72,
    ) -> None:
        self._db_path = Path(db_path)
        self._table = table_name
        self._max_age_hours = max_age_hours
        self._cache: dict[str, PipelineState] = {}
        self._logger: structlog.BoundLogger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create the table, index, and prune stale sessions.

        Must be called once before use (typically during app startup).
        """
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.execute(_CREATE_TABLE_SQL.format(table=self._table))
            conn.execute(_CREATE_INDEX_SQL.format(table=self._table))
            if self._max_age_hours > 0:
                cursor = conn.execute(
                    _PRUNE_SQL.format(table=self._table, hours=self._max_age_hours)
                )
                pruned = cursor.rowcount
                if pruned:
                    self._logger.info(
                        "sessions_pruned",
                        table=self._table,
                        pruned=pruned,
                        max_age_hours=self._max_age_hours,
                    )
            conn.commit()
        finally:
            conn.close()

        self._logger.info(
            "session_store_initialized",
            db_path=str(self._db_path),
            table=self._table,
            existing_sessions=len(self),
        )

    # ------------------------------------------------------------------
    # MutableMapping interface
    # ------------------------------------------------------------------

    def __setitem__(self, session_id: str, state: PipelineState) -> None:
        """Write to both in-memory cache and SQLite."""
        self._cache[session_id] = state
        state_json = state.model_dump_json()
        conn = self._connect()
        try:
            conn.execute(
                _UPSERT_SQL.format(table=self._table),
                (session_id, state_json),
            )
            conn.commit()
        finally:
            conn.close()

    def __getitem__(self, session_id: str) -> PipelineState:
        """Read from cache first, then SQLite.  Raises KeyError if missing."""
        if session_id in self._cache:
            return self._cache[session_id]

        state = self._load_from_db(session_id)
        if state is None:
            raise KeyError(session_id)

        self._cache[session_id] = state
        return state

    def __delitem__(self, session_id: str) -> None:
        """Remove from both cache and SQLite.  Raises KeyError if missing."""
        if session_id not in self:
            raise KeyError(session_id)
        self._cache.pop(session_id, None)
        conn = self._connect()
        try:
            conn.execute(
                _DELETE_SQL.format(table=self._table),
                (session_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def __contains__(self, session_id: object) -> bool:
        """Check existence in cache first, then SQLite."""
        if session_id in self._cache:
            return True
        if not isinstance(session_id, str):
            return False
        conn = self._connect()
        try:
            cursor = conn.execute(
                _EXISTS_SQL.format(table=self._table),
                (session_id,),
            )
            return cursor.fetchone() is not None
        finally:
            conn.close()

    def __iter__(self) -> Iterator[str]:
        """Iterate over all session IDs in the SQLite table."""
        conn = self._connect()
        try:
            cursor = conn.execute(_ALL_IDS_SQL.format(table=self._table))
            return iter([row[0] for row in cursor.fetchall()])
        finally:
            conn.close()

    def __len__(self) -> int:
        """Return the number of sessions in SQLite."""
        conn = self._connect()
        try:
            cursor = conn.execute(_COUNT_SQL.format(table=self._table))
            row = cursor.fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def pop(self, session_id: str, *args: PipelineState | None) -> PipelineState | None:
        """Remove and return a session.  Used by ConfirmationGate."""
        try:
            state = self[session_id]
            del self[session_id]
            return state
        except KeyError:
            if args:
                return args[0]
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open a new SQLite connection with WAL mode for concurrency."""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _load_from_db(self, session_id: str) -> PipelineState | None:
        """Deserialize a PipelineState from SQLite, or return None."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                _SELECT_SQL.format(table=self._table),
                (session_id,),
            )
            row = cursor.fetchone()
        finally:
            conn.close()

        if row is None:
            return None

        try:
            return PipelineState.model_validate_json(row[0])
        except Exception as exc:
            self._logger.warning(
                "session_deserialize_failed",
                session_id=session_id,
                error=str(exc)[:200],
            )
            return None

    def get_provider_name(self) -> str:
        """Return a human-readable identifier for this provider."""
        return f"sqlite_session_store:{self._table}"

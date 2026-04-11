"""SQLite-backed feedback provider.

Persists user thumbs up/down ratings to a local SQLite database at
``data/feedback.db``.  Uses ``aiosqlite`` for async I/O.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import aiosqlite
import structlog

from src.interfaces.feedback_provider import IFeedbackProvider

logger = structlog.get_logger(logger_name=__name__)

_DEFAULT_DB_PATH = Path("data/feedback.db")

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS ratings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    item_type   TEXT    NOT NULL,
    item_key    TEXT    NOT NULL,
    rating      INTEGER NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(session_id, item_type, item_key)
);
"""

_CREATE_INDICES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_ratings_session ON ratings(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_ratings_type ON ratings(item_type);",
    "CREATE INDEX IF NOT EXISTS idx_ratings_type_key ON ratings(item_type, item_key);",
]

_UPSERT_SQL = """\
INSERT INTO ratings (session_id, item_type, item_key, rating)
VALUES (?, ?, ?, ?)
ON CONFLICT(session_id, item_type, item_key)
DO UPDATE SET rating     = excluded.rating,
              updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now');
"""

_SELECT_LAST_SQL = """\
SELECT id, session_id, item_type, item_key, rating, created_at, updated_at
FROM ratings
WHERE session_id = ? AND item_type = ? AND item_key = ?;
"""

_SELECT_NEGATIVE_KEYS_SQL = """\
SELECT item_key
FROM ratings
WHERE item_type = ? AND item_key LIKE ?
GROUP BY item_key
HAVING SUM(rating) < 0;
"""

# -- Calibration sample table (Optimization I) ----------------------------
# Stores (predicted_score, ground_truth) observations collected from the
# entity confirmation gate.  The ConfidenceCalibrator reads these on
# startup to learn how raw LLM/fuzzy-match scores map to real accuracy.

_CREATE_CALIBRATION_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS calibration_samples (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    score_type      TEXT    NOT NULL,
    predicted_score REAL    NOT NULL,
    ground_truth    INTEGER NOT NULL,
    entity_type     TEXT,
    session_id      TEXT,
    created_at      TEXT    DEFAULT CURRENT_TIMESTAMP
);
"""

_CREATE_CALIBRATION_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_calibration_score_type "
    "ON calibration_samples(score_type);"
)

_INSERT_CALIBRATION_SQL = """\
INSERT INTO calibration_samples
    (score_type, predicted_score, ground_truth, entity_type, session_id)
VALUES (?, ?, ?, ?, ?);
"""

_SELECT_CALIBRATION_SQL = """\
SELECT predicted_score, ground_truth
FROM calibration_samples
WHERE score_type = ?;
"""


class SQLiteFeedbackProvider(IFeedbackProvider):
    """SQLite-backed feedback persistence."""

    def __init__(self, db_path: str | Path = _DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)

    async def initialize(self) -> None:
        """Create the ratings and calibration_samples tables if they don't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(_CREATE_TABLE_SQL)
            for idx_sql in _CREATE_INDICES_SQL:
                await db.execute(idx_sql)
            # Calibration samples table stores (predicted_score, ground_truth)
            # pairs used to train the ConfidenceCalibrator (Optimization I).
            await db.execute(_CREATE_CALIBRATION_TABLE_SQL)
            await db.execute(_CREATE_CALIBRATION_INDEX_SQL)
            await db.commit()
        logger.info("feedback_db_initialized", path=str(self._db_path))

    async def submit_rating(
        self,
        session_id: str,
        item_type: str,
        item_key: str,
        rating: int,
    ) -> dict[str, Any]:
        """Store or update a rating.  Returns the upserted row."""
        if rating not in (1, -1):
            msg = f"Rating must be +1 or -1, got {rating}"
            raise ValueError(msg)

        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            await db.execute(
                _UPSERT_SQL,
                (session_id, item_type.upper(), item_key, rating),
            )
            await db.commit()
            cursor = await db.execute(
                _SELECT_LAST_SQL,
                (session_id, item_type.upper(), item_key),
            )
            row = await cursor.fetchone()

        result = dict(row)
        logger.info(
            "rating_submitted",
            session_id=session_id,
            item_type=item_type,
            item_key=item_key,
            rating=rating,
        )
        return result

    async def get_ratings(self, session_id: str) -> list[dict[str, Any]]:
        """Return all ratings for a session, newest first."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT id, session_id, item_type, item_key, rating, created_at, updated_at "
                "FROM ratings WHERE session_id = ? ORDER BY updated_at DESC",
                (session_id,),
            )
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_rating_summary(
        self,
        item_type: str | None = None,
    ) -> dict[str, Any]:
        """Return aggregate rating statistics across all sessions."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            if item_type:
                cursor = await db.execute(
                    "SELECT item_type, "
                    "COUNT(*) as total, "
                    "SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) as positive, "
                    "SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) as negative "
                    "FROM ratings WHERE item_type = ? GROUP BY item_type",
                    (item_type.upper(),),
                )
            else:
                cursor = await db.execute(
                    "SELECT item_type, "
                    "COUNT(*) as total, "
                    "SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) as positive, "
                    "SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) as negative "
                    "FROM ratings GROUP BY item_type",
                )
            rows = await cursor.fetchall()

        by_type: dict[str, dict[str, int]] = {}
        total = positive = negative = 0
        for row in rows:
            r = dict(row)
            by_type[r["item_type"]] = {
                "total": r["total"],
                "positive": r["positive"],
                "negative": r["negative"],
            }
            total += r["total"]
            positive += r["positive"]
            negative += r["negative"]

        return {
            "total_ratings": total,
            "positive": positive,
            "negative": negative,
            "by_type": by_type,
        }

    async def get_negative_item_keys(
        self,
        item_type: str,
        item_key_prefix: str,
    ) -> set[str]:
        """Return item_keys with net-negative ratings across all sessions."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                _SELECT_NEGATIVE_KEYS_SQL,
                (item_type.upper(), item_key_prefix + "%"),
            )
            rows = await cursor.fetchall()
        return {row["item_key"] for row in rows}

    # -- Calibration data methods (Optimization I) -----------------------------
    # These persist the ground-truth observations that the ConfidenceCalibrator
    # uses to learn the mapping from raw LLM/fuzzy-match scores to empirical
    # accuracy.  Called by ConfirmationGate after each user confirmation.

    async def store_calibration_sample(
        self,
        score_type: str,
        predicted_score: float,
        ground_truth: int,
        entity_type: str,
        session_id: str,
    ) -> None:
        """Persist a single calibration observation."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(
                _INSERT_CALIBRATION_SQL,
                (score_type, predicted_score, ground_truth, entity_type, session_id),
            )
            await db.commit()
        logger.debug(
            "calibration_sample_stored",
            score_type=score_type,
            predicted_score=predicted_score,
            ground_truth=ground_truth,
            entity_type=entity_type,
            session_id=session_id,
        )

    async def get_calibration_data(self, score_type: str) -> list[dict]:
        """Load all calibration observations for the given score type."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(_SELECT_CALIBRATION_SQL, (score_type,))
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    def get_provider_name(self) -> str:
        """Return a human-readable identifier for this provider."""
        return "sqlite_feedback"

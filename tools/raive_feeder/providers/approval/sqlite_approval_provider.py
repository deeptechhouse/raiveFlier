"""SQLite-backed content approval queue provider.

# ─── ARCHITECTURE ────────────────────────────────────────────────────
#
# Follows the exact same pattern as src/providers/feedback/sqlite_feedback_provider.py:
#   - aiosqlite for async I/O
#   - WAL mode for concurrent reads during writes
#   - Parameterized queries throughout (no string interpolation in SQL)
#   - idempotent initialize() with CREATE TABLE IF NOT EXISTS
#
# The pending_submissions table stores all content submitted through
# raiveFeeder's ingestion tabs.  Content sits in "pending" status until
# an admin approves or rejects it via the approval dashboard.
#
# File uploads are staged on disk at /data/pending_uploads/{id}_{filename}
# and referenced by path in content_data.  Text and URL submissions store
# their content directly in the content_data column.
#
# Layer: Providers (implements IApprovalProvider interface)
# Depends on: aiosqlite, structlog
# ─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite
import structlog

from tools.raive_feeder.interfaces.approval_provider import IApprovalProvider

logger = structlog.get_logger(logger_name=__name__)

_DEFAULT_DB_PATH = Path("data/feeder_approvals.db")

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS pending_submissions (
    id                  TEXT PRIMARY KEY,
    submitted_at        TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'pending',
    decided_at          TEXT,
    decision_reason     TEXT DEFAULT '',
    title               TEXT NOT NULL,
    source_type         TEXT NOT NULL DEFAULT 'article',
    citation_tier       INTEGER NOT NULL DEFAULT 3,
    author              TEXT DEFAULT '',
    year                INTEGER DEFAULT 0,
    content_type        TEXT NOT NULL,
    content_data        TEXT NOT NULL,
    chunk_count_est     INTEGER DEFAULT 0,
    github_issue_number INTEGER,
    github_issue_url    TEXT
);
"""

_CREATE_INDICES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_pending_status ON pending_submissions(status);",
    "CREATE INDEX IF NOT EXISTS idx_pending_submitted ON pending_submissions(submitted_at);",
]


class SQLiteApprovalProvider(IApprovalProvider):
    """SQLite-backed content approval queue.

    Stores pending submissions on the persistent /data disk so they survive
    container restarts on Render.
    """

    def __init__(self, db_path: str | Path = _DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)

    async def initialize(self) -> None:
        """Create the pending_submissions table and indices (idempotent)."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(self._db_path)) as db:
            # WAL mode allows concurrent reads during writes.
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.execute(_CREATE_TABLE_SQL)
            for idx_sql in _CREATE_INDICES_SQL:
                await db.execute(idx_sql)
            await db.commit()
        logger.info("approval_db_initialized", path=str(self._db_path))

    async def submit(self, submission: dict[str, Any]) -> str:
        """Insert a new pending submission.  Returns the submission ID."""
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(
                "INSERT INTO pending_submissions "
                "(id, submitted_at, status, title, source_type, citation_tier, "
                "author, year, content_type, content_data, chunk_count_est) "
                "VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    submission["id"],
                    now,
                    submission.get("title", "Untitled"),
                    submission.get("source_type", "article"),
                    submission.get("citation_tier", 3),
                    submission.get("author", ""),
                    submission.get("year", 0),
                    submission["content_type"],
                    submission["content_data"],
                    submission.get("chunk_count_est", 0),
                ),
            )
            await db.commit()

        logger.info(
            "submission_queued",
            submission_id=submission["id"],
            title=submission.get("title"),
            content_type=submission["content_type"],
        )
        return submission["id"]

    async def list_pending(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """Return pending submissions, newest first."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM pending_submissions "
                "WHERE status = 'pending' "
                "ORDER BY submitted_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_by_id(self, submission_id: str) -> dict[str, Any] | None:
        """Return a single submission by ID."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM pending_submissions WHERE id = ?",
                (submission_id,),
            )
            row = await cursor.fetchone()
        return dict(row) if row else None

    async def approve(self, submission_id: str) -> bool:
        """Mark a submission as approved."""
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(str(self._db_path)) as db:
            cursor = await db.execute(
                "UPDATE pending_submissions SET status = 'approved', decided_at = ? "
                "WHERE id = ? AND status = 'pending'",
                (now, submission_id),
            )
            await db.commit()
            updated = cursor.rowcount > 0
        if updated:
            logger.info("submission_approved", submission_id=submission_id)
        return updated

    async def reject(self, submission_id: str, reason: str = "") -> bool:
        """Mark a submission as rejected."""
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(str(self._db_path)) as db:
            cursor = await db.execute(
                "UPDATE pending_submissions SET status = 'rejected', decided_at = ?, "
                "decision_reason = ? WHERE id = ? AND status = 'pending'",
                (now, reason, submission_id),
            )
            await db.commit()
            updated = cursor.rowcount > 0
        if updated:
            logger.info("submission_rejected", submission_id=submission_id, reason=reason)
        return updated

    async def bulk_approve(self, submission_ids: list[str]) -> int:
        """Approve multiple submissions at once."""
        if not submission_ids:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        placeholders = ",".join("?" for _ in submission_ids)
        async with aiosqlite.connect(str(self._db_path)) as db:
            cursor = await db.execute(
                f"UPDATE pending_submissions SET status = 'approved', decided_at = ? "
                f"WHERE id IN ({placeholders}) AND status = 'pending'",
                [now, *submission_ids],
            )
            await db.commit()
            count = cursor.rowcount
        logger.info("bulk_approved", count=count, ids=submission_ids)
        return count

    async def bulk_reject(self, submission_ids: list[str], reason: str = "") -> int:
        """Reject multiple submissions at once."""
        if not submission_ids:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        placeholders = ",".join("?" for _ in submission_ids)
        async with aiosqlite.connect(str(self._db_path)) as db:
            cursor = await db.execute(
                f"UPDATE pending_submissions SET status = 'rejected', decided_at = ?, "
                f"decision_reason = ? WHERE id IN ({placeholders}) AND status = 'pending'",
                [now, reason, *submission_ids],
            )
            await db.commit()
            count = cursor.rowcount
        logger.info("bulk_rejected", count=count, ids=submission_ids, reason=reason)
        return count

    async def get_stats(self) -> dict[str, int]:
        """Return submission counts by status."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT status, COUNT(*) as cnt FROM pending_submissions GROUP BY status"
            )
            rows = await cursor.fetchall()

        stats = {"pending": 0, "approved": 0, "rejected": 0, "total": 0}
        for row in rows:
            status = row["status"]
            count = row["cnt"]
            if status in stats:
                stats[status] = count
            stats["total"] += count
        return stats

    async def update_github_issue(
        self, submission_id: str, issue_number: int, issue_url: str
    ) -> bool:
        """Attach GitHub Issue metadata to a submission."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            cursor = await db.execute(
                "UPDATE pending_submissions "
                "SET github_issue_number = ?, github_issue_url = ? "
                "WHERE id = ?",
                (issue_number, issue_url, submission_id),
            )
            await db.commit()
            return cursor.rowcount > 0

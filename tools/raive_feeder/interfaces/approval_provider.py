"""Abstract interface for the content approval queue provider.

# ─── DESIGN RATIONALE ────────────────────────────────────────────────
#
# This interface defines the contract for persisting content approval
# decisions.  Following the adapter pattern used throughout raiveFlier,
# the concrete implementation (SQLite) can be swapped for any other
# storage backend (PostgreSQL, DynamoDB, etc.) by implementing this ABC.
#
# The approval queue holds submissions in a "pending" state until an
# admin approves or rejects them.  Only approved submissions trigger
# actual corpus ingestion into ChromaDB.
#
# Layer: Interfaces (depended on by Services and Providers)
# ─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class IApprovalProvider(ABC):
    """Abstract base class for content approval queue storage."""

    @abstractmethod
    async def initialize(self) -> None:
        """Create storage schema if it doesn't exist (idempotent)."""

    @abstractmethod
    async def submit(self, submission: dict[str, Any]) -> str:
        """Store a new pending submission.  Returns the submission ID."""

    @abstractmethod
    async def list_pending(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """Return pending submissions, newest first."""

    @abstractmethod
    async def get_by_id(self, submission_id: str) -> dict[str, Any] | None:
        """Return a single submission by ID, or None if not found."""

    @abstractmethod
    async def approve(self, submission_id: str) -> bool:
        """Mark a submission as approved.  Returns True if found and updated."""

    @abstractmethod
    async def reject(self, submission_id: str, reason: str = "") -> bool:
        """Mark a submission as rejected.  Returns True if found and updated."""

    @abstractmethod
    async def bulk_approve(self, submission_ids: list[str]) -> int:
        """Approve multiple submissions.  Returns count of updated rows."""

    @abstractmethod
    async def bulk_reject(self, submission_ids: list[str], reason: str = "") -> int:
        """Reject multiple submissions.  Returns count of updated rows."""

    @abstractmethod
    async def get_stats(self) -> dict[str, int]:
        """Return counts by status: {pending, approved, rejected, total}."""

    @abstractmethod
    async def update_github_issue(
        self, submission_id: str, issue_number: int, issue_url: str
    ) -> bool:
        """Attach GitHub Issue metadata to a submission."""

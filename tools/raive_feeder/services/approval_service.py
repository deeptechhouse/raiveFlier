"""Content approval service — orchestrates the submission-to-ingestion lifecycle.

# ─── ARCHITECTURE ────────────────────────────────────────────────────
#
# ApprovalService sits between the API routes and the storage/ingestion
# layers.  It coordinates:
#
#   1. submit()  → stores content as pending, creates GitHub Issue
#   2. approve() → triggers actual ingestion into ChromaDB, closes Issue
#   3. reject()  → deletes staged content, closes Issue with reason
#
# Dependencies (all injected, never created internally):
#   - IApprovalProvider  — storage for pending submissions
#   - IngestionService   — ChromaDB ingestion (used only on approve)
#   - GitHubIssueService — optional Issue notifications (graceful if absent)
#
# Layer: Services (depends on Interfaces, used by API routes)
# ─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from tools.raive_feeder.interfaces.approval_provider import IApprovalProvider

logger = structlog.get_logger(logger_name=__name__)


class ApprovalService:
    """Orchestrates content submission, approval, and rejection.

    All external dependencies are injected — no globals, no service locator.
    """

    def __init__(
        self,
        *,
        approval_provider: IApprovalProvider,
        ingestion_service: Any | None = None,
        github_issue_service: Any | None = None,
    ) -> None:
        self._provider = approval_provider
        self._ingestion = ingestion_service
        self._github = github_issue_service

    async def submit(self, submission: dict[str, Any]) -> dict[str, Any]:
        """Queue a new content submission for admin review.

        Parameters
        ----------
        submission:
            Must contain at minimum: id, title, content_type, content_data.
            Optional: source_type, citation_tier, author, year, chunk_count_est.

        Returns
        -------
        dict with submission_id and status.
        """
        submission_id = await self._provider.submit(submission)

        # Create GitHub Issue for notification (non-blocking, graceful failure).
        if self._github is not None:
            try:
                result = await self._github.create_submission_issue(submission)
                if result is not None:
                    issue_number, issue_url = result
                    await self._provider.update_github_issue(
                        submission_id, issue_number, issue_url
                    )
            except Exception as exc:
                logger.warning(
                    "github_issue_creation_failed",
                    submission_id=submission_id,
                    error=str(exc),
                )

        return {"submission_id": submission_id, "status": "pending"}

    async def approve(self, submission_id: str) -> dict[str, Any]:
        """Approve a submission and trigger actual corpus ingestion.

        Returns
        -------
        dict with approval result and ingestion details.
        """
        submission = await self._provider.get_by_id(submission_id)
        if submission is None:
            return {"error": "Submission not found", "submission_id": submission_id}

        if submission["status"] != "pending":
            return {"error": f"Submission already {submission['status']}", "submission_id": submission_id}

        # Mark as approved in the queue.
        updated = await self._provider.approve(submission_id)
        if not updated:
            return {"error": "Failed to update submission", "submission_id": submission_id}

        # Trigger ingestion into ChromaDB.
        ingestion_result = await self._ingest_submission(submission)

        # Close the GitHub Issue with approval comment.
        await self._close_github_issue(submission, "approved", "")

        return {
            "submission_id": submission_id,
            "status": "approved",
            "ingestion": ingestion_result,
        }

    async def reject(self, submission_id: str, reason: str = "") -> dict[str, Any]:
        """Reject a submission and clean up staged content."""
        submission = await self._provider.get_by_id(submission_id)
        if submission is None:
            return {"error": "Submission not found", "submission_id": submission_id}

        if submission["status"] != "pending":
            return {"error": f"Submission already {submission['status']}", "submission_id": submission_id}

        updated = await self._provider.reject(submission_id, reason)
        if not updated:
            return {"error": "Failed to update submission", "submission_id": submission_id}

        # Clean up staged file if it was a file upload.
        if submission["content_type"] == "file_path":
            staged_path = Path(submission["content_data"])
            if staged_path.exists():
                staged_path.unlink(missing_ok=True)
                logger.info("staged_file_deleted", path=str(staged_path))

        # Close the GitHub Issue with rejection comment.
        await self._close_github_issue(submission, "rejected", reason)

        return {"submission_id": submission_id, "status": "rejected"}

    async def bulk_approve(self, submission_ids: list[str]) -> dict[str, Any]:
        """Approve multiple submissions and trigger ingestion for each."""
        results = []
        for sid in submission_ids:
            result = await self.approve(sid)
            results.append(result)
        approved_count = sum(1 for r in results if r.get("status") == "approved")
        return {"approved": approved_count, "total": len(submission_ids), "results": results}

    async def bulk_reject(self, submission_ids: list[str], reason: str = "") -> dict[str, Any]:
        """Reject multiple submissions."""
        results = []
        for sid in submission_ids:
            result = await self.reject(sid, reason)
            results.append(result)
        rejected_count = sum(1 for r in results if r.get("status") == "rejected")
        return {"rejected": rejected_count, "total": len(submission_ids), "results": results}

    async def list_pending(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """Return pending submissions for the dashboard."""
        return await self._provider.list_pending(limit, offset)

    async def get_stats(self) -> dict[str, int]:
        """Return approval queue statistics."""
        return await self._provider.get_stats()

    async def get_by_id(self, submission_id: str) -> dict[str, Any] | None:
        """Return a single submission."""
        return await self._provider.get_by_id(submission_id)

    # ─── Private helpers ─────────────────────────────────────────────

    async def _ingest_submission(self, submission: dict[str, Any]) -> dict[str, Any]:
        """Trigger actual corpus ingestion for an approved submission.

        Routes to the appropriate IngestionService method based on content_type.
        """
        if self._ingestion is None:
            return {"error": "Ingestion service unavailable"}

        content_type = submission["content_type"]
        title = submission.get("title", "Untitled")
        author = submission.get("author", "")
        year = submission.get("year", 0)

        try:
            if content_type == "url":
                # Scrape and ingest the URL.
                result = await self._ingestion.ingest_article(url=submission["content_data"])
                return {
                    "source_id": result.source_id,
                    "chunks_created": result.chunks_created,
                }

            elif content_type == "text":
                # Ingest raw text as a book/document.
                result = await self._ingestion.ingest_text(
                    text=submission["content_data"],
                    title=title,
                    author=author,
                    year=year,
                    source_type=submission.get("source_type", "article"),
                )
                return {
                    "source_id": result.source_id,
                    "chunks_created": result.chunks_created,
                }

            elif content_type == "file_path":
                # Ingest the staged file.
                file_path = submission["content_data"]
                suffix = Path(file_path).suffix.lower()
                if suffix == ".pdf":
                    result = await self._ingestion.ingest_pdf(
                        file_path=file_path, title=title, author=author, year=year
                    )
                elif suffix == ".epub":
                    result = await self._ingestion.ingest_epub(
                        file_path=file_path, title=title, author=author, year=year
                    )
                else:
                    result = await self._ingestion.ingest_book(
                        file_path=file_path, title=title, author=author, year=year
                    )
                return {
                    "source_id": result.source_id,
                    "chunks_created": result.chunks_created,
                }

            else:
                return {"error": f"Unknown content_type: {content_type}"}

        except Exception as exc:
            logger.error(
                "ingestion_after_approval_failed",
                submission_id=submission["id"],
                error=str(exc),
            )
            return {"error": str(exc)}

    async def _close_github_issue(
        self, submission: dict[str, Any], decision: str, reason: str
    ) -> None:
        """Close the GitHub Issue associated with a submission."""
        if self._github is None:
            return

        issue_number = submission.get("github_issue_number")
        if not issue_number:
            return

        try:
            await self._github.close_issue_with_comment(issue_number, decision, reason)
        except Exception as exc:
            logger.warning(
                "github_issue_close_failed",
                issue_number=issue_number,
                error=str(exc),
            )

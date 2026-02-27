"""Pydantic v2 schemas for the content approval queue API.

# ─── SCHEMA DESIGN ───────────────────────────────────────────────────
#
# Follows the same frozen=True immutability pattern as schemas.py.
# These models shape the JSON exchanged between the approval dashboard
# frontend and the approval API endpoints.
#
# Naming:
#   PendingSubmission  — a single queued submission (read)
#   SubmissionResponse — confirmation after submitting content
#   ApprovalDecision   — approve/reject action (write)
#   BulkDecisionRequest — batch approve/reject (write)
#   ApprovalStats      — dashboard summary counts (read)
# ─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PendingSubmission(BaseModel):
    """A single content submission awaiting admin review."""
    model_config = ConfigDict(frozen=True)

    id: str
    submitted_at: str
    status: str = Field(default="pending")
    decided_at: str | None = None
    decision_reason: str = Field(default="")
    title: str
    source_type: str = Field(default="article")
    citation_tier: int = Field(default=3, ge=1, le=6)
    author: str = Field(default="")
    year: int = Field(default=0)
    content_type: str = Field(description="'text', 'url', or 'file_path'")
    content_data: str = Field(description="Actual text, URL, or path to staged file")
    chunk_count_est: int = Field(default=0)
    github_issue_number: int | None = None
    github_issue_url: str | None = None


class SubmissionResponse(BaseModel):
    """Confirmation returned after content is submitted to the approval queue."""
    model_config = ConfigDict(frozen=True)

    submission_id: str
    status: str = Field(default="pending")
    message: str = Field(default="Submission queued for admin approval")


class ApprovalDecision(BaseModel):
    """Request body for approving or rejecting a single submission."""
    model_config = ConfigDict(frozen=True)

    reason: str = Field(default="", description="Optional reason for rejection")


class BulkDecisionRequest(BaseModel):
    """Request body for bulk approve/reject operations."""
    model_config = ConfigDict(frozen=True)

    ids: list[str] = Field(description="Submission IDs to approve or reject")
    reason: str = Field(default="", description="Rejection reason (only for reject)")


class ApprovalStats(BaseModel):
    """Summary counts for the approval dashboard."""
    model_config = ConfigDict(frozen=True)

    pending: int = Field(default=0)
    approved: int = Field(default=0)
    rejected: int = Field(default=0)
    total: int = Field(default=0)

"""Approval queue API endpoints for raiveFeeder.

# ─── ROUTE ARCHITECTURE ─────────────────────────────────────────────
#
# These endpoints power the admin approval dashboard in the Corpus tab.
# All routes delegate to ApprovalService (stored on app.state) — the
# route handlers are thin wrappers that validate input and format output.
#
# Endpoints:
#   GET  /approval/pending         — List pending submissions
#   GET  /approval/stats           — Dashboard summary counts
#   GET  /approval/{id}            — Get submission details
#   POST /approval/{id}/approve    — Approve and trigger ingestion
#   POST /approval/{id}/reject     — Reject and clean up
#   POST /approval/bulk-approve    — Batch approve
#   POST /approval/bulk-reject     — Batch reject
#
# Layer: API (depends on Services via app.state)
# ─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from tools.raive_feeder.api.approval_schemas import (
    ApprovalDecision,
    ApprovalStats,
    BulkDecisionRequest,
    PendingSubmission,
)

approval_router = APIRouter(prefix="/approval", tags=["approval"])


def _get_approval_service(request: Request):
    """Retrieve ApprovalService from app.state."""
    svc = getattr(request.app.state, "approval_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Approval service unavailable")
    return svc


@approval_router.get("/pending", response_model=list[PendingSubmission])
async def list_pending(
    request: Request,
    limit: int = 50,
    offset: int = 0,
) -> list[PendingSubmission]:
    """List pending submissions, newest first."""
    svc = _get_approval_service(request)
    rows = await svc.list_pending(limit=limit, offset=offset)
    return [PendingSubmission(**r) for r in rows]


@approval_router.get("/stats", response_model=ApprovalStats)
async def approval_stats(request: Request) -> ApprovalStats:
    """Return approval queue summary counts for the dashboard."""
    svc = _get_approval_service(request)
    stats = await svc.get_stats()
    return ApprovalStats(**stats)


@approval_router.get("/{submission_id}", response_model=PendingSubmission)
async def get_submission(request: Request, submission_id: str) -> PendingSubmission:
    """Get details for a specific submission."""
    svc = _get_approval_service(request)
    row = await svc.get_by_id(submission_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Submission not found")
    return PendingSubmission(**row)


@approval_router.post("/{submission_id}/approve")
async def approve_submission(
    request: Request, submission_id: str
) -> dict:
    """Approve a pending submission and trigger corpus ingestion."""
    svc = _get_approval_service(request)
    result = await svc.approve(submission_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@approval_router.post("/{submission_id}/reject")
async def reject_submission(
    request: Request, submission_id: str, body: ApprovalDecision | None = None
) -> dict:
    """Reject a pending submission."""
    svc = _get_approval_service(request)
    reason = body.reason if body else ""
    result = await svc.reject(submission_id, reason)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@approval_router.post("/bulk-approve")
async def bulk_approve(request: Request, body: BulkDecisionRequest) -> dict:
    """Approve multiple submissions at once."""
    svc = _get_approval_service(request)
    return await svc.bulk_approve(body.ids)


@approval_router.post("/bulk-reject")
async def bulk_reject(request: Request, body: BulkDecisionRequest) -> dict:
    """Reject multiple submissions at once."""
    svc = _get_approval_service(request)
    return await svc.bulk_reject(body.ids, body.reason)

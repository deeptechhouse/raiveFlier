"""Batch processing service for queued ingestion jobs.

# ─── DESIGN ────────────────────────────────────────────────────────────
#
# BatchProcessor manages a queue of ingestion jobs with:
#   - Sequential or configurable concurrency (asyncio.Semaphore)
#   - Per-item progress tracking via callbacks
#   - Error handling per-item (batch continues on failure)
#   - Pause/resume/cancel support
#
# Each job is identified by a UUID and tracked in an in-memory dict.
# WebSocket clients subscribe to progress updates via the job_id.
#
# Pattern: Observer (progress callbacks) + Strategy (ingestion method
# selection based on item type).
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable
from uuid import uuid4

import structlog

from tools.raive_feeder.api.schemas import JobStatus, JobStatusResponse

logger = structlog.get_logger(logger_name=__name__)

# Type alias for progress callback: (job_id, item_name, progress_pct, message)
ProgressCallback = Callable[[str, str, float, str], Awaitable[None]]


class _JobState:
    """Internal mutable state for a single batch job.

    Not exposed outside BatchProcessor — encapsulation enforced via
    the JobStatusResponse frozen model for external consumers.
    """

    def __init__(self, job_id: str, items: list[dict[str, Any]]) -> None:
        self._job_id = job_id
        self._items = items
        self._status = JobStatus.QUEUED
        self._completed = 0
        self._failed = 0
        self._current_item: str | None = None
        self._errors: list[str] = []
        self._created_at = datetime.now(timezone.utc)
        self._cancel_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused initially.
        self._task: asyncio.Task[None] | None = None
        self._listeners: list[ProgressCallback] = []

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def status(self) -> JobStatus:
        return self._status

    def to_response(self) -> JobStatusResponse:
        """Build an immutable status snapshot for the API."""
        return JobStatusResponse(
            job_id=self._job_id,
            status=self._status,
            total_items=len(self._items),
            completed_items=self._completed,
            failed_items=self._failed,
            current_item=self._current_item,
            errors=list(self._errors),
        )


class BatchProcessor:
    """Manages queued batch ingestion jobs with progress tracking.

    Parameters
    ----------
    ingestion_service:
        The shared IngestionService instance for processing documents.
    max_concurrency:
        Maximum number of items processed concurrently within a single job.
    """

    def __init__(
        self,
        ingestion_service: Any | None,
        max_concurrency: int = 2,
    ) -> None:
        self._ingestion_service = ingestion_service
        self._max_concurrency = max(1, max_concurrency)
        self._jobs: dict[str, _JobState] = {}

    # ─── Job lifecycle ─────────────────────────────────────────────────

    def create_job(self, items: list[dict[str, Any]]) -> str:
        """Create a new batch job and return its ID.

        The job is queued but not started — call :meth:`start_job` to begin.
        """
        job_id = str(uuid4())
        state = _JobState(job_id=job_id, items=items)
        self._jobs[job_id] = state
        logger.info("batch_job_created", job_id=job_id, items=len(items))
        return job_id

    async def start_job(
        self,
        job_id: str,
        process_fn: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    ) -> None:
        """Start executing a batch job in the background.

        Parameters
        ----------
        job_id:
            The job to start.
        process_fn:
            Async function that processes a single item dict and returns
            a result dict.  Called once per item in the batch.
        """
        state = self._jobs.get(job_id)
        if state is None:
            raise ValueError(f"Unknown job: {job_id}")

        state._status = JobStatus.RUNNING
        state._task = asyncio.create_task(
            self._run_job(state, process_fn)
        )

    async def _run_job(
        self,
        state: _JobState,
        process_fn: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    ) -> None:
        """Execute all items in a batch job sequentially."""
        semaphore = asyncio.Semaphore(self._max_concurrency)

        for idx, item in enumerate(state._items):
            # Check for cancellation.
            if state._cancel_event.is_set():
                state._status = JobStatus.CANCELLED
                logger.info("batch_job_cancelled", job_id=state.job_id)
                return

            # Wait if paused.
            await state._pause_event.wait()

            item_name = item.get("filename", item.get("url", f"item_{idx}"))
            state._current_item = item_name

            # Notify listeners of progress.
            progress = (idx / len(state._items)) * 100
            await self._notify(state, item_name, progress, f"Processing {item_name}")

            async with semaphore:
                try:
                    await process_fn(item)
                    state._completed += 1
                except Exception as exc:
                    state._failed += 1
                    error_msg = f"{item_name}: {exc}"
                    state._errors.append(error_msg)
                    logger.warning(
                        "batch_item_failed",
                        job_id=state.job_id,
                        item=item_name,
                        error=str(exc),
                    )

        # Final status.
        state._current_item = None
        if state._failed == len(state._items):
            state._status = JobStatus.FAILED
        else:
            state._status = JobStatus.COMPLETED

        await self._notify(state, "", 100.0, "Batch complete")
        logger.info(
            "batch_job_finished",
            job_id=state.job_id,
            completed=state._completed,
            failed=state._failed,
        )

    # ─── Job control ───────────────────────────────────────────────────

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job. Returns True if the job was found."""
        state = self._jobs.get(job_id)
        if state is None:
            return False
        state._cancel_event.set()
        state._pause_event.set()  # Unpause so it can exit.
        return True

    def pause_job(self, job_id: str) -> bool:
        """Pause a running job. Returns True if the job was found."""
        state = self._jobs.get(job_id)
        if state is None:
            return False
        state._pause_event.clear()
        state._status = JobStatus.PAUSED
        return True

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job. Returns True if the job was found."""
        state = self._jobs.get(job_id)
        if state is None:
            return False
        state._pause_event.set()
        state._status = JobStatus.RUNNING
        return True

    # ─── Job queries ───────────────────────────────────────────────────

    def get_job_status(self, job_id: str) -> JobStatusResponse | None:
        """Get the current status of a job, or None if not found."""
        state = self._jobs.get(job_id)
        return state.to_response() if state else None

    def list_jobs(self) -> list[JobStatusResponse]:
        """List all jobs (active and completed)."""
        return [s.to_response() for s in self._jobs.values()]

    # ─── Progress listeners (Observer pattern) ─────────────────────────

    def register_listener(self, job_id: str, callback: ProgressCallback) -> None:
        """Register a WebSocket progress callback for a job."""
        state = self._jobs.get(job_id)
        if state is not None:
            state._listeners.append(callback)

    def unregister_listener(self, job_id: str, callback: ProgressCallback) -> None:
        """Remove a progress callback for a job."""
        state = self._jobs.get(job_id)
        if state is not None:
            state._listeners = [cb for cb in state._listeners if cb is not callback]

    async def _notify(
        self, state: _JobState, item: str, progress: float, message: str
    ) -> None:
        """Push progress to all registered listeners for a job."""
        for cb in state._listeners:
            try:
                await cb(state.job_id, item, progress, message)
            except Exception:
                pass

    # ─── Cleanup ───────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Cancel all running jobs during app shutdown."""
        for state in self._jobs.values():
            if state._task is not None and not state._task.done():
                state._cancel_event.set()
                state._pause_event.set()
                state._task.cancel()
        logger.info("batch_processor_shutdown", jobs=len(self._jobs))

"""Pipeline progress tracking with callback-based listener notification.

Tracks the current phase and progress percentage for each pipeline session
and broadcasts updates to registered listener callbacks.  Listeners are
keyed by session ID so multiple pipelines can run concurrently without
cross-talk.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

import structlog

from src.models.pipeline import PipelinePhase
from src.utils.logging import get_logger


@dataclass
class _SessionStatus:
    """Internal snapshot of a single session's progress."""

    phase: PipelinePhase = PipelinePhase.UPLOAD
    progress: float = 0.0
    message: str = ""


class ProgressTracker:
    """Tracks and broadcasts pipeline progress via callbacks.

    Each pipeline session is identified by a string ``session_id``.
    External consumers (e.g. WebSocket handlers, UI controllers) register
    async callbacks that are invoked whenever :meth:`update` is called for
    that session.
    """

    def __init__(self) -> None:
        self._statuses: dict[str, _SessionStatus] = {}
        self._listeners: dict[str, list[Callable]] = {}
        self._logger: structlog.BoundLogger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def update(
        self,
        session_id: str,
        phase: PipelinePhase,
        progress: float,
        message: str,
    ) -> None:
        """Record a progress update and notify all registered listeners.

        Parameters
        ----------
        session_id:
            The pipeline session to update.
        phase:
            The current pipeline phase.
        progress:
            Completion percentage (0.0 – 100.0).
        message:
            Human-readable status message.
        """
        progress = max(0.0, min(100.0, progress))

        self._statuses[session_id] = _SessionStatus(
            phase=phase,
            progress=progress,
            message=message,
        )

        self._logger.debug(
            "progress_update",
            session_id=session_id,
            phase=phase.value,
            progress=round(progress, 1),
            message=message,
        )

        await self._notify_listeners(session_id, phase, progress, message)

    def register_listener(self, session_id: str, callback: Callable) -> None:
        """Register a callback to receive progress updates for a session.

        Parameters
        ----------
        session_id:
            The pipeline session to listen to.
        callback:
            An async or sync callable accepting
            ``(session_id, phase, progress, message)``.
        """
        if session_id not in self._listeners:
            self._listeners[session_id] = []

        if callback not in self._listeners[session_id]:
            self._listeners[session_id].append(callback)
            self._logger.debug(
                "listener_registered",
                session_id=session_id,
                total_listeners=len(self._listeners[session_id]),
            )

    def unregister_listener(self, session_id: str, callback: Callable) -> None:
        """Remove a previously registered callback for a session.

        Parameters
        ----------
        session_id:
            The pipeline session.
        callback:
            The callback to remove.
        """
        listeners = self._listeners.get(session_id, [])
        if callback in listeners:
            listeners.remove(callback)
            self._logger.debug(
                "listener_unregistered",
                session_id=session_id,
                remaining_listeners=len(listeners),
            )

    def get_status(self, session_id: str) -> dict:
        """Return the current phase and progress for a session.

        Parameters
        ----------
        session_id:
            The pipeline session to query.

        Returns
        -------
        dict
            Keys: ``phase`` (:class:`str`), ``progress`` (:class:`float`),
            ``message`` (:class:`str`).  Returns zeroed defaults when the
            session has not yet been tracked.
        """
        status = self._statuses.get(session_id)
        if status is None:
            return {
                "phase": PipelinePhase.UPLOAD.value,
                "progress": 0.0,
                "message": "",
            }

        return {
            "phase": status.phase.value,
            "progress": status.progress,
            "message": status.message,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _notify_listeners(
        self,
        session_id: str,
        phase: PipelinePhase,
        progress: float,
        message: str,
    ) -> None:
        """Invoke all registered listeners for a session.

        Listeners that raise exceptions are logged and silently skipped
        so a single faulty listener cannot block progress updates.
        """
        listeners = self._listeners.get(session_id, [])
        if not listeners:
            return

        for callback in listeners:
            try:
                result = callback(session_id, phase, progress, message)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                self._logger.warning(
                    "listener_callback_error",
                    session_id=session_id,
                    error=str(exc),
                    callback=getattr(callback, "__name__", repr(callback)),
                )

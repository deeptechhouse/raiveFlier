"""WebSocket endpoint for real-time pipeline progress updates.

Connects a client to a specific pipeline session via the ``ProgressTracker``
listener mechanism.  Progress updates are pushed as JSON messages containing
``session_id``, ``phase``, ``progress``, and ``message``.
"""

from __future__ import annotations

import contextlib

import structlog
from fastapi import WebSocket, WebSocketDisconnect

from src.models.pipeline import PipelinePhase
from src.pipeline.progress_tracker import ProgressTracker
from src.utils.logging import get_logger

_logger: structlog.BoundLogger = get_logger(__name__)


async def websocket_progress(websocket: WebSocket, session_id: str) -> None:
    """Stream pipeline progress updates to the client over WebSocket.

    Lifecycle:
        1. Accept the WebSocket connection.
        2. Register a listener callback with the :class:`ProgressTracker`.
        3. Send the current status snapshot immediately.
        4. On every progress update, push a JSON message.
        5. On disconnect, unregister the listener and clean up.

    Parameters
    ----------
    websocket:
        The WebSocket connection managed by FastAPI / Starlette.
    session_id:
        The pipeline session to subscribe to.
    """
    progress_tracker: ProgressTracker = websocket.app.state.progress_tracker

    await websocket.accept()
    _logger.info("websocket_connected", session_id=session_id)

    # --- Listener callback ---

    async def _on_progress(
        sid: str,
        phase: PipelinePhase,
        progress: float,
        message: str,
    ) -> None:
        """Push a progress update to the connected WebSocket client."""
        with contextlib.suppress(Exception):
            await websocket.send_json(
                {
                    "session_id": sid,
                    "phase": phase.value,
                    "progress": round(progress, 1),
                    "message": message,
                }
            )

    progress_tracker.register_listener(session_id, _on_progress)

    try:
        # Send the current status snapshot so the client is up to date
        status = progress_tracker.get_status(session_id)
        await websocket.send_json({"session_id": session_id, **status})

        # Keep the connection alive — receive pings / keep-alive messages
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        _logger.info("websocket_disconnected", session_id=session_id)

    finally:
        progress_tracker.unregister_listener(session_id, _on_progress)
        _logger.debug("websocket_listener_cleaned_up", session_id=session_id)

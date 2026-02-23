"""WebSocket endpoint for real-time pipeline progress updates.

Connects a client to a specific pipeline session via the ``ProgressTracker``
listener mechanism.  Progress updates are pushed as JSON messages containing
``session_id``, ``phase``, ``progress``, and ``message``.

# ─── HOW WEBSOCKET PROGRESS WORKS (Junior Developer Guide) ────────────
#
# Instead of the frontend polling /status every second, we use a WebSocket
# for real-time push updates.  The flow:
#
#   Frontend (websocket.js)              Backend (this file)
#   ───────────────────────              ──────────────────
#   ws = new WebSocket(url)   ──────→   websocket.accept()
#                                        register_listener(callback)
#                             ←──────   send initial status snapshot
#                                        ...pipeline runs...
#                             ←──────   push progress update (JSON)
#                             ←──────   push progress update (JSON)
#                             ←──────   push progress update (JSON)
#   ws.close()                ──────→   WebSocketDisconnect
#                                        unregister_listener(callback)
#
# JSON message format:
#   { "session_id": "abc", "phase": "RESEARCH", "progress": 45.0, "message": "..." }
#
# The `while True: await websocket.receive_text()` loop keeps the
# connection alive.  The actual progress pushes happen via the
# _on_progress callback registered with ProgressTracker.
# ──────────────────────────────────────────────────────────────────────
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
    # Access the singleton ProgressTracker from app.state (set during startup).
    progress_tracker: ProgressTracker = websocket.app.state.progress_tracker

    await websocket.accept()
    _logger.info("websocket_connected", session_id=session_id)

    # --- Listener callback ---
    # This closure captures the `websocket` variable and pushes JSON
    # to the client whenever the ProgressTracker fires an update.

    async def _on_progress(
        sid: str,
        phase: PipelinePhase,
        progress: float,
        message: str,
    ) -> None:
        """Push a progress update to the connected WebSocket client."""
        # contextlib.suppress(Exception) silently ignores errors — this is
        # intentional because the WebSocket may have disconnected between
        # the time we checked and the time we send.  The cleanup happens
        # in the `finally` block below.
        with contextlib.suppress(Exception):
            await websocket.send_json(
                {
                    "session_id": sid,
                    "phase": phase.value,
                    "progress": round(progress, 1),
                    "message": message,
                }
            )

    # Register our callback with the ProgressTracker's Observer mechanism.
    progress_tracker.register_listener(session_id, _on_progress)

    try:
        # Send the current status snapshot so the client is immediately up
        # to date (in case progress happened before the WebSocket connected).
        status = progress_tracker.get_status(session_id)
        await websocket.send_json({"session_id": session_id, **status})

        # Keep the connection alive — receive pings / keep-alive messages.
        # This blocks until the client disconnects (raises WebSocketDisconnect).
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        _logger.info("websocket_disconnected", session_id=session_id)

    finally:
        # Always clean up the listener to prevent memory leaks.
        progress_tracker.unregister_listener(session_id, _on_progress)
        _logger.debug("websocket_listener_cleaned_up", session_id=session_id)

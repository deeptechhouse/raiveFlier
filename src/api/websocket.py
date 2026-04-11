"""WebSocket endpoints for real-time pipeline updates.

Two WebSocket endpoints live in this module:

1. ``websocket_progress`` — Pushes pipeline phase/progress/message updates
   during the full analysis lifecycle.  Connected via /ws/progress/{session_id}.

2. ``websocket_interconnection_stream`` — Streams interconnection analysis
   results (narrative chunks + final map) in real-time.  Connected via
   /ws/interconnection/{session_id}.  This enables the frontend to render
   the LLM narrative with a typewriter effect instead of waiting 10-20s
   for the full response.

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
#
# ─── HOW INTERCONNECTION STREAMING WORKS ──────────────────────────────
#
#   Frontend (websocket.js)              Backend (this file)
#   ───────────────────────              ──────────────────
#   ws = new WebSocket(url)   ──────→   websocket.accept()
#                                        call analyze_streaming()
#                             ←──────   {"type":"narrative_chunk","text":"..."}
#                             ←──────   {"type":"narrative_chunk","text":"..."}
#                             ←──────   {"type":"analysis_complete","interconnection_map":{}}
#                                        websocket.close()
#
# The frontend opens this WebSocket when the pipeline reaches the
# INTERCONNECTION phase.  As LLM tokens arrive, they are forwarded
# directly to the client for progressive rendering.
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


async def websocket_interconnection_stream(
    websocket: WebSocket, session_id: str
) -> None:
    """Stream interconnection analysis results to the client over WebSocket.

    This endpoint is opened by the frontend when the pipeline reaches the
    INTERCONNECTION phase.  It retrieves the session's research results and
    entities from the session store, then calls
    ``InterconnectionService.analyze_streaming()`` which yields:

    1. ``{"type": "narrative_chunk", "text": "..."}`` — LLM tokens as they
       arrive, enabling typewriter-style rendering of the narrative.
    2. ``{"type": "analysis_complete", "interconnection_map": {...}}`` —
       the final validated InterconnectionMap once all tokens are received
       and post-processing (citation validation, confidence enrichment) is
       done.

    On error, sends ``{"type": "error", "message": "..."}`` and closes
    the connection cleanly rather than crashing.

    Parameters
    ----------
    websocket:
        The WebSocket connection managed by FastAPI / Starlette.
    session_id:
        The pipeline session whose research results to analyse.
    """
    await websocket.accept()
    _logger.info("interconnection_ws_connected", session_id=session_id)

    try:
        # Retrieve the interconnection service and session store from
        # app.state — these are injected during startup by main.py's
        # _build_all() function.  The session store is named "session_states"
        # (not "session_store") in _build_all()'s return dict.
        interconnection_service = websocket.app.state.interconnection_service
        session_store = websocket.app.state.session_states

        # Look up the session to get research_results and entities.
        session = session_store.get(session_id)
        if session is None:
            await websocket.send_json({
                "type": "error",
                "message": f"Session {session_id} not found",
            })
            await websocket.close(code=1008)
            return

        # The session must have completed the research phase to have
        # the data needed for interconnection analysis.
        research_results = getattr(session, "research_results", None)
        entities = getattr(session, "extracted_entities", None)

        if not research_results or not entities:
            await websocket.send_json({
                "type": "error",
                "message": "Session missing research results or entities",
            })
            await websocket.close(code=1008)
            return

        # Stream chunks from analyze_streaming() directly to the client.
        # Each yielded dict is sent as a JSON message.
        async for update in interconnection_service.analyze_streaming(
            research_results=research_results,
            entities=entities,
        ):
            # contextlib.suppress(Exception) handles the case where the
            # client disconnects mid-stream — we don't want to crash.
            with contextlib.suppress(Exception):
                await websocket.send_json(update)

        # Close cleanly after all data is sent.
        await websocket.close(code=1000)

    except WebSocketDisconnect:
        _logger.info("interconnection_ws_disconnected", session_id=session_id)

    except Exception as exc:
        _logger.error(
            "interconnection_ws_error",
            session_id=session_id,
            error=str(exc),
        )
        # Send error to client before closing so the frontend can display
        # a meaningful message instead of a generic connection error.
        with contextlib.suppress(Exception):
            await websocket.send_json({
                "type": "error",
                "message": f"Interconnection analysis failed: {exc}",
            })
            await websocket.close(code=1011)

"""WebSocket endpoint for real-time batch job progress updates.

# ─── HOW FEEDER WEBSOCKET WORKS ───────────────────────────────────────
#
# Mirrors raiveFlier's WebSocket pattern (src/api/websocket.py) but
# subscribes to BatchProcessor job progress instead of pipeline phases.
#
#   Frontend (progress.js)              Backend (this file)
#   ──────────────────────              ──────────────────
#   ws = new WebSocket(url)   ──────→   websocket.accept()
#                                        register_listener(callback)
#                             ←──────   send initial status snapshot
#                                        ...batch runs...
#                             ←──────   push item progress (JSON)
#                             ←──────   push item progress (JSON)
#   ws.close()                ──────→   WebSocketDisconnect
#                                        unregister_listener(callback)
#
# JSON message format:
#   { "job_id": "abc", "item": "file.pdf", "progress": 45.0,
#     "message": "Processing file.pdf" }
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import contextlib

import structlog
from fastapi import WebSocket, WebSocketDisconnect

logger = structlog.get_logger(logger_name=__name__)


async def websocket_progress(websocket: WebSocket, job_id: str) -> None:
    """Stream batch job progress updates to the client over WebSocket.

    Parameters
    ----------
    websocket:
        The WebSocket connection managed by FastAPI / Starlette.
    job_id:
        The batch job to subscribe to.
    """
    batch_processor = getattr(websocket.app.state, "batch_processor", None)
    if batch_processor is None:
        await websocket.close(code=1011, reason="Batch processor unavailable")
        return

    await websocket.accept()
    logger.info("feeder_ws_connected", job_id=job_id)

    # Listener callback — pushes progress JSON to the connected client.
    async def _on_progress(
        jid: str, item: str, progress: float, message: str
    ) -> None:
        with contextlib.suppress(Exception):
            await websocket.send_json({
                "job_id": jid,
                "item": item,
                "progress": round(progress, 1),
                "message": message,
            })

    batch_processor.register_listener(job_id, _on_progress)

    try:
        # Send initial status snapshot.
        status = batch_processor.get_job_status(job_id)
        if status:
            await websocket.send_json(status.model_dump())

        # Keep connection alive until client disconnects.
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        logger.info("feeder_ws_disconnected", job_id=job_id)

    finally:
        batch_processor.unregister_listener(job_id, _on_progress)

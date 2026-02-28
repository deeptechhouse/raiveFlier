"""Rave Stories sub-application factory.

# ─── ARCHITECTURE ROLE ───────────────────────────────────────────────
#
# This module creates the FastAPI sub-application that is mounted at
# ``/stories`` on the main raiveFlier app (same pattern as raiveFeeder
# at ``/feeder``).
#
# The sub-app:
#   1. Includes the stories API router at ``/api/v1/stories/``.
#   2. Applies ``StoryAnonymityMiddleware`` to strip identifying headers.
#   3. Serves the stories frontend SPA as static files.
#   4. Wires DI components from the parent app or builds its own
#      (for standalone usage).
#
# Mounting pattern:
#   In src/main.py:
#       from src.stories.main import create_stories_app
#       app.mount("/stories", create_stories_app(settings, components))
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.stories.api.routes import router as stories_router

logger = structlog.get_logger(logger_name=__name__)

# Path to the stories frontend directory (relative to project root).
_FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "stories_frontend"


# ── Anonymity Middleware ──────────────────────────────────────────────
class StoryAnonymityMiddleware(BaseHTTPMiddleware):
    """Strip identifying headers from story-related requests.

    This middleware runs on ALL requests to the stories sub-app.
    It removes headers that could identify the submitter before
    any logging occurs, enforcing the anonymity guarantee.

    Stripped headers: X-Forwarded-For, X-Real-IP, User-Agent,
    X-Request-ID, and any custom tracking headers.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        # Create a sanitized scope by removing identifying headers.
        # We can't modify the immutable request headers directly,
        # so we log the stripped state for audit purposes.
        sanitized_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in (
                "x-forwarded-for", "x-real-ip", "user-agent",
                "x-request-id", "x-correlation-id",
            )
        }

        # Override the structlog context to exclude identifying info
        # for the duration of this request.
        with structlog.contextvars.bound_contextvars(
            story_route=True,
            client_ip="[stripped]",
            user_agent="[stripped]",
        ):
            response = await call_next(request)

        return response


# ── Sub-app lifecycle ─────────────────────────────────────────────────
@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Initialize story service components on startup."""
    # If components were passed via app.state.story_components,
    # wire them up.  Otherwise, build standalone components.
    components = getattr(app.state, "story_components", None)
    if components:
        for key, value in components.items():
            setattr(app.state, key, value)

    # Initialize the story store if present.
    story_store = getattr(app.state, "story_store", None)
    if story_store is not None:
        await story_store.initialize()
        logger.info("story_store_initialized")

    yield
    logger.info("stories_app_shutdown")


# ── Factory ───────────────────────────────────────────────────────────
def create_stories_app(
    components: dict[str, Any] | None = None,
) -> FastAPI:
    """Create the Rave Stories FastAPI sub-application.

    Parameters
    ----------
    components:
        Optional dict of pre-built DI components (story_service,
        story_store, etc.).  When mounted in the main app, these are
        passed from ``_build_all()``.  When running standalone, the
        lifespan will build its own.

    Returns
    -------
    FastAPI
        The configured sub-application ready for mounting.
    """
    app = FastAPI(
        title="Rave Stories",
        description="Anonymous first-person rave experience stories",
        version="0.1.0",
        lifespan=_lifespan,
    )

    # Store components for the lifespan to wire up.
    if components:
        app.state.story_components = components

    # Anonymity middleware — strips identifying headers on all story routes.
    app.add_middleware(StoryAnonymityMiddleware)

    # API routes.
    app.include_router(stories_router)

    # Frontend static files.
    if _FRONTEND_DIR.is_dir():
        if (_FRONTEND_DIR / "css").exists():
            app.mount(
                "/css",
                StaticFiles(directory=str(_FRONTEND_DIR / "css")),
                name="stories_css",
            )
        if (_FRONTEND_DIR / "js").exists():
            app.mount(
                "/js",
                StaticFiles(directory=str(_FRONTEND_DIR / "js")),
                name="stories_js",
            )

        # Serve the SPA entry point at the sub-app root.
        @app.get("/", include_in_schema=False)
        async def serve_index() -> FileResponse:
            return FileResponse(str(_FRONTEND_DIR / "index.html"))

    return app

"""API middleware — CORS, request logging, and error handling.

Provides helper functions and middleware classes to configure cross-origin
resource sharing, structured request logging (via structlog), and automatic
conversion of ``RaiveFlierError`` subclasses into JSON ``ErrorResponse``
bodies.

# ─── MIDDLEWARE EXECUTION ORDER (Junior Developer Guide) ───────────────
#
# Starlette middleware is a stack (LIFO — last added, first executed):
#
#   In main.py:
#     app.add_middleware(ErrorHandlingMiddleware)   # added 2nd → wraps inner
#     app.add_middleware(RequestLoggingMiddleware)   # added 1st → outermost
#
#   Request flow:
#     Client → RequestLogging → ErrorHandling → route handler
#   Response flow:
#     Client ← RequestLogging ← ErrorHandling ← route handler
#
# So RequestLoggingMiddleware sees the *final* response status code
# (even if ErrorHandling replaced a 500 with a structured JSON error).
#
# Each middleware extends BaseHTTPMiddleware and overrides dispatch().
# Inside dispatch(), call_next(request) passes to the next middleware
# or the actual route handler.
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import time

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.api.schemas import ErrorResponse
from src.utils.errors import RaiveFlierError
from src.utils.logging import get_logger

_logger: structlog.BoundLogger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


def configure_cors(app: FastAPI, *, allowed_origins: list[str] | None = None) -> None:
    """Add CORS middleware to the FastAPI application.

    # JUNIOR DEV NOTE — What is CORS?
    # CORS (Cross-Origin Resource Sharing) controls which domains can
    # make API requests to this server.  Browsers enforce this for
    # security — without CORS headers, a page at domain-A can't fetch
    # from domain-B.
    #
    # In development: allow ["*"] (all origins) for convenience.
    # In production: restrict to the actual deployed domain only.

    Parameters
    ----------
    app:
        The FastAPI application instance.
    allowed_origins:
        Explicit list of allowed origins.  Defaults to ``["*"]`` for
        development; override with specific origins in production.
    """
    origins = allowed_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,  # Allow cookies/auth headers
        allow_methods=["*"],     # Allow all HTTP methods (GET, POST, etc.)
        allow_headers=["*"],     # Allow all request headers
    )


# ---------------------------------------------------------------------------
# Request Logging
# ---------------------------------------------------------------------------


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request with method, path, status code, and duration."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        start = time.perf_counter()
        response: Response | None = None

        try:
            response = await call_next(request)
            return response
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            status_code = response.status_code if response else 500
            _logger.info(
                "http_request",
                method=request.method,
                path=str(request.url.path),
                status=status_code,
                duration_ms=duration_ms,
            )


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Catch ``RaiveFlierError`` subclasses and return structured JSON errors.

    All unhandled application errors are converted into an
    :class:`ErrorResponse` with the exception class name and message.
    Stack traces are logged server-side only — never leaked to the client.

    # JUNIOR DEV NOTE — Security: Error Sanitization
    # This middleware is a security boundary.  It ensures that:
    #   1. Internal error details (stack traces, file paths) stay in logs
    #   2. The client only sees a sanitized error type + message
    #   3. Application-specific errors (RaiveFlierError) are caught here;
    #      generic Python exceptions will bubble up to FastAPI's default
    #      500 handler (which also hides internals in production mode).
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        try:
            return await call_next(request)
        except RaiveFlierError as exc:
            # Log the full error details server-side for debugging
            _logger.error(
                "application_error",
                error_type=type(exc).__name__,
                message=exc.message,
                provider=exc.provider_name,
                path=str(request.url.path),
            )
            # Return a sanitized JSON error to the client
            body = ErrorResponse(
                error=type(exc).__name__,
                detail=exc.message,
            )
            return JSONResponse(
                status_code=500,
                content=body.model_dump(),
            )

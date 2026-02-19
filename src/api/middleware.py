"""API middleware — CORS, request logging, and error handling.

Provides helper functions and middleware classes to configure cross-origin
resource sharing, structured request logging (via structlog), and automatic
conversion of ``RaiveFlierError`` subclasses into JSON ``ErrorResponse``
bodies.
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
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
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
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        try:
            return await call_next(request)
        except RaiveFlierError as exc:
            _logger.error(
                "application_error",
                error_type=type(exc).__name__,
                message=exc.message,
                provider=exc.provider_name,
                path=str(request.url.path),
            )
            body = ErrorResponse(
                error=type(exc).__name__,
                detail=exc.message,
            )
            return JSONResponse(
                status_code=500,
                content=body.model_dump(),
            )

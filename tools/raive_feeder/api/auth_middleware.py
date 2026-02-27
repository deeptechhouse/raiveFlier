"""Passcode authentication middleware for raiveFeeder.

# ─── HOW THE AUTH MIDDLEWARE WORKS ───────────────────────────────────
#
# FeederAuthMiddleware intercepts every request to the feeder sub-app
# and checks for a valid ``raive_feeder_session`` cookie.  If the cookie
# is missing or invalid, browser requests are redirected to the login
# page and API requests receive a 401 JSON response.
#
# Design decisions:
#   - Middleware on the feeder sub-app ONLY — raiveFlier stays fully public.
#   - Dev mode bypass: if RAIVE_FEEDER_PASSCODE is empty, auth is disabled
#     entirely so local development doesn't require a passcode.
#   - Exempt paths allow the login page and static assets to load without auth.
#   - Uses BaseHTTPMiddleware (Starlette) — no new dependencies.
#
# Layer position: after CORS, before route handlers.
# ─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response

from tools.raive_feeder.api.auth_utils import validate_session_cookie

# Cookie name used for the feeder session.
COOKIE_NAME = "raive_feeder_session"

# File extensions that are always allowed without auth (static assets).
_STATIC_EXTENSIONS = (".css", ".js", ".ico", ".woff2", ".woff", ".ttf", ".png", ".jpg", ".svg")

# Path prefixes exempt from auth checks.
_EXEMPT_PATHS = ("/login", "/api/v1/auth/login", "/api/v1/auth/status", "/api/v1/health")


class FeederAuthMiddleware(BaseHTTPMiddleware):
    """Enforces passcode-based session authentication on the feeder sub-app.

    Constructor injection: the passcode secret and TTL are passed in from
    main.py so the middleware doesn't read config globals directly.
    """

    def __init__(self, app: object, secret: str, ttl_hours: int = 168) -> None:
        super().__init__(app)
        self._secret = secret
        self._ttl_hours = ttl_hours

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Check auth for every request; exempt login and static assets."""
        # Dev mode: no passcode configured — skip auth entirely.
        if not self._secret:
            return await call_next(request)

        path = request.url.path

        # Allow exempt paths through without auth.
        if self._is_exempt(path):
            return await call_next(request)

        # Validate the session cookie.
        cookie = request.cookies.get(COOKIE_NAME, "")
        if validate_session_cookie(cookie, self._secret, self._ttl_hours):
            return await call_next(request)

        # Auth failed — determine response type.
        accept = request.headers.get("accept", "")
        if "text/html" in accept:
            # Browser request — redirect to login page.
            # Use root_path so the redirect includes the sub-app mount prefix
            # (e.g. /feeder/login when mounted at /feeder/).
            root_path = request.scope.get("root_path", "")
            return RedirectResponse(url=f"{root_path}/login", status_code=303)
        # API/fetch request — return 401 JSON.
        return JSONResponse(
            status_code=401,
            content={"detail": "Authentication required"},
        )

    @staticmethod
    def _is_exempt(path: str) -> bool:
        """Check if the path is exempt from authentication."""
        # Exempt specific paths.
        for exempt in _EXEMPT_PATHS:
            if path == exempt or path.startswith(exempt + "/"):
                return True
        # Exempt static asset files.
        if any(path.endswith(ext) for ext in _STATIC_EXTENSIONS):
            return True
        return False

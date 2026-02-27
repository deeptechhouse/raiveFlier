"""Authentication routes for raiveFeeder passcode gate.

# ─── ROUTE ARCHITECTURE ─────────────────────────────────────────────
#
# These routes handle the login/logout flow for raiveFeeder's passcode
# gate.  The login page is served as inline HTML (no separate template
# file) to keep the auth system self-contained.
#
# Endpoints:
#   GET  /login              — Serves the login page (inline HTML)
#   POST /api/v1/auth/login  — Validates passcode, sets session cookie
#   POST /api/v1/auth/logout — Clears session cookie
#   GET  /api/v1/auth/status — Returns current auth state
#
# The login page uses the same Bunker + Emerald theme as the main app
# for visual consistency.
# ─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import hmac
import os

from fastapi import APIRouter, Request
from starlette.responses import HTMLResponse, JSONResponse

from tools.raive_feeder.api.auth_middleware import COOKIE_NAME
from tools.raive_feeder.api.auth_utils import create_session_cookie, validate_session_cookie

# Router for auth endpoints — mounted by main.py.
auth_router = APIRouter()


# ─── Login page (inline HTML) ────────────────────────────────────────

_LOGIN_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>raiveFeeder — Login</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
    html { color-scheme: dark; }
    body {
      font-family: 'IBM Plex Mono', monospace;
      background: #0a0a0c;
      color: #d8d5cd;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .login-card {
      background: #14141a;
      border: 1px solid #2a2a30;
      border-radius: 16px;
      padding: 3rem 2.5rem;
      width: 100%;
      max-width: 380px;
      text-align: center;
    }
    .login-card h1 {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 1.5rem;
      color: #50c080;
      margin-bottom: 0.5rem;
    }
    .login-card p {
      font-size: 0.75rem;
      color: #6a6a70;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 2rem;
    }
    .login-input {
      width: 100%;
      padding: 0.75rem 1rem;
      background: #1e1e24;
      border: 1px solid #2a2a30;
      border-radius: 8px;
      color: #d8d5cd;
      font-family: inherit;
      font-size: 1rem;
      text-align: center;
      letter-spacing: 0.15em;
      outline: none;
      transition: border-color 150ms;
    }
    .login-input:focus { border-color: #2a8a5a; }
    .login-input::placeholder { color: #3a3a40; letter-spacing: 0.1em; }
    .login-btn {
      width: 100%;
      padding: 0.75rem;
      margin-top: 1.25rem;
      background: #2a8a5a;
      color: #fff;
      border: none;
      border-radius: 8px;
      font-family: inherit;
      font-size: 0.875rem;
      font-weight: 500;
      cursor: pointer;
      transition: background 150ms;
    }
    .login-btn:hover { background: #3aaa6a; }
    .login-btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .login-error {
      margin-top: 1rem;
      color: #c87080;
      font-size: 0.8rem;
      min-height: 1.2em;
    }
  </style>
</head>
<body>
  <div class="login-card">
    <h1>raiveFeeder</h1>
    <p>Corpus ingestion access</p>
    <form id="login-form">
      <input type="password" class="login-input" id="passcode"
             placeholder="Enter passcode" autocomplete="off" autofocus>
      <button type="submit" class="login-btn" id="login-btn">Authenticate</button>
    </form>
    <div class="login-error" id="login-error"></div>
  </div>
  <script>
    const form = document.getElementById('login-form');
    const input = document.getElementById('passcode');
    const btn = document.getElementById('login-btn');
    const errorEl = document.getElementById('login-error');

    // Derive path prefix so login API call works at /feeder/api/v1/auth/login.
    const prefix = window.location.pathname.replace(/\\/login\\/?$/, '');

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      errorEl.textContent = '';
      btn.disabled = true;
      btn.textContent = 'Verifying...';
      try {
        const resp = await fetch(prefix + '/api/v1/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ passcode: input.value }),
        });
        const data = await resp.json();
        if (resp.ok && data.authenticated) {
          window.location.href = prefix + '/';
        } else {
          errorEl.textContent = data.detail || 'Invalid passcode';
        }
      } catch (err) {
        errorEl.textContent = 'Connection error';
      } finally {
        btn.disabled = false;
        btn.textContent = 'Authenticate';
      }
    });
  </script>
</body>
</html>
"""


@auth_router.get("/login", response_class=HTMLResponse, include_in_schema=False)
async def login_page() -> HTMLResponse:
    """Serve the inline HTML login page."""
    return HTMLResponse(content=_LOGIN_HTML)


# ─── Auth API endpoints ──────────────────────────────────────────────


@auth_router.post("/api/v1/auth/login")
async def auth_login(request: Request) -> JSONResponse:
    """Validate passcode and set session cookie.

    Expects JSON body: ``{"passcode": "the-shared-passcode"}``
    Uses constant-time comparison to prevent timing attacks.
    """
    body = await request.json()
    provided = body.get("passcode", "")
    settings = request.app.state.settings

    expected = settings.feeder_passcode

    # Constant-time comparison prevents timing-based passcode guessing.
    if not expected or not hmac.compare_digest(provided, expected):
        return JSONResponse(
            status_code=401,
            content={"authenticated": False, "detail": "Invalid passcode"},
        )

    # Create signed session cookie.
    cookie_value = create_session_cookie(expected, settings.session_cookie_ttl_hours)

    # Determine if we're behind HTTPS (production).
    is_secure = os.getenv("APP_ENV", "development") == "production"

    response = JSONResponse(content={"authenticated": True})
    response.set_cookie(
        key=COOKIE_NAME,
        value=cookie_value,
        httponly=True,
        samesite="strict",
        secure=is_secure,
        max_age=settings.session_cookie_ttl_hours * 3600,
        path="/",
    )
    return response


@auth_router.post("/api/v1/auth/logout")
async def auth_logout() -> JSONResponse:
    """Clear the session cookie."""
    response = JSONResponse(content={"authenticated": False})
    response.delete_cookie(key=COOKIE_NAME, path="/")
    return response


@auth_router.get("/api/v1/auth/status")
async def auth_status(request: Request) -> JSONResponse:
    """Return current authentication state."""
    settings = request.app.state.settings

    # No passcode configured — auth is disabled.
    if not settings.feeder_passcode:
        return JSONResponse(content={"authenticated": True, "auth_enabled": False})

    cookie = request.cookies.get(COOKIE_NAME, "")
    is_valid = validate_session_cookie(
        cookie, settings.feeder_passcode, settings.session_cookie_ttl_hours
    )
    return JSONResponse(content={"authenticated": is_valid, "auth_enabled": True})

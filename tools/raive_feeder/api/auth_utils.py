"""HMAC-signed session cookie utilities for raiveFeeder authentication.

# ─── HOW AUTH COOKIES WORK ───────────────────────────────────────────
#
# The passcode gate uses an HMAC-SHA256 signed cookie instead of
# server-side sessions.  This keeps the system stateless (no session
# DB needed) while preventing cookie forgery.
#
# Cookie format:  {unix_timestamp}:{hmac_hex_digest}
#   - timestamp: when the cookie was issued (UTC epoch seconds)
#   - hmac:      HMAC-SHA256(secret, timestamp_str)
#
# Validation checks:
#   1. Cookie format matches expected pattern
#   2. HMAC signature is valid (constant-time comparison)
#   3. Timestamp is within the TTL window
#
# Zero external dependencies — uses only Python stdlib (hmac, hashlib, time).
# ─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import hashlib
import hmac
import time


def create_session_cookie(secret: str, ttl_hours: int = 168) -> str:
    """Create an HMAC-signed session cookie value.

    Parameters
    ----------
    secret:
        The shared secret (typically the feeder passcode).
    ttl_hours:
        Cookie lifetime in hours (default 168 = 7 days).
        Stored for documentation; actual expiry checked at validation time.

    Returns
    -------
    Cookie string in the format ``{timestamp}:{hmac_hex}``.
    """
    timestamp = str(int(time.time()))
    signature = hmac.new(
        secret.encode("utf-8"),
        timestamp.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{timestamp}:{signature}"


def validate_session_cookie(
    cookie: str,
    secret: str,
    ttl_hours: int = 168,
) -> bool:
    """Validate an HMAC-signed session cookie.

    Parameters
    ----------
    cookie:
        The cookie value to validate.
    secret:
        The shared secret used when the cookie was created.
    ttl_hours:
        Maximum age in hours before the cookie is considered expired.

    Returns
    -------
    True if the cookie is valid and not expired, False otherwise.
    """
    if not cookie or ":" not in cookie:
        return False

    parts = cookie.split(":", 1)
    if len(parts) != 2:
        return False

    timestamp_str, provided_hmac = parts

    # Verify the timestamp is a valid integer.
    try:
        timestamp = int(timestamp_str)
    except ValueError:
        return False

    # Check expiry.
    max_age_seconds = ttl_hours * 3600
    if time.time() - timestamp > max_age_seconds:
        return False

    # Recompute the expected HMAC and compare in constant time.
    expected_hmac = hmac.new(
        secret.encode("utf-8"),
        timestamp_str.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(provided_hmac, expected_hmac)

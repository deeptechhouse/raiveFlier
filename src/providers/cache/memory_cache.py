"""In-memory cache provider using cachetools.TTLCache.

Simple, fast cache suitable for development and single-process deployments.
Can be swapped for Redis or another backend via the ICacheProvider interface.
"""

from __future__ import annotations

from typing import Any

import structlog
from cachetools import TTLCache

from src.interfaces.cache_provider import ICacheProvider

logger = structlog.get_logger(logger_name=__name__)


class MemoryCacheProvider(ICacheProvider):
    """In-memory TTL cache backed by ``cachetools.TTLCache``.

    Parameters
    ----------
    max_size:
        Maximum number of entries before the least-recently-used entry
        is evicted.
    ttl:
        Default time-to-live in seconds for cache entries.
    """

    def __init__(self, max_size: int = 1000, ttl: int = 3600) -> None:
        self._default_ttl = ttl
        self._cache: TTLCache[str, Any] = TTLCache(maxsize=max_size, ttl=ttl)

    # ------------------------------------------------------------------
    # ICacheProvider implementation
    # ------------------------------------------------------------------

    async def get(self, key: str) -> Any | None:
        """Retrieve the cached value for *key*, or ``None`` if missing/expired."""
        value = self._cache.get(key)
        if value is not None:
            logger.debug("cache_hit", key=key)
        else:
            logger.debug("cache_miss", key=key)
        return value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store *value* under *key*.

        The per-item *ttl* parameter is noted but the underlying
        ``TTLCache`` applies a uniform TTL set at construction time.
        If a different TTL is needed, consider using a Redis-backed
        cache provider.
        """
        self._cache[key] = value
        logger.debug("cache_set", key=key)

    async def delete(self, key: str) -> None:
        """Remove *key* from the cache (no-op if absent)."""
        self._cache.pop(key, None)
        logger.debug("cache_delete", key=key)

    async def exists(self, key: str) -> bool:
        """Return ``True`` if *key* is present and not expired."""
        return key in self._cache

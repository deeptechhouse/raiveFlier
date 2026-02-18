"""Abstract base class for cache service providers.

Defines the contract for key-value caching used across the raiveFlier
pipeline (OCR results, API responses, research data).  Implementations may
use an in-memory dict, SQLite, Redis, or any other storage backend.  The
adapter pattern (CLAUDE.md Section 6) allows the cache backend to be
swapped without touching business logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ICacheProvider(ABC):
    """Contract for key-value cache services.

    All operations are async to allow for network-backed stores (e.g. Redis)
    without blocking the event loop.
    """

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Retrieve the value stored under *key*.

        Parameters
        ----------
        key:
            The cache key to look up.

        Returns
        -------
        Any or None
            The cached value if present and not expired; ``None`` otherwise.
        """

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store *value* under *key* with an optional time-to-live.

        Parameters
        ----------
        key:
            The cache key.
        value:
            The value to store.  Implementations should handle serialisation
            of common Python types (str, int, float, dict, list).
        ttl:
            Time-to-live in seconds.  ``None`` means the entry does not
            expire automatically.
        """

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove the entry stored under *key*.

        This is a no-op if the key does not exist.

        Parameters
        ----------
        key:
            The cache key to delete.
        """

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Return ``True`` if *key* is present in the cache and not expired.

        Parameters
        ----------
        key:
            The cache key to check.
        """

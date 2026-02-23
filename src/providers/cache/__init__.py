"""Cache providers.

In-memory TTL-based cache used to avoid redundant API calls during a single
analysis session (e.g., if the same artist name appears on multiple fliers
processed in sequence, the Discogs lookup result is cached).

MemoryCacheProvider is a dict-based cache — fast but not shared across
processes. For multi-worker deployments, swap in a Redis adapter implementing
ICacheProvider without changing any business logic.
"""

from src.providers.cache.memory_cache import MemoryCacheProvider

__all__ = ["MemoryCacheProvider"]

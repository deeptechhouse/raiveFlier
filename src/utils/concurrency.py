"""Shared concurrency primitives for research pipeline parallelization.

Provides a global search semaphore that all researchers share to limit
concurrent web search requests, preventing rate-limiting from DuckDuckGo
and other search providers while still allowing parallel execution.

Two main patterns are exposed:

1. **throttled_gather** -- A drop-in replacement for ``asyncio.gather`` that
   wraps each awaitable in a semaphore acquire/release.  Used when you have
   a list of arbitrary coroutines to run with bounded concurrency.

2. **parallel_search** -- A higher-level convenience function for the
   fan-out-then-merge pattern used by every researcher module: dispatch
   N search queries in parallel, collect results, log failures, and return
   a flat list of all successful results.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, TypeVar

import structlog

from src.utils.logging import get_logger

_T = TypeVar("_T")

# Global semaphore limiting concurrent web search requests across ALL
# researcher instances.  A value of 5 allows meaningful parallelism
# while staying well under DuckDuckGo's rate-limit threshold (~20 req/min).
# This is shared (not per-researcher) so that bursts from multiple
# researchers running simultaneously don't stack up and trip rate limits.
_SEARCH_SEMAPHORE = asyncio.Semaphore(5)

_logger: structlog.BoundLogger = get_logger(__name__)


async def throttled_gather(
    coros: list[Awaitable[_T]],
    semaphore: asyncio.Semaphore | None = None,
    return_exceptions: bool = True,
) -> list[_T | BaseException]:
    """Run awaitables concurrently with optional semaphore throttling.

    Each coroutine is wrapped so it acquires the semaphore before executing
    and releases it afterward, ensuring at most ``semaphore._value``
    coroutines run simultaneously.

    Parameters
    ----------
    coros:
        Awaitable objects to execute concurrently.
    semaphore:
        Optional semaphore for concurrency control.  Defaults to the
        module-level ``_SEARCH_SEMAPHORE``.
    return_exceptions:
        If ``True``, exceptions are returned in the results list rather
        than being raised.  Mirrors ``asyncio.gather`` semantics.

    Returns
    -------
    list[_T | BaseException]
        Results in the same order as the input coroutines.
    """
    if semaphore is None:
        semaphore = _SEARCH_SEMAPHORE

    async def _wrapped(coro: Awaitable[_T]) -> _T:
        # The semaphore context manager blocks here until a slot opens.
        async with semaphore:
            return await coro

    # All tasks are created immediately but only ``semaphore._value``
    # will actually be executing at any moment.
    tasks = [_wrapped(c) for c in coros]
    return await asyncio.gather(*tasks, return_exceptions=return_exceptions)


async def parallel_search(
    search_fn: Callable[..., Awaitable[list[Any]]],
    queries: list[dict[str, Any]],
    logger: structlog.BoundLogger | None = None,
    error_msg: str = "Search query failed",
) -> list[Any]:
    """Execute multiple search queries in parallel with throttling.

    Implements the **fan-out / merge** pattern used by every researcher:
    1. Fan out -- dispatch each query dict as ``search_fn(**query)``
    2. Throttle -- all calls share the global semaphore
    3. Merge -- flatten successful results into one list; log failures

    Parameters
    ----------
    search_fn:
        The async search function to call for each query.  Called with
        keyword arguments from each dict in ``queries``.
    queries:
        List of keyword-argument dicts, one per search call.
    logger:
        Optional structured logger for warnings on failures.
    error_msg:
        Log message prefix for failed queries.

    Returns
    -------
    list[Any]
        Flattened list of all search results from successful queries.
    """
    if logger is None:
        logger = _logger

    # Build coroutines by splatting each query dict as kwargs.
    coros = [search_fn(**q) for q in queries]
    raw_results = await throttled_gather(coros, return_exceptions=True)

    # Post-processing: separate successes from failures and flatten lists.
    all_results: list[Any] = []
    for idx, result in enumerate(raw_results):
        if isinstance(result, BaseException):
            # Log the failure but don't abort -- partial results are valuable.
            logger.warning(error_msg, query=queries[idx], error=str(result))
        elif isinstance(result, list):
            all_results.extend(result)  # Flatten list results
        else:
            all_results.append(result)

    return all_results

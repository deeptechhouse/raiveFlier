"""Resident Advisor (RA.co) event scraper via their undocumented GraphQL API.

RA.co is a client-side rendered React app.  Direct HTML scraping returns 403.
This provider issues POST requests to ``https://ra.co/graphql`` using the
``GET_EVENT_LISTINGS`` operation discovered from open-source scrapers.

Rate-limited to ~15 requests/minute (4-second delay) to avoid detection.
Includes retry with exponential backoff on 403/429 responses.  Returns
typed :class:`~src.models.ra_event.RAEvent` Pydantic models.

Follows the same adapter pattern as :class:`DiscogsScrapeProvider`:
injected ``httpx.AsyncClient``, ``_throttle()``, graceful error handling.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import date
from typing import Any, Callable

import httpx

from src.models.ra_event import RAEvent, RAEventPage
from src.utils.logging import get_logger

_GRAPHQL_URL = "https://ra.co/graphql"
_SCRAPE_DELAY = 4.0  # seconds between requests (~15 req/min)
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)
_PAGE_SIZE = 20  # RA returns up to 20 events per page
_MAX_RETRIES = 3
_RETRY_BACKOFF = 10.0  # seconds base backoff on rate-limit

# ---------------------------------------------------------------------------
# RA area ID mapping — verified IDs marked with (v), others need discovery.
# Use ``discover_area_id()`` or ``verify_area_ids()`` to validate at runtime.
# ---------------------------------------------------------------------------
RA_AREA_IDS: dict[str, int] = {
    # USA
    "chicago": 218,
    "detroit": 219,
    "new_york": 8,
    "san_francisco": 111,
    "minneapolis": 221,
    "madison": 335,
    "milwaukee": 336,
    "los_angeles": 17,
    # Europe
    "berlin": 34,  # (v)
    "london": 13,
    "manchester": 45,
    "cologne": 143,  # (v)
    "amsterdam": 29,
    "brussels": 48,
    "ibiza": 25,
    "barcelona": 44,
    # Asia
    "tokyo": 127,
}

# Display-friendly city names.
CITY_DISPLAY_NAMES: dict[str, str] = {
    "chicago": "Chicago",
    "detroit": "Detroit",
    "new_york": "New York",
    "san_francisco": "San Francisco",
    "minneapolis": "Minneapolis",
    "madison": "Madison",
    "milwaukee": "Milwaukee",
    "los_angeles": "Los Angeles",
    "berlin": "Berlin",
    "london": "London",
    "manchester": "Manchester",
    "cologne": "Cologne",
    "amsterdam": "Amsterdam",
    "brussels": "Brussels",
    "ibiza": "Ibiza",
    "barcelona": "Barcelona",
    "tokyo": "Tokyo",
}

# GraphQL query extracted from djb-gt/resident-advisor-events-scraper.
_GRAPHQL_QUERY = (
    "query GET_EVENT_LISTINGS("
    "$filters: FilterInputDtoInput, "
    "$filterOptions: FilterOptionsInputDtoInput, "
    "$page: Int, "
    "$pageSize: Int"
    ") {"
    "eventListings("
    "filters: $filters, "
    "filterOptions: $filterOptions, "
    "pageSize: $pageSize, "
    "page: $page"
    ") {"
    "data {"
    "id listingDate "
    "event {"
    "...eventListingsFields "
    "artists {id name __typename} "
    "__typename"
    "} __typename"
    "} "
    "totalResults __typename"
    "}"
    "}"
    "fragment eventListingsFields on Event {"
    "id date startTime endTime title contentUrl flyerFront "
    "isTicketed attending queueItEnabled newEventForm "
    "images {id filename alt type crop __typename} "
    "pick {id blurb __typename} "
    "venue {id name contentUrl live __typename} "
    "__typename"
    "}"
)


class RAGraphQLProvider:
    """Scrapes RA.co event listings via their undocumented GraphQL API.

    Parameters
    ----------
    http_client:
        Injected ``httpx.AsyncClient`` for testability and connection pooling.
    scrape_delay:
        Minimum seconds between requests (default 4.0).
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        scrape_delay: float = _SCRAPE_DELAY,
    ) -> None:
        self._http = http_client
        self._scrape_delay = scrape_delay
        self._last_request_time: float = 0.0
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _throttle(self) -> None:
        """Enforce minimum delay between requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if self._last_request_time > 0 and elapsed < self._scrape_delay:
            await asyncio.sleep(self._scrape_delay - elapsed)
        self._last_request_time = time.monotonic()

    async def _graphql_request(
        self,
        variables: dict[str, Any],
    ) -> dict[str, Any] | None:
        """POST to RA's GraphQL endpoint with retry logic.

        Returns the ``data`` portion of the response, or ``None`` on failure.
        """
        await self._throttle()

        headers = {
            "User-Agent": _USER_AGENT,
            "Content-Type": "application/json",
            "Referer": "https://ra.co/events",
            "Accept": "application/json",
        }
        payload = {
            "operationName": "GET_EVENT_LISTINGS",
            "variables": variables,
            "query": _GRAPHQL_QUERY,
        }

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = await self._http.post(
                    _GRAPHQL_URL,
                    json=payload,
                    headers=headers,
                    timeout=30.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    if "errors" in data:
                        self._logger.warning(
                            "ra_graphql_errors",
                            errors=data["errors"][:3],
                        )
                        return None
                    return data.get("data")

                if response.status_code in (403, 429):
                    backoff = _RETRY_BACKOFF * attempt
                    self._logger.warning(
                        "ra_rate_limited",
                        status=response.status_code,
                        attempt=attempt,
                        backoff_s=backoff,
                    )
                    await asyncio.sleep(backoff)
                    continue

                if response.status_code >= 500:
                    self._logger.warning(
                        "ra_server_error",
                        status=response.status_code,
                        attempt=attempt,
                    )
                    await asyncio.sleep(_RETRY_BACKOFF * attempt)
                    continue

                self._logger.warning(
                    "ra_unexpected_status",
                    status=response.status_code,
                )
                return None

            except httpx.HTTPError as exc:
                self._logger.warning(
                    "ra_request_failed",
                    error=str(exc),
                    attempt=attempt,
                )
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(_RETRY_BACKOFF * attempt)

        return None

    @staticmethod
    def _build_variables(
        area_id: int,
        start_date: str,
        end_date: str,
        page: int = 1,
    ) -> dict[str, Any]:
        """Build the GraphQL variables for an event listings query.

        Parameters
        ----------
        area_id:
            RA numeric area ID.
        start_date:
            ISO date string ``YYYY-MM-DD``.
        end_date:
            ISO date string ``YYYY-MM-DD``.
        page:
            1-indexed page number.
        """
        return {
            "filters": {
                "areas": {"eq": area_id},
                "listingDate": {
                    "gte": f"{start_date}T00:00:00.000Z",
                    "lte": f"{end_date}T23:59:59.999Z",
                },
            },
            "filterOptions": {"genre": True},
            "pageSize": _PAGE_SIZE,
            "page": page,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_events_page(
        self,
        area_id: int,
        start_date: str,
        end_date: str,
        page: int = 1,
        city: str = "",
    ) -> RAEventPage:
        """Fetch one page of events for a given area and date range.

        Parameters
        ----------
        area_id:
            RA numeric area ID.
        start_date:
            ISO date string ``YYYY-MM-DD`` (inclusive).
        end_date:
            ISO date string ``YYYY-MM-DD`` (inclusive).
        page:
            1-indexed page number.
        city:
            City label to stamp on each event.

        Returns
        -------
        RAEventPage
            Events on this page plus total result count.
        """
        variables = self._build_variables(area_id, start_date, end_date, page)
        data = await self._graphql_request(variables)

        if data is None or "eventListings" not in data:
            return RAEventPage(events=[], total_results=0)

        listings = data["eventListings"]
        total = listings.get("totalResults", 0)
        raw_items = listings.get("data", [])

        events: list[RAEvent] = []
        for item in raw_items:
            if not item:
                continue
            try:
                event = RAEvent.from_graphql(item, city=city)
                events.append(event)
            except Exception:
                self._logger.warning(
                    "ra_event_parse_failed",
                    event_data=str(item)[:200],
                )

        self._logger.debug(
            "ra_page_fetched",
            area_id=area_id,
            page=page,
            events_on_page=len(events),
            total_results=total,
        )
        return RAEventPage(events=events, total_results=total)

    async def fetch_all_events(
        self,
        area_id: int,
        start_date: str,
        end_date: str,
        city: str = "",
        on_page_complete: Callable[[int, int, int], None] | None = None,
    ) -> list[RAEvent]:
        """Fetch all events for an area within a date range, paginating.

        Parameters
        ----------
        area_id:
            RA numeric area ID.
        start_date:
            ISO date string ``YYYY-MM-DD``.
        end_date:
            ISO date string ``YYYY-MM-DD``.
        city:
            City label to stamp on each event.
        on_page_complete:
            Optional callback ``(page, fetched_so_far, total_results)``.

        Returns
        -------
        list[RAEvent]
            All events within the date range.
        """
        all_events: list[RAEvent] = []
        page = 1

        while True:
            result = await self.fetch_events_page(
                area_id=area_id,
                start_date=start_date,
                end_date=end_date,
                page=page,
                city=city,
            )

            all_events.extend(result.events)

            if on_page_complete:
                on_page_complete(page, len(all_events), result.total_results)

            # Stop when we've retrieved all results or got an empty page.
            if len(all_events) >= result.total_results or not result.events:
                break

            page += 1

        return all_events

    async def verify_area_id(self, area_id: int) -> bool:
        """Test whether an area ID returns events.

        Makes a single request for recent events to check if the area
        exists on RA.  Returns ``True`` if the query succeeds with
        ``totalResults > 0``.
        """
        today = date.today().isoformat()
        variables = self._build_variables(
            area_id=area_id,
            start_date="2024-01-01",
            end_date=today,
            page=1,
        )
        # Override page size to minimize data transfer.
        variables["pageSize"] = 1

        data = await self._graphql_request(variables)
        if data is None or "eventListings" not in data:
            return False

        total = data["eventListings"].get("totalResults", 0)
        return total > 0

    async def discover_area_id(
        self,
        search_range: range | None = None,
    ) -> dict[int, int]:
        """Probe a range of area IDs to find active ones.

        Useful for discovering IDs for cities not in :data:`RA_AREA_IDS`.
        Returns a mapping ``{area_id: total_results}``.
        """
        if search_range is None:
            search_range = range(1, 50)

        results: dict[int, int] = {}
        today = date.today().isoformat()

        for aid in search_range:
            variables = self._build_variables(
                area_id=aid,
                start_date="2024-01-01",
                end_date=today,
                page=1,
            )
            variables["pageSize"] = 1

            data = await self._graphql_request(variables)
            if data and "eventListings" in data:
                total = data["eventListings"].get("totalResults", 0)
                if total > 0:
                    results[aid] = total
                    self._logger.info(
                        "ra_area_discovered",
                        area_id=aid,
                        total_results=total,
                    )

        return results

    def get_provider_name(self) -> str:
        """Return ``'ra_graphql'``."""
        return "ra_graphql"

    def is_available(self) -> bool:
        """Always ``True`` — no API key required."""
        return True

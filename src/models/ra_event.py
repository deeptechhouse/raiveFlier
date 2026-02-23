"""Pydantic v2 models for Resident Advisor (RA.co) event data.

All models use frozen config (immutable) per project convention (Section 28).
These models represent the structured event data returned by RA's undocumented
GraphQL API, normalized for corpus ingestion.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RAArtist(BaseModel):
    """An artist listed on an RA event."""

    model_config = ConfigDict(frozen=True)

    ra_id: str = Field(description="RA's internal artist ID.")
    name: str = Field(description="Artist/DJ name as listed on RA.")


class RAVenue(BaseModel):
    """A venue from an RA event listing."""

    model_config = ConfigDict(frozen=True)

    ra_id: str = Field(description="RA's internal venue ID.")
    name: str = Field(description="Venue name.")
    area_name: str | None = Field(
        default=None, description="Area/city name from RA's taxonomy."
    )


class RAPromoter(BaseModel):
    """A promoter / event organizer from an RA event listing."""

    model_config = ConfigDict(frozen=True)

    ra_id: str = Field(description="RA's internal promoter ID.")
    name: str = Field(description="Promoter/organizer name.")


class RAEvent(BaseModel):
    """A single event scraped from RA.co.

    Represents the full event record as returned by the GraphQL API,
    normalized into a flat structure suitable for corpus text generation
    and ``DocumentChunk`` conversion.
    """

    model_config = ConfigDict(frozen=True)

    ra_id: str = Field(description="RA's internal event ID.")
    title: str = Field(description="Event title/name.")
    event_date: date = Field(description="Event date (UTC).")
    content_url: str | None = Field(
        default=None,
        description="Relative URL path on ra.co (e.g. '/events/123456').",
    )
    flyer_url: str | None = Field(
        default=None,
        description="URL to the event flyer image, if available.",
    )
    venue: RAVenue | None = Field(default=None, description="Venue details.")
    artists: list[RAArtist] = Field(
        default_factory=list, description="Artists on the lineup."
    )
    promoters: list[RAPromoter] = Field(
        default_factory=list, description="Event promoters/organizers."
    )
    city: str = Field(
        default="", description="City name (set during scrape from area config)."
    )
    attending: int = Field(
        default=0, description="Number of guests marked as attending on RA."
    )

    @classmethod
    def from_graphql(cls, item: dict[str, Any], city: str = "") -> RAEvent:
        """Parse a single event from RA's ``eventListings`` GraphQL response.

        Parameters
        ----------
        item:
            A single element from ``data.eventListings.data[]``.  Each
            element has an ``event`` key containing the nested event fields.
        city:
            City label to stamp onto the event.

        Returns
        -------
        RAEvent
        """
        event_data = item.get("event") or item
        venue_raw = event_data.get("venue") or {}

        venue: RAVenue | None = None
        if venue_raw.get("name"):
            venue = RAVenue(
                ra_id=str(venue_raw.get("id", "")),
                name=venue_raw["name"],
                area_name=None,
            )

        artists = [
            RAArtist(ra_id=str(a["id"]), name=a["name"])
            for a in (event_data.get("artists") or [])
            if a.get("name")
        ]

        # RA's eventListings query doesn't return promoters directly.
        # Promoters may be extractable from event detail pages later.
        promoters: list[RAPromoter] = []

        # Parse date — RA returns ISO-8601 strings.
        event_date_str = event_data.get("date") or item.get("listingDate") or ""
        try:
            event_date = datetime.fromisoformat(
                event_date_str.replace("Z", "+00:00")
            ).date()
        except (ValueError, AttributeError):
            event_date = date(2000, 1, 1)

        attending_raw = event_data.get("attending") or item.get("attending") or 0

        return cls(
            ra_id=str(event_data.get("id") or item.get("id", "")),
            title=event_data.get("title") or "Untitled Event",
            event_date=event_date,
            content_url=event_data.get("contentUrl"),
            flyer_url=event_data.get("flyerFront"),
            venue=venue,
            artists=artists,
            promoters=promoters,
            city=city,
            attending=attending_raw if isinstance(attending_raw, int) else 0,
        )


class RAEventPage(BaseModel):
    """A single page of results from an RA GraphQL event listings query."""

    model_config = ConfigDict(frozen=True)

    events: list[RAEvent] = Field(default_factory=list)
    total_results: int = Field(default=0)


class RAScrapeProgress(BaseModel):
    """Checkpoint state for resumable scraping.

    Serialized to JSON after each date-range batch completes so scraping
    can resume from the last successfully completed month on interruption.

    Progress is tracked **per date-range** — Phase A (2016-present) and
    Phase B (2003-2015) each get their own progress file while sharing
    the same events JSON.
    """

    model_config = ConfigDict(frozen=True)

    city: str = Field(description="City key (lowercase, e.g. 'chicago').")
    area_id: int = Field(description="RA area ID used for this city.")
    scrape_start_year: int = Field(
        description="Start year of the requested scrape range."
    )
    scrape_end_year: int = Field(
        description="End year of the requested scrape range."
    )
    total_events_scraped: int = Field(
        default=0, description="Cumulative events fetched so far."
    )
    last_completed_year: int | None = Field(
        default=None, description="Last year fully scraped."
    )
    last_completed_month: int | None = Field(
        default=None, description="Last month (1-12) fully scraped."
    )
    is_complete: bool = Field(
        default=False, description="True when the full date range is done."
    )
    started_at: str = Field(
        default="", description="ISO timestamp of scrape start."
    )
    updated_at: str = Field(
        default="", description="ISO timestamp of last checkpoint update."
    )

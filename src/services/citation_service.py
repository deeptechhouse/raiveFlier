"""Citation ranking, tier assignment, and verification service.

Centralises the logic for mapping source URLs and source types to a
six-tier citation hierarchy.  Provides ranking, automatic tier assignment
from URLs, and async link verification via HTTP HEAD requests.

Tier 1 is the highest authority (published books, first-hand accounts);
tier 6 is the lowest (community forums).
"""

from __future__ import annotations

import asyncio
import re

import httpx
import structlog

from src.models.analysis import Citation
from src.utils.logging import get_logger

# Timeout for HEAD requests during citation verification (seconds)
_VERIFY_TIMEOUT = 5.0

# Wayback Machine availability API endpoint
_WAYBACK_API = "https://archive.org/wayback/available"

# Source type keywords mapped to tiers (used when no URL is available)
_SOURCE_TYPE_TIERS: dict[str, int] = {
    "book": 1,
    "academic": 1,
    "interview": 2,
    "magazine": 2,
    "press": 2,
    "review": 2,
    "database": 3,
    "discography": 3,
    "wiki": 4,
    "video": 4,
    "social_media": 5,
    "forum": 5,
    "blog": 5,
    "unknown": 6,
}


class CitationService:
    """Manages citation ranking, tier assignment, and link verification.

    Citation tiers reflect a hierarchy of source authority, from published
    books (tier 1) down through community forums (tier 6).  URL patterns
    are used to auto-assign tiers for web-based sources.

    All external HTTP calls are made via ``httpx.AsyncClient`` and the
    service can be used without network access for pure ranking operations.
    """

    # -- Citation tier hierarchy (class constant) -----------------------------
    TIER_MAP: dict[str, int] = {
        "book": 1,
        "press": 2,
        "flier": 3,
        "database": 4,
        "web": 5,
        "forum": 6,
    }

    # Compiled URL patterns mapped to tier values for automatic assignment.
    # More specific patterns (e.g. ``discogs.com/forum``) appear before
    # their broader parent domain pattern to ensure correct matching.
    URL_TIER_PATTERNS: list[tuple[re.Pattern[str], int]] = [
        # Tier 2 — established music press
        (re.compile(r"djmag\.com", re.IGNORECASE), 2),
        (re.compile(r"residentadvisor\.net", re.IGNORECASE), 2),
        (re.compile(r"mixmag\.net", re.IGNORECASE), 2),
        (re.compile(r"xlr8r\.com", re.IGNORECASE), 2),
        (re.compile(r"thequietus\.com", re.IGNORECASE), 2),
        (re.compile(r"pitchfork\.com", re.IGNORECASE), 2),
        (re.compile(r"factmag\.com", re.IGNORECASE), 2),
        # Tier 3 — event / flier archives
        (re.compile(r"19hz\.info", re.IGNORECASE), 3),
        (re.compile(r"flyerarchive", re.IGNORECASE), 3),
        # Tier 6 — forums (must precede broader domain matches below)
        (re.compile(r"discogs\.com/forum", re.IGNORECASE), 6),
        (re.compile(r"reddit\.com", re.IGNORECASE), 6),
        # Tier 4 — music databases
        (re.compile(r"discogs\.com", re.IGNORECASE), 4),
        (re.compile(r"musicbrainz\.org", re.IGNORECASE), 4),
        # Tier 5 — general web
        (re.compile(r"wikipedia\.org", re.IGNORECASE), 5),
    ]

    def __init__(self) -> None:
        self._logger: structlog.BoundLogger = get_logger(__name__)

    # -- Public API -----------------------------------------------------------

    def rank_citations(self, citations: list[Citation]) -> list[Citation]:
        """Sort citations by tier (ascending) then by date (newest first).

        Parameters
        ----------
        citations:
            The citations to rank.

        Returns
        -------
        list[Citation]
            A new list sorted by (tier ASC, source_date DESC).  Citations
            without a date sort after those with dates within the same tier.
        """

        def _sort_key(c: Citation) -> tuple[int, int]:
            # For date: negate ordinal so newer dates come first.
            # Citations without a date get a very low value (0) so they
            # sort after dated citations within the same tier.
            date_val = -c.source_date.toordinal() if c.source_date else 0
            return (c.tier, date_val)

        ranked = sorted(citations, key=_sort_key)

        self._logger.debug(
            "Citations ranked",
            total=len(ranked),
            tiers_present=sorted({c.tier for c in ranked}),
        )

        return ranked

    def assign_tier_from_url(self, url: str) -> int:
        """Determine the citation tier for a URL based on domain patterns.

        Parameters
        ----------
        url:
            The source URL to classify.

        Returns
        -------
        int
            The assigned tier (1–6).  Defaults to 5 (general web) when
            no pattern matches.
        """
        for pattern, tier in self.URL_TIER_PATTERNS:
            if pattern.search(url):
                return tier
        return 5

    def assign_tier(
        self,
        source_url: str | None = None,
        source_type: str | None = None,
    ) -> int:
        """Determine the citation tier for a given source.

        Checks URL patterns first, then falls back to source-type keyword
        lookup.

        Parameters
        ----------
        source_url:
            URL of the source (preferred signal).
        source_type:
            Keyword describing the source type (fallback signal).

        Returns
        -------
        int
            Citation tier between 1 (highest) and 6 (lowest).
        """
        if source_url:
            tier = self.assign_tier_from_url(source_url)
            if tier != 5:
                return tier

        if source_type:
            normalised = source_type.strip().lower().replace(" ", "_")
            if normalised in _SOURCE_TYPE_TIERS:
                return _SOURCE_TYPE_TIERS[normalised]

        if source_url:
            return 5

        return 6

    def build_citation(
        self,
        text: str,
        source_name: str,
        source_type: str = "unknown",
        source_url: str | None = None,
        source_date: str | None = None,
        page_number: str | None = None,
    ) -> Citation:
        """Build a :class:`Citation` with an automatically assigned tier.

        Parameters
        ----------
        text:
            The claim or fact this citation supports.
        source_name:
            Human-readable name of the source.
        source_type:
            Keyword describing the source category.
        source_url:
            URL of the source (optional).
        source_date:
            ISO-format date string (optional).
        page_number:
            Page reference (optional, for books/magazines).

        Returns
        -------
        Citation
            Immutable citation model with tier assigned.
        """
        from datetime import date as _date

        tier = self.assign_tier(source_url=source_url, source_type=source_type)

        parsed_date: _date | None = None
        if source_date:
            try:
                parsed_date = _date.fromisoformat(source_date)
            except ValueError:
                self._logger.warning(
                    "invalid_citation_date",
                    source_name=source_name,
                    raw_date=source_date,
                )

        return Citation(
            text=text,
            source_type=source_type,
            source_name=source_name,
            source_url=source_url,
            source_date=parsed_date,
            tier=tier,
            page_number=page_number,
        )

    async def verify_citation(self, citation: Citation) -> tuple[Citation, bool]:
        """Check whether a citation's URL is still accessible.

        Sends an HTTP HEAD request to the source URL with a 5-second timeout.
        If the URL is inaccessible, attempts to locate it via the Wayback
        Machine availability API.

        Parameters
        ----------
        citation:
            The citation to verify.

        Returns
        -------
        tuple[Citation, bool]
            A ``(citation, is_accessible)`` tuple.  ``is_accessible`` is
            ``True`` if the original URL or a Wayback Machine snapshot
            responded with a success/redirect status.
        """
        if not citation.source_url:
            return (citation, False)

        accessible = await self._head_check(citation.source_url)
        if accessible:
            self._logger.debug(
                "citation_url_accessible",
                url=citation.source_url,
            )
            return (citation, True)

        # Original URL is down — try Wayback Machine
        wayback_ok = await self._check_wayback(citation.source_url)
        if wayback_ok:
            self._logger.debug(
                "citation_url_found_in_wayback",
                url=citation.source_url,
            )
            return (citation, True)

        self._logger.debug(
            "citation_url_inaccessible",
            url=citation.source_url,
        )
        return (citation, False)

    async def verify_batch(
        self,
        citations: list[Citation],
        max_concurrent: int = 10,
    ) -> list[tuple[Citation, bool]]:
        """Verify all citations concurrently with a concurrency limit.

        Parameters
        ----------
        citations:
            The citations to verify.
        max_concurrent:
            Maximum number of concurrent verification requests.

        Returns
        -------
        list[tuple[Citation, bool]]
            A list of ``(citation, is_accessible)`` tuples in the same
            order as the input.
        """
        if not citations:
            return []

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _limited_verify(
            c: Citation,
        ) -> tuple[Citation, bool]:
            async with semaphore:
                return await self.verify_citation(c)

        tasks = [_limited_verify(c) for c in citations]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        verified: list[tuple[Citation, bool]] = []
        for citation, result in zip(citations, results, strict=True):
            if isinstance(result, Exception):
                self._logger.debug(
                    "verification_task_exception",
                    url=citation.source_url,
                    error=str(result),
                )
                verified.append((citation, False))
            else:
                verified.append(result)  # type: ignore[arg-type]

        self._logger.info(
            "bulk_citation_verification_complete",
            total=len(verified),
            accessible=sum(1 for _, ok in verified if ok),
            inaccessible=sum(1 for _, ok in verified if not ok),
        )

        return verified

    def format_citation(self, citation: Citation) -> str:
        """Format a citation as a human-readable string.

        Formatting varies by tier:

        - **Tier 1** (book/academic): ``Author, Title, Publisher, Year, p.XX``
        - **Tier 2** (press): ``Title — Publication, Date. URL``
        - **Tier 4** (database): ``Discogs: Artist — Release. URL``
        - **Tiers 5–6** (web/forum): ``Title, Source. URL``

        Parameters
        ----------
        citation:
            The citation to format.

        Returns
        -------
        str
            Human-readable citation string.
        """
        date_str = citation.source_date.isoformat() if citation.source_date else ""
        url_str = citation.source_url or ""

        if citation.tier == 1:
            # Book / academic format
            parts: list[str] = [citation.source_name, citation.text]
            if date_str:
                parts.append(date_str)
            if citation.page_number:
                parts.append(f"p.{citation.page_number}")
            return ", ".join(p for p in parts if p)

        if citation.tier == 2:
            # Press format: Title — Publication, Date. URL
            result = f"{citation.text} — {citation.source_name}"
            if date_str:
                result += f", {date_str}"
            if url_str:
                result += f". {url_str}"
            return result

        if citation.tier == 4:
            # Database format: Discogs: Artist — Release. URL
            result = f"{citation.source_name}: {citation.text}"
            if url_str:
                result += f". {url_str}"
            return result

        # Tiers 3, 5, 6 — general web / forum / flier archive
        result = f"{citation.text}, {citation.source_name}"
        if url_str:
            result += f". {url_str}"
        return result

    # -- Private helpers ------------------------------------------------------

    async def _head_check(self, url: str) -> bool:
        """Send an HTTP HEAD request and return whether the URL is reachable.

        Parameters
        ----------
        url:
            The URL to check.

        Returns
        -------
        bool
            ``True`` if the server responded with ``2xx`` or ``3xx``.
        """
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=_VERIFY_TIMEOUT,
            ) as client:
                response = await client.head(url)
                return response.status_code < 400
        except (httpx.HTTPError, httpx.InvalidURL, Exception):
            return False

    async def _check_wayback(self, url: str) -> bool:
        """Check the Wayback Machine for an archived snapshot of a URL.

        Parameters
        ----------
        url:
            The original URL to look up.

        Returns
        -------
        bool
            ``True`` if the Wayback Machine has an accessible snapshot.
        """
        try:
            async with httpx.AsyncClient(
                timeout=_VERIFY_TIMEOUT,
            ) as client:
                response = await client.get(
                    _WAYBACK_API,
                    params={"url": url},
                )
                if response.status_code != 200:
                    return False
                data = response.json()
                snapshot = data.get("archived_snapshots", {}).get("closest", {})
                return snapshot.get("available", False) is True
        except (httpx.HTTPError, ValueError, Exception):
            return False

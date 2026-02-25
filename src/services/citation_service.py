"""Citation ranking, tier assignment, and verification service.

Centralises the logic for mapping source URLs and source types to a
six-tier citation hierarchy.  Provides ranking, automatic tier assignment
from URLs, and async link verification via HTTP HEAD requests.

Tier 1 is the highest authority (published books, first-hand accounts);
tier 6 is the lowest (community forums).

Architecture overview for junior developers
--------------------------------------------
This service implements a six-tier citation authority system used across
the entire pipeline.  Every source (URL, book, article, forum post) is
assigned a numerical tier from 1 (most authoritative) to 6 (least):

    Tier 1 -- Book / Academic   (published books, first-hand accounts)
    Tier 2 -- Press             (DJ Mag, Resident Advisor, Mixmag, etc.)
    Tier 3 -- Flier / Archive   (19hz.info, flyer archives -- the flier itself)
    Tier 4 -- Database          (Discogs, MusicBrainz -- structured metadata)
    Tier 5 -- Web               (Wikipedia, general websites)
    Tier 6 -- Forum             (Reddit, Discogs forums, community discussion)

Tier assignment uses two strategies, checked in order:
  1. URL pattern matching -- regex patterns mapped to known domains
  2. Source-type keyword fallback -- e.g., "book" -> tier 1

The service also provides async URL verification: it sends an HTTP HEAD
request to check if a citation's URL is still live.  If the URL is dead,
it falls back to the Wayback Machine availability API to check for an
archived snapshot.  This two-step verification ensures citations remain
valid even when original pages go offline.
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

# Source type keywords mapped to tiers (used when no URL is available).
# This is the fallback lookup when URL pattern matching does not apply
# (e.g., the source is a book with no URL).  The keyword comes from the
# source_type field on a Citation model instance.
_SOURCE_TYPE_TIERS: dict[str, int] = {
    "book": 1,
    "academic": 1,
    "interview": 2,
    "magazine": 2,
    "press": 2,
    "review": 2,
    "database": 3,
    "discography": 3,
    "event_listing": 3,  # RA event listings — tier 3 (same as database sources)
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
    # The canonical tier map.  Lower number = higher authority.
    # Why this ordering?  Books and first-hand interviews are primary sources
    # that cannot be edited after publication; press articles are reviewed;
    # flier archives are physical evidence; databases are crowd-sourced but
    # moderated; general web is unmoderated; forums are ephemeral opinion.
    TIER_MAP: dict[str, int] = {
        "book": 1,
        "press": 2,
        "flier": 3,
        "database": 4,
        "web": 5,
        "forum": 6,
    }

    # Compiled URL patterns mapped to tier values for automatic assignment.
    # ORDERING MATTERS: more specific patterns (e.g. ``discogs.com/forum``)
    # appear BEFORE their broader parent domain (``discogs.com``) to ensure
    # the first match wins correctly.  A forum thread on Discogs is tier 6,
    # not tier 4.
    URL_TIER_PATTERNS: list[tuple[re.Pattern[str], int]] = [
        # Tier 2 -- established music press (editorially reviewed content)
        (re.compile(r"djmag\.com", re.IGNORECASE), 2),
        (re.compile(r"residentadvisor\.net", re.IGNORECASE), 2),
        (re.compile(r"mixmag\.net", re.IGNORECASE), 2),
        (re.compile(r"xlr8r\.com", re.IGNORECASE), 2),
        (re.compile(r"thequietus\.com", re.IGNORECASE), 2),
        (re.compile(r"pitchfork\.com", re.IGNORECASE), 2),
        (re.compile(r"factmag\.com", re.IGNORECASE), 2),
        # Tier 3 -- event / flier archives (physical evidence of events)
        (re.compile(r"19hz\.info", re.IGNORECASE), 3),
        (re.compile(r"flyerarchive", re.IGNORECASE), 3),
        # Tier 6 -- forums.  NOTE: discogs.com/forum MUST come before the
        # broader discogs.com pattern below, otherwise forums would be
        # misclassified as tier 4 database content.
        (re.compile(r"discogs\.com/forum", re.IGNORECASE), 6),
        (re.compile(r"reddit\.com", re.IGNORECASE), 6),
        # Tier 4 -- music databases (structured, crowd-sourced, moderated)
        (re.compile(r"discogs\.com", re.IGNORECASE), 4),
        (re.compile(r"musicbrainz\.org", re.IGNORECASE), 4),
        # Tier 5 -- general web (unmoderated or loosely moderated)
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
            # Composite sort: (tier ASC, date DESC within each tier).
            # Negating the date ordinal makes newer dates sort first.
            # Citations without a date get 0, placing them after dated
            # citations within the same tier (since negative ordinals
            # are always less than 0).
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
        # Strategy 1: URL pattern matching (preferred -- most specific signal).
        # If the URL matches a known domain, use that tier.  The default
        # return from assign_tier_from_url is 5 ("general web"), so we only
        # accept the result if it is NOT 5 (meaning a specific pattern hit).
        if source_url:
            tier = self.assign_tier_from_url(source_url)
            if tier != 5:
                return tier

        # Strategy 2: source-type keyword lookup (fallback when URL is
        # unknown or matched as generic "web").
        if source_type:
            normalised = source_type.strip().lower().replace(" ", "_")
            if normalised in _SOURCE_TYPE_TIERS:
                return _SOURCE_TYPE_TIERS[normalised]

        # If we have a URL but no keyword match, default to tier 5 (web).
        if source_url:
            return 5

        # No URL, no recognized type -- lowest possible authority.
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
        # No URL means nothing to verify -- return immediately.
        if not citation.source_url:
            return (citation, False)

        # Step 1: Try the original URL with an HTTP HEAD request.
        # HEAD is used (not GET) to avoid downloading large page bodies.
        accessible = await self._head_check(citation.source_url)
        if accessible:
            self._logger.debug(
                "citation_url_accessible",
                url=citation.source_url,
            )
            return (citation, True)

        # Step 2: Original URL is down -- fall back to the Wayback Machine.
        # The Internet Archive's availability API checks if an archived
        # snapshot exists.  This is crucial for rave history research where
        # many original pages from the 90s/2000s no longer exist.
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

        # Semaphore-based concurrency limiter: prevents flooding external
        # servers with too many simultaneous HEAD requests.  Default of 10
        # balances throughput with politeness.
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
                # The Wayback Machine API returns a JSON structure like:
                # {"archived_snapshots": {"closest": {"available": true, ...}}}
                # We check the "available" flag on the closest snapshot.
                data = response.json()
                snapshot = data.get("archived_snapshots", {}).get("closest", {})
                return snapshot.get("available", False) is True
        except (httpx.HTTPError, ValueError, Exception):
            return False

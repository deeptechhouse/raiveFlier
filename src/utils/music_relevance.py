# ─── MUSIC RELEVANCE UTILITY ───────────────────────────────────────────
#
# Shared module for determining whether a web search result is relevant to
# rave / electronic music culture.  Extracted from artist_researcher.py so
# that both the research pipeline and the corpus sidebar web-search tier
# can reuse the same domain-specific relevance logic.
#
# Two compiled regexes power the check:
#   • _MUSIC_DOMAINS   — URL patterns for known music sites (always relevant)
#   • _MUSIC_RELEVANCE_TERMS — keyword signals in titles/snippets
#
# The public function ``is_music_relevant()`` is the single entry point.
# ────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import re


# Domains known to be music-related — results from these always pass the
# relevance check without needing keyword signals in the title/snippet.
MUSIC_DOMAINS = re.compile(
    r"ra\.co|residentadvisor|djmag\.com|mixmag\.net|xlr8r\.com|pitchfork\.com|"
    r"thequietus\.com|factmag\.com|factmagazine|discogs\.com|musicbrainz\.org|"
    r"bandcamp\.com|beatport\.com|soundcloud\.com|youtube\.com|youtu\.be|"
    r"boilerroom\.tv|traxsource\.com|juno\.co\.uk|boomkat\.com|"
    r"electronicbeats\.net|djtechtools\.com|attackmagazine\.com",
    re.IGNORECASE,
)

# Terms that signal music/electronic-music relevance in titles and snippets
MUSIC_RELEVANCE_TERMS = re.compile(
    r"\b(?:dj|producer|remix|techno|house|drum\s*(?:and|&|n)\s*bass|"
    r"electronic\s*music|rave|club|vinyl|label|release|mix|track|"
    r"bpm|ep\b|lp\b|album|record|boiler\s*room|soundsystem|"
    r"beatport|discogs|bandcamp|resident\s*advisor|soundcloud|"
    r"dance\s*music|edm|jungle|garage|dubstep|acid|trance|"
    r"breakbeat|ambient|industrial|synth|turntable|decks|"
    r"festival|warehouse|nightclub|set\b|lineup|b2b)\b",
    re.IGNORECASE,
)


def is_music_relevant(url: str, title: str = "", snippet: str = "") -> bool:
    """Check whether a web search result is plausibly about music/electronic music.

    Known music domains always pass.  For other domains, at least one
    music-relevance keyword must appear in the title or snippet text.

    Parameters
    ----------
    url:
        The result URL to check against known music domains.
    title:
        The page title (may be empty).
    snippet:
        A text excerpt/description from the search result (may be empty).

    Returns
    -------
    bool
        ``True`` if the result is likely music-relevant.
    """
    # Known music domains always pass
    if MUSIC_DOMAINS.search(url):
        return True

    # Check title and snippet for music relevance signals
    text_to_check = f"{title} {snippet}"
    return bool(MUSIC_RELEVANCE_TERMS.search(text_to_check))

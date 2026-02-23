"""Music-database provider implementations.

Five concrete implementations of IMusicDatabaseProvider, tried in priority
order by artist_researcher.py for looking up artist discographies and labels:

    1. DiscogsAPIProvider     — official Discogs REST API (requires DISCOGS_CONSUMER_KEY
       and DISCOGS_CONSUMER_SECRET). Most comprehensive data. Rate limit: 60 req/min.
    2. DiscogsScrapeProvider  — Discogs web scraping fallback (no API key needed).
       Slower and less data, but works when API keys aren't configured.
    3. MusicBrainzProvider    — MusicBrainz open API (requires MUSICBRAINZ_CONTACT email).
       Free and open-source. Rate limit: 1 req/sec. Good for cross-referencing.
    4. BandcampProvider       — Bandcamp HTML scraping (no API). Finds independent/
       underground artists that aren't on Discogs or MusicBrainz.
    5. BeatportProvider       — Beatport HTML scraping (no API). Electronic music focus.
       Good for DJ-oriented release data (tracks, BPM).

Each provider returns the same types (ArtistSearchResult, Release, Label) so
the researcher can seamlessly merge results from multiple sources.
"""

from src.providers.music_db.bandcamp_provider import BandcampProvider
from src.providers.music_db.beatport_provider import BeatportProvider
from src.providers.music_db.discogs_api_provider import DiscogsAPIProvider
from src.providers.music_db.discogs_scrape_provider import DiscogsScrapeProvider
from src.providers.music_db.musicbrainz_provider import MusicBrainzProvider

__all__ = [
    "BandcampProvider",
    "BeatportProvider",
    "DiscogsAPIProvider",
    "DiscogsScrapeProvider",
    "MusicBrainzProvider",
]

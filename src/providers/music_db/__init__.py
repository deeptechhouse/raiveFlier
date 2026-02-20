"""Music-database provider implementations."""

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

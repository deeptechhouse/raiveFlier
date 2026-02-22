"""Unit tests for music database provider adapters."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.settings import Settings


def _settings(**overrides) -> Settings:
    defaults = {
        "discogs_consumer_key": "test-key",
        "discogs_consumer_secret": "test-secret",
        "musicbrainz_app_name": "raiveFlier-test",
        "musicbrainz_app_version": "0.1.0",
        "musicbrainz_contact": "test@test.com",
    }
    defaults.update(overrides)
    return Settings(**defaults)


# ======================================================================
# Discogs API Provider
# ======================================================================


class TestDiscogsAPIProvider:
    @pytest.fixture()
    def settings(self) -> Settings:
        return _settings()

    def test_get_provider_name(self, settings: Settings) -> None:
        from src.providers.music_db.discogs_api_provider import DiscogsAPIProvider
        provider = DiscogsAPIProvider(settings)
        assert provider.get_provider_name() == "discogs_api"

    def test_is_available_with_keys(self, settings: Settings) -> None:
        from src.providers.music_db.discogs_api_provider import DiscogsAPIProvider
        provider = DiscogsAPIProvider(settings)
        assert provider.is_available() is True

    def test_is_available_without_keys(self) -> None:
        from src.providers.music_db.discogs_api_provider import DiscogsAPIProvider
        provider = DiscogsAPIProvider(_settings(discogs_consumer_key="", discogs_consumer_secret=""))
        assert provider.is_available() is False

    @pytest.mark.asyncio
    async def test_search_artist_success(self, settings: Settings) -> None:
        from src.providers.music_db.discogs_api_provider import DiscogsAPIProvider

        # _search_sync iterates results directly with enumerate(), not .page()
        mock_result = MagicMock()
        mock_result.id = 12345
        mock_result.data = {"title": "Carl Cox"}

        mock_search_results = MagicMock()
        mock_search_results.__iter__ = MagicMock(return_value=iter([mock_result]))

        mock_client = MagicMock()
        mock_client.search.return_value = mock_search_results

        with patch("src.providers.music_db.discogs_api_provider.discogs_client") as mock_dc:
            mock_dc.Client.return_value = mock_client
            provider = DiscogsAPIProvider(settings)
            results = await provider.search_artist("Carl Cox")

        assert len(results) >= 1
        assert results[0].name == "Carl Cox"

    @pytest.mark.asyncio
    async def test_search_artist_empty(self, settings: Settings) -> None:
        from src.providers.music_db.discogs_api_provider import DiscogsAPIProvider

        mock_search_results = MagicMock()
        mock_search_results.__iter__ = MagicMock(return_value=iter([]))

        mock_client = MagicMock()
        mock_client.search.return_value = mock_search_results

        with patch("src.providers.music_db.discogs_api_provider.discogs_client") as mock_dc:
            mock_dc.Client.return_value = mock_client
            provider = DiscogsAPIProvider(settings)
            results = await provider.search_artist("xyznonexistent")

        assert results == []

    @pytest.mark.asyncio
    async def test_get_artist_releases(self, settings: Settings) -> None:
        from src.providers.music_db.discogs_api_provider import DiscogsAPIProvider

        # _get_releases_sync iterates artist.releases directly with enumerate()
        mock_release_item = MagicMock()
        mock_release_item.data = {
            "id": 1,
            "title": "Phat Trax",
            "year": 1995,
            "label": "React",
            "format": '12"',
        }

        mock_artist = MagicMock()
        mock_artist.releases.__iter__ = MagicMock(return_value=iter([mock_release_item]))

        mock_client = MagicMock()
        mock_client.artist.return_value = mock_artist

        with patch("src.providers.music_db.discogs_api_provider.discogs_client") as mock_dc:
            mock_dc.Client.return_value = mock_client
            provider = DiscogsAPIProvider(settings)
            releases = await provider.get_artist_releases("12345")

        assert len(releases) >= 1
        assert releases[0].title == "Phat Trax"

    @pytest.mark.asyncio
    async def test_get_artist_labels(self, settings: Settings) -> None:
        from src.providers.music_db.discogs_api_provider import DiscogsAPIProvider

        # _get_releases_sync iterates artist.releases directly
        mock_release_item = MagicMock()
        mock_release_item.data = {
            "id": 2,
            "title": "Some Track",
            "year": 1996,
            "label": "Intec",
        }

        mock_artist = MagicMock()
        mock_artist.releases.__iter__ = MagicMock(return_value=iter([mock_release_item]))

        # _search_label_sync iterates label search results directly
        mock_label_result = MagicMock()
        mock_label_result.id = 100
        mock_label_result.data = {"title": "Intec"}

        mock_label_search = MagicMock()
        mock_label_search.__iter__ = MagicMock(return_value=iter([mock_label_result]))

        mock_client = MagicMock()
        mock_client.artist.return_value = mock_artist
        mock_client.search.return_value = mock_label_search

        with patch("src.providers.music_db.discogs_api_provider.discogs_client") as mock_dc:
            mock_dc.Client.return_value = mock_client
            provider = DiscogsAPIProvider(settings)
            labels = await provider.get_artist_labels("12345")

        assert len(labels) >= 1
        assert labels[0].name == "Intec"


# ======================================================================
# Discogs Scrape Provider
# ======================================================================


class TestDiscogsScrapeProvider:
    def test_get_provider_name(self) -> None:
        from src.providers.music_db.discogs_scrape_provider import DiscogsScrapeProvider
        mock_client = AsyncMock()
        provider = DiscogsScrapeProvider(mock_client)
        assert provider.get_provider_name() == "discogs_scrape"

    def test_is_available(self) -> None:
        from src.providers.music_db.discogs_scrape_provider import DiscogsScrapeProvider
        mock_client = AsyncMock()
        provider = DiscogsScrapeProvider(mock_client)
        assert provider.is_available() is True

    @pytest.mark.asyncio
    async def test_search_artist_success(self) -> None:
        from src.providers.music_db.discogs_scrape_provider import DiscogsScrapeProvider

        html = """
        <div id="search_results">
            <div class="card" data-object-type="artist">
                <a href="/artist/12345-Carl-Cox" class="search_result_title">Carl Cox</a>
            </div>
        </div>
        """
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = DiscogsScrapeProvider(mock_client)
        results = await provider.search_artist("Carl Cox")

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_artist_http_error(self) -> None:
        from src.providers.music_db.discogs_scrape_provider import DiscogsScrapeProvider
        import httpx

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.HTTPStatusError(
            "403 Forbidden", request=MagicMock(), response=MagicMock(status_code=403)
        ))

        provider = DiscogsScrapeProvider(mock_client)
        results = await provider.search_artist("Test")

        assert results == []

    @pytest.mark.asyncio
    async def test_get_artist_releases_success(self) -> None:
        """Test parsing releases from a Discogs artist discography table."""
        from src.providers.music_db.discogs_scrape_provider import DiscogsScrapeProvider

        html = """
        <html><body>
        <table>
            <tr>
                <td class="label"><a href="/label/123-Intec">Intec</a></td>
                <td class="catno">INTEC-001</td>
                <td><a href="/release/99999-Phat-Trax">Phat Trax</a></td>
                <td class="year">1995</td>
                <td class="format">12"</td>
            </tr>
            <tr>
                <td class="label"><a href="/label/456-React">React</a></td>
                <td class="catno">REACT-002</td>
                <td><a href="/master/88888-Two-Paintings">Two Paintings</a></td>
                <td class="year">1996</td>
                <td class="format">12", EP</td>
            </tr>
        </table>
        </body></html>
        """
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = DiscogsScrapeProvider(mock_client)
        releases = await provider.get_artist_releases("12345")

        assert len(releases) == 2
        assert releases[0].title == "Phat Trax"
        assert releases[0].label == "Intec"
        assert releases[0].catalog_number == "INTEC-001"
        assert releases[0].year == 1995
        assert releases[0].format == '12"'
        assert releases[1].title == "Two Paintings"
        assert releases[1].label == "React"

    @pytest.mark.asyncio
    async def test_get_artist_releases_before_date(self) -> None:
        """Test that releases after before_date are filtered out."""
        from src.providers.music_db.discogs_scrape_provider import DiscogsScrapeProvider

        html = """
        <html><body>
        <table>
            <tr>
                <td class="label">Label A</td>
                <td><a href="/release/111-Old-Release">Old Release</a></td>
                <td class="year">1995</td>
            </tr>
            <tr>
                <td class="label">Label B</td>
                <td><a href="/release/222-New-Release">New Release</a></td>
                <td class="year">2024</td>
            </tr>
        </table>
        </body></html>
        """
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = DiscogsScrapeProvider(mock_client)
        releases = await provider.get_artist_releases(
            "12345", before_date=date(2000, 1, 1)
        )

        assert len(releases) == 1
        assert releases[0].title == "Old Release"

    @pytest.mark.asyncio
    async def test_get_artist_releases_empty(self) -> None:
        """Test get_artist_releases returns empty on page fetch failure."""
        from src.providers.music_db.discogs_scrape_provider import DiscogsScrapeProvider

        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = DiscogsScrapeProvider(mock_client)
        releases = await provider.get_artist_releases("12345")

        assert releases == []

    @pytest.mark.asyncio
    async def test_get_artist_labels_success(self) -> None:
        """Test extracting unique labels with Discogs IDs from discography."""
        from src.providers.music_db.discogs_scrape_provider import DiscogsScrapeProvider

        html = """
        <html><body>
        <table>
            <tr>
                <td class="label"><a href="/label/100-Intec">Intec</a></td>
                <td><a href="/release/1-Track-A">Track A</a></td>
                <td class="year">1996</td>
            </tr>
            <tr>
                <td class="label"><a href="/label/200-Drumcode">Drumcode</a></td>
                <td><a href="/release/2-Track-B">Track B</a></td>
                <td class="year">1998</td>
            </tr>
            <tr>
                <td class="label"><a href="/label/100-Intec">Intec</a></td>
                <td><a href="/release/3-Track-C">Track C</a></td>
                <td class="year">2000</td>
            </tr>
        </table>
        </body></html>
        """
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = DiscogsScrapeProvider(mock_client)
        labels = await provider.get_artist_labels("12345")

        label_names = [lb.name for lb in labels]
        assert "Intec" in label_names
        assert "Drumcode" in label_names
        # Intec appears twice in HTML but should be deduplicated
        assert label_names.count("Intec") == 1
        # Labels should have discogs_id from the URL
        intec = next(lb for lb in labels if lb.name == "Intec")
        assert intec.discogs_id == 100
        assert intec.discogs_url == "https://www.discogs.com/label/100"

    @pytest.mark.asyncio
    async def test_get_artist_labels_empty(self) -> None:
        """Test get_artist_labels returns empty on page fetch failure."""
        from src.providers.music_db.discogs_scrape_provider import DiscogsScrapeProvider

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = DiscogsScrapeProvider(mock_client)
        labels = await provider.get_artist_labels("12345")

        assert labels == []

    @pytest.mark.asyncio
    async def test_get_release_details_success(self) -> None:
        """Test scraping an individual release page for full details."""
        from src.providers.music_db.discogs_scrape_provider import DiscogsScrapeProvider

        html = """
        <html><body>
            <h1>Carl Cox - Phat Trax</h1>
            <div class="label"><a href="/label/100">Intec</a></div>
            <span class="catno">INTEC-001</span>
            <span class="year">1995</span>
            <span class="format">12"</span>
            <div class="genre">
                <a href="/genre/electronic">Electronic</a>
            </div>
            <div class="style">
                <a href="/style/techno">Techno</a>
                <a href="/style/hard-techno">Hard Techno</a>
            </div>
        </body></html>
        """
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = DiscogsScrapeProvider(mock_client)
        release = await provider.get_release_details("99999")

        assert release is not None
        assert release.title == "Carl Cox - Phat Trax"
        assert release.label == "Intec"
        assert release.catalog_number == "INTEC-001"
        assert release.year == 1995
        assert release.format == '12"'
        assert "Electronic" in release.genres
        assert "Techno" in release.styles
        assert "Hard Techno" in release.styles

    @pytest.mark.asyncio
    async def test_get_release_details_not_found(self) -> None:
        """Test get_release_details returns None when page fetch fails."""
        from src.providers.music_db.discogs_scrape_provider import DiscogsScrapeProvider

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = DiscogsScrapeProvider(mock_client)
        release = await provider.get_release_details("nonexistent")

        assert release is None

    def test_safe_int_valid(self) -> None:
        """Test _safe_int with valid integer strings."""
        from src.providers.music_db.discogs_scrape_provider import DiscogsScrapeProvider

        assert DiscogsScrapeProvider._safe_int("1995") == 1995
        assert DiscogsScrapeProvider._safe_int("  2020  ") == 2020

    def test_safe_int_invalid(self) -> None:
        """Test _safe_int returns None for invalid inputs."""
        from src.providers.music_db.discogs_scrape_provider import DiscogsScrapeProvider

        assert DiscogsScrapeProvider._safe_int("not-a-number") is None
        assert DiscogsScrapeProvider._safe_int("") is None

    def test_extract_artist_id(self) -> None:
        """Test _extract_artist_id with valid and invalid hrefs."""
        from src.providers.music_db.discogs_scrape_provider import DiscogsScrapeProvider

        assert DiscogsScrapeProvider._extract_artist_id("/artist/12345-Carl-Cox") == "12345"
        assert DiscogsScrapeProvider._extract_artist_id("/some/other/path") is None


# ======================================================================
# Bandcamp Provider
# ======================================================================


class TestBandcampProvider:
    def test_get_provider_name(self) -> None:
        from src.providers.music_db.bandcamp_provider import BandcampProvider
        mock_client = AsyncMock()
        provider = BandcampProvider(mock_client)
        assert provider.get_provider_name() == "bandcamp"

    def test_is_available(self) -> None:
        from src.providers.music_db.bandcamp_provider import BandcampProvider
        mock_client = AsyncMock()
        provider = BandcampProvider(mock_client)
        assert provider.is_available() is True

    @pytest.mark.asyncio
    async def test_search_artist_success(self) -> None:
        from src.providers.music_db.bandcamp_provider import BandcampProvider

        html = """
        <div class="results">
            <li class="searchresult band">
                <div class="result-info">
                    <div class="heading"><a href="https://carlcox.bandcamp.com">Carl Cox</a></div>
                    <div class="subhead">artist</div>
                </div>
            </li>
        </div>
        """
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = BandcampProvider(mock_client)
        results = await provider.search_artist("Carl Cox")

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_artist_empty(self) -> None:
        from src.providers.music_db.bandcamp_provider import BandcampProvider

        html = '<div class="results"></div>'
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = BandcampProvider(mock_client)
        results = await provider.search_artist("xyznonexistent")

        assert results == []

    @pytest.mark.asyncio
    async def test_get_artist_releases_data_items(self) -> None:
        """Test get_artist_releases parsing releases from [data-item-id] elements."""
        from src.providers.music_db.bandcamp_provider import BandcampProvider

        html = """
        <html>
        <div id="band-name-location">
            <span class="title">Some Artist</span>
        </div>
        <ol id="music-grid">
            <li data-item-id="album-1234">
                <a href="/album/first-ep">
                    <div class="title">First EP</div>
                </a>
                <div class="released">released March 15, 2020</div>
            </li>
            <li data-item-id="album-5678">
                <a href="/album/second-lp">
                    <div class="title">Second LP</div>
                </a>
                <div class="released">released January 1, 2022</div>
            </li>
        </ol>
        </html>
        """
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = BandcampProvider(mock_client)
        releases = await provider.get_artist_releases("https://someartist.bandcamp.com")

        assert len(releases) == 2
        assert releases[0].title == "First EP"
        assert releases[0].year == 2020
        assert releases[0].format == "Digital"
        assert releases[1].title == "Second LP"
        assert releases[1].year == 2022

    @pytest.mark.asyncio
    async def test_get_artist_releases_music_grid_fallback(self) -> None:
        """Test fallback to .music-grid-item when no [data-item-id] elements."""
        from src.providers.music_db.bandcamp_provider import BandcampProvider

        html = """
        <html>
        <div class="music-grid">
            <li class="music-grid-item">
                <a href="/album/grid-release">
                    <div class="title">Grid Release</div>
                </a>
            </li>
        </div>
        </html>
        """
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = BandcampProvider(mock_client)
        releases = await provider.get_artist_releases("https://someartist.bandcamp.com")

        assert len(releases) == 1
        assert releases[0].title == "Grid Release"
        assert releases[0].label == "Self-released"

    @pytest.mark.asyncio
    async def test_get_artist_releases_empty_page(self) -> None:
        """Test get_artist_releases returns empty when fetch fails."""
        from src.providers.music_db.bandcamp_provider import BandcampProvider

        # Simulate 429 rate limit on both /music and base URL
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = BandcampProvider(mock_client)
        releases = await provider.get_artist_releases("https://someartist.bandcamp.com")

        assert releases == []

    @pytest.mark.asyncio
    async def test_get_artist_releases_with_label(self) -> None:
        """Test that label is extracted from page when present as a label link."""
        from src.providers.music_db.bandcamp_provider import BandcampProvider

        html = """
        <html>
        <div id="band-name-location">
            <a href="https://superlabel.bandcamp.com/label/main">
                <span class="title">Super Label Records</span>
            </a>
        </div>
        <ol>
            <li data-item-id="album-100">
                <a href="/album/release-one">
                    <div class="title">Release One</div>
                </a>
            </li>
        </ol>
        </html>
        """
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = BandcampProvider(mock_client)
        releases = await provider.get_artist_releases("https://someartist.bandcamp.com")

        assert len(releases) == 1
        assert releases[0].label == "Super Label Records"

    @pytest.mark.asyncio
    async def test_get_artist_releases_before_date(self) -> None:
        """Test that releases after before_date are filtered out."""
        from src.providers.music_db.bandcamp_provider import BandcampProvider

        html = """
        <html>
        <ol>
            <li data-item-id="album-1">
                <a href="/album/old-ep">
                    <div class="title">Old EP</div>
                </a>
                <div class="released">released June 10, 1999</div>
            </li>
            <li data-item-id="album-2">
                <a href="/album/new-ep">
                    <div class="title">New EP</div>
                </a>
                <div class="released">released January 5, 2023</div>
            </li>
        </ol>
        </html>
        """
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = BandcampProvider(mock_client)
        releases = await provider.get_artist_releases(
            "https://someartist.bandcamp.com",
            before_date=date(2000, 1, 1),
        )

        assert len(releases) == 1
        assert releases[0].title == "Old EP"

    @pytest.mark.asyncio
    async def test_get_artist_labels_success(self) -> None:
        """Test extracting labels from an artist's releases."""
        from src.providers.music_db.bandcamp_provider import BandcampProvider

        html = """
        <html>
        <div id="band-name-location">
            <a href="https://labelx.bandcamp.com/label/releases">
                <span class="title">Label X</span>
            </a>
        </div>
        <ol>
            <li data-item-id="album-10">
                <a href="/album/track-a">
                    <div class="title">Track A</div>
                </a>
            </li>
        </ol>
        </html>
        """
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = BandcampProvider(mock_client)
        labels = await provider.get_artist_labels("https://someartist.bandcamp.com")

        assert len(labels) == 1
        assert labels[0].name == "Label X"

    @pytest.mark.asyncio
    async def test_get_artist_labels_self_released_excluded(self) -> None:
        """Test that 'Self-released' labels are excluded from labels list."""
        from src.providers.music_db.bandcamp_provider import BandcampProvider

        # No label link means label = "Self-released", which gets filtered
        html = """
        <html>
        <ol>
            <li data-item-id="album-1">
                <a href="/album/demo"><div class="title">Demo</div></a>
            </li>
        </ol>
        </html>
        """
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = BandcampProvider(mock_client)
        labels = await provider.get_artist_labels("https://someartist.bandcamp.com")

        assert labels == []

    @pytest.mark.asyncio
    async def test_get_release_details(self) -> None:
        """Test get_release_details parses an individual release page."""
        from src.providers.music_db.bandcamp_provider import BandcampProvider

        html = """
        <html>
        <div id="name-section">
            <h2 class="trackTitle">Night Drive EP</h2>
        </div>
        <div id="band-name-location">
            <span class="title">Techno Records</span>
        </div>
        <div class="tralbumData tralbum-credits">
            released October 15, 2019
        </div>
        <div class="tralbumData tralbum-tags">
            <a class="tag" href="/tag/techno">techno</a>
            <a class="tag" href="/tag/dark-techno">dark techno</a>
        </div>
        </html>
        """
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = BandcampProvider(mock_client)
        release = await provider.get_release_details("https://someartist.bandcamp.com/album/night-drive-ep")

        assert release is not None
        assert release.title == "Night Drive EP"
        assert release.label == "Techno Records"
        assert release.year == 2019
        assert "techno" in release.genres
        assert "dark techno" in release.genres

    @pytest.mark.asyncio
    async def test_get_release_details_none_on_failure(self) -> None:
        """Test get_release_details returns None when page fetch fails."""
        from src.providers.music_db.bandcamp_provider import BandcampProvider

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = BandcampProvider(mock_client)
        release = await provider.get_release_details("https://someartist.bandcamp.com/album/gone")

        assert release is None


# ======================================================================
# Beatport Provider
# ======================================================================


class TestBeatportProvider:
    def test_get_provider_name(self) -> None:
        from src.providers.music_db.beatport_provider import BeatportProvider
        mock_client = AsyncMock()
        provider = BeatportProvider(mock_client)
        assert provider.get_provider_name() == "beatport"

    def test_is_available(self) -> None:
        from src.providers.music_db.beatport_provider import BeatportProvider
        mock_client = AsyncMock()
        provider = BeatportProvider(mock_client)
        assert provider.is_available() is True

    @pytest.mark.asyncio
    async def test_search_artist_success(self) -> None:
        from src.providers.music_db.beatport_provider import BeatportProvider

        # Beatport JSON API response
        json_data = {
            "results": [
                {
                    "id": 1234,
                    "name": "Carl Cox",
                    "slug": "carl-cox",
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.json.return_value = json_data
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = BeatportProvider(mock_client)
        results = await provider.search_artist("Carl Cox")

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_artist_empty_results(self) -> None:
        """Test search_artist returns empty list when no results from both API and scrape."""
        from src.providers.music_db.beatport_provider import BeatportProvider

        # Both JSON API responses return nothing useful
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.text = "<html><body>No results</body></html>"

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = BeatportProvider(mock_client)
        results = await provider.search_artist("xyznonexistent")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_artist_fallback_to_scrape(self) -> None:
        """Test fallback to HTML scraping when JSON API returns 403."""
        from src.providers.music_db.beatport_provider import BeatportProvider

        # First two calls: JSON API blocked (403)
        api_response = MagicMock()
        api_response.status_code = 403
        api_response.json.return_value = {}
        api_response.raise_for_status = MagicMock()

        # Third call: HTML scrape page
        scrape_html = """
        <html><body>
            <a href="/artist/carl-cox/1234">Carl Cox</a>
        </body></html>
        """
        scrape_response = MagicMock()
        scrape_response.status_code = 200
        scrape_response.text = scrape_html
        scrape_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=[api_response, api_response, scrape_response]
        )

        provider = BeatportProvider(mock_client)
        results = await provider.search_artist("Carl Cox")

        assert len(results) >= 1
        assert results[0].name == "Carl Cox"
        assert results[0].id == "carl-cox/1234"

    @pytest.mark.asyncio
    async def test_get_artist_releases_json_api(self) -> None:
        """Test get_artist_releases parsing from JSON API response."""
        from src.providers.music_db.beatport_provider import BeatportProvider

        json_data = {
            "results": [
                {
                    "id": 5001,
                    "name": "Night Techno EP",
                    "slug": "night-techno-ep",
                    "new_release_date": "2021-06-15",
                    "label": {"name": "Drumcode"},
                    "type": {"name": "EP"},
                    "genre": [{"name": "Techno"}, {"name": "Hard Techno"}],
                },
                {
                    "id": 5002,
                    "name": "Bass Driver",
                    "slug": "bass-driver",
                    "publish_date": "2019-03-01",
                    "label": {"name": "Intec"},
                    "type": {"name": "Single"},
                    "genre": [{"name": "Tech House"}],
                },
            ]
        }
        mock_response = MagicMock()
        mock_response.json.return_value = json_data
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = BeatportProvider(mock_client)
        releases = await provider.get_artist_releases("carl-cox/1234")

        assert len(releases) == 2
        assert releases[0].title == "Night Techno EP"
        assert releases[0].label == "Drumcode"
        assert releases[0].year == 2021
        assert releases[0].format == "EP"
        assert "Techno" in releases[0].genres
        assert releases[1].title == "Bass Driver"
        assert releases[1].label == "Intec"

    @pytest.mark.asyncio
    async def test_get_artist_releases_before_date_filter(self) -> None:
        """Test that releases after before_date are filtered out."""
        from src.providers.music_db.beatport_provider import BeatportProvider

        json_data = {
            "results": [
                {
                    "id": 6001,
                    "name": "Old Track",
                    "slug": "old-track",
                    "new_release_date": "1998-01-01",
                    "label": {"name": "React"},
                    "type": {"name": "Single"},
                },
                {
                    "id": 6002,
                    "name": "New Track",
                    "slug": "new-track",
                    "new_release_date": "2025-06-01",
                    "label": {"name": "React"},
                    "type": {"name": "Single"},
                },
            ]
        }
        mock_response = MagicMock()
        mock_response.json.return_value = json_data
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = BeatportProvider(mock_client)
        releases = await provider.get_artist_releases(
            "carl-cox/1234", before_date=date(2000, 1, 1)
        )

        assert len(releases) == 1
        assert releases[0].title == "Old Track"

    @pytest.mark.asyncio
    async def test_get_artist_releases_scrape_fallback(self) -> None:
        """Test fallback to HTML scraping when JSON API returns nothing."""
        from src.providers.music_db.beatport_provider import BeatportProvider

        # JSON API returns empty
        api_response = MagicMock()
        api_response.json.return_value = {"results": []}
        api_response.status_code = 200
        api_response.raise_for_status = MagicMock()

        # HTML scrape page with release links
        scrape_html = """
        <html><body>
            <a href="/release/dark-matter/9001">Dark Matter</a>
            <a href="/release/light-pulse/9002">Light Pulse</a>
        </body></html>
        """
        scrape_response = MagicMock()
        scrape_response.status_code = 200
        scrape_response.text = scrape_html
        scrape_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=[api_response, scrape_response]
        )

        provider = BeatportProvider(mock_client)
        releases = await provider.get_artist_releases("carl-cox/1234")

        assert len(releases) == 2
        assert releases[0].title == "Dark Matter"
        assert releases[1].title == "Light Pulse"
        assert "beatport.com/release/dark-matter/9001" in releases[0].beatport_url

    @pytest.mark.asyncio
    async def test_get_release_details_json_api(self) -> None:
        """Test get_release_details via JSON API."""
        from src.providers.music_db.beatport_provider import BeatportProvider

        json_data = {
            "id": 7001,
            "name": "Warehouse Sessions",
            "slug": "warehouse-sessions",
            "new_release_date": "2020-09-15",
            "label": {"name": "Intec"},
            "type": {"name": "Album"},
            "genre": [{"name": "Techno"}],
        }
        mock_response = MagicMock()
        mock_response.json.return_value = json_data
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = BeatportProvider(mock_client)
        release = await provider.get_release_details("warehouse-sessions/7001")

        assert release is not None
        assert release.title == "Warehouse Sessions"
        assert release.label == "Intec"
        assert release.year == 2020

    @pytest.mark.asyncio
    async def test_get_release_details_not_found(self) -> None:
        """Test get_release_details returns None when API returns nothing."""
        from src.providers.music_db.beatport_provider import BeatportProvider

        # JSON API blocked
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = BeatportProvider(mock_client)
        release = await provider.get_release_details("nonexistent/9999")

        assert release is None

    @pytest.mark.asyncio
    async def test_get_artist_labels(self) -> None:
        """Test extracting unique labels from artist releases."""
        from src.providers.music_db.beatport_provider import BeatportProvider

        json_data = {
            "results": [
                {
                    "id": 8001,
                    "name": "Track A",
                    "slug": "track-a",
                    "label": {"name": "Drumcode"},
                    "type": {"name": "Single"},
                },
                {
                    "id": 8002,
                    "name": "Track B",
                    "slug": "track-b",
                    "label": {"name": "Intec"},
                    "type": {"name": "EP"},
                },
                {
                    "id": 8003,
                    "name": "Track C",
                    "slug": "track-c",
                    "label": {"name": "Drumcode"},
                    "type": {"name": "Single"},
                },
            ]
        }
        mock_response = MagicMock()
        mock_response.json.return_value = json_data
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = BeatportProvider(mock_client)
        labels = await provider.get_artist_labels("carl-cox/1234")

        label_names = [lb.name for lb in labels]
        assert "Drumcode" in label_names
        assert "Intec" in label_names
        # "Unknown" labels should be excluded
        assert "Unknown" not in label_names
        # Drumcode should appear only once (deduplicated)
        assert label_names.count("Drumcode") == 1

    def test_parse_artist_id_composite(self) -> None:
        """Test _parse_artist_id splits slug/numeric format."""
        from src.providers.music_db.beatport_provider import BeatportProvider

        slug, numeric = BeatportProvider._parse_artist_id("carl-cox/1234")
        assert slug == "carl-cox"
        assert numeric == "1234"

    def test_parse_artist_id_plain(self) -> None:
        """Test _parse_artist_id handles plain ID (no slash)."""
        from src.providers.music_db.beatport_provider import BeatportProvider

        slug, numeric = BeatportProvider._parse_artist_id("1234")
        assert slug == "1234"
        assert numeric == "1234"


# ======================================================================
# MusicBrainz Provider
# ======================================================================


class TestMusicBrainzProvider:
    @pytest.fixture()
    def settings(self) -> Settings:
        return _settings()

    def test_get_provider_name(self, settings: Settings) -> None:
        from src.providers.music_db.musicbrainz_provider import MusicBrainzProvider

        with patch("src.providers.music_db.musicbrainz_provider.musicbrainzngs"):
            provider = MusicBrainzProvider(settings)
            assert provider.get_provider_name() == "musicbrainz"

    def test_is_available(self, settings: Settings) -> None:
        from src.providers.music_db.musicbrainz_provider import MusicBrainzProvider

        with patch("src.providers.music_db.musicbrainz_provider.musicbrainzngs"):
            provider = MusicBrainzProvider(settings)
            assert provider.is_available() is True

    @pytest.mark.asyncio
    async def test_search_artist_success(self, settings: Settings) -> None:
        from src.providers.music_db.musicbrainz_provider import MusicBrainzProvider

        mb_response = {
            "artist-list": [
                {
                    "id": "abc-def-123",
                    "name": "Carl Cox",
                    "disambiguation": "UK DJ",
                    "ext:score": "100",
                }
            ]
        }

        with patch("src.providers.music_db.musicbrainz_provider.musicbrainzngs") as mock_mb:
            mock_mb.search_artists.return_value = mb_response
            provider = MusicBrainzProvider(settings)
            results = await provider.search_artist("Carl Cox")

        assert len(results) >= 1
        assert results[0].name == "Carl Cox"

    @pytest.mark.asyncio
    async def test_search_artist_empty(self, settings: Settings) -> None:
        from src.providers.music_db.musicbrainz_provider import MusicBrainzProvider

        with patch("src.providers.music_db.musicbrainz_provider.musicbrainzngs") as mock_mb:
            mock_mb.search_artists.return_value = {"artist-list": []}
            provider = MusicBrainzProvider(settings)
            results = await provider.search_artist("xyznonexistent")

        assert results == []

    @pytest.mark.asyncio
    async def test_get_artist_releases(self, settings: Settings) -> None:
        from src.providers.music_db.musicbrainz_provider import MusicBrainzProvider

        # Code uses browse_releases (not browse_release_groups)
        # _map_release expects: title, date, label-info-list
        mb_response = {
            "release-list": [
                {
                    "id": "rel-123",
                    "title": "Phat Trax",
                    "date": "1995",
                    "label-info-list": [
                        {
                            "catalog-number": "REACT-001",
                            "label": {"name": "React"},
                        }
                    ],
                }
            ],
            "release-count": 1,
        }

        with patch("src.providers.music_db.musicbrainz_provider.musicbrainzngs") as mock_mb:
            mock_mb.browse_releases.return_value = mb_response
            provider = MusicBrainzProvider(settings)
            releases = await provider.get_artist_releases("abc-def-123")

        assert len(releases) >= 1
        assert releases[0].title == "Phat Trax"

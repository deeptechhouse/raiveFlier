"""Unit tests for CorpusPublisher service.

Tests GitHub release upload, version marker writing, Render deploy
trigger, and publish status reporting — all with mocked HTTP calls.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.raive_feeder.services.corpus_publisher import CorpusPublisher


def _make_mock_manager(tarball_content: bytes = b"fake tarball"):
    """Create a mock CorpusManager that returns a temp tarball on export."""
    manager = MagicMock()

    async def _fake_export(chromadb_dir: str) -> str:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz")
        tmp.write(tarball_content)
        tmp.close()
        return tmp.name

    manager.export_corpus = AsyncMock(side_effect=_fake_export)
    return manager


@pytest.fixture
def publisher():
    """CorpusPublisher with mocked manager and test token."""
    return CorpusPublisher(
        corpus_manager=_make_mock_manager(),
        github_token="ghp_test_token_123",
        corpus_repo="testorg/test-corpus",
        render_deploy_hook_url="",
    )


@pytest.fixture
def publisher_with_deploy():
    """CorpusPublisher with Render deploy hook configured."""
    return CorpusPublisher(
        corpus_manager=_make_mock_manager(),
        github_token="ghp_test_token_123",
        corpus_repo="testorg/test-corpus",
        render_deploy_hook_url="https://api.render.com/deploy/srv-test?key=abc",
    )


class TestVersionMarker:
    """Tests for corpus_version.txt writing."""

    def test_writes_version_file(self):
        """Publishing should write corpus_version.txt with the tag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            CorpusPublisher._write_version_marker(tmpdir, "v1.0.5")

            version_file = Path(tmpdir) / "corpus_version.txt"
            assert version_file.exists()

            content = version_file.read_text()
            assert content.startswith("v1.0.5\n")
            # Second line should be a timestamp.
            lines = content.strip().split("\n")
            assert len(lines) == 2


class TestPublishValidation:
    """Tests for publish input validation."""

    @pytest.mark.asyncio
    async def test_no_token_raises(self):
        """Publishing without GITHUB_TOKEN should raise ValueError."""
        publisher = CorpusPublisher(
            corpus_manager=_make_mock_manager(),
            github_token="",
            corpus_repo="testorg/test-corpus",
        )

        with pytest.raises(ValueError, match="GITHUB_TOKEN"):
            await publisher.publish("v1.0.0", "/tmp/chromadb")


class TestPublishFlow:
    """Tests for the full publish flow with mocked GitHub API."""

    @pytest.mark.asyncio
    async def test_publish_creates_release_and_uploads(self, publisher):
        """Publish should create release, delete old asset, upload new one."""
        with tempfile.TemporaryDirectory() as chromadb_dir:
            # Create a fake chroma.sqlite3 so export can "work".
            (Path(chromadb_dir) / "chroma.sqlite3").write_bytes(b"fake db")

            # Mock all HTTP calls.
            mock_responses = [
                # _ensure_release: GET release by tag → 404 (not found).
                MagicMock(status_code=404),
                # _ensure_release: POST create release → 201.
                MagicMock(status_code=201, json=lambda: {"id": 42}),
                # _delete_existing_asset: GET assets → empty list.
                MagicMock(status_code=200, json=lambda: []),
                # _upload_asset: POST upload → 201.
                MagicMock(status_code=201),
            ]

            with patch("tools.raive_feeder.services.corpus_publisher.httpx.AsyncClient") as MockClient:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(side_effect=[mock_responses[0], mock_responses[2]])
                mock_client.post = AsyncMock(side_effect=[mock_responses[1], mock_responses[3]])
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = mock_client

                result = await publisher.publish("v1.0.1", chromadb_dir)

            assert result["tag"] == "v1.0.1"
            assert result["status"] == "uploaded"
            assert result["deploy_triggered"] is False
            assert isinstance(result["size_mb"], float)

            # Version marker should have been written.
            version_file = Path(chromadb_dir) / "corpus_version.txt"
            assert version_file.exists()
            assert version_file.read_text().startswith("v1.0.1\n")


class TestDeployTrigger:
    """Tests for Render deploy hook triggering."""

    @pytest.mark.asyncio
    async def test_trigger_deploy_success(self, publisher_with_deploy):
        """Successful POST to deploy hook should return True."""
        with patch("tools.raive_feeder.services.corpus_publisher.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=MagicMock(status_code=200))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await publisher_with_deploy.trigger_deploy()

        assert result is True

    @pytest.mark.asyncio
    async def test_trigger_deploy_no_hook(self, publisher):
        """No deploy hook URL should return False without making a request."""
        result = await publisher.trigger_deploy()
        assert result is False

    @pytest.mark.asyncio
    async def test_trigger_deploy_failure(self, publisher_with_deploy):
        """Failed POST should return False."""
        with patch("tools.raive_feeder.services.corpus_publisher.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=MagicMock(status_code=500, text="Internal Server Error"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await publisher_with_deploy.trigger_deploy()

        assert result is False


class TestPublishStatus:
    """Tests for publish status reporting."""

    @pytest.mark.asyncio
    async def test_status_with_token(self, publisher):
        """Status should report token set and fetch latest tag."""
        with patch("tools.raive_feeder.services.corpus_publisher.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=MagicMock(
                status_code=200,
                json=lambda: {"tag_name": "v1.0.3"},
            ))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            status = await publisher.get_publish_status()

        assert status["github_token_set"] is True
        assert status["latest_tag"] == "v1.0.3"
        assert status["corpus_repo"] == "testorg/test-corpus"
        assert status["deploy_hook_configured"] is False

    @pytest.mark.asyncio
    async def test_status_without_token(self):
        """Status without token should report token not set."""
        publisher = CorpusPublisher(
            corpus_manager=_make_mock_manager(),
            github_token="",
            corpus_repo="testorg/test-corpus",
        )

        status = await publisher.get_publish_status()

        assert status["github_token_set"] is False
        assert status["latest_tag"] is None

    @pytest.mark.asyncio
    async def test_status_with_deploy_hook(self, publisher_with_deploy):
        """Status should report deploy hook configured."""
        with patch("tools.raive_feeder.services.corpus_publisher.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=MagicMock(
                status_code=200,
                json=lambda: {"tag_name": "v1.0.0"},
            ))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            status = await publisher_with_deploy.get_publish_status()

        assert status["deploy_hook_configured"] is True

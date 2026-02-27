"""Corpus publishing service — automates GitHub release upload and Render deploy.

# ─── ARCHITECTURE ────────────────────────────────────────────────────
#
# CorpusPublisher bridges the gap between local corpus building (via
# raiveFeeder's ingestion tabs) and production deployment on Render.
#
# The manual workflow it replaces:
#   1. Run scripts/package_corpus.sh --upload v1.0.x
#   2. Manually trigger a Render deploy
#   3. Manually update CORPUS_TAG env var on Render
#
# Now it's one API call: POST /api/v1/corpus/publish
#
# Dependencies:
#   - CorpusManager.export_corpus() for tarball creation
#   - GitHub Releases REST API (via httpx) for upload
#   - Render Deploy Hook URL (optional, for auto-deploy)
#   - GITHUB_TOKEN env var for authentication
#
# The service writes a corpus_version.txt into the ChromaDB directory
# before packaging.  entrypoint.sh reads this to decide whether to
# re-download, solving the "corpus >50MB skips download" problem.
# ─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import structlog

from tools.raive_feeder.services.corpus_manager import CorpusManager

logger = structlog.get_logger(logger_name=__name__)

# GitHub Releases API base URL.
_GITHUB_API = "https://api.github.com"
# GitHub upload endpoint uses a different host.
_GITHUB_UPLOADS = "https://uploads.github.com"
# Expected tarball asset name (must match what entrypoint.sh downloads).
_ASSET_NAME = "chromadb_corpus.tar.gz"


class CorpusPublisher:
    """Publishes the local ChromaDB corpus to a GitHub release and optionally
    triggers a Render deploy.

    Constructor injection keeps this testable — all external dependencies
    (token, repo, URLs) are passed in rather than read from globals.
    """

    def __init__(
        self,
        *,
        corpus_manager: CorpusManager,
        github_token: str,
        corpus_repo: str = "deeptechhouse/raiveflier-corpus",
        render_deploy_hook_url: str = "",
    ) -> None:
        self._manager = corpus_manager
        self._token = github_token
        self._repo = corpus_repo
        self._deploy_hook_url = render_deploy_hook_url

    # ─── Public API ──────────────────────────────────────────────────

    async def publish(self, tag: str, chromadb_dir: str) -> dict[str, Any]:
        """Export the corpus, upload to GitHub release, optionally trigger deploy.

        Parameters
        ----------
        tag:
            Release tag (e.g. "v1.0.2").  Created if it doesn't exist.
        chromadb_dir:
            Path to the local ChromaDB directory to package.

        Returns
        -------
        dict with keys: tag, size_mb, status, deploy_triggered
        """
        if not self._token:
            raise ValueError("GITHUB_TOKEN is required for publishing")

        # Write version marker before packaging so it's included in the tarball.
        self._write_version_marker(chromadb_dir, tag)

        # Reuse CorpusManager's export to create the tarball.
        tarball_path = await self._manager.export_corpus(chromadb_dir)
        size_mb = round(Path(tarball_path).stat().st_size / (1024 * 1024), 1)

        logger.info(
            "corpus_publish_start",
            tag=tag,
            size_mb=size_mb,
            repo=self._repo,
        )

        # Ensure the release exists (create if needed).
        release_id = await self._ensure_release(tag)

        # Delete existing asset with same name (clobber semantics).
        await self._delete_existing_asset(release_id)

        # Upload the new tarball.
        await self._upload_asset(release_id, tarball_path)

        # Clean up the temp tarball.
        Path(tarball_path).unlink(missing_ok=True)

        # Optionally trigger a Render deploy.
        deploy_triggered = False
        if self._deploy_hook_url:
            deploy_triggered = await self.trigger_deploy()

        logger.info(
            "corpus_publish_complete",
            tag=tag,
            size_mb=size_mb,
            deploy_triggered=deploy_triggered,
        )

        return {
            "tag": tag,
            "size_mb": size_mb,
            "status": "uploaded",
            "deploy_triggered": deploy_triggered,
        }

    async def trigger_deploy(self) -> bool:
        """POST to the Render Deploy Hook URL to trigger a fresh deploy.

        Returns True if the deploy was triggered successfully.
        """
        if not self._deploy_hook_url:
            logger.warning("render_deploy_hook_not_configured")
            return False

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(self._deploy_hook_url)

            if resp.status_code == 200:
                logger.info("render_deploy_triggered")
                return True

            logger.warning(
                "render_deploy_failed",
                status_code=resp.status_code,
                body=resp.text[:200],
            )
            return False

    async def get_publish_status(self) -> dict[str, Any]:
        """Return the current publish configuration and latest release info.

        Used by the UI to show the current state before publishing.
        """
        status: dict[str, Any] = {
            "github_token_set": bool(self._token),
            "corpus_repo": self._repo,
            "deploy_hook_configured": bool(self._deploy_hook_url),
            "latest_tag": None,
        }

        if self._token:
            status["latest_tag"] = await self._get_latest_release_tag()

        return status

    # ─── Private: GitHub API helpers ─────────────────────────────────

    def _headers(self) -> dict[str, str]:
        """Standard headers for GitHub API requests."""
        return {
            "Authorization": f"token {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def _ensure_release(self, tag: str) -> int:
        """Get or create a GitHub release for the given tag.  Returns release ID."""
        async with httpx.AsyncClient(timeout=30) as client:
            # Try to get existing release.
            resp = await client.get(
                f"{_GITHUB_API}/repos/{self._repo}/releases/tags/{tag}",
                headers=self._headers(),
            )

            if resp.status_code == 200:
                return resp.json()["id"]

            # Create the release.
            resp = await client.post(
                f"{_GITHUB_API}/repos/{self._repo}/releases",
                headers=self._headers(),
                json={
                    "tag_name": tag,
                    "name": f"ChromaDB Corpus {tag}",
                    "body": (
                        f"Pre-built ChromaDB corpus for raiveFlier deployment.\n"
                        f"Published by raiveFeeder on "
                        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}."
                    ),
                },
            )
            resp.raise_for_status()
            release_id = resp.json()["id"]

            logger.info("github_release_created", tag=tag, release_id=release_id)
            return release_id

    async def _delete_existing_asset(self, release_id: int) -> None:
        """Delete any existing asset named chromadb_corpus.tar.gz on the release."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{_GITHUB_API}/repos/{self._repo}/releases/{release_id}/assets",
                headers=self._headers(),
            )

            if resp.status_code != 200:
                return

            for asset in resp.json():
                if asset["name"] == _ASSET_NAME:
                    await client.delete(
                        f"{_GITHUB_API}/repos/{self._repo}/releases/assets/{asset['id']}",
                        headers=self._headers(),
                    )
                    logger.info("github_asset_deleted", asset_id=asset["id"])

    async def _upload_asset(self, release_id: int, tarball_path: str) -> None:
        """Upload the tarball as a release asset."""
        file_size = Path(tarball_path).stat().st_size

        async with httpx.AsyncClient(timeout=600) as client:
            with open(tarball_path, "rb") as f:
                resp = await client.post(
                    f"{_GITHUB_UPLOADS}/repos/{self._repo}/releases/{release_id}/assets",
                    params={"name": _ASSET_NAME},
                    headers={
                        **self._headers(),
                        "Content-Type": "application/gzip",
                        "Content-Length": str(file_size),
                    },
                    content=f.read(),
                )
                resp.raise_for_status()

        logger.info(
            "github_asset_uploaded",
            release_id=release_id,
            size_bytes=file_size,
        )

    async def _get_latest_release_tag(self) -> str | None:
        """Fetch the latest release tag from the corpus repo."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"{_GITHUB_API}/repos/{self._repo}/releases/latest",
                    headers=self._headers(),
                )

                if resp.status_code == 200:
                    return resp.json().get("tag_name")

                return None
        except httpx.HTTPError:
            return None

    # ─── Private: version marker ─────────────────────────────────────

    @staticmethod
    def _write_version_marker(chromadb_dir: str, tag: str) -> None:
        """Write corpus_version.txt into the ChromaDB directory.

        This file is included in the tarball and also stays on the local
        filesystem.  entrypoint.sh on Render compares the local version
        file against CORPUS_TAG to decide whether to re-download.
        """
        version_file = Path(chromadb_dir) / "corpus_version.txt"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        version_file.write_text(f"{tag}\n{timestamp}\n")

        logger.info(
            "corpus_version_marker_written",
            tag=tag,
            path=str(version_file),
        )

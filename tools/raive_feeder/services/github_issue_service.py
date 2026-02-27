"""GitHub Issue notification service for raiveFeeder content submissions.

# ─── ARCHITECTURE ────────────────────────────────────────────────────
#
# GitHubIssueService creates and closes GitHub Issues when content is
# submitted to or decided on in the approval queue.  This provides
# admin notifications via email/mobile (GitHub's default notification
# settings) without adding a separate notification service.
#
# Follows the same pattern as corpus_publisher.py:
#   - httpx for async HTTP requests
#   - Same token auth headers
#   - Same error handling (log + graceful None return on failure)
#
# Graceful degradation: if no GITHUB_TOKEN is set or the API call fails,
# the approval queue still works — Issues are nice-to-have notifications,
# not a hard dependency.
#
# Layer: Services (depends on httpx, structlog)
# ─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

logger = structlog.get_logger(logger_name=__name__)

_GITHUB_API = "https://api.github.com"


class GitHubIssueService:
    """Creates and manages GitHub Issues for content submission notifications.

    Constructor injection: token and repo are passed in, not read from env.
    """

    def __init__(self, *, github_token: str, repo: str) -> None:
        self._token = github_token
        self._repo = repo

    def _headers(self) -> dict[str, str]:
        """Standard headers for GitHub API requests."""
        return {
            "Authorization": f"token {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def create_submission_issue(
        self, submission: dict[str, Any]
    ) -> tuple[int, str] | None:
        """Create a GitHub Issue for a new content submission.

        Parameters
        ----------
        submission:
            The submission dict with id, title, content_type, content_data, etc.

        Returns
        -------
        Tuple of (issue_number, issue_url), or None if creation failed.
        """
        if not self._token:
            return None

        title = f"[raive-feeder] New submission: {submission.get('title', 'Untitled')}"
        content_type = submission.get("content_type", "unknown")
        content_preview = submission.get("content_data", "")[:200]

        body = (
            f"## New Content Submission\n\n"
            f"- **ID:** `{submission.get('id', 'unknown')}`\n"
            f"- **Title:** {submission.get('title', 'Untitled')}\n"
            f"- **Type:** {submission.get('source_type', 'article')}\n"
            f"- **Content Type:** {content_type}\n"
            f"- **Citation Tier:** {submission.get('citation_tier', 3)}\n"
            f"- **Author:** {submission.get('author', 'N/A')}\n"
            f"- **Year:** {submission.get('year', 'N/A')}\n\n"
            f"### Content Preview\n"
            f"```\n{content_preview}\n```\n\n"
            f"---\n"
            f"*Approve or reject this submission in the raiveFeeder dashboard.*"
        )

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    f"{_GITHUB_API}/repos/{self._repo}/issues",
                    headers=self._headers(),
                    json={
                        "title": title,
                        "body": body,
                        "labels": ["raive-feeder"],
                    },
                )

                if resp.status_code == 201:
                    data = resp.json()
                    issue_number = data["number"]
                    issue_url = data["html_url"]
                    logger.info(
                        "github_issue_created",
                        issue_number=issue_number,
                        submission_id=submission.get("id"),
                    )
                    return issue_number, issue_url

                logger.warning(
                    "github_issue_creation_failed",
                    status_code=resp.status_code,
                    body=resp.text[:200],
                )
                return None

        except httpx.HTTPError as exc:
            logger.warning("github_issue_http_error", error=str(exc))
            return None

    async def close_issue_with_comment(
        self, issue_number: int, decision: str, reason: str = ""
    ) -> bool:
        """Close a GitHub Issue with an approval/rejection comment.

        Parameters
        ----------
        issue_number:
            The GitHub Issue number to close.
        decision:
            "approved" or "rejected".
        reason:
            Optional rejection reason.

        Returns
        -------
        True if the Issue was closed successfully, False otherwise.
        """
        if not self._token:
            return False

        # Build the comment body.
        emoji = "\u2705" if decision == "approved" else "\u274c"
        comment_body = f"{emoji} **Submission {decision}**"
        if reason:
            comment_body += f"\n\nReason: {reason}"

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                # Add comment.
                await client.post(
                    f"{_GITHUB_API}/repos/{self._repo}/issues/{issue_number}/comments",
                    headers=self._headers(),
                    json={"body": comment_body},
                )

                # Close the issue.
                resp = await client.patch(
                    f"{_GITHUB_API}/repos/{self._repo}/issues/{issue_number}",
                    headers=self._headers(),
                    json={"state": "closed"},
                )

                if resp.status_code == 200:
                    logger.info(
                        "github_issue_closed",
                        issue_number=issue_number,
                        decision=decision,
                    )
                    return True

                logger.warning(
                    "github_issue_close_failed",
                    issue_number=issue_number,
                    status_code=resp.status_code,
                )
                return False

        except httpx.HTTPError as exc:
            logger.warning(
                "github_issue_close_http_error",
                issue_number=issue_number,
                error=str(exc),
            )
            return False

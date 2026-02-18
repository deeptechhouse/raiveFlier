"""Abstract base class for article-extraction service providers.

Defines the contract for extracting readable content from web pages.
Implementations may wrap Newspaper3k, Trafilatura, Readability, or a
custom scraper.  The adapter pattern (CLAUDE.md Section 6) keeps the
research layer independent of the underlying extraction engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ArticleContent:
    """Extracted content from a web article.

    Attributes
    ----------
    title:
        The article's headline or page title.
    text:
        The main body text with markup stripped.
    author:
        The article author if identifiable.
    date:
        The publication date if identifiable.
    url:
        The source URL the content was extracted from.
    """

    title: str
    text: str
    author: str | None = None
    date: date | None = None
    url: str = ""


class IArticleProvider(ABC):
    """Contract for services that extract readable content from web URLs.

    Used during the research phase to pull article text from URLs found
    via web search, enabling downstream LLM analysis and citation.
    """

    @abstractmethod
    async def extract_content(self, url: str) -> ArticleContent | None:
        """Fetch and extract readable content from *url*.

        Parameters
        ----------
        url:
            The web page URL to extract content from.

        Returns
        -------
        ArticleContent or None
            The extracted article content, or ``None`` if extraction
            failed or the page contains no usable text.

        Raises
        ------
        src.core.errors.ResearchError
            If the HTTP request fails or a non-recoverable error occurs.
        """

    @abstractmethod
    async def check_availability(self, url: str) -> bool:
        """Return ``True`` if *url* is reachable and likely contains extractable content.

        This is a lightweight probe (e.g. an HTTP HEAD request) — it does
        not perform full extraction.

        Parameters
        ----------
        url:
            The URL to check.

        Returns
        -------
        bool
            ``True`` if the URL appears accessible; ``False`` otherwise.
        """

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return a human-readable identifier for this article provider.

        Example return values: ``"newspaper3k"``, ``"trafilatura"``.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if the provider's dependencies are installed and usable.

        Implementations should verify that required libraries or network
        access are in place without performing actual extraction.
        """

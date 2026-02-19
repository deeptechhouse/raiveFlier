"""Source processor for web articles and local HTML/text article files.

Fetches or reads article content and wraps it in :class:`~src.models.rag.DocumentChunk`
objects with source type and citation tier inferred from URL patterns.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import TYPE_CHECKING

import structlog

from src.models.rag import DocumentChunk

if TYPE_CHECKING:
    from src.interfaces.article_provider import IArticleProvider

logger = structlog.get_logger(logger_name=__name__)

# URL-to-source-type and citation tier mapping.
# More specific patterns are checked first.
_URL_TIER_MAP: list[tuple[re.Pattern[str], str, int]] = [
    (re.compile(r"residentadvisor\.net", re.IGNORECASE), "press", 2),
    (re.compile(r"ra\.co", re.IGNORECASE), "press", 2),
    (re.compile(r"mixmag\.net", re.IGNORECASE), "press", 2),
    (re.compile(r"djmag\.com", re.IGNORECASE), "press", 2),
    (re.compile(r"thequietus\.com", re.IGNORECASE), "press", 2),
    (re.compile(r"pitchfork\.com", re.IGNORECASE), "press", 2),
    (re.compile(r"factmag\.com", re.IGNORECASE), "press", 2),
    (re.compile(r"thefader\.com", re.IGNORECASE), "press", 2),
    (re.compile(r"xlr8r\.com", re.IGNORECASE), "press", 2),
    (re.compile(r"electronicbeats\.net", re.IGNORECASE), "press", 2),
    (re.compile(r"redbullmusicacademy\.", re.IGNORECASE), "interview", 3),
    (re.compile(r"rbma\.", re.IGNORECASE), "interview", 3),
    (re.compile(r"boilerroom\.", re.IGNORECASE), "interview", 3),
    (re.compile(r"wikipedia\.org", re.IGNORECASE), "reference", 4),
    (re.compile(r"discogs\.com", re.IGNORECASE), "reference", 4),
    (re.compile(r"musicbrainz\.org", re.IGNORECASE), "reference", 4),
    (re.compile(r"reddit\.com", re.IGNORECASE), "forum", 6),
    (re.compile(r"forums?\.", re.IGNORECASE), "forum", 6),
]


class ArticleProcessor:
    """Processes web articles and local article files into :class:`DocumentChunk` objects.

    URL-based processing infers source type and citation tier from domain
    patterns.  Local file processing accepts explicit metadata overrides.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_url(
        self,
        url: str,
        article_scraper: IArticleProvider,
    ) -> list[DocumentChunk]:
        """Fetch an article from *url* and convert to document chunks.

        Parameters
        ----------
        url:
            The article URL to fetch.
        article_scraper:
            An article extraction provider (e.g. trafilatura-backed).

        Returns
        -------
        list[DocumentChunk]
            A single-element list containing the full article text as one
            chunk.  Returns an empty list if extraction fails.
        """
        content = await article_scraper.extract_content(url)
        if not content or not content.text.strip():
            logger.warning("article_extraction_empty", url=url)
            return []

        source_type, tier = self._classify_url(url)
        source_id = hashlib.sha256(url.encode()).hexdigest()

        chunk = DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            text=content.text,
            source_id=source_id,
            source_title=content.title or url,
            source_type=source_type,
            author=content.author,
            publication_date=content.date,
            citation_tier=tier,
        )

        logger.info(
            "article_url_processed",
            url=url,
            source_type=source_type,
            tier=tier,
            text_length=len(content.text),
        )
        return [chunk]

    def process_file(
        self,
        file_path: str,
        source_type: str = "article",
        tier: int = 5,
    ) -> list[DocumentChunk]:
        """Read a local article file and convert to a document chunk.

        Parameters
        ----------
        file_path:
            Path to a UTF-8 text or HTML file.
        source_type:
            Source type label (default ``"article"``).
        tier:
            Citation tier (default 5 — web content).

        Returns
        -------
        list[DocumentChunk]
            A single-element list with the file's content, or empty on read failure.
        """
        try:
            with open(file_path, encoding="utf-8") as fh:
                text = fh.read()
        except OSError:
            logger.error("article_file_read_failed", file_path=file_path)
            return []

        if not text.strip():
            return []

        source_id = hashlib.sha256(text.encode()).hexdigest()
        chunk = DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            text=text,
            source_id=source_id,
            source_title=file_path.rsplit("/", maxsplit=1)[-1],
            source_type=source_type,
            citation_tier=tier,
        )

        logger.info(
            "article_file_processed",
            file_path=file_path,
            source_type=source_type,
            text_length=len(text),
        )
        return [chunk]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_url(url: str) -> tuple[str, int]:
        """Return ``(source_type, citation_tier)`` based on URL domain patterns."""
        for pattern, stype, tier in _URL_TIER_MAP:
            if pattern.search(url):
                return stype, tier
        # Default: generic web article.
        return "article", 5

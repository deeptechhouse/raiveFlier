"""Source processor for EPUB book files.

Reads EPUB files using ebooklib, extracts text from each document item
(typically one per chapter), and returns logical sections as
:class:`~src.models.rag.DocumentChunk` objects with ``source_type = "book"``
and ``citation_tier = 1``.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import date

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

import structlog

from src.models.rag import DocumentChunk

logger = structlog.get_logger(logger_name=__name__)

# Regex to collapse excessive whitespace while preserving paragraph breaks.
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


class EPUBProcessor:
    """Processes EPUB files into :class:`DocumentChunk` sections.

    Extracts text from each XHTML document item in the EPUB spine,
    strips HTML tags via BeautifulSoup, and returns one
    :class:`DocumentChunk` per chapter/document for downstream chunking
    by :class:`TextChunker`.
    """

    def process(
        self,
        file_path: str,
        title: str,
        author: str,
        year: int,
    ) -> list[DocumentChunk]:
        """Read an EPUB file and split into chapter-level document chunks.

        Parameters
        ----------
        file_path:
            Path to the EPUB file.
        title:
            Human-readable book title.
        author:
            Book author name.
        year:
            Publication year (used for ``publication_date``).

        Returns
        -------
        list[DocumentChunk]
            One chunk per chapter/document item.
        """
        chapters = self._extract_chapters(file_path)
        if not chapters:
            return []

        full_text = "\n\n".join(text for _, text in chapters)
        source_id = hashlib.sha256(full_text.encode()).hexdigest()
        pub_date = date(year, 1, 1)

        chunks: list[DocumentChunk] = []
        for chapter_title, chapter_text in chapters:
            display_title = f"{title} — {chapter_title}" if chapter_title else title
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                text=chapter_text,
                source_id=source_id,
                source_title=display_title,
                source_type="book",
                author=author,
                publication_date=pub_date,
                citation_tier=1,
            )
            chunks.append(chunk)

        logger.info(
            "epub_processed",
            file_path=file_path,
            title=title,
            chapters=len(chunks),
            source_id=source_id[:12],
        )
        return chunks

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_chapters(file_path: str) -> list[tuple[str, str]]:
        """Extract chapter text from each document item in the EPUB.

        Returns
        -------
        list[tuple[str, str]]
            List of ``(chapter_title, chapter_text)`` tuples.  Chapters
            with no meaningful text content are skipped.
        """
        try:
            book = epub.read_epub(file_path, options={"ignore_ncx": True})
        except Exception:
            logger.error("epub_open_failed", file_path=file_path)
            return []

        chapters: list[tuple[str, str]] = []

        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            html_content = item.get_content().decode("utf-8", errors="replace")
            soup = BeautifulSoup(html_content, "html.parser")

            # Extract chapter title from first heading tag.
            heading = soup.find(re.compile(r"^h[1-3]$"))
            chapter_title = heading.get_text(strip=True) if heading else ""

            # Extract all text, preserving paragraph structure.
            text = soup.get_text(separator="\n")
            text = _MULTI_SPACE.sub(" ", text)
            text = _MULTI_NEWLINE.sub("\n\n", text)
            text = text.strip()

            # Skip chapters with very little text (TOC, copyright, etc.).
            if len(text) < 100:
                continue

            chapters.append((chapter_title, text))

        if not chapters:
            logger.warning("epub_no_chapters_extracted", file_path=file_path)

        return chapters

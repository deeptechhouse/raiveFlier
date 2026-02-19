"""Source processor for plain-text book files.

Reads a UTF-8 text file, detects chapter boundaries, and returns logical
sections as :class:`~src.models.rag.DocumentChunk` objects with ``source_type
= "book"`` and ``citation_tier = 1``.  Actual chunking into embedding-sized
windows happens downstream in :class:`~src.services.ingestion.chunker.TextChunker`.
"""

from __future__ import annotations

import hashlib
import re
import uuid

import structlog

from src.models.rag import DocumentChunk

logger = structlog.get_logger(logger_name=__name__)

# Regex patterns for detecting chapter boundaries.
_CHAPTER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^Chapter\s+\d+", re.IGNORECASE),  # "Chapter 1", "chapter 12"
    re.compile(r"^PART\s+[IVXLCDM\d]+", re.IGNORECASE),  # "PART I", "PART 3"
    re.compile(r"^\d+\.\s+\S"),  # "1. Introduction"
    re.compile(r"^[A-Z][A-Z\s]{4,}$"),  # ALL-CAPS heading lines
]

# Regex for inline page markers like [p.142] or [page 42].
_PAGE_MARKER = re.compile(r"\[p(?:age)?\.?\s*(\d+)\]", re.IGNORECASE)


class BookProcessor:
    """Processes plain-text book files into :class:`DocumentChunk` sections.

    The processor does **not** perform token-level chunking — it returns
    one :class:`DocumentChunk` per detected chapter/section so the caller
    can feed sections into :class:`TextChunker` for fine-grained splitting.
    """

    def process(
        self,
        file_path: str,
        title: str,
        author: str,
        year: int,
    ) -> list[DocumentChunk]:
        """Read *file_path* and split into chapter-level document chunks.

        Parameters
        ----------
        file_path:
            Path to a UTF-8 plain-text file.
        title:
            Human-readable book title.
        author:
            Book author name.
        year:
            Publication year (used for ``publication_date``).

        Returns
        -------
        list[DocumentChunk]
            One chunk per detected chapter/section.
        """
        text = self._read_file(file_path)
        if not text:
            return []

        source_id = hashlib.sha256(text.encode()).hexdigest()
        sections = self._detect_sections(text)

        from datetime import date

        pub_date = date(year, 1, 1)

        chunks: list[DocumentChunk] = []
        for section_text, page_number in sections:
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                text=section_text,
                source_id=source_id,
                source_title=title,
                source_type="book",
                author=author,
                publication_date=pub_date,
                citation_tier=1,
                page_number=page_number,
            )
            chunks.append(chunk)

        logger.info(
            "book_processed",
            file_path=file_path,
            title=title,
            sections=len(chunks),
            source_id=source_id[:12],
        )
        return chunks

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _read_file(file_path: str) -> str:
        """Read and return the full text of *file_path* (UTF-8)."""
        try:
            with open(file_path, encoding="utf-8") as fh:
                return fh.read()
        except OSError:
            logger.error("book_read_failed", file_path=file_path)
            return ""

    def _detect_sections(self, text: str) -> list[tuple[str, str | None]]:
        """Split *text* into ``(section_text, page_number)`` tuples.

        Chapter boundaries are detected via heading patterns.  If no
        chapters are found the entire text is returned as a single section.
        Page numbers are extracted from inline markers like ``[p.142]``.
        """
        lines = text.split("\n")
        boundary_indices: list[int] = []

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            for pattern in _CHAPTER_PATTERNS:
                if pattern.match(stripped):
                    boundary_indices.append(idx)
                    break

        # If no chapter boundaries detected, return entire text as one section.
        if not boundary_indices:
            page = self._first_page_marker(text)
            return [(text.strip(), page)]

        sections: list[tuple[str, str | None]] = []

        # Content before the first chapter heading (e.g. preface).
        if boundary_indices[0] > 0:
            pre_text = "\n".join(lines[: boundary_indices[0]]).strip()
            if pre_text:
                sections.append((pre_text, self._first_page_marker(pre_text)))

        for i, start in enumerate(boundary_indices):
            end = boundary_indices[i + 1] if i + 1 < len(boundary_indices) else len(lines)
            section_text = "\n".join(lines[start:end]).strip()
            if section_text:
                page = self._first_page_marker(section_text)
                sections.append((section_text, page))

        return sections

    @staticmethod
    def _first_page_marker(text: str) -> str | None:
        """Return the first ``[p.NNN]`` page marker found in *text*, or ``None``."""
        match = _PAGE_MARKER.search(text)
        return match.group(1) if match else None

"""Source processor for PDF book and document files.

Reads PDF files using PyMuPDF (fitz), extracts text page-by-page, detects
chapter boundaries, and returns logical sections as
:class:`~src.models.rag.DocumentChunk` objects with ``citation_tier = 1``.

Supports both text-based PDFs and scanned PDFs with embedded OCR text layers.
When no chapter headings are detected (e.g. scanned documents without structure),
pages are grouped into fixed-size ~10-page sections as a fallback.

Like BookProcessor, this returns chapter-level chunks for downstream
fine-grained splitting by TextChunker.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import date

import fitz  # PyMuPDF -- the "fitz" import name is a PyMuPDF convention
import structlog

from src.models.rag import DocumentChunk

logger = structlog.get_logger(logger_name=__name__)

# Same chapter-detection patterns as BookProcessor for consistency.
_CHAPTER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^Chapter\s+\d+", re.IGNORECASE),
    re.compile(r"^PART\s+[IVXLCDM\d]+", re.IGNORECASE),
    re.compile(r"^\d+\.\s+\S"),
    re.compile(r"^[A-Z][A-Z\s]{4,}$"),
]


class PDFProcessor:
    """Processes PDF files into :class:`DocumentChunk` sections.

    Extracts text page-by-page, groups pages into chapter-level sections
    using heading detection, and returns one :class:`DocumentChunk` per
    section for downstream chunking by :class:`TextChunker`.
    """

    def process(
        self,
        file_path: str,
        title: str,
        author: str,
        year: int,
    ) -> list[DocumentChunk]:
        """Read a PDF file and split into chapter-level document chunks.

        Parameters
        ----------
        file_path:
            Path to the PDF file.
        title:
            Human-readable document title.
        author:
            Document author name.
        year:
            Publication year (used for ``publication_date``).

        Returns
        -------
        list[DocumentChunk]
            One chunk per detected chapter/section.
        """
        pages = self._extract_pages(file_path)
        if not pages:
            return []

        full_text = "\n\n".join(text for _, text in pages)
        source_id = hashlib.sha256(full_text.encode()).hexdigest()
        pub_date = date(year, 1, 1)

        sections = self._detect_sections(pages)

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
            "pdf_processed",
            file_path=file_path,
            title=title,
            pages=len(pages),
            sections=len(chunks),
            source_id=source_id[:12],
        )
        return chunks

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pages(file_path: str) -> list[tuple[int, str]]:
        """Extract text from each page of the PDF.

        Returns
        -------
        list[tuple[int, str]]
            List of ``(page_number, page_text)`` tuples.  Page numbers are
            1-based.  Pages with no extractable text are skipped.
        """
        try:
            doc = fitz.open(file_path)
        except Exception:
            logger.error("pdf_open_failed", file_path=file_path)
            return []

        pages: list[tuple[int, str]] = []
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text").strip()
                if text:
                    pages.append((page_num + 1, text))
        finally:
            doc.close()

        if not pages:
            logger.warning("pdf_no_text_extracted", file_path=file_path)

        return pages

    def _detect_sections(
        self,
        pages: list[tuple[int, str]],
    ) -> list[tuple[str, str]]:
        """Group pages into chapter-level sections.

        Scans the first few lines of each page for chapter heading patterns.
        When a heading is found, a new section begins.  If no headings are
        found, pages are grouped into fixed-size sections of
        ``_PAGES_PER_SECTION`` pages.

        Returns
        -------
        list[tuple[str, str]]
            List of ``(section_text, start_page_number)`` tuples.
        """
        # Find chapter boundary pages.
        boundary_indices: list[int] = []
        for idx, (_, page_text) in enumerate(pages):
            first_lines = page_text.split("\n")[:5]
            for line in first_lines:
                stripped = line.strip()
                if not stripped:
                    continue
                for pattern in _CHAPTER_PATTERNS:
                    if pattern.match(stripped):
                        boundary_indices.append(idx)
                        break
                else:
                    continue
                break

        # If no chapter boundaries detected, group into ~10-page sections.
        if not boundary_indices:
            return self._group_by_page_count(pages, pages_per_section=10)

        sections: list[tuple[str, str]] = []

        # Content before first chapter (e.g. preface, table of contents).
        if boundary_indices[0] > 0:
            pre_pages = pages[: boundary_indices[0]]
            pre_text = "\n\n".join(text for _, text in pre_pages)
            if pre_text.strip():
                sections.append((pre_text.strip(), str(pre_pages[0][0])))

        for i, start_idx in enumerate(boundary_indices):
            end_idx = (
                boundary_indices[i + 1]
                if i + 1 < len(boundary_indices)
                else len(pages)
            )
            section_pages = pages[start_idx:end_idx]
            section_text = "\n\n".join(text for _, text in section_pages)
            if section_text.strip():
                sections.append(
                    (section_text.strip(), str(section_pages[0][0]))
                )

        return sections

    @staticmethod
    def _group_by_page_count(
        pages: list[tuple[int, str]],
        pages_per_section: int = 10,
    ) -> list[tuple[str, str]]:
        """Group pages into fixed-size sections when no chapters are detected."""
        sections: list[tuple[str, str]] = []
        for start in range(0, len(pages), pages_per_section):
            batch = pages[start : start + pages_per_section]
            text = "\n\n".join(t for _, t in batch)
            if text.strip():
                sections.append((text.strip(), str(batch[0][0])))
        return sections

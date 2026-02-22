"""Unit tests for source processors (PDF, EPUB, Analysis).

Tests cover the main processing paths for each processor, including
success cases, empty inputs, error handling, and entity extraction.
All file I/O and external library calls are mocked.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.models.rag import DocumentChunk


# ======================================================================
# PDFProcessor
# ======================================================================


class TestPDFProcessor:
    """Tests for the PDFProcessor that reads PDFs via PyMuPDF (fitz)."""

    @patch("src.services.ingestion.source_processors.pdf_processor.fitz")
    def test_process_success(self, mock_fitz: MagicMock) -> None:
        """Test successful PDF processing with multiple pages and chapter detection."""
        from src.services.ingestion.source_processors.pdf_processor import PDFProcessor

        # Create mock pages with chapter heading on page 1
        mock_page_1 = MagicMock()
        mock_page_1.get_text.return_value = (
            "Chapter 1\n\nThe history of Detroit techno begins in the "
            "mid-1980s with the Belleville Three."
        )

        mock_page_2 = MagicMock()
        mock_page_2.get_text.return_value = (
            "Juan Atkins, Derrick May, and Kevin Saunderson created a "
            "futuristic sound that drew from Kraftwerk."
        )

        mock_page_3 = MagicMock()
        mock_page_3.get_text.return_value = (
            "Chapter 2\n\nAcross the Atlantic, acid house exploded in "
            "the UK during the late 1980s."
        )

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=3)
        mock_doc.__getitem__ = MagicMock(
            side_effect=lambda i: [mock_page_1, mock_page_2, mock_page_3][i]
        )
        mock_doc.close = MagicMock()

        mock_fitz.open.return_value = mock_doc

        processor = PDFProcessor()
        chunks = processor.process(
            file_path="/data/books/energy_flash.pdf",
            title="Energy Flash",
            author="Simon Reynolds",
            year=1998,
        )

        assert len(chunks) >= 1
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert all(c.source_type == "book" for c in chunks)
        assert all(c.citation_tier == 1 for c in chunks)
        assert all(c.author == "Simon Reynolds" for c in chunks)
        assert all(c.publication_date == date(1998, 1, 1) for c in chunks)
        # All chunks share the same source_id (hash of full text)
        source_ids = {c.source_id for c in chunks}
        assert len(source_ids) == 1

    @patch("src.services.ingestion.source_processors.pdf_processor.fitz")
    def test_process_empty_pdf(self, mock_fitz: MagicMock) -> None:
        """Test processing a PDF with no extractable text returns empty list."""
        from src.services.ingestion.source_processors.pdf_processor import PDFProcessor

        mock_page = MagicMock()
        mock_page.get_text.return_value = ""

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_doc.close = MagicMock()

        mock_fitz.open.return_value = mock_doc

        processor = PDFProcessor()
        chunks = processor.process(
            file_path="/data/books/empty.pdf",
            title="Empty Book",
            author="Unknown",
            year=2000,
        )

        assert chunks == []

    @patch("src.services.ingestion.source_processors.pdf_processor.fitz")
    def test_process_open_error(self, mock_fitz: MagicMock) -> None:
        """Test processing when fitz.open raises an exception returns empty list."""
        from src.services.ingestion.source_processors.pdf_processor import PDFProcessor

        mock_fitz.open.side_effect = RuntimeError("Corrupted PDF")

        processor = PDFProcessor()
        chunks = processor.process(
            file_path="/data/books/corrupted.pdf",
            title="Corrupted PDF",
            author="Unknown",
            year=2000,
        )

        assert chunks == []

    @patch("src.services.ingestion.source_processors.pdf_processor.fitz")
    def test_process_no_chapter_boundaries(self, mock_fitz: MagicMock) -> None:
        """Test fallback to page-count grouping when no chapter headings detected."""
        from src.services.ingestion.source_processors.pdf_processor import PDFProcessor

        # Create 15 pages with no chapter headings
        mock_pages = []
        for i in range(15):
            page = MagicMock()
            page.get_text.return_value = f"Some text on page {i + 1} about electronic music."
            mock_pages.append(page)

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=15)
        mock_doc.__getitem__ = MagicMock(side_effect=lambda i: mock_pages[i])
        mock_doc.close = MagicMock()

        mock_fitz.open.return_value = mock_doc

        processor = PDFProcessor()
        chunks = processor.process(
            file_path="/data/books/no_chapters.pdf",
            title="No Chapters",
            author="Writer",
            year=2010,
        )

        # With 15 pages and default 10 pages per section, expect 2 sections
        assert len(chunks) == 2
        assert all(c.source_type == "book" for c in chunks)

    @patch("src.services.ingestion.source_processors.pdf_processor.fitz")
    def test_process_with_preface_before_chapter(self, mock_fitz: MagicMock) -> None:
        """Test that content before the first chapter heading is captured."""
        from src.services.ingestion.source_processors.pdf_processor import PDFProcessor

        mock_page_preface = MagicMock()
        mock_page_preface.get_text.return_value = "Preface\n\nThis book covers the history of rave."

        mock_page_ch1 = MagicMock()
        mock_page_ch1.get_text.return_value = (
            "Chapter 1\n\nThe rave scene emerged from underground warehouses."
        )

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=2)
        mock_doc.__getitem__ = MagicMock(
            side_effect=lambda i: [mock_page_preface, mock_page_ch1][i]
        )
        mock_doc.close = MagicMock()

        mock_fitz.open.return_value = mock_doc

        processor = PDFProcessor()
        chunks = processor.process(
            file_path="/data/books/with_preface.pdf",
            title="Rave History",
            author="Author X",
            year=2005,
        )

        # Expect preface section + Chapter 1 section = 2 chunks
        assert len(chunks) == 2
        assert "Preface" in chunks[0].text
        assert "Chapter 1" in chunks[1].text

    @patch("src.services.ingestion.source_processors.pdf_processor.fitz")
    def test_process_page_numbers_in_chunks(self, mock_fitz: MagicMock) -> None:
        """Test that page_number is set on the produced chunks."""
        from src.services.ingestion.source_processors.pdf_processor import PDFProcessor

        mock_page = MagicMock()
        mock_page.get_text.return_value = "Some content about acid house in 1988."

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_doc.close = MagicMock()

        mock_fitz.open.return_value = mock_doc

        processor = PDFProcessor()
        chunks = processor.process(
            file_path="/data/books/one_page.pdf",
            title="One Page",
            author="Author",
            year=2000,
        )

        assert len(chunks) == 1
        assert chunks[0].page_number is not None
        assert chunks[0].page_number == "1"


# ======================================================================
# EPUBProcessor
# ======================================================================


class TestEPUBProcessor:
    """Tests for the EPUBProcessor that reads EPUB files via ebooklib."""

    @patch("src.services.ingestion.source_processors.epub_processor.epub")
    def test_process_success(self, mock_epub_module: MagicMock) -> None:
        """Test successful EPUB processing with chapter extraction."""
        from src.services.ingestion.source_processors.epub_processor import EPUBProcessor

        # Create mock EPUB items (chapters)
        mock_item_1 = MagicMock()
        mock_item_1.get_content.return_value = (
            b"<html><body><h1>Chapter 1: Origins</h1>"
            b"<p>Detroit techno emerged from the creative vision of the Belleville Three. "
            b"Their music drew from European synth-pop and funk, creating a futuristic "
            b"sound that would reshape electronic music forever.</p>"
            b"</body></html>"
        )

        mock_item_2 = MagicMock()
        mock_item_2.get_content.return_value = (
            b"<html><body><h2>Chapter 2: Acid House</h2>"
            b"<p>The acid house movement transformed British youth culture in 1988. "
            b"Warehouse parties and outdoor raves attracted thousands, powered by the "
            b"revolutionary sound of the Roland TB-303 bass synthesizer.</p>"
            b"</body></html>"
        )

        # Short item that should be skipped (< 100 chars)
        mock_item_short = MagicMock()
        mock_item_short.get_content.return_value = (
            b"<html><body><p>Copyright page</p></body></html>"
        )

        mock_book = MagicMock()
        mock_book.get_items_of_type.return_value = [
            mock_item_1,
            mock_item_short,
            mock_item_2,
        ]

        mock_epub_module.read_epub.return_value = mock_book

        processor = EPUBProcessor()
        chunks = processor.process(
            file_path="/data/books/rave_history.epub",
            title="Rave History",
            author="Simon Reynolds",
            year=1998,
        )

        assert len(chunks) == 2  # Short chapter should be skipped
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert all(c.source_type == "book" for c in chunks)
        assert all(c.citation_tier == 1 for c in chunks)
        assert all(c.author == "Simon Reynolds" for c in chunks)
        assert all(c.publication_date == date(1998, 1, 1) for c in chunks)

        # Chapter titles should be incorporated into source_title
        assert "Chapter 1: Origins" in chunks[0].source_title
        assert "Chapter 2: Acid House" in chunks[1].source_title

        # All chunks share the same source_id
        source_ids = {c.source_id for c in chunks}
        assert len(source_ids) == 1

    @patch("src.services.ingestion.source_processors.epub_processor.epub")
    def test_process_empty_epub(self, mock_epub_module: MagicMock) -> None:
        """Test processing an EPUB with no usable chapters returns empty list."""
        from src.services.ingestion.source_processors.epub_processor import EPUBProcessor

        # Only a copyright page (too short)
        mock_item = MagicMock()
        mock_item.get_content.return_value = b"<html><body><p>Short.</p></body></html>"

        mock_book = MagicMock()
        mock_book.get_items_of_type.return_value = [mock_item]

        mock_epub_module.read_epub.return_value = mock_book

        processor = EPUBProcessor()
        chunks = processor.process(
            file_path="/data/books/empty.epub",
            title="Empty Book",
            author="Unknown",
            year=2000,
        )

        assert chunks == []

    @patch("src.services.ingestion.source_processors.epub_processor.epub")
    def test_process_open_error(self, mock_epub_module: MagicMock) -> None:
        """Test processing when epub.read_epub raises an exception returns empty list."""
        from src.services.ingestion.source_processors.epub_processor import EPUBProcessor

        mock_epub_module.read_epub.side_effect = Exception("Corrupted EPUB")

        processor = EPUBProcessor()
        chunks = processor.process(
            file_path="/data/books/corrupted.epub",
            title="Corrupted EPUB",
            author="Unknown",
            year=2000,
        )

        assert chunks == []

    @patch("src.services.ingestion.source_processors.epub_processor.epub")
    def test_process_chapter_without_heading(self, mock_epub_module: MagicMock) -> None:
        """Test that chapters without h1-h3 headings use the base title."""
        from src.services.ingestion.source_processors.epub_processor import EPUBProcessor

        mock_item = MagicMock()
        mock_item.get_content.return_value = (
            b"<html><body>"
            b"<p>This chapter has no heading element but contains substantial text about "
            b"the development of jungle and drum and bass music in the mid-1990s UK scene.</p>"
            b"</body></html>"
        )

        mock_book = MagicMock()
        mock_book.get_items_of_type.return_value = [mock_item]

        mock_epub_module.read_epub.return_value = mock_book

        processor = EPUBProcessor()
        chunks = processor.process(
            file_path="/data/books/no_headings.epub",
            title="Jungle Music",
            author="Author Y",
            year=2001,
        )

        assert len(chunks) == 1
        # Without a chapter heading, source_title should just be the base title
        assert chunks[0].source_title == "Jungle Music"

    @patch("src.services.ingestion.source_processors.epub_processor.epub")
    def test_process_whitespace_normalization(self, mock_epub_module: MagicMock) -> None:
        """Test that excessive whitespace is collapsed in extracted text."""
        from src.services.ingestion.source_processors.epub_processor import EPUBProcessor

        mock_item = MagicMock()
        mock_item.get_content.return_value = (
            b"<html><body>"
            b"<h1>Chapter 1</h1>"
            b"<p>First paragraph about techno.</p>"
            b"<p></p><p></p><p></p><p></p>"
            b"<p>Second paragraph with    extra     spaces about house music culture.</p>"
            b"</body></html>"
        )

        mock_book = MagicMock()
        mock_book.get_items_of_type.return_value = [mock_item]

        mock_epub_module.read_epub.return_value = mock_book

        processor = EPUBProcessor()
        chunks = processor.process(
            file_path="/data/books/whitespace.epub",
            title="Whitespace Test",
            author="Author Z",
            year=2015,
        )

        assert len(chunks) == 1
        text = chunks[0].text
        # Should not have triple+ newlines or double+ spaces
        assert "\n\n\n" not in text
        assert "    " not in text


# ======================================================================
# AnalysisProcessor
# ======================================================================


class TestAnalysisProcessor:
    """Tests for the AnalysisProcessor that converts pipeline results to chunks."""

    def _make_pipeline_state(
        self,
        *,
        artists: list | None = None,
        venues: list | None = None,
        promoters: list | None = None,
        narrative: str | None = None,
    ) -> MagicMock:
        """Build a minimal mock PipelineState with configurable research results."""
        from src.models.entities import Artist, Promoter, Release, Venue

        mock_state = MagicMock()
        mock_state.session_id = "test-session-001"

        research_results = []
        for a in (artists or []):
            result = MagicMock()
            result.artist = a
            result.venue = None
            result.promoter = None
            research_results.append(result)

        for v in (venues or []):
            result = MagicMock()
            result.artist = None
            result.venue = v
            result.promoter = None
            research_results.append(result)

        for p in (promoters or []):
            result = MagicMock()
            result.artist = None
            result.venue = None
            result.promoter = p
            research_results.append(result)

        mock_state.research_results = research_results

        if narrative:
            mock_imap = MagicMock()
            mock_imap.narrative = narrative
            mock_state.interconnection_map = mock_imap
        else:
            mock_state.interconnection_map = None

        return mock_state

    def test_process_success_with_artist(self) -> None:
        """Test processing a pipeline state with artist research results."""
        from src.models.entities import Artist, Release
        from src.services.ingestion.source_processors.analysis_processor import (
            AnalysisProcessor,
        )

        artist = Artist(
            name="Carl Cox",
            aliases=["The King"],
            confidence=0.95,
            profile_summary="UK techno DJ and producer.",
            releases=[
                Release(title="Phat Trax", label="React", year=1995),
                Release(title="Two Paintings", label="Intec", year=1996),
            ],
        )

        state = self._make_pipeline_state(artists=[artist])
        processor = AnalysisProcessor()
        chunks = processor.process(state)

        assert len(chunks) >= 1
        artist_chunk = chunks[0]
        assert isinstance(artist_chunk, DocumentChunk)
        assert artist_chunk.source_type == "analysis"
        assert artist_chunk.citation_tier == 5
        assert "Carl Cox" in artist_chunk.text
        assert "Carl Cox" in artist_chunk.entity_tags
        assert "The King" in artist_chunk.entity_tags
        assert "Analysis: Carl Cox" in artist_chunk.source_title
        assert "analysis-test-session-001" == artist_chunk.source_id

    def test_process_success_with_venue(self) -> None:
        """Test processing a pipeline state with venue research results."""
        from src.models.entities import Venue
        from src.services.ingestion.source_processors.analysis_processor import (
            AnalysisProcessor,
        )

        venue = Venue(
            name="Tresor",
            city="Berlin",
            country="Germany",
            history="Opened in 1991 in a former department store vault.",
            notable_events=["Love Parade afterparty"],
            cultural_significance="Epicenter of Berlin techno",
        )

        state = self._make_pipeline_state(venues=[venue])
        processor = AnalysisProcessor()
        chunks = processor.process(state)

        assert len(chunks) >= 1
        venue_chunk = chunks[0]
        assert venue_chunk.source_type == "analysis"
        assert "Tresor" in venue_chunk.text
        assert "Tresor" in venue_chunk.entity_tags
        assert "Berlin" in venue_chunk.geographic_tags
        assert "Germany" in venue_chunk.geographic_tags

    def test_process_success_with_promoter(self) -> None:
        """Test processing a pipeline state with promoter research results."""
        from src.models.entities import Promoter
        from src.services.ingestion.source_processors.analysis_processor import (
            AnalysisProcessor,
        )

        promoter = Promoter(
            name="Tresor Records",
            event_history=["Tresor nights", "Love Parade"],
            affiliated_artists=["Carl Cox", "Jeff Mills"],
            affiliated_venues=["Tresor Berlin"],
        )

        state = self._make_pipeline_state(promoters=[promoter])
        processor = AnalysisProcessor()
        chunks = processor.process(state)

        assert len(chunks) >= 1
        promoter_chunk = chunks[0]
        assert promoter_chunk.source_type == "analysis"
        assert "Tresor Records" in promoter_chunk.text
        assert "Tresor Records" in promoter_chunk.entity_tags

    def test_process_with_interconnection_narrative(self) -> None:
        """Test that the interconnection narrative produces a chunk."""
        from src.models.entities import Artist
        from src.services.ingestion.source_processors.analysis_processor import (
            AnalysisProcessor,
        )

        artist = Artist(name="Derrick May", confidence=0.9)
        narrative = "Derrick May was a key figure connecting Detroit and Berlin techno scenes."

        state = self._make_pipeline_state(artists=[artist], narrative=narrative)
        processor = AnalysisProcessor()
        chunks = processor.process(state)

        # One artist chunk + one narrative chunk
        assert len(chunks) == 2
        narrative_chunk = chunks[-1]
        assert narrative_chunk.text == narrative
        assert "Interconnection:" in narrative_chunk.source_title
        assert "Derrick May" in narrative_chunk.entity_tags

    def test_process_empty_pipeline_state(self) -> None:
        """Test processing a pipeline state with no research results."""
        from src.services.ingestion.source_processors.analysis_processor import (
            AnalysisProcessor,
        )

        state = self._make_pipeline_state()
        processor = AnalysisProcessor()
        chunks = processor.process(state)

        assert chunks == []

    def test_process_with_multiple_entities(self) -> None:
        """Test processing multiple artists, venues, and promoters at once."""
        from src.models.entities import Artist, Promoter, Venue
        from src.services.ingestion.source_processors.analysis_processor import (
            AnalysisProcessor,
        )

        artists = [
            Artist(name="Carl Cox", confidence=0.95),
            Artist(name="Jeff Mills", confidence=0.90),
        ]
        venues = [
            Venue(name="Tresor", city="Berlin", country="Germany"),
        ]
        promoters = [
            Promoter(name="Tresor Records"),
        ]

        state = self._make_pipeline_state(
            artists=artists, venues=venues, promoters=promoters,
            narrative="A narrative about Berlin techno.",
        )
        processor = AnalysisProcessor()
        chunks = processor.process(state)

        # 2 artists + 1 venue + 1 promoter + 1 narrative = 5
        assert len(chunks) == 5
        assert all(c.source_type == "analysis" for c in chunks)
        assert all(c.citation_tier == 5 for c in chunks)

        # All share the same source_id
        source_ids = {c.source_id for c in chunks}
        assert len(source_ids) == 1

    def test_process_artist_with_releases_in_summary(self) -> None:
        """Test that artist releases appear in the summary text."""
        from src.models.entities import Artist, Release
        from src.services.ingestion.source_processors.analysis_processor import (
            AnalysisProcessor,
        )

        artist = Artist(
            name="Jeff Mills",
            confidence=0.9,
            releases=[
                Release(title="The Bells", label="Axis", year=1996),
            ],
        )

        state = self._make_pipeline_state(artists=[artist])
        processor = AnalysisProcessor()
        chunks = processor.process(state)

        assert len(chunks) == 1
        assert "The Bells" in chunks[0].text
        assert "Axis" in chunks[0].text
        assert "1996" in chunks[0].text

    def test_process_venue_with_full_details(self) -> None:
        """Test venue summary includes location, history, significance, and events."""
        from src.models.entities import Venue
        from src.services.ingestion.source_processors.analysis_processor import (
            AnalysisProcessor,
        )

        venue = Venue(
            name="Berghain",
            city="Berlin",
            country="Germany",
            history="Opened in 2004 in a former power station.",
            cultural_significance="World-renowned techno club.",
            notable_events=["Panorama Bar opening", "New Year Marathon"],
        )

        state = self._make_pipeline_state(venues=[venue])
        processor = AnalysisProcessor()
        chunks = processor.process(state)

        assert len(chunks) == 1
        text = chunks[0].text
        assert "Berghain" in text
        assert "Berlin" in text
        assert "Germany" in text
        assert "2004" in text
        assert "World-renowned techno club" in text
        assert "Panorama Bar opening" in text

    def test_process_narrative_entity_tags_deduplicated(self) -> None:
        """Test that narrative chunk entity tags are deduplicated."""
        from src.models.entities import Artist
        from src.services.ingestion.source_processors.analysis_processor import (
            AnalysisProcessor,
        )

        # Two research results for the same artist name
        artist1 = Artist(name="Carl Cox", confidence=0.95)
        artist2 = Artist(name="Carl Cox", confidence=0.90)

        state = self._make_pipeline_state(
            artists=[artist1, artist2],
            narrative="Carl Cox connection narrative.",
        )
        processor = AnalysisProcessor()
        chunks = processor.process(state)

        # 2 artist chunks + 1 narrative chunk = 3
        assert len(chunks) == 3
        narrative_chunk = chunks[-1]
        # "Carl Cox" should appear only once in the narrative's entity_tags
        assert narrative_chunk.entity_tags.count("Carl Cox") == 1

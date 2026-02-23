"""Source processors for the raiveFlier ingestion pipeline.

Each processor converts a specific source format into chapter/section-level
:class:`~src.models.rag.DocumentChunk` objects.  These chunks are then fed
to the TextChunker for fine-grained splitting, and onward through the
tag -> embed -> store stages.

Available processors and their input formats:

- **BookProcessor**     -- Plain-text (.txt) books with chapter headings
- **PDFProcessor**      -- PDF books/documents via PyMuPDF page extraction
- **EPUBProcessor**     -- EPUB books via ebooklib + BeautifulSoup HTML stripping
- **ArticleProcessor**  -- Web articles (via URL scraping) or local .txt/.html files
- **RAEventProcessor**  -- Structured RA.co event data (RAEvent models)
- **AnalysisProcessor** -- Completed pipeline analyses (feedback loop re-ingestion)

Each processor assigns a ``citation_tier`` reflecting source authority:
  Tier 1 = books, Tier 2 = press/music publications, Tier 3 = events/interviews,
  Tier 4 = reference databases, Tier 5 = analyses/web content, Tier 6 = forums.
"""

from src.services.ingestion.source_processors.analysis_processor import (
    AnalysisProcessor,
)
from src.services.ingestion.source_processors.article_processor import (
    ArticleProcessor,
)
from src.services.ingestion.source_processors.book_processor import BookProcessor
from src.services.ingestion.source_processors.epub_processor import EPUBProcessor
from src.services.ingestion.source_processors.pdf_processor import PDFProcessor
from src.services.ingestion.source_processors.ra_event_processor import (
    RAEventProcessor,
)

__all__ = [
    "AnalysisProcessor",
    "ArticleProcessor",
    "BookProcessor",
    "EPUBProcessor",
    "PDFProcessor",
    "RAEventProcessor",
]

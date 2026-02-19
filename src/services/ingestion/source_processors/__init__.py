"""Source processors for the raiveFlier ingestion pipeline.

Each processor converts a specific source type (book, article, analysis)
into :class:`~src.models.rag.DocumentChunk` objects for downstream chunking,
tagging, embedding, and vector-store indexing.
"""

from src.services.ingestion.source_processors.analysis_processor import (
    AnalysisProcessor,
)
from src.services.ingestion.source_processors.article_processor import (
    ArticleProcessor,
)
from src.services.ingestion.source_processors.book_processor import BookProcessor
from src.services.ingestion.source_processors.pdf_processor import PDFProcessor

__all__ = [
    "AnalysisProcessor",
    "ArticleProcessor",
    "BookProcessor",
    "PDFProcessor",
]

"""Document ingestion pipeline for the raiveFlier RAG knowledge base.

Orchestrates the full pipeline: **process -> chunk -> tag -> embed -> store**.

Pipeline stages overview:

1. **Process** (source_processors/) -- Format-specific readers convert raw
   sources (books, PDFs, EPUBs, articles, RA events, completed analyses)
   into chapter/section-level DocumentChunk objects.

2. **Chunk** (chunker.py / TextChunker) -- Splits large sections into
   ~500-token overlapping windows, preserving paragraph and sentence
   boundaries so no chunk starts or ends mid-thought.

3. **Tag** (metadata_extractor.py / MetadataExtractor) -- Enriches each
   chunk with LLM-derived entity, geographic, and genre tags to enable
   filtered semantic retrieval at query time.

4. **Embed** (via IEmbeddingProvider) -- Generates dense vector embeddings
   for each chunk's text content.

5. **Store** (via IVectorStoreProvider) -- Persists embedded chunks to the
   vector database (Qdrant) for semantic similarity search.

The IngestionService class orchestrates all five stages and provides
per-format entry points (ingest_book, ingest_pdf, ingest_article, etc.).
"""

from src.services.ingestion.chunker import TextChunker
from src.services.ingestion.ingestion_service import IngestionService
from src.services.ingestion.metadata_extractor import MetadataExtractor

__all__ = [
    "IngestionService",
    "MetadataExtractor",
    "TextChunker",
]

"""Document ingestion pipeline for the raiveFlier RAG knowledge base.

Orchestrates the full pipeline: **process → chunk → tag → embed → store**.
Source processors handle different document types (books, articles, prior
analyses); the chunker splits text into overlapping windows preserving
paragraph boundaries; the metadata extractor enriches chunks with LLM-derived
entity, geographic, and genre tags; and finally chunks are embedded and
persisted to the vector store.
"""

from src.services.ingestion.chunker import TextChunker
from src.services.ingestion.ingestion_service import IngestionService
from src.services.ingestion.metadata_extractor import MetadataExtractor

__all__ = [
    "IngestionService",
    "MetadataExtractor",
    "TextChunker",
]

#!/usr/bin/env python3
# =============================================================================
# scripts/batch_ingest.py — Bulk Book Ingestion Script
# =============================================================================
#
# Batch-ingests all PDF and EPUB files from a directory into the ChromaDB
# vector store. Designed for one-time corpus building when you have a
# collection of ebooks about electronic music, rave culture, DJ history, etc.
#
# The script automatically parses metadata (title, author, year) from
# Anna's Archive filename format, which follows this pattern:
#   "Title -- Author -- Edition/Location, Year -- Publisher -- ISBN -- Hash -- Anna's Archive.pdf"
#
# For files not matching this pattern, it falls back to using the filename
# stem as the title with placeholder author and year values.
#
# Ingestion pipeline per book:
#   1. Parse metadata from filename
#   2. Extract text (PDF: page-by-page, EPUB: chapter-by-chapter)
#   3. Chunk text into ~512-token windows with overlap
#   4. Tag chunks with semantic metadata via LLM
#   5. Generate vector embeddings
#   6. Store in ChromaDB
#
# Usage:
#   python scripts/batch_ingest.py /path/to/books/
#
# Environment: Requires .env with at least one embedding provider key
# (OPENAI_API_KEY or OLLAMA_BASE_URL) and one LLM provider key.
# =============================================================================

"""Batch ingest all PDF and EPUB files from a directory.

Automatically extracts title, author, and year from Anna's Archive
filenames (format: Title -- Author -- ..., Year -- Publisher -- ...).
Falls back to filename-based guesses when parsing fails.

Usage:
    python scripts/batch_ingest.py /path/to/books/
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

# Add the project root to sys.path so we can import from src/.
# This is necessary because this script lives in scripts/ and is run
# directly (not as a module via python -m).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import Settings
from src.services.ingestion.chunker import TextChunker
from src.services.ingestion.ingestion_service import IngestionService
from src.services.ingestion.metadata_extractor import MetadataExtractor


def _build_embedding_provider(app_settings: Settings):
    """Select the first available embedding provider.

    Same fallback chain as src/cli/ingest.py: OpenAI -> Nomic/Ollama.
    Note: This duplicates the logic from the CLI module because scripts
    run standalone and cannot easily share the CLI's factory functions.
    """
    if app_settings.openai_api_key:
        from src.providers.embedding.openai_embedding_provider import (
            OpenAIEmbeddingProvider,
        )
        provider = OpenAIEmbeddingProvider(settings=app_settings)
        if provider.is_available():
            return provider

    from src.providers.embedding.nomic_embedding_provider import (
        NomicEmbeddingProvider,
    )
    provider = NomicEmbeddingProvider(settings=app_settings)
    if provider.is_available():
        return provider

    return None


def _build_llm_provider(app_settings: Settings):
    """Select the first available LLM provider."""
    if app_settings.anthropic_api_key:
        from src.providers.llm.anthropic_provider import AnthropicLLMProvider
        return AnthropicLLMProvider(settings=app_settings)
    if app_settings.openai_api_key:
        from src.providers.llm.openai_provider import OpenAILLMProvider
        return OpenAILLMProvider(settings=app_settings)
    from src.providers.llm.ollama_provider import OllamaLLMProvider
    return OllamaLLMProvider(settings=app_settings)


def parse_annas_archive_filename(filename: str) -> dict:
    """Parse Anna's Archive filename format into metadata.

    Anna's Archive uses a double-dash delimited filename format:
      "Title -- Author -- Edition/Location, Year -- Publisher -- ISBN -- Hash -- Anna's Archive.ext"

    This function extracts title, author, and publication year from that pattern.

    Cleaning steps:
      1. Strip the "-- Anna's Archive" suffix
      2. Split on " -- " delimiter
      3. Clean title: replace underscores with spaces, collapse whitespace
      4. Clean author: remove parenthetical descriptions, trailing punctuation
      5. Search all parts for a 4-digit year (1900-2099 range)

    Falls back to year=2000 if no year is found (provides a reasonable default
    for the vector store metadata without breaking the ingestion pipeline).

    Returns: dict with keys "title", "author", "year"
    """
    stem = Path(filename).stem
    # Remove the "-- Anna's Archive" suffix that appears on all downloaded files.
    stem = re.sub(r"\s*--\s*Anna'?s Archive$", "", stem)

    # Split filename into parts on the " -- " delimiter.
    parts = [p.strip() for p in stem.split(" -- ")]

    # Part 1: Title (always the first segment)
    title = parts[0] if parts else stem
    title = re.sub(r"_", " ", title).strip()       # Underscores to spaces
    title = re.sub(r"\s{2,}", ": ", title).strip()  # Multiple spaces -> colon
    title = title.rstrip(",:").strip()               # Clean trailing punctuation

    author = ""
    year = 0

    # Part 2: Author (second segment if present)
    if len(parts) >= 2:
        author = parts[1]
        author = re.sub(r"_", " ", author).strip()
        # Remove parenthetical like "(Psychologist)" or "(Editor)"
        author = re.sub(r"\s*\(.*?\)\s*", " ", author).strip()
        author = author.rstrip(",").strip()

    # Search ALL parts for a 4-digit year (not just the author field).
    # The year often appears in the third part with edition/location info.
    for part in parts:
        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", part)
        if year_match:
            year = int(year_match.group(1))
            break

    if not year:
        year = 2000  # Fallback: prevents null year in vector store metadata

    return {"title": title, "author": author, "year": year}


async def main() -> None:
    """Main entry point: discovers books, builds ingestion service, processes each.

    The script processes books sequentially (not in parallel) to avoid
    overwhelming the embedding API with concurrent requests and to provide
    clear progress output. Each book is try/except wrapped so a single
    failure does not abort the entire batch.
    """
    if len(sys.argv) < 2:
        print("Usage: python scripts/batch_ingest.py /path/to/books/")
        sys.exit(1)

    book_dir = Path(sys.argv[1])
    if not book_dir.is_dir():
        print(f"Error: {book_dir} is not a directory")
        sys.exit(1)

    # Collect all PDF and EPUB files. Sorted for deterministic processing order.
    files = sorted(book_dir.glob("*.pdf")) + sorted(book_dir.glob("*.epub"))
    if not files:
        print(f"No PDF or EPUB files found in {book_dir}")
        sys.exit(0)

    print(f"Found {len(files)} books to ingest:\n")

    # Parse metadata from filenames
    books: list[dict] = []
    for f in files:
        meta = parse_annas_archive_filename(f.name)
        meta["file"] = str(f)
        meta["ext"] = f.suffix.lower()
        books.append(meta)
        print(f"  [{meta['ext'][1:].upper():4s}] {meta['title']}")
        print(f"         by {meta['author']} ({meta['year']})")
        print()

    # Build ingestion service
    settings = Settings()
    embedding_provider = _build_embedding_provider(settings)
    if embedding_provider is None:
        print("Error: No embedding provider available.")
        sys.exit(1)

    from src.providers.vector_store.chromadb_provider import ChromaDBProvider

    llm_provider = _build_llm_provider(settings)
    vector_store = ChromaDBProvider(
        embedding_provider=embedding_provider,
        persist_directory=settings.chromadb_persist_dir,
        collection_name=settings.chromadb_collection,
    )
    chunker = TextChunker()
    metadata_extractor = MetadataExtractor(llm=llm_provider)

    service = IngestionService(
        chunker=chunker,
        metadata_extractor=metadata_extractor,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    print(f"Embedding: {embedding_provider.get_provider_name()}")
    print(f"LLM: {llm_provider.get_provider_name()}")
    print("=" * 60)
    print()

    # Ingest each book sequentially with per-book error handling.
    total_chunks = 0
    total_tokens = 0
    succeeded = 0
    failed = 0

    for i, book in enumerate(books, 1):
        print(f"[{i}/{len(books)}] Ingesting: {book['title']}...")

        try:
            if book["ext"] == ".pdf":
                result = await service.ingest_pdf(
                    file_path=book["file"],
                    title=book["title"],
                    author=book["author"],
                    year=book["year"],
                )
            elif book["ext"] == ".epub":
                result = await service.ingest_epub(
                    file_path=book["file"],
                    title=book["title"],
                    author=book["author"],
                    year=book["year"],
                )
            else:
                print(f"  SKIP: Unsupported format {book['ext']}")
                continue

            total_chunks += result.chunks_created
            total_tokens += result.total_tokens
            succeeded += 1
            print(f"  OK: {result.chunks_created} chunks, {result.total_tokens} tokens, {result.ingestion_time:.1f}s")

        except Exception as exc:
            failed += 1
            print(f"  FAIL: {exc}")

        print()

    # Summary
    print("=" * 60)
    print("BATCH INGESTION COMPLETE")
    print(f"  Succeeded:    {succeeded}/{len(books)}")
    print(f"  Failed:       {failed}/{len(books)}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Total tokens: {total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())

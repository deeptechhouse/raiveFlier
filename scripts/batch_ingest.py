#!/usr/bin/env python3
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

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import Settings
from src.services.ingestion.chunker import TextChunker
from src.services.ingestion.ingestion_service import IngestionService
from src.services.ingestion.metadata_extractor import MetadataExtractor


def _build_embedding_provider(app_settings: Settings):
    """Select the first available embedding provider."""
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

    Format: Title -- Author -- Edition/Location, Year -- Publisher -- ISBN -- Hash -- Anna's Archive.ext
    """
    stem = Path(filename).stem
    # Remove "-- Anna's Archive" suffix
    stem = re.sub(r"\s*--\s*Anna'?s Archive$", "", stem)

    parts = [p.strip() for p in stem.split(" -- ")]

    title = parts[0] if parts else stem
    # Clean up title: underscores to spaces, fix colons
    title = re.sub(r"_", " ", title).strip()
    # Collapse multiple spaces to single, fix ": " patterns
    title = re.sub(r"\s{2,}", ": ", title).strip()
    # Remove trailing commas or colons
    title = title.rstrip(",:").strip()

    author = ""
    year = 0

    if len(parts) >= 2:
        author = parts[1]
        # Clean underscores in author name
        author = re.sub(r"_", " ", author).strip()
        # Remove parenthetical descriptions like "(Psychologist)"
        author = re.sub(r"\s*\(.*?\)\s*", " ", author).strip()
        # Remove trailing commas
        author = author.rstrip(",").strip()

    # Search all parts for a 4-digit year
    for part in parts:
        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", part)
        if year_match:
            year = int(year_match.group(1))
            break

    if not year:
        year = 2000  # fallback

    return {"title": title, "author": author, "year": year}


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/batch_ingest.py /path/to/books/")
        sys.exit(1)

    book_dir = Path(sys.argv[1])
    if not book_dir.is_dir():
        print(f"Error: {book_dir} is not a directory")
        sys.exit(1)

    # Collect all PDF and EPUB files
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

    # Ingest each book
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

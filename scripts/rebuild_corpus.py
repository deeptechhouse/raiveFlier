#!/usr/bin/env python3
"""Rebuild the entire RAG corpus from scratch.

Ingests all source material in order:
  1. Reference corpus (curated text files)
  2. RA Exchange transcripts
  3. Books (PDF + EPUB from a specified directory)

Usage:
    python scripts/rebuild_corpus.py /path/to/books/
    python scripts/rebuild_corpus.py  # skip books if no path given
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
    """Parse Anna's Archive filename format into metadata."""
    stem = Path(filename).stem
    stem = re.sub(r"\s*--\s*Anna'?s Archive$", "", stem)
    parts = [p.strip() for p in stem.split(" -- ")]

    title = parts[0] if parts else stem
    title = re.sub(r"_", " ", title).strip()
    title = re.sub(r"\s{2,}", ": ", title).strip()
    title = title.rstrip(",:").strip()

    author = ""
    year = 0

    if len(parts) >= 2:
        author = parts[1]
        author = re.sub(r"_", " ", author).strip()
        author = re.sub(r"\s*\(.*?\)\s*", " ", author).strip()
        author = author.rstrip(",").strip()

    for part in parts:
        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", part)
        if year_match:
            year = int(year_match.group(1))
            break

    if not year:
        year = 2000

    return {"title": title, "author": author, "year": year}


def build_ingestion_service() -> tuple[IngestionService, str]:
    """Bootstrap the ingestion service from project settings."""
    settings = Settings()

    embedding_provider = _build_embedding_provider(settings)
    if embedding_provider is None:
        print("Error: No embedding provider available. Check .env config.")
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

    model_name = embedding_provider.get_provider_name()
    dim = embedding_provider.get_dimension()
    print(f"Embedding: {model_name} ({dim} dimensions)")
    print(f"ChromaDB:  {settings.chromadb_persist_dir}")
    print()

    return service, settings.chromadb_persist_dir


async def ingest_reference_corpus(service: IngestionService) -> int:
    """Ingest the curated reference text files."""
    corpus_dir = PROJECT_ROOT / "data" / "reference_corpus"
    if not corpus_dir.is_dir():
        print(f"  Skipping — {corpus_dir} not found")
        return 0

    files = sorted(corpus_dir.glob("*.txt"))
    print(f"  Found {len(files)} reference files")

    results = await service.ingest_directory(str(corpus_dir), source_type="reference")
    total = sum(r.chunks_created for r in results)
    print(f"  Ingested {total} chunks from {len(results)} files")
    return total


async def ingest_transcripts(service: IngestionService) -> int:
    """Ingest all RA Exchange transcripts."""
    transcript_dir = PROJECT_ROOT / "transcripts" / "ra_exchange"
    if not transcript_dir.is_dir():
        print(f"  Skipping — {transcript_dir} not found")
        return 0

    files = sorted(transcript_dir.glob("*.txt"))
    print(f"  Found {len(files)} transcript files")

    # Process in batches to avoid overwhelming the API
    batch_size = 10
    total_chunks = 0
    total_files = 0

    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        results = []
        for f in batch:
            result = await _ingest_single_transcript(service, f)
            if result:
                results.append(result)

        batch_chunks = sum(r.chunks_created for r in results)
        total_chunks += batch_chunks
        total_files += len(results)
        print(f"  Batch {i // batch_size + 1}: {len(results)} files, {batch_chunks} chunks (total: {total_chunks})")

    print(f"  Ingested {total_chunks} chunks from {total_files} transcripts")
    return total_chunks


async def _ingest_single_transcript(service, file_path: Path):
    """Ingest a single transcript file as an interview source."""
    from src.services.ingestion.source_processors.article_processor import ArticleProcessor
    from src.models.rag import DocumentChunk

    processor = ArticleProcessor()
    raw_chunks = processor.process_file(str(file_path), source_type="interview")
    if not raw_chunks:
        return None

    from src.utils.text_normalizer import preprocess_transcript

    source_id = raw_chunks[0].source_id
    all_chunks: list[DocumentChunk] = []
    for rc in raw_chunks:
        text = preprocess_transcript(rc.text)
        metadata = {
            "source_id": rc.source_id,
            "source_title": rc.source_title,
            "source_type": rc.source_type,
            "citation_tier": rc.citation_tier,
        }
        chunks = service._chunker.chunk(text, metadata)
        all_chunks.extend(chunks)

    if not all_chunks:
        return None

    return await service._tag_embed_store(
        all_chunks, source_id=source_id, title=file_path.stem, start=time.monotonic()
    )


async def ingest_books(service: IngestionService, book_dir: Path) -> int:
    """Ingest all PDF and EPUB books from a directory."""
    if not book_dir.is_dir():
        print(f"  Skipping — {book_dir} not found")
        return 0

    files = sorted(book_dir.glob("*.pdf")) + sorted(book_dir.glob("*.epub"))
    files = [f for f in files if not f.name.startswith(".")]
    print(f"  Found {len(files)} books")

    total_chunks = 0
    for i, f in enumerate(files, 1):
        meta = parse_annas_archive_filename(f.name)
        ext = f.suffix.lower()
        print(f"  [{i}/{len(files)}] {meta['title']} ({ext[1:].upper()})")

        try:
            if ext == ".epub":
                result = await service.ingest_epub(
                    str(f), meta["title"], meta["author"], meta["year"]
                )
            elif ext == ".pdf":
                result = await service.ingest_pdf(
                    str(f), meta["title"], meta["author"], meta["year"]
                )
            else:
                print(f"    Skipping unsupported format: {ext}")
                continue

            total_chunks += result.chunks_created
            print(f"    -> {result.chunks_created} chunks in {result.ingestion_time}s")
        except Exception as exc:
            print(f"    ERROR: {exc}")

    print(f"  Ingested {total_chunks} chunks from {len(files)} books")
    return total_chunks


async def main() -> None:
    book_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    print("=" * 60)
    print("raiveFlier Corpus Rebuild")
    print("=" * 60)
    print()

    service, persist_dir = build_ingestion_service()
    grand_total = 0
    start = time.monotonic()

    # Phase 1: Reference corpus
    print("[1/3] Reference corpus")
    grand_total += await ingest_reference_corpus(service)
    print()

    # Phase 2: RA Exchange transcripts
    print("[2/3] RA Exchange transcripts")
    grand_total += await ingest_transcripts(service)
    print()

    # Phase 3: Books
    print("[3/3] Books")
    if book_dir:
        grand_total += await ingest_books(service, book_dir)
    else:
        print("  Skipping — no book directory provided")
    print()

    elapsed = time.monotonic() - start
    print("=" * 60)
    print(f"COMPLETE: {grand_total} total chunks in {elapsed:.1f}s")
    print(f"ChromaDB: {persist_dir}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

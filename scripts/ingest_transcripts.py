#!/usr/bin/env python3
"""Ingest existing RA Exchange transcripts into ChromaDB.

Reads the _progress.json to find completed-but-not-ingested transcripts
and runs them through the ingestion pipeline (chunk -> tag -> embed -> store).

Safe to run while transcribe_ra_exchange.py is still running — it only
processes transcripts that are already written to disk.

Usage:
    python scripts/ingest_transcripts.py
    python scripts/ingest_transcripts.py --batch-size 10
    python scripts/ingest_transcripts.py --transcript-dir transcripts/ra_exchange
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import Settings
from src.services.ingestion.chunker import TextChunker
from src.services.ingestion.ingestion_service import IngestionService
from src.services.ingestion.metadata_extractor import MetadataExtractor
from src.services.ingestion.source_processors.article_processor import ArticleProcessor
from src.models.rag import DocumentChunk


async def build_ingestion_service() -> tuple[IngestionService, str]:
    """Bootstrap the ingestion service from project settings."""
    settings = Settings()

    # Embedding provider — try OpenAI-compatible first, then local
    # sentence-transformers (same model, no API needed), then Nomic/Ollama.
    embedding_provider = None
    if settings.openai_api_key:
        from src.providers.embedding.openai_embedding_provider import (
            OpenAIEmbeddingProvider,
        )
        provider = OpenAIEmbeddingProvider(settings=settings)
        if provider.is_available():
            # Quick connectivity check — skip if API credits exhausted
            try:
                await provider.embed_single("test")
                embedding_provider = provider
            except Exception as _e:
                print(f"  OpenAI-compatible provider unavailable: {str(_e)[:120]}")
                print("  Falling back to local sentence-transformers...")

    if embedding_provider is None:
        from src.providers.embedding.sentence_transformer_embedding_provider import (
            SentenceTransformerEmbeddingProvider,
        )
        provider = SentenceTransformerEmbeddingProvider(
            model_name=settings.openai_embedding_model or None
        )
        if provider.is_available():
            embedding_provider = provider

    if embedding_provider is None:
        from src.providers.embedding.nomic_embedding_provider import (
            NomicEmbeddingProvider,
        )
        provider = NomicEmbeddingProvider(settings=settings)
        if provider.is_available():
            embedding_provider = provider

    if embedding_provider is None:
        print("Error: No embedding provider available. Check .env config.")
        sys.exit(1)

    # LLM provider
    if settings.anthropic_api_key:
        from src.providers.llm.anthropic_provider import AnthropicLLMProvider
        llm_provider = AnthropicLLMProvider(settings=settings)
    elif settings.openai_api_key:
        from src.providers.llm.openai_provider import OpenAILLMProvider
        llm_provider = OpenAILLMProvider(settings=settings)
    else:
        from src.providers.llm.ollama_provider import OllamaLLMProvider
        llm_provider = OllamaLLMProvider(settings=settings)

    from src.providers.vector_store.chromadb_provider import ChromaDBProvider

    vector_store = ChromaDBProvider(
        embedding_provider=embedding_provider,
        persist_directory=settings.chromadb_persist_dir,
        collection_name=settings.chromadb_collection,
    )

    service = IngestionService(
        chunker=TextChunker(),
        metadata_extractor=MetadataExtractor(llm=llm_provider),
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    return service, embedding_provider.get_provider_name()


async def ingest_transcript(
    service: IngestionService,
    file_path: str,
    title: str,
) -> tuple[int, int]:
    """Ingest a single transcript into ChromaDB. Returns (chunks, tokens)."""
    processor = ArticleProcessor()
    raw_chunks = processor.process_file(file_path, source_type="interview", tier=3)
    if not raw_chunks:
        return 0, 0

    source_id = raw_chunks[0].source_id
    chunker = TextChunker()

    from src.utils.text_normalizer import preprocess_transcript

    all_chunks: list[DocumentChunk] = []
    for rc in raw_chunks:
        text = preprocess_transcript(rc.text)
        metadata = {
            "source_id": rc.source_id,
            "source_title": title,
            "source_type": "interview",
            "citation_tier": 3,
            "author": "Resident Advisor",
        }
        chunks = chunker.chunk(text, metadata)
        all_chunks.extend(chunks)

    if not all_chunks:
        return 0, 0

    result = await service._tag_embed_store(
        all_chunks, source_id=source_id, title=title, start=time.monotonic()
    )
    return result.chunks_created, result.total_tokens


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest existing RA Exchange transcripts into ChromaDB"
    )
    parser.add_argument(
        "--transcript-dir",
        default="transcripts/ra_exchange",
        help="Directory containing transcripts (default: transcripts/ra_exchange)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Ingestion batch size for progress reporting (default: 5)",
    )
    args = parser.parse_args()

    transcript_dir = Path(args.transcript_dir)
    transcribe_progress_file = transcript_dir / "_progress.json"
    ingest_progress_file = transcript_dir / "_ingest_progress.json"

    if not transcribe_progress_file.exists():
        print(f"Error: No _progress.json found in {transcript_dir}")
        sys.exit(1)

    transcribe_progress = json.loads(transcribe_progress_file.read_text())
    completed = transcribe_progress.get("completed", {})

    # Use separate file for ingestion tracking to avoid race condition
    # with the transcription script
    if ingest_progress_file.exists():
        ingest_progress = json.loads(ingest_progress_file.read_text())
    else:
        ingest_progress = {"ingested": {}}
    ingested = ingest_progress.get("ingested", {})

    # Find transcripts that are completed but not yet ingested
    to_ingest = []
    for url, info in completed.items():
        if url not in ingested:
            file_path = transcript_dir / info["file"]
            if file_path.exists():
                to_ingest.append({
                    "url": url,
                    "title": info["title"],
                    "file_path": str(file_path),
                })

    if not to_ingest:
        print("Nothing to ingest — all completed transcripts are already in the DB.")
        return

    print("=" * 60)
    print("RA EXCHANGE TRANSCRIPT INGESTION")
    print("=" * 60)
    print(f"  Completed transcripts:  {len(completed)}")
    print(f"  Already ingested:       {len(ingested)}")
    print(f"  To ingest now:          {len(to_ingest)}")
    print()

    # Bootstrap ingestion service
    print("Initializing ingestion pipeline...")
    service, emb_name = await build_ingestion_service()
    print(f"  Embedding provider: {emb_name}")
    print()

    total_chunks = 0
    total_tokens = 0
    succeeded = 0
    failed = 0

    for i, item in enumerate(to_ingest, 1):
        print(f"[{i}/{len(to_ingest)}] {item['title']}")

        try:
            chunks, tokens = await ingest_transcript(
                service, item["file_path"], item["title"]
            )
            total_chunks += chunks
            total_tokens += tokens
            succeeded += 1
            print(f"  OK: {chunks} chunks, {tokens} tokens")

            # Update ingestion progress file (separate from transcription)
            ingest_progress.setdefault("ingested", {})[item["url"]] = {
                "chunks": chunks,
                "tokens": tokens,
            }
            ingest_progress_file.write_text(json.dumps(ingest_progress, indent=2))

        except Exception as exc:
            failed += 1
            print(f"  FAIL: {str(exc)[:200]}")

        # Progress summary every batch_size episodes
        if i % args.batch_size == 0:
            print(f"\n  --- Progress: {i}/{len(to_ingest)} | "
                  f"{total_chunks} chunks, {total_tokens} tokens ---\n")

    print()
    print("=" * 60)
    print("INGESTION COMPLETE")
    print(f"  Succeeded:    {succeeded}/{len(to_ingest)}")
    print(f"  Failed:       {failed}/{len(to_ingest)}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Total tokens: {total_tokens}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

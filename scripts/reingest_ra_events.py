#!/usr/bin/env python3
"""Re-ingest RA event files with correct source_type and citation_tier.

Fixes a bug where ra_events_*.txt files were ingested as
source_type="reference" / citation_tier=5 instead of
source_type="event" / citation_tier=3.

Steps:
  1. Find all source_ids in ChromaDB whose source_title matches
     ra_events_*.txt and source_type="reference".
  2. Delete those chunks.
  3. Re-ingest the reference corpus directory — the fixed
     ingest_directory() code will assign the correct metadata.

Usage:
    python scripts/reingest_ra_events.py
"""

import asyncio
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so "src" imports work.
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.config.settings import Settings  # noqa: E402
from src.providers.vector_store.chromadb_provider import ChromaDBProvider  # noqa: E402


async def _find_ra_event_source_ids(vector_store: ChromaDBProvider) -> set[str]:
    """Return source_ids for reference-typed RA event chunks."""
    # ChromaDB doesn't support LIKE queries, so we retrieve all
    # reference-typed chunks in pages and filter in Python.
    collection = vector_store._collection
    page_size = 5000
    offset = 0
    ra_source_ids: set[str] = set()

    while True:
        batch = collection.get(
            where={"source_type": "reference"},
            limit=page_size,
            offset=offset,
            include=["metadatas"],
        )
        if not batch["ids"]:
            break

        for meta in batch["metadatas"]:
            title = meta.get("source_title", "")
            if title.startswith("ra_events_") and title.endswith(".txt"):
                ra_source_ids.add(meta.get("source_id", ""))

        offset += page_size
        # Safety: stop if we've paged through all chunks
        if len(batch["ids"]) < page_size:
            break

    ra_source_ids.discard("")
    return ra_source_ids


async def main() -> int:
    settings = Settings()

    # Build embedding provider (needed for ChromaDB init, not for delete)
    from src.cli.ingest import _build_embedding_provider
    embedding_provider = _build_embedding_provider(settings)
    if embedding_provider is None:
        print("Error: No embedding provider available.", file=sys.stderr)
        return 1

    vector_store = ChromaDBProvider(
        embedding_provider=embedding_provider,
        persist_directory=settings.chromadb_persist_dir,
        collection_name=settings.chromadb_collection,
    )

    if not vector_store.is_available():
        print("Error: ChromaDB not available.", file=sys.stderr)
        return 1

    # --- Step 1: Find RA event source_ids with wrong metadata ---
    print("Step 1: Finding RA event chunks with source_type='reference'...")
    ra_ids = await _find_ra_event_source_ids(vector_store)

    if not ra_ids:
        print("  No RA event chunks found with source_type='reference'.")
        print("  They may already be correctly ingested as 'event'.")
        # Show current stats for verification
        stats = await vector_store.get_stats()
        el_count = stats.sources_by_type.get("event", 0)
        ref_count = stats.sources_by_type.get("reference", 0)
        print(f"  Current counts: event={el_count}, reference={ref_count}")
        return 0

    print(f"  Found {len(ra_ids)} RA event sources to re-ingest.")

    # --- Step 2: Delete old chunks ---
    print("\nStep 2: Deleting old RA event chunks...")
    total_deleted = 0
    for sid in sorted(ra_ids):
        deleted = await vector_store.delete_by_source(sid)
        total_deleted += deleted
        print(f"  Deleted {deleted:>5} chunks for source {sid[:12]}...")

    print(f"  Total deleted: {total_deleted} chunks from {len(ra_ids)} sources.")

    # --- Step 3: Re-ingest the reference corpus directory ---
    print("\nStep 3: Re-ingesting reference corpus with correct metadata...")
    corpus_dir = _project_root / "data" / "reference_corpus"
    if not corpus_dir.is_dir():
        print(f"Error: Corpus directory not found: {corpus_dir}", file=sys.stderr)
        return 1

    # Get remaining existing IDs to skip non-RA-event files
    existing_ids = await vector_store.get_source_ids(source_type="reference")
    existing_ids |= await vector_store.get_source_ids(source_type="event")
    print(f"  Existing source_ids to skip: {len(existing_ids)}")

    from src.services.ingestion.chunker import TextChunker
    from src.services.ingestion.ingestion_service import IngestionService
    from src.services.ingestion.metadata_extractor import MetadataExtractor

    chunker = TextChunker()
    # MetadataExtractor needs an LLM, but skip_tagging=True means it
    # won't be called.  Pass None and let the extractor handle it.
    metadata_extractor = MetadataExtractor(llm=None)
    ingestion_service = IngestionService(
        chunker=chunker,
        metadata_extractor=metadata_extractor,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    start = time.monotonic()
    results = await ingestion_service.ingest_directory(
        str(corpus_dir),
        source_type="reference",
        skip_source_ids=existing_ids if existing_ids else None,
        skip_tagging=True,
    )
    elapsed = time.monotonic() - start

    total_chunks = sum(r.chunks_created for r in results)
    print(f"\n  Re-ingested {len(results)} files, {total_chunks} chunks in {elapsed:.1f}s")

    # --- Verify ---
    print("\nStep 4: Verifying...")
    stats = await vector_store.get_stats()
    print(f"  Total chunks: {stats.total_chunks}")
    print(f"  Total sources: {stats.total_sources}")
    print("  Sources by type:")
    for st, count in sorted(stats.sources_by_type.items()):
        print(f"    {st:<20} {count:>6}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

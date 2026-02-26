# =============================================================================
# src/cli/ingest.py — CLI Ingest Command (RAG Corpus Management)
# =============================================================================
#
# Standalone CLI for managing the raiveFlier RAG (Retrieval-Augmented Generation)
# vector store corpus. The corpus is a ChromaDB collection of text chunks
# from books, articles, web pages, and event listings about electronic music
# history, rave culture, and the underground dance music scene.
#
# During the research phase (Phase 3) of the analysis pipeline, the system
# queries this corpus to provide rich cultural context about artists, venues,
# labels, and events found on a flier.
#
# Supported subcommands:
#
#   book      — Ingest a plain-text (.txt) book file
#   pdf       — Ingest a PDF book or document
#   epub      — Ingest an EPUB ebook
#   article   — Ingest a web article by URL (auto-fetches and parses HTML)
#   directory — Bulk-ingest all files in a directory
#   purge     — Delete all chunks of a given source type (e.g., "interview")
#   stats     — Display corpus statistics (chunk counts, source types)
#
# The ingestion pipeline for each document:
#   1. Read/parse the document (text extraction for PDF/EPUB, HTTP fetch for URLs)
#   2. Chunk the text into embedding-sized windows (~512 tokens)
#   3. Extract metadata tags via LLM (artist names, genres, locations, eras)
#   4. Generate vector embeddings (OpenAI text-embedding-3-small or Nomic)
#   5. Store chunks + embeddings + metadata in ChromaDB
#
# Provider Selection:
#   - Embedding: OpenAI (if key available and no custom base_url) -> Nomic/Ollama
#   - LLM (for metadata tagging): Anthropic -> OpenAI -> Ollama
#   - Vector Store: ChromaDB (always)
#
# Usage examples:
#   python -m src.cli.ingest book --file /path/to/book.txt \
#       --title "Energy Flash" --author "Simon Reynolds" --year 1998
#   python -m src.cli.ingest article --url https://example.com/article
#   python -m src.cli.ingest directory --path /path/to/articles/ --type article
#   python -m src.cli.ingest stats
#   python -m src.cli.ingest purge --type interview --yes
# =============================================================================

"""Standalone CLI for building the raiveFlier RAG vector-store corpus.

Usage::

    python -m src.cli.ingest book --file /path/to/book.txt \\
        --title "Energy Flash" --author "Simon Reynolds" --year 1998

    python -m src.cli.ingest article --url https://example.com/article

    python -m src.cli.ingest directory --path /path/to/articles/ --type article

    python -m src.cli.ingest stats

No extra dependencies beyond the core project requirements.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

# Settings is a Pydantic BaseSettings model that reads configuration from
# environment variables and .env files. It provides all API keys, model
# names, and directory paths needed to construct providers.
from src.config.settings import Settings


def _build_embedding_provider(app_settings: Settings):  # noqa: ANN202
    """Select the first available embedding provider.

    Fallback chain mirrors ``main.py`` to ensure the CLI uses the same
    embedding model as the deployed app (critical for dimension compatibility
    with an existing ChromaDB collection):

    Priority: OpenAI (if real key, not a custom base_url) ->
              FastEmbed (ONNX, 1024-dim, matches production corpus) ->
              SentenceTransformer (PyTorch, 1024-dim) ->
              Nomic/Ollama (768-dim — only works with a fresh collection).

    Imports are deferred inside the function to avoid loading heavy
    dependencies (httpx, openai SDK, fastembed, etc.) unless actually needed.

    Returns
    -------
    IEmbeddingProvider or None
        The first available embedding provider, or ``None`` if none are
        configured.
    """
    # --- Tier 1: OpenAI API (if real OpenAI key, not TogetherAI) ---
    if app_settings.openai_api_key:
        from src.providers.embedding.openai_embedding_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(settings=app_settings)
        if provider.is_available():
            return provider

    # --- Tier 2: FastEmbed (ONNX, no PyTorch, 1024-dim default) ---
    # Matches the production corpus built with intfloat/multilingual-e5-large.
    from src.providers.embedding.fastembed_embedding_provider import (
        FastEmbedEmbeddingProvider,
    )

    fe_provider = FastEmbedEmbeddingProvider()
    if fe_provider.is_available():
        return fe_provider

    # --- Tier 3: SentenceTransformer (PyTorch, 1024-dim default) ---
    from src.providers.embedding.sentence_transformer_embedding_provider import (
        SentenceTransformerEmbeddingProvider,
    )

    st_provider = SentenceTransformerEmbeddingProvider()
    if st_provider.is_available():
        return st_provider

    # --- Tier 4: Nomic via Ollama (768-dim — dimension mismatch warning) ---
    from src.providers.embedding.nomic_embedding_provider import (
        NomicEmbeddingProvider,
    )

    provider = NomicEmbeddingProvider(settings=app_settings)
    if provider.is_available():
        return provider

    return None


def _build_llm_provider(app_settings: Settings):  # noqa: ANN202
    """Select the first available LLM provider for metadata extraction.

    The LLM is used during ingestion to auto-tag chunks with semantic
    metadata (artist names, genres, geographic locations, time periods).
    This metadata enables filtered vector search during the research phase.

    Priority: Anthropic (Claude) -> OpenAI (GPT) -> Ollama (local/free).
    """
    if app_settings.anthropic_api_key:
        from src.providers.llm.anthropic_provider import AnthropicLLMProvider

        return AnthropicLLMProvider(settings=app_settings)
    if app_settings.openai_api_key:
        from src.providers.llm.openai_provider import OpenAILLMProvider

        return OpenAILLMProvider(settings=app_settings)

    from src.providers.llm.ollama_provider import OllamaLLMProvider

    return OllamaLLMProvider(settings=app_settings)


def _build_ingestion_service(app_settings: Settings):  # noqa: ANN202
    """Construct the full ingestion service with all providers.

    Assembles the complete ingestion pipeline by wiring together:
      - TextChunker: Splits documents into ~512-token windows with overlap
      - MetadataExtractor: Uses LLM to auto-tag chunks with semantic metadata
      - EmbeddingProvider: Converts text chunks to vector embeddings
      - ChromaDBProvider: Stores embeddings + metadata in the vector database

    This is also called by scrape_ra.py's ingest command, so it serves as
    the shared factory for the ingestion pipeline.

    Returns
    -------
    tuple[IngestionService, str] or tuple[None, str]
        The ingestion service and a status message.  Returns ``None`` with
        an error message if a required provider is unavailable.
    """
    embedding_provider = _build_embedding_provider(app_settings)
    if embedding_provider is None:
        return None, (
            "No embedding provider available.\n"
            "Set one of:\n"
            "  OPENAI_API_KEY  — for OpenAI text-embedding-3-small\n"
            "  OLLAMA_BASE_URL — for Nomic nomic-embed-text (default: http://localhost:11434)\n"
        )

    from src.providers.vector_store.chromadb_provider import ChromaDBProvider
    from src.services.ingestion.chunker import TextChunker
    from src.services.ingestion.ingestion_service import IngestionService
    from src.services.ingestion.metadata_extractor import MetadataExtractor

    llm_provider = _build_llm_provider(app_settings)

    vector_store = ChromaDBProvider(
        embedding_provider=embedding_provider,
        persist_directory=app_settings.chromadb_persist_dir,
        collection_name=app_settings.chromadb_collection,
    )

    chunker = TextChunker()
    metadata_extractor = MetadataExtractor(llm=llm_provider)

    service = IngestionService(
        chunker=chunker,
        metadata_extractor=metadata_extractor,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    provider_name = embedding_provider.get_provider_name()
    return service, f"Embedding: {provider_name} | Store: chromadb"


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


async def _handle_book(args: argparse.Namespace, service) -> int:  # noqa: ANN001
    """Ingest a plain-text book."""
    print(f"Ingesting book: {args.title} by {args.author} ({args.year})")
    print(f"  File: {args.file}")

    result = await service.ingest_book(
        file_path=args.file,
        title=args.title,
        author=args.author,
        year=args.year,
    )

    print("\nIngestion complete:")
    print(f"  Chunks created: {result.chunks_created}")
    print(f"  Total tokens:   {result.total_tokens}")
    print(f"  Time:           {result.ingestion_time:.2f}s")
    print(f"  Source ID:       {result.source_id}")
    return 0


async def _handle_pdf(args: argparse.Namespace, service) -> int:  # noqa: ANN001
    """Ingest a PDF book or document."""
    print(f"Ingesting PDF: {args.title} by {args.author} ({args.year})")
    print(f"  File: {args.file}")

    result = await service.ingest_pdf(
        file_path=args.file,
        title=args.title,
        author=args.author,
        year=args.year,
    )

    print("\nIngestion complete:")
    print(f"  Chunks created: {result.chunks_created}")
    print(f"  Total tokens:   {result.total_tokens}")
    print(f"  Time:           {result.ingestion_time:.2f}s")
    print(f"  Source ID:       {result.source_id}")
    return 0


async def _handle_epub(args: argparse.Namespace, service) -> int:  # noqa: ANN001
    """Ingest an EPUB book."""
    print(f"Ingesting EPUB: {args.title} by {args.author} ({args.year})")
    print(f"  File: {args.file}")

    result = await service.ingest_epub(
        file_path=args.file,
        title=args.title,
        author=args.author,
        year=args.year,
    )

    print("\nIngestion complete:")
    print(f"  Chunks created: {result.chunks_created}")
    print(f"  Total tokens:   {result.total_tokens}")
    print(f"  Time:           {result.ingestion_time:.2f}s")
    print(f"  Source ID:       {result.source_id}")
    return 0


async def _handle_article(args: argparse.Namespace, service) -> int:  # noqa: ANN001
    """Ingest a web article."""
    print(f"Ingesting article: {args.url}")

    result = await service.ingest_article(url=args.url)

    print("\nIngestion complete:")
    print(f"  Chunks created: {result.chunks_created}")
    print(f"  Total tokens:   {result.total_tokens}")
    print(f"  Time:           {result.ingestion_time:.2f}s")
    print(f"  Source ID:       {result.source_id}")
    return 0


async def _handle_directory(args: argparse.Namespace, service) -> int:  # noqa: ANN001
    """Ingest all files in a directory."""
    source_type = args.type
    skip_tagging = getattr(args, "skip_tagging", False)
    print(f"Ingesting directory: {args.path} (type: {source_type})")
    if skip_tagging:
        print("  Skipping LLM metadata tagging (embeddings-only mode)")

    results = await service.ingest_directory(
        dir_path=args.path,
        source_type=source_type,
        skip_tagging=skip_tagging,
    )

    total_chunks = sum(r.chunks_created for r in results)
    total_tokens = sum(r.total_tokens for r in results)
    total_time = sum(r.ingestion_time for r in results)

    print("\nDirectory ingestion complete:")
    print(f"  Files processed: {len(results)}")
    print(f"  Total chunks:    {total_chunks}")
    print(f"  Total tokens:    {total_tokens}")
    print(f"  Total time:      {total_time:.2f}s")
    return 0


async def _handle_purge(args: argparse.Namespace, app_settings: Settings) -> int:
    """Delete all chunks of a given source type.

    This is a destructive operation — it removes all vector embeddings and
    metadata for chunks matching the specified source_type (e.g., "interview",
    "reference", "event_listing"). Requires confirmation unless --yes is passed.
    """
    source_type = args.type
    print(f"Purging all chunks with source_type='{source_type}'")

    embedding_provider = _build_embedding_provider(app_settings)
    if embedding_provider is None:
        print("Error: No embedding provider available.", file=sys.stderr)
        return 1

    from src.providers.vector_store.chromadb_provider import ChromaDBProvider

    vector_store = ChromaDBProvider(
        embedding_provider=embedding_provider,
        persist_directory=app_settings.chromadb_persist_dir,
        collection_name=app_settings.chromadb_collection,
    )

    # Show current count before deleting
    stats = await vector_store.get_stats()
    type_count = stats.sources_by_type.get(source_type, 0)
    if type_count == 0:
        print(f"No chunks found with source_type='{source_type}'. Nothing to purge.")
        return 0

    print(f"  Found {type_count} chunks with source_type='{source_type}'")

    if not args.yes:
        confirm = input(f"  Delete all {type_count} chunks? [y/N] ").strip().lower()
        if confirm not in ("y", "yes"):
            print("  Aborted.")
            return 0

    deleted = await vector_store.delete_by_source_type(source_type)
    print(f"\n  Deleted {deleted} chunks.")
    return 0


async def _handle_stats(app_settings: Settings) -> int:
    """Display corpus statistics."""
    embedding_provider = _build_embedding_provider(app_settings)
    if embedding_provider is None:
        print("RAG not configured — no embedding provider available.")
        print("Set OPENAI_API_KEY or OLLAMA_BASE_URL to enable RAG.")
        return 0

    from src.providers.vector_store.chromadb_provider import ChromaDBProvider

    vector_store = ChromaDBProvider(
        embedding_provider=embedding_provider,
        persist_directory=app_settings.chromadb_persist_dir,
        collection_name=app_settings.chromadb_collection,
    )

    if not vector_store.is_available():
        print("Vector store not available.")
        return 1

    stats = await vector_store.get_stats()

    print("Corpus Statistics")
    print("=" * 40)
    print(f"  Total chunks:     {stats.total_chunks}")
    print(f"  Total sources:    {stats.total_sources}")
    print(f"  Entity tags:      {stats.entity_tag_count}")
    print(f"  Geographic tags:  {stats.geographic_tag_count}")

    if stats.sources_by_type:
        print("\n  Sources by type:")
        for src_type, count in sorted(stats.sources_by_type.items()):
            print(f"    {src_type:<15} {count}")

    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the ingestion CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.ingest",
        description="Manage the raiveFlier RAG vector-store corpus.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Ingestion commands")

    # -- book --
    book_parser = subparsers.add_parser("book", help="Ingest a plain-text book")
    book_parser.add_argument("--file", required=True, help="Path to the text file")
    book_parser.add_argument("--title", required=True, help="Book title")
    book_parser.add_argument("--author", required=True, help="Book author")
    book_parser.add_argument("--year", required=True, type=int, help="Publication year")

    # -- pdf --
    pdf_parser = subparsers.add_parser("pdf", help="Ingest a PDF book or document")
    pdf_parser.add_argument("--file", required=True, help="Path to the PDF file")
    pdf_parser.add_argument("--title", required=True, help="Book/document title")
    pdf_parser.add_argument("--author", required=True, help="Author name")
    pdf_parser.add_argument("--year", required=True, type=int, help="Publication year")

    # -- epub --
    epub_parser = subparsers.add_parser("epub", help="Ingest an EPUB book")
    epub_parser.add_argument("--file", required=True, help="Path to the EPUB file")
    epub_parser.add_argument("--title", required=True, help="Book title")
    epub_parser.add_argument("--author", required=True, help="Author name")
    epub_parser.add_argument("--year", required=True, type=int, help="Publication year")

    # -- article --
    article_parser = subparsers.add_parser("article", help="Ingest a web article")
    article_parser.add_argument("--url", required=True, help="Article URL")

    # -- directory --
    dir_parser = subparsers.add_parser("directory", help="Ingest all files in a directory")
    dir_parser.add_argument("--path", required=True, help="Directory path")
    dir_parser.add_argument(
        "--type",
        default="article",
        help="Source type label (default: article)",
    )
    dir_parser.add_argument(
        "--skip-tagging",
        action="store_true",
        dest="skip_tagging",
        help="Skip LLM metadata extraction (faster, embeddings-only)",
    )

    # -- purge --
    purge_parser = subparsers.add_parser(
        "purge", help="Delete all chunks of a given source type"
    )
    purge_parser.add_argument(
        "--type", required=True, help="Source type to purge (e.g. interview, reference)"
    )
    purge_parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompt"
    )

    # -- stats --
    subparsers.add_parser("stats", help="Show corpus statistics")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the ingestion tool.

    Parses the subcommand and arguments, initializes the Settings from
    environment variables / .env file, and dispatches to the appropriate
    handler function.

    The "stats" and "purge" commands only need a vector store connection,
    so they initialize fewer dependencies. All other commands require the
    full ingestion service (chunker + metadata extractor + embedding + store).
    """
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Load all configuration from environment variables and .env file.
    app_settings = Settings()

    # Stats and purge are lightweight — they only need the vector store,
    # not the full ingestion pipeline with LLM tagging.
    if args.command == "stats":
        exit_code = asyncio.run(_handle_stats(app_settings))
        sys.exit(exit_code)

    if args.command == "purge":
        exit_code = asyncio.run(_handle_purge(args, app_settings))
        sys.exit(exit_code)

    # All other commands (book, pdf, epub, article, directory) require the
    # full ingestion service with embedding provider, LLM, chunker, and store.
    service, status_msg = _build_ingestion_service(app_settings)
    if service is None:
        print(f"Error: {status_msg}", file=sys.stderr)
        sys.exit(1)

    print(f"Providers: {status_msg}")
    print()

    if args.command == "book":
        exit_code = asyncio.run(_handle_book(args, service))
    elif args.command == "pdf":
        exit_code = asyncio.run(_handle_pdf(args, service))
    elif args.command == "epub":
        exit_code = asyncio.run(_handle_epub(args, service))
    elif args.command == "article":
        exit_code = asyncio.run(_handle_article(args, service))
    elif args.command == "directory":
        exit_code = asyncio.run(_handle_directory(args, service))
    else:
        parser.print_help()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

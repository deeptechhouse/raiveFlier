# =============================================================================
# src/cli/scrape_rbma.py — RBMA (Red Bull Music Academy) Article Scraper CLI
# =============================================================================
#
# CLI tool for discovering, scraping, and ingesting articles and lectures from
# Red Bull Music Academy (RBMA) — one of the most important archives of
# electronic music journalism ever produced (1998-2019). The scraped content
# covers house, techno, drum & bass, jungle, disco, rave culture, DJ history,
# and more, providing rich cultural context for flier analysis.
#
# The RBMA site is JS-rendered, so direct scraping returns minimal content.
# This tool uses the Wayback Machine CDX API for URL discovery and attempts
# the live site first with Wayback archive as fallback for content extraction.
#
# Workflow (4-step process):
#   1. DISCOVER:  Query Wayback CDX API for article + lecture URLs
#   2. SCRAPE:    Fetch articles with live-site + Wayback fallback chain
#   3. GENERATE:  Convert scraped articles into genre-grouped corpus .txt files
#   4. INGEST:    Embed corpus files into ChromaDB (or use startup auto-ingest)
#
# Each step is idempotent and resume-safe:
#   - Discovery saves URL lists to JSON checkpoint files
#   - Scraping checkpoints after each article (resume from interruption)
#   - Corpus generation is stateless (always regenerates from checkpoint data)
#
# Rate Limiting:
#   Default 2-second delay between requests. RBMA is an archived/legacy site;
#   there's no need for aggressive scraping.
#
# Subcommands:
#   discover        — Query CDX API, save URL lists (articles + lectures)
#   scrape          — Fetch articles from discovered URLs (resume-safe)
#   generate-corpus — Generate genre-grouped .txt corpus files
#   ingest          — Generate corpus + ingest into ChromaDB
#   status          — Display discovery/scrape progress
# =============================================================================

"""CLI for scraping RBMA articles and ingesting them into the RAG corpus.

Usage::

    # Step 1: Discover all RBMA article and lecture URLs via Wayback CDX
    python -m src.cli.scrape_rbma discover

    # Step 2: Scrape all genre-relevant articles (resume-safe)
    python -m src.cli.scrape_rbma scrape

    # Step 2 (filtered): Scrape only specific genres
    python -m src.cli.scrape_rbma scrape --genre house,techno

    # Step 3: Generate corpus text files from scraped articles
    python -m src.cli.scrape_rbma generate-corpus

    # Step 4: Generate + ingest into ChromaDB (skips expensive LLM tagging)
    python -m src.cli.scrape_rbma ingest --skip-tagging

    # Check progress at any time
    python -m src.cli.scrape_rbma status

No extra dependencies beyond the core project requirements.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# HTTP client factory
# ---------------------------------------------------------------------------

def _make_http_client() -> httpx.AsyncClient:
    """Create an httpx.AsyncClient with reasonable defaults for scraping.

    Uses a browser-like User-Agent and follows redirects (some RBMA URLs
    redirect between http/https or www/non-www variants).
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(15.0),
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; raiveFlier/0.1; "
                "+https://github.com/raiveFlier)"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
        follow_redirects=True,
    )


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


async def _handle_discover(args: argparse.Namespace) -> int:
    """Run Wayback CDX discovery for RBMA article and lecture URLs.

    Queries the CDX API twice — once for daily.redbullmusicacademy.com/*
    (articles) and once for www.redbullmusicacademy.com/lectures/* (lectures).
    Results are filtered to valid article/lecture URL patterns and saved
    to checkpoint JSON files in data/rbma_scrape/.
    """
    from src.services.rbma_scrape_service import RBMAScrapeService

    print("Discovering RBMA article and lecture URLs via Wayback CDX API...")
    print()

    async with _make_http_client() as client:
        service = RBMAScrapeService(http_client=client)
        result = await service.discover_urls()

    article_count = len(result.get("articles", []))
    lecture_count = len(result.get("lectures", []))

    print(f"  Articles discovered: {article_count:,}")
    print(f"  Lectures discovered: {lecture_count:,}")
    print(f"  Total:               {article_count + lecture_count:,}")
    print()
    print("URL lists saved to data/rbma_scrape/")
    print("Next: run 'python -m src.cli.scrape_rbma scrape' to fetch articles")
    return 0


async def _handle_scrape(args: argparse.Namespace) -> int:
    """Scrape articles from discovered RBMA URLs.

    Loads the discovered URL lists, applies genre filtering (Pass 1 — URL
    slug keyword matching), then fetches each URL using the live-site-first,
    Wayback-fallback strategy. Checkpoints after each article for resume.

    The --genre flag allows filtering to specific genre keywords (comma-
    separated). Without --genre, all genre-relevant articles are scraped.
    """
    from src.services.rbma_scrape_service import (
        GENRE_KEYWORDS,
        RBMAScrapeService,
    )

    delay = getattr(args, "delay", 2.0)
    genre_filter: set[str] | None = None

    # Parse optional genre filter
    if args.genre:
        genre_filter = {g.strip().lower() for g in args.genre.split(",")}
        invalid = genre_filter - GENRE_KEYWORDS
        if invalid:
            print(f"Warning: Unknown genre keywords: {', '.join(sorted(invalid))}")
            genre_filter -= invalid
        if not genre_filter:
            print("Error: No valid genre keywords specified.", file=sys.stderr)
            return 1
        print(f"Genre filter: {', '.join(sorted(genre_filter))}")

    async with _make_http_client() as client:
        service = RBMAScrapeService(
            http_client=client, scrape_delay=delay
        )

        # Load discovered URLs from checkpoint
        article_urls = service._load_discovered_urls("articles")
        lecture_urls = service._load_discovered_urls("lectures")

        if not article_urls and not lecture_urls:
            print(
                "No discovered URLs found. Run 'discover' first.",
                file=sys.stderr,
            )
            return 1

        # Apply genre filtering (Pass 1) to articles
        filtered_articles = service.filter_by_genre(
            article_urls, genres=genre_filter
        )
        # Lectures are always genre-relevant (primary-source artist perspectives)
        filtered_lectures = [
            {**entry, "matched_genres": []} for entry in lecture_urls
        ]

        all_urls = filtered_articles + filtered_lectures
        total = len(all_urls)

        print(f"Scraping {total:,} URLs "
              f"({len(filtered_articles):,} articles + {len(filtered_lectures):,} lectures)")
        print(f"Rate limit: {delay}s between requests")
        print()

        # Progress callback
        def on_progress(current: int, total_count: int, url: str) -> None:
            pct = (current / total_count * 100) if total_count > 0 else 0
            # Truncate URL for display
            short_url = url[:70] + "..." if len(url) > 73 else url
            print(f"  [{current:,}/{total_count:,}] ({pct:.0f}%) {short_url}")

        articles = await service.scrape_articles(
            urls=all_urls, on_progress=on_progress
        )

    print()
    print(f"Scrape complete: {len(articles):,} articles extracted")
    print("Next: run 'python -m src.cli.scrape_rbma generate-corpus'")
    return 0


async def _handle_generate_corpus(args: argparse.Namespace) -> int:
    """Generate genre-grouped corpus .txt files from scraped articles.

    Reads all scraped articles from the checkpoint JSON and groups them
    by genre into separate .txt files in data/reference_corpus/. Each
    file follows the standard corpus format (header + article blocks).
    """
    from src.services.rbma_scrape_service import RBMAScrapeService

    print("Generating RBMA corpus files...")
    print()

    async with _make_http_client() as client:
        service = RBMAScrapeService(http_client=client)
        generated = service.generate_corpus_files()

    if not generated:
        print("No articles to generate corpus from. Scrape first.")
        return 1

    for path in generated:
        # Read file to count articles
        content = path.read_text(encoding="utf-8")
        article_count = content.count("\n---\n")
        size_kb = path.stat().st_size / 1024
        print(f"  {path.name:<30} {article_count:>5} articles  {size_kb:>8.1f} KB")

    print()
    print(f"Generated {len(generated)} corpus files in data/reference_corpus/")
    print("Next: run 'python -m src.cli.scrape_rbma ingest --skip-tagging'")
    return 0


async def _handle_ingest(args: argparse.Namespace) -> int:
    """Generate corpus files and ingest them into ChromaDB.

    This is the full pipeline endpoint: generates corpus .txt files from
    scraped articles, then ingests them into the ChromaDB vector store.
    Uses the shared _build_ingestion_service factory from ingest.py.

    --skip-tagging bypasses LLM metadata extraction (recommended for
    large corpora — saves API cost, and semantic search via embeddings
    is sufficient for reference material).
    """
    from src.cli.ingest import _build_ingestion_service
    from src.config.settings import Settings
    from src.services.rbma_scrape_service import RBMAScrapeService

    # First generate corpus files
    print("Step 1: Generating corpus files...")
    async with _make_http_client() as client:
        service = RBMAScrapeService(http_client=client)
        generated = service.generate_corpus_files()

    if not generated:
        print("No articles to ingest. Scrape first.", file=sys.stderr)
        return 1

    for path in generated:
        print(f"  Generated: {path.name}")

    # Build ingestion service
    print()
    print("Step 2: Ingesting into ChromaDB...")
    app_settings = Settings()
    ingestion_service, status_msg = _build_ingestion_service(app_settings)
    if ingestion_service is None:
        print(f"Error: {status_msg}", file=sys.stderr)
        return 1

    print(f"Providers: {status_msg}")

    skip_tagging = getattr(args, "skip_tagging", False)

    # Ingest ONLY the new RBMA files — not the entire corpus directory.
    # The ingest_directory method reads ALL files before checking skip_source_ids,
    # which causes OOM when the directory contains large RA event files (20MB+).
    # Solution: create a temp directory with symlinks to only the RBMA files,
    # then ingest that isolated directory one file at a time to limit memory.
    import tempfile

    total_chunks = 0
    total_files = 0

    for path in generated:
        print(f"  Ingesting: {path.name} ...", end=" ", flush=True)

        # Create a temp dir with a single symlink to process one file at a time.
        # This prevents ingest_directory from reading the entire corpus dir.
        with tempfile.TemporaryDirectory(prefix="rbma_ingest_") as tmpdir:
            link = Path(tmpdir) / path.name
            link.symlink_to(path.resolve())

            results = await ingestion_service.ingest_directory(
                tmpdir,
                source_type="reference",
                skip_tagging=skip_tagging,
            )
            chunks = sum(r.chunks_created for r in results)
            total_chunks += chunks
            total_files += len(results)
            print(f"{chunks} chunks")

    print(f"\nIngestion complete: {total_chunks:,} chunks from {total_files} files")
    return 0


def _handle_status() -> int:
    """Show discovery and scrape progress for RBMA.

    Reads checkpoint files from disk — no network requests, no async needed.
    Displays a summary table of discovery counts, scrape progress, and
    corpus file status.
    """
    from src.services.rbma_scrape_service import RBMAScrapeService

    # Build service without real HTTP client — only checkpoint I/O is used
    service = RBMAScrapeService(http_client=httpx.AsyncClient())
    status = service.get_scrape_status()

    print("RBMA Scrape Status")
    print("=" * 60)
    print()
    print("Discovery:")
    print(f"  Article URLs:  {status['discovered_articles']:,}")
    print(f"  Lecture URLs:  {status['discovered_lectures']:,}")
    print()
    print("Scraping:")
    print(f"  Articles scraped: {status['scraped_articles']:,}")
    progress = status.get("scrape_progress", {})
    if progress:
        last_idx = progress.get("last_index", 0)
        total_urls = progress.get("total_urls", 0)
        pct = (last_idx / total_urls * 100) if total_urls > 0 else 0
        print(f"  Progress:         {last_idx:,}/{total_urls:,} ({pct:.0f}%)")
        updated = progress.get("updated_at", "")
        if updated:
            print(f"  Last updated:     {updated}")
    print()
    print("Corpus Files:")
    corpus_files = status.get("corpus_files", [])
    if corpus_files:
        for name in corpus_files:
            print(f"  - {name}")
    else:
        print("  (none generated yet)")

    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the RBMA scrape CLI.

    Subcommands mirror the 4-step workflow: discover → scrape →
    generate-corpus → ingest, plus a status command.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.scrape_rbma",
        description=(
            "Scrape Red Bull Music Academy articles and lectures, "
            "and ingest into the raiveFlier RAG corpus."
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command", help="RBMA scrape commands"
    )

    # -- discover --
    subparsers.add_parser(
        "discover",
        help="Discover RBMA article and lecture URLs via Wayback CDX API",
    )

    # -- scrape --
    scrape_parser = subparsers.add_parser(
        "scrape",
        help="Scrape articles from discovered URLs (resume-safe)",
    )
    scrape_parser.add_argument(
        "--genre",
        default=None,
        help="Comma-separated genre keywords to filter (e.g. 'house,techno')",
    )
    scrape_parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds between requests (default: 2.0)",
    )

    # -- generate-corpus --
    subparsers.add_parser(
        "generate-corpus",
        help="Generate genre-grouped .txt corpus files from scraped articles",
    )

    # -- ingest --
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Generate corpus files and ingest into ChromaDB",
    )
    ingest_parser.add_argument(
        "--skip-tagging",
        action="store_true",
        dest="skip_tagging",
        help="Skip LLM metadata tagging (recommended for large corpora)",
    )

    # -- status --
    subparsers.add_parser(
        "status",
        help="Show discovery and scrape progress",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the RBMA scrape tool.

    Dispatches to the appropriate subcommand handler. The "status" command
    is synchronous (reads local checkpoint files only). All other commands
    are async because they make HTTP requests or interact with the vector store.
    """
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Status is synchronous — reads local files only
    if args.command == "status":
        exit_code = _handle_status()
        sys.exit(exit_code)

    # All other commands are async
    if args.command == "discover":
        exit_code = asyncio.run(_handle_discover(args))
    elif args.command == "scrape":
        exit_code = asyncio.run(_handle_scrape(args))
    elif args.command == "generate-corpus":
        exit_code = asyncio.run(_handle_generate_corpus(args))
    elif args.command == "ingest":
        exit_code = asyncio.run(_handle_ingest(args))
    else:
        parser.print_help()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

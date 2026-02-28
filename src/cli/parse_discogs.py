# =============================================================================
# src/cli/parse_discogs.py — Discogs Data Dump Parser & Corpus Generator CLI
# =============================================================================
#
# CLI tool for parsing Discogs monthly XML data dumps (~90+ GB uncompressed),
# extracting electronic music data (artists, labels, master releases) for
# house, techno, electro, and drum & bass, and converting it into prose-format
# text files suitable for ingestion into the raiveFlier RAG corpus.
#
# This is a LOCAL tool — it runs on your development machine (not on the
# 512 MB Render instance) because:
#   1. The XML dumps are 14-16 GB compressed / 90+ GB uncompressed
#   2. Streaming parse requires ~100-500 MB RAM (fine locally, tight on Render)
#   3. Embedding 50K-80K chunks takes 1-3 hours of compute
#
# After generating and ingesting the corpus locally, rsync the ChromaDB
# directory to the Render persistent disk for deployment.
#
# Workflow (recommended order):
#   1. Download dumps from data.discogs.com (releases, artists, labels, masters)
#   2. build-style-maps  — Scan releases to map artists/labels to styles (~30-60 min)
#   3. parse-artists     — Extract electronic music artists (~10-20 min)
#   4. parse-labels      — Extract electronic music labels (~5-10 min)
#   5. parse-masters     — Extract electronic music master releases (~20-40 min)
#   6. generate-corpus   — Convert checkpoints to prose text files (~1 min)
#   7. ingest            — Embed and store in ChromaDB (~1-3 hours)
#
# Steps 2-5 produce JSONL checkpoint files (idempotent — re-running overwrites).
# Step 6 reads checkpoints and writes .txt files to data/discogs_corpus/.
# Step 7 uses the existing ingest_directory() pipeline.
#
# Subcommands:
#   download            — Print download instructions and URLs
#   build-style-maps    — Scan releases XML to build artist/label → style maps
#   parse-artists       — Parse artists XML using style maps for filtering
#   parse-labels        — Parse labels XML using style maps for filtering
#   parse-masters       — Parse masters XML (self-filtering by genre/style)
#   parse-all           — Run build-style-maps + all three parsers sequentially
#   generate-corpus     — Convert JSONL checkpoints → prose .txt corpus files
#   ingest              — Generate corpus + embed + store in ChromaDB
#   status              — Show parsing progress and corpus file stats
# =============================================================================

"""CLI for parsing Discogs data dumps and building the RAG corpus.

Usage::

    # Show download instructions for Discogs data dumps
    python -m src.cli.parse_discogs download

    # Build style maps from releases (required before artist/label parsing)
    python -m src.cli.parse_discogs build-style-maps \\
        --releases-xml /path/to/discogs_releases.xml.gz

    # Parse all three data types using pre-built style maps
    python -m src.cli.parse_discogs parse-all \\
        --releases-xml /path/to/discogs_releases.xml.gz \\
        --artists-xml /path/to/discogs_artists.xml.gz \\
        --labels-xml /path/to/discogs_labels.xml.gz \\
        --masters-xml /path/to/discogs_masters.xml.gz

    # Generate prose corpus files from parsed checkpoints
    python -m src.cli.parse_discogs generate-corpus

    # Generate + ingest into ChromaDB (skip LLM tagging for speed)
    python -m src.cli.parse_discogs ingest --skip-tagging

    # Check progress
    python -m src.cli.parse_discogs status

No extra dependencies beyond the core project requirements.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

# ─── DISCOGS DATA DUMP DOWNLOAD URLs ─────────────────────────────────────────
# Discogs publishes monthly dumps on the first of each month.
# URL format: https://discogs-data-dumps.s3.us-west-2.amazonaws.com/data/YYYY/discogs_YYYYMMDD_TYPE.xml.gz
# The latest dump date must be filled in by the user.

_DOWNLOAD_INSTRUCTIONS = """
Discogs Data Dump Download Instructions
========================================

1. Visit https://www.discogs.com/data/ and accept the data usage terms.

2. Download the four XML dump files (use the most recent month available):

   Artists:  discogs_YYYYMMDD_artists.xml.gz   (~1.5-2 GB)
   Labels:   discogs_YYYYMMDD_labels.xml.gz    (~500 MB-1 GB)
   Masters:  discogs_YYYYMMDD_masters.xml.gz   (~2-3 GB)
   Releases: discogs_YYYYMMDD_releases.xml.gz  (~14-16 GB)

   Replace YYYYMMDD with the actual dump date (e.g., 20260201).

3. Save them to a local directory (e.g., ~/discogs_dumps/).
   Do NOT decompress them — the parser reads .gz files directly.

4. Run the parser in order:

   # Step 1: Build style maps (~30-60 min, scans the 14 GB releases file)
   python -m src.cli.parse_discogs build-style-maps \\
       --releases-xml ~/discogs_dumps/discogs_YYYYMMDD_releases.xml.gz

   # Step 2: Parse all three data types (uses pre-built style maps)
   python -m src.cli.parse_discogs parse-all \\
       --releases-xml ~/discogs_dumps/discogs_YYYYMMDD_releases.xml.gz \\
       --artists-xml ~/discogs_dumps/discogs_YYYYMMDD_artists.xml.gz \\
       --labels-xml ~/discogs_dumps/discogs_YYYYMMDD_labels.xml.gz \\
       --masters-xml ~/discogs_dumps/discogs_YYYYMMDD_masters.xml.gz

   # Step 3: Generate prose corpus files (~1 min)
   python -m src.cli.parse_discogs generate-corpus

   # Step 4: Ingest into ChromaDB (~1-3 hours)
   python -m src.cli.parse_discogs ingest --skip-tagging

   # Step 5: Check results
   python -m src.cli.parse_discogs status

Total disk space needed: ~20 GB for compressed dumps + ~200-400 MB for corpus.
"""


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _handle_download() -> int:
    """Print download instructions for Discogs data dumps."""
    print(_DOWNLOAD_INSTRUCTIONS)
    return 0


def _handle_build_style_maps(args: argparse.Namespace) -> int:
    """Scan the releases XML to build artist/label → style cross-reference maps.

    This is the longest-running step (~30-60 min) because it streams through
    the entire releases dump (~90+ GB uncompressed). The resulting maps are
    saved as JSON checkpoints so subsequent parse-artists and parse-labels
    commands can use them without re-scanning.
    """
    from src.services.discogs_dump_service import DiscogsDumpParser

    parser = DiscogsDumpParser(output_dir=args.output_dir)

    print(f"Building style maps from: {args.releases_xml}")
    print("This scans the full releases dump (~19M records). Expect 30-60 minutes.")
    print()

    start = time.monotonic()
    artist_map, label_map = parser.build_style_maps(args.releases_xml)
    elapsed = time.monotonic() - start

    print(f"\nStyle maps built in {elapsed:.0f}s:")
    print(f"  Unique artists:  {len(artist_map):,}")
    print(f"  Unique labels:   {len(label_map):,}")
    print(f"  Saved to: {args.output_dir}/checkpoints/")
    return 0


def _handle_parse_artists(args: argparse.Namespace) -> int:
    """Parse the artists XML dump using pre-built style maps for filtering."""
    from src.services.discogs_dump_service import DiscogsDumpParser

    parser = DiscogsDumpParser(output_dir=args.output_dir)

    # Load style maps from checkpoint.
    maps = parser.load_style_maps()
    if maps is None:
        print(
            "Error: Style maps not found. Run build-style-maps first.",
            file=sys.stderr,
        )
        return 1

    artist_map, _ = maps
    print(f"Loaded artist style map ({len(artist_map):,} entries)")
    print(f"Parsing artists from: {args.artists_xml}")
    print()

    count = parser.parse_artists(args.artists_xml, artist_style_map=artist_map)
    print(f"\nExtracted {count:,} artists to checkpoint.")
    return 0


def _handle_parse_labels(args: argparse.Namespace) -> int:
    """Parse the labels XML dump using pre-built style maps for filtering."""
    from src.services.discogs_dump_service import DiscogsDumpParser

    parser = DiscogsDumpParser(output_dir=args.output_dir)

    maps = parser.load_style_maps()
    if maps is None:
        print(
            "Error: Style maps not found. Run build-style-maps first.",
            file=sys.stderr,
        )
        return 1

    _, label_map = maps
    print(f"Loaded label style map ({len(label_map):,} entries)")
    print(f"Parsing labels from: {args.labels_xml}")
    print()

    count = parser.parse_labels(args.labels_xml, label_style_map=label_map)
    print(f"\nExtracted {count:,} labels to checkpoint.")
    return 0


def _handle_parse_masters(args: argparse.Namespace) -> int:
    """Parse the masters XML dump, filtering by Electronic genre + target styles."""
    from src.services.discogs_dump_service import DiscogsDumpParser

    parser = DiscogsDumpParser(output_dir=args.output_dir)

    print(f"Parsing masters from: {args.masters_xml}")
    print("Filtering for Electronic genre with target styles.")
    print()

    count = parser.parse_masters(args.masters_xml)
    print(f"\nExtracted {count:,} master releases to checkpoint.")
    return 0


def _handle_parse_all(args: argparse.Namespace) -> int:
    """Run the full parse pipeline: build-style-maps → parse all three data types.

    This is the convenience command that runs everything in the correct order.
    Takes ~60-90 minutes total for the full Discogs database.
    """
    from src.services.discogs_dump_service import DiscogsDumpParser

    parser = DiscogsDumpParser(output_dir=args.output_dir)
    overall_start = time.monotonic()

    # Step 1: Build style maps (or reuse existing ones).
    maps = parser.load_style_maps()
    if maps is not None and not args.force_rebuild:
        artist_map, label_map = maps
        print(f"Reusing existing style maps (artists: {len(artist_map):,}, labels: {len(label_map):,})")
        print("  Use --force-rebuild to regenerate from releases XML.")
        print()
    else:
        if not args.releases_xml:
            print(
                "Error: --releases-xml is required for building style maps.",
                file=sys.stderr,
            )
            return 1
        print("Step 1/4: Building style maps from releases XML...")
        artist_map, label_map = parser.build_style_maps(args.releases_xml)
        print(f"  Artists: {len(artist_map):,} | Labels: {len(label_map):,}")
        print()

    # Step 2: Parse artists.
    if not args.artists_xml:
        print("Skipping artists (no --artists-xml provided)")
    else:
        print("Step 2/4: Parsing artists XML...")
        count = parser.parse_artists(args.artists_xml, artist_style_map=artist_map)
        print(f"  Extracted {count:,} artists")
    print()

    # Step 3: Parse labels.
    if not args.labels_xml:
        print("Skipping labels (no --labels-xml provided)")
    else:
        print("Step 3/4: Parsing labels XML...")
        count = parser.parse_labels(args.labels_xml, label_style_map=label_map)
        print(f"  Extracted {count:,} labels")
    print()

    # Step 4: Parse masters.
    if not args.masters_xml:
        print("Skipping masters (no --masters-xml provided)")
    else:
        print("Step 4/4: Parsing masters XML...")
        count = parser.parse_masters(args.masters_xml)
        print(f"  Extracted {count:,} master releases")
    print()

    elapsed = time.monotonic() - overall_start
    print(f"All parsing complete in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"Checkpoints saved to: {args.output_dir}/checkpoints/")
    print()
    print("Next step: python -m src.cli.parse_discogs generate-corpus")
    return 0


def _handle_generate_corpus(args: argparse.Namespace) -> int:
    """Convert JSONL checkpoints into prose-format corpus text files."""
    from src.services.discogs_dump_service import DiscogsDumpParser

    parser = DiscogsDumpParser(output_dir=args.output_dir)

    print("Generating prose corpus files from checkpoints...")
    print()

    results = parser.generate_corpus()

    if not results:
        print("No checkpoint files found. Run parse-all or individual parse commands first.")
        return 1

    print("Corpus files generated:")
    total_entries = 0
    for filename, count in sorted(results.items()):
        print(f"  {filename:<30} {count:>8,} entries")
        total_entries += count

    print(f"\n  {'Total':<30} {total_entries:>8,} entries")
    print(f"\nCorpus directory: {args.output_dir}/")
    print()
    print("Next step: python -m src.cli.parse_discogs ingest --skip-tagging")
    return 0


def _handle_ingest(args: argparse.Namespace) -> int:
    """Generate corpus files and ingest them into ChromaDB.

    Runs generate_corpus() first (idempotent), then uses the existing
    IngestionService.ingest_directory() to chunk, embed, and store
    the prose text files in ChromaDB.
    """
    from src.services.discogs_dump_service import DiscogsDumpParser

    parser = DiscogsDumpParser(output_dir=args.output_dir)

    # Step 1: Generate corpus (fast, idempotent).
    if not args.skip_generate:
        print("Step 1: Generating prose corpus files...")
        results = parser.generate_corpus()
        total = sum(results.values())
        print(f"  Generated {len(results)} files with {total:,} entries")
        print()
    else:
        print("Skipping corpus generation (--skip-generate)")
        print()

    # Step 2: Ingest into ChromaDB via existing pipeline.
    print("Step 2: Ingesting corpus into ChromaDB...")

    # Deferred import to avoid loading heavy ML dependencies until needed.
    from src.config.settings import Settings
    from src.cli.ingest import _build_ingestion_service

    app_settings = Settings()
    service, status_msg = _build_ingestion_service(app_settings)
    if service is None:
        print(f"Error: {status_msg}", file=sys.stderr)
        return 1

    print(f"  Providers: {status_msg}")
    skip_tagging = args.skip_tagging

    if skip_tagging:
        print("  LLM metadata tagging: SKIPPED (embeddings-only mode)")
    print()

    # Use ingest_directory on the corpus output directory.
    # Source type "discogs" distinguishes these chunks from other corpus
    # sources (books, articles, RA events, RBMA content) in the vector store.
    ingest_results = asyncio.run(
        service.ingest_directory(
            dir_path=args.output_dir,
            source_type="discogs",
            skip_tagging=skip_tagging,
        )
    )

    total_chunks = sum(r.chunks_created for r in ingest_results)
    total_tokens = sum(r.total_tokens for r in ingest_results)
    total_time = sum(r.ingestion_time for r in ingest_results)

    print("\nIngestion complete:")
    print(f"  Files processed: {len(ingest_results)}")
    print(f"  Total chunks:    {total_chunks:,}")
    print(f"  Total tokens:    {total_tokens:,}")
    print(f"  Total time:      {total_time:.1f}s ({total_time / 60:.1f} min)")
    return 0


def _handle_status(args: argparse.Namespace) -> int:
    """Display parsing progress and corpus file statistics."""
    from src.services.discogs_dump_service import DiscogsDumpParser

    parser = DiscogsDumpParser(output_dir=args.output_dir)
    status = parser.get_status()

    print("Discogs Dump Parser Status")
    print("=" * 50)
    print(f"  Output dir:     {status['output_dir']}")
    print(f"  Checkpoint dir: {status['checkpoint_dir']}")
    print()

    # Style maps.
    print("Style Maps (from releases scan):")
    for name in ("artist_style_map", "label_style_map"):
        info = status.get(name)
        if info:
            print(f"  {name}: {info['entries']:,} entries")
        else:
            print(f"  {name}: not built yet")
    print()

    # Checkpoints.
    print("JSONL Checkpoints:")
    for name in ("artists", "labels", "masters"):
        key = f"{name}_checkpoint"
        info = status.get(key)
        if info:
            print(f"  {name}: {info['records']:,} records ({info['size_mb']:.1f} MB)")
        else:
            print(f"  {name}: not parsed yet")
    print()

    # Corpus files.
    corpus_files = status.get("corpus_files", {})
    if corpus_files:
        print("Corpus Text Files:")
        total_size = 0.0
        total_lines = 0
        for filename, info in corpus_files.items():
            print(f"  {filename:<30} {info['lines']:>8,} lines  ({info['size_mb']:.1f} MB)")
            total_size += info["size_mb"]
            total_lines += info["lines"]
        print(f"  {'─' * 30} {'─' * 8}       {'─' * 8}")
        print(f"  {'Total':<30} {total_lines:>8,} lines  ({total_size:.1f} MB)")
    else:
        print("Corpus Text Files: none generated yet")

    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the Discogs dump CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.parse_discogs",
        description=(
            "Parse Discogs monthly XML data dumps and build prose corpus "
            "files for the raiveFlier RAG system."
        ),
    )
    # Global option: output directory for all generated files.
    parser.add_argument(
        "--output-dir",
        default="./data/discogs_corpus",
        help="Output directory for checkpoints and corpus files (default: ./data/discogs_corpus)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Parser commands")

    # -- download --
    subparsers.add_parser(
        "download",
        help="Print download instructions for Discogs data dumps",
    )

    # -- build-style-maps --
    bsm_parser = subparsers.add_parser(
        "build-style-maps",
        help="Scan releases XML to build artist/label → style cross-reference maps",
    )
    bsm_parser.add_argument(
        "--releases-xml",
        required=True,
        help="Path to discogs_*_releases.xml or .xml.gz",
    )

    # -- parse-artists --
    pa_parser = subparsers.add_parser(
        "parse-artists",
        help="Parse artists XML using pre-built style maps",
    )
    pa_parser.add_argument(
        "--artists-xml",
        required=True,
        help="Path to discogs_*_artists.xml or .xml.gz",
    )

    # -- parse-labels --
    pl_parser = subparsers.add_parser(
        "parse-labels",
        help="Parse labels XML using pre-built style maps",
    )
    pl_parser.add_argument(
        "--labels-xml",
        required=True,
        help="Path to discogs_*_labels.xml or .xml.gz",
    )

    # -- parse-masters --
    pm_parser = subparsers.add_parser(
        "parse-masters",
        help="Parse masters XML (filters by Electronic genre + target styles)",
    )
    pm_parser.add_argument(
        "--masters-xml",
        required=True,
        help="Path to discogs_*_masters.xml or .xml.gz",
    )

    # -- parse-all --
    pa_all = subparsers.add_parser(
        "parse-all",
        help="Run full parse pipeline: build style maps + parse artists/labels/masters",
    )
    pa_all.add_argument(
        "--releases-xml",
        help="Path to releases XML (required for style maps if not already built)",
    )
    pa_all.add_argument(
        "--artists-xml",
        help="Path to artists XML (skipped if not provided)",
    )
    pa_all.add_argument(
        "--labels-xml",
        help="Path to labels XML (skipped if not provided)",
    )
    pa_all.add_argument(
        "--masters-xml",
        help="Path to masters XML (skipped if not provided)",
    )
    pa_all.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild style maps even if checkpoints exist",
    )

    # -- generate-corpus --
    subparsers.add_parser(
        "generate-corpus",
        help="Convert JSONL checkpoints → prose .txt corpus files",
    )

    # -- ingest --
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Generate corpus + embed + store in ChromaDB",
    )
    ingest_parser.add_argument(
        "--skip-tagging",
        action="store_true",
        dest="skip_tagging",
        help="Skip LLM metadata extraction (faster, embeddings-only)",
    )
    ingest_parser.add_argument(
        "--skip-generate",
        action="store_true",
        dest="skip_generate",
        help="Skip corpus generation step (use existing .txt files)",
    )

    # -- status --
    subparsers.add_parser(
        "status",
        help="Show parsing progress and corpus file statistics",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for the Discogs dump parser.

    Parses the subcommand and arguments, then dispatches to the appropriate
    handler. All handlers construct their own DiscogsDumpParser instance
    rather than sharing one, since this is a one-shot CLI tool.
    """
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Dispatch to the appropriate handler.
    if args.command == "download":
        exit_code = _handle_download()
    elif args.command == "build-style-maps":
        exit_code = _handle_build_style_maps(args)
    elif args.command == "parse-artists":
        exit_code = _handle_parse_artists(args)
    elif args.command == "parse-labels":
        exit_code = _handle_parse_labels(args)
    elif args.command == "parse-masters":
        exit_code = _handle_parse_masters(args)
    elif args.command == "parse-all":
        exit_code = _handle_parse_all(args)
    elif args.command == "generate-corpus":
        exit_code = _handle_generate_corpus(args)
    elif args.command == "ingest":
        exit_code = _handle_ingest(args)
    elif args.command == "status":
        exit_code = _handle_status(args)
    else:
        parser.print_help()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

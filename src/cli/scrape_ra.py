# =============================================================================
# src/cli/scrape_ra.py — RA.co (Resident Advisor) Event Scraper CLI
# =============================================================================
#
# CLI tool for scraping electronic music event listings from Resident Advisor
# (RA.co), the world's largest electronic music platform. Scraped events are
# converted into corpus text files and ingested into the ChromaDB vector store
# to provide rich event-history context during flier analysis.
#
# The scraper targets 17 global cities (Berlin, London, New York, Detroit,
# Chicago, Tokyo, etc.) and covers events from 2016 to present. Data is
# fetched via RA's GraphQL API, which requires specific area IDs for each city.
#
# Workflow (3-step process):
#   1. SCRAPE:  Fetch raw event data from RA GraphQL API -> JSON checkpoints
#   2. GENERATE: Convert JSON checkpoints -> plain text corpus files
#   3. INGEST:  Chunk, embed, and store corpus files -> ChromaDB
#
# Each step is idempotent and resumable:
#   - Scraping saves per-city progress checkpoints (year/month completed)
#   - Re-running skips already-scraped months
#   - Ingestion can use --skip-tagging to bypass expensive LLM metadata
#     extraction and use pre-extracted tags from the event data instead
#
# Rate Limiting:
#   The scraper enforces a configurable delay between API requests (default:
#   4 seconds) to avoid being blocked by RA.co. This is critical — aggressive
#   scraping will result in IP bans.
#
# Subcommands:
#   scrape          — Fetch events from RA.co for one or all cities
#   generate-corpus — Convert scraped JSON to plain text corpus files
#   ingest          — Generate corpus + ingest into ChromaDB
#   verify-ids      — Test-query each city's area ID to confirm validity
#   status          — Display scrape progress for all cities
# =============================================================================

"""CLI for scraping RA.co events and ingesting them into the RAG corpus.

Usage::

    # Scrape a single city (resumes from checkpoint)
    python -m src.cli.scrape_ra scrape --city chicago

    # Scrape all 17 target cities (2016-present)
    python -m src.cli.scrape_ra scrape --all --start-year 2016

    # Generate corpus text files from scraped JSON
    python -m src.cli.scrape_ra generate-corpus --city chicago
    python -m src.cli.scrape_ra generate-corpus --all

    # Ingest corpus files into ChromaDB (full pipeline)
    python -m src.cli.scrape_ra ingest --city chicago --skip-tagging

    # Verify area IDs with test queries
    python -m src.cli.scrape_ra verify-ids

    # Show scrape progress for all cities
    python -m src.cli.scrape_ra status

No extra dependencies beyond the core project requirements.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import httpx

# RA_AREA_IDS maps city keys (e.g., "chicago") to RA's internal area IDs
# used in their GraphQL API. CITY_DISPLAY_NAMES maps keys to human-readable
# names (e.g., "chicago" -> "Chicago").
from src.providers.event.ra_graphql_provider import (
    CITY_DISPLAY_NAMES,
    RA_AREA_IDS,
)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


async def _handle_scrape(args: argparse.Namespace) -> int:
    """Scrape events from RA.co for one or all cities.

    This handler fetches event listings from RA's GraphQL API month-by-month,
    saving progress checkpoints after each month. If interrupted, re-running
    with the same city will resume from the last completed month.

    The scraper uses httpx.AsyncClient for HTTP requests with a configurable
    delay between requests to respect RA's rate limits.
    """
    from src.providers.event.ra_graphql_provider import RAGraphQLProvider
    from src.services.ra_scrape_service import RAScrapeService

    # Default delay between API requests (seconds). Higher = safer from IP bans.
    delay = getattr(args, "delay", 4.0)

    # httpx.AsyncClient is used as a context manager to ensure connections are
    # properly closed when scraping completes (or if an error occurs).
    async with httpx.AsyncClient() as client:
        provider = RAGraphQLProvider(http_client=client, scrape_delay=delay)
        service = RAScrapeService(provider=provider)

        # Progress callback invoked after each month is scraped.
        # Prints running totals to give the operator visibility during long scrapes.
        def on_progress(city: str, count: int, year: int, month: int) -> None:
            display = CITY_DISPLAY_NAMES.get(city, city)
            print(f"  [{display}] {count:,} events ({year}-{month:02d})")

        # Determine which cities to scrape: all 17 target cities or a single one.
        # City keys are normalized to lowercase with underscores (e.g., "new_york").
        if args.all:
            cities = list(RA_AREA_IDS.keys())
        else:
            city_key = args.city.lower().replace(" ", "_")
            if city_key not in RA_AREA_IDS:
                print(f"Error: Unknown city '{args.city}'.", file=sys.stderr)
                print(f"Available cities: {', '.join(sorted(RA_AREA_IDS))}")
                return 1
            cities = [city_key]

        # --skip flag allows excluding specific cities from an --all scrape.
        # Useful for skipping cities that have already been fully scraped.
        skip_set = {s.lower().replace(" ", "_") for s in args.skip}
        if skip_set:
            cities = [c for c in cities if c not in skip_set]
            print(f"Skipping: {', '.join(sorted(skip_set))}")

        start_year = args.start_year
        end_year = args.end_year

        print(f"Scraping {len(cities)} city(ies), {start_year}-{end_year or 'present'}")
        print(f"Rate limit: {delay}s between requests")
        print()

        for city in cities:
            display = CITY_DISPLAY_NAMES.get(city, city)
            area_id = RA_AREA_IDS[city]
            print(f"--- {display} (area_id={area_id}) ---")

            events = await service.scrape_city(
                city=city,
                start_year=start_year,
                end_year=end_year,
                on_progress=on_progress,
            )

            print(f"  Total: {len(events):,} events")
            print()

    print("Scrape complete.")
    return 0


async def _handle_generate_corpus(args: argparse.Namespace) -> int:
    """Generate corpus text files from scraped JSON checkpoints.

    Reads the raw JSON event data saved by the scrape command and converts
    each city's events into a structured plain text file suitable for
    chunking and embedding. The text format includes event name, date,
    venue, artists, and other metadata in a consistent format.
    """
    from src.providers.event.ra_graphql_provider import RAGraphQLProvider
    from src.services.ra_scrape_service import RAScrapeService

    async with httpx.AsyncClient() as client:
        provider = RAGraphQLProvider(http_client=client)
        service = RAScrapeService(provider=provider)

        if args.all:
            cities = list(RA_AREA_IDS.keys())
        else:
            city_key = args.city.lower().replace(" ", "_")
            cities = [city_key]

        for city in cities:
            display = CITY_DISPLAY_NAMES.get(city, city)
            path = service.generate_corpus_file(city)
            if path:
                print(f"  {display}: {path}")
            else:
                print(f"  {display}: no data (scrape first)")

    print("\nCorpus generation complete.")
    return 0


async def _handle_ingest(args: argparse.Namespace) -> int:
    """Ingest scraped RA events into ChromaDB via the full pipeline.

    This is the most complex subcommand. It combines corpus generation and
    ingestion into one step:
      1. Generate corpus text files from scraped JSON (if not already done)
      2. Either:
         a. --skip-tagging: Use RAEventProcessor to extract tags directly from
            the structured event data (fast, free, no LLM calls)
         b. Default: Run full ingestion pipeline with LLM metadata tagging
            (slower, costs API tokens, but produces richer semantic tags)

    The --skip-tagging mode is recommended for large corpora because LLM
    tagging costs ~$0.01-0.05 per chunk and the pre-extracted event tags
    (artist names, venue, date, city) are already quite rich.
    """
    # Reuse the ingestion service factory from ingest.py to avoid duplication.
    from src.cli.ingest import _build_ingestion_service
    from src.config.settings import Settings

    app_settings = Settings()
    service, status_msg = _build_ingestion_service(app_settings)
    if service is None:
        print(f"Error: {status_msg}", file=sys.stderr)
        return 1

    print(f"Providers: {status_msg}")

    # First generate the corpus text files.
    from src.providers.event.ra_graphql_provider import RAGraphQLProvider
    from src.services.ra_scrape_service import RAScrapeService

    async with httpx.AsyncClient() as client:
        ra_provider = RAGraphQLProvider(http_client=client)
        scrape_service = RAScrapeService(provider=ra_provider)

        if args.all:
            cities = list(RA_AREA_IDS.keys())
        else:
            city_key = args.city.lower().replace(" ", "_")
            cities = [city_key]

        corpus_files: list[str] = []
        for city in cities:
            path = scrape_service.generate_corpus_file(city)
            if path:
                corpus_files.append(str(path))
                display = CITY_DISPLAY_NAMES.get(city, city)
                print(f"  Generated: {display} ({path})")

    if not corpus_files:
        print("No corpus files generated. Scrape first.")
        return 1

    skip_tagging = getattr(args, "skip_tagging", False)

    if skip_tagging:
        # FAST PATH: Bypass LLM tagging entirely.
        # RAEventProcessor extracts tags (artists, venue, city, date) directly
        # from the structured event JSON data. This is much faster and free
        # (no LLM API calls) compared to running the full metadata extractor.
        # Trade-off: tags are limited to what's explicitly in the event data
        # (no inferred genres, scene connections, or cultural context).
        from src.services.ingestion.source_processors.ra_event_processor import (
            RAEventProcessor,
        )

        processor = RAEventProcessor()
        total_chunks = 0
        total_events = 0

        for city in cities:
            # Load raw event data from the JSON checkpoint file.
            events = scrape_service._load_events(city)
            if not events:
                continue

            display = CITY_DISPLAY_NAMES.get(city, city)
            # Process events into DocumentChunk objects with pre-extracted metadata.
            chunks = processor.process_events(events, display)

            if chunks:
                # Directly embed and store, bypassing the metadata extractor.
                # This accesses private members of the ingestion service — not
                # ideal OOP but acceptable for a CLI utility that needs to
                # optimize the hot path for large corpus ingestion.
                texts = [c.text for c in chunks]
                embeddings = await service._embedding_provider.embed(texts)
                stored = await service._vector_store.add_chunks(chunks, embeddings)
                total_chunks += stored
                total_events += len(events)
                print(f"  Ingested {display}: {stored} chunks from {len(events):,} events")

        print(f"\nTotal: {total_chunks} chunks from {total_events:,} events")
    else:
        # FULL PATH: Use the standard ingestion pipeline with LLM metadata tagging.
        # This produces richer semantic tags but costs API tokens and takes longer.
        # The corpus text files are ingested as "event" source type.
        import os

        corpus_dir = os.path.dirname(corpus_files[0]) if corpus_files else "data/reference_corpus"
        results = await service.ingest_directory(
            dir_path=corpus_dir,
            source_type="event",
        )
        total_chunks = sum(r.chunks_created for r in results)
        total_tokens = sum(r.total_tokens for r in results)
        print(f"\nIngested {len(results)} files: {total_chunks} chunks, {total_tokens:,} tokens")

    print("Ingestion complete.")
    return 0


async def _handle_verify_ids(args: argparse.Namespace) -> int:
    """Verify RA area IDs with test queries.

    Sends a single test query per city to RA's GraphQL API and reports
    whether each area ID returns valid results. This is useful when
    RA changes their area ID scheme or when adding new cities.

    Each city is queried for events in 2024 (a known good year).
    """
    from src.providers.event.ra_graphql_provider import RAGraphQLProvider

    async with httpx.AsyncClient() as client:
        provider = RAGraphQLProvider(http_client=client, scrape_delay=3.0)

        print("Verifying RA area IDs (1 test query per city)...")
        print()
        print(f"{'City':<20} {'Area ID':>8} {'Status':<12} {'Events':>10}")
        print("-" * 55)

        for city_key, area_id in RA_AREA_IDS.items():
            display = CITY_DISPLAY_NAMES.get(city_key, city_key)
            page = await provider.fetch_events_page(
                area_id=area_id,
                start_date="2024-01-01",
                end_date="2024-12-31",
                page=1,
                city=display,
            )

            if page.total_results > 0:
                status = "OK"
            else:
                status = "NO DATA"

            print(f"{display:<20} {area_id:>8} {status:<12} {page.total_results:>10,}")

    print("\nVerification complete.")
    return 0


def _handle_status() -> int:
    """Show scrape progress for all cities.

    Reads checkpoint files from disk to display a table of scrape progress
    for each city. This is a synchronous, read-only operation — no network
    requests are made. The HTTP client is created but never used (required
    by RAGraphQLProvider constructor).

    Output columns:
      City     — Human-readable city name
      Area ID  — RA's internal area identifier
      Events   — Total events scraped so far
      Range    — Date range of scraped events
      Last     — Last completed year-month
      Done     — Whether scraping is complete for this city
    """
    from src.providers.event.ra_graphql_provider import RAGraphQLProvider
    from src.services.ra_scrape_service import RAScrapeService

    # Build service without needing a real HTTP client — only checkpoint I/O is used.
    provider = RAGraphQLProvider(http_client=httpx.AsyncClient())
    service = RAScrapeService(provider=provider)

    statuses = service.get_scrape_status()

    print("RA Scrape Progress")
    print("=" * 78)
    print(
        f"{'City':<20} {'Area ID':>8} {'Events':>10}"
        f" {'Range':<12} {'Last':>10} {'Done':>6}"
    )
    print("-" * 78)

    seen_cities: set[str] = set()
    total_events = 0
    for s in statuses:
        city_key = str(s["city"])
        display = CITY_DISPLAY_NAMES.get(city_key, city_key)
        events = int(s["events"])
        last_year = int(s["last_year"])
        last_month = int(s["last_month"])
        complete = bool(s["complete"])
        range_str = str(s.get("range", ""))

        if city_key not in seen_cities:
            seen_cities.add(city_key)
            total_events += events

        last_str = f"{last_year}-{last_month:02d}" if last_year else "-"
        done_str = "YES" if complete else "no"

        print(
            f"{display:<20} {s['area_id']:>8} {events:>10,}"
            f" {range_str:<12} {last_str:>10} {done_str:>6}"
        )

    print("-" * 78)
    print(f"{'TOTAL':<20} {'':>8} {total_events:>10,}")

    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the RA scrape CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.scrape_ra",
        description="Scrape RA.co events and ingest into the raiveFlier RAG corpus.",
    )
    subparsers = parser.add_subparsers(dest="command", help="RA scrape commands")

    # -- scrape --
    scrape_parser = subparsers.add_parser(
        "scrape", help="Scrape events from RA.co"
    )
    scrape_group = scrape_parser.add_mutually_exclusive_group(required=True)
    scrape_group.add_argument("--city", help="City to scrape (e.g. 'chicago')")
    scrape_group.add_argument(
        "--all", action="store_true", help="Scrape all 17 target cities"
    )
    scrape_parser.add_argument(
        "--start-year",
        type=int,
        default=2016,
        help="Start year (default: 2016)",
    )
    scrape_parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="End year (default: current year)",
    )
    scrape_parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        help="Cities to skip (e.g. --skip tokyo barcelona manchester)",
    )
    scrape_parser.add_argument(
        "--delay",
        type=float,
        default=4.0,
        help="Seconds between requests (default: 4.0)",
    )

    # -- generate-corpus --
    gen_parser = subparsers.add_parser(
        "generate-corpus",
        help="Generate corpus text files from scraped JSON",
    )
    gen_group = gen_parser.add_mutually_exclusive_group(required=True)
    gen_group.add_argument("--city", help="City to generate corpus for")
    gen_group.add_argument(
        "--all", action="store_true", help="Generate for all scraped cities"
    )

    # -- ingest --
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Generate corpus files and ingest into ChromaDB",
    )
    ingest_group = ingest_parser.add_mutually_exclusive_group(required=True)
    ingest_group.add_argument("--city", help="City to ingest")
    ingest_group.add_argument(
        "--all", action="store_true", help="Ingest all scraped cities"
    )
    ingest_parser.add_argument(
        "--skip-tagging",
        action="store_true",
        help="Skip LLM metadata tagging (uses pre-extracted tags, saves cost)",
    )

    # -- verify-ids --
    subparsers.add_parser(
        "verify-ids",
        help="Verify RA area IDs with test queries",
    )

    # -- status --
    subparsers.add_parser("status", help="Show scrape progress for all cities")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the RA scrape tool.

    Dispatches to the appropriate subcommand handler. The "status" command
    is synchronous (no async needed — just reads checkpoint files from disk).
    All other commands are async because they make HTTP requests or interact
    with the vector store.
    """
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Status is the only synchronous command (reads local files only).
    if args.command == "status":
        exit_code = _handle_status()
        sys.exit(exit_code)

    # All other commands are async — wrap in asyncio.run().
    if args.command == "scrape":
        exit_code = asyncio.run(_handle_scrape(args))
    elif args.command == "generate-corpus":
        exit_code = asyncio.run(_handle_generate_corpus(args))
    elif args.command == "ingest":
        exit_code = asyncio.run(_handle_ingest(args))
    elif args.command == "verify-ids":
        exit_code = asyncio.run(_handle_verify_ids(args))
    else:
        parser.print_help()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

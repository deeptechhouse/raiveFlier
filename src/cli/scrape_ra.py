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

from src.providers.event.ra_graphql_provider import (
    CITY_DISPLAY_NAMES,
    RA_AREA_IDS,
)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


async def _handle_scrape(args: argparse.Namespace) -> int:
    """Scrape events from RA.co for one or all cities."""
    from src.providers.event.ra_graphql_provider import RAGraphQLProvider
    from src.services.ra_scrape_service import RAScrapeService

    delay = getattr(args, "delay", 4.0)

    async with httpx.AsyncClient() as client:
        provider = RAGraphQLProvider(http_client=client, scrape_delay=delay)
        service = RAScrapeService(provider=provider)

        def on_progress(city: str, count: int, year: int, month: int) -> None:
            display = CITY_DISPLAY_NAMES.get(city, city)
            print(f"  [{display}] {count:,} events ({year}-{month:02d})")

        if args.all:
            cities = list(RA_AREA_IDS.keys())
        else:
            city_key = args.city.lower().replace(" ", "_")
            if city_key not in RA_AREA_IDS:
                print(f"Error: Unknown city '{args.city}'.", file=sys.stderr)
                print(f"Available cities: {', '.join(sorted(RA_AREA_IDS))}")
                return 1
            cities = [city_key]

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
    """Generate corpus text files from scraped JSON checkpoints."""
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
    """Ingest scraped RA events into ChromaDB via the full pipeline."""
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
        # Bypass LLM tagging — use pre-extracted tags from the event processor.
        from src.services.ingestion.source_processors.ra_event_processor import (
            RAEventProcessor,
        )

        processor = RAEventProcessor()
        total_chunks = 0
        total_events = 0

        for city in cities:
            events = scrape_service._load_events(city)
            if not events:
                continue

            display = CITY_DISPLAY_NAMES.get(city, city)
            chunks = processor.process_events(events, display)

            if chunks:
                # Embed and store directly (skip metadata extractor).
                texts = [c.text for c in chunks]
                embeddings = await service._embedding_provider.embed(texts)
                stored = await service._vector_store.add_chunks(chunks, embeddings)
                total_chunks += stored
                total_events += len(events)
                print(f"  Ingested {display}: {stored} chunks from {len(events):,} events")

        print(f"\nTotal: {total_chunks} chunks from {total_events:,} events")
    else:
        # Full pipeline with LLM tagging via directory ingestion.
        import os

        corpus_dir = os.path.dirname(corpus_files[0]) if corpus_files else "data/reference_corpus"
        results = await service.ingest_directory(
            dir_path=corpus_dir,
            source_type="event_listing",
        )
        total_chunks = sum(r.chunks_created for r in results)
        total_tokens = sum(r.total_tokens for r in results)
        print(f"\nIngested {len(results)} files: {total_chunks} chunks, {total_tokens:,} tokens")

    print("Ingestion complete.")
    return 0


async def _handle_verify_ids(args: argparse.Namespace) -> int:
    """Verify RA area IDs with test queries."""
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
    """Show scrape progress for all cities."""
    from src.providers.event.ra_graphql_provider import RAGraphQLProvider
    from src.services.ra_scrape_service import RAScrapeService

    # Build service without HTTP client (only checkpoint I/O needed).
    provider = RAGraphQLProvider(http_client=httpx.AsyncClient())
    service = RAScrapeService(provider=provider)

    statuses = service.get_scrape_status()

    print("RA Scrape Progress")
    print("=" * 65)
    print(f"{'City':<20} {'Area ID':>8} {'Events':>10} {'Last':>10} {'Done':>6}")
    print("-" * 65)

    total_events = 0
    for s in statuses:
        city_key = str(s["city"])
        display = CITY_DISPLAY_NAMES.get(city_key, city_key)
        events = int(s["events"])
        total_events += events
        last_year = int(s["last_year"])
        last_month = int(s["last_month"])
        complete = bool(s["complete"])

        last_str = f"{last_year}-{last_month:02d}" if last_year else "-"
        done_str = "YES" if complete else "no"

        print(f"{display:<20} {s['area_id']:>8} {events:>10,} {last_str:>10} {done_str:>6}")

    print("-" * 65)
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
    """CLI entry point for the RA scrape tool."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "status":
        exit_code = _handle_status()
        sys.exit(exit_code)

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

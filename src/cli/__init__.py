# =============================================================================
# src/cli/__init__.py — CLI Module Overview
# =============================================================================
#
# This package provides standalone command-line tools for the raiveFlier
# application. Each submodule is a self-contained CLI utility that can be
# run directly via `python -m src.cli.<module>`.
#
# The CLI layer is the primary interface for operators and developers who
# need to interact with the system outside of the web frontend/API. It
# covers three major workflows:
#
#   1. ANALYSIS  (analyze.py)
#      Runs the full 5-phase flier analysis pipeline (OCR -> entity
#      extraction -> research -> interconnection -> output) on a local
#      image file and prints results to stdout.
#
#   2. INGESTION (ingest.py)
#      Manages the RAG (Retrieval-Augmented Generation) vector store
#      corpus. Supports ingesting books (TXT, PDF, EPUB), web articles,
#      and bulk directory imports. Also provides corpus stats and purge.
#
#   3. RA SCRAPER (scrape_ra.py)
#      Scrapes event listings from Resident Advisor (RA.co) via their
#      GraphQL API, generates corpus text files, and optionally ingests
#      them into ChromaDB. Covers 17 global cities from 2016-present.
#
# Architecture Notes:
#   - All CLI modules use argparse for argument parsing (not Click/Typer)
#     to minimize external dependencies.
#   - Heavy imports (LLM providers, vector stores, etc.) are deferred
#     inside functions to keep startup time fast for simple commands.
#   - Each module constructs its own service dependencies rather than
#     relying on a central DI container, because CLI tools run as
#     one-shot scripts, not long-lived servers.
# =============================================================================

"""CLI tools for the raiveFlier pipeline.

Provides standalone command-line utilities:

- ``python -m src.cli.analyze`` — analyze a rave flier image (full pipeline)
- ``python -m src.cli.ingest`` — ingest books, articles, and directories
  into the vector store.
- ``python -m src.cli.scrape_ra`` — scrape RA.co events and ingest into
  the RAG corpus.
"""

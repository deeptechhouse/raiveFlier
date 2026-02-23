# =============================================================================
# src/cli/__main__.py — Package Entry Point
# =============================================================================
#
# This file enables running the CLI package itself as a module:
#     python -m src.cli
#
# When Python encounters `python -m src.cli`, it looks for __main__.py
# inside the package and executes it. This delegates to the ingestion
# CLI (ingest.py) as the default subcommand, since corpus management is
# the most common CLI operation during development and deployment.
#
# For other CLI tools, run them directly:
#     python -m src.cli.analyze /path/to/flier.jpg
#     python -m src.cli.scrape_ra status
# =============================================================================

"""Allow ``python -m src.cli.ingest`` execution."""

from src.cli.ingest import main

# Invoke the ingestion CLI's main() when this package is run as a module.
main()

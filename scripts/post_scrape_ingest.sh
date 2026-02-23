#!/bin/bash
# =============================================================================
# scripts/post_scrape_ingest.sh — Post-Scrape Ingestion Pipeline
# =============================================================================
#
# Automates the two-step process that follows an RA.co event scrape:
#   1. Generate corpus text files from the scraped JSON data
#   2. Ingest those corpus files into ChromaDB (with --skip-tagging for speed)
#
# Designed to be kicked off in the background while a scrape is running.
# If a scrape PID is provided, the script polls every 30 seconds until
# that process exits, then proceeds with corpus generation and ingestion.
#
# Typical workflow:
#   # Terminal 1: Start the scrape
#   python3 -m src.cli.scrape_ra scrape --all &
#   SCRAPE_PID=$!
#
#   # Terminal 2: Queue the post-scrape pipeline
#   ./scripts/post_scrape_ingest.sh $SCRAPE_PID
#
#   # Or, run immediately (no PID — skip the wait):
#   ./scripts/post_scrape_ingest.sh
#
# All output is logged to scrape_ingest_pipeline.log in the project root.
#
# Uses --skip-tagging for ingestion because RA event data already contains
# structured metadata (artist names, venue, date, city) that the
# RAEventProcessor can extract directly without LLM calls.
# =============================================================================

# Exit on any error (-e), undefined variable (-u), or pipe failure (-o pipefail).
set -euo pipefail
# Change to project root (parent of scripts/ directory).
cd "$(dirname "$0")/.."

LOG="scrape_ingest_pipeline.log"

# Timestamped logging function — outputs to both terminal and log file.
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

# Optional: PID of a running scrape process to wait for.
SCRAPE_PID="${1:-}"

# Phase 0 (optional): Wait for the scrape process to complete.
# kill -0 checks if a process exists without actually sending a signal.
# We poll every 30 seconds to avoid busy-waiting.
if [ -n "$SCRAPE_PID" ]; then
    log "Waiting for scrape process (PID $SCRAPE_PID) to finish..."
    while kill -0 "$SCRAPE_PID" 2>/dev/null; do
        sleep 30
    done
    log "Scrape process finished."
fi

# Phase 1: Generate corpus text files from scraped JSON checkpoints.
# Converts the raw event JSON (data/ra_scrape/ra_events_*.json) into
# plain text files (data/reference_corpus/ra_events_*.txt).
log "=== Generating corpus files ==="
python3 -m src.cli.scrape_ra generate-corpus --all 2>&1 | tee -a "$LOG"

# Phase 2: Ingest corpus files into ChromaDB.
# --skip-tagging bypasses LLM metadata extraction, using pre-extracted
# tags from RAEventProcessor instead (free, fast, and sufficient for
# structured event data).
log "=== Ingesting into ChromaDB ==="
python3 -m src.cli.scrape_ra ingest --all --skip-tagging 2>&1 | tee -a "$LOG"

log "=== Pipeline complete ==="

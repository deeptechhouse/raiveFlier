#!/bin/bash
# =============================================================================
# scripts/scrape_status.sh — Live RA Scrape Progress Monitor
# =============================================================================
#
# Provides a live-updating terminal dashboard showing the progress of the
# RA.co event scraper. Refreshes every 30 seconds. Press Ctrl+C to exit.
#
# Displays:
#   - Per-city event counts and progress (which year/month is completed)
#   - Total event count across all cities
#   - Whether the scrape process is currently running
#   - Whether the post-scrape ingestion pipeline is waiting
#
# Data Source:
#   Reads directly from the scrape checkpoint files in data/ra_scrape/:
#     - ra_events_<city>.json — Raw event data (event count = array length)
#     - ra_progress_<city>_*.json — Progress checkpoints (year/month, complete flag)
#
# Process Detection:
#   Uses `pgrep -f` to check if scrape_ra or post_scrape_ingest processes
#   are currently running. This is a simple heuristic — it matches on the
#   command line string, so it may false-positive if other processes contain
#   these strings.
#
# Usage:
#   ./scripts/scrape_status.sh
# =============================================================================

# Change to project root so relative paths to data/ work correctly.
cd "$(dirname "$0")/.."

# Infinite loop: clear screen, display status, sleep 30 seconds.
while true; do
    clear
    # The status display is implemented as an inline Python script because
    # parsing JSON and formatting tables is much easier in Python than bash.
    python3 -c "
import json, subprocess
from pathlib import Path

# Read all per-city event data files from the scrape directory.
base = Path('data/ra_scrape')
total = 0
lines = []
for f in sorted(base.glob('ra_events_*.json')):
    # Extract city name from filename (e.g., 'ra_events_chicago.json' -> 'chicago')
    city = f.stem.replace('ra_events_', '')
    count = len(json.loads(f.read_text()))
    total += count

    # Find the most recent progress checkpoint for this city.
    # Progress files are named ra_progress_<city>_<timestamp>.json.
    progs = sorted(base.glob(f'ra_progress_{city}_*.json'))
    p = json.loads(progs[-1].read_text()) if progs else {}

    # Determine status: either COMPLETE or showing the last completed month.
    if p.get('is_complete'):
        status = 'COMPLETE'
    else:
        y = p.get('last_completed_year', '?')
        m = p.get('last_completed_month', 0)
        status = f'through {y}-{m:02d}'
    lines.append(f'{city:18s} {count:>7,}   {status}')

# Display the formatted table.
print('RA Scrape Progress')
print('=' * 48)
for l in lines:
    print(l)
print('-' * 48)
print(f'{\"TOTAL\":18s} {total:>7,}')
print()

# Check if the scrape process is currently running.
result = subprocess.run(['pgrep', '-f', 'scrape_ra'], capture_output=True)
if result.returncode == 0:
    print('Scrape: RUNNING')
else:
    print('Scrape: FINISHED')

# Check if the post-scrape ingestion pipeline is waiting.
result2 = subprocess.run(['pgrep', '-f', 'post_scrape_ingest'], capture_output=True)
if result2.returncode == 0:
    print('Ingest pipeline: WAITING')
print()
print('Refreshing every 30s — Ctrl+C to exit')
"
    sleep 30
done

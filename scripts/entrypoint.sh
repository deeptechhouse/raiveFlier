#!/bin/bash
# =============================================================================
# scripts/entrypoint.sh — Docker Entrypoint for raiveFlier
# =============================================================================
#
# This script is the CMD for the Docker container. It runs two phases:
#
#   Phase 1: CORPUS DOWNLOAD (optional, non-blocking)
#     If the ChromaDB persistent directory is empty or contains only the
#     small reference corpus (~36 chunks, <1 MB), this script downloads
#     the full pre-built corpus (~154 MB, ~15K chunks) from a private
#     GitHub release. This saves 10-15 minutes of ingestion time on
#     first deploy.
#
#     Requirements for corpus download:
#       - GITHUB_TOKEN env var must be set (personal access token with repo scope)
#       - RAG_ENABLED must be "true"
#       - The private repo (deeptechhouse/raiveflier-corpus) must have a
#         release tagged with the version in CORPUS_TAG
#
#     If download fails for any reason, the app still starts — it will
#     just use the small reference corpus baked into the Docker image.
#
#   Phase 2: APPLICATION START
#     Starts the FastAPI application via uvicorn with a single worker.
#     Uses `exec` to replace the shell process so uvicorn becomes PID 1,
#     ensuring it receives SIGTERM directly for graceful shutdown.
#
# Deployment constraints (Render Starter):
#   - 512 MB RAM — single worker, no preloading
#   - /data is persistent disk — only reliable storage across redeploys
#   - $PORT is injected by Render (overrides the default 8000)
#
# Environment Variables:
#   CHROMADB_PERSIST_DIR — ChromaDB directory (default: /data/chromadb)
#   CORPUS_REPO          — GitHub repo for corpus download (default: deeptechhouse/raiveflier-corpus)
#   CORPUS_TAG           — Release tag to download (default: v1.0.0)
#   GITHUB_TOKEN         — GitHub PAT for private repo access
#   RAG_ENABLED          — Must be "true" to enable corpus download
#   PORT                 — HTTP port (default: 8000, overridden by Render)
# =============================================================================

# Read configuration from environment variables with sensible defaults.
CHROMADB_DIR="${CHROMADB_PERSIST_DIR:-/data/chromadb}"
CORPUS_REPO="${CORPUS_REPO:-deeptechhouse/raiveflier-corpus}"
CORPUS_TAG="${CORPUS_TAG:-v1.0.0}"

# =============================================================================
# Phase 1: Corpus Download
# =============================================================================
# Downloads the pre-built ChromaDB corpus from a private GitHub release.
# This function is designed to be resilient — any failure is non-fatal
# and the application will start regardless.
download_corpus() {
    # Guard: skip if no GitHub token is available for authentication.
    if [ -z "$GITHUB_TOKEN" ]; then
        echo "[entrypoint] No GITHUB_TOKEN set — skipping corpus download"
        return 0
    fi

    # Guard: skip if RAG is not enabled (no point downloading a corpus).
    if [ "$RAG_ENABLED" != "true" ]; then
        echo "[entrypoint] RAG_ENABLED is not true — skipping corpus download"
        return 0
    fi

    # Size check: determine if the existing corpus is the full version or
    # just the small auto-ingested reference set.
    # Full corpus: chroma.sqlite3 > 50 MB (~154 MB with ~15K chunks)
    # Reference set: chroma.sqlite3 < 1 MB (~36 chunks)
    DB_FILE="$CHROMADB_DIR/chroma.sqlite3"
    MIN_CORPUS_SIZE=50000000  # 50 MB threshold
    if [ -f "$DB_FILE" ]; then
        DB_SIZE=$(wc -c < "$DB_FILE" 2>/dev/null || echo "0")
        if [ "$DB_SIZE" -gt "$MIN_CORPUS_SIZE" ]; then
            echo "[entrypoint] Full corpus already present at $CHROMADB_DIR ($DB_SIZE bytes)"
            return 0
        fi
        echo "[entrypoint] Corpus DB too small ($DB_SIZE bytes < ${MIN_CORPUS_SIZE}) — downloading full corpus..."
    else
        echo "[entrypoint] No corpus found at $CHROMADB_DIR — downloading..."
    fi

    # Step 1: Query the GitHub Releases API to find the download URL.
    # We need the asset ID to download the binary release asset.
    echo "[entrypoint] Fetching release info from $CORPUS_REPO/$CORPUS_TAG..."
    RELEASE_JSON=$(curl -sL \
        -H "Authorization: token $GITHUB_TOKEN" \
        -H "Accept: application/vnd.github+json" \
        "https://api.github.com/repos/$CORPUS_REPO/releases/tags/$CORPUS_TAG")

    # Parse the asset ID from the JSON response using Python (available in the
    # container since we use python:3.12-slim as the base image).
    # Parse asset ID, routing stderr to /dev/null so error messages
    # don't get captured into the ASSET_ID variable (the old 2>&1
    # redirect was mixing stderr text into ASSET_ID, causing the
    # download to always fail with "Could not find corpus release asset").
    ASSET_ID=$(echo "$RELEASE_JSON" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    assets = data.get('assets', [])
    if assets:
        print(assets[0]['id'])
    else:
        print('No assets in release', file=sys.stderr)
        sys.exit(1)
except Exception as e:
    print(f'JSON parse error: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null)

    if [ -z "$ASSET_ID" ] || [ "$ASSET_ID" = "None" ]; then
        echo "[entrypoint] WARNING: Could not find corpus release asset"
        echo "[entrypoint] API response: $(echo "$RELEASE_JSON" | head -c 500)"
        return 0
    fi

    # Step 2: Download and extract the corpus tarball.
    # The tarball is expected to contain a chromadb/ directory at its root,
    # which extracts directly into the parent of CHROMADB_DIR.
    echo "[entrypoint] Downloading corpus (asset $ASSET_ID, ~154 MB)..."
    mkdir -p "$CHROMADB_DIR"

    PARENT_DIR=$(dirname "$CHROMADB_DIR")
    MAX_RETRIES=3       # Retry up to 3 times on network failure
    RETRY_DELAY=5       # Wait 5 seconds between retries
    DOWNLOAD_OK=false

    for attempt in $(seq 1 $MAX_RETRIES); do
        echo "[entrypoint] Download attempt $attempt of $MAX_RETRIES..."
        # Download the binary asset and pipe directly to tar for extraction.
        # The Accept: application/octet-stream header tells GitHub to return
        # the raw binary file instead of JSON metadata.
        if curl -sL --fail \
            -H "Authorization: token $GITHUB_TOKEN" \
            -H "Accept: application/octet-stream" \
            "https://api.github.com/repos/$CORPUS_REPO/releases/assets/$ASSET_ID" \
            | tar xz -C "$PARENT_DIR"; then
            DOWNLOAD_OK=true
            echo "[entrypoint] Corpus downloaded and extracted to $CHROMADB_DIR"
            break
        else
            echo "[entrypoint] Attempt $attempt failed"
            if [ "$attempt" -lt "$MAX_RETRIES" ]; then
                echo "[entrypoint] Retrying in ${RETRY_DELAY}s..."
                sleep "$RETRY_DELAY"
            fi
        fi
    done

    # Step 3: Verify the extraction produced a valid corpus file.
    if [ "$DOWNLOAD_OK" = true ]; then
        if [ -f "$CHROMADB_DIR/chroma.sqlite3" ]; then
            FINAL_SIZE=$(wc -c < "$CHROMADB_DIR/chroma.sqlite3")
            echo "[entrypoint] Verified: chroma.sqlite3 is $FINAL_SIZE bytes"
            # Sanity check: if the file is suspiciously small, something went wrong
            # during download/extraction (partial download, corrupt tarball, etc.)
            if [ "$FINAL_SIZE" -lt 100000 ]; then
                echo "[entrypoint] WARNING: Corpus file suspiciously small ($FINAL_SIZE bytes)"
            fi
        else
            echo "[entrypoint] WARNING: chroma.sqlite3 not found after extraction"
        fi
    else
        echo "[entrypoint] WARNING: Corpus download failed after $MAX_RETRIES attempts"
    fi
}

# Run corpus download — errors must NOT prevent the app from starting.
# The || clause catches any unhandled failure in download_corpus().
download_corpus || echo "[entrypoint] Corpus download encountered an error — continuing..."

# =============================================================================
# Phase 1b: Clear Stale ChromaDB Data (if needed)
# =============================================================================
# If the corpus download failed (or was skipped) and the existing ChromaDB
# data is undersized, it may have been created by an older ChromaDB version.
# Version mismatches cause "'PersistentData' object has no attribute
# 'max_seq_id'" errors that prevent any new data from being stored.
# Clearing the stale data lets ChromaDB create a fresh database with the
# current version's schema.  The reference corpus will be re-ingested
# automatically by the background task on startup.
DB_FILE="$CHROMADB_DIR/chroma.sqlite3"
if [ -f "$DB_FILE" ]; then
    DB_SIZE=$(wc -c < "$DB_FILE" 2>/dev/null || echo "0")
    if [ "$DB_SIZE" -lt 50000000 ]; then
        echo "[entrypoint] Clearing stale ChromaDB data ($DB_SIZE bytes < 50 MB threshold)"
        echo "[entrypoint] This ensures the current ChromaDB version creates a compatible schema"
        rm -rf "${CHROMADB_DIR:?}"/*
        echo "[entrypoint] Cleared — fresh corpus will be built by background ingestion"
    fi
fi

# =============================================================================
# Phase 2: Start the Application
# =============================================================================
# Start uvicorn (ASGI server) with the FastAPI application.
#
# Key flags:
#   --host 0.0.0.0     — Bind to all interfaces (required for Docker/Render)
#   --port $PORT        — Use Render's injected port (default: 8000)
#   --workers 1         — Single worker to stay within 512 MB RAM limit
#   --timeout-keep-alive 65 — Slightly above Render's 60s load balancer timeout
#                              to prevent premature connection closures
#
# `exec` replaces this shell process with uvicorn, making it PID 1.
# This is important because Docker sends SIGTERM to PID 1 on container stop,
# and uvicorn needs to receive it directly for graceful shutdown.
echo "[entrypoint] Starting uvicorn on port ${PORT:-8000}..."
exec python3 -m uvicorn src.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 1 \
    --timeout-keep-alive 65

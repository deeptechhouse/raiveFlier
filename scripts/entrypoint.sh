#!/bin/bash
# =============================================================================
# scripts/entrypoint.sh — Docker Entrypoint for raiveFlier
# =============================================================================
#
# This script is the CMD for the Docker container. It runs two phases:
#
#   Phase 1: CORPUS DOWNLOAD (optional, non-blocking)
#     Automatically fetches the LATEST corpus release from GitHub.
#     Supports both single-asset and multi-part releases (split archives
#     that exceed GitHub's 2 GB per-asset limit are concatenated before
#     extraction).
#
#     Downloads if:
#       - No corpus exists locally, OR
#       - Local corpus is small (<50 MB), OR
#       - A newer release is available (version mismatch)
#
#     Requirements:
#       - GITHUB_TOKEN env var (personal access token with repo scope)
#       - RAG_ENABLED must be "true"
#
#     If download fails, the app starts anyway with the reference corpus.
#
#   Phase 2: APPLICATION START
#     Starts FastAPI via uvicorn (single worker, 2 GB RAM — Standard plan).
#
# Environment Variables:
#   CHROMADB_PERSIST_DIR — ChromaDB directory (default: /data/chromadb)
#   CORPUS_REPO          — GitHub repo for corpus releases (default: deeptechhouse/raiveflier-corpus)
#   GITHUB_TOKEN         — GitHub PAT for private repo access
#   RAG_ENABLED          — Must be "true" to enable corpus download
#   PORT                 — HTTP port (default: 8000, overridden by Render)
# =============================================================================

# Read configuration from environment variables with sensible defaults.
CHROMADB_DIR="${CHROMADB_PERSIST_DIR:-/data/chromadb}"
CORPUS_REPO="${CORPUS_REPO:-deeptechhouse/raiveflier-corpus}"

# =============================================================================
# Phase 1: Corpus Download
# =============================================================================
# Downloads the pre-built ChromaDB corpus from the LATEST GitHub release.
# Compares local corpus_version.txt against the latest release tag to
# decide whether a download is needed — no manual CORPUS_TAG required.
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

    # Step 1: Fetch the latest release tag from GitHub.
    echo "[entrypoint] Checking latest corpus release from $CORPUS_REPO..."
    RELEASE_JSON=$(curl -sL \
        -H "Authorization: token $GITHUB_TOKEN" \
        -H "Accept: application/vnd.github+json" \
        "https://api.github.com/repos/$CORPUS_REPO/releases/latest")

    # Parse the tag name and asset ID from the response.
    LATEST_TAG=$(echo "$RELEASE_JSON" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('tag_name', ''))
except Exception:
    pass
" 2>/dev/null)

    if [ -z "$LATEST_TAG" ]; then
        echo "[entrypoint] WARNING: Could not determine latest release tag"
        echo "[entrypoint] API response: $(echo "$RELEASE_JSON" | head -c 500)"
        return 0
    fi

    echo "[entrypoint] Latest release: $LATEST_TAG"

    # Step 2: Compare local version against latest release.
    DB_FILE="$CHROMADB_DIR/chroma.sqlite3"
    VERSION_FILE="$CHROMADB_DIR/corpus_version.txt"
    MIN_CORPUS_SIZE=50000000  # 50 MB threshold

    # Read local version (first line of corpus_version.txt, or "unknown").
    LOCAL_VERSION="unknown"
    if [ -f "$VERSION_FILE" ]; then
        LOCAL_VERSION=$(head -n1 "$VERSION_FILE" 2>/dev/null || echo "unknown")
    fi

    if [ -f "$DB_FILE" ]; then
        DB_SIZE=$(wc -c < "$DB_FILE" 2>/dev/null || echo "0")
        if [ "$DB_SIZE" -gt "$MIN_CORPUS_SIZE" ]; then
            # Large corpus exists — only re-download if version changed.
            if [ "$LOCAL_VERSION" = "$LATEST_TAG" ]; then
                echo "[entrypoint] Corpus up-to-date (version=$LOCAL_VERSION, size=$DB_SIZE bytes)"
                return 0
            fi
            echo "[entrypoint] New corpus available (local=$LOCAL_VERSION, latest=$LATEST_TAG) — downloading..."
        elif [ "$LOCAL_VERSION" = "unknown" ]; then
            # Small data with no version file means auto-ingest is building
            # the corpus from reference files.  Do NOT overwrite with the
            # GitHub release — it may use an incompatible ChromaDB version
            # (e.g. 0.5.x schema vs pinned 0.4.x), causing a clear-and-
            # rebuild loop on every container restart.
            echo "[entrypoint] Auto-ingest in progress ($DB_SIZE bytes, no version file) — skipping download"
            return 0
        else
            echo "[entrypoint] Corpus DB too small ($DB_SIZE bytes < ${MIN_CORPUS_SIZE}) — downloading..."
        fi
    else
        echo "[entrypoint] No corpus found at $CHROMADB_DIR — downloading..."
    fi

    # Step 3: Parse asset IDs for download.
    # The corpus may be a single tarball or split into multiple parts
    # (chromadb_corpus_part_*) when it exceeds GitHub's 2 GB per-asset limit.
    # Both formats are handled: single-asset streams directly to tar, while
    # multi-part downloads are concatenated before extraction.
    ASSET_INFO=$(echo "$RELEASE_JSON" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    assets = data.get('assets', [])
    if not assets:
        print('No assets in release', file=sys.stderr)
        sys.exit(1)
    # Sort by name so split parts (part_aa, part_ab, ...) are in order.
    assets.sort(key=lambda a: a['name'])
    for a in assets:
        print(f\"{a['id']}|{a['name']}|{a['size']}\")
except Exception as e:
    print(f'JSON parse error: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null)

    if [ -z "$ASSET_INFO" ]; then
        echo "[entrypoint] WARNING: Could not find corpus release assets"
        return 0
    fi

    ASSET_COUNT=$(echo "$ASSET_INFO" | wc -l | tr -d ' ')
    echo "[entrypoint] Found $ASSET_COUNT release asset(s)"

    # Step 4: Download and extract the corpus.
    mkdir -p "$CHROMADB_DIR"
    PARENT_DIR=$(dirname "$CHROMADB_DIR")
    MAX_RETRIES=3
    RETRY_DELAY=5
    DOWNLOAD_OK=false

    # Helper: download a single asset by ID to a file path.
    download_asset() {
        local asset_id="$1"
        local dest_path="$2"
        local attempt
        for attempt in $(seq 1 $MAX_RETRIES); do
            if curl -sL --fail \
                -H "Authorization: token $GITHUB_TOKEN" \
                -H "Accept: application/octet-stream" \
                "https://api.github.com/repos/$CORPUS_REPO/releases/assets/$asset_id" \
                -o "$dest_path"; then
                return 0
            fi
            echo "[entrypoint] Asset $asset_id attempt $attempt failed"
            [ "$attempt" -lt "$MAX_RETRIES" ] && sleep "$RETRY_DELAY"
        done
        return 1
    }

    if [ "$ASSET_COUNT" -eq 1 ]; then
        # Single-asset release: stream directly to tar (no temp file needed).
        ASSET_ID=$(echo "$ASSET_INFO" | head -1 | cut -d'|' -f1)
        ASSET_NAME=$(echo "$ASSET_INFO" | head -1 | cut -d'|' -f2)
        echo "[entrypoint] Downloading single asset: $ASSET_NAME..."

        for attempt in $(seq 1 $MAX_RETRIES); do
            echo "[entrypoint] Download attempt $attempt of $MAX_RETRIES..."
            if curl -sL --fail \
                -H "Authorization: token $GITHUB_TOKEN" \
                -H "Accept: application/octet-stream" \
                "https://api.github.com/repos/$CORPUS_REPO/releases/assets/$ASSET_ID" \
                | tar xz -C "$PARENT_DIR"; then
                DOWNLOAD_OK=true
                echo "[entrypoint] Corpus downloaded and extracted to $CHROMADB_DIR"
                break
            fi
            echo "[entrypoint] Attempt $attempt failed"
            [ "$attempt" -lt "$MAX_RETRIES" ] && sleep "$RETRY_DELAY"
        done
    else
        # Multi-part release: download all parts, concatenate, then extract.
        # Split parts are named chromadb_corpus_part_aa, part_ab, etc. and
        # must be concatenated in alphabetical order to reconstruct the
        # original tar.gz before extraction.
        echo "[entrypoint] Downloading $ASSET_COUNT corpus parts..."
        # Download parts to the persistent disk, NOT /tmp.
        # Render limits /tmp to 2 GB, but the corpus parts total ~3.8 GB.
        # The persistent disk at /data has enough space (20 GB).
        TMPDIR="$PARENT_DIR/_corpus_download"
        mkdir -p "$TMPDIR"
        ALL_PARTS_OK=true

        while IFS='|' read -r asset_id asset_name asset_size; do
            echo "[entrypoint] Downloading $asset_name ($(echo "$asset_size" | awk '{printf "%.0f MB", $1/1048576}'))..."
            if ! download_asset "$asset_id" "$TMPDIR/$asset_name"; then
                echo "[entrypoint] FAILED to download $asset_name after $MAX_RETRIES attempts"
                ALL_PARTS_OK=false
                break
            fi
        done <<< "$ASSET_INFO"

        if [ "$ALL_PARTS_OK" = true ]; then
            echo "[entrypoint] All parts downloaded — concatenating and extracting..."
            # Concatenate parts in sorted order and pipe to tar.
            cat "$TMPDIR"/chromadb_corpus_part_* | tar xz -C "$PARENT_DIR" && DOWNLOAD_OK=true
            echo "[entrypoint] Corpus extracted to $CHROMADB_DIR"
        fi

        # Clean up temporary download directory.
        rm -rf "$TMPDIR"
    fi

    # Step 5: Verify extraction and record the version.
    if [ "$DOWNLOAD_OK" = true ]; then
        if [ -f "$CHROMADB_DIR/chroma.sqlite3" ]; then
            FINAL_SIZE=$(wc -c < "$CHROMADB_DIR/chroma.sqlite3")
            echo "[entrypoint] Verified: chroma.sqlite3 is $FINAL_SIZE bytes"
            if [ "$FINAL_SIZE" -lt 100000 ]; then
                echo "[entrypoint] WARNING: Corpus file suspiciously small ($FINAL_SIZE bytes)"
            fi

            # Record the version so the next startup skips re-downloading.
            echo "$LATEST_TAG" > "$CHROMADB_DIR/corpus_version.txt"
            echo "[entrypoint] Recorded corpus version: $LATEST_TAG"
        else
            echo "[entrypoint] WARNING: chroma.sqlite3 not found after extraction"
        fi
    else
        echo "[entrypoint] WARNING: Corpus download failed — app will use auto-ingest fallback"
    fi
}

# Run corpus download — errors must NOT prevent the app from starting.
download_corpus || echo "[entrypoint] Corpus download encountered an error — continuing..."

# =============================================================================
# Phase 1b: Clear Incompatible ChromaDB Data
# =============================================================================
# ChromaDB 0.5.x+ changed the internal SQLite schema (added _type to
# collection config JSON).  If the persistent disk has data written by
# 0.5.x/0.6.x but the pinned version is 0.4.x, startup will crash with
# KeyError: '_type'.  Detect this and wipe the data so the auto-ingest
# rebuilds it with the compatible schema.
DB_FILE="$CHROMADB_DIR/chroma.sqlite3"
if [ -f "$DB_FILE" ]; then
    DB_SIZE=$(wc -c < "$DB_FILE" 2>/dev/null || echo "0")
    # Check for schema incompatibility: try to open the collection.
    # If it fails with a KeyError or any import error, the data is
    # from an incompatible ChromaDB version.
    COMPAT_OK=$(python3 -c "
import sys
try:
    import chromadb
    client = chromadb.PersistentClient(path='$CHROMADB_DIR')
    # Attempt to list collections — triggers config deserialization
    client.list_collections()
    print('ok')
except Exception as e:
    print(f'fail: {e}', file=sys.stderr)
    print('fail')
" 2>&1 | tail -1)

    if [ "$COMPAT_OK" != "ok" ]; then
        echo "[entrypoint] ChromaDB data incompatible with installed version"
        echo "[entrypoint] Clearing $CHROMADB_DIR so auto-ingest rebuilds the corpus..."
        rm -rf "${CHROMADB_DIR:?}"/*
        mkdir -p "$CHROMADB_DIR"
        echo "[entrypoint] Cleared — fresh corpus will be built by background ingestion"
    elif [ "$DB_SIZE" -lt 50000000 ]; then
        # Small but valid data — only clear if it came from a GitHub corpus
        # download (corpus_version.txt exists).  If no version file, the data
        # is from an in-progress auto-ingest rebuild and must be preserved
        # across container restarts to avoid a Sisyphean wipe-and-rebuild loop.
        VERSION_FILE="$CHROMADB_DIR/corpus_version.txt"
        if [ -f "$VERSION_FILE" ]; then
            echo "[entrypoint] Clearing stale downloaded corpus ($DB_SIZE bytes < 50 MB threshold)"
            rm -rf "${CHROMADB_DIR:?}"/*
            echo "[entrypoint] Cleared — fresh corpus will be built by background ingestion"
        else
            echo "[entrypoint] ChromaDB data small ($DB_SIZE bytes) but auto-ingest in progress — preserving"
        fi
    else
        echo "[entrypoint] ChromaDB data OK (size=$DB_SIZE bytes)"
    fi
fi

# =============================================================================
# Phase 2: Start the Application
# =============================================================================
# Single worker despite 2 GB RAM — EasyOCR loads PyTorch models per-worker
# (~600 MB each). Two workers would consume ~1.2 GB for models alone, leaving
# <800 MB for ChromaDB HNSW index + request headroom. FastAPI's async event
# loop handles concurrency within 1 worker.
echo "[entrypoint] Starting uvicorn on port ${PORT:-8000}..."
exec python3 -m uvicorn src.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 1 \
    --timeout-keep-alive 65

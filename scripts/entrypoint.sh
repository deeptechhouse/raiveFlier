#!/bin/bash
# =============================================================================
# scripts/entrypoint.sh — Docker Entrypoint for raiveFlier
# =============================================================================
#
# This script is the CMD for the Docker container. It runs two phases:
#
#   Phase 1: CORPUS DOWNLOAD (optional, non-blocking)
#     Automatically fetches the LATEST corpus release from GitHub.
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
#     Starts FastAPI via uvicorn (single worker, 512 MB RAM constraint).
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
        else
            echo "[entrypoint] Corpus DB too small ($DB_SIZE bytes < ${MIN_CORPUS_SIZE}) — downloading..."
        fi
    else
        echo "[entrypoint] No corpus found at $CHROMADB_DIR — downloading..."
    fi

    # Step 3: Parse the asset ID for download.
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
        return 0
    fi

    # Step 4: Download and extract the corpus tarball.
    echo "[entrypoint] Downloading corpus $LATEST_TAG (asset $ASSET_ID)..."
    mkdir -p "$CHROMADB_DIR"

    PARENT_DIR=$(dirname "$CHROMADB_DIR")
    MAX_RETRIES=3
    RETRY_DELAY=5
    DOWNLOAD_OK=false

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
        else
            echo "[entrypoint] Attempt $attempt failed"
            if [ "$attempt" -lt "$MAX_RETRIES" ]; then
                echo "[entrypoint] Retrying in ${RETRY_DELAY}s..."
                sleep "$RETRY_DELAY"
            fi
        fi
    done

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
        echo "[entrypoint] WARNING: Corpus download failed after $MAX_RETRIES attempts"
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
        echo "[entrypoint] Clearing stale ChromaDB data ($DB_SIZE bytes < 50 MB threshold)"
        rm -rf "${CHROMADB_DIR:?}"/*
        echo "[entrypoint] Cleared — fresh corpus will be built by background ingestion"
    else
        echo "[entrypoint] ChromaDB data OK (size=$DB_SIZE bytes)"
    fi
fi

# =============================================================================
# Phase 2: Start the Application
# =============================================================================
echo "[entrypoint] Starting uvicorn on port ${PORT:-8000}..."
exec python3 -m uvicorn src.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 1 \
    --timeout-keep-alive 65

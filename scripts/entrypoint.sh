#!/bin/bash
# ──────────────────────────────────────────────────────────────
# raiveFlier — Docker entrypoint
# ──────────────────────────────────────────────────────────────
# Downloads pre-built ChromaDB corpus from a private GitHub
# release if the local corpus is empty.  Then starts uvicorn.
# ──────────────────────────────────────────────────────────────

CHROMADB_DIR="${CHROMADB_PERSIST_DIR:-/data/chromadb}"
CORPUS_REPO="${CORPUS_REPO:-deeptechhouse/raiveflier-corpus}"
CORPUS_TAG="${CORPUS_TAG:-v1.0.0}"

# ── Download corpus if empty ──────────────────────────────────
download_corpus() {
    if [ -z "$GITHUB_TOKEN" ]; then
        echo "[entrypoint] No GITHUB_TOKEN set — skipping corpus download"
        return 0
    fi

    if [ "$RAG_ENABLED" != "true" ]; then
        echo "[entrypoint] RAG_ENABLED is not true — skipping corpus download"
        return 0
    fi

    # Check if ChromaDB already has data
    DB_FILE="$CHROMADB_DIR/chroma.sqlite3"
    if [ -f "$DB_FILE" ]; then
        DB_SIZE=$(wc -c < "$DB_FILE" 2>/dev/null || echo "0")
        if [ "$DB_SIZE" -gt 100000 ]; then
            echo "[entrypoint] Corpus already present at $CHROMADB_DIR ($DB_SIZE bytes)"
            return 0
        fi
        echo "[entrypoint] Corpus DB too small ($DB_SIZE bytes) — re-downloading..."
    else
        echo "[entrypoint] No corpus found at $CHROMADB_DIR — downloading..."
    fi

    # Get the asset ID from the release
    echo "[entrypoint] Fetching release info from $CORPUS_REPO/$CORPUS_TAG..."
    RELEASE_JSON=$(curl -sL \
        -H "Authorization: token $GITHUB_TOKEN" \
        -H "Accept: application/vnd.github+json" \
        "https://api.github.com/repos/$CORPUS_REPO/releases/tags/$CORPUS_TAG")

    ASSET_ID=$(echo "$RELEASE_JSON" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    assets = data.get('assets', [])
    if assets:
        print(assets[0]['id'])
    else:
        print('')
except Exception as e:
    print('', file=sys.stderr)
    print(f'JSON parse error: {e}', file=sys.stderr)
" 2>&1)

    if [ -z "$ASSET_ID" ] || [ "$ASSET_ID" = "None" ]; then
        echo "[entrypoint] WARNING: Could not find corpus release asset"
        echo "[entrypoint] API response: $(echo "$RELEASE_JSON" | head -c 500)"
        return 0
    fi

    echo "[entrypoint] Downloading corpus (asset $ASSET_ID, ~154 MB)..."
    mkdir -p "$CHROMADB_DIR"

    # Download and extract — the tarball contains chromadb/ as top dir
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

    if [ "$DOWNLOAD_OK" = true ]; then
        # Verify extraction
        if [ -f "$CHROMADB_DIR/chroma.sqlite3" ]; then
            FINAL_SIZE=$(wc -c < "$CHROMADB_DIR/chroma.sqlite3")
            echo "[entrypoint] Verified: chroma.sqlite3 is $FINAL_SIZE bytes"
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

# Run corpus download (errors here should not prevent app startup)
download_corpus || echo "[entrypoint] Corpus download encountered an error — continuing..."

# ── Start the application ────────────────────────────────────
echo "[entrypoint] Starting uvicorn on port ${PORT:-8000}..."
exec python3 -m uvicorn src.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 1 \
    --timeout-keep-alive 65

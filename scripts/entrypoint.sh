#!/bin/bash
# ──────────────────────────────────────────────────────────────
# raiveFlier — Docker entrypoint
# ──────────────────────────────────────────────────────────────
# Downloads pre-built ChromaDB corpus from a private GitHub
# release if the local corpus is empty.  Then starts uvicorn.
# ──────────────────────────────────────────────────────────────

set -e

CHROMADB_DIR="${CHROMADB_PERSIST_DIR:-/data/chromadb}"
CORPUS_REPO="${CORPUS_REPO:-deeptechhouse/raiveflier-corpus}"
CORPUS_TAG="${CORPUS_TAG:-v1.0.0}"

# ── Download corpus if empty ──────────────────────────────────
if [ -n "$GITHUB_TOKEN" ] && [ "$RAG_ENABLED" = "true" ]; then
    # Check if ChromaDB has data (look for chroma.sqlite3 with non-trivial size)
    DB_FILE="$CHROMADB_DIR/chroma.sqlite3"
    NEEDS_DOWNLOAD=false

    if [ ! -f "$DB_FILE" ]; then
        NEEDS_DOWNLOAD=true
        echo "[entrypoint] No corpus found at $CHROMADB_DIR — downloading..."
    elif [ "$(stat -c%s "$DB_FILE" 2>/dev/null || stat -f%z "$DB_FILE" 2>/dev/null)" -lt 100000 ]; then
        NEEDS_DOWNLOAD=true
        echo "[entrypoint] Corpus DB too small — re-downloading..."
    fi

    if [ "$NEEDS_DOWNLOAD" = true ]; then
        # Get the asset ID from the release
        ASSET_ID=$(curl -sL \
            -H "Authorization: token $GITHUB_TOKEN" \
            -H "Accept: application/vnd.github+json" \
            "https://api.github.com/repos/$CORPUS_REPO/releases/tags/$CORPUS_TAG" \
            | python -c "import sys,json; assets=json.load(sys.stdin).get('assets',[]); print(assets[0]['id'] if assets else '')" 2>/dev/null)

        if [ -n "$ASSET_ID" ]; then
            echo "[entrypoint] Downloading corpus (asset $ASSET_ID)..."
            mkdir -p "$CHROMADB_DIR"
            curl -sL \
                -H "Authorization: token $GITHUB_TOKEN" \
                -H "Accept: application/octet-stream" \
                "https://api.github.com/repos/$CORPUS_REPO/releases/assets/$ASSET_ID" \
                | tar xz -C "$(dirname "$CHROMADB_DIR")"
            echo "[entrypoint] Corpus downloaded and extracted to $CHROMADB_DIR"
        else
            echo "[entrypoint] WARNING: Could not find corpus release asset"
        fi
    else
        echo "[entrypoint] Corpus already present at $CHROMADB_DIR"
    fi
else
    if [ -z "$GITHUB_TOKEN" ]; then
        echo "[entrypoint] No GITHUB_TOKEN set — skipping corpus download"
    fi
fi

# ── Start the application ────────────────────────────────────
exec python -m uvicorn src.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 1 \
    --timeout-keep-alive 65

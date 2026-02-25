#!/bin/bash
# =============================================================================
# scripts/package_corpus.sh — Package ChromaDB Corpus for GitHub Release
# =============================================================================
#
# Creates a tarball of the local ChromaDB directory in the format that
# entrypoint.sh's download_corpus() function expects.  The tarball must
# contain a "chromadb/" directory at its root, because entrypoint.sh
# extracts with `tar xz -C /data` and expects the result at /data/chromadb/.
#
# Usage:
#   ./scripts/package_corpus.sh                      # Build tarball only
#   ./scripts/package_corpus.sh --upload v1.0.0      # Build + upload to release
#   ./scripts/package_corpus.sh --upload v1.0.1      # Build + upload to new tag
#
# Environment Variables:
#   CHROMADB_PERSIST_DIR — Local ChromaDB directory (default: ./data/chromadb)
#   CORPUS_REPO          — GitHub repo (default: deeptechhouse/raiveflier-corpus)
#
# Prerequisites:
#   - gh CLI installed and authenticated (for --upload)
#   - A built ChromaDB corpus at CHROMADB_PERSIST_DIR with chroma.sqlite3
# =============================================================================

set -euo pipefail

CHROMADB_DIR="${CHROMADB_PERSIST_DIR:-./data/chromadb}"
OUTPUT="chromadb_corpus.tar.gz"
REPO="${CORPUS_REPO:-deeptechhouse/raiveflier-corpus}"

# ── Validate corpus exists and is substantial ───────────────
if [ ! -f "$CHROMADB_DIR/chroma.sqlite3" ]; then
    echo "ERROR: $CHROMADB_DIR/chroma.sqlite3 not found"
    echo "Run the ingestion pipeline first to build the corpus."
    exit 1
fi

DB_SIZE=$(wc -c < "$CHROMADB_DIR/chroma.sqlite3")
echo "Corpus DB size: $DB_SIZE bytes"

if [ "$DB_SIZE" -lt 50000000 ]; then
    echo "WARNING: Corpus is smaller than 50 MB — this may be only the reference set."
    echo "Consider running a full corpus rebuild before packaging."
fi

# ── Create tarball with chromadb/ at root ───────────────────
# The -C flag changes to the parent directory so the tarball contains
# "chromadb/" at its root (matching what entrypoint.sh expects when
# extracting with `tar xz -C /data`).
PARENT_DIR=$(dirname "$CHROMADB_DIR")
DIR_NAME=$(basename "$CHROMADB_DIR")

echo "Packaging $CHROMADB_DIR as $OUTPUT..."
tar czf "$OUTPUT" -C "$PARENT_DIR" "$DIR_NAME"

TARBALL_SIZE=$(wc -c < "$OUTPUT")
echo "Created $OUTPUT ($TARBALL_SIZE bytes)"

# ── Optional upload to GitHub release ───────────────────────
if [ "${1:-}" = "--upload" ] && [ -n "${2:-}" ]; then
    TAG="$2"
    echo "Uploading $OUTPUT to $REPO release $TAG..."

    # Create the release if it doesn't exist yet
    if ! gh release view "$TAG" --repo "$REPO" > /dev/null 2>&1; then
        echo "Creating release $TAG..."
        gh release create "$TAG" \
            --repo "$REPO" \
            --title "ChromaDB Corpus $TAG" \
            --notes "Pre-built ChromaDB corpus for raiveFlier deployment."
    fi

    # Upload (--clobber replaces existing asset with same name)
    gh release upload "$TAG" "$OUTPUT" --repo "$REPO" --clobber
    echo "Upload complete: $REPO@$TAG"
elif [ "${1:-}" = "--upload" ]; then
    echo "ERROR: --upload requires a tag argument (e.g. --upload v1.0.0)"
    exit 1
fi

#!/bin/bash
# =============================================================================
# scripts/package_corpus.sh — Package ChromaDB Corpus for GitHub Release
# =============================================================================
#
# Creates a tarball of the local ChromaDB directory and optionally uploads it
# to a GitHub release.  Handles corpora that exceed GitHub's 2 GB per-asset
# limit by splitting the tarball into parts and uploading each as a separate
# release asset.
#
# The tarball contains a "chromadb/" directory at its root, matching what
# entrypoint.sh's download_corpus() expects when extracting with:
#   tar xz -C /data  →  /data/chromadb/
#
# Usage:
#   ./scripts/package_corpus.sh                      # Build tarball only
#   ./scripts/package_corpus.sh --upload v2.0.0      # Build + upload to release
#
# Environment Variables:
#   CHROMADB_PERSIST_DIR — Local ChromaDB directory (default: ./data/chromadb)
#   CORPUS_REPO          — GitHub repo (default: deeptechhouse/raiveflier-corpus)
#   SPLIT_SIZE_MB        — Max size per part in MB (default: 1800)
#
# Prerequisites:
#   - gh CLI installed and authenticated (for --upload)
#   - A built ChromaDB corpus at CHROMADB_PERSIST_DIR with chroma.sqlite3
# =============================================================================

set -euo pipefail

CHROMADB_DIR="${CHROMADB_PERSIST_DIR:-./data/chromadb}"
OUTPUT="chromadb_corpus.tar.gz"
REPO="${CORPUS_REPO:-deeptechhouse/raiveflier-corpus}"
# GitHub's per-asset limit is 2 GB. Use 1800 MB to stay safely under.
SPLIT_SIZE_MB="${SPLIT_SIZE_MB:-1800}"

# ── Validate corpus exists and is substantial ───────────────
if [ ! -f "$CHROMADB_DIR/chroma.sqlite3" ]; then
    echo "ERROR: $CHROMADB_DIR/chroma.sqlite3 not found"
    echo "Run the ingestion pipeline first to build the corpus."
    exit 1
fi

DB_SIZE=$(wc -c < "$CHROMADB_DIR/chroma.sqlite3")
echo "Corpus DB size: $DB_SIZE bytes ($(echo "$DB_SIZE" | awk '{printf "%.1f GB", $1/1073741824}'))"

if [ "$DB_SIZE" -lt 50000000 ]; then
    echo "WARNING: Corpus is smaller than 50 MB — this may be only the reference set."
    echo "Consider running a full corpus rebuild before packaging."
fi

# ── Create tarball with chromadb/ at root ───────────────────
PARENT_DIR=$(dirname "$CHROMADB_DIR")
DIR_NAME=$(basename "$CHROMADB_DIR")

echo "Packaging $CHROMADB_DIR as $OUTPUT..."
tar czf "$OUTPUT" -C "$PARENT_DIR" "$DIR_NAME"

TARBALL_SIZE=$(wc -c < "$OUTPUT")
TARBALL_MB=$((TARBALL_SIZE / 1048576))
echo "Created $OUTPUT ($TARBALL_MB MB)"

# ── Split if larger than SPLIT_SIZE_MB ──────────────────────
# GitHub releases have a 2 GB per-asset limit.  If the tarball exceeds
# SPLIT_SIZE_MB, split it into sequential parts that can be concatenated
# back together on the download side (cat part_* | tar xz).
SPLIT_PREFIX="chromadb_corpus_part_"
NEEDS_SPLIT=false

if [ "$TARBALL_MB" -gt "$SPLIT_SIZE_MB" ]; then
    NEEDS_SPLIT=true
    echo "Tarball exceeds ${SPLIT_SIZE_MB} MB — splitting into parts..."
    split -b "${SPLIT_SIZE_MB}m" "$OUTPUT" "$SPLIT_PREFIX"

    # List the created parts.
    PART_COUNT=0
    for part in ${SPLIT_PREFIX}*; do
        PART_SIZE=$(wc -c < "$part")
        echo "  $part: $(echo "$PART_SIZE" | awk '{printf "%.0f MB", $1/1048576}')"
        PART_COUNT=$((PART_COUNT + 1))
    done
    echo "Split into $PART_COUNT parts"

    # Clean up the monolithic tarball — only the parts are needed.
    rm "$OUTPUT"
else
    echo "Tarball fits in a single asset (under ${SPLIT_SIZE_MB} MB)"
fi

# ── Optional upload to GitHub release ───────────────────────
if [ "${1:-}" = "--upload" ] && [ -n "${2:-}" ]; then
    TAG="$2"
    echo ""
    echo "Uploading to $REPO release $TAG..."

    # Create the release if it doesn't exist yet.
    if ! gh release view "$TAG" --repo "$REPO" > /dev/null 2>&1; then
        echo "Creating release $TAG..."
        gh release create "$TAG" \
            --repo "$REPO" \
            --title "ChromaDB Corpus $TAG" \
            --notes "Pre-built ChromaDB corpus for raiveFlier deployment.
Chunks: ~486K | Sources: ~3K | Size: ${TARBALL_MB} MB compressed

$(if [ "$NEEDS_SPLIT" = true ]; then echo "Multi-part archive: download ALL parts and concatenate before extracting."; fi)"
    fi

    if [ "$NEEDS_SPLIT" = true ]; then
        # Upload each split part as a separate release asset.
        for part in ${SPLIT_PREFIX}*; do
            echo "Uploading $part..."
            gh release upload "$TAG" "$part" --repo "$REPO" --clobber
        done
        # Clean up local split files after successful upload.
        rm -f ${SPLIT_PREFIX}*
    else
        # Single-file upload.
        gh release upload "$TAG" "$OUTPUT" --repo "$REPO" --clobber
        rm -f "$OUTPUT"
    fi

    echo "Upload complete: $REPO@$TAG"

elif [ "${1:-}" = "--upload" ]; then
    echo "ERROR: --upload requires a tag argument (e.g. --upload v2.0.0)"
    exit 1
else
    echo ""
    echo "To upload, run:"
    if [ "$NEEDS_SPLIT" = true ]; then
        echo "  ./scripts/package_corpus.sh --upload v2.0.0"
    else
        echo "  ./scripts/package_corpus.sh --upload v2.0.0"
    fi
fi

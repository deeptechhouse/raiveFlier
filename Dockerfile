# =============================================================================
# Dockerfile — Production Docker Build for raiveFlier
# =============================================================================
#
# This Dockerfile builds the production image for the raiveFlier application.
# It targets Render.com's Standard plan ($25/mo) with these resources:
#
#   - 2 GB RAM limit (Render Standard)
#   - 20 GB persistent disk mounted at /data (for ChromaDB corpus + SQLite)
#   - No GPU access (all ML inference is CPU-only)
#
# Key Design Decisions:
#
#   1. EasyOCR is INCLUDED — uses CPU-only PyTorch (~300 MB) which fits
#      within the 2 GB RAM budget. The OCR fallback chain is:
#      LLM Vision (API) -> EasyOCR (local ML) -> Tesseract (local binary).
#
#   2. sentence-transformers is INCLUDED (shared PyTorch dep with EasyOCR).
#      FastEmbed remains the default embedding provider; SentenceTransformers
#      is available as a fallback via the existing import guard chain.
#
#   3. CPU-only PyTorch is installed via --extra-index-url to keep the image
#      lean (~300 MB vs ~2 GB for full CUDA PyTorch).
#
#   4. The reference corpus (curated text files about rave history) is baked
#      into the image under data/reference_corpus/. On first boot, the
#      entrypoint script checks if ChromaDB has a full corpus and downloads
#      a pre-built one from a private GitHub release if needed.
#
# Build:  docker build -t raiveflier .
# Run:    docker run -p 8000:8000 --env-file .env raiveflier
# =============================================================================

FROM python:3.12-slim AS base

# PYTHONDONTWRITEBYTECODE=1 — Prevents .pyc bytecode files (saves disk I/O)
# PYTHONUNBUFFERED=1 — Forces stdout/stderr to be unbuffered (logs appear immediately)
# ANONYMIZED_TELEMETRY=False — Disables ChromaDB PostHog telemetry to prevent
#   "capture() takes 1 positional argument but 3 were given" errors caused by
#   a version mismatch between ChromaDB's bundled PostHog client and the server.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ANONYMIZED_TELEMETRY=False

# ── System dependencies ──────────────────────────────────────
# These are native libraries required by the application and its Python deps:
#   - tesseract-ocr/eng: Fallback OCR engine for text extraction from images
#   - libgl1, libglib2.0-0, libsm6, libxext6, libxrender-dev: OpenCV dependencies
#     (used by image processing code even though we do not run EasyOCR)
#   - curl: Used by the health check and entrypoint corpus download
# The rm -rf /var/lib/apt/lists/* cleans up apt cache to reduce image size.
# build-essential provides g++ required by chroma-hnswlib (ChromaDB dep)
# which compiles from source when no pre-built wheel exists for this platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        tesseract-ocr \
        tesseract-ocr-eng \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ──────────────────────────────────────
# Copy requirements.txt first (before application code) to leverage Docker's
# layer caching. If only application code changes, this layer is cached and
# pip install is skipped, dramatically speeding up rebuilds.
COPY requirements.txt .

# Install all Python dependencies including EasyOCR and sentence-transformers.
# --extra-index-url pulls CPU-only PyTorch (~300 MB vs ~2 GB full CUDA build)
# since Render has no GPU. This single flag unblocks both EasyOCR and
# sentence-transformers which share PyTorch as a dependency.
# --no-cache-dir prevents pip from storing downloaded wheels (saves ~100 MB).
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# ── Application code ─────────────────────────────────────────
# Each COPY creates a separate layer. Order matters for cache efficiency:
# source code changes most often, so it goes last.
# Python application code
COPY src/ src/
# raiveFeeder companion app (mounted at /feeder/ in production)
COPY tools/ tools/
# Static HTML/CSS/JS frontend
COPY frontend/ frontend/
# YAML configuration files
COPY config/ config/
# Curated RAG corpus text files
COPY data/reference_corpus/ data/reference_corpus/
# RA Exchange interview transcripts (489 episodes, ~25 MB)
COPY transcripts/ra_exchange/ transcripts/ra_exchange/
# User-provided book text files (auto-ingested as tier 1)
COPY data/books/ data/books/
# Docker entrypoint script
COPY scripts/entrypoint.sh scripts/entrypoint.sh

# Create writable directories for runtime data.
# On Render, /data is a persistent disk mount (survives redeploys).
# The local data/chromadb and uploads/ dirs are fallbacks for local Docker
# runs where no external volume is mounted.
# /data/pending_uploads stages files awaiting content approval.
RUN mkdir -p data/chromadb /data/chromadb /data/pending_uploads uploads \
    && chmod +x scripts/entrypoint.sh

# ── Runtime ──────────────────────────────────────────────────
# Expose the default application port. Render injects its own $PORT env var
# at runtime, which the entrypoint script reads.
EXPOSE 8000

# Default port — Render overrides this via the PORT environment variable.
ENV PORT=8000

# Docker health check: pings the /api/v1/health endpoint every 30 seconds.
# If 3 consecutive checks fail, Docker marks the container as unhealthy.
# --start-period=15s gives the app time to boot before checks begin.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/v1/health || exit 1

# Use the entrypoint script as the container's main process.
# It downloads the pre-built ChromaDB corpus (if needed) and then exec's
# into uvicorn, which replaces the shell process (exec ensures PID 1
# signal handling works correctly for graceful shutdown).
CMD ["/bin/bash", "/app/scripts/entrypoint.sh"]

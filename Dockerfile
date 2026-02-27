# =============================================================================
# Dockerfile — Production Docker Build for raiveFlier
# =============================================================================
#
# This Dockerfile builds the production image for the raiveFlier application.
# It is optimized for Render.com's Starter plan ($7/mo) with strict constraints:
#
#   - 512 MB RAM limit (Render Starter)
#   - 1 GB persistent disk mounted at /data (for ChromaDB corpus + SQLite)
#   - No GPU access (all ML inference is API-based, not local)
#
# Key Design Decisions:
#
#   1. EasyOCR is EXCLUDED because it depends on PyTorch (~561 MB), which
#      alone would exceed the 512 MB RAM budget. Instead, the OCR fallback
#      chain is: LLM Vision (API) -> Tesseract (local binary, ~20 MB).
#
#   2. sentence-transformers is EXCLUDED for the same reason (PyTorch dep).
#      The fastembed library provides equivalent embedding via ONNX Runtime,
#      which runs within the memory budget.
#
#   3. Requirements are filtered at build time using grep to strip excluded
#      packages, keeping a single requirements.txt source of truth.
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
RUN apt-get update && apt-get install -y --no-install-recommends \
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

# Filter out heavy packages that exceed the 512 MB RAM budget:
#   - easyocr: Pulls PyTorch (~561 MB) — replaced by LLM Vision API + Tesseract
#   - sentence-transformers: Also pulls PyTorch — replaced by fastembed (ONNX)
# The grep -ivE does case-insensitive, extended regex exclusion.
# --no-cache-dir prevents pip from storing downloaded wheels (saves ~100 MB).
RUN grep -ivE 'easyocr|sentence.transformers' requirements.txt > requirements-deploy.txt \
    && pip install --no-cache-dir -r requirements-deploy.txt \
    && rm requirements-deploy.txt

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

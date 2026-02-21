# ──────────────────────────────────────────────────────────────
# RaiveFlier — Production Dockerfile (Render / any Docker host)
# ──────────────────────────────────────────────────────────────
# Targets: Render Starter ($7/mo) with persistent disk at /data.
# EasyOCR is excluded to avoid pulling PyTorch (~2 GB).
# Primary OCR: LLM Vision (API-based).  Fallback: Tesseract.
# Reference corpus is auto-ingested into ChromaDB on first boot.
# ──────────────────────────────────────────────────────────────

FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── System dependencies ──────────────────────────────────────
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
# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install everything EXCEPT easyocr (too heavy for <=512 MB RAM)
RUN grep -iv 'easyocr' requirements.txt > requirements-deploy.txt \
    && pip install --no-cache-dir -r requirements-deploy.txt \
    && rm requirements-deploy.txt

# ── Application code ─────────────────────────────────────────
COPY src/ src/
COPY frontend/ frontend/
COPY config/ config/
COPY data/reference_corpus/ data/reference_corpus/

# Create writable directories for runtime data
# On Render, /data is a persistent disk mount — these are fallbacks
# for local Docker runs where no volume is mounted.
RUN mkdir -p data/chromadb /data/chromadb uploads

# ── Runtime ──────────────────────────────────────────────────
EXPOSE 8000

# Default port (Render overrides via $PORT env var)
ENV PORT=8000

# Health-check against the /api/v1/health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/v1/health || exit 1

# Run with uvicorn — Render sets PORT env var automatically
CMD python -m uvicorn src.main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers 1 \
    --timeout-keep-alive 65

"""raiveFeeder configuration settings.

# ─── HOW FEEDER SETTINGS WORK ─────────────────────────────────────────
#
# FeederSettings extends the base raiveFlier Settings with feeder-specific
# fields (port, transcription model, crawl limits, etc.).  All base
# settings (API keys, ChromaDB paths, embedding config) are inherited
# so raiveFeeder uses the same providers and vector store as raiveFlier.
#
# Environment variables and .env override defaults just like the base.
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from src.config.settings import Settings


class FeederSettings(Settings):
    """raiveFeeder application settings.

    Inherits all raiveFlier settings (API keys, ChromaDB config, embedding
    providers, etc.) and adds feeder-specific configuration for the
    ingestion GUI, audio transcription, and web crawling.
    """

    # === Feeder App ===
    # Port 8001 avoids collision with raiveFlier on 8000.
    feeder_port: int = 8001
    feeder_host: str = "0.0.0.0"

    # === Audio Transcription ===
    # Whisper model size for faster-whisper local transcription.
    # Options: tiny, base, small, medium, large-v2, large-v3.
    # Larger models = better accuracy but more RAM/VRAM.
    whisper_model_size: str = "base"
    # Maximum audio duration in seconds (30 min default).
    max_audio_duration_seconds: int = 1800

    # === Web Crawling ===
    # Defaults for the URL scraping tab.  Users can override per-job in the UI.
    crawl_max_depth: int = 2
    crawl_max_pages: int = 20
    # Seconds between requests to the same domain (politeness).
    crawl_rate_limit_seconds: float = 1.0
    # Minimum LLM relevance score (0-10) for a page to be ingested.
    crawl_relevance_threshold: float = 5.0

    # === Batch Processing ===
    # Maximum number of concurrent ingestion jobs.
    batch_max_concurrency: int = 2
    # Maximum number of items in a single batch.
    batch_max_items: int = 100

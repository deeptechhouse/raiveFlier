"""Utility modules for RaiveFlier.

Available utility modules (all re-exported here for convenience):

- **confidence** -- Weighted scoring math and human-readable level mapping
  used throughout the pipeline to rate extraction and research quality.
- **errors** -- Domain-specific exception hierarchy rooted at RaiveFlierError;
  each pipeline stage raises its own subclass so callers can handle failures
  granularly without broad ``except Exception`` blocks.
- **image_preprocessor** -- Multi-pass OpenCV/PIL image preprocessing that
  generates 7+ variants of a flier image (inverted, color channels, CLAHE,
  denoised, saturation, Otsu) to maximise OCR text recovery.
- **concurrency** -- asyncio semaphore throttling and fan-out helpers that
  keep parallel web-search and LLM calls under provider rate limits.
- **logging** -- structlog setup with a dual-renderer pattern: coloured
  console output in development, structured JSON in production.
- **text_normalizer** -- DJ/artist name normalization, OCR error correction
  for rave-flier typography, and transcript cleaning for RAG ingestion.
- **ocr_helpers** (not re-exported here) -- Cross-pass fuzzy deduplication
  and result merging shared by the EasyOCR and Tesseract providers.
"""

# -- Confidence scoring utilities ------------------------------------------
from src.utils.confidence import (
    ConfidenceLevel,
    calculate_confidence,
    confidence_to_level,
    merge_confidence,
)

# -- Domain exception hierarchy --------------------------------------------
from src.utils.errors import (
    ConfigurationError,
    EntityExtractionError,
    OCRExtractionError,
    PipelineError,
    ProviderUnavailableError,
    RaiveFlierError,
    RateLimitError,
    ResearchError,
)

# -- Image preprocessing for OCR -------------------------------------------
from src.utils.image_preprocessor import ImagePreprocessor

# -- Async concurrency helpers ---------------------------------------------
from src.utils.concurrency import parallel_search, throttled_gather

# -- Structured logging setup ----------------------------------------------
from src.utils.logging import configure_logging, get_logger

# -- Text normalization (artist names, OCR corrections, transcript prep) ---
from src.utils.text_normalizer import fuzzy_match, normalize_artist_name, split_artist_names

__all__ = [
    "ConfidenceLevel",
    "ConfigurationError",
    "EntityExtractionError",
    "ImagePreprocessor",
    "OCRExtractionError",
    "PipelineError",
    "ProviderUnavailableError",
    "RaiveFlierError",
    "RateLimitError",
    "ResearchError",
    "calculate_confidence",
    "confidence_to_level",
    "configure_logging",
    "fuzzy_match",
    "get_logger",
    "merge_confidence",
    "normalize_artist_name",
    "parallel_search",
    "split_artist_names",
    "throttled_gather",
]

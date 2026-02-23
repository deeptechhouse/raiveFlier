"""Utility modules for RaiveFlier."""

from src.utils.confidence import (
    ConfidenceLevel,
    calculate_confidence,
    confidence_to_level,
    merge_confidence,
)
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
from src.utils.image_preprocessor import ImagePreprocessor
from src.utils.concurrency import parallel_search, throttled_gather
from src.utils.logging import configure_logging, get_logger
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

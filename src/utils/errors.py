"""Custom exception hierarchy for RaiveFlier.

All application exceptions inherit from :class:`RaiveFlierError`, which
carries an optional ``provider_name`` so error handlers can identify which
external service (e.g. "openai", "tesseract", "qdrant") caused the failure.

The hierarchy is organized by pipeline domain:

    RaiveFlierError  (base -- catch-all for any raiveFlier error)
    +-- OCRExtractionError       (Phase 1: image-to-text extraction)
    +-- EntityExtractionError    (Phase 2: LLM entity parsing)
    +-- ResearchError            (Phase 3: web research)
    +-- PipelineError            (orchestration / phase transitions)
    +-- ConfigurationError       (startup / missing config)
    +-- LLMError                 (any LLM API call failure)
    +-- RateLimitError           (provider rate-limit exceeded)
    +-- ProviderUnavailableError (external service down / unreachable)
    +-- RAGError                 (embedding or vector-store failure)

This granular hierarchy lets callers handle errors at exactly the right
level -- e.g. retry on RateLimitError, fall back to another provider on
ProviderUnavailableError, or abort on ConfigurationError.
"""


class RaiveFlierError(Exception):
    """Base exception for all RaiveFlier errors.

    Every subclass carries a human-readable ``message`` and an optional
    ``provider_name`` identifying which external service triggered the
    error.  The ``__str__`` method prefixes the provider name in brackets
    for structured log output, e.g. ``[openai] Rate limit exceeded``.
    """

    def __init__(
        self,
        message: str = "An unexpected error occurred",
        provider_name: str | None = None,
    ) -> None:
        # Use private attributes with @property accessors to enforce
        # read-only access (data hiding / encapsulation).
        self._message = message
        self._provider_name = provider_name
        super().__init__(self._message)

    @property
    def message(self) -> str:
        return self._message

    @property
    def provider_name(self) -> str | None:
        return self._provider_name

    def __str__(self) -> str:
        # Prefix the provider name for easier log scanning.
        if self._provider_name:
            return f"[{self._provider_name}] {self._message}"
        return self._message


# ---------------------------------------------------------------------------
# Phase 1 & 2: Extraction errors
# ---------------------------------------------------------------------------

class OCRExtractionError(RaiveFlierError):
    """Raised when OCR text extraction fails (Tesseract, EasyOCR, LLM Vision)."""

    def __init__(
        self,
        message: str = "OCR text extraction failed",
        provider_name: str | None = None,
    ) -> None:
        super().__init__(message=message, provider_name=provider_name)


class EntityExtractionError(RaiveFlierError):
    """Raised when entity parsing from OCR text fails (LLM entity extraction)."""

    def __init__(
        self,
        message: str = "Entity extraction failed",
        provider_name: str | None = None,
    ) -> None:
        super().__init__(message=message, provider_name=provider_name)


# ---------------------------------------------------------------------------
# Phase 3: Research errors
# ---------------------------------------------------------------------------

class ResearchError(RaiveFlierError):
    """Raised when the research phase encounters a failure (web search, scraping)."""

    def __init__(
        self,
        message: str = "Research phase failed",
        provider_name: str | None = None,
    ) -> None:
        super().__init__(message=message, provider_name=provider_name)


# ---------------------------------------------------------------------------
# External service / provider errors
# ---------------------------------------------------------------------------

class ProviderUnavailableError(RaiveFlierError):
    """Raised when an external service or provider is unreachable.

    The pipeline's fallback logic catches this to try the next provider
    in the configured priority order.
    """

    def __init__(
        self,
        message: str = "External service is unavailable",
        provider_name: str | None = None,
    ) -> None:
        super().__init__(message=message, provider_name=provider_name)


class RateLimitError(RaiveFlierError):
    """Raised when an API rate limit is exceeded.

    Callers should implement exponential backoff or switch to a
    secondary provider when this is caught.
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        provider_name: str | None = None,
    ) -> None:
        super().__init__(message=message, provider_name=provider_name)


class LLMError(RaiveFlierError):
    """Raised when an LLM API call fails or returns an unparseable response."""

    def __init__(
        self,
        message: str = "LLM API call failed",
        provider_name: str | None = None,
    ) -> None:
        super().__init__(message=message, provider_name=provider_name)


# ---------------------------------------------------------------------------
# Orchestration / configuration errors
# ---------------------------------------------------------------------------

class PipelineError(RaiveFlierError):
    """Raised when pipeline orchestration fails (invalid state transition, etc.)."""

    def __init__(
        self,
        message: str = "Pipeline orchestration failed",
        provider_name: str | None = None,
    ) -> None:
        super().__init__(message=message, provider_name=provider_name)


class ConfigurationError(RaiveFlierError):
    """Raised when configuration is invalid or missing at startup."""

    def __init__(
        self,
        message: str = "Invalid or missing configuration",
        provider_name: str | None = None,
    ) -> None:
        super().__init__(message=message, provider_name=provider_name)


# ---------------------------------------------------------------------------
# RAG / vector-store errors
# ---------------------------------------------------------------------------

class RAGError(RaiveFlierError):
    """Raised when a RAG pipeline operation fails (embedding or vector store)."""

    def __init__(
        self,
        message: str = "RAG pipeline operation failed",
        provider_name: str | None = None,
    ) -> None:
        super().__init__(message=message, provider_name=provider_name)

"""Custom exception hierarchy for RaiveFlier."""


class RaiveFlierError(Exception):
    """Base exception for all RaiveFlier errors."""

    def __init__(
        self,
        message: str = "An unexpected error occurred",
        provider_name: str | None = None,
    ) -> None:
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
        if self._provider_name:
            return f"[{self._provider_name}] {self._message}"
        return self._message


class OCRExtractionError(RaiveFlierError):
    """Raised when OCR text extraction fails."""

    def __init__(
        self,
        message: str = "OCR text extraction failed",
        provider_name: str | None = None,
    ) -> None:
        super().__init__(message=message, provider_name=provider_name)


class EntityExtractionError(RaiveFlierError):
    """Raised when entity parsing from OCR text fails."""

    def __init__(
        self,
        message: str = "Entity extraction failed",
        provider_name: str | None = None,
    ) -> None:
        super().__init__(message=message, provider_name=provider_name)


class ProviderUnavailableError(RaiveFlierError):
    """Raised when an external service or provider is unavailable."""

    def __init__(
        self,
        message: str = "External service is unavailable",
        provider_name: str | None = None,
    ) -> None:
        super().__init__(message=message, provider_name=provider_name)


class ResearchError(RaiveFlierError):
    """Raised when the research phase encounters a failure."""

    def __init__(
        self,
        message: str = "Research phase failed",
        provider_name: str | None = None,
    ) -> None:
        super().__init__(message=message, provider_name=provider_name)


class PipelineError(RaiveFlierError):
    """Raised when pipeline orchestration fails."""

    def __init__(
        self,
        message: str = "Pipeline orchestration failed",
        provider_name: str | None = None,
    ) -> None:
        super().__init__(message=message, provider_name=provider_name)


class ConfigurationError(RaiveFlierError):
    """Raised when configuration is invalid or missing."""

    def __init__(
        self,
        message: str = "Invalid or missing configuration",
        provider_name: str | None = None,
    ) -> None:
        super().__init__(message=message, provider_name=provider_name)


class LLMError(RaiveFlierError):
    """Raised when an LLM API call fails or returns an invalid response."""

    def __init__(
        self,
        message: str = "LLM API call failed",
        provider_name: str | None = None,
    ) -> None:
        super().__init__(message=message, provider_name=provider_name)


class RateLimitError(RaiveFlierError):
    """Raised when an API rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        provider_name: str | None = None,
    ) -> None:
        super().__init__(message=message, provider_name=provider_name)

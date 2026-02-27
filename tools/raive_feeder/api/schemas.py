"""Request and response Pydantic models for raiveFeeder API endpoints.

# ─── SCHEMA DESIGN ─────────────────────────────────────────────────────
#
# All schemas use Pydantic v2 with frozen=True for immutability
# (CLAUDE.md Section 28).  Request models validate user input at the
# API boundary; response models shape the JSON sent to the frontend.
#
# Naming convention:
#   - *Request  — incoming from client
#   - *Response — outgoing to client
#   - *Status   — status/progress payloads
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


# ─── Enums ─────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    """Lifecycle states for a batch processing job."""
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SourceType(str, Enum):
    """Supported source types for ingested documents."""
    BOOK = "book"
    ARTICLE = "article"
    INTERVIEW = "interview"
    REFERENCE = "reference"
    FLIER = "flier"
    EVENT_LISTING = "event_listing"
    TRANSCRIPT = "transcript"


class TranscriptionProvider(str, Enum):
    """Available audio transcription backends."""
    WHISPER_LOCAL = "whisper_local"
    WHISPER_API = "whisper_api"


class ImageMode(str, Enum):
    """Image ingestion modes."""
    SINGLE_FLIER = "single_flier"
    MULTI_PAGE_SCAN = "multi_page_scan"


# ─── Document Ingestion ───────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    """Metadata provided by the user when uploading a document."""
    model_config = ConfigDict(frozen=True)

    title: str = Field(description="Document title.")
    author: str = Field(default="", description="Author name.")
    year: int = Field(default=0, ge=0, description="Publication year.")
    source_type: SourceType = Field(
        default=SourceType.BOOK,
        description="Type classification for the document.",
    )
    citation_tier: int = Field(
        default=2, ge=1, le=6,
        description="Citation reliability tier (1=best, 6=worst).",
    )


class IngestDocumentResponse(BaseModel):
    """Response after document ingestion completes."""
    model_config = ConfigDict(frozen=True)

    job_id: str = Field(description="Unique job identifier.")
    source_id: str = Field(default="", description="Source ID in the vector store.")
    source_title: str = Field(default="", description="Title of the ingested source.")
    chunks_created: int = Field(default=0, description="Number of chunks stored.")
    total_tokens: int = Field(default=0, description="Approximate total tokens.")
    ingestion_time: float = Field(default=0.0, description="Ingestion duration in seconds.")
    status: JobStatus = Field(default=JobStatus.COMPLETED)
    error: str | None = Field(default=None, description="Error message if failed.")


# ─── Audio Transcription ──────────────────────────────────────────────

class TranscriptionRequest(BaseModel):
    """Parameters for audio transcription (submitted alongside audio file upload)."""
    model_config = ConfigDict(frozen=True)

    provider: TranscriptionProvider = Field(
        default=TranscriptionProvider.WHISPER_LOCAL,
        description="Which transcription backend to use.",
    )
    language: str | None = Field(
        default=None,
        description="ISO 639-1 language code (e.g. 'en'). None for auto-detect.",
    )
    title: str = Field(default="", description="Title for the resulting document.")
    source_type: SourceType = Field(
        default=SourceType.INTERVIEW,
        description="Source type classification.",
    )
    citation_tier: int = Field(default=3, ge=1, le=6)


class TranscriptionResponse(BaseModel):
    """Response containing the transcription result before ingestion."""
    model_config = ConfigDict(frozen=True)

    job_id: str
    transcript: str = Field(description="Full transcribed text.")
    language: str = Field(default="en", description="Detected or specified language.")
    duration_seconds: float = Field(default=0.0)
    provider_used: str = Field(default="")
    status: JobStatus = Field(default=JobStatus.COMPLETED)
    error: str | None = None


# ─── Image Ingestion ──────────────────────────────────────────────────

class ImageIngestRequest(BaseModel):
    """Parameters for image OCR ingestion."""
    model_config = ConfigDict(frozen=True)

    mode: ImageMode = Field(
        default=ImageMode.SINGLE_FLIER,
        description="Single flier or multi-page scan mode.",
    )
    title: str = Field(default="", description="Title for the resulting document.")
    author: str = Field(default="")
    source_type: SourceType = Field(default=SourceType.FLIER)
    citation_tier: int = Field(default=4, ge=1, le=6)


class ImageIngestResponse(BaseModel):
    """Response from image OCR ingestion."""
    model_config = ConfigDict(frozen=True)

    job_id: str
    ocr_text: str = Field(default="", description="Extracted text from OCR.")
    chunks_created: int = Field(default=0)
    status: JobStatus = Field(default=JobStatus.COMPLETED)
    error: str | None = None


# ─── URL Scraping ─────────────────────────────────────────────────────

class ScrapeURLRequest(BaseModel):
    """Parameters for URL scraping."""
    model_config = ConfigDict(frozen=True)

    url: str = Field(description="Seed URL to start crawling from.")
    max_depth: int = Field(default=0, ge=0, le=5, description="Maximum link-follow depth.")
    max_pages: int = Field(default=1, ge=1, le=100, description="Maximum pages to scrape.")
    nl_query: str | None = Field(
        default=None,
        description="Optional natural language query for LLM-guided relevance filtering.",
    )
    auto_ingest: bool = Field(
        default=False,
        description="If True, automatically ingest all scraped pages.",
    )


class ScrapedPage(BaseModel):
    """A single page discovered during web crawling."""
    model_config = ConfigDict(frozen=True)

    url: str
    title: str = Field(default="")
    text_preview: str = Field(default="", description="First ~500 chars of extracted text.")
    relevance_score: float | None = Field(
        default=None, description="LLM relevance score (0-10) if NL query was provided.",
    )
    word_count: int = Field(default=0)


class ScrapeURLResponse(BaseModel):
    """Response from URL scraping with discovered pages."""
    model_config = ConfigDict(frozen=True)

    job_id: str
    pages: list[ScrapedPage] = Field(default_factory=list)
    status: JobStatus = Field(default=JobStatus.COMPLETED)
    error: str | None = None


# ─── Batch Processing ─────────────────────────────────────────────────

class BatchIngestRequest(BaseModel):
    """Parameters for queuing a batch ingestion job."""
    model_config = ConfigDict(frozen=True)

    # File paths or URLs are provided as multipart form data, not here.
    # This schema carries the shared metadata for all items in the batch.
    source_type: SourceType = Field(default=SourceType.ARTICLE)
    citation_tier: int = Field(default=3, ge=1, le=6)
    skip_tagging: bool = Field(
        default=False,
        description="Skip LLM metadata extraction for faster processing.",
    )


class JobStatusResponse(BaseModel):
    """Status of a batch processing job."""
    model_config = ConfigDict(frozen=True)

    job_id: str
    status: JobStatus
    total_items: int = Field(default=0)
    completed_items: int = Field(default=0)
    failed_items: int = Field(default=0)
    current_item: str | None = Field(default=None, description="Currently processing item.")
    errors: list[str] = Field(default_factory=list)
    message: str = Field(default="", description="Human-readable status message (e.g. approval queue confirmation).")


# ─── Corpus Management ────────────────────────────────────────────────

class CorpusStatsResponse(BaseModel):
    """Corpus statistics for the dashboard."""
    model_config = ConfigDict(frozen=True)

    total_chunks: int = Field(default=0)
    total_sources: int = Field(default=0)
    sources_by_type: dict[str, int] = Field(default_factory=dict)
    entity_tag_count: int = Field(default=0)
    geographic_tag_count: int = Field(default=0)
    genre_tag_count: int = Field(default=0)
    genre_tags: list[str] = Field(default_factory=list)
    time_periods: list[str] = Field(default_factory=list)


class CorpusSourceSummary(BaseModel):
    """Summary of a single ingested source in the corpus."""
    model_config = ConfigDict(frozen=True)

    source_id: str
    source_title: str = Field(default="")
    source_type: str = Field(default="unknown")
    chunk_count: int = Field(default=0)
    author: str | None = None
    citation_tier: int = Field(default=6)


class CorpusSearchRequest(BaseModel):
    """Request body for semantic corpus search."""
    model_config = ConfigDict(frozen=True)

    query: str = Field(description="Natural language search query.")
    top_k: int = Field(default=20, ge=1, le=100)
    source_type: str | None = None
    genre: str | None = None
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)


class ChunkMetadataUpdate(BaseModel):
    """Partial metadata update for a chunk."""
    model_config = ConfigDict(frozen=True)

    entity_tags: list[str] | None = None
    geographic_tags: list[str] | None = None
    genre_tags: list[str] | None = None
    time_period: str | None = None
    citation_tier: int | None = Field(default=None, ge=1, le=6)


# ─── Health / System ──────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Health check response."""
    model_config = ConfigDict(frozen=True)

    status: str = Field(default="ok")
    ingestion_available: bool = Field(default=False)
    vector_store_available: bool = Field(default=False)
    llm_available: bool = Field(default=False)
    ocr_providers: int = Field(default=0)
    ingestion_status: str = Field(default="")


class ProviderInfo(BaseModel):
    """Information about an available provider."""
    model_config = ConfigDict(frozen=True)

    name: str
    provider_type: str
    available: bool = Field(default=True)

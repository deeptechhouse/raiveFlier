"""RAG pipeline data models for the raiveFlier knowledge base.

Defines Pydantic v2 models for document chunks, retrieval results, ingestion
metadata, and corpus statistics.  These models power the vector-store-backed
retrieval-augmented generation layer described in ``RAG_PIPELINE_PLAN.md``.
All models use frozen config to enforce immutability (CLAUDE.md Section 28).
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class DocumentChunk(BaseModel):
    """A chunk of text from a source document, ready for embedding and storage.

    Each chunk carries rich metadata enabling filtered semantic retrieval —
    e.g. retrieve only book passages mentioning a specific artist that were
    published before the date on a flier.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(description="Unique identifier (UUID) for this chunk.")
    text: str = Field(description="The chunk's textual content.")
    source_id: str = Field(description="Identifier of the parent source document.")
    source_title: str = Field(description="Human-readable title of the source document.")
    source_type: str = Field(
        description=(
            'Type of the source document: "book", "article", '
            '"interview", "flier", or "analysis".'
        )
    )
    author: str | None = Field(default=None, description="Author of the source document, if known.")
    publication_date: date | None = Field(
        default=None,
        description="Publication or creation date of the source document.",
    )
    citation_tier: int = Field(
        default=6,
        ge=1,
        le=6,
        description=(
            "Citation reliability tier (1 = published book, " "6 = unverified web content)."
        ),
    )
    page_number: str | None = Field(
        default=None,
        description="Page or section reference within the source document.",
    )
    entity_tags: list[str] = Field(
        default_factory=list,
        description="Artist, venue, and label names mentioned in this chunk.",
    )
    geographic_tags: list[str] = Field(
        default_factory=list,
        description="Cities, countries, and regions referenced in this chunk.",
    )
    genre_tags: list[str] = Field(
        default_factory=list,
        description="Musical genres and sub-genres referenced in this chunk.",
    )


class RetrievedChunk(BaseModel):
    """A document chunk returned from a vector-store query with its score.

    Wraps a :class:`DocumentChunk` with the similarity score produced by the
    vector store and a pre-formatted citation string for downstream display.
    """

    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk = Field(description="The retrieved document chunk.")
    similarity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score between the query and this chunk.",
    )
    formatted_citation: str = Field(
        default="",
        description=(
            "Human-readable citation string, e.g. "
            '"Energy Flash, Simon Reynolds, p.142, Faber & Faber, 1998 [Tier 1]".'
        ),
    )


class IngestionResult(BaseModel):
    """Summary of a single document ingestion run.

    Returned by the ingestion pipeline after processing and indexing a source
    document into the vector store.
    """

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(description="Identifier assigned to the ingested source.")
    source_title: str = Field(description="Title of the ingested source document.")
    chunks_created: int = Field(
        default=0, ge=0, description="Number of chunks produced and stored."
    )
    total_tokens: int = Field(
        default=0,
        ge=0,
        description="Approximate total token count across all chunks.",
    )
    ingestion_time: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock time in seconds for the ingestion run.",
    )


class CorpusStats(BaseModel):
    """Aggregate statistics for the vector-store corpus.

    Provides a snapshot of the knowledge base's size and composition.
    """

    model_config = ConfigDict(frozen=True)

    total_chunks: int = Field(default=0, ge=0, description="Total number of chunks in the store.")
    total_sources: int = Field(
        default=0, ge=0, description="Total number of distinct source documents."
    )
    sources_by_type: dict[str, int] = Field(
        default_factory=dict,
        description=("Breakdown of source count by type " '(e.g. {"book": 5, "article": 42}).'),
    )
    entity_tag_count: int = Field(
        default=0,
        ge=0,
        description="Total number of distinct entity tags across all chunks.",
    )
    geographic_tag_count: int = Field(
        default=0,
        ge=0,
        description="Total number of distinct geographic tags across all chunks.",
    )

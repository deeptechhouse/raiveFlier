"""RAG pipeline data models for the raiveFlier knowledge base.

Defines Pydantic v2 models for document chunks, retrieval results, ingestion
metadata, and corpus statistics.  These models power the vector-store-backed
retrieval-augmented generation layer described in ``RAG_PIPELINE_PLAN.md``.
All models use frozen config to enforce immutability (CLAUDE.md Section 28).

RAG (Retrieval-Augmented Generation) overview for junior developers:
    RAG is a technique where an LLM's responses are enhanced by first
    searching a curated knowledge base for relevant passages. In raiveFlier:

    1. INGESTION: Reference documents (books like "Energy Flash", articles
       from Resident Advisor, etc.) are split into small text chunks.
    2. EMBEDDING: Each chunk is converted into a numeric vector (embedding)
       that captures its semantic meaning.
    3. STORAGE: Chunks + embeddings are stored in ChromaDB (a vector database).
    4. RETRIEVAL: When analyzing a flier, the system searches ChromaDB for
       chunks relevant to the entities found (e.g. "DJ Shadow techno").
    5. GENERATION: Retrieved chunks are included in the LLM prompt as
       context, producing more informed and citation-backed analysis.

    See src/services/ingestion/ for the ingestion pipeline and
    src/providers/vector_store/ for the ChromaDB integration.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# DocumentChunk — the fundamental unit of the RAG knowledge base.
# ---------------------------------------------------------------------------
class DocumentChunk(BaseModel):
    """A chunk of text from a source document, ready for embedding and storage.

    Each chunk carries rich metadata enabling filtered semantic retrieval —
    e.g. retrieve only book passages mentioning a specific artist that were
    published before the date on a flier.

    Chunks are created by src/services/ingestion/chunker.py which splits
    documents into overlapping windows of ~500 tokens each.
    """

    model_config = ConfigDict(frozen=True)

    # UUID uniquely identifying this chunk in the vector store.
    chunk_id: str = Field(description="Unique identifier (UUID) for this chunk.")
    # The actual text content of this chunk.
    text: str = Field(description="The chunk's textual content.")
    # Approximate token count — used for LLM context window budgeting.
    token_count: int = Field(default=0, ge=0, description="Approximate token count for this chunk.")
    # --- Source provenance fields: track where this chunk came from. ---
    # Links back to the parent document (e.g. "energy_flash_reynolds").
    source_id: str = Field(description="Identifier of the parent source document.")
    # Human-readable title (e.g. "Energy Flash by Simon Reynolds").
    source_title: str = Field(description="Human-readable title of the source document.")
    # What kind of document this chunk comes from — affects citation tier.
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
    # --- Semantic tags: extracted during ingestion by metadata_extractor.py.
    # These tags enable FILTERED vector search — e.g. "find chunks about
    # Aphex Twin from books" rather than searching the entire corpus. ---
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
    entity_types: list[str] = Field(
        default_factory=list,
        description=(
            "Parallel list of entity type classifications — one per entity_tag. "
            "Values: ARTIST, VENUE, LABEL, EVENT, COLLECTIVE. Empty for legacy chunks."
        ),
    )
    time_period: str | None = Field(
        default=None,
        description="Decade or year range referenced in the text, e.g. '1990s', '1987-1989'.",
    )


# ---------------------------------------------------------------------------
# RetrievedChunk — a search result from the vector store.
# ---------------------------------------------------------------------------
class RetrievedChunk(BaseModel):
    """A document chunk returned from a vector-store query with its score.

    Wraps a :class:`DocumentChunk` with the similarity score produced by the
    vector store and a pre-formatted citation string for downstream display.

    The corpus search endpoint (/api/v1/corpus/search) and the Q&A service
    (src/services/qa_service.py) both return lists of RetrievedChunk objects.
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


# ---------------------------------------------------------------------------
# IngestionResult — output of the ingestion pipeline for one document.
# ---------------------------------------------------------------------------
class IngestionResult(BaseModel):
    """Summary of a single document ingestion run.

    Returned by the ingestion pipeline after processing and indexing a source
    document into the vector store. Used by the CLI (src/cli/ingest.py) to
    report progress to the operator.
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


# ---------------------------------------------------------------------------
# CorpusStats — a snapshot of the knowledge base's size and composition.
# ---------------------------------------------------------------------------
class CorpusStats(BaseModel):
    """Aggregate statistics for the vector-store corpus.

    Provides a snapshot of the knowledge base's size and composition.
    Returned by the /api/v1/corpus/stats endpoint and displayed in the
    frontend's corpus sidebar to show what reference material is available.
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
    # Genre and time_period lists for populating frontend filter dropdowns.
    # Collected during get_stats() by iterating chunk metadata.
    genre_tag_count: int = Field(
        default=0,
        ge=0,
        description="Total number of distinct genre tags across all chunks.",
    )
    genre_tags: list[str] = Field(
        default_factory=list,
        description="Sorted list of distinct genre tag strings for filter dropdowns.",
    )
    time_periods: list[str] = Field(
        default_factory=list,
        description="Sorted list of distinct time period strings for filter dropdowns.",
    )
    # Full tag lists for autocomplete — sorted for consistent UI display.
    # Kept separate from the *_count fields so callers that only need
    # counts don't pay for the list overhead.
    entity_tags_list: list[str] = Field(
        default_factory=list,
        description="Sorted list of distinct entity tag strings for autocomplete.",
    )
    geographic_tags_list: list[str] = Field(
        default_factory=list,
        description="Sorted list of distinct geographic tag strings for autocomplete.",
    )

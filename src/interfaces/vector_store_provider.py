"""Abstract base class for vector-store service providers.

Defines the contract for storing, querying, and managing embedded document
chunks.  Implementations may wrap ChromaDB (local/free), Qdrant, Pinecone,
or any other vector database.  The adapter pattern (CLAUDE.md Section 6)
keeps the RAG layer independent of the chosen backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.models.rag import CorpusStats, DocumentChunk, RetrievedChunk


# Concrete implementation: ChromaDBProvider (src/providers/vector_store/)
# ChromaDB is a free, open-source vector database. Data persists to disk at
# CHROMADB_PERSIST_DIR (default: /data/chromadb on Render's persistent disk).
# Could be swapped for Qdrant, Pinecone, or Weaviate via this interface.
class IVectorStoreProvider(ABC):
    """Contract for vector-store services used by the RAG pipeline.

    All query and mutation methods are async to support network-backed stores
    without blocking the event loop.

    **Supported filter syntax** (passed via the *filters* dict in :meth:`query`):

    * ``{"date": {"$lte": "YYYY-MM-DD"}}`` — chunks from sources published
      on or before a given date.
    * ``{"entity_tags": {"$contains": "name"}}`` — chunks mentioning a
      specific artist, venue, or label.
    * ``{"source_type": {"$in": ["book", "article"]}}`` — restrict to
      specific source types.
    * ``{"geographic_tags": {"$contains": "city"}}`` — chunks referencing
      a specific geographic location.
    * ``{"genre_tags": {"$contains": "techno"}}`` — chunks tagged with a
      specific genre (substring match on comma-separated metadata).
    * ``{"citation_tier": {"$lte": 3}}`` — chunks with citation tier at
      or better than the given value (1 = best, 6 = unverified).

    Concrete providers translate this common filter syntax into their
    backend-specific query language.
    """

    @abstractmethod
    async def query(
        self,
        query_text: str,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        max_per_source: int = 3,
    ) -> list[RetrievedChunk]:
        """Perform a semantic search against the vector store.

        Parameters
        ----------
        query_text:
            The natural-language query to embed and search for.
        top_k:
            Maximum number of results to return.
        filters:
            Optional metadata filters to narrow the search (see class
            docstring for supported syntax).
        max_per_source:
            Maximum number of chunks from any single source document.
            Prevents one long document from dominating results while
            still allowing multiple relevant passages.

        Returns
        -------
        list[RetrievedChunk]
            Zero or more results ranked by similarity score (descending).

        Raises
        ------
        src.core.errors.RAGError
            If the vector store query fails.
        """

    @abstractmethod
    async def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> int:
        """Add pre-embedded document chunks to the vector store.

        Parameters
        ----------
        chunks:
            The document chunks to store.  Each chunk's ``chunk_id`` is used
            as the primary key.
        embeddings:
            Embedding vectors corresponding positionally to *chunks*.  The
            caller is responsible for generating embeddings via an
            :class:`~src.interfaces.embedding_provider.IEmbeddingProvider`
            before calling this method.

        Returns
        -------
        int
            The number of chunks successfully added.

        Raises
        ------
        ValueError
            If ``len(chunks) != len(embeddings)``.
        src.core.errors.RAGError
            If the store operation fails.
        """

    @abstractmethod
    async def delete_by_source(self, source_id: str) -> int:
        """Delete all chunks originating from the given source.

        Parameters
        ----------
        source_id:
            The ``source_id`` of the document whose chunks should be removed.

        Returns
        -------
        int
            The number of chunks deleted.
        """

    @abstractmethod
    async def delete_by_source_type(self, source_type: str) -> int:
        """Delete all chunks with the given source type.

        Parameters
        ----------
        source_type:
            The ``source_type`` metadata value (e.g. ``"interview"``,
            ``"reference"``, ``"book"``).

        Returns
        -------
        int
            The number of chunks deleted.
        """

    @abstractmethod
    async def get_source_ids(self, source_type: str | None = None) -> set[str]:
        """Return the set of unique ``source_id`` values in the store.

        Parameters
        ----------
        source_type:
            If provided, only return source IDs for chunks of this type
            (e.g. ``"reference"``, ``"analysis"``).  If ``None``, return
            all source IDs regardless of type.

        Returns
        -------
        set[str]
            Unique source identifiers.
        """

    @abstractmethod
    async def get_stats(self) -> CorpusStats:
        """Return aggregate statistics about the corpus.

        Returns
        -------
        CorpusStats
            A snapshot of the knowledge base's size and composition.
        """

    @abstractmethod
    async def update_chunk_metadata(
        self, chunk_id: str, metadata: dict[str, Any]
    ) -> bool:
        """Update metadata fields on an existing chunk without re-embedding.

        Allows editing tags, citation tier, time period, etc. on stored
        chunks without the cost of regenerating embeddings.  Used by the
        raiveFeeder Corpus Management tab for metadata corrections.

        Parameters
        ----------
        chunk_id:
            The unique identifier of the chunk to update.
        metadata:
            Dictionary of metadata field names to new values.  Only the
            provided fields are updated; others remain unchanged.

        Returns
        -------
        bool
            ``True`` if the chunk was found and updated, ``False`` if no
            chunk with the given ID exists.
        """

    @abstractmethod
    async def list_all_metadata(
        self,
        page_size: int = 5000,
        include_documents: bool = False,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, dict[str, Any], str | None]]:
        """Iterate over all stored chunks, returning (id, metadata, document) tuples.

        Provides a public, backend-agnostic way to enumerate corpus contents
        without reaching into provider internals.  Used by CorpusManager for
        source listing and detail views.

        Parameters
        ----------
        page_size:
            Number of chunks to fetch per internal page (implementation hint
            for providers that paginate their storage layer).
        include_documents:
            If ``True``, include the chunk text in the third tuple element.
            If ``False``, the third element is ``None`` (saves memory).
        where:
            Optional filter clause (same syntax as :meth:`query` filters)
            to restrict which chunks are returned.

        Returns
        -------
        list[tuple[str, dict[str, Any], str | None]]
            Each tuple is ``(chunk_id, metadata_dict, document_text_or_None)``.
        """

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return a human-readable identifier for this vector-store provider.

        Example return values: ``"chromadb"``, ``"qdrant"``.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if the provider is configured and reachable.

        Implementations should verify that the store is accessible without
        performing a full query.
        """

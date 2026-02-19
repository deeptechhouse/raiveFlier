"""ChromaDB vector store provider adapter.

Wraps `chromadb.PersistentClient` to implement :class:`IVectorStoreProvider`.
Uses cosine distance for similarity search.  Fully local, free, and
Python-native — no external service required.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import chromadb
import structlog

from src.interfaces.embedding_provider import IEmbeddingProvider
from src.interfaces.vector_store_provider import IVectorStoreProvider
from src.models.rag import CorpusStats, DocumentChunk, RetrievedChunk
from src.utils.errors import RAGError

logger = structlog.get_logger(logger_name=__name__)


class ChromaDBProvider(IVectorStoreProvider):
    """Vector store provider backed by ChromaDB with local persistence.

    ChromaDB stores embeddings on disk and performs cosine-similarity search.
    An :class:`IEmbeddingProvider` is injected at init time so the provider
    can embed query text before passing it to ChromaDB.
    """

    def __init__(
        self,
        embedding_provider: IEmbeddingProvider,
        persist_directory: str = "./data/chromadb",
        collection_name: str = "raiveflier_corpus",
    ) -> None:
        self._embedding_provider = embedding_provider
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # IVectorStoreProvider implementation
    # ------------------------------------------------------------------

    async def query(
        self,
        query_text: str,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Perform semantic search against the ChromaDB collection."""
        try:
            query_embedding = await self._embedding_provider.embed_single(query_text)
            where_clause = self._translate_filters(filters) if filters else None

            kwargs: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
            }
            if where_clause:
                kwargs["where"] = where_clause

            results = self._collection.query(**kwargs)

            retrieved: list[RetrievedChunk] = []
            if not results["documents"] or not results["documents"][0]:
                return retrieved

            documents = results["documents"][0]
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(documents)

            for doc_text, meta, distance in zip(documents, metadatas, distances, strict=True):
                similarity = 1.0 - distance
                chunk = self._metadata_to_chunk(meta, doc_text)
                citation = self._format_citation(chunk, similarity)
                retrieved.append(
                    RetrievedChunk(
                        chunk=chunk,
                        similarity_score=max(0.0, min(1.0, similarity)),
                        formatted_citation=citation,
                    )
                )

            logger.info(
                "chromadb_query",
                query_length=len(query_text),
                results_count=len(retrieved),
                top_score=retrieved[0].similarity_score if retrieved else 0.0,
            )
            return retrieved

        except RAGError:
            raise
        except Exception as exc:
            raise RAGError(
                message=f"ChromaDB query failed: {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

    async def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> int:
        """Add pre-embedded document chunks to the ChromaDB collection (upsert)."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings length mismatch: {len(chunks)} != {len(embeddings)}"
            )
        if not chunks:
            return 0

        try:
            ids = [c.chunk_id for c in chunks]
            documents = [c.text for c in chunks]
            metadatas = [self._chunk_to_metadata(c) for c in chunks]

            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            logger.info(
                "chromadb_add_chunks",
                count=len(chunks),
            )
            return len(chunks)

        except Exception as exc:
            raise RAGError(
                message=f"ChromaDB add_chunks failed: {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

    async def delete_by_source(self, source_id: str) -> int:
        """Delete all chunks originating from the given source."""
        try:
            existing = self._collection.get(where={"source_id": source_id})
            count = len(existing["ids"]) if existing["ids"] else 0

            if count > 0:
                self._collection.delete(where={"source_id": source_id})

            logger.info(
                "chromadb_delete_by_source",
                source_id=source_id,
                deleted_count=count,
            )
            return count

        except Exception as exc:
            raise RAGError(
                message=f"ChromaDB delete_by_source failed: {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

    async def get_stats(self) -> CorpusStats:
        """Return aggregate statistics about the corpus."""
        try:
            total_chunks = self._collection.count()

            sources_by_type: dict[str, int] = {}
            source_ids: set[str] = set()
            entity_tags: set[str] = set()
            geographic_tags: set[str] = set()

            if total_chunks > 0:
                all_meta = self._collection.get(include=["metadatas"])
                for meta in all_meta["metadatas"] or []:
                    source_ids.add(meta.get("source_id", ""))
                    src_type = meta.get("source_type", "unknown")
                    sources_by_type[src_type] = sources_by_type.get(src_type, 0) + 1

                    for tag in self._split_tags(meta.get("entity_tags", "")):
                        entity_tags.add(tag)
                    for tag in self._split_tags(meta.get("geographic_tags", "")):
                        geographic_tags.add(tag)

            return CorpusStats(
                total_chunks=total_chunks,
                total_sources=len(source_ids),
                sources_by_type=sources_by_type,
                entity_tag_count=len(entity_tags),
                geographic_tag_count=len(geographic_tags),
            )

        except Exception as exc:
            raise RAGError(
                message=f"ChromaDB get_stats failed: {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

    def get_provider_name(self) -> str:
        return "chromadb"

    def is_available(self) -> bool:
        """Return ``True`` if the ChromaDB collection is accessible."""
        try:
            self._collection.count()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_to_metadata(chunk: DocumentChunk) -> dict[str, str | int | float | bool]:
        """Convert a DocumentChunk to a ChromaDB-compatible metadata dict.

        ChromaDB metadata values must be str, int, float, or bool.
        Lists are serialized as comma-separated strings.
        """
        meta: dict[str, str | int | float | bool] = {
            "source_id": chunk.source_id,
            "source_title": chunk.source_title,
            "source_type": chunk.source_type,
            "citation_tier": chunk.citation_tier,
            "entity_tags": ",".join(chunk.entity_tags),
            "geographic_tags": ",".join(chunk.geographic_tags),
            "genre_tags": ",".join(chunk.genre_tags),
        }
        if chunk.author is not None:
            meta["author"] = chunk.author
        if chunk.publication_date is not None:
            meta["publication_date"] = chunk.publication_date.isoformat()
        if chunk.page_number is not None:
            meta["page_number"] = chunk.page_number
        return meta

    @staticmethod
    def _metadata_to_chunk(meta: dict[str, Any], text: str) -> DocumentChunk:
        """Convert a ChromaDB metadata dict back to a DocumentChunk.

        Reverses the serialization done by :meth:`_chunk_to_metadata`.
        """
        pub_date_str = meta.get("publication_date")
        pub_date = date.fromisoformat(pub_date_str) if pub_date_str else None

        return DocumentChunk(
            chunk_id=meta.get("chunk_id", ""),
            text=text,
            source_id=meta.get("source_id", ""),
            source_title=meta.get("source_title", ""),
            source_type=meta.get("source_type", "unknown"),
            author=meta.get("author"),
            publication_date=pub_date,
            citation_tier=int(meta.get("citation_tier", 6)),
            page_number=meta.get("page_number"),
            entity_tags=ChromaDBProvider._split_tags(meta.get("entity_tags", "")),
            geographic_tags=ChromaDBProvider._split_tags(meta.get("geographic_tags", "")),
            genre_tags=ChromaDBProvider._split_tags(meta.get("genre_tags", "")),
        )

    @staticmethod
    def _split_tags(value: str | Any) -> list[str]:
        """Split a comma-separated tag string back into a list."""
        if not value or not isinstance(value, str):
            return []
        return [tag.strip() for tag in value.split(",") if tag.strip()]

    @staticmethod
    def _translate_filters(filters: dict[str, Any]) -> dict[str, Any] | None:
        """Translate the common filter syntax to ChromaDB ``where`` clauses.

        Supported input keys:
        - ``date``: ``{"$lte": "YYYY-MM-DD"}`` → ``publication_date`` filter
        - ``entity_tags``: ``{"$contains": "name"}`` → string ``$contains``
        - ``geographic_tags``: ``{"$contains": "city"}`` → string ``$contains``
        - ``source_type``: ``{"$in": [...]}`` → ``$in`` filter
        """
        clauses: list[dict[str, Any]] = []

        date_filter = filters.get("date")
        if date_filter and isinstance(date_filter, dict):
            lte_val = date_filter.get("$lte")
            if lte_val:
                clauses.append({"publication_date": {"$lte": str(lte_val)}})

        entity_filter = filters.get("entity_tags")
        if entity_filter and isinstance(entity_filter, dict):
            contains_val = entity_filter.get("$contains")
            if contains_val:
                clauses.append({"entity_tags": {"$contains": str(contains_val)}})

        geo_filter = filters.get("geographic_tags")
        if geo_filter and isinstance(geo_filter, dict):
            contains_val = geo_filter.get("$contains")
            if contains_val:
                clauses.append({"geographic_tags": {"$contains": str(contains_val)}})

        source_filter = filters.get("source_type")
        if source_filter and isinstance(source_filter, dict):
            in_val = source_filter.get("$in")
            if in_val and isinstance(in_val, list):
                clauses.append({"source_type": {"$in": in_val}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    @staticmethod
    def _format_citation(chunk: DocumentChunk, score: float) -> str:
        """Build a human-readable citation string from chunk metadata."""
        parts: list[str] = [chunk.source_title]
        if chunk.author:
            parts.append(chunk.author)
        if chunk.page_number:
            parts.append(f"p.{chunk.page_number}")
        if chunk.publication_date:
            parts.append(str(chunk.publication_date.year))
        parts.append(f"[Tier {chunk.citation_tier}]")
        return ", ".join(parts)

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
        self._cached_stats: CorpusStats | None = None
        self._cached_stats_count: int = -1

        # Validate embedding dimensions match existing corpus data
        self._validate_embedding_dimensions()

    # ------------------------------------------------------------------
    # Startup validation
    # ------------------------------------------------------------------

    def _validate_embedding_dimensions(self) -> None:
        """Verify embedding provider dimensions match existing corpus vectors.

        Peeks at a single stored vector and compares its length to the
        provider's declared dimension.  A mismatch means every query will
        produce garbage results — fail loud and fast.
        """
        try:
            collection_count = self._collection.count()
            if collection_count == 0:
                return  # Empty collection — nothing to validate against

            sample = self._collection.peek(limit=1)
            embeddings = sample.get("embeddings") if sample else None
            if embeddings is None or len(embeddings) == 0:
                return  # No embeddings stored (shouldn't happen, but defensive)

            stored_dim = len(embeddings[0])
            expected_dim = self._embedding_provider.get_dimension()

            if stored_dim != expected_dim:
                logger.error(
                    "embedding_dimension_mismatch",
                    stored_dim=stored_dim,
                    expected_dim=expected_dim,
                    provider=self._embedding_provider.get_provider_name(),
                )
                raise RAGError(
                    message=(
                        f"Embedding dimension mismatch: corpus has {stored_dim}-dim vectors "
                        f"but provider '{self._embedding_provider.get_provider_name()}' "
                        f"produces {expected_dim}-dim vectors. "
                        f"Set OPENAI_EMBEDDING_MODEL to the model used to build the corpus."
                    ),
                    provider_name="chromadb",
                )

            logger.info(
                "embedding_dimension_validated",
                dimension=stored_dim,
                corpus_chunks=collection_count,
            )
        except RAGError:
            raise
        except Exception as exc:
            logger.warning("embedding_dimension_check_skipped", error=str(exc))

    # ------------------------------------------------------------------
    # IVectorStoreProvider implementation
    # ------------------------------------------------------------------

    async def query(
        self,
        query_text: str,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        max_per_source: int = 3,
    ) -> list[RetrievedChunk]:
        """Perform semantic search against the ChromaDB collection.

        Results are deduplicated by ``source_id``: when multiple chunks from
        the same source document match, only the top *max_per_source* chunks
        are kept.  To compensate, ChromaDB is asked for up to ``3 * top_k``
        raw results before dedup and trimming.
        """
        try:
            query_embedding = await self._embedding_provider.embed_single(query_text)
            where_clause = self._translate_filters(filters) if filters else None

            # Over-fetch to have enough results after per-source limiting
            fetch_k = min(top_k * 3, max(top_k, self._collection.count()))

            kwargs: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": fetch_k,
            }
            if where_clause:
                kwargs["where"] = where_clause

            results = self._collection.query(**kwargs)

            if not results["documents"] or not results["documents"][0]:
                return []

            documents = results["documents"][0]
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(documents)

            # Group by source_id, keeping top max_per_source chunks per source
            chunks_per_source: dict[str, list[RetrievedChunk]] = {}

            for doc_text, meta, distance in zip(documents, metadatas, distances, strict=True):
                similarity = max(0.0, min(1.0, 1.0 - distance))
                chunk = self._metadata_to_chunk(meta, doc_text)
                source_id = chunk.source_id
                citation = self._format_citation(chunk, similarity)
                rc = RetrievedChunk(
                    chunk=chunk,
                    similarity_score=similarity,
                    formatted_citation=citation,
                )
                chunks_per_source.setdefault(source_id, []).append(rc)

            # Keep top N per source, flatten, sort globally, trim to top_k
            deduped: list[RetrievedChunk] = []
            for entries in chunks_per_source.values():
                entries.sort(key=lambda rc: rc.similarity_score, reverse=True)
                deduped.extend(entries[:max_per_source])

            retrieved = sorted(
                deduped,
                key=lambda rc: rc.similarity_score,
                reverse=True,
            )[:top_k]

            logger.info(
                "chromadb_query",
                query_length=len(query_text),
                raw_results=len(documents),
                unique_sources=len(chunks_per_source),
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

        self._cached_stats = None  # Invalidate stats cache

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
        self._cached_stats = None  # Invalidate stats cache
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

    async def delete_by_source_type(self, source_type: str) -> int:
        """Delete all chunks with the given source type."""
        self._cached_stats = None  # Invalidate stats cache
        try:
            existing = self._collection.get(where={"source_type": source_type})
            count = len(existing["ids"]) if existing["ids"] else 0

            if count > 0:
                self._collection.delete(where={"source_type": source_type})

            logger.info(
                "chromadb_delete_by_source_type",
                source_type=source_type,
                deleted_count=count,
            )
            return count

        except Exception as exc:
            raise RAGError(
                message=f"ChromaDB delete_by_source_type failed: {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

    async def get_stats(self) -> CorpusStats:
        """Return aggregate statistics about the corpus.

        Results are cached and invalidated when chunks are added or deleted.
        The cache also detects external changes by comparing the current
        collection count against the count at cache time.
        """
        try:
            current_count = self._collection.count()

            # Return cached stats if still valid
            if self._cached_stats is not None and self._cached_stats_count == current_count:
                return self._cached_stats

            sources_by_type: dict[str, int] = {}
            source_ids: set[str] = set()
            entity_tags: set[str] = set()
            geographic_tags: set[str] = set()

            if current_count > 0:
                all_meta = self._collection.get(include=["metadatas"])
                for meta in all_meta["metadatas"] or []:
                    source_ids.add(meta.get("source_id", ""))
                    src_type = meta.get("source_type", "unknown")
                    sources_by_type[src_type] = sources_by_type.get(src_type, 0) + 1

                    for tag in self._split_tags(meta.get("entity_tags", "")):
                        entity_tags.add(tag)
                    for tag in self._split_tags(meta.get("geographic_tags", "")):
                        geographic_tags.add(tag)

            result = CorpusStats(
                total_chunks=current_count,
                total_sources=len(source_ids),
                sources_by_type=sources_by_type,
                entity_tag_count=len(entity_tags),
                geographic_tag_count=len(geographic_tags),
            )

            # Cache the result
            self._cached_stats = result
            self._cached_stats_count = current_count

            return result

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

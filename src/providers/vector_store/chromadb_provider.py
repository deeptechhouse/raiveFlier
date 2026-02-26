"""ChromaDB vector store provider adapter.

Wraps `chromadb.PersistentClient` to implement :class:`IVectorStoreProvider`.
Uses cosine distance for similarity search.  Fully local, free, and
Python-native — no external service required.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Any

# Disable ChromaDB telemetry completely before importing chromadb.
# ChromaDB uses PostHog for anonymous telemetry, but a version mismatch
# between ChromaDB's bundled PostHog client and the installed version
# causes "capture() takes 1 positional argument but 3 were given" errors.
# Three layers of defense:
#   1. ANONYMIZED_TELEMETRY env var — respected by some ChromaDB versions
#   2. posthog.disabled = True — disables the PostHog SDK directly
#   3. Settings(anonymized_telemetry=False) — passed to PersistentClient
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import posthog

posthog.disabled = True

import chromadb
import structlog

from src.interfaces.embedding_provider import IEmbeddingProvider
from src.interfaces.vector_store_provider import IVectorStoreProvider
from src.models.rag import CorpusStats, DocumentChunk, RetrievedChunk
from src.utils.errors import RAGError

logger = structlog.get_logger(logger_name=__name__)


class _NoopEmbeddingFunction(chromadb.EmbeddingFunction[list[str]]):
    """No-op embedding function that prevents ChromaDB from loading a model.

    raiveFlier always passes pre-computed embeddings to ``add_chunks()``,
    so ChromaDB's built-in embedding is never invoked.  Without this,
    ChromaDB downloads and loads the default all-MiniLM-L6-v2 ONNX model
    (~80 MB RAM) on collection creation — wasted memory on the 512 MB
    Render instance.
    """

    def __call__(self, input: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "raiveFlier uses pre-computed embeddings; "
            "ChromaDB's built-in embedding should never be called."
        )

    def name(self) -> str:
        """Return function name (required by ChromaDB's EmbeddingFunction protocol)."""
        return "noop_precomputed"


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
        # Disable ChromaDB's PostHog telemetry via Settings to prevent
        # "capture() takes 1 positional argument but 3 were given" errors.
        # The env var ANONYMIZED_TELEMETRY alone is insufficient for some
        # ChromaDB versions — passing it through Settings is authoritative.
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
        # Pass _NoopEmbeddingFunction to prevent ChromaDB from downloading
        # and loading its default ONNX embedding model (~80 MB).  All
        # embeddings are pre-computed by our IEmbeddingProvider adapter
        # and passed explicitly to add_chunks().
        #
        # Newer ChromaDB versions enforce that the embedding function must
        # match the one persisted in the collection.  If the collection was
        # created with the default embedding function (e.g., by an older
        # ChromaDB version), passing _NoopEmbeddingFunction triggers a
        # ValueError.  We handle this by falling back to opening the
        # collection without specifying an embedding function — ChromaDB
        # then uses whatever was persisted, which is fine because we
        # pre-compute all embeddings externally anyway.
        try:
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=_NoopEmbeddingFunction(),
            )
        except ValueError:
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
            # and post-filtering (genre, era, similarity threshold).
            # Multiplier raised from 3x to 4x to compensate for more
            # aggressive post-filters introduced in the enhanced search.
            fetch_k = min(top_k * 4, max(top_k, self._collection.count()))

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
        batch_size: int = 500,
    ) -> int:
        """Add pre-embedded document chunks to the ChromaDB collection (upsert).

        Chunks are upserted in batches of *batch_size* to keep memory
        bounded.  A single upsert of thousands of chunks can spike memory
        beyond the 512 MB Render budget because ChromaDB builds internal
        data structures proportional to the batch.  Paginating lets the
        previous batch's allocations be garbage-collected before the next.

        The *batch_size* parameter is an implementation detail — the
        :class:`IVectorStoreProvider` interface is unchanged.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings length mismatch: {len(chunks)} != {len(embeddings)}"
            )
        if not chunks:
            return 0

        self._cached_stats = None  # Invalidate stats cache

        try:
            total_stored = 0

            # Paginate upserts to bound peak memory.  Each iteration
            # builds id/document/metadata lists only for its slice,
            # letting the previous batch's allocations be GC'd.
            for start in range(0, len(chunks), batch_size):
                end = min(start + batch_size, len(chunks))
                batch_chunks = chunks[start:end]
                batch_embeddings = embeddings[start:end]

                ids = [c.chunk_id for c in batch_chunks]
                documents = [c.text for c in batch_chunks]
                metadatas = [self._chunk_to_metadata(c) for c in batch_chunks]

                self._collection.upsert(
                    ids=ids,
                    embeddings=batch_embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
                total_stored += len(batch_chunks)

            logger.info(
                "chromadb_add_chunks",
                count=total_stored,
                batches=(len(chunks) + batch_size - 1) // batch_size,
            )
            return total_stored

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

    async def get_source_ids(self, source_type: str | None = None) -> set[str]:
        """Return unique source_ids, optionally filtered by source_type.

        Uses a ChromaDB ``where`` clause when *source_type* is given to avoid
        fetching metadata for every chunk in the collection.  Paginates in
        5K-row pages to avoid SQLite's bind-parameter limit on large corpora.
        """
        _PAGE_SIZE = 5000
        try:
            kwargs: dict[str, Any] = {"include": ["metadatas"]}
            if source_type:
                kwargs["where"] = {"source_type": source_type}

            source_ids: set[str] = set()
            offset = 0
            while True:
                page = self._collection.get(
                    **kwargs, limit=_PAGE_SIZE, offset=offset
                )
                metadatas = page["metadatas"] or []
                if not metadatas:
                    break
                for m in metadatas:
                    sid = m.get("source_id")
                    if sid:
                        source_ids.add(sid)
                if len(metadatas) < _PAGE_SIZE:
                    break
                offset += _PAGE_SIZE
            return source_ids
        except Exception as exc:
            raise RAGError(
                message=f"ChromaDB get_source_ids failed: {exc}",
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
            # Collect genre tags and time periods for frontend filter dropdowns
            genre_tags: set[str] = set()
            time_periods: set[str] = set()

            # Paginate the metadata fetch to avoid SQLite's "too many SQL
            # variables" limit (~999 bind params).  ChromaDB's .get() with
            # no IDs fetches ALL rows in a single query, which breaks when
            # the collection exceeds ~40K chunks.  Fetching in 5K-row pages
            # keeps each query well under the SQLite variable ceiling.
            _PAGE_SIZE = 5000
            if current_count > 0:
                for page_offset in range(0, current_count, _PAGE_SIZE):
                    page = self._collection.get(
                        include=["metadatas"],
                        limit=_PAGE_SIZE,
                        offset=page_offset,
                    )
                    for meta in page["metadatas"] or []:
                        source_ids.add(meta.get("source_id", ""))
                        src_type = meta.get("source_type", "unknown")
                        sources_by_type[src_type] = sources_by_type.get(src_type, 0) + 1

                        for tag in self._split_tags(meta.get("entity_tags", "")):
                            entity_tags.add(tag)
                        for tag in self._split_tags(meta.get("geographic_tags", "")):
                            geographic_tags.add(tag)
                        for tag in self._split_tags(meta.get("genre_tags", "")):
                            genre_tags.add(tag)
                        tp = meta.get("time_period")
                        if tp:
                            time_periods.add(tp)

            result = CorpusStats(
                total_chunks=current_count,
                total_sources=len(source_ids),
                sources_by_type=sources_by_type,
                entity_tag_count=len(entity_tags),
                geographic_tag_count=len(geographic_tags),
                genre_tag_count=len(genre_tags),
                genre_tags=sorted(genre_tags),
                time_periods=sorted(time_periods),
                # Full tag lists for autocomplete — reuse the sets already
                # collected above to avoid a second pass over metadata.
                entity_tags_list=sorted(entity_tags),
                geographic_tags_list=sorted(geographic_tags),
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

    async def update_chunk_metadata(
        self, chunk_id: str, metadata: dict[str, Any]
    ) -> bool:
        """Update metadata fields on an existing chunk without re-embedding.

        Uses ChromaDB's collection.update() to modify only the specified
        metadata fields.  The chunk text and embedding remain unchanged.
        """
        self._cached_stats = None  # Invalidate stats cache.
        try:
            # Verify the chunk exists before attempting update.
            existing = self._collection.get(ids=[chunk_id])
            if not existing["ids"]:
                logger.warning("update_chunk_not_found", chunk_id=chunk_id)
                return False

            self._collection.update(
                ids=[chunk_id],
                metadatas=[metadata],
            )

            logger.info(
                "chromadb_update_chunk_metadata",
                chunk_id=chunk_id,
                fields=list(metadata.keys()),
            )
            return True

        except Exception as exc:
            raise RAGError(
                message=f"ChromaDB update_chunk_metadata failed: {exc}",
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
            "entity_types": ",".join(chunk.entity_types),
        }
        if chunk.time_period is not None:
            meta["time_period"] = chunk.time_period
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
            entity_types=ChromaDBProvider._split_tags(meta.get("entity_types", "")),
            geographic_tags=ChromaDBProvider._split_tags(meta.get("geographic_tags", "")),
            genre_tags=ChromaDBProvider._split_tags(meta.get("genre_tags", "")),
            time_period=meta.get("time_period"),
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

        # Genre filter: match chunks whose genre_tags metadata contains the
        # genre string.  Genre tags are stored as comma-separated strings,
        # so ChromaDB's $contains does a substring match (e.g. "techno"
        # matches "techno,acid techno,detroit techno").
        genre_filter = filters.get("genre_tags")
        if genre_filter and isinstance(genre_filter, dict):
            contains_val = genre_filter.get("$contains")
            if contains_val:
                clauses.append({"genre_tags": {"$contains": str(contains_val)}})

        # Citation tier filter: only include chunks with citation_tier at
        # or better than the threshold.  Lower number = higher quality,
        # so $lte finds "at least this good" (e.g. $lte: 3 returns T1-T3).
        tier_filter = filters.get("citation_tier")
        if tier_filter and isinstance(tier_filter, dict):
            lte_val = tier_filter.get("$lte")
            if lte_val is not None:
                clauses.append({"citation_tier": {"$lte": int(lte_val)}})

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

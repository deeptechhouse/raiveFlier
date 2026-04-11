"""SQLite FTS5-based BM25 keyword search provider.

Provides lexical (keyword) search as a complement to ChromaDB's semantic
(embedding-based) search.  Uses SQLite's built-in FTS5 full-text search
engine with BM25 ranking.  Designed for reciprocal rank fusion with
semantic results to catch exact-term matches that embedding similarity misses.

# ─── ARCHITECTURE ROLE ─────────────────────────────────────────────────
# Layer: Provider (concrete adapter)
# Pattern: Parallel retrieval path alongside ChromaDB, NOT a replacement
#
# Semantic search (ChromaDB) excels at finding conceptually similar text
# even when wording differs ("Detroit techno pioneers" finds passages
# about Juan Atkins, Derrick May, etc.).  But it can miss exact name
# matches — a query for "Berghain" might rank a passage about "the
# Berlin techno institution" higher than one that literally says
# "Berghain".  BM25 keyword search captures these lexical hits.
#
# The BM25 index mirrors chunk text from ChromaDB.  It is kept in sync
# via upsert/delete calls from the ChromaDB provider during ingestion.
#
# Data flow:
#   Ingestion: ChromaDBProvider.add_chunks() → BM25Provider.upsert_chunks()
#   Search:    routes.py corpus_search → parallel(ChromaDB, BM25) → RRF merge
#
# Dependencies: aiosqlite (async SQLite), no external services
# ────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite
import structlog

logger = structlog.get_logger(logger_name=__name__)


# ─── BM25Result ────────────────────────────────────────────────────────
# Simple dataclass representing a single BM25 search hit.  Carries enough
# metadata to build a CorpusSearchChunk during rank fusion without needing
# to look up the chunk in ChromaDB.
@dataclass
class BM25Result:
    """A single BM25 keyword search result from the FTS5 index."""

    chunk_id: str
    source_id: str
    source_title: str
    source_type: str
    bm25_score: float
    text: str
    citation_tier: int = 6


class BM25Provider:
    """SQLite FTS5-based keyword search provider for hybrid retrieval.

    Uses the Porter stemmer tokenizer so that queries like "raving" match
    documents containing "rave", "raves", "raver" etc.  The ``unicode61``
    tokenizer handles Unicode normalization (accented characters, etc.).

    The FTS5 virtual table stores chunk metadata as UNINDEXED columns —
    they are returned in results but not searchable, keeping the index
    lean.  Only the ``text`` column is indexed for full-text search.
    """

    def __init__(self, db_path: str = "data/bm25_index.db") -> None:
        self._db_path = Path(db_path)
        # Ensure parent directory exists (e.g. /data/ on Render persistent disk)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """Create the FTS5 virtual table if it does not already exist.

        FTS5 is built into Python's sqlite3 module — no extra C extensions
        or pip packages needed.  The ``tokenize`` option selects Porter
        stemming with Unicode normalization for language-aware matching.
        """
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS corpus_fts USING fts5(
                    chunk_id UNINDEXED,
                    source_id UNINDEXED,
                    source_title UNINDEXED,
                    source_type UNINDEXED,
                    citation_tier UNINDEXED,
                    text,
                    tokenize='porter unicode61'
                )
            """)
            await db.commit()

    async def upsert_chunks(self, chunks: list[Any]) -> int:
        """Insert or replace chunks into the FTS5 index.

        FTS5 does not support native UPSERT (INSERT OR REPLACE).  Instead
        we delete any existing row with the same chunk_id, then insert the
        new row.  This is wrapped in a transaction for atomicity.

        Parameters
        ----------
        chunks:
            List of DocumentChunk objects (from src/models/rag.py).  Each
            must have: chunk_id, source_id, source_title, source_type,
            citation_tier, text.

        Returns
        -------
        int
            Number of chunks successfully indexed.
        """
        if not chunks:
            return 0

        count = 0
        async with aiosqlite.connect(self._db_path) as db:
            for chunk in chunks:
                # FTS5 tables use a hidden rowid but don't support
                # INSERT OR REPLACE directly.  Delete-then-insert
                # simulates upsert behavior.
                await db.execute(
                    "DELETE FROM corpus_fts WHERE chunk_id = ?",
                    (chunk.chunk_id,),
                )
                await db.execute(
                    """INSERT INTO corpus_fts
                       (chunk_id, source_id, source_title, source_type,
                        citation_tier, text)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        chunk.chunk_id,
                        chunk.source_id,
                        chunk.source_title,
                        chunk.source_type,
                        chunk.citation_tier,
                        chunk.text,
                    ),
                )
                count += 1
            await db.commit()

        logger.info("bm25_upsert_chunks", count=count)
        return count

    async def delete_by_source(self, source_id: str) -> int:
        """Delete all chunks from a given source.

        Uses a standard DELETE on the FTS5 content table filtered by the
        UNINDEXED source_id column.

        Returns
        -------
        int
            Number of rows deleted.
        """
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "DELETE FROM corpus_fts WHERE source_id = ?",
                (source_id,),
            )
            await db.commit()
            deleted = cursor.rowcount
            logger.info(
                "bm25_delete_by_source",
                source_id=source_id,
                deleted_count=deleted,
            )
            return deleted

    async def search(self, query: str, top_k: int = 50) -> list[BM25Result]:
        """Search using FTS5 MATCH with BM25 ranking.

        The ``bm25()`` auxiliary function returns NEGATIVE scores where
        more negative = better match (lower retrieval cost).  Results are
        ordered ascending so the best matches come first.

        Query text is sanitized to prevent FTS5 syntax errors from user
        input containing special characters (*, ", etc.).  Each token is
        wrapped in double-quotes to force literal matching, then joined
        with implicit AND.

        Parameters
        ----------
        query:
            Raw search query text.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[BM25Result]
            BM25-ranked results, best match first.
        """
        if not query or not query.strip():
            return []

        # Sanitize: strip FTS5 operators and quote each token for literal match.
        # This prevents syntax errors from queries like 'DJ "Shadow"' or 'acid*'.
        tokens = query.strip().split()
        safe_tokens = []
        for token in tokens:
            # Remove characters that have special meaning in FTS5 query syntax
            cleaned = token.replace('"', "").replace("*", "").replace("(", "").replace(")", "")
            if cleaned:
                safe_tokens.append(f'"{cleaned}"')
        if not safe_tokens:
            return []

        # Join with space = implicit AND in FTS5
        fts_query = " ".join(safe_tokens)

        results: list[BM25Result] = []
        try:
            async with aiosqlite.connect(self._db_path) as db:
                db.row_factory = aiosqlite.Row
                # bm25() returns negative scores; ORDER BY score ASC puts
                # the strongest matches first (most negative = best).
                cursor = await db.execute(
                    """SELECT chunk_id, source_id, source_title, source_type,
                              citation_tier, text, bm25(corpus_fts) AS score
                       FROM corpus_fts
                       WHERE corpus_fts MATCH ?
                       ORDER BY score
                       LIMIT ?""",
                    (fts_query, top_k),
                )
                rows = await cursor.fetchall()
                for row in rows:
                    results.append(
                        BM25Result(
                            chunk_id=row["chunk_id"],
                            source_id=row["source_id"],
                            source_title=row["source_title"],
                            source_type=row["source_type"],
                            bm25_score=float(row["score"]),
                            text=row["text"],
                            citation_tier=int(row["citation_tier"]),
                        )
                    )

            logger.debug(
                "bm25_search",
                query=query[:80],
                fts_query=fts_query[:80],
                results=len(results),
                top_score=results[0].bm25_score if results else None,
            )
        except Exception as exc:
            # Graceful degradation: BM25 failure must never crash the pipeline.
            # Log and return empty results so semantic-only retrieval continues.
            logger.warning("bm25_search_failed", error=str(exc), query=query[:80])

        return results

    async def get_count(self) -> int:
        """Return total number of indexed chunks in the FTS5 table."""
        try:
            async with aiosqlite.connect(self._db_path) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM corpus_fts")
                row = await cursor.fetchone()
                return row[0] if row else 0
        except Exception:
            return 0

    async def rebuild_from_chromadb(self, vector_store: Any) -> int:
        """One-time migration: read all chunks from ChromaDB and index them.

        Called at startup when the BM25 index is empty but ChromaDB has data.
        Uses vector_store.list_all_metadata(include_documents=True) to page
        through all chunks without loading them all into memory at once.

        This ensures the BM25 index is populated from existing corpus data
        without requiring a full re-ingestion.

        Parameters
        ----------
        vector_store:
            The ChromaDB provider instance (must have list_all_metadata method).

        Returns
        -------
        int
            Number of chunks indexed.
        """
        logger.info("bm25_rebuild_from_chromadb_start")
        try:
            all_data = await vector_store.list_all_metadata(
                include_documents=True,
            )
        except Exception as exc:
            logger.error("bm25_rebuild_chromadb_read_failed", error=str(exc))
            return 0

        if not all_data:
            logger.info("bm25_rebuild_from_chromadb_empty")
            return 0

        count = 0
        async with aiosqlite.connect(self._db_path) as db:
            # Clear existing index before full rebuild to prevent stale entries
            await db.execute("DELETE FROM corpus_fts")

            for chunk_id, meta, doc_text in all_data:
                if not doc_text:
                    continue
                await db.execute(
                    """INSERT INTO corpus_fts
                       (chunk_id, source_id, source_title, source_type,
                        citation_tier, text)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        chunk_id,
                        meta.get("source_id", ""),
                        meta.get("source_title", ""),
                        meta.get("source_type", "unknown"),
                        int(meta.get("citation_tier", 6)),
                        doc_text,
                    ),
                )
                count += 1

            await db.commit()

        logger.info("bm25_rebuild_from_chromadb_complete", chunks_indexed=count)
        return count

    def is_available(self) -> bool:
        """Return True if the BM25 index database file exists on disk."""
        return self._db_path.exists()

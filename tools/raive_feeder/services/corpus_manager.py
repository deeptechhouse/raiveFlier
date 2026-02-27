"""Corpus management service for viewing, editing, and exporting the vector store.

# ─── DESIGN ────────────────────────────────────────────────────────────
#
# CorpusManager provides CRUD operations on the ChromaDB vector store
# that go beyond what the IngestionService offers.  It enables:
#
#   - Browsing all ingested sources with metadata
#   - Viewing all chunks for a specific source
#   - Searching the corpus with filters
#   - Editing chunk metadata (tags, tier) without re-embedding
#   - Deleting sources or entire source types
#   - Exporting the corpus as a tarball for backup/transfer
#   - Importing a corpus tarball
#
# All operations go through the IVectorStoreProvider interface so the
# corpus manager works with any vector store backend.
#
# Pattern: Facade (simplifies complex vector store operations for the API layer).
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path
from typing import Any

import structlog

from tools.raive_feeder.api.schemas import CorpusSourceSummary

logger = structlog.get_logger(logger_name=__name__)


class CorpusManager:
    """Manages the ChromaDB corpus for browsing, editing, and export.

    Parameters
    ----------
    vector_store:
        The IVectorStoreProvider instance (typically ChromaDBProvider).
    """

    def __init__(self, vector_store: Any) -> None:
        self._vs = vector_store

    async def list_sources(self) -> list[CorpusSourceSummary]:
        """List all ingested sources with summary metadata.

        Uses the vector store's public ``list_all_metadata()`` method to
        enumerate chunks without accessing provider internals.  Groups
        results by source_id to count chunks per source.
        """
        sources: dict[str, dict[str, Any]] = {}

        all_entries = await self._vs.list_all_metadata(
            include_documents=False,
        )

        for _chunk_id, meta, _doc in all_entries:
            sid = meta.get("source_id", "")
            if sid not in sources:
                sources[sid] = {
                    "source_id": sid,
                    "source_title": meta.get("source_title", ""),
                    "source_type": meta.get("source_type", "unknown"),
                    "author": meta.get("author"),
                    "citation_tier": int(meta.get("citation_tier", 6)),
                    "chunk_count": 0,
                }
            sources[sid]["chunk_count"] += 1

        return [
            CorpusSourceSummary(**data)
            for data in sorted(sources.values(), key=lambda s: s["source_title"])
        ]

    async def get_source_detail(self, source_id: str) -> dict[str, Any]:
        """Get all chunks and metadata for a specific source.

        Uses the vector store's public ``list_all_metadata()`` with a where
        filter to retrieve chunks for a single source without accessing
        provider internals.
        """
        entries = await self._vs.list_all_metadata(
            include_documents=True,
            where={"source_id": source_id},
        )

        chunks = [
            {
                "chunk_id": chunk_id,
                "text": doc,
                "metadata": meta,
            }
            for chunk_id, meta, doc in entries
        ]

        return {
            "source_id": source_id,
            "chunk_count": len(chunks),
            "chunks": chunks,
        }

    async def export_corpus(self, chromadb_dir: str) -> str:
        """Package the ChromaDB directory as a gzipped tarball.

        Returns the path to the created tarball.
        """
        chromadb_path = Path(chromadb_dir)
        if not chromadb_path.exists():
            raise FileNotFoundError(f"ChromaDB directory not found: {chromadb_dir}")

        tarball_path = tempfile.mktemp(suffix=".tar.gz")
        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add(str(chromadb_path), arcname="chromadb")

        size_mb = Path(tarball_path).stat().st_size / (1024 * 1024)
        logger.info("corpus_exported", path=tarball_path, size_mb=round(size_mb, 1))
        return tarball_path

    async def import_corpus(self, tarball_path: str, chromadb_dir: str) -> None:
        """Extract a corpus tarball into the ChromaDB directory.

        WARNING: This overwrites the existing corpus data.
        """
        chromadb_path = Path(chromadb_dir)

        with tarfile.open(tarball_path, "r:gz") as tar:
            # Security: validate all paths are within the extraction target.
            for member in tar.getmembers():
                member_path = Path(chromadb_path.parent / member.name).resolve()
                if not str(member_path).startswith(str(chromadb_path.parent.resolve())):
                    raise ValueError(f"Unsafe path in tarball: {member.name}")

            tar.extractall(path=str(chromadb_path.parent))

        logger.info("corpus_imported", path=chromadb_dir)

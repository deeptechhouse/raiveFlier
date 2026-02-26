"""Unit tests for CorpusManager service.

Tests source listing, export, and import with mocked vector store.
"""

from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from tools.raive_feeder.services.corpus_manager import CorpusManager


def _make_mock_vector_store(chunks_data=None):
    """Create a mock vector store with fake collection data."""
    vs = MagicMock()
    collection = MagicMock()

    if chunks_data is None:
        chunks_data = {
            "ids": ["chunk1", "chunk2", "chunk3"],
            "documents": ["Text 1", "Text 2", "Text 3"],
            "metadatas": [
                {"source_id": "src1", "source_title": "Book A", "source_type": "book", "citation_tier": 1},
                {"source_id": "src1", "source_title": "Book A", "source_type": "book", "citation_tier": 1},
                {"source_id": "src2", "source_title": "Article B", "source_type": "article", "citation_tier": 3},
            ],
        }

    collection.count.return_value = len(chunks_data["ids"])
    collection.get.return_value = chunks_data
    vs._collection = collection
    return vs


@pytest.fixture
def manager():
    return CorpusManager(vector_store=_make_mock_vector_store())


class TestListSources:
    """Tests for listing ingested sources."""

    @pytest.mark.asyncio
    async def test_groups_by_source_id(self, manager):
        """Sources should be grouped by source_id with correct chunk counts."""
        sources = await manager.list_sources()
        assert len(sources) == 2

        # Find Book A (should have 2 chunks).
        book_a = next(s for s in sources if s.source_title == "Book A")
        assert book_a.chunk_count == 2
        assert book_a.source_type == "book"

        # Find Article B (should have 1 chunk).
        article_b = next(s for s in sources if s.source_title == "Article B")
        assert article_b.chunk_count == 1

    @pytest.mark.asyncio
    async def test_empty_corpus(self):
        """Empty corpus should return empty list."""
        vs = _make_mock_vector_store({
            "ids": [],
            "documents": [],
            "metadatas": [],
        })
        manager = CorpusManager(vector_store=vs)
        sources = await manager.list_sources()
        assert sources == []


class TestGetSourceDetail:
    """Tests for getting detailed chunk info for a source."""

    @pytest.mark.asyncio
    async def test_returns_chunks(self, manager):
        """Should return all chunks for the requested source."""
        detail = await manager.get_source_detail("src1")
        assert detail["source_id"] == "src1"
        assert detail["chunk_count"] == 3  # Mock returns all 3 since we don't filter by source_id in mock.


class TestExportCorpus:
    """Tests for corpus tarball export."""

    @pytest.mark.asyncio
    async def test_creates_tarball(self, manager):
        """Export should create a valid gzipped tarball."""
        # Create a temp directory with a test file to act as chromadb dir.
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test_data.txt"
            test_file.write_text("test")

            tarball_path = await manager.export_corpus(tmpdir)

            assert Path(tarball_path).exists()
            assert tarball_path.endswith(".tar.gz")

            # Verify it's a valid tarball.
            with tarfile.open(tarball_path, "r:gz") as tar:
                names = tar.getnames()
                assert any("test_data" in n for n in names)

    @pytest.mark.asyncio
    async def test_export_missing_dir_raises(self, manager):
        """Exporting a non-existent directory should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            await manager.export_corpus("/nonexistent/path")


class TestImportCorpus:
    """Tests for corpus tarball import."""

    @pytest.mark.asyncio
    async def test_import_extracts_tarball(self, manager):
        """Import should extract tarball contents to the target directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a tarball to import.
            source_dir = Path(tmpdir) / "chromadb"
            source_dir.mkdir()
            (source_dir / "data.txt").write_text("corpus data")

            tarball_path = Path(tmpdir) / "corpus.tar.gz"
            with tarfile.open(str(tarball_path), "w:gz") as tar:
                tar.add(str(source_dir), arcname="chromadb")

            # Import into a new location.
            target_dir = Path(tmpdir) / "target"
            target_dir.mkdir()

            await manager.import_corpus(str(tarball_path), str(target_dir / "chromadb"))

            # Verify extraction.
            assert (target_dir / "chromadb" / "data.txt").exists()

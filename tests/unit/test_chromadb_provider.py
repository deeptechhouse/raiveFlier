"""Unit tests for ChromaDB vector store provider.

Tests cover add_chunks, query, query with filters, delete_by_source,
get_stats, is_available, cache invalidation, and the static helper
methods (_chunk_to_metadata, _metadata_to_chunk, _translate_filters,
_split_tags, _format_citation).
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.rag import DocumentChunk
from src.providers.vector_store.chromadb_provider import ChromaDBProvider
from src.utils.errors import RAGError


def _make_chunk(
    chunk_id: str = "c1",
    source_id: str = "s1",
    text: str = "test text",
    source_type: str = "book",
    source_title: str = "Test Book",
    citation_tier: int = 1,
    entity_tags: list[str] | None = None,
    geographic_tags: list[str] | None = None,
    genre_tags: list[str] | None = None,
    author: str | None = None,
    publication_date: date | None = None,
    page_number: str | None = None,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        source_id=source_id,
        source_type=source_type,
        source_title=source_title,
        text=text,
        citation_tier=citation_tier,
        entity_tags=entity_tags if entity_tags is not None else ["Carl Cox"],
        geographic_tags=geographic_tags if geographic_tags is not None else ["Berlin"],
        genre_tags=genre_tags if genre_tags is not None else [],
        author=author,
        publication_date=publication_date,
        page_number=page_number,
    )


class TestChromaDBProvider:
    @pytest.fixture()
    def mock_embedding_provider(self):
        from src.interfaces.embedding_provider import IEmbeddingProvider

        mock = MagicMock(spec=IEmbeddingProvider)
        mock.embed_single = AsyncMock(return_value=[0.1] * 128)
        mock.embed = AsyncMock(return_value=[[0.1] * 128])
        mock.get_dimension.return_value = 128
        mock.get_provider_name.return_value = "mock"
        mock.is_available.return_value = True
        return mock

    def test_get_provider_name(self, mock_embedding_provider, tmp_path) -> None:
        from src.providers.vector_store.chromadb_provider import ChromaDBProvider

        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma"),
            collection_name="test_collection",
        )
        assert provider.get_provider_name() == "chromadb"

    def test_is_available(self, mock_embedding_provider, tmp_path) -> None:
        from src.providers.vector_store.chromadb_provider import ChromaDBProvider

        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma"),
            collection_name="test_collection",
        )
        assert provider.is_available() is True

    @pytest.mark.asyncio
    async def test_add_chunks(self, mock_embedding_provider, tmp_path) -> None:
        from src.providers.vector_store.chromadb_provider import ChromaDBProvider

        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma"),
            collection_name="test_add",
        )

        chunk = _make_chunk()
        embeddings = [[0.1] * 128]
        count = await provider.add_chunks([chunk], embeddings)
        assert count == 1

    @pytest.mark.asyncio
    async def test_add_and_query(self, mock_embedding_provider, tmp_path) -> None:
        from src.providers.vector_store.chromadb_provider import ChromaDBProvider

        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma"),
            collection_name="test_query",
        )

        chunk = _make_chunk(text="Carl Cox played at Tresor Berlin")
        embeddings = [[0.1] * 128]
        await provider.add_chunks([chunk], embeddings)

        results = await provider.query("Carl Cox Tresor", top_k=5)
        assert len(results) >= 1
        assert "Carl Cox" in results[0].chunk.text

    @pytest.mark.asyncio
    async def test_query_empty_store(self, mock_embedding_provider, tmp_path) -> None:
        from src.providers.vector_store.chromadb_provider import ChromaDBProvider

        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma"),
            collection_name="test_empty",
        )

        results = await provider.query("test query", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_delete_by_source(self, mock_embedding_provider, tmp_path) -> None:
        from src.providers.vector_store.chromadb_provider import ChromaDBProvider

        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma"),
            collection_name="test_delete",
        )

        chunk = _make_chunk(source_id="source_to_delete")
        embeddings = [[0.1] * 128]
        await provider.add_chunks([chunk], embeddings)

        deleted = await provider.delete_by_source("source_to_delete")
        assert deleted >= 1

    @pytest.mark.asyncio
    async def test_get_stats(self, mock_embedding_provider, tmp_path) -> None:
        from src.providers.vector_store.chromadb_provider import ChromaDBProvider

        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma"),
            collection_name="test_stats",
        )

        chunk = _make_chunk()
        embeddings = [[0.1] * 128]
        await provider.add_chunks([chunk], embeddings)

        stats = await provider.get_stats()
        assert stats.total_chunks >= 1


# ======================================================================
# Extended tests — query with filters
# ======================================================================


class TestChromaDBQueryFilters:
    """Tests for filtered queries against ChromaDB."""

    @pytest.fixture()
    def mock_embedding_provider(self):
        from src.interfaces.embedding_provider import IEmbeddingProvider

        mock = MagicMock(spec=IEmbeddingProvider)
        mock.embed_single = AsyncMock(return_value=[0.1] * 128)
        mock.embed = AsyncMock(return_value=[[0.1] * 128])
        mock.get_dimension.return_value = 128
        mock.get_provider_name.return_value = "mock"
        mock.is_available.return_value = True
        return mock

    @pytest.mark.asyncio
    async def test_query_entity_tags_filter_not_applied_at_db_level(
        self, mock_embedding_provider, tmp_path
    ) -> None:
        """entity_tags $contains is NOT supported by ChromaDB — the provider
        skips this filter and returns all results.  Post-filtering happens
        in the route handler (routes.py), not at the database level."""
        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma_ef"),
            collection_name="test_entity_filter",
        )
        chunks = [
            _make_chunk(chunk_id="ef1", text="Carl Cox at Tresor", entity_tags=["Carl Cox", "Tresor"]),
            _make_chunk(chunk_id="ef2", text="Jeff Mills Detroit", entity_tags=["Jeff Mills"]),
        ]
        mock_embedding_provider.embed = AsyncMock(return_value=[[0.1] * 128, [0.2] * 128])
        embeddings = await mock_embedding_provider.embed([c.text for c in chunks])
        await provider.add_chunks(chunks, embeddings)

        # ChromaDB returns all results — entity_tags filter is ignored at
        # the provider level (post-filtered in Python at the route level).
        results = await provider.query(
            "techno DJ",
            filters={"entity_tags": {"$contains": "Carl Cox"}},
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_query_source_type_filter(
        self, mock_embedding_provider, tmp_path
    ) -> None:
        """Query with source_type $in filter restricts results."""
        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma_st"),
            collection_name="test_source_filter",
        )
        chunks = [
            _make_chunk(chunk_id="sf1", text="Book passage", source_type="book"),
            _make_chunk(chunk_id="sf2", text="Article passage", source_type="article"),
        ]
        mock_embedding_provider.embed = AsyncMock(return_value=[[0.1] * 128, [0.2] * 128])
        embeddings = await mock_embedding_provider.embed([c.text for c in chunks])
        await provider.add_chunks(chunks, embeddings)

        results = await provider.query(
            "passage",
            filters={"source_type": {"$in": ["book"]}},
        )
        for r in results:
            assert r.chunk.source_type == "book"

    @pytest.mark.asyncio
    async def test_query_geographic_tags_filter_not_applied_at_db_level(
        self, mock_embedding_provider, tmp_path
    ) -> None:
        """geographic_tags $contains is NOT supported by ChromaDB — the
        provider skips this filter and returns all results.  Post-filtering
        happens in the route handler (routes.py)."""
        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma_gf"),
            collection_name="test_geo_filter",
        )
        chunks = [
            _make_chunk(chunk_id="gf1", text="Berlin techno", geographic_tags=["Berlin"]),
            _make_chunk(chunk_id="gf2", text="Detroit techno", geographic_tags=["Detroit"]),
        ]
        mock_embedding_provider.embed = AsyncMock(return_value=[[0.1] * 128, [0.2] * 128])
        embeddings = await mock_embedding_provider.embed([c.text for c in chunks])
        await provider.add_chunks(chunks, embeddings)

        # ChromaDB returns all results — geographic_tags filter is ignored
        # at the provider level (post-filtered in Python at the route level).
        results = await provider.query(
            "techno",
            filters={"geographic_tags": {"$contains": "Berlin"}},
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_query_result_has_citation_string(
        self, mock_embedding_provider, tmp_path
    ) -> None:
        """Each result has a non-empty formatted_citation."""
        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma_cit"),
            collection_name="test_citation",
        )
        chunk = _make_chunk(
            chunk_id="cit1",
            text="Tresor history",
            source_title="Berlin Techno Guide",
            author="Test Writer",
            citation_tier=2,
        )
        await provider.add_chunks([chunk], [[0.1] * 128])

        results = await provider.query("Tresor")
        assert len(results) > 0
        assert "Berlin Techno Guide" in results[0].formatted_citation

    @pytest.mark.asyncio
    async def test_query_similarity_score_range(
        self, mock_embedding_provider, tmp_path
    ) -> None:
        """All result similarity scores are between 0.0 and 1.0."""
        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma_sim"),
            collection_name="test_sim_range",
        )
        chunk = _make_chunk(chunk_id="sim1", text="Techno music history")
        await provider.add_chunks([chunk], [[0.1] * 128])

        results = await provider.query("Techno")
        for r in results:
            assert 0.0 <= r.similarity_score <= 1.0


# ======================================================================
# Multi-chunk per-source dedup tests
# ======================================================================


class TestMultiChunkPerSourceDedup:
    """Tests for max_per_source dedup in query()."""

    @pytest.fixture()
    def mock_embedding_provider(self):
        from src.interfaces.embedding_provider import IEmbeddingProvider

        mock = MagicMock(spec=IEmbeddingProvider)
        mock.embed_single = AsyncMock(return_value=[0.1] * 128)
        mock.embed = AsyncMock(return_value=[[0.1] * 128])
        mock.get_dimension.return_value = 128
        mock.get_provider_name.return_value = "mock"
        mock.is_available.return_value = True
        return mock

    @pytest.mark.asyncio
    async def test_returns_multiple_chunks_per_source(
        self, mock_embedding_provider, tmp_path
    ) -> None:
        """With max_per_source=3, up to 3 chunks from the same source are returned."""
        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma_multi"),
            collection_name="test_multi_chunk",
        )

        # Add 5 chunks from the same source
        chunks = [
            _make_chunk(chunk_id=f"c{i}", source_id="transcript-001", text=f"Topic {i} content")
            for i in range(5)
        ]
        embeddings = [[0.1] * 128 for _ in range(5)]
        mock_embedding_provider.embed = AsyncMock(return_value=embeddings)
        await provider.add_chunks(chunks, embeddings)

        results = await provider.query("Topic content", top_k=10, max_per_source=3)
        assert len(results) <= 3
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_max_per_source_one_gives_single_result(
        self, mock_embedding_provider, tmp_path
    ) -> None:
        """max_per_source=1 should give at most 1 result per source (old behavior)."""
        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma_single"),
            collection_name="test_single_chunk",
        )

        chunks = [
            _make_chunk(chunk_id=f"c{i}", source_id="same-source", text=f"Content {i}")
            for i in range(3)
        ]
        embeddings = [[0.1] * 128 for _ in range(3)]
        mock_embedding_provider.embed = AsyncMock(return_value=embeddings)
        await provider.add_chunks(chunks, embeddings)

        results = await provider.query("Content", top_k=10, max_per_source=1)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_different_sources_not_limited(
        self, mock_embedding_provider, tmp_path
    ) -> None:
        """Chunks from different sources are not affected by per-source limit."""
        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma_diff"),
            collection_name="test_diff_sources",
        )

        chunks = [
            _make_chunk(chunk_id=f"c{i}", source_id=f"source-{i}", text=f"Detroit techno content {i}")
            for i in range(5)
        ]
        embeddings = [[0.1] * 128 for _ in range(5)]
        mock_embedding_provider.embed = AsyncMock(return_value=embeddings)
        await provider.add_chunks(chunks, embeddings)

        results = await provider.query("Detroit techno", top_k=10, max_per_source=3)
        assert len(results) == 5


# ======================================================================
# Extended tests — cache invalidation
# ======================================================================


class TestChromaDBCacheInvalidation:
    """Tests for the stats cache invalidation logic."""

    @pytest.fixture()
    def mock_embedding_provider(self):
        from src.interfaces.embedding_provider import IEmbeddingProvider

        mock = MagicMock(spec=IEmbeddingProvider)
        mock.embed_single = AsyncMock(return_value=[0.1] * 128)
        mock.embed = AsyncMock(return_value=[[0.1] * 128])
        mock.get_dimension.return_value = 128
        mock.get_provider_name.return_value = "mock"
        mock.is_available.return_value = True
        return mock

    @pytest.mark.asyncio
    async def test_add_chunks_invalidates_cache(
        self, mock_embedding_provider, tmp_path
    ) -> None:
        """Adding chunks clears the cached stats."""
        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma_ci1"),
            collection_name="test_cache_inv_add",
        )
        # Populate cache
        await provider.get_stats()
        assert provider._cached_stats is not None

        chunk = _make_chunk(chunk_id="ci1")
        await provider.add_chunks([chunk], [[0.1] * 128])
        assert provider._cached_stats is None

    @pytest.mark.asyncio
    async def test_delete_invalidates_cache(
        self, mock_embedding_provider, tmp_path
    ) -> None:
        """Deleting chunks clears the cached stats."""
        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma_ci2"),
            collection_name="test_cache_inv_del",
        )
        chunk = _make_chunk(chunk_id="ci2", source_id="src-ci2")
        await provider.add_chunks([chunk], [[0.1] * 128])

        # Populate cache
        await provider.get_stats()
        assert provider._cached_stats is not None

        await provider.delete_by_source("src-ci2")
        assert provider._cached_stats is None

    @pytest.mark.asyncio
    async def test_stats_cache_returns_same_object(
        self, mock_embedding_provider, tmp_path
    ) -> None:
        """Consecutive get_stats calls without mutations return the same object."""
        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma_ci3"),
            collection_name="test_cache_same",
        )
        chunk = _make_chunk(chunk_id="ci3")
        await provider.add_chunks([chunk], [[0.1] * 128])

        stats1 = await provider.get_stats()
        stats2 = await provider.get_stats()
        assert stats1 is stats2

    @pytest.mark.asyncio
    async def test_stats_empty_collection(
        self, mock_embedding_provider, tmp_path
    ) -> None:
        """Stats on an empty collection return zero counts."""
        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma_ci4"),
            collection_name="test_empty_stats",
        )
        stats = await provider.get_stats()
        assert stats.total_chunks == 0
        assert stats.total_sources == 0
        assert stats.sources_by_type == {}

    @pytest.mark.asyncio
    async def test_stats_counts_sources_and_types(
        self, mock_embedding_provider, tmp_path
    ) -> None:
        """Stats correctly count sources, types, entity tags, and geographic tags."""
        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma_ci5"),
            collection_name="test_stats_detail",
        )
        chunks = [
            _make_chunk(
                chunk_id="sd1", source_id="src-a", source_type="book",
                entity_tags=["Carl Cox"], geographic_tags=["Berlin"],
            ),
            _make_chunk(
                chunk_id="sd2", source_id="src-a", source_type="book",
                entity_tags=["Jeff Mills"], geographic_tags=["Detroit"],
            ),
            _make_chunk(
                chunk_id="sd3", source_id="src-b", source_type="article",
                entity_tags=["Carl Cox"], geographic_tags=[],
            ),
        ]
        mock_embedding_provider.embed = AsyncMock(
            return_value=[[0.1] * 128, [0.2] * 128, [0.3] * 128]
        )
        embeddings = await mock_embedding_provider.embed([c.text for c in chunks])
        await provider.add_chunks(chunks, embeddings)

        stats = await provider.get_stats()
        assert stats.total_chunks == 3
        assert stats.total_sources == 2
        assert stats.sources_by_type.get("book") == 2
        assert stats.sources_by_type.get("article") == 1
        assert stats.entity_tag_count >= 2
        assert stats.geographic_tag_count >= 1


# ======================================================================
# Extended tests — delete_by_source edge cases
# ======================================================================


class TestChromaDBDeleteExtended:
    """Extended delete_by_source tests."""

    @pytest.fixture()
    def mock_embedding_provider(self):
        from src.interfaces.embedding_provider import IEmbeddingProvider

        mock = MagicMock(spec=IEmbeddingProvider)
        mock.embed_single = AsyncMock(return_value=[0.1] * 128)
        mock.embed = AsyncMock(return_value=[[0.1] * 128])
        mock.get_dimension.return_value = 128
        mock.get_provider_name.return_value = "mock"
        mock.is_available.return_value = True
        return mock

    @pytest.mark.asyncio
    async def test_delete_nonexistent_source(
        self, mock_embedding_provider, tmp_path
    ) -> None:
        """Deleting a source that doesn't exist returns 0."""
        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma_de1"),
            collection_name="test_del_none",
        )
        deleted = await provider.delete_by_source("nonexistent-source")
        assert deleted == 0

    @pytest.mark.asyncio
    async def test_delete_one_source_keeps_other(
        self, mock_embedding_provider, tmp_path
    ) -> None:
        """Deleting one source leaves other sources intact."""
        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma_de2"),
            collection_name="test_del_partial",
        )
        chunks = [
            _make_chunk(chunk_id="dp1", source_id="delete-me", text="Delete this"),
            _make_chunk(chunk_id="dp2", source_id="delete-me", text="Delete also"),
            _make_chunk(chunk_id="dp3", source_id="keep-me", text="Keep this"),
        ]
        mock_embedding_provider.embed = AsyncMock(
            return_value=[[0.1] * 128, [0.2] * 128, [0.3] * 128]
        )
        embeddings = await mock_embedding_provider.embed([c.text for c in chunks])
        await provider.add_chunks(chunks, embeddings)

        deleted = await provider.delete_by_source("delete-me")
        assert deleted == 2

        stats = await provider.get_stats()
        assert stats.total_chunks == 1


# ======================================================================
# delete_by_source_type tests
# ======================================================================


class TestDeleteBySourceType:
    """Tests for delete_by_source_type()."""

    @pytest.fixture()
    def mock_embedding_provider(self):
        from src.interfaces.embedding_provider import IEmbeddingProvider

        mock = MagicMock(spec=IEmbeddingProvider)
        mock.embed_single = AsyncMock(return_value=[0.1] * 128)
        mock.embed = AsyncMock(return_value=[[0.1] * 128])
        mock.get_dimension.return_value = 128
        mock.get_provider_name.return_value = "mock"
        mock.is_available.return_value = True
        return mock

    @pytest.mark.asyncio
    async def test_deletes_all_chunks_of_type(
        self, mock_embedding_provider, tmp_path
    ) -> None:
        """All chunks with the given source_type are deleted."""
        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma_dst1"),
            collection_name="test_dst_all",
        )
        chunks = [
            _make_chunk(chunk_id="dst1", source_id="int-1", source_type="interview"),
            _make_chunk(chunk_id="dst2", source_id="int-1", source_type="interview"),
            _make_chunk(chunk_id="dst3", source_id="int-2", source_type="interview"),
            _make_chunk(chunk_id="dst4", source_id="ref-1", source_type="reference"),
        ]
        mock_embedding_provider.embed = AsyncMock(
            return_value=[[0.1] * 128, [0.2] * 128, [0.3] * 128, [0.4] * 128]
        )
        embeddings = await mock_embedding_provider.embed([c.text for c in chunks])
        await provider.add_chunks(chunks, embeddings)

        deleted = await provider.delete_by_source_type("interview")
        assert deleted == 3

        stats = await provider.get_stats()
        assert stats.total_chunks == 1
        assert stats.sources_by_type.get("interview", 0) == 0
        assert stats.sources_by_type.get("reference") == 1

    @pytest.mark.asyncio
    async def test_nonexistent_type_returns_zero(
        self, mock_embedding_provider, tmp_path
    ) -> None:
        """Purging a type that doesn't exist returns 0."""
        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma_dst2"),
            collection_name="test_dst_none",
        )
        deleted = await provider.delete_by_source_type("nonexistent")
        assert deleted == 0

    @pytest.mark.asyncio
    async def test_invalidates_stats_cache(
        self, mock_embedding_provider, tmp_path
    ) -> None:
        """delete_by_source_type invalidates the stats cache."""
        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma_dst3"),
            collection_name="test_dst_cache",
        )
        chunk = _make_chunk(chunk_id="dst5", source_type="interview")
        await provider.add_chunks([chunk], [[0.1] * 128])

        await provider.get_stats()
        assert provider._cached_stats is not None

        await provider.delete_by_source_type("interview")
        assert provider._cached_stats is None


# ======================================================================
# Static helper methods
# ======================================================================


class TestStaticHelpers:
    """Tests for ChromaDBProvider static helper methods."""

    def test_split_tags_empty_string(self) -> None:
        assert ChromaDBProvider._split_tags("") == []

    def test_split_tags_none_value(self) -> None:
        assert ChromaDBProvider._split_tags(None) == []

    def test_split_tags_non_string_type(self) -> None:
        assert ChromaDBProvider._split_tags(123) == []

    def test_split_tags_comma_separated(self) -> None:
        result = ChromaDBProvider._split_tags("Carl Cox,Jeff Mills,Derrick May")
        assert result == ["Carl Cox", "Jeff Mills", "Derrick May"]

    def test_split_tags_with_whitespace(self) -> None:
        result = ChromaDBProvider._split_tags(" Carl Cox , Jeff Mills ")
        assert result == ["Carl Cox", "Jeff Mills"]

    def test_split_tags_single_value(self) -> None:
        result = ChromaDBProvider._split_tags("Carl Cox")
        assert result == ["Carl Cox"]

    def test_chunk_to_metadata_basic(self) -> None:
        """Converts a chunk to a flat dict of scalar values."""
        chunk = _make_chunk(
            entity_tags=["Carl Cox"],
            geographic_tags=["Berlin"],
            genre_tags=["techno"],
            author="Test Author",
            page_number="42",
            publication_date=date(1997, 3, 15),
        )
        meta = ChromaDBProvider._chunk_to_metadata(chunk)

        assert meta["source_id"] == "s1"
        assert meta["source_title"] == "Test Book"
        assert meta["source_type"] == "book"
        assert meta["citation_tier"] == 1
        assert meta["entity_tags"] == "Carl Cox"
        assert meta["geographic_tags"] == "Berlin"
        assert meta["genre_tags"] == "techno"
        assert meta["author"] == "Test Author"
        assert meta["page_number"] == "42"
        assert meta["publication_date"] == "1997-03-15"

    def test_chunk_to_metadata_no_optional_fields(self) -> None:
        """Optional fields (author, publication_date, page_number) are omitted when None."""
        chunk = _make_chunk(author=None, publication_date=None, page_number=None)
        meta = ChromaDBProvider._chunk_to_metadata(chunk)
        assert "author" not in meta
        assert "publication_date" not in meta
        assert "page_number" not in meta

    def test_metadata_to_chunk_roundtrip(self) -> None:
        """_chunk_to_metadata -> _metadata_to_chunk preserves key fields."""
        original = _make_chunk(
            entity_tags=["Carl Cox", "Tresor"],
            geographic_tags=["Berlin"],
            genre_tags=["techno"],
            author="Simon Reynolds",
            publication_date=date(1998, 5, 1),
            page_number="142",
            citation_tier=1,
        )
        meta = ChromaDBProvider._chunk_to_metadata(original)
        reconstructed = ChromaDBProvider._metadata_to_chunk(meta, original.text)

        assert reconstructed.source_id == original.source_id
        assert reconstructed.source_title == original.source_title
        assert reconstructed.source_type == original.source_type
        assert reconstructed.author == original.author
        assert reconstructed.publication_date == original.publication_date
        assert reconstructed.page_number == original.page_number
        assert reconstructed.citation_tier == original.citation_tier
        assert reconstructed.entity_tags == original.entity_tags
        assert reconstructed.geographic_tags == original.geographic_tags
        assert reconstructed.genre_tags == original.genre_tags

    def test_metadata_to_chunk_no_publication_date(self) -> None:
        """When publication_date is absent, it reconstructs as None."""
        meta = {
            "source_id": "s1",
            "source_title": "Test",
            "source_type": "article",
            "citation_tier": 3,
            "entity_tags": "",
            "geographic_tags": "",
            "genre_tags": "",
        }
        chunk = ChromaDBProvider._metadata_to_chunk(meta, "Some text")
        assert chunk.publication_date is None
        assert chunk.author is None

    def test_translate_filters_empty(self) -> None:
        assert ChromaDBProvider._translate_filters({}) is None

    def test_translate_filters_single_date(self) -> None:
        result = ChromaDBProvider._translate_filters({"date": {"$lte": "1997-03-15"}})
        assert result == {"publication_date": {"$lte": "1997-03-15"}}

    def test_translate_filters_skips_entity_tags(self) -> None:
        """entity_tags $contains is NOT a valid ChromaDB operator — it is
        post-filtered in Python (routes.py), so _translate_filters skips it."""
        result = ChromaDBProvider._translate_filters({"entity_tags": {"$contains": "Carl Cox"}})
        assert result is None

    def test_translate_filters_skips_geographic_tags(self) -> None:
        """geographic_tags $contains is NOT a valid ChromaDB operator — it is
        post-filtered in Python (routes.py), so _translate_filters skips it."""
        result = ChromaDBProvider._translate_filters({"geographic_tags": {"$contains": "Berlin"}})
        assert result is None

    def test_translate_filters_single_source_type(self) -> None:
        result = ChromaDBProvider._translate_filters({"source_type": {"$in": ["book", "article"]}})
        assert result == {"source_type": {"$in": ["book", "article"]}}

    def test_translate_filters_multiple_produces_and(self) -> None:
        """Multiple ChromaDB-supported filter keys produce a $and clause.
        entity_tags is skipped (post-filtered in Python), so only the
        date + source_type combo produces $and here."""
        result = ChromaDBProvider._translate_filters(
            {
                "date": {"$lte": "1997-03-15"},
                "source_type": {"$in": ["book"]},
            }
        )
        assert "$and" in result
        assert len(result["$and"]) == 2

    def test_translate_filters_ignores_empty_values(self) -> None:
        """Filters with empty/falsy nested dicts are ignored."""
        result = ChromaDBProvider._translate_filters({"date": {}, "entity_tags": None})
        assert result is None

    def test_translate_filters_ignores_non_dict_values(self) -> None:
        """Non-dict filter values are skipped."""
        result = ChromaDBProvider._translate_filters({"date": "not-a-dict"})
        assert result is None

    def test_format_citation_full(self) -> None:
        """_format_citation includes all available metadata."""
        chunk = _make_chunk(
            source_title="Energy Flash",
            author="Simon Reynolds",
            page_number="142",
            publication_date=date(1998, 5, 1),
            citation_tier=1,
        )
        citation = ChromaDBProvider._format_citation(chunk, 0.95)
        assert "Energy Flash" in citation
        assert "Simon Reynolds" in citation
        assert "p.142" in citation
        assert "1998" in citation
        assert "[Tier 1]" in citation

    def test_format_citation_minimal(self) -> None:
        """_format_citation works with only source_title and tier."""
        chunk = _make_chunk(source_title="Unknown Source", citation_tier=3)
        citation = ChromaDBProvider._format_citation(chunk, 0.5)
        assert "Unknown Source" in citation
        assert "[Tier 3]" in citation

    def test_format_citation_no_author_no_page(self) -> None:
        """Citation without author or page still includes title, year, and tier."""
        chunk = _make_chunk(
            source_title="Acid House",
            publication_date=date(1988, 1, 1),
            citation_tier=2,
        )
        citation = ChromaDBProvider._format_citation(chunk, 0.8)
        assert "Acid House" in citation
        assert "1988" in citation
        assert "[Tier 2]" in citation


# ======================================================================
# add_chunks — mismatched lengths and upsert
# ======================================================================


class TestAddChunksExtended:
    """Extended add_chunks tests for edge cases."""

    @pytest.fixture()
    def mock_embedding_provider(self):
        from src.interfaces.embedding_provider import IEmbeddingProvider

        mock = MagicMock(spec=IEmbeddingProvider)
        mock.embed_single = AsyncMock(return_value=[0.1] * 128)
        mock.embed = AsyncMock(return_value=[[0.1] * 128])
        mock.get_dimension.return_value = 128
        mock.get_provider_name.return_value = "mock"
        mock.is_available.return_value = True
        return mock

    @pytest.mark.asyncio
    async def test_add_empty_list_returns_zero(
        self, mock_embedding_provider, tmp_path
    ) -> None:
        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma_ae1"),
            collection_name="test_add_empty",
        )
        result = await provider.add_chunks([], [])
        assert result == 0

    @pytest.mark.asyncio
    async def test_add_mismatched_raises_value_error(
        self, mock_embedding_provider, tmp_path
    ) -> None:
        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma_ae2"),
            collection_name="test_add_mismatch",
        )
        chunk = _make_chunk(chunk_id="mm1")
        with pytest.raises(ValueError, match="mismatch"):
            await provider.add_chunks([chunk], [])

    @pytest.mark.asyncio
    async def test_upsert_updates_existing_chunk(
        self, mock_embedding_provider, tmp_path
    ) -> None:
        """Adding a chunk with the same chunk_id overwrites (upsert behavior)."""
        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_provider,
            persist_directory=str(tmp_path / "chroma_ae3"),
            collection_name="test_upsert",
        )
        chunk_v1 = _make_chunk(chunk_id="upsert1", text="Original text")
        await provider.add_chunks([chunk_v1], [[0.1] * 128])

        chunk_v2 = _make_chunk(chunk_id="upsert1", text="Updated text")
        await provider.add_chunks([chunk_v2], [[0.2] * 128])

        # Still only one chunk in the store
        stats = await provider.get_stats()
        assert stats.total_chunks == 1


# ======================================================================
# Embedding dimension validation
# ======================================================================


class TestEmbeddingDimensionValidation:
    """Tests for the startup embedding dimension check."""

    @pytest.fixture()
    def mock_embedding_128(self):
        from src.interfaces.embedding_provider import IEmbeddingProvider

        mock = MagicMock(spec=IEmbeddingProvider)
        mock.embed_single = AsyncMock(return_value=[0.1] * 128)
        mock.embed = AsyncMock(return_value=[[0.1] * 128])
        mock.get_dimension.return_value = 128
        mock.get_provider_name.return_value = "mock-128"
        mock.is_available.return_value = True
        return mock

    @pytest.fixture()
    def mock_embedding_256(self):
        from src.interfaces.embedding_provider import IEmbeddingProvider

        mock = MagicMock(spec=IEmbeddingProvider)
        mock.embed_single = AsyncMock(return_value=[0.1] * 256)
        mock.embed = AsyncMock(return_value=[[0.1] * 256])
        mock.get_dimension.return_value = 256
        mock.get_provider_name.return_value = "mock-256"
        mock.is_available.return_value = True
        return mock

    def test_empty_collection_passes_validation(self, mock_embedding_128, tmp_path) -> None:
        """Empty collection skips dimension check — nothing to compare against."""
        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_128,
            persist_directory=str(tmp_path / "chroma_dv1"),
            collection_name="test_dim_empty",
        )
        assert provider.is_available()

    @pytest.mark.asyncio
    async def test_matching_dimensions_pass(self, mock_embedding_128, tmp_path) -> None:
        """Same dimension corpus + provider passes validation."""
        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_128,
            persist_directory=str(tmp_path / "chroma_dv2"),
            collection_name="test_dim_match",
        )
        chunk = _make_chunk(chunk_id="dv1")
        await provider.add_chunks([chunk], [[0.1] * 128])

        # Creating a new provider instance against the same data should succeed
        provider2 = ChromaDBProvider(
            embedding_provider=mock_embedding_128,
            persist_directory=str(tmp_path / "chroma_dv2"),
            collection_name="test_dim_match",
        )
        assert provider2.is_available()

    @pytest.mark.asyncio
    async def test_mismatched_dimensions_raise_rag_error(
        self, mock_embedding_128, mock_embedding_256, tmp_path
    ) -> None:
        """Mismatched dimensions between corpus and provider raises RAGError."""
        # Build corpus with 128-dim embeddings
        provider = ChromaDBProvider(
            embedding_provider=mock_embedding_128,
            persist_directory=str(tmp_path / "chroma_dv3"),
            collection_name="test_dim_mismatch",
        )
        chunk = _make_chunk(chunk_id="dv2")
        await provider.add_chunks([chunk], [[0.1] * 128])

        # Try to create a new provider with 256-dim — should fail
        with pytest.raises(RAGError, match="dimension mismatch"):
            ChromaDBProvider(
                embedding_provider=mock_embedding_256,
                persist_directory=str(tmp_path / "chroma_dv3"),
                collection_name="test_dim_mismatch",
            )

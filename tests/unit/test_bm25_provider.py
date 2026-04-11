"""Unit tests for the BM25Provider (SQLite FTS5 keyword search).

# ─── MODULE OVERVIEW ────────────────────────────────────────────────
# Tests the BM25Provider's core operations: table initialization,
# chunk ingestion via upsert_chunks, keyword search with BM25 ranking,
# deletion by source, and count accuracy.  Each test uses a temporary
# SQLite database created by pytest's tmp_path fixture for isolation.
#
# The BM25Provider complements the semantic vector store (ChromaDB)
# by providing exact keyword matching.  These tests verify the FTS5
# index behaves correctly independent of the vector store.
# ────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import pytest_asyncio

from src.providers.vector_store.bm25_provider import BM25Provider

# ─── Helper: lightweight chunk-like objects for FTS5 ingestion ─────


@dataclass
class FakeChunk:
    """Minimal chunk object matching the attributes BM25Provider.upsert_chunks expects."""

    chunk_id: str
    source_id: str
    source_title: str
    source_type: str
    text: str
    citation_tier: int = 3


# ─── Fixtures ──────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def provider(tmp_path: Path) -> BM25Provider:
    """Create and initialize a BM25Provider with a temporary database."""
    db_path = str(tmp_path / "test_bm25.db")
    prov = BM25Provider(db_path=db_path)
    await prov.initialize()
    return prov


def _make_chunks() -> list[FakeChunk]:
    """Build 3 chunks about different topics for search tests."""
    return [
        FakeChunk(
            chunk_id="chunk-detroit-1",
            source_id="src-detroit",
            source_title="Energy Flash",
            source_type="book",
            text=(
                "Detroit techno emerged in the mid-1980s from the Belleville Three: "
                "Juan Atkins, Derrick May, and Kevin Saunderson."
            ),
            citation_tier=1,
        ),
        FakeChunk(
            chunk_id="chunk-berlin-1",
            source_id="src-berlin",
            source_title="Der Klang der Familie",
            source_type="book",
            text=(
                "The Berlin techno scene developed after the fall of the Wall in 1989. "
                "Tresor opened in the vault of a former department store."
            ),
            citation_tier=2,
        ),
        FakeChunk(
            chunk_id="chunk-chicago-1",
            source_id="src-chicago",
            source_title="Last Night a DJ Saved My Life",
            source_type="book",
            text=(
                "Chicago house music was built in the clubs of the South Side. "
                "Frankie Knuckles held legendary residencies at the Warehouse."
            ),
            citation_tier=1,
        ),
    ]


# ─── Tests ─────────────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_initialize_creates_table(provider: BM25Provider) -> None:
    """After initialization, the FTS5 table exists and count returns 0."""
    count = await provider.get_count()
    assert count == 0


@pytest.mark.asyncio()
async def test_upsert_and_search(provider: BM25Provider) -> None:
    """Upserting 3 chunks then searching for a keyword finds the right chunk."""
    chunks = _make_chunks()
    inserted = await provider.upsert_chunks(chunks)
    assert inserted == 3

    results = await provider.search("Tresor", top_k=5)
    assert len(results) >= 1
    # The Berlin chunk mentions "Tresor" — it should appear in results
    texts = [r.text for r in results]
    assert any("Tresor" in t for t in texts)


@pytest.mark.asyncio()
async def test_search_returns_empty_for_no_match(provider: BM25Provider) -> None:
    """Searching for a term not present in any chunk returns an empty list."""
    chunks = _make_chunks()
    await provider.upsert_chunks(chunks)

    results = await provider.search("gabber", top_k=5)
    assert results == []


@pytest.mark.asyncio()
async def test_delete_by_source(provider: BM25Provider) -> None:
    """Deleting by source_id removes only chunks from that source."""
    # Two chunks with the same source_id
    chunks = [
        FakeChunk(
            chunk_id="chunk-a",
            source_id="src-same",
            source_title="Same Source",
            source_type="article",
            text="First chunk from the same source.",
        ),
        FakeChunk(
            chunk_id="chunk-b",
            source_id="src-same",
            source_title="Same Source",
            source_type="article",
            text="Second chunk from the same source.",
        ),
        FakeChunk(
            chunk_id="chunk-c",
            source_id="src-other",
            source_title="Other Source",
            source_type="book",
            text="Chunk from a different source.",
        ),
    ]
    await provider.upsert_chunks(chunks)
    assert await provider.get_count() == 3

    deleted = await provider.delete_by_source("src-same")
    assert deleted == 2
    assert await provider.get_count() == 1


@pytest.mark.asyncio()
async def test_get_count_accurate(provider: BM25Provider) -> None:
    """get_count returns the exact number of indexed chunks."""
    chunks = _make_chunks()
    await provider.upsert_chunks(chunks)
    assert await provider.get_count() == 3

    # Add one more
    extra = FakeChunk(
        chunk_id="chunk-extra",
        source_id="src-extra",
        source_title="Extra",
        source_type="article",
        text="An extra chunk for counting.",
    )
    await provider.upsert_chunks([extra])
    assert await provider.get_count() == 4


@pytest.mark.asyncio()
async def test_search_ranking_order(provider: BM25Provider) -> None:
    """A chunk containing the search term more often should rank higher.

    BM25 gives higher scores to documents where the query term appears
    more frequently (term frequency component).  We verify this by
    inserting one chunk with "techno" once and another with "techno"
    three times, then checking the 3x chunk ranks first.
    """
    once = FakeChunk(
        chunk_id="chunk-once",
        source_id="src-once",
        source_title="Once Source",
        source_type="article",
        text="This article mentions techno in passing.",
    )
    thrice = FakeChunk(
        chunk_id="chunk-thrice",
        source_id="src-thrice",
        source_title="Thrice Source",
        source_type="article",
        text="Techno is the heart of techno culture. Techno forever.",
    )
    await provider.upsert_chunks([once, thrice])

    results = await provider.search("techno", top_k=5)
    assert len(results) == 2

    # BM25 in FTS5 returns NEGATIVE scores (more negative = better match).
    # The chunk with "techno" 3 times ranks first (most negative score).
    assert results[0].chunk_id == "chunk-thrice"
    assert results[0].bm25_score <= results[1].bm25_score

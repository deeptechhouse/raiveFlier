"""Unit tests for the relationship graph storage in SQLiteFlierHistoryProvider.

# ─── MODULE OVERVIEW ────────────────────────────────────────────────
# Tests the Phase 3 relationship graph methods added to the flier
# history provider: store_relationship_edges, get_label_mates, and
# get_co_billing_edges.
#
# The relationship graph persists cross-flier entity relationships
# (label mates, co-billing, shared scene participation) discovered
# during interconnection analysis.  UPSERT semantics mean storing
# the same edge again increments its strength — relationships seen
# across multiple fliers are reinforced.
#
# Each test uses a temporary SQLite database for isolation.
# ────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

from src.providers.flier_history.sqlite_flier_history_provider import (
    SQLiteFlierHistoryProvider,
)

# ─── Fixtures ──────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def provider(tmp_path: Path) -> SQLiteFlierHistoryProvider:
    """Create and initialize a provider with a temporary database."""
    db_path = tmp_path / "test_relationship_graph.db"
    prov = SQLiteFlierHistoryProvider(db_path=db_path)
    await prov.initialize()
    return prov


# ─── Tests ─────────────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_store_and_retrieve_label_mates(
    provider: SQLiteFlierHistoryProvider,
) -> None:
    """Store a label_mate edge, then retrieve it by artist name."""
    edges = [
        {
            "source_entity": "Carl Cox",
            "target_entity": "Adam Beyer",
            "relationship_type": "label_mate",
            "evidence": {"label": "Drumcode"},
            "strength": 1.0,
        },
    ]
    stored = await provider.store_relationship_edges(edges)
    assert stored == 1

    results = await provider.get_label_mates(["Carl Cox"])
    assert len(results) == 1
    assert results[0]["source_entity"] == "Carl Cox"
    assert results[0]["target_entity"] == "Adam Beyer"
    assert results[0]["relationship_type"] == "label_mate"
    assert results[0]["evidence"]["label"] == "Drumcode"
    assert results[0]["strength"] == 1.0


@pytest.mark.asyncio()
async def test_upsert_increments_strength(
    provider: SQLiteFlierHistoryProvider,
) -> None:
    """Storing the same edge twice should increment its strength.

    The UPSERT semantics use ON CONFLICT to add the new strength value
    to the existing one.  Two inserts with strength=1.0 each should
    result in strength=2.0.
    """
    edge = {
        "source_entity": "Jeff Mills",
        "target_entity": "Robert Hood",
        "relationship_type": "label_mate",
        "evidence": {"label": "Axis Records"},
        "strength": 1.0,
    }
    await provider.store_relationship_edges([edge])
    await provider.store_relationship_edges([edge])

    results = await provider.get_label_mates(["Jeff Mills"])
    assert len(results) == 1
    assert results[0]["strength"] > 1.0


@pytest.mark.asyncio()
async def test_get_co_billing_edges(
    provider: SQLiteFlierHistoryProvider,
) -> None:
    """Store a co_billing edge and retrieve it by relationship type."""
    edges = [
        {
            "source_entity": "Derrick May",
            "target_entity": "Kevin Saunderson",
            "relationship_type": "co_billing",
            "evidence": {"event": "Tresor Night 1995"},
            "strength": 1.0,
        },
    ]
    await provider.store_relationship_edges(edges)

    # get_co_billing_edges should find it
    results = await provider.get_co_billing_edges(["Derrick May"])
    assert len(results) == 1
    assert results[0]["relationship_type"] == "co_billing"
    assert results[0]["target_entity"] == "Kevin Saunderson"

    # get_label_mates should NOT find it (different type)
    label_results = await provider.get_label_mates(["Derrick May"])
    assert len(label_results) == 0


@pytest.mark.asyncio()
async def test_bidirectional_lookup(
    provider: SQLiteFlierHistoryProvider,
) -> None:
    """Storing (A -> B) should be findable by querying for B.

    The query checks both source_entity and target_entity columns,
    so relationships are discoverable from either direction without
    storing duplicate rows.
    """
    edges = [
        {
            "source_entity": "Aphex Twin",
            "target_entity": "Squarepusher",
            "relationship_type": "label_mate",
            "evidence": {"label": "Warp Records"},
            "strength": 1.0,
        },
    ]
    await provider.store_relationship_edges(edges)

    # Query by target entity (Squarepusher), should still find the edge
    results = await provider.get_label_mates(["Squarepusher"])
    assert len(results) == 1
    assert results[0]["source_entity"] == "Aphex Twin"
    assert results[0]["target_entity"] == "Squarepusher"


@pytest.mark.asyncio()
async def test_empty_query_returns_empty(
    provider: SQLiteFlierHistoryProvider,
) -> None:
    """Querying for an artist with no stored edges returns an empty list."""
    # Store some edges for other artists
    edges = [
        {
            "source_entity": "Carl Craig",
            "target_entity": "Moodymann",
            "relationship_type": "label_mate",
            "evidence": {"label": "Planet E"},
            "strength": 1.0,
        },
    ]
    await provider.store_relationship_edges(edges)

    # Query for a nonexistent artist
    results = await provider.get_label_mates(["DJ Nobody"])
    assert results == []

    results = await provider.get_co_billing_edges(["DJ Nobody"])
    assert results == []

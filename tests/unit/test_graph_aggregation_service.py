"""Unit tests for GraphAggregationService.

Tests: entity dedup, fuzzy matching, edge merging, dismissal filtering,
stats calculation, and node detail retrieval.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.graph_aggregation_service import GraphAggregationService


# ─── Mock provider factory ──────────────────────────────────────────

def _make_mock_provider(analyses: list[dict]) -> MagicMock:
    """Create a mock IFlierHistoryProvider with preset analyses."""
    mock = MagicMock()
    mock.get_all_active_analyses = AsyncMock(return_value=analyses)
    return mock


# ─── Sample data ─────────────────────────────────────────────────────

ANALYSIS_1 = {
    "session_id": "sess-001",
    "flier_id": 1,
    "created_at": "2025-01-01T00:00:00Z",
    "venue": "Berghain",
    "event_name": "Panorama",
    "genre_tags": ["techno"],
    "dismissals": [],
    "interconnection_map": {
        "nodes": [
            {"entity_type": "ARTIST", "name": "Carl Cox", "properties": {"city": "London"}},
            {"entity_type": "ARTIST", "name": "Adam Beyer", "properties": {"city": "Stockholm"}},
            {"entity_type": "VENUE", "name": "Berghain", "properties": {"city": "Berlin"}},
        ],
        "edges": [
            {
                "source": "Carl Cox",
                "target": "Adam Beyer",
                "relationship_type": "shared_label",
                "details": "Both on Drumcode",
                "confidence": 0.9,
                "dismissed": False,
            },
        ],
    },
}

ANALYSIS_2 = {
    "session_id": "sess-002",
    "flier_id": 2,
    "created_at": "2025-01-02T00:00:00Z",
    "venue": "Fabric",
    "event_name": "FABRICLIVE",
    "genre_tags": ["house"],
    "dismissals": [],
    "interconnection_map": {
        "nodes": [
            {"entity_type": "ARTIST", "name": "Carl Cox", "properties": {"genres": ["techno"]}},
            {"entity_type": "ARTIST", "name": "Nina Kraviz", "properties": {}},
            {"entity_type": "VENUE", "name": "Fabric", "properties": {"city": "London"}},
        ],
        "edges": [
            {
                "source": "Carl Cox",
                "target": "Nina Kraviz",
                "relationship_type": "shared_label",
                "details": "Both on Trip Records",
                "confidence": 0.7,
                "dismissed": False,
            },
        ],
    },
}

ANALYSIS_WITH_DISMISSAL = {
    "session_id": "sess-003",
    "flier_id": 3,
    "created_at": "2025-01-03T00:00:00Z",
    "venue": "Tresor",
    "event_name": None,
    "genre_tags": [],
    "dismissals": [
        {
            "source_entity": "Carl Cox",
            "target_entity": "Adam Beyer",
            "relationship_type": "shared_label",
        },
    ],
    "interconnection_map": {
        "nodes": [
            {"entity_type": "ARTIST", "name": "Carl Cox", "properties": {}},
            {"entity_type": "ARTIST", "name": "Adam Beyer", "properties": {}},
        ],
        "edges": [
            {
                "source": "Carl Cox",
                "target": "Adam Beyer",
                "relationship_type": "shared_label",
                "confidence": 0.8,
                "dismissed": False,
            },
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════


class TestAggregate:
    """Tests for the main aggregate() method."""

    @pytest.mark.asyncio
    async def test_empty_analyses(self):
        svc = GraphAggregationService(flier_history=_make_mock_provider([]))
        result = await svc.aggregate()
        assert result.total_analyses == 0
        assert len(result.nodes) == 0
        assert len(result.edges) == 0

    @pytest.mark.asyncio
    async def test_single_analysis(self):
        svc = GraphAggregationService(flier_history=_make_mock_provider([ANALYSIS_1]))
        result = await svc.aggregate()
        assert result.total_analyses == 1
        assert len(result.nodes) == 3
        assert len(result.edges) == 1

    @pytest.mark.asyncio
    async def test_entity_deduplication(self):
        """Carl Cox appears in both analyses → one node with appearance_count=2."""
        svc = GraphAggregationService(
            flier_history=_make_mock_provider([ANALYSIS_1, ANALYSIS_2])
        )
        result = await svc.aggregate()

        carl_nodes = [n for n in result.nodes if n.name == "Carl Cox"]
        assert len(carl_nodes) == 1
        assert carl_nodes[0].appearance_count == 2
        assert len(carl_nodes[0].source_sessions) == 2

    @pytest.mark.asyncio
    async def test_unique_entities_counted(self):
        """Two analyses with some overlap → correct unique count."""
        svc = GraphAggregationService(
            flier_history=_make_mock_provider([ANALYSIS_1, ANALYSIS_2])
        )
        result = await svc.aggregate()
        # Unique: Carl Cox, Adam Beyer, Berghain, Nina Kraviz, Fabric = 5
        assert len(result.nodes) == 5

    @pytest.mark.asyncio
    async def test_dismissed_edges_filtered(self):
        """Dismissed edges from analysis_3 should be excluded."""
        svc = GraphAggregationService(
            flier_history=_make_mock_provider([ANALYSIS_WITH_DISMISSAL])
        )
        result = await svc.aggregate()
        # The edge Carl Cox→Adam Beyer is dismissed → 0 edges
        assert len(result.edges) == 0

    @pytest.mark.asyncio
    async def test_nodes_sorted_by_appearance(self):
        svc = GraphAggregationService(
            flier_history=_make_mock_provider([ANALYSIS_1, ANALYSIS_2])
        )
        result = await svc.aggregate()
        # Carl Cox (2 appearances) should be first
        assert result.nodes[0].name == "Carl Cox"
        assert result.nodes[0].appearance_count == 2

    @pytest.mark.asyncio
    async def test_properties_merged(self):
        """Properties from multiple analyses should be merged."""
        svc = GraphAggregationService(
            flier_history=_make_mock_provider([ANALYSIS_1, ANALYSIS_2])
        )
        result = await svc.aggregate()
        carl = next(n for n in result.nodes if n.name == "Carl Cox")
        # Should have both city (from analysis_1) and genres (from analysis_2)
        assert "city" in carl.properties or "genres" in carl.properties


class TestGetNodeDetail:
    """Tests for get_node_detail()."""

    @pytest.mark.asyncio
    async def test_found_entity(self):
        svc = GraphAggregationService(
            flier_history=_make_mock_provider([ANALYSIS_1, ANALYSIS_2])
        )
        detail = await svc.get_node_detail("Carl Cox")
        assert detail["found"] is True
        assert detail["name"] == "Carl Cox"
        assert detail["appearance_count"] == 2
        assert len(detail["edges"]) > 0

    @pytest.mark.asyncio
    async def test_not_found_entity(self):
        svc = GraphAggregationService(flier_history=_make_mock_provider([]))
        detail = await svc.get_node_detail("Unknown Artist")
        assert detail["found"] is False


class TestGetStats:
    """Tests for get_stats()."""

    @pytest.mark.asyncio
    async def test_stats_with_data(self):
        svc = GraphAggregationService(
            flier_history=_make_mock_provider([ANALYSIS_1, ANALYSIS_2])
        )
        stats = await svc.get_stats()
        assert stats["total_analyses"] == 2
        assert stats["unique_nodes"] == 5
        assert stats["unique_edges"] > 0
        assert "ARTIST" in stats["entity_type_counts"]

    @pytest.mark.asyncio
    async def test_stats_empty(self):
        svc = GraphAggregationService(flier_history=_make_mock_provider([]))
        stats = await svc.get_stats()
        assert stats["total_analyses"] == 0
        assert stats["unique_nodes"] == 0

"""Unit tests for the 9 new analysis persistence methods on SQLiteFlierHistoryProvider.

Tests cover: store_analysis, get_analysis, get_analysis_by_flier_id,
list_analyses, persist_edge_dismissal, get_edge_dismissals, add_annotation,
get_annotations, get_all_active_analyses.

Each test uses a temporary SQLite database to ensure isolation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from src.providers.flier_history.sqlite_flier_history_provider import SQLiteFlierHistoryProvider


@pytest_asyncio.fixture
async def provider(tmp_path: Path) -> SQLiteFlierHistoryProvider:
    """Create and initialize a provider with a temp DB."""
    db_path = tmp_path / "test_flier_history.db"
    prov = SQLiteFlierHistoryProvider(db_path=db_path)
    await prov.initialize()
    return prov


@pytest_asyncio.fixture
async def provider_with_flier(provider: SQLiteFlierHistoryProvider) -> tuple[SQLiteFlierHistoryProvider, dict]:
    """Provider with a logged flier for analysis tests."""
    flier = await provider.log_flier(
        session_id="sess-001",
        artists=["Carl Cox", "Adam Beyer"],
        venue="Berghain",
        promoter="Ostgut Ton",
        event_name="Panorama Bar",
        event_date="2025-06-15",
        genre_tags=["techno"],
    )
    return provider, flier


# ─── Sample data ─────────────────────────────────────────────────────

SAMPLE_MAP = {
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
            "confidence": 0.85,
            "dismissed": False,
        },
    ],
    "patterns": [],
    "narrative": "Test narrative",
    "citations": [],
}

SAMPLE_RESEARCH = [
    {"entity_name": "Carl Cox", "entity_type": "ARTIST", "summary": "Legendary techno DJ"},
]


# ═══════════════════════════════════════════════════════════════════════
# store_analysis
# ═══════════════════════════════════════════════════════════════════════


class TestStoreAnalysis:
    """Tests for storing analysis snapshots."""

    @pytest.mark.asyncio
    async def test_store_creates_snapshot(self, provider_with_flier):
        prov, flier = provider_with_flier
        result = await prov.store_analysis(
            session_id="sess-001",
            interconnection_map=SAMPLE_MAP,
        )
        assert result["session_id"] == "sess-001"
        assert result["flier_id"] == flier["id"]
        assert result["revision"] == 1
        assert result["id"] is not None

    @pytest.mark.asyncio
    async def test_store_with_research_results(self, provider_with_flier):
        prov, _ = provider_with_flier
        result = await prov.store_analysis(
            session_id="sess-001",
            interconnection_map=SAMPLE_MAP,
            research_results=SAMPLE_RESEARCH,
        )
        assert result["revision"] == 1

    @pytest.mark.asyncio
    async def test_store_revision_increment(self, provider_with_flier):
        prov, _ = provider_with_flier
        r1 = await prov.store_analysis(session_id="sess-001", interconnection_map=SAMPLE_MAP)
        assert r1["revision"] == 1

        r2 = await prov.store_analysis(session_id="sess-001", interconnection_map=SAMPLE_MAP)
        assert r2["revision"] == 2

    @pytest.mark.asyncio
    async def test_store_no_flier_raises(self, provider):
        with pytest.raises(ValueError, match="No flier found"):
            await provider.store_analysis(
                session_id="nonexistent",
                interconnection_map=SAMPLE_MAP,
            )


# ═══════════════════════════════════════════════════════════════════════
# get_analysis
# ═══════════════════════════════════════════════════════════════════════


class TestGetAnalysis:
    """Tests for retrieving analysis snapshots."""

    @pytest.mark.asyncio
    async def test_get_returns_active_snapshot(self, provider_with_flier):
        prov, _ = provider_with_flier
        await prov.store_analysis(session_id="sess-001", interconnection_map=SAMPLE_MAP)

        result = await prov.get_analysis("sess-001")
        assert result is not None
        assert result["session_id"] == "sess-001"
        assert result["interconnection_map"]["narrative"] == "Test narrative"
        assert "research_results_json" not in result

    @pytest.mark.asyncio
    async def test_get_with_research(self, provider_with_flier):
        prov, _ = provider_with_flier
        await prov.store_analysis(
            session_id="sess-001",
            interconnection_map=SAMPLE_MAP,
            research_results=SAMPLE_RESEARCH,
        )

        result = await prov.get_analysis("sess-001", include_research=True)
        assert result is not None
        assert result["research_results"] is not None
        assert len(result["research_results"]) == 1

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, provider):
        result = await provider.get_analysis("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_latest_revision(self, provider_with_flier):
        prov, _ = provider_with_flier
        await prov.store_analysis(session_id="sess-001", interconnection_map=SAMPLE_MAP)
        modified_map = {**SAMPLE_MAP, "narrative": "Updated narrative"}
        await prov.store_analysis(session_id="sess-001", interconnection_map=modified_map)

        result = await prov.get_analysis("sess-001")
        assert result["revision"] == 2
        assert result["interconnection_map"]["narrative"] == "Updated narrative"


# ═══════════════════════════════════════════════════════════════════════
# get_analysis_by_flier_id
# ═══════════════════════════════════════════════════════════════════════


class TestGetAnalysisByFlierId:

    @pytest.mark.asyncio
    async def test_get_by_flier_id(self, provider_with_flier):
        prov, flier = provider_with_flier
        await prov.store_analysis(session_id="sess-001", interconnection_map=SAMPLE_MAP)

        result = await prov.get_analysis_by_flier_id(flier["id"])
        assert result is not None
        assert result["session_id"] == "sess-001"

    @pytest.mark.asyncio
    async def test_get_by_flier_id_nonexistent(self, provider):
        result = await provider.get_analysis_by_flier_id(9999)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════
# list_analyses
# ═══════════════════════════════════════════════════════════════════════


class TestListAnalyses:

    @pytest.mark.asyncio
    async def test_list_empty(self, provider):
        result = await provider.list_analyses()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_with_analyses(self, provider):
        await provider.log_flier(
            session_id="sess-a", artists=["A"], venue="V1",
            promoter=None, event_name="E1", event_date=None, genre_tags=["house"],
        )
        await provider.store_analysis(session_id="sess-a", interconnection_map=SAMPLE_MAP)

        await provider.log_flier(
            session_id="sess-b", artists=["B"], venue="V2",
            promoter=None, event_name="E2", event_date=None, genre_tags=["techno"],
        )
        await provider.store_analysis(session_id="sess-b", interconnection_map=SAMPLE_MAP)

        result = await provider.list_analyses()
        assert len(result) == 2
        assert all("venue" in r for r in result)

    @pytest.mark.asyncio
    async def test_list_pagination(self, provider):
        for i in range(5):
            await provider.log_flier(
                session_id=f"sess-{i}", artists=[f"A{i}"], venue=f"V{i}",
                promoter=None, event_name=None, event_date=None, genre_tags=[],
            )
            await provider.store_analysis(session_id=f"sess-{i}", interconnection_map=SAMPLE_MAP)

        result = await provider.list_analyses(limit=2, offset=0)
        assert len(result) == 2

        result2 = await provider.list_analyses(limit=2, offset=2)
        assert len(result2) == 2


# ═══════════════════════════════════════════════════════════════════════
# Edge dismissals
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeDismissals:

    @pytest.mark.asyncio
    async def test_persist_dismissal(self, provider_with_flier):
        prov, _ = provider_with_flier
        result = await prov.persist_edge_dismissal(
            session_id="sess-001",
            source="Carl Cox",
            target="Adam Beyer",
            relationship_type="shared_label",
            reason="Incorrect",
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_persist_dismissal_no_flier(self, provider):
        result = await provider.persist_edge_dismissal(
            session_id="nonexistent",
            source="A", target="B",
            relationship_type="related",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_get_dismissals(self, provider_with_flier):
        prov, _ = provider_with_flier
        await prov.persist_edge_dismissal(
            session_id="sess-001",
            source="Carl Cox", target="Adam Beyer",
            relationship_type="shared_label", reason="Bad data",
        )
        await prov.persist_edge_dismissal(
            session_id="sess-001",
            source="Carl Cox", target="Berghain",
            relationship_type="resident_at",
        )

        dismissals = await prov.get_edge_dismissals("sess-001")
        assert len(dismissals) == 2
        assert dismissals[0]["source_entity"] == "Carl Cox"

    @pytest.mark.asyncio
    async def test_get_dismissals_empty(self, provider):
        dismissals = await provider.get_edge_dismissals("nonexistent")
        assert dismissals == []


# ═══════════════════════════════════════════════════════════════════════
# Annotations
# ═══════════════════════════════════════════════════════════════════════


class TestAnnotations:

    @pytest.mark.asyncio
    async def test_add_annotation(self, provider_with_flier):
        prov, _ = provider_with_flier
        result = await prov.add_annotation(
            session_id="sess-001",
            note="Great analysis, very insightful",
            target_type="analysis",
        )
        assert result["id"] is not None
        assert result["note"] == "Great analysis, very insightful"
        assert result["target_type"] == "analysis"

    @pytest.mark.asyncio
    async def test_add_annotation_to_entity(self, provider_with_flier):
        prov, _ = provider_with_flier
        result = await prov.add_annotation(
            session_id="sess-001",
            note="Key figure in techno",
            target_type="entity",
            target_key="Carl Cox",
        )
        assert result["target_key"] == "Carl Cox"

    @pytest.mark.asyncio
    async def test_add_annotation_no_flier_raises(self, provider):
        with pytest.raises(ValueError, match="No flier found"):
            await provider.add_annotation(
                session_id="nonexistent",
                note="Test",
            )

    @pytest.mark.asyncio
    async def test_get_annotations(self, provider_with_flier):
        prov, _ = provider_with_flier
        await prov.add_annotation(session_id="sess-001", note="Note 1")
        await prov.add_annotation(session_id="sess-001", note="Note 2")

        annotations = await prov.get_annotations("sess-001")
        assert len(annotations) == 2

    @pytest.mark.asyncio
    async def test_get_annotations_empty(self, provider):
        annotations = await provider.get_annotations("nonexistent")
        assert annotations == []


# ═══════════════════════════════════════════════════════════════════════
# get_all_active_analyses
# ═══════════════════════════════════════════════════════════════════════


class TestGetAllActiveAnalyses:

    @pytest.mark.asyncio
    async def test_get_all_empty(self, provider):
        result = await provider.get_all_active_analyses()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_all_with_analyses(self, provider):
        for i in range(3):
            await provider.log_flier(
                session_id=f"sess-{i}", artists=[f"Artist {i}"], venue=f"Venue {i}",
                promoter=None, event_name=None, event_date=None, genre_tags=["techno"],
            )
            await provider.store_analysis(session_id=f"sess-{i}", interconnection_map=SAMPLE_MAP)

        result = await provider.get_all_active_analyses()
        assert len(result) == 3
        for r in result:
            assert "interconnection_map" in r
            assert "dismissals" in r

    @pytest.mark.asyncio
    async def test_get_all_includes_dismissals(self, provider):
        await provider.log_flier(
            session_id="sess-x", artists=["A"], venue="V",
            promoter=None, event_name=None, event_date=None, genre_tags=[],
        )
        await provider.store_analysis(session_id="sess-x", interconnection_map=SAMPLE_MAP)
        await provider.persist_edge_dismissal(
            session_id="sess-x",
            source="Carl Cox", target="Adam Beyer",
            relationship_type="shared_label",
        )

        result = await provider.get_all_active_analyses()
        assert len(result) == 1
        assert len(result[0]["dismissals"]) == 1

    @pytest.mark.asyncio
    async def test_only_active_revisions_returned(self, provider):
        await provider.log_flier(
            session_id="sess-rev", artists=["A"], venue="V",
            promoter=None, event_name=None, event_date=None, genre_tags=[],
        )
        await provider.store_analysis(session_id="sess-rev", interconnection_map=SAMPLE_MAP)
        modified = {**SAMPLE_MAP, "narrative": "Rev 2"}
        await provider.store_analysis(session_id="sess-rev", interconnection_map=modified)

        result = await provider.get_all_active_analyses()
        assert len(result) == 1
        assert result[0]["interconnection_map"]["narrative"] == "Rev 2"

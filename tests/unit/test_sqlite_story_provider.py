"""Unit tests for SQLiteStoryProvider.

Tests CRUD operations, event queries, narrative caching, tag retrieval,
and statistics against a temporary in-memory-like SQLite database.

Uses a temp file to avoid polluting the real data directory.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from src.models.story import RaveStory, StoryMetadata, StoryStatus
from src.providers.story.sqlite_story_provider import SQLiteStoryProvider


@pytest.fixture
async def provider():
    """Create a SQLiteStoryProvider with a temporary database."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    p = SQLiteStoryProvider(db_path=tmp.name)
    await p.initialize()
    yield p
    os.unlink(tmp.name)


def _make_story(
    story_id: str = "test-001",
    text: str = "The bass was incredible at this underground warehouse party.",
    status: StoryStatus = StoryStatus.APPROVED,
    event_name: str = "Tresor Tuesday",
    event_year: int = 1995,
    city: str = "Berlin",
    genre: str = "techno",
    **kwargs,
) -> RaveStory:
    return RaveStory(
        story_id=story_id,
        text=text,
        word_count=len(text.split()),
        status=status,
        created_at="2025-06-15",
        moderated_at="2025-06-15",
        metadata=StoryMetadata(
            event_name=event_name,
            event_year=event_year,
            city=city,
            genre=genre,
        ),
        entity_tags=kwargs.get("entity_tags", ["Jeff Mills"]),
        genre_tags=kwargs.get("genre_tags", ["techno"]),
        geographic_tags=kwargs.get("geographic_tags", ["Berlin"]),
    )


# ─── Initialization ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_initialize_creates_tables(provider):
    """Provider initialization should create all required tables."""
    assert provider.get_provider_name() == "sqlite_story"


@pytest.mark.asyncio
async def test_double_initialize_is_idempotent(provider):
    """Calling initialize() twice should not raise."""
    await provider.initialize()


# ─── Submit + Get ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_submit_and_get_story(provider):
    story = _make_story()
    result = await provider.submit_story(story)
    assert result["story_id"] == "test-001"
    assert result["status"] == "APPROVED"

    retrieved = await provider.get_story("test-001")
    assert retrieved is not None
    assert retrieved.story_id == "test-001"
    assert retrieved.text == story.text
    assert retrieved.metadata.event_name == "Tresor Tuesday"
    assert "Jeff Mills" in retrieved.entity_tags
    assert "techno" in retrieved.genre_tags


@pytest.mark.asyncio
async def test_get_nonexistent_story(provider):
    result = await provider.get_story("nonexistent-id")
    assert result is None


# ─── List Stories ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_stories_with_status_filter(provider):
    await provider.submit_story(_make_story(story_id="s1", status=StoryStatus.APPROVED))
    await provider.submit_story(_make_story(story_id="s2", status=StoryStatus.REJECTED))
    await provider.submit_story(_make_story(story_id="s3", status=StoryStatus.APPROVED))

    approved = await provider.list_stories(status="APPROVED")
    assert len(approved) == 2

    rejected = await provider.list_stories(status="REJECTED")
    assert len(rejected) == 1


@pytest.mark.asyncio
async def test_list_stories_with_tag_filter(provider):
    await provider.submit_story(_make_story(story_id="s1", genre_tags=["techno"]))
    await provider.submit_story(_make_story(story_id="s2", genre_tags=["jungle"]))

    techno = await provider.list_stories(tag_type="genre", tag_value="techno")
    assert len(techno) == 1
    assert techno[0].story_id == "s1"


@pytest.mark.asyncio
async def test_list_stories_pagination(provider):
    for i in range(5):
        await provider.submit_story(_make_story(story_id=f"p{i}"))

    page1 = await provider.list_stories(limit=2, offset=0)
    assert len(page1) == 2

    page2 = await provider.list_stories(limit=2, offset=2)
    assert len(page2) == 2

    page3 = await provider.list_stories(limit=2, offset=4)
    assert len(page3) == 1


# ─── Update Status ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_update_story_status(provider):
    await provider.submit_story(_make_story(status=StoryStatus.PENDING_MODERATION))

    updated = await provider.update_story_status(
        "test-001", "APPROVED", moderation_flags=["clean"]
    )
    assert updated is True

    story = await provider.get_story("test-001")
    assert story.status == StoryStatus.APPROVED


@pytest.mark.asyncio
async def test_update_nonexistent_story(provider):
    updated = await provider.update_story_status("fake-id", "APPROVED")
    assert updated is False


# ─── Event Collections ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_events(provider):
    await provider.submit_story(_make_story(story_id="e1", event_name="Tresor"))
    await provider.submit_story(_make_story(story_id="e2", event_name="Tresor"))
    await provider.submit_story(_make_story(story_id="e3", event_name="Berghain"))

    events = await provider.list_events()
    assert len(events) == 2
    # Tresor has more stories, should appear first.
    assert events[0]["event_name"] == "Tresor"
    assert events[0]["story_count"] == 2


@pytest.mark.asyncio
async def test_get_event_stories(provider):
    await provider.submit_story(_make_story(story_id="es1", event_name="Tresor"))
    await provider.submit_story(_make_story(story_id="es2", event_name="Tresor"))
    await provider.submit_story(_make_story(story_id="es3", event_name="Other"))

    stories = await provider.get_event_stories("Tresor")
    assert len(stories) == 2


# ─── Narratives ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_save_and_get_narrative(provider):
    await provider.save_narrative(
        event_name="Tresor",
        event_year=1995,
        narrative="The bass vibrated through the concrete walls...",
        themes=["bass weight", "unity"],
        story_count=5,
    )

    cached = await provider.get_narrative("Tresor", 1995)
    assert cached is not None
    assert cached["narrative"].startswith("The bass")
    assert len(cached["themes"]) == 2
    assert cached["story_count"] == 5


@pytest.mark.asyncio
async def test_get_nonexistent_narrative(provider):
    result = await provider.get_narrative("Nonexistent", None)
    assert result is None


@pytest.mark.asyncio
async def test_narrative_upsert(provider):
    """Saving a narrative for the same event should update, not duplicate."""
    await provider.save_narrative("Tresor", 1995, "v1", ["theme1"], 3)
    await provider.save_narrative("Tresor", 1995, "v2", ["theme2"], 5)

    cached = await provider.get_narrative("Tresor", 1995)
    assert cached["narrative"] == "v2"
    assert cached["story_count"] == 5


# ─── Tags ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_tags(provider):
    await provider.submit_story(_make_story(story_id="t1", genre_tags=["techno", "acid"]))
    await provider.submit_story(_make_story(story_id="t2", genre_tags=["jungle"]))

    genres = await provider.get_tags("genre")
    assert set(genres) == {"acid", "jungle", "techno"}


@pytest.mark.asyncio
async def test_tags_only_from_approved_stories(provider):
    """Tags should only come from APPROVED stories."""
    await provider.submit_story(_make_story(story_id="a1", status=StoryStatus.APPROVED, genre_tags=["techno"]))
    await provider.submit_story(_make_story(story_id="r1", status=StoryStatus.REJECTED, genre_tags=["forbidden"]))

    genres = await provider.get_tags("genre")
    assert "techno" in genres
    assert "forbidden" not in genres


# ─── Statistics ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_stats(provider):
    await provider.submit_story(_make_story(story_id="st1", status=StoryStatus.APPROVED))
    await provider.submit_story(_make_story(story_id="st2", status=StoryStatus.REJECTED))
    await provider.submit_story(_make_story(story_id="st3", status=StoryStatus.APPROVED))

    stats = await provider.get_stats()
    assert stats["total_stories"] == 3
    assert stats["approved_stories"] == 2
    assert stats["stories_by_status"]["APPROVED"] == 2
    assert stats["stories_by_status"]["REJECTED"] == 1

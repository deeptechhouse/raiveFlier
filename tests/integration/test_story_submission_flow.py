"""Integration test for the full story submission flow.

Tests the end-to-end pipeline: submit → moderate → extract → store → retrieve.
Uses a real SQLite database (temp file) with a mocked LLM provider.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.story import StoryMetadata, StoryStatus
from src.providers.story.sqlite_story_provider import SQLiteStoryProvider
from src.services.story_service import StoryService


@pytest.fixture
async def service():
    """Build a StoryService with real SQLite and mocked LLM."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()

    store = SQLiteStoryProvider(db_path=tmp.name)
    await store.initialize()

    llm = MagicMock()
    llm.get_provider_name.return_value = "mock_llm"

    svc = StoryService(llm=llm, story_store=store)
    yield svc, llm, store
    os.unlink(tmp.name)


@pytest.mark.asyncio
async def test_full_text_submission_flow(service):
    """Submit a text story and verify it appears in browse and event views."""
    svc, llm, store = service

    # Configure LLM to approve and extract entities.
    llm.complete = AsyncMock(side_effect=[
        '{"is_safe": true, "flags": [], "pii_found": [], "reason": null}',
        '{"artists": ["Jeff Mills"], "venues": ["Tresor"], "genres": ["techno"], "cities": ["Berlin"], "promoters": []}',
    ])

    # 1. Submit a story.
    result = await svc.submit_text_story(
        "The thundering kick drums at Tresor were unlike anything I had experienced. "
        "Jeff Mills commanded the room with precision, each transition seamless. "
        "The concrete walls vibrated with every bassline.",
        StoryMetadata(event_name="Tresor Tuesday", event_year=1995, city="Berlin", genre="techno"),
    )

    assert result["status"] == "APPROVED"
    story_id = result["story_id"]

    # 2. Retrieve it.
    story = await svc.get_story(story_id)
    assert story is not None
    assert story.status == StoryStatus.APPROVED
    assert "thundering kick drums" in story.text

    # 3. Verify it appears in the browse list.
    stories = await svc.list_stories()
    assert len(stories) == 1
    assert stories[0].story_id == story_id

    # 4. Verify the event appears in the events list.
    events = await svc.list_events()
    assert len(events) == 1
    assert events[0]["event_name"] == "Tresor Tuesday"

    # 5. Verify event stories.
    collection = await svc.get_event_stories("Tresor Tuesday", 1995)
    assert collection.story_count == 1

    # 6. Verify stats.
    stats = await svc.get_stats()
    assert stats["total_stories"] == 1
    assert stats["approved_stories"] == 1


@pytest.mark.asyncio
async def test_rejected_story_not_in_browse(service):
    """Rejected stories should not appear in browse or event views."""
    svc, llm, store = service

    llm.complete = AsyncMock(return_value='{"is_safe": false, "flags": ["hate_speech"], "pii_found": [], "reason": "Contains hate speech."}')

    result = await svc.submit_text_story(
        "A story that contains problematic content and should be rejected by moderation. " * 3,
        StoryMetadata(event_name="Bad Event"),
    )

    assert result["status"] == "REJECTED"

    # Should not appear in approved list.
    stories = await svc.list_stories()
    assert len(stories) == 0


@pytest.mark.asyncio
async def test_pii_redaction_flow(service):
    """Stories with PII should have it redacted but still be approved."""
    svc, llm, store = service

    llm.complete = AsyncMock(side_effect=[
        '{"is_safe": true, "flags": ["pii_detected"], "pii_found": ["John Smith"], "reason": null}',
        '{"artists": [], "venues": [], "genres": ["techno"], "cities": [], "promoters": []}',
    ])

    result = await svc.submit_text_story(
        "I saw John Smith spinning records at the underground warehouse party last weekend. " * 3,
        StoryMetadata(event_name="Underground Party"),
    )

    assert result["status"] == "APPROVED"
    assert "pii_redacted" in result["moderation_flags"]

    story = await svc.get_story(result["story_id"])
    assert "[REDACTED]" in story.text
    assert "John Smith" not in story.text


@pytest.mark.asyncio
async def test_narrative_generation_flow(service):
    """Narrative generation requires >= 3 stories and uses LLM synthesis."""
    svc, llm, store = service

    # Submit 3 stories (each call needs 2 LLM responses: moderation + extraction).
    for i in range(3):
        llm.complete = AsyncMock(side_effect=[
            '{"is_safe": true, "flags": [], "pii_found": [], "reason": null}',
            '{"artists": [], "venues": [], "genres": ["techno"], "cities": ["Berlin"], "promoters": []}',
        ])
        await svc.submit_text_story(
            f"Story {i}: The bass was incredible and the energy was electric at the warehouse. " * 3,
            StoryMetadata(event_name="Warehouse Rave", event_year=1997, city="Berlin"),
        )

    # Generate narrative.
    llm.complete = AsyncMock(return_value='{"narrative": "The warehouse throbbed with communal energy...", "themes": ["bass", "unity", "sunrise"]}')

    result = await svc.get_or_generate_narrative("Warehouse Rave", 1997)
    assert result.get("narrative") is not None
    assert "warehouse" in result["narrative"].lower()
    assert len(result["themes"]) == 3

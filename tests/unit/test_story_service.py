"""Unit tests for StoryService with mocked dependencies.

Tests the orchestration logic: text sanitization, moderation flow,
entity extraction, narrative generation, and the submission pipeline.
Dependencies (LLM, story store, vector store) are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.story import RaveStory, StoryMetadata, StoryStatus
from src.services.story_service import StoryService


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.complete = AsyncMock(return_value='{"is_safe": true, "flags": [], "pii_found": [], "reason": null}')
    llm.get_provider_name.return_value = "mock_llm"
    return llm


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.submit_story = AsyncMock(return_value={
        "story_id": "mock-uuid",
        "status": "APPROVED",
        "created_at": "2025-06-15",
    })
    store.get_story = AsyncMock(return_value=None)
    store.list_stories = AsyncMock(return_value=[])
    store.list_events = AsyncMock(return_value=[])
    store.get_event_stories = AsyncMock(return_value=[])
    store.get_narrative = AsyncMock(return_value=None)
    store.save_narrative = AsyncMock()
    store.get_tags = AsyncMock(return_value=[])
    store.get_stats = AsyncMock(return_value={"total_stories": 0, "approved_stories": 0})
    return store


@pytest.fixture
def service(mock_llm, mock_store):
    return StoryService(
        llm=mock_llm,
        story_store=mock_store,
    )


# ─── Text Sanitization ────────────────────────────────────────────

class TestTextSanitization:
    def test_strips_html_tags(self, service):
        result = service._sanitize_text("<script>alert('xss')</script>Hello <b>world</b>")
        assert "<script>" not in result
        assert "<b>" not in result
        assert "Hello" in result
        assert "world" in result

    def test_strips_whitespace(self, service):
        result = service._sanitize_text("  text with whitespace  ")
        assert result == "text with whitespace"


# ─── Submit Text Story ────────────────────────────────────────────

class TestSubmitTextStory:
    @pytest.mark.asyncio
    async def test_rejects_too_short(self, service):
        result = await service.submit_text_story(
            "Too short.",
            StoryMetadata(event_name="Test"),
        )
        assert result["status"] == "REJECTED"
        assert "at least" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_rejects_no_metadata(self, service):
        long_text = "This is a story about a rave that happened. " * 5
        result = await service.submit_text_story(long_text, StoryMetadata())
        assert result["status"] == "REJECTED"
        assert "metadata" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_successful_submission(self, service, mock_llm, mock_store):
        # LLM moderation returns safe.
        mock_llm.complete = AsyncMock(side_effect=[
            '{"is_safe": true, "flags": [], "pii_found": [], "reason": null}',
            '{"artists": ["DJ Shadow"], "venues": [], "genres": ["trip hop"], "cities": ["San Francisco"], "promoters": []}',
        ])

        long_text = "The sound system was massive and the DJ played all night long with incredible sets. " * 3
        result = await service.submit_text_story(
            long_text,
            StoryMetadata(event_name="Warehouse Party", city="San Francisco"),
        )

        assert result["story_id"] == "mock-uuid"
        assert result["status"] == "APPROVED"
        mock_store.submit_story.assert_called_once()

    @pytest.mark.asyncio
    async def test_rejects_unsafe_content(self, service, mock_llm, mock_store):
        mock_llm.complete = AsyncMock(return_value='{"is_safe": false, "flags": ["hate_speech"], "pii_found": [], "reason": "Contains hate speech."}')

        # Make submit_story return the status that was passed to it.
        async def capture_submit(story):
            return {"story_id": story.story_id, "status": story.status.value, "created_at": story.created_at}
        mock_store.submit_story = AsyncMock(side_effect=capture_submit)

        long_text = "A story with some problematic content that should be flagged. " * 3
        result = await service.submit_text_story(
            long_text,
            StoryMetadata(event_name="Test Event"),
        )

        assert result["status"] == "REJECTED"
        mock_store.submit_story.assert_called_once()  # Still stored with REJECTED status.


# ─── Moderation ────────────────────────────────────────────────────

class TestModeration:
    @pytest.mark.asyncio
    async def test_pii_redaction(self, service, mock_llm):
        mock_llm.complete = AsyncMock(return_value='{"is_safe": true, "flags": ["pii_detected"], "pii_found": ["John Smith", "555-1234"], "reason": null}')

        result = await service._moderate_content("I saw John Smith at the event and called 555-1234.")
        assert result.sanitized_text is not None
        assert "[REDACTED]" in result.sanitized_text
        assert "pii_redacted" in result.flags

    @pytest.mark.asyncio
    async def test_moderation_llm_failure_fails_open(self, service, mock_llm):
        """If LLM moderation fails, the story should still be approved."""
        mock_llm.complete = AsyncMock(side_effect=Exception("LLM unavailable"))

        result = await service._moderate_content("A perfectly safe story about a rave.")
        assert result.is_safe
        assert "moderation_skipped" in result.flags


# ─── Entity Extraction ────────────────────────────────────────────

class TestEntityExtraction:
    @pytest.mark.asyncio
    async def test_extracts_entities_from_metadata(self, service, mock_llm):
        mock_llm.complete = AsyncMock(return_value='{"artists": [], "venues": [], "genres": [], "cities": [], "promoters": []}')

        result = await service._extract_entities(
            "Some story text.",
            StoryMetadata(artist="Jeff Mills", genre="techno", city="Detroit"),
        )

        assert "Jeff Mills" in result["entities"]
        assert "techno" in result["genres"]
        assert "Detroit" in result["geographic"]

    @pytest.mark.asyncio
    async def test_merges_llm_extraction_with_metadata(self, service, mock_llm):
        mock_llm.complete = AsyncMock(return_value='{"artists": ["Aphex Twin"], "venues": ["Tresor"], "genres": ["ambient"], "cities": ["London"], "promoters": []}')

        result = await service._extract_entities(
            "Some story text.",
            StoryMetadata(artist="Jeff Mills", genre="techno"),
        )

        assert "Jeff Mills" in result["entities"]
        assert "Aphex Twin" in result["entities"]
        assert "techno" in result["genres"]
        assert "ambient" in result["genres"]


# ─── Narrative Generation ─────────────────────────────────────────

class TestNarrativeGeneration:
    @pytest.mark.asyncio
    async def test_requires_minimum_stories(self, service, mock_store):
        mock_store.get_event_stories = AsyncMock(return_value=[
            MagicMock(), MagicMock(),  # Only 2 stories — below threshold.
        ])

        result = await service.get_or_generate_narrative("Tresor")
        assert "error" in result
        assert "3" in result["error"]

    @pytest.mark.asyncio
    async def test_generates_narrative(self, service, mock_llm, mock_store):
        stories = [
            RaveStory(
                story_id=f"s{i}", text=f"Story {i} about the bass.", word_count=5,
                created_at="2025-01-01", metadata=StoryMetadata(event_name="Tresor"),
            )
            for i in range(3)
        ]
        mock_store.get_event_stories = AsyncMock(return_value=stories)
        mock_store.get_narrative = AsyncMock(return_value=None)
        mock_llm.complete = AsyncMock(return_value='{"narrative": "The crowd felt the bass...", "themes": ["bass", "unity"]}')

        result = await service.get_or_generate_narrative("Tresor")
        assert result["narrative"] == "The crowd felt the bass..."
        assert "bass" in result["themes"]
        mock_store.save_narrative.assert_called_once()


# ─── JSON Parsing ──────────────────────────────────────────────────

class TestJsonParsing:
    def test_parses_clean_json(self):
        result = StoryService._parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parses_json_in_code_fence(self):
        result = StoryService._parse_json_response('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_parses_json_with_surrounding_text(self):
        result = StoryService._parse_json_response('Here is the result: {"key": "value"} end.')
        assert result == {"key": "value"}

    def test_returns_none_for_invalid_json(self):
        result = StoryService._parse_json_response("not json at all")
        assert result is None

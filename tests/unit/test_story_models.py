"""Unit tests for Rave Stories domain models.

Tests the Pydantic v2 frozen models: StoryStatus, StoryMetadata,
ModerationResult, RaveStory, and EventStoryCollection.

Verifies:
  - Frozen immutability (model_copy for transitions).
  - StoryMetadata.has_any_field() validation.
  - Enum serialization.
  - Default values and field constraints.
"""

from __future__ import annotations

import pytest

from src.models.story import (
    EventStoryCollection,
    ModerationResult,
    RaveStory,
    StoryMetadata,
    StoryStatus,
)


# ─── StoryStatus Enum ─────────────────────────────────────────────

class TestStoryStatus:
    def test_values(self):
        assert StoryStatus.PENDING_MODERATION == "PENDING_MODERATION"
        assert StoryStatus.APPROVED == "APPROVED"
        assert StoryStatus.REJECTED == "REJECTED"

    def test_json_serialization(self):
        """StoryStatus inherits (str, Enum) so it serializes as a string."""
        assert StoryStatus.APPROVED.value == "APPROVED"


# ─── StoryMetadata ─────────────────────────────────────────────────

class TestStoryMetadata:
    def test_empty_metadata_has_no_fields(self):
        meta = StoryMetadata()
        assert not meta.has_any_field()

    def test_event_name_triggers_has_any_field(self):
        meta = StoryMetadata(event_name="Berghain NYE")
        assert meta.has_any_field()

    def test_year_triggers_has_any_field(self):
        meta = StoryMetadata(event_year=1997)
        assert meta.has_any_field()

    def test_city_triggers_has_any_field(self):
        meta = StoryMetadata(city="Berlin")
        assert meta.has_any_field()

    def test_genre_triggers_has_any_field(self):
        meta = StoryMetadata(genre="techno")
        assert meta.has_any_field()

    def test_frozen_immutability(self):
        meta = StoryMetadata(event_name="Test")
        with pytest.raises(Exception):
            meta.event_name = "Changed"

    def test_year_validation(self):
        """Year must be between 1980 and 2030."""
        meta = StoryMetadata(event_year=1995)
        assert meta.event_year == 1995

        with pytest.raises(Exception):
            StoryMetadata(event_year=1900)

        with pytest.raises(Exception):
            StoryMetadata(event_year=2100)


# ─── ModerationResult ─────────────────────────────────────────────

class TestModerationResult:
    def test_safe_result(self):
        result = ModerationResult(is_safe=True)
        assert result.is_safe
        assert result.flags == []
        assert result.sanitized_text is None
        assert result.reason is None

    def test_unsafe_result_with_flags(self):
        result = ModerationResult(
            is_safe=False,
            flags=["hate_speech", "threats"],
            reason="Contains hate speech targeting protected group.",
        )
        assert not result.is_safe
        assert len(result.flags) == 2
        assert result.reason is not None

    def test_pii_redacted(self):
        result = ModerationResult(
            is_safe=True,
            flags=["pii_redacted"],
            sanitized_text="I saw [REDACTED] at the event.",
        )
        assert result.sanitized_text is not None
        assert "[REDACTED]" in result.sanitized_text


# ─── RaveStory ─────────────────────────────────────────────────────

class TestRaveStory:
    def _make_story(self, **overrides):
        defaults = {
            "story_id": "test-uuid-123",
            "text": "The bass hit like a wall of sound at midnight.",
            "word_count": 10,
            "created_at": "2025-06-15",
            "metadata": StoryMetadata(event_name="Tresor"),
        }
        defaults.update(overrides)
        return RaveStory(**defaults)

    def test_default_status(self):
        story = self._make_story()
        assert story.status == StoryStatus.PENDING_MODERATION

    def test_default_input_mode(self):
        story = self._make_story()
        assert story.input_mode == "text"

    def test_frozen_immutability(self):
        story = self._make_story()
        with pytest.raises(Exception):
            story.text = "Modified text"

    def test_status_transition_via_model_copy(self):
        """State transitions use model_copy(update=...) — never mutate."""
        story = self._make_story()
        approved = story.model_copy(update={"status": StoryStatus.APPROVED})
        assert approved.status == StoryStatus.APPROVED
        assert story.status == StoryStatus.PENDING_MODERATION  # Original unchanged.

    def test_date_only_no_timestamp(self):
        """created_at stores only the date (YYYY-MM-DD), never a timestamp."""
        story = self._make_story(created_at="2025-06-15")
        assert len(story.created_at) == 10  # YYYY-MM-DD
        assert "T" not in story.created_at  # No timestamp separator.

    def test_entity_tags(self):
        story = self._make_story(
            entity_tags=["Jeff Mills", "Tresor"],
            genre_tags=["techno", "acid"],
            geographic_tags=["Berlin"],
        )
        assert "Jeff Mills" in story.entity_tags
        assert "techno" in story.genre_tags
        assert "Berlin" in story.geographic_tags

    def test_audio_story(self):
        story = self._make_story(input_mode="audio", audio_duration=123.5)
        assert story.input_mode == "audio"
        assert story.audio_duration == 123.5

    def test_no_user_identifiers(self):
        """Verify the model has no user-identifying fields."""
        story = self._make_story()
        field_names = set(RaveStory.model_fields.keys())
        # These fields must NEVER exist on RaveStory.
        forbidden = {"user_id", "session_id", "ip_address", "user_agent"}
        assert forbidden.isdisjoint(field_names)


# ─── EventStoryCollection ─────────────────────────────────────────

class TestEventStoryCollection:
    def test_empty_collection(self):
        collection = EventStoryCollection(event_name="Tresor Tuesday")
        assert collection.story_count == 0
        assert collection.stories == []
        assert collection.narrative is None

    def test_collection_with_narrative(self):
        collection = EventStoryCollection(
            event_name="Tresor Tuesday",
            event_year=1995,
            story_count=5,
            narrative="The bass vibrated through the concrete...",
            themes=["bass weight", "communal energy"],
        )
        assert collection.narrative is not None
        assert len(collection.themes) == 2

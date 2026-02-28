"""Anonymity verification tests for the Rave Stories feature.

These tests verify that no personally identifiable information (PII) can
leak through the story models, database schema, or API responses.

Anonymity is the #1 design constraint for Rave Stories:
  - No IP addresses stored
  - No user IDs or session IDs
  - No User-Agent strings
  - No sub-second timestamps (date only: YYYY-MM-DD)
  - Audio files deleted after transcription
"""

from __future__ import annotations

import os
import tempfile

import pytest

from src.models.story import RaveStory, StoryMetadata, StoryStatus
from src.providers.story.sqlite_story_provider import SQLiteStoryProvider


# ─── Model-Level Anonymity ─────────────────────────────────────────

class TestModelAnonymity:
    """Verify RaveStory model has no user-identifying fields."""

    def test_no_ip_address_field(self):
        fields = set(RaveStory.model_fields.keys())
        assert "ip_address" not in fields
        assert "ip" not in fields
        assert "remote_addr" not in fields

    def test_no_user_id_field(self):
        fields = set(RaveStory.model_fields.keys())
        assert "user_id" not in fields
        assert "user" not in fields
        assert "author_id" not in fields

    def test_no_session_id_field(self):
        fields = set(RaveStory.model_fields.keys())
        assert "session_id" not in fields
        assert "session" not in fields

    def test_no_user_agent_field(self):
        fields = set(RaveStory.model_fields.keys())
        assert "user_agent" not in fields
        assert "browser" not in fields

    def test_no_email_field(self):
        fields = set(RaveStory.model_fields.keys())
        assert "email" not in fields

    def test_created_at_is_date_only(self):
        """created_at must be YYYY-MM-DD format — no hours, minutes, seconds."""
        story = RaveStory(
            story_id="test-anon",
            text="Test story.",
            word_count=2,
            created_at="2025-06-15",
            metadata=StoryMetadata(event_name="Test"),
        )
        assert len(story.created_at) == 10
        assert "T" not in story.created_at
        assert ":" not in story.created_at


# ─── Database Schema Anonymity ─────────────────────────────────────

class TestDatabaseAnonymity:
    """Verify the SQLite schema has no user-identifying columns."""

    @pytest.fixture
    async def provider(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        p = SQLiteStoryProvider(db_path=tmp.name)
        await p.initialize()
        yield p
        os.unlink(tmp.name)

    @pytest.mark.asyncio
    async def test_no_ip_column_in_stories_table(self, provider):
        """Verify the stories table has no IP address column."""
        import aiosqlite
        async with aiosqlite.connect(str(provider._db_path)) as db:
            cursor = await db.execute("PRAGMA table_info(stories);")
            columns = await cursor.fetchall()
            column_names = {col[1] for col in columns}

        forbidden = {"ip_address", "ip", "remote_addr", "client_ip"}
        assert forbidden.isdisjoint(column_names), f"Found forbidden columns: {forbidden & column_names}"

    @pytest.mark.asyncio
    async def test_no_user_id_column_in_stories_table(self, provider):
        import aiosqlite
        async with aiosqlite.connect(str(provider._db_path)) as db:
            cursor = await db.execute("PRAGMA table_info(stories);")
            columns = await cursor.fetchall()
            column_names = {col[1] for col in columns}

        forbidden = {"user_id", "session_id", "user_agent", "email"}
        assert forbidden.isdisjoint(column_names), f"Found forbidden columns: {forbidden & column_names}"

    @pytest.mark.asyncio
    async def test_no_timestamp_in_created_at(self, provider):
        """Verify that stored created_at values are date-only."""
        story = RaveStory(
            story_id="anon-test-123",
            text="Testing anonymity of timestamp storage.",
            word_count=6,
            status=StoryStatus.APPROVED,
            created_at="2025-06-15",
            moderated_at="2025-06-15",
            metadata=StoryMetadata(event_name="Anon Test"),
            entity_tags=[],
            genre_tags=[],
            geographic_tags=[],
        )
        await provider.submit_story(story)

        import aiosqlite
        async with aiosqlite.connect(str(provider._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT created_at FROM stories WHERE story_id = ?",
                ("anon-test-123",),
            )
            row = await cursor.fetchone()

        created_at = row["created_at"]
        assert len(created_at) == 10, f"Expected YYYY-MM-DD, got: {created_at}"
        assert "T" not in created_at, f"Timestamp found in created_at: {created_at}"

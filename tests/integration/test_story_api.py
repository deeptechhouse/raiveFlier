"""Integration tests for the Rave Stories API endpoints.

Tests the full request/response cycle through FastAPI's TestClient,
using the stories sub-app with a temporary SQLite database and mocked
LLM provider.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.models.story import RaveStory, StoryMetadata, StoryStatus
from src.providers.story.sqlite_story_provider import SQLiteStoryProvider
from src.services.story_service import StoryService
from src.stories.main import create_stories_app


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    # Default: moderation says safe, entity extraction returns empty.
    llm.complete = AsyncMock(side_effect=[
        '{"is_safe": true, "flags": [], "pii_found": [], "reason": null}',
        '{"artists": [], "venues": [], "genres": [], "cities": [], "promoters": []}',
    ])
    llm.get_provider_name.return_value = "mock_llm"
    return llm


@pytest.fixture
def tmp_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def client(mock_llm, tmp_db):
    """Create a TestClient for the stories sub-app with mocked dependencies."""
    store = SQLiteStoryProvider(db_path=tmp_db)
    service = StoryService(llm=mock_llm, story_store=store)

    app = create_stories_app(components={
        "story_store": store,
        "story_service": service,
    })

    with TestClient(app) as c:
        yield c


# ─── Submit Text Story ────────────────────────────────────────────

class TestSubmitStory:
    def test_submit_valid_story(self, client, mock_llm):
        # Reset side_effect for each test.
        mock_llm.complete = AsyncMock(side_effect=[
            '{"is_safe": true, "flags": [], "pii_found": [], "reason": null}',
            '{"artists": [], "venues": [], "genres": ["techno"], "cities": ["Berlin"], "promoters": []}',
        ])

        response = client.post("/api/v1/stories/submit", json={
            "text": "The bass was incredible and the lights were mesmerizing in the underground bunker. " * 3,
            "metadata": {
                "event_name": "Tresor Tuesday",
                "city": "Berlin",
                "genre": "techno",
            },
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "APPROVED"
        assert data["story_id"] is not None

    def test_submit_too_short_story(self, client):
        response = client.post("/api/v1/stories/submit", json={
            "text": "Too short",
            "metadata": {"event_name": "Test"},
        })

        # Pydantic validation will reject text < 20 chars.
        assert response.status_code == 422

    def test_submit_without_metadata(self, client):
        response = client.post("/api/v1/stories/submit", json={
            "text": "A long enough story about an incredible rave experience in the warehouse. " * 3,
            "metadata": {},
        })

        assert response.status_code == 200
        data = response.json()
        # Should be rejected because no metadata fields are filled.
        assert data["status"] == "REJECTED"


# ─── List Stories ──────────────────────────────────────────────────

class TestListStories:
    def test_list_empty(self, client):
        response = client.get("/api/v1/stories/")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_with_pagination(self, client):
        response = client.get("/api/v1/stories/?limit=10&offset=0")
        assert response.status_code == 200


# ─── Events ───────────────────────────────────────────────────────

class TestEvents:
    def test_list_events_empty(self, client):
        response = client.get("/api/v1/stories/events")
        assert response.status_code == 200
        assert response.json() == []


# ─── Tags ──────────────────────────────────────────────────────────

class TestTags:
    def test_get_genre_tags(self, client):
        response = client.get("/api/v1/stories/tags/genre")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_invalid_tag_type(self, client):
        response = client.get("/api/v1/stories/tags/invalid")
        assert response.status_code == 400


# ─── Stats ─────────────────────────────────────────────────────────

class TestStats:
    def test_get_stats(self, client):
        response = client.get("/api/v1/stories/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_stories" in data
        assert "approved_stories" in data


# ─── Search ────────────────────────────────────────────────────────

class TestSearch:
    def test_search_without_vector_store(self, client):
        """Search should return empty list when no vector store is configured."""
        response = client.post("/api/v1/stories/search", json={
            "query": "bass at Tresor",
        })
        assert response.status_code == 200
        assert response.json() == []

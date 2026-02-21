"""Tests for the thumbs up/down rating system.

Covers:
- Pydantic schema validation (SubmitRatingRequest, RatingResponse, etc.)
- API endpoint integration (submit, get, summary)
- SQLiteFeedbackProvider unit tests (upsert, get_ratings, get_summary)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import router as api_router
from src.api.schemas import (
    RatingResponse,
    RatingSummaryResponse,
    SessionRatingsResponse,
    SubmitRatingRequest,
)
from src.providers.feedback.sqlite_feedback_provider import SQLiteFeedbackProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_app(*, feedback_provider: object | None = None) -> FastAPI:
    """Build a minimal FastAPI app with mocked state for testing."""
    app = FastAPI()
    app.include_router(api_router)

    app.state.session_states = {}
    app.state.pipeline = MagicMock()
    app.state.confirmation_gate = MagicMock()
    app.state.progress_tracker = MagicMock()
    app.state.rag_enabled = False
    app.state.vector_store = None
    app.state.feedback_provider = feedback_provider

    return app


def _mock_feedback_provider(
    submit_return: dict | None = None,
    ratings_return: list | None = None,
    summary_return: dict | None = None,
) -> AsyncMock:
    """Create a mock feedback provider with configurable return values."""
    mock = AsyncMock()

    if submit_return is None:
        submit_return = {
            "id": 1,
            "session_id": "test-session",
            "item_type": "ARTIST",
            "item_key": "Carl Cox",
            "rating": 1,
            "created_at": "2026-02-20T12:00:00.000Z",
            "updated_at": "2026-02-20T12:00:00.000Z",
        }
    mock.submit_rating = AsyncMock(return_value=submit_return)

    if ratings_return is None:
        ratings_return = [submit_return]
    mock.get_ratings = AsyncMock(return_value=ratings_return)

    if summary_return is None:
        summary_return = {
            "total_ratings": 1,
            "positive": 1,
            "negative": 0,
            "by_type": {"ARTIST": {"total": 1, "positive": 1, "negative": 0}},
        }
    mock.get_rating_summary = AsyncMock(return_value=summary_return)

    return mock


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestSubmitRatingRequestSchema:
    """Test SubmitRatingRequest Pydantic validation."""

    def test_valid_thumbs_up(self) -> None:
        req = SubmitRatingRequest(item_type="ARTIST", item_key="Carl Cox", rating=1)
        assert req.rating == 1
        assert req.item_type == "ARTIST"

    def test_valid_thumbs_down(self) -> None:
        req = SubmitRatingRequest(item_type="VENUE", item_key="Tresor", rating=-1)
        assert req.rating == -1

    def test_empty_item_key_rejected(self) -> None:
        with pytest.raises(Exception):
            SubmitRatingRequest(item_type="ARTIST", item_key="", rating=1)

    def test_item_key_too_long_rejected(self) -> None:
        with pytest.raises(Exception):
            SubmitRatingRequest(item_type="ARTIST", item_key="x" * 501, rating=1)


class TestRatingResponseSchema:
    """Test RatingResponse Pydantic model."""

    def test_valid_response(self) -> None:
        resp = RatingResponse(
            id=1,
            session_id="test-session",
            item_type="ARTIST",
            item_key="Carl Cox",
            rating=1,
            created_at="2026-02-20T12:00:00.000Z",
            updated_at="2026-02-20T12:00:00.000Z",
        )
        assert resp.id == 1
        assert resp.rating == 1

    def test_session_ratings_response(self) -> None:
        resp = SessionRatingsResponse(session_id="test", ratings=[], total=0)
        assert resp.total == 0
        assert resp.ratings == []

    def test_summary_response_defaults(self) -> None:
        resp = RatingSummaryResponse()
        assert resp.total_ratings == 0
        assert resp.positive == 0
        assert resp.negative == 0
        assert resp.by_type == {}


# ---------------------------------------------------------------------------
# API endpoint integration tests
# ---------------------------------------------------------------------------


class TestSubmitRatingEndpoint:
    """Test POST /api/v1/fliers/{session_id}/rate."""

    def test_submit_thumbs_up(self) -> None:
        mock = _mock_feedback_provider()
        app = _create_test_app(feedback_provider=mock)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/fliers/test-session/rate",
            json={"item_type": "ARTIST", "item_key": "Carl Cox", "rating": 1},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["item_type"] == "ARTIST"
        assert data["item_key"] == "Carl Cox"
        assert data["rating"] == 1
        mock.submit_rating.assert_awaited_once()

    def test_submit_thumbs_down(self) -> None:
        submit_return = {
            "id": 2,
            "session_id": "test-session",
            "item_type": "VENUE",
            "item_key": "Tresor",
            "rating": -1,
            "created_at": "2026-02-20T12:00:00.000Z",
            "updated_at": "2026-02-20T12:00:00.000Z",
        }
        mock = _mock_feedback_provider(submit_return=submit_return)
        app = _create_test_app(feedback_provider=mock)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/fliers/test-session/rate",
            json={"item_type": "VENUE", "item_key": "Tresor", "rating": -1},
        )

        assert resp.status_code == 200
        assert resp.json()["rating"] == -1

    def test_invalid_rating_value(self) -> None:
        mock = _mock_feedback_provider()
        app = _create_test_app(feedback_provider=mock)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/fliers/test-session/rate",
            json={"item_type": "ARTIST", "item_key": "Carl Cox", "rating": 5},
        )

        assert resp.status_code == 400
        assert "Rating must be +1 or -1" in resp.json()["detail"]

    def test_invalid_item_type(self) -> None:
        mock = _mock_feedback_provider()
        app = _create_test_app(feedback_provider=mock)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/fliers/test-session/rate",
            json={"item_type": "INVALID", "item_key": "test", "rating": 1},
        )

        assert resp.status_code == 400
        assert "Invalid item_type" in resp.json()["detail"]

    def test_no_feedback_provider_returns_503(self) -> None:
        app = _create_test_app(feedback_provider=None)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/fliers/test-session/rate",
            json={"item_type": "ARTIST", "item_key": "Carl Cox", "rating": 1},
        )

        assert resp.status_code == 503

    def test_all_valid_item_types(self) -> None:
        """Verify every item type is accepted."""
        valid_types = [
            "ARTIST", "VENUE", "PROMOTER", "DATE", "EVENT",
            "CONNECTION", "PATTERN", "QA", "CORPUS",
        ]
        for item_type in valid_types:
            mock = _mock_feedback_provider(
                submit_return={
                    "id": 1,
                    "session_id": "s",
                    "item_type": item_type,
                    "item_key": "key",
                    "rating": 1,
                    "created_at": "2026-02-20T12:00:00.000Z",
                    "updated_at": "2026-02-20T12:00:00.000Z",
                }
            )
            app = _create_test_app(feedback_provider=mock)
            client = TestClient(app)

            resp = client.post(
                "/api/v1/fliers/s/rate",
                json={"item_type": item_type, "item_key": "key", "rating": 1},
            )
            assert resp.status_code == 200, f"Failed for item_type={item_type}"


class TestGetSessionRatingsEndpoint:
    """Test GET /api/v1/fliers/{session_id}/ratings."""

    def test_returns_ratings(self) -> None:
        mock = _mock_feedback_provider()
        app = _create_test_app(feedback_provider=mock)
        client = TestClient(app)

        resp = client.get("/api/v1/fliers/test-session/ratings")

        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "test-session"
        assert data["total"] == 1
        assert len(data["ratings"]) == 1

    def test_empty_session(self) -> None:
        mock = _mock_feedback_provider(ratings_return=[])
        app = _create_test_app(feedback_provider=mock)
        client = TestClient(app)

        resp = client.get("/api/v1/fliers/empty-session/ratings")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["ratings"] == []

    def test_no_feedback_provider_returns_503(self) -> None:
        app = _create_test_app(feedback_provider=None)
        client = TestClient(app)

        resp = client.get("/api/v1/fliers/test-session/ratings")

        assert resp.status_code == 503


class TestGetRatingSummaryEndpoint:
    """Test GET /api/v1/ratings/summary."""

    def test_returns_summary(self) -> None:
        mock = _mock_feedback_provider()
        app = _create_test_app(feedback_provider=mock)
        client = TestClient(app)

        resp = client.get("/api/v1/ratings/summary")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_ratings"] == 1
        assert data["positive"] == 1
        assert data["negative"] == 0
        assert "ARTIST" in data["by_type"]

    def test_summary_with_type_filter(self) -> None:
        mock = _mock_feedback_provider()
        app = _create_test_app(feedback_provider=mock)
        client = TestClient(app)

        resp = client.get("/api/v1/ratings/summary?item_type=ARTIST")

        assert resp.status_code == 200
        mock.get_rating_summary.assert_awaited_once_with(item_type="ARTIST")

    def test_no_feedback_provider_returns_503(self) -> None:
        app = _create_test_app(feedback_provider=None)
        client = TestClient(app)

        resp = client.get("/api/v1/ratings/summary")

        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# SQLiteFeedbackProvider unit tests
# ---------------------------------------------------------------------------


class TestSQLiteFeedbackProvider:
    """Test SQLiteFeedbackProvider with a temporary database."""

    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "test_feedback.db"

    @pytest.fixture
    async def provider(self, db_path: Path) -> SQLiteFeedbackProvider:
        p = SQLiteFeedbackProvider(db_path=db_path)
        await p.initialize()
        return p

    @pytest.mark.asyncio
    async def test_initialize_creates_db(self, db_path: Path) -> None:
        provider = SQLiteFeedbackProvider(db_path=db_path)
        await provider.initialize()
        assert db_path.exists()

    @pytest.mark.asyncio
    async def test_submit_rating_returns_row(self, provider: SQLiteFeedbackProvider) -> None:
        result = await provider.submit_rating(
            session_id="sess-1",
            item_type="ARTIST",
            item_key="Carl Cox",
            rating=1,
        )
        assert result["session_id"] == "sess-1"
        assert result["item_type"] == "ARTIST"
        assert result["item_key"] == "Carl Cox"
        assert result["rating"] == 1
        assert "id" in result
        assert "created_at" in result

    @pytest.mark.asyncio
    async def test_upsert_replaces_rating(self, provider: SQLiteFeedbackProvider) -> None:
        await provider.submit_rating("sess-1", "ARTIST", "Carl Cox", 1)
        result = await provider.submit_rating("sess-1", "ARTIST", "Carl Cox", -1)
        assert result["rating"] == -1

        ratings = await provider.get_ratings("sess-1")
        assert len(ratings) == 1
        assert ratings[0]["rating"] == -1

    @pytest.mark.asyncio
    async def test_invalid_rating_raises(self, provider: SQLiteFeedbackProvider) -> None:
        with pytest.raises(ValueError, match="must be .1 or -1"):
            await provider.submit_rating("sess-1", "ARTIST", "Carl Cox", 5)

    @pytest.mark.asyncio
    async def test_get_ratings_empty(self, provider: SQLiteFeedbackProvider) -> None:
        ratings = await provider.get_ratings("nonexistent")
        assert ratings == []

    @pytest.mark.asyncio
    async def test_get_ratings_returns_all(self, provider: SQLiteFeedbackProvider) -> None:
        await provider.submit_rating("sess-1", "ARTIST", "Carl Cox", 1)
        await provider.submit_rating("sess-1", "VENUE", "Tresor", -1)
        await provider.submit_rating("sess-1", "PROMOTER", "Tresor Records", 1)

        ratings = await provider.get_ratings("sess-1")
        assert len(ratings) == 3

    @pytest.mark.asyncio
    async def test_get_ratings_session_isolation(self, provider: SQLiteFeedbackProvider) -> None:
        await provider.submit_rating("sess-1", "ARTIST", "Carl Cox", 1)
        await provider.submit_rating("sess-2", "ARTIST", "Jeff Mills", -1)

        ratings_1 = await provider.get_ratings("sess-1")
        ratings_2 = await provider.get_ratings("sess-2")
        assert len(ratings_1) == 1
        assert len(ratings_2) == 1
        assert ratings_1[0]["item_key"] == "Carl Cox"
        assert ratings_2[0]["item_key"] == "Jeff Mills"

    @pytest.mark.asyncio
    async def test_get_summary_empty(self, provider: SQLiteFeedbackProvider) -> None:
        summary = await provider.get_rating_summary()
        assert summary["total_ratings"] == 0
        assert summary["positive"] == 0
        assert summary["negative"] == 0
        assert summary["by_type"] == {}

    @pytest.mark.asyncio
    async def test_get_summary_aggregate(self, provider: SQLiteFeedbackProvider) -> None:
        await provider.submit_rating("s1", "ARTIST", "Carl Cox", 1)
        await provider.submit_rating("s1", "ARTIST", "Jeff Mills", -1)
        await provider.submit_rating("s1", "VENUE", "Tresor", 1)
        await provider.submit_rating("s2", "ARTIST", "Derrick May", 1)

        summary = await provider.get_rating_summary()
        assert summary["total_ratings"] == 4
        assert summary["positive"] == 3
        assert summary["negative"] == 1
        assert summary["by_type"]["ARTIST"]["total"] == 3
        assert summary["by_type"]["ARTIST"]["positive"] == 2
        assert summary["by_type"]["ARTIST"]["negative"] == 1
        assert summary["by_type"]["VENUE"]["total"] == 1

    @pytest.mark.asyncio
    async def test_get_summary_filtered(self, provider: SQLiteFeedbackProvider) -> None:
        await provider.submit_rating("s1", "ARTIST", "Carl Cox", 1)
        await provider.submit_rating("s1", "VENUE", "Tresor", -1)

        summary = await provider.get_rating_summary(item_type="ARTIST")
        assert summary["total_ratings"] == 1
        assert summary["by_type"]["ARTIST"]["total"] == 1
        assert "VENUE" not in summary["by_type"]

    @pytest.mark.asyncio
    async def test_provider_name(self, provider: SQLiteFeedbackProvider) -> None:
        assert provider.get_provider_name() == "sqlite_feedback"

    @pytest.mark.asyncio
    async def test_item_type_case_insensitive(self, provider: SQLiteFeedbackProvider) -> None:
        await provider.submit_rating("s1", "artist", "Carl Cox", 1)
        ratings = await provider.get_ratings("s1")
        assert len(ratings) == 1
        assert ratings[0]["item_type"] == "ARTIST"

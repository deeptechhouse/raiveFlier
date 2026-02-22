"""Unit tests for MemoryCacheProvider and ProgressTracker."""

from __future__ import annotations

import pytest

from src.models.pipeline import PipelinePhase
from src.pipeline.progress_tracker import ProgressTracker
from src.providers.cache.memory_cache import MemoryCacheProvider


# ======================================================================
# MemoryCacheProvider
# ======================================================================


class TestMemoryCacheProvider:
    @pytest.fixture()
    def cache(self) -> MemoryCacheProvider:
        return MemoryCacheProvider(max_size=100, ttl=3600)

    @pytest.mark.asyncio
    async def test_get_missing_key_returns_none(self, cache: MemoryCacheProvider) -> None:
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get(self, cache: MemoryCacheProvider) -> None:
        await cache.set("key1", "value1")
        result = await cache.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_set_overwrites_existing(self, cache: MemoryCacheProvider) -> None:
        await cache.set("key1", "old")
        await cache.set("key1", "new")
        result = await cache.get("key1")
        assert result == "new"

    @pytest.mark.asyncio
    async def test_delete_removes_key(self, cache: MemoryCacheProvider) -> None:
        await cache.set("key1", "value1")
        await cache.delete("key1")
        result = await cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_noop(self, cache: MemoryCacheProvider) -> None:
        await cache.delete("nonexistent")  # should not raise

    @pytest.mark.asyncio
    async def test_exists_returns_true_for_present_key(self, cache: MemoryCacheProvider) -> None:
        await cache.set("key1", "value1")
        assert await cache.exists("key1") is True

    @pytest.mark.asyncio
    async def test_exists_returns_false_for_missing_key(self, cache: MemoryCacheProvider) -> None:
        assert await cache.exists("missing") is False

    @pytest.mark.asyncio
    async def test_stores_complex_values(self, cache: MemoryCacheProvider) -> None:
        data = {"artists": ["Carl Cox", "Jeff Mills"], "count": 2}
        await cache.set("complex", data)
        result = await cache.get("complex")
        assert result == data


# ======================================================================
# ProgressTracker
# ======================================================================


class TestProgressTracker:
    @pytest.fixture()
    def tracker(self) -> ProgressTracker:
        return ProgressTracker()

    @pytest.mark.asyncio
    async def test_update_stores_status(self, tracker: ProgressTracker) -> None:
        await tracker.update("s1", PipelinePhase.OCR, 25.0, "Running OCR")
        status = tracker.get_status("s1")
        assert status["phase"] == "OCR"
        assert status["progress"] == 25.0
        assert status["message"] == "Running OCR"

    @pytest.mark.asyncio
    async def test_get_status_unknown_session(self, tracker: ProgressTracker) -> None:
        status = tracker.get_status("unknown")
        assert status["phase"] == "UPLOAD"
        assert status["progress"] == 0.0
        assert status["message"] == ""

    @pytest.mark.asyncio
    async def test_progress_clamped_to_0_100(self, tracker: ProgressTracker) -> None:
        await tracker.update("s1", PipelinePhase.OCR, -10.0, "Negative")
        assert tracker.get_status("s1")["progress"] == 0.0

        await tracker.update("s1", PipelinePhase.OCR, 150.0, "Over")
        assert tracker.get_status("s1")["progress"] == 100.0

    @pytest.mark.asyncio
    async def test_register_and_notify_async_listener(self, tracker: ProgressTracker) -> None:
        received: list[tuple] = []

        async def callback(sid: str, phase: PipelinePhase, prog: float, msg: str) -> None:
            received.append((sid, phase, prog, msg))

        tracker.register_listener("s1", callback)
        await tracker.update("s1", PipelinePhase.RESEARCH, 50.0, "Researching")

        assert len(received) == 1
        assert received[0][0] == "s1"
        assert received[0][2] == 50.0

    @pytest.mark.asyncio
    async def test_register_and_notify_sync_listener(self, tracker: ProgressTracker) -> None:
        received: list[tuple] = []

        def callback(sid: str, phase: PipelinePhase, prog: float, msg: str) -> None:
            received.append((sid, phase, prog, msg))

        tracker.register_listener("s1", callback)
        await tracker.update("s1", PipelinePhase.RESEARCH, 50.0, "Researching")

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_unregister_listener(self, tracker: ProgressTracker) -> None:
        received: list[tuple] = []

        async def callback(sid: str, phase: PipelinePhase, prog: float, msg: str) -> None:
            received.append((sid, phase, prog, msg))

        tracker.register_listener("s1", callback)
        tracker.unregister_listener("s1", callback)
        await tracker.update("s1", PipelinePhase.OCR, 10.0, "Should not notify")

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_is_noop(self, tracker: ProgressTracker) -> None:
        async def callback(sid: str, phase: PipelinePhase, prog: float, msg: str) -> None:
            pass

        tracker.unregister_listener("s1", callback)  # should not raise

    @pytest.mark.asyncio
    async def test_duplicate_register_ignored(self, tracker: ProgressTracker) -> None:
        received: list[tuple] = []

        async def callback(sid: str, phase: PipelinePhase, prog: float, msg: str) -> None:
            received.append((sid, phase, prog, msg))

        tracker.register_listener("s1", callback)
        tracker.register_listener("s1", callback)  # duplicate
        await tracker.update("s1", PipelinePhase.OCR, 10.0, "test")

        assert len(received) == 1  # called only once

    @pytest.mark.asyncio
    async def test_listener_error_isolation(self, tracker: ProgressTracker) -> None:
        received: list[tuple] = []

        async def bad_callback(sid: str, phase: PipelinePhase, prog: float, msg: str) -> None:
            raise RuntimeError("Listener broke")

        async def good_callback(sid: str, phase: PipelinePhase, prog: float, msg: str) -> None:
            received.append((sid, phase, prog, msg))

        tracker.register_listener("s1", bad_callback)
        tracker.register_listener("s1", good_callback)
        await tracker.update("s1", PipelinePhase.OUTPUT, 100.0, "Done")

        # Good callback still fires despite bad one raising
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_no_listeners_no_error(self, tracker: ProgressTracker) -> None:
        # Update without any listeners should not raise
        await tracker.update("s1", PipelinePhase.UPLOAD, 0.0, "Starting")

    @pytest.mark.asyncio
    async def test_multiple_sessions_independent(self, tracker: ProgressTracker) -> None:
        await tracker.update("s1", PipelinePhase.OCR, 30.0, "OCR session 1")
        await tracker.update("s2", PipelinePhase.RESEARCH, 60.0, "Research session 2")

        s1 = tracker.get_status("s1")
        s2 = tracker.get_status("s2")

        assert s1["phase"] == "OCR"
        assert s2["phase"] == "RESEARCH"

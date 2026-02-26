"""Unit tests for AudioTranscriber service.

Tests provider selection, format validation, and audio conversion logic
with mocked transcription providers.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.raive_feeder.config.settings import FeederSettings
from tools.raive_feeder.services.audio_transcriber import AudioTranscriber


@pytest.fixture
def settings():
    return FeederSettings(
        openai_api_key="test-key",
        whisper_model_size="base",
    )


@pytest.fixture
def transcriber(settings):
    return AudioTranscriber(settings=settings)


class TestProviderSelection:
    """Tests for _get_provider dispatch."""

    def test_local_provider_selection(self, transcriber):
        """Requesting whisper_local should return WhisperLocalProvider."""
        provider = transcriber._get_provider("whisper_local")
        assert "whisper_local" in provider.get_provider_name()

    def test_api_provider_selection(self, transcriber):
        """Requesting whisper_api should return WhisperAPIProvider."""
        provider = transcriber._get_provider("whisper_api")
        assert "whisper_api" in provider.get_provider_name()

    def test_api_provider_without_key_raises(self):
        """Whisper API without an API key should raise ValueError."""
        settings = FeederSettings(openai_api_key="")
        transcriber = AudioTranscriber(settings=settings)
        with pytest.raises(ValueError, match="API key required"):
            transcriber._get_provider("whisper_api")


class TestFormatValidation:
    """Tests for audio format validation."""

    @pytest.mark.asyncio
    async def test_unsupported_format_raises(self, transcriber):
        """Unsupported audio format should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported audio format"):
            await transcriber.transcribe("/tmp/test.xyz")


class TestTranscribeFlow:
    """Tests for the full transcribe pipeline with mocked provider."""

    @pytest.mark.asyncio
    async def test_transcribe_success(self, transcriber):
        """Successful transcription returns expected structure."""
        mock_result = MagicMock()
        mock_result.text = "Hello world"
        mock_result.language = "en"
        mock_result.duration_seconds = 5.0
        mock_result.segments = []

        mock_provider = MagicMock()
        mock_provider.transcribe = AsyncMock(return_value=mock_result)
        mock_provider.get_provider_name.return_value = "test_provider"
        mock_provider.supported_formats.return_value = [".wav", ".mp3"]

        with patch.object(transcriber, "_get_provider", return_value=mock_provider):
            result = await transcriber.transcribe("/tmp/test.wav")

        assert result["text"] == "Hello world"
        assert result["language"] == "en"
        assert result["duration"] == 5.0
        assert result["provider"] == "test_provider"

"""Audio transcription orchestrator service.

# ─── DESIGN ────────────────────────────────────────────────────────────
#
# AudioTranscriber sits between the API layer and the transcription
# providers.  It handles:
#   1. Audio format validation and conversion (via pydub/ffmpeg)
#   2. Provider selection (local Whisper vs. API, based on user choice)
#   3. Transcript post-processing (cleanup, normalization)
#
# The user reviews/edits the transcript in the UI before ingestion.
# This service does NOT auto-ingest — it returns the raw transcript
# for human review.
#
# Pattern: Facade (wraps provider complexity behind a simple interface).
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import structlog

from tools.raive_feeder.config.settings import FeederSettings

logger = structlog.get_logger(logger_name=__name__)

# Audio formats that may need conversion to WAV before local Whisper.
_SUPPORTED_AUDIO = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".webm", ".mp4", ".mpeg", ".mpga"}


class AudioTranscriber:
    """Orchestrates audio transcription via selectable providers.

    Parameters
    ----------
    settings:
        FeederSettings with whisper model size and API keys.
    """

    def __init__(self, settings: FeederSettings) -> None:
        self._settings = settings

    async def transcribe(
        self,
        audio_path: str,
        provider_name: str = "whisper_local",
        language: str | None = None,
    ) -> dict[str, Any]:
        """Transcribe an audio file and return the result.

        Parameters
        ----------
        audio_path:
            Path to the audio file on disk.
        provider_name:
            "whisper_local" or "whisper_api".
        language:
            Optional ISO 639-1 code.  None for auto-detect.

        Returns
        -------
        dict with keys: text, language, duration, provider, segments
        """
        # Validate file extension.
        suffix = Path(audio_path).suffix.lower()
        if suffix not in _SUPPORTED_AUDIO:
            raise ValueError(f"Unsupported audio format: {suffix}")

        # Select provider.
        provider = self._get_provider(provider_name)

        # Convert to WAV if the provider needs it and format isn't directly supported.
        if suffix not in provider.supported_formats():
            audio_path = await self._convert_to_wav(audio_path)

        # Transcribe.
        result = await provider.transcribe(audio_path, language=language)

        return {
            "text": result.text,
            "language": result.language,
            "duration": result.duration_seconds,
            "provider": provider.get_provider_name(),
            "segments": result.segments,
        }

    def _get_provider(self, name: str) -> Any:
        """Instantiate the requested transcription provider."""
        if name == "whisper_api":
            from tools.raive_feeder.providers.transcription.whisper_api_provider import (
                WhisperAPIProvider,
            )
            if not self._settings.openai_api_key:
                raise ValueError("OpenAI API key required for Whisper API provider")
            return WhisperAPIProvider(api_key=self._settings.openai_api_key)
        else:
            from tools.raive_feeder.providers.transcription.whisper_local_provider import (
                WhisperLocalProvider,
            )
            return WhisperLocalProvider(
                model_size=self._settings.whisper_model_size,
            )

    @staticmethod
    async def _convert_to_wav(audio_path: str) -> str:
        """Convert audio to WAV format using pydub (requires ffmpeg)."""
        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "ffmpeg not installed. Install via: brew install ffmpeg "
                "(macOS) or apt install ffmpeg (Linux)"
            )

        from pydub import AudioSegment

        audio = AudioSegment.from_file(audio_path)
        wav_path = tempfile.mktemp(suffix=".wav")
        audio.export(wav_path, format="wav")

        logger.info("audio_converted_to_wav", source=audio_path, output=wav_path)
        return wav_path

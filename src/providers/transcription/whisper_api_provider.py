"""OpenAI Whisper API transcription provider.

# ─── CLOUD TRANSCRIPTION ────────────────────────────────────────────
#
# The Whisper API is a cloud-based option for audio transcription.
# Used by the Rave Stories feature to convert audio submissions to text.
#
# Cost: $0.006 per minute of audio.
# Max file size: 25 MB per request.
# Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, webm.
#
# The API handles all preprocessing internally — no pydub/ffmpeg needed.
#
# Copied from tools/raive_feeder/providers/transcription/ and relocated
# to src/providers/transcription/ with updated imports pointing to the
# canonical ITranscriptionProvider at src/interfaces/.
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path

import structlog

from src.interfaces.transcription_provider import (
    ITranscriptionProvider,
    TranscriptionResult,
)

logger = structlog.get_logger(logger_name=__name__)


class WhisperAPIProvider(ITranscriptionProvider):
    """Transcription via the OpenAI Whisper API ($0.006/min).

    Parameters
    ----------
    api_key:
        OpenAI API key.  Required for authentication.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio using the OpenAI Whisper API."""
        # Deferred import — only load openai when actually transcribing.
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self._api_key)

        audio_file = Path(audio_path)
        with open(audio_file, "rb") as f:
            kwargs = {
                "model": "whisper-1",
                "file": f,
                "response_format": "verbose_json",
            }
            if language:
                kwargs["language"] = language

            response = await client.audio.transcriptions.create(**kwargs)

        # Parse the verbose JSON response which includes segments.
        segments = []
        if hasattr(response, "segments") and response.segments:
            for seg in response.segments:
                segments.append({
                    "start": seg.get("start", 0.0) if isinstance(seg, dict) else getattr(seg, "start", 0.0),
                    "end": seg.get("end", 0.0) if isinstance(seg, dict) else getattr(seg, "end", 0.0),
                    "text": (seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")).strip(),
                })

        duration = getattr(response, "duration", 0.0) or 0.0
        detected_language = getattr(response, "language", language or "en")

        logger.info(
            "whisper_api_transcription_complete",
            duration=duration,
            language=detected_language,
            segments=len(segments),
        )

        return TranscriptionResult(
            text=response.text,
            language=detected_language,
            duration_seconds=duration,
            segments=segments,
        )

    def get_provider_name(self) -> str:
        return "whisper_api (OpenAI)"

    def is_available(self) -> bool:
        """Available if openai SDK is installed and API key is set."""
        if not self._api_key:
            return False
        try:
            import openai  # noqa: F401
            return True
        except ImportError:
            return False

    def supported_formats(self) -> list[str]:
        return [".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"]

"""Abstract base class for audio transcription providers.

# ─── ADAPTER PATTERN ───────────────────────────────────────────────────
#
# ITranscriptionProvider follows the same adapter pattern used throughout
# raiveFlier (ILLMProvider, IOCRProvider, etc.).  Concrete implementations
# wrap a specific transcription backend (faster-whisper local, OpenAI API)
# behind this common interface so the AudioTranscriber service doesn't
# need to know which backend is in use.
#
# The user selects the provider per-job in the UI (toggle between local
# and cloud transcription).  Both produce identical TranscriptionResult
# objects.
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field


class TranscriptionResult(BaseModel):
    """Immutable result from an audio transcription.

    Contains the full transcribed text, detected language, segment-level
    timing data, and the duration of the audio file.
    """
    model_config = ConfigDict(frozen=True)

    text: str = Field(description="Full transcribed text.")
    language: str = Field(default="en", description="Detected or specified language code.")
    duration_seconds: float = Field(default=0.0, description="Audio duration in seconds.")
    segments: list[dict[str, float | str]] = Field(
        default_factory=list,
        description="Segment-level data: [{'start': 0.0, 'end': 2.5, 'text': '...'}]",
    )


class ITranscriptionProvider(ABC):
    """Contract for audio transcription backends.

    Implementations must handle audio file reading, preprocessing (format
    conversion if needed), and transcription.  The provider is expected
    to support at least WAV input; other formats may require pydub/ffmpeg
    conversion before calling the underlying engine.
    """

    @abstractmethod
    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file to text.

        Parameters
        ----------
        audio_path:
            Path to the audio file on disk.
        language:
            Optional ISO 639-1 language code (e.g. "en", "de").
            If None, the provider should auto-detect the language.

        Returns
        -------
        TranscriptionResult
            The transcription with full text, language, and segments.
        """

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return a human-readable name for this provider."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the provider is ready to accept transcription requests."""

    @abstractmethod
    def supported_formats(self) -> list[str]:
        """Return list of supported audio file extensions (e.g. ['.wav', '.mp3'])."""

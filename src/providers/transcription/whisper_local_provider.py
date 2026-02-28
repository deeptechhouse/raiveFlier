"""Local Whisper transcription provider using faster-whisper (CTranslate2).

# ─── LOCAL TRANSCRIPTION ─────────────────────────────────────────────
#
# faster-whisper uses CTranslate2 for inference, which is significantly
# faster than OpenAI's original Whisper implementation and uses less RAM.
# It runs entirely locally with no API costs.
#
# Model sizes and approximate requirements:
#   tiny   — ~75 MB RAM, fast, lower accuracy
#   base   — ~150 MB RAM, good balance
#   small  — ~500 MB RAM, better accuracy
#   medium — ~1.5 GB RAM, near-large accuracy
#   large-v3 — ~3 GB RAM, best accuracy
#
# For the Rave Stories feature, "tiny" is used by default to stay within
# the 512MB RAM budget on Render (CLAUDE.md deployment constraints).
#
# Copied from tools/raive_feeder/providers/transcription/ and relocated
# to src/providers/transcription/ with updated imports.
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import structlog

from src.interfaces.transcription_provider import (
    ITranscriptionProvider,
    TranscriptionResult,
)

logger = structlog.get_logger(logger_name=__name__)


class WhisperLocalProvider(ITranscriptionProvider):
    """Transcription via faster-whisper running locally on CPU/GPU.

    Parameters
    ----------
    model_size:
        Whisper model variant: tiny, base, small, medium, large-v2, large-v3.
    device:
        Compute device: "cpu" or "cuda".  Auto-detected if not specified.
    compute_type:
        CTranslate2 compute type: "int8" (fast/low RAM), "float16" (GPU),
        "float32" (CPU accurate).  Defaults to "int8" for CPU efficiency.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model = None

    def _ensure_model(self) -> None:
        """Lazy-load the model on first use to avoid startup RAM overhead."""
        if self._model is not None:
            return

        # Deferred import — only load faster_whisper when actually needed.
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )
        logger.info(
            "whisper_model_loaded",
            model=self._model_size,
            device=self._device,
        )

    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio using faster-whisper locally."""
        self._ensure_model()
        assert self._model is not None

        kwargs = {"beam_size": 5}
        if language:
            kwargs["language"] = language

        segments_iter, info = self._model.transcribe(audio_path, **kwargs)

        # Collect all segments from the generator.
        segments = []
        full_text_parts = []
        for seg in segments_iter:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            })
            full_text_parts.append(seg.text.strip())

        full_text = " ".join(full_text_parts)

        logger.info(
            "whisper_local_transcription_complete",
            duration=info.duration,
            language=info.language,
            segments=len(segments),
        )

        return TranscriptionResult(
            text=full_text,
            language=info.language or "en",
            duration_seconds=info.duration,
            segments=segments,
        )

    def get_provider_name(self) -> str:
        return f"whisper_local ({self._model_size})"

    def is_available(self) -> bool:
        """Check if faster-whisper is importable."""
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def supported_formats(self) -> list[str]:
        return [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm"]

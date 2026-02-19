"""OpenAI-compatible LLM provider adapter.

Wraps the ``openai`` async client to implement :class:`ILLMProvider`.
Supports both text completion and vision analysis.  When a custom
``openai_base_url`` is configured (e.g. TogetherAI, Anyscale, Fireworks),
the client points at that URL instead of the default OpenAI endpoint.
"""

from __future__ import annotations

import base64

import openai
import structlog

from src.config.settings import Settings
from src.interfaces.llm_provider import ILLMProvider
from src.utils.errors import LLMError

logger = structlog.get_logger(logger_name=__name__)


def _detect_media_type(image_bytes: bytes) -> str:
    """Detect the MIME type of an image from its magic bytes."""
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    return "image/jpeg"  # safe fallback


class OpenAILLMProvider(ILLMProvider):
    """LLM provider backed by an OpenAI-compatible API.

    Uses ``gpt-4o`` for vision tasks and ``gpt-4o-mini`` for text-only
    completions by default.  These can be overridden via settings for
    OpenAI-compatible providers like TogetherAI, Anyscale, or Fireworks.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._api_key = settings.openai_api_key

        # Build client kwargs — add base_url only when configured
        client_kwargs: dict = {"api_key": self._api_key}
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url

        self._client = openai.AsyncOpenAI(**client_kwargs)
        self._text_model = settings.openai_text_model or "gpt-4o-mini"
        self._vision_model = settings.openai_vision_model or "gpt-4o"
        self._has_vision = bool(settings.openai_vision_model) or not settings.openai_base_url
        self._provider_label = (
            "openai-compatible" if settings.openai_base_url else "openai"
        )

    # ------------------------------------------------------------------
    # ILLMProvider implementation
    # ------------------------------------------------------------------

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 4000,
    ) -> str:
        """Generate a text completion via the OpenAI-compatible chat API."""
        try:
            response = await self._client.chat.completions.create(
                model=self._text_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if content is None:
                raise LLMError(
                    message=f"{self._provider_label} returned empty response",
                    provider_name=self.get_provider_name(),
                )
            logger.info(
                "openai_completion",
                model=self._text_model,
                provider=self._provider_label,
                tokens=response.usage.total_tokens if response.usage else None,
            )
            return content
        except openai.APIError as exc:
            raise LLMError(
                message=f"{self._provider_label} API error: {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

    async def vision_extract(self, image_bytes: bytes, prompt: str) -> str:
        """Analyse an image using the configured vision model."""
        if not self._has_vision:
            raise LLMError(
                message="Vision not supported by this provider configuration",
                provider_name=self.get_provider_name(),
            )
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        media_type = _detect_media_type(image_bytes)
        try:
            response = await self._client.chat.completions.create(
                model=self._vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{b64}",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=4000,
            )
            content = response.choices[0].message.content
            if content is None:
                raise LLMError(
                    message=f"{self._provider_label} vision returned empty response",
                    provider_name=self.get_provider_name(),
                )
            logger.info(
                "openai_vision_extract",
                model=self._vision_model,
                provider=self._provider_label,
                tokens=response.usage.total_tokens if response.usage else None,
            )
            return content
        except openai.APIError as exc:
            raise LLMError(
                message=f"{self._provider_label} vision API error: {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

    def supports_vision(self) -> bool:
        return self._has_vision

    def is_available(self) -> bool:
        """Return ``True`` if an API key is configured."""
        return bool(self._api_key)

    async def validate_credentials(self) -> bool:
        """Try listing models to verify the API key works."""
        if not self.is_available():
            return False
        try:
            await self._client.models.list()
            return True
        except openai.APIError:
            return False

    def get_provider_name(self) -> str:
        return self._provider_label

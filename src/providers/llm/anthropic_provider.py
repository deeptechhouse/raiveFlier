"""Anthropic LLM provider adapter.

Wraps the ``anthropic`` async client to implement :class:`ILLMProvider`.
Supports both text completion and vision analysis via the Claude Messages API.
"""

from __future__ import annotations

import base64

import anthropic
import structlog

from src.config.settings import Settings
from src.interfaces.llm_provider import ILLMProvider
from src.utils.errors import LLMError

logger = structlog.get_logger(logger_name=__name__)


class AnthropicLLMProvider(ILLMProvider):
    """LLM provider backed by the Anthropic Claude API.

    Uses ``claude-sonnet-4-20250514`` by default for both text and vision tasks.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._api_key = settings.anthropic_api_key
        self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        self._model = "claude-sonnet-4-20250514"

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
        """Generate a text completion via the Anthropic Messages API."""
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
            )
            text_blocks = [block.text for block in response.content if block.type == "text"]
            if not text_blocks:
                raise LLMError(
                    message="Anthropic returned no text content",
                    provider_name=self.get_provider_name(),
                )
            result = "\n".join(text_blocks)
            logger.info(
                "anthropic_completion",
                model=self._model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
            return result
        except anthropic.APIError as exc:
            raise LLMError(
                message=f"Anthropic API error: {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

    async def vision_extract(self, image_bytes: bytes, prompt: str) -> str:
        """Analyse an image using Claude's vision capability."""
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=4000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": b64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            text_blocks = [block.text for block in response.content if block.type == "text"]
            if not text_blocks:
                raise LLMError(
                    message="Anthropic vision returned no text content",
                    provider_name=self.get_provider_name(),
                )
            result = "\n".join(text_blocks)
            logger.info(
                "anthropic_vision_extract",
                model=self._model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
            return result
        except anthropic.APIError as exc:
            raise LLMError(
                message=f"Anthropic vision API error: {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

    def supports_vision(self) -> bool:
        return True

    def is_available(self) -> bool:
        """Return ``True`` if an Anthropic API key is configured."""
        return bool(self._api_key)

    async def validate_credentials(self) -> bool:
        """Try a minimal completion to verify the API key works."""
        if not self.is_available():
            return False
        try:
            await self._client.messages.create(
                model=self._model,
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True
        except anthropic.APIError:
            return False

    def get_provider_name(self) -> str:
        return "anthropic"

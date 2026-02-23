"""Anthropic LLM provider adapter.

Wraps the ``anthropic`` async client to implement :class:`ILLMProvider`.
Supports both text completion and vision analysis via the Claude Messages API.

Key differences from OpenAI adapter:
    - Uses Anthropic's Messages API (not chat.completions)
    - System prompt is a separate parameter, not a message in the list
    - Vision uses "image" content type with base64 source (not image_url)
    - Response content is a list of blocks (may include text + tool_use),
      so we filter for text blocks and join them
    - Claude always supports vision — no capability flag needed
"""

from __future__ import annotations

import base64

# The official Anthropic Python SDK (async version).
import anthropic
import structlog

from src.config.settings import Settings
from src.interfaces.llm_provider import ILLMProvider
from src.utils.errors import LLMError

logger = structlog.get_logger(logger_name=__name__)


def _detect_media_type(image_bytes: bytes) -> str:
    """Detect the MIME type of an image from its magic bytes.
    Same logic as openai_provider.py — see that file for detailed comments.
    """
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    return "image/jpeg"  # safe fallback


class AnthropicLLMProvider(ILLMProvider):
    """LLM provider backed by the Anthropic Claude API.

    Uses ``claude-sonnet-4-20250514`` by default for both text and vision tasks.
    Claude Sonnet provides excellent vision capabilities for reading stylized
    rave flier typography, making it the preferred OCR backend when available.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # API key from ANTHROPIC_API_KEY env var.
        self._api_key = settings.anthropic_api_key
        # AsyncAnthropic is the async client — all calls return coroutines.
        self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        # Single model for both text and vision — Claude handles both natively.
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
        """Generate a text completion via the Anthropic Messages API.

        Note the key difference from OpenAI: the system_prompt is a
        top-level parameter here, not a message in the messages list.
        """
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                # Anthropic takes system prompt as a separate kwarg, not a message.
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
            )
            # Anthropic responses contain a list of content blocks (text, tool_use,
            # etc.). We filter for text blocks only and join them. In practice,
            # non-tool-use completions will have exactly one text block.
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
        """Analyse an image using Claude's vision capability.

        Anthropic's vision API differs from OpenAI's:
            - Content type is "image" (not "image_url")
            - Source is an object with type, media_type, and data fields
            - Image goes BEFORE the text prompt in the content array
        """
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        media_type = _detect_media_type(image_bytes)
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=4000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            # Anthropic vision format: image content block with
                            # inline base64 data (not a URL like OpenAI).
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
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

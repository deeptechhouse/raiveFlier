"""Ollama LLM provider adapter.

Wraps a local Ollama server via its OpenAI-compatible API endpoint.
Uses the ``openai`` client library pointed at the Ollama base URL.

Ollama is a free, open-source tool for running LLMs locally. This adapter
enables raiveFlier to work completely offline with no API costs. The trade-off
is that local models (llama3.1, llava) are typically less capable than
cloud models (GPT-4o, Claude) for entity extraction and analysis.

Setup: Install Ollama (https://ollama.ai), then ``ollama pull llama3.1``
and ``ollama pull llava`` for vision. Set OLLAMA_BASE_URL=http://localhost:11434
"""

from __future__ import annotations

import base64

# httpx is an async HTTP client (like requests but async-native).
# Used here only for validate_credentials() to check if Ollama is running.
import httpx
import openai
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


class OllamaLLMProvider(ILLMProvider):
    """LLM provider backed by a local Ollama server.

    Ollama exposes an OpenAI-compatible ``/v1`` API, so this adapter
    reuses the ``openai.AsyncOpenAI`` client pointed at the local URL.
    Defaults to ``llama3.1`` for text and ``llava`` for vision.

    This is clever reuse — instead of writing a separate HTTP client for
    Ollama's native API, we leverage the fact that Ollama supports the
    OpenAI protocol and reuse the same openai SDK with a different base_url.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # OLLAMA_BASE_URL env var, typically "http://localhost:11434".
        self._base_url = settings.ollama_base_url
        # Point the OpenAI client at Ollama's /v1 endpoint.
        self._client = openai.AsyncOpenAI(
            base_url=f"{self._base_url.rstrip('/')}/v1",
            # Ollama doesn't require an API key, but the openai SDK requires
            # the parameter to be non-empty. "ollama" is a dummy value.
            api_key="ollama",
        )
        # Default local models — user can pull different ones with `ollama pull`.
        self._text_model = "llama3.1"
        # llava is a vision-language model that can process images locally.
        self._vision_model = "llava"

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
        """Generate a text completion via Ollama's OpenAI-compatible API."""
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
                    message="Ollama returned empty response",
                    provider_name=self.get_provider_name(),
                )
            logger.info("ollama_completion", model=self._text_model)
            return content
        except openai.APIError as exc:
            raise LLMError(
                message=f"Ollama API error: {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

    async def vision_extract(self, image_bytes: bytes, prompt: str) -> str:
        """Analyse an image using Ollama's vision model (llava)."""
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
                    message="Ollama vision returned empty response",
                    provider_name=self.get_provider_name(),
                )
            logger.info("ollama_vision_extract", model=self._vision_model)
            return content
        except openai.APIError as exc:
            raise LLMError(
                message=f"Ollama vision API error: {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

    def supports_vision(self) -> bool:
        return True

    def is_available(self) -> bool:
        """Return ``True`` if the Ollama base URL is configured."""
        return bool(self._base_url)

    async def validate_credentials(self) -> bool:
        """Try listing models from the Ollama server.

        Unlike OpenAI/Anthropic which verify API keys, this checks that
        the Ollama server is actually running and reachable at the configured URL.
        The /api/tags endpoint lists installed models without running inference.
        """
        if not self.is_available():
            return False
        try:
            # Use httpx directly (not the openai client) to hit Ollama's
            # native /api/tags endpoint which lists installed models.
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._base_url.rstrip('/')}/api/tags")
                return response.status_code == 200
        except httpx.HTTPError:
            return False

    def get_provider_name(self) -> str:
        return "ollama"

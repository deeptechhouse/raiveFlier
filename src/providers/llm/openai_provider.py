"""OpenAI LLM provider adapter.

Wraps the ``openai`` async client to implement :class:`ILLMProvider`.
Supports both text completion (GPT-4o-mini) and vision analysis (GPT-4o).
"""

from __future__ import annotations

import base64

import openai
import structlog

from src.config.settings import Settings
from src.interfaces.llm_provider import ILLMProvider
from src.utils.errors import LLMError

logger = structlog.get_logger(logger_name=__name__)


class OpenAILLMProvider(ILLMProvider):
    """LLM provider backed by the OpenAI API.

    Uses ``gpt-4o`` for vision tasks and ``gpt-4o-mini`` for text-only
    completions by default.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._api_key = settings.openai_api_key
        self._client = openai.AsyncOpenAI(api_key=self._api_key)
        self._text_model = "gpt-4o-mini"
        self._vision_model = "gpt-4o"

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
        """Generate a text completion via the OpenAI chat API."""
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
                    message="OpenAI returned empty response",
                    provider_name=self.get_provider_name(),
                )
            logger.info(
                "openai_completion",
                model=self._text_model,
                tokens=response.usage.total_tokens if response.usage else None,
            )
            return content
        except openai.APIError as exc:
            raise LLMError(
                message=f"OpenAI API error: {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

    async def vision_extract(self, image_bytes: bytes, prompt: str) -> str:
        """Analyse an image using GPT-4o vision."""
        b64 = base64.b64encode(image_bytes).decode("utf-8")
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
                                    "url": f"data:image/jpeg;base64,{b64}",
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
                    message="OpenAI vision returned empty response",
                    provider_name=self.get_provider_name(),
                )
            logger.info(
                "openai_vision_extract",
                model=self._vision_model,
                tokens=response.usage.total_tokens if response.usage else None,
            )
            return content
        except openai.APIError as exc:
            raise LLMError(
                message=f"OpenAI vision API error: {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

    def supports_vision(self) -> bool:
        return True

    def is_available(self) -> bool:
        """Return ``True`` if an OpenAI API key is configured."""
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
        return "openai"

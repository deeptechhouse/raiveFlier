"""OpenAI-compatible LLM provider adapter.

Wraps the ``openai`` async client to implement :class:`ILLMProvider`.
Supports both text completion and vision analysis.  When a custom
``openai_base_url`` is configured (e.g. TogetherAI, Anyscale, Fireworks),
the client points at that URL instead of the default OpenAI endpoint.

This is the most versatile LLM adapter because many third-party LLM
providers (TogetherAI, Anyscale, Fireworks, Groq) expose OpenAI-compatible
REST APIs. By pointing the openai client at a different base_url, this
single adapter can talk to dozens of different model providers.
"""

from __future__ import annotations

# base64 is used to encode image bytes into a base64 string for the vision API.
# The OpenAI vision endpoint requires images as base64 data URIs.
import base64

# The official OpenAI Python SDK (async version). Provides type-safe access
# to the chat completions, models, and other endpoints.
import openai
# structlog provides structured JSON logging (see src/utils/logging.py).
# Every log entry includes context like model name and token usage.
import structlog

# Settings is the Pydantic Settings class that loads env vars and config.
from src.config.settings import Settings
# ILLMProvider is the abstract interface this class implements.
from src.interfaces.llm_provider import ILLMProvider
# LLMError is a custom exception that wraps provider-specific errors with
# a consistent interface for the rest of the codebase to catch.
from src.utils.errors import LLMError

logger = structlog.get_logger(logger_name=__name__)


def _detect_media_type(image_bytes: bytes) -> str:
    """Detect the MIME type of an image from its magic bytes.

    Magic bytes are the first few bytes of a file that identify its format.
    This is more reliable than trusting file extensions. We need the correct
    MIME type for the base64 data URI sent to the vision API.

    PNG starts with: 89 50 4E 47 0D 0A 1A 0A
    WEBP starts with: RIFF....WEBP
    JPEG starts with: FF D8
    """
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    return "image/jpeg"  # safe fallback — JPEG is the most common flier format


class OpenAILLMProvider(ILLMProvider):
    """LLM provider backed by an OpenAI-compatible API.

    Uses ``gpt-4o`` for vision tasks and ``gpt-4o-mini`` for text-only
    completions by default.  These can be overridden via settings for
    OpenAI-compatible providers like TogetherAI, Anyscale, or Fireworks.

    This class demonstrates the Adapter Pattern:
        - It implements ILLMProvider (the interface the app expects)
        - It wraps the openai SDK (the third-party library)
        - The rest of the app never imports or calls openai directly
    """

    def __init__(self, settings: Settings) -> None:
        # Store settings for potential future reference.
        self._settings = settings
        # API key loaded from OPENAI_API_KEY env var via Pydantic Settings.
        self._api_key = settings.openai_api_key

        # Build client kwargs — add base_url only when a custom endpoint is
        # configured (e.g. for TogetherAI, Anyscale, or Fireworks).
        # Timeout is set to 25 seconds so the LLM call finishes before
        # Render's 30-second request timeout kills the connection.
        client_kwargs: dict = {
            "api_key": self._api_key,
            "timeout": openai.Timeout(25.0, connect=5.0),
        }
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url

        # AsyncOpenAI is the async version of the OpenAI client. All methods
        # return coroutines and must be awaited. This is essential in a FastAPI
        # app where blocking calls would freeze the event loop.
        self._client = openai.AsyncOpenAI(**client_kwargs)
        # Separate models for text-only and vision tasks — vision models cost more.
        self._text_model = settings.openai_text_model or "gpt-4o-mini"
        self._vision_model = settings.openai_vision_model or "gpt-4o"
        # Assume vision is available unless using a custom base_url that might
        # not support it. Can be explicitly enabled via openai_vision_model setting.
        self._has_vision = bool(settings.openai_vision_model) or not settings.openai_base_url
        # Label used in logs and error messages to identify this provider.
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
        """Generate a text completion via the OpenAI-compatible chat API.

        Used for entity extraction, interconnection analysis, and any other
        text-only LLM tasks in the pipeline. The system_prompt sets the LLM's
        role (e.g. "You are an expert at reading rave fliers") and user_prompt
        contains the actual data to process.
        """
        try:
            # The chat.completions.create() call is the core OpenAI API method.
            # It sends a list of messages (system + user) and returns the model's
            # response. temperature=0.3 keeps output fairly deterministic —
            # we want consistent entity extraction, not creative writing.
            response = await self._client.chat.completions.create(
                model=self._text_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # The response contains a list of "choices" (usually just one).
            # Each choice has a message with content (the model's text output).
            content = response.choices[0].message.content
            if content is None:
                raise LLMError(
                    message=f"{self._provider_label} returned empty response",
                    provider_name=self.get_provider_name(),
                )
            # Log token usage for cost tracking and debugging. structlog
            # automatically includes timestamp, level, and logger name.
            logger.info(
                "openai_completion",
                model=self._text_model,
                provider=self._provider_label,
                tokens=response.usage.total_tokens if response.usage else None,
            )
            return content
        except openai.APITimeoutError as exc:
            # Timeout fires before Render's 30s connection limit so the
            # caller gets a clean error instead of a dropped connection.
            raise LLMError(
                message=f"{self._provider_label} timed out after 25s",
                provider_name=self.get_provider_name(),
            ) from exc
        except openai.APIError as exc:
            # Wrap the SDK-specific exception in our custom LLMError so
            # callers don't need to import openai to catch errors.
            # "from exc" preserves the original stack trace for debugging.
            raise LLMError(
                message=f"{self._provider_label} API error: {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

    async def vision_extract(self, image_bytes: bytes, prompt: str) -> str:
        """Analyse an image using the configured vision model.

        This is the primary OCR path for rave fliers — the LLM looks at the
        image directly and extracts text. Much better at reading stylized/
        distorted rave typography than traditional OCR engines.
        """
        if not self._has_vision:
            raise LLMError(
                message="Vision not supported by this provider configuration",
                provider_name=self.get_provider_name(),
            )
        # Encode image bytes to base64 string — required by the vision API.
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        # Detect MIME type from magic bytes for the data URI.
        media_type = _detect_media_type(image_bytes)
        try:
            # Vision requests use a multi-part content array: one text part
            # (the prompt/instructions) and one image_url part (the base64
            # data URI). The model processes both together.
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
                                    # data URI format: data:<mime>;base64,<encoded_data>
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
        """OpenAI supports vision unless explicitly using a custom endpoint without it."""
        return self._has_vision

    def is_available(self) -> bool:
        """Return ``True`` if an API key is configured (doesn't verify it works)."""
        return bool(self._api_key)

    async def validate_credentials(self) -> bool:
        """Try listing models to verify the API key works.

        This is a lightweight API call that confirms the key is accepted
        without incurring inference costs. Called at startup by main.py.
        """
        if not self.is_available():
            return False
        try:
            await self._client.models.list()
            return True
        except openai.APIError:
            return False

    def get_provider_name(self) -> str:
        """Return 'openai' or 'openai-compatible' depending on configuration."""
        return self._provider_label

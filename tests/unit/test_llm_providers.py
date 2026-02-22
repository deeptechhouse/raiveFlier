"""Unit tests for LLM provider adapters — OpenAI, Anthropic, Ollama."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.settings import Settings


# ======================================================================
# Shared helpers
# ======================================================================

def _settings(**overrides) -> Settings:
    defaults = {
        "openai_api_key": "sk-test",
        "openai_base_url": "",
        "openai_text_model": "",
        "openai_vision_model": "",
        "openai_embedding_model": "",
        "anthropic_api_key": "test-anthropic",
        "ollama_base_url": "http://localhost:11434",
    }
    defaults.update(overrides)
    return Settings(**defaults)


# ======================================================================
# OpenAI LLM Provider
# ======================================================================


class TestOpenAILLMProvider:
    @pytest.fixture()
    def settings(self) -> Settings:
        return _settings()

    def test_get_provider_name(self, settings: Settings) -> None:
        from src.providers.llm.openai_provider import OpenAILLMProvider
        provider = OpenAILLMProvider(settings)
        name = provider.get_provider_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_is_available_with_key(self, settings: Settings) -> None:
        from src.providers.llm.openai_provider import OpenAILLMProvider
        provider = OpenAILLMProvider(settings)
        assert provider.is_available() is True

    def test_is_available_without_key(self) -> None:
        from src.providers.llm.openai_provider import OpenAILLMProvider
        provider = OpenAILLMProvider(_settings(openai_api_key=""))
        assert provider.is_available() is False

    def test_supports_vision(self, settings: Settings) -> None:
        from src.providers.llm.openai_provider import OpenAILLMProvider
        provider = OpenAILLMProvider(settings)
        assert isinstance(provider.supports_vision(), bool)

    @pytest.mark.asyncio
    async def test_complete_success(self, settings: Settings) -> None:
        from src.providers.llm.openai_provider import OpenAILLMProvider

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="LLM response text"))]
        mock_response.usage = MagicMock(total_tokens=100)

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("src.providers.llm.openai_provider.openai.AsyncOpenAI", return_value=mock_client):
            provider = OpenAILLMProvider(settings)
            result = await provider.complete("system prompt", "user prompt")

        assert result == "LLM response text"

    @pytest.mark.asyncio
    async def test_complete_error(self, settings: Settings) -> None:
        from src.providers.llm.openai_provider import OpenAILLMProvider
        import openai

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=openai.APIError(
                message="Rate limit exceeded",
                request=MagicMock(),
                body=None,
            )
        )

        with patch("src.providers.llm.openai_provider.openai.AsyncOpenAI", return_value=mock_client):
            provider = OpenAILLMProvider(settings)
            with pytest.raises(Exception):
                await provider.complete("system", "user")

    @pytest.mark.asyncio
    async def test_validate_credentials_success(self, settings: Settings) -> None:
        from src.providers.llm.openai_provider import OpenAILLMProvider

        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(return_value=MagicMock())

        with patch("src.providers.llm.openai_provider.openai.AsyncOpenAI", return_value=mock_client):
            provider = OpenAILLMProvider(settings)
            result = await provider.validate_credentials()

        assert result is True

    @pytest.mark.asyncio
    async def test_validate_credentials_failure(self, settings: Settings) -> None:
        from src.providers.llm.openai_provider import OpenAILLMProvider
        import openai

        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(
            side_effect=openai.APIError(
                message="Invalid key",
                request=MagicMock(),
                body=None,
            )
        )

        with patch("src.providers.llm.openai_provider.openai.AsyncOpenAI", return_value=mock_client):
            provider = OpenAILLMProvider(settings)
            result = await provider.validate_credentials()

        assert result is False

    @pytest.mark.asyncio
    async def test_vision_extract(self, settings: Settings) -> None:
        from src.providers.llm.openai_provider import OpenAILLMProvider

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="CARL COX\nTRESOR"))]
        mock_response.usage = MagicMock(total_tokens=200)

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("src.providers.llm.openai_provider.openai.AsyncOpenAI", return_value=mock_client):
            provider = OpenAILLMProvider(_settings(openai_vision_model="gpt-4o"))
            result = await provider.vision_extract(b"fake_image_bytes", "Extract text")

        assert "CARL COX" in result


# ======================================================================
# Anthropic LLM Provider
# ======================================================================


class TestAnthropicLLMProvider:
    @pytest.fixture()
    def settings(self) -> Settings:
        return _settings()

    def test_get_provider_name(self, settings: Settings) -> None:
        from src.providers.llm.anthropic_provider import AnthropicLLMProvider
        provider = AnthropicLLMProvider(settings)
        assert provider.get_provider_name() == "anthropic"

    def test_is_available_with_key(self, settings: Settings) -> None:
        from src.providers.llm.anthropic_provider import AnthropicLLMProvider
        provider = AnthropicLLMProvider(settings)
        assert provider.is_available() is True

    def test_is_available_without_key(self) -> None:
        from src.providers.llm.anthropic_provider import AnthropicLLMProvider
        provider = AnthropicLLMProvider(_settings(anthropic_api_key=""))
        assert provider.is_available() is False

    def test_supports_vision(self, settings: Settings) -> None:
        from src.providers.llm.anthropic_provider import AnthropicLLMProvider
        provider = AnthropicLLMProvider(settings)
        assert provider.supports_vision() is True

    @pytest.mark.asyncio
    async def test_complete_success(self, settings: Settings) -> None:
        from src.providers.llm.anthropic_provider import AnthropicLLMProvider

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "Anthropic response"

        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.usage = MagicMock(input_tokens=50, output_tokens=50)

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("src.providers.llm.anthropic_provider.anthropic.AsyncAnthropic", return_value=mock_client):
            provider = AnthropicLLMProvider(settings)
            result = await provider.complete("system", "user")

        assert result == "Anthropic response"

    @pytest.mark.asyncio
    async def test_complete_error(self, settings: Settings) -> None:
        from src.providers.llm.anthropic_provider import AnthropicLLMProvider

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=Exception("API error"))

        with patch("src.providers.llm.anthropic_provider.anthropic.AsyncAnthropic", return_value=mock_client):
            provider = AnthropicLLMProvider(settings)
            with pytest.raises(Exception):
                await provider.complete("system", "user")

    @pytest.mark.asyncio
    async def test_validate_credentials_success(self, settings: Settings) -> None:
        from src.providers.llm.anthropic_provider import AnthropicLLMProvider

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "ok"
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.usage = MagicMock(input_tokens=5, output_tokens=5)

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("src.providers.llm.anthropic_provider.anthropic.AsyncAnthropic", return_value=mock_client):
            provider = AnthropicLLMProvider(settings)
            result = await provider.validate_credentials()

        assert result is True

    @pytest.mark.asyncio
    async def test_vision_extract(self, settings: Settings) -> None:
        from src.providers.llm.anthropic_provider import AnthropicLLMProvider

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "JEFF MILLS\nTRESOR BERLIN"

        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("src.providers.llm.anthropic_provider.anthropic.AsyncAnthropic", return_value=mock_client):
            provider = AnthropicLLMProvider(settings)
            result = await provider.vision_extract(b"fake_image", "Extract text")

        assert "JEFF MILLS" in result


# ======================================================================
# Ollama LLM Provider
# ======================================================================


class TestOllamaLLMProvider:
    @pytest.fixture()
    def settings(self) -> Settings:
        return _settings()

    def test_get_provider_name(self, settings: Settings) -> None:
        from src.providers.llm.ollama_provider import OllamaLLMProvider
        provider = OllamaLLMProvider(settings)
        assert provider.get_provider_name() == "ollama"

    def test_is_available(self, settings: Settings) -> None:
        from src.providers.llm.ollama_provider import OllamaLLMProvider
        provider = OllamaLLMProvider(settings)
        assert provider.is_available() is True

    @pytest.mark.asyncio
    async def test_complete_success(self, settings: Settings) -> None:
        from src.providers.llm.ollama_provider import OllamaLLMProvider

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Ollama response"))]
        mock_response.usage = MagicMock(total_tokens=80)

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("src.providers.llm.ollama_provider.openai.AsyncOpenAI", return_value=mock_client):
            provider = OllamaLLMProvider(settings)
            result = await provider.complete("system", "user")

        assert result == "Ollama response"

    @pytest.mark.asyncio
    async def test_validate_credentials_success(self, settings: Settings) -> None:
        from src.providers.llm.ollama_provider import OllamaLLMProvider

        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_http_client = AsyncMock()
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)
        mock_http_client.get = AsyncMock(return_value=mock_response)

        with patch("src.providers.llm.ollama_provider.httpx.AsyncClient", return_value=mock_http_client):
            provider = OllamaLLMProvider(settings)
            result = await provider.validate_credentials()

        assert result is True

    @pytest.mark.asyncio
    async def test_validate_credentials_failure(self, settings: Settings) -> None:
        from src.providers.llm.ollama_provider import OllamaLLMProvider
        import httpx

        mock_http_client = AsyncMock()
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)
        mock_http_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

        with patch("src.providers.llm.ollama_provider.httpx.AsyncClient", return_value=mock_http_client):
            provider = OllamaLLMProvider(settings)
            result = await provider.validate_credentials()

        assert result is False

    def test_supports_vision_returns_true(self, settings: Settings) -> None:
        """OllamaLLMProvider always reports vision support (via llava)."""
        from src.providers.llm.ollama_provider import OllamaLLMProvider

        provider = OllamaLLMProvider(settings)
        assert provider.supports_vision() is True

    def test_is_available_with_base_url(self, settings: Settings) -> None:
        """is_available returns True when ollama_base_url is configured."""
        from src.providers.llm.ollama_provider import OllamaLLMProvider

        provider = OllamaLLMProvider(settings)
        assert provider.is_available() is True

    def test_is_available_without_base_url(self) -> None:
        """is_available returns False when ollama_base_url is empty."""
        from src.providers.llm.ollama_provider import OllamaLLMProvider

        provider = OllamaLLMProvider(_settings(ollama_base_url=""))
        assert provider.is_available() is False

    @pytest.mark.asyncio
    async def test_validate_credentials_returns_false_when_unavailable(self) -> None:
        """validate_credentials returns False when is_available is False."""
        from src.providers.llm.ollama_provider import OllamaLLMProvider

        provider = OllamaLLMProvider(_settings(ollama_base_url=""))
        result = await provider.validate_credentials()
        assert result is False

    @pytest.mark.asyncio
    async def test_vision_extract_success(self, settings: Settings) -> None:
        """vision_extract sends an image to the vision model and returns text."""
        from src.providers.llm.ollama_provider import OllamaLLMProvider

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="CARL COX\nTRESOR BERLIN"))]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("src.providers.llm.ollama_provider.openai.AsyncOpenAI", return_value=mock_client):
            provider = OllamaLLMProvider(settings)
            # JPEG magic bytes
            image_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100
            result = await provider.vision_extract(image_bytes, "Extract text from this flier")

        assert "CARL COX" in result
        assert "TRESOR BERLIN" in result
        # Verify the vision model was used
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "llava"

    @pytest.mark.asyncio
    async def test_vision_extract_empty_response_raises(self, settings: Settings) -> None:
        """vision_extract raises LLMError when the model returns empty content."""
        from src.providers.llm.ollama_provider import OllamaLLMProvider
        from src.utils.errors import LLMError

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=None))]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("src.providers.llm.ollama_provider.openai.AsyncOpenAI", return_value=mock_client):
            provider = OllamaLLMProvider(settings)
            with pytest.raises(LLMError, match="empty response"):
                await provider.vision_extract(b"\xff\xd8" + b"\x00" * 50, "Extract text")

    @pytest.mark.asyncio
    async def test_vision_extract_api_error_raises(self, settings: Settings) -> None:
        """vision_extract wraps openai.APIError in LLMError."""
        from src.providers.llm.ollama_provider import OllamaLLMProvider
        from src.utils.errors import LLMError
        import openai

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=openai.APIError(
                message="Model not found",
                request=MagicMock(),
                body=None,
            )
        )

        with patch("src.providers.llm.ollama_provider.openai.AsyncOpenAI", return_value=mock_client):
            provider = OllamaLLMProvider(settings)
            with pytest.raises(LLMError, match="vision API error"):
                await provider.vision_extract(b"\xff\xd8" + b"\x00" * 50, "Extract text")

    @pytest.mark.asyncio
    async def test_complete_empty_response_raises(self, settings: Settings) -> None:
        """complete raises LLMError when the model returns None content."""
        from src.providers.llm.ollama_provider import OllamaLLMProvider
        from src.utils.errors import LLMError

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=None))]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("src.providers.llm.ollama_provider.openai.AsyncOpenAI", return_value=mock_client):
            provider = OllamaLLMProvider(settings)
            with pytest.raises(LLMError, match="empty response"):
                await provider.complete("system", "user")

    @pytest.mark.asyncio
    async def test_complete_api_error_raises(self, settings: Settings) -> None:
        """complete wraps openai.APIError in LLMError."""
        from src.providers.llm.ollama_provider import OllamaLLMProvider
        from src.utils.errors import LLMError
        import openai

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=openai.APIError(
                message="Server error",
                request=MagicMock(),
                body=None,
            )
        )

        with patch("src.providers.llm.ollama_provider.openai.AsyncOpenAI", return_value=mock_client):
            provider = OllamaLLMProvider(settings)
            with pytest.raises(LLMError, match="Ollama API error"):
                await provider.complete("system", "user")

    def test_vision_extract_detects_png_media_type(self, settings: Settings) -> None:
        """_detect_media_type correctly identifies PNG files."""
        from src.providers.llm.ollama_provider import _detect_media_type

        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        assert _detect_media_type(png_header) == "image/png"

    def test_vision_extract_detects_jpeg_media_type(self, settings: Settings) -> None:
        """_detect_media_type correctly identifies JPEG files."""
        from src.providers.llm.ollama_provider import _detect_media_type

        jpeg_header = b"\xff\xd8" + b"\x00" * 100
        assert _detect_media_type(jpeg_header) == "image/jpeg"

    def test_vision_extract_detects_webp_media_type(self, settings: Settings) -> None:
        """_detect_media_type correctly identifies WebP files."""
        from src.providers.llm.ollama_provider import _detect_media_type

        webp_header = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 100
        assert _detect_media_type(webp_header) == "image/webp"

    def test_vision_extract_unknown_defaults_jpeg(self, settings: Settings) -> None:
        """_detect_media_type returns image/jpeg for unknown formats."""
        from src.providers.llm.ollama_provider import _detect_media_type

        unknown_bytes = b"\x00\x01\x02\x03" * 10
        assert _detect_media_type(unknown_bytes) == "image/jpeg"

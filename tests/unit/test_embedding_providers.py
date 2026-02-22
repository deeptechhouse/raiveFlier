"""Unit tests for embedding provider adapters — OpenAI, Nomic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.settings import Settings


def _settings(**overrides) -> Settings:
    defaults = {
        "openai_api_key": "sk-test",
        "openai_base_url": "",
        "openai_embedding_model": "",
        "ollama_base_url": "http://localhost:11434",
    }
    defaults.update(overrides)
    return Settings(**defaults)


# ======================================================================
# OpenAI Embedding Provider
# ======================================================================


class TestOpenAIEmbeddingProvider:
    @pytest.fixture()
    def settings(self) -> Settings:
        return _settings()

    def test_get_provider_name(self, settings: Settings) -> None:
        from src.providers.embedding.openai_embedding_provider import OpenAIEmbeddingProvider
        provider = OpenAIEmbeddingProvider(settings)
        name = provider.get_provider_name()
        assert isinstance(name, str)
        assert "embedding" in name

    def test_is_available_with_key(self, settings: Settings) -> None:
        from src.providers.embedding.openai_embedding_provider import OpenAIEmbeddingProvider
        provider = OpenAIEmbeddingProvider(settings)
        assert provider.is_available() is True

    def test_is_available_without_key(self) -> None:
        from src.providers.embedding.openai_embedding_provider import OpenAIEmbeddingProvider
        provider = OpenAIEmbeddingProvider(_settings(openai_api_key=""))
        assert provider.is_available() is False

    def test_get_dimension(self, settings: Settings) -> None:
        from src.providers.embedding.openai_embedding_provider import OpenAIEmbeddingProvider
        provider = OpenAIEmbeddingProvider(settings)
        dim = provider.get_dimension()
        assert isinstance(dim, int)
        assert dim > 0

    @pytest.mark.asyncio
    async def test_embed_success(self, settings: Settings) -> None:
        from src.providers.embedding.openai_embedding_provider import OpenAIEmbeddingProvider

        dim = 1536
        mock_embedding_1 = MagicMock(embedding=[0.1] * dim)
        mock_embedding_2 = MagicMock(embedding=[0.2] * dim)

        mock_response = MagicMock()
        mock_response.data = [mock_embedding_1, mock_embedding_2]
        mock_response.usage = MagicMock(total_tokens=50)

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with patch(
            "src.providers.embedding.openai_embedding_provider.openai.AsyncOpenAI",
            return_value=mock_client,
        ):
            provider = OpenAIEmbeddingProvider(settings)
            result = await provider.embed(["hello", "world"])

        assert len(result) == 2
        assert len(result[0]) == dim

    @pytest.mark.asyncio
    async def test_embed_single(self, settings: Settings) -> None:
        from src.providers.embedding.openai_embedding_provider import OpenAIEmbeddingProvider

        dim = 1536
        mock_embedding = MagicMock(embedding=[0.5] * dim)
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_response.usage = MagicMock(total_tokens=10)

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with patch(
            "src.providers.embedding.openai_embedding_provider.openai.AsyncOpenAI",
            return_value=mock_client,
        ):
            provider = OpenAIEmbeddingProvider(settings)
            result = await provider.embed_single("hello")

        assert len(result) == dim

    @pytest.mark.asyncio
    async def test_embed_error(self, settings: Settings) -> None:
        from src.providers.embedding.openai_embedding_provider import OpenAIEmbeddingProvider
        import openai

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(
            side_effect=openai.APIError(
                message="Rate limit",
                request=MagicMock(),
                body=None,
            )
        )

        with patch(
            "src.providers.embedding.openai_embedding_provider.openai.AsyncOpenAI",
            return_value=mock_client,
        ):
            provider = OpenAIEmbeddingProvider(settings)
            with pytest.raises(Exception):
                await provider.embed(["test"])


# ======================================================================
# Nomic Embedding Provider
# ======================================================================


class TestNomicEmbeddingProvider:
    @pytest.fixture()
    def settings(self) -> Settings:
        return _settings()

    def test_get_provider_name(self, settings: Settings) -> None:
        from src.providers.embedding.nomic_embedding_provider import NomicEmbeddingProvider
        provider = NomicEmbeddingProvider(settings)
        name = provider.get_provider_name()
        assert isinstance(name, str)
        assert "nomic" in name

    def test_get_dimension(self, settings: Settings) -> None:
        from src.providers.embedding.nomic_embedding_provider import NomicEmbeddingProvider
        provider = NomicEmbeddingProvider(settings)
        assert provider.get_dimension() == 768

    def test_is_available_success(self, settings: Settings) -> None:
        from src.providers.embedding.nomic_embedding_provider import NomicEmbeddingProvider

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("src.providers.embedding.nomic_embedding_provider.httpx.get", return_value=mock_response):
            provider = NomicEmbeddingProvider(settings)
            assert provider.is_available() is True

    def test_is_available_failure(self, settings: Settings) -> None:
        from src.providers.embedding.nomic_embedding_provider import NomicEmbeddingProvider
        import httpx

        with patch(
            "src.providers.embedding.nomic_embedding_provider.httpx.get",
            side_effect=httpx.ConnectError("refused"),
        ):
            provider = NomicEmbeddingProvider(settings)
            assert provider.is_available() is False

    @pytest.mark.asyncio
    async def test_embed_success(self, settings: Settings) -> None:
        from src.providers.embedding.nomic_embedding_provider import NomicEmbeddingProvider

        dim = 768
        mock_embedding = MagicMock(embedding=[0.3] * dim)
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_response.usage = MagicMock(total_tokens=20)

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with patch(
            "src.providers.embedding.nomic_embedding_provider.openai.AsyncOpenAI",
            return_value=mock_client,
        ):
            provider = NomicEmbeddingProvider(settings)
            result = await provider.embed(["test text"])

        assert len(result) == 1
        assert len(result[0]) == dim

    @pytest.mark.asyncio
    async def test_embed_single(self, settings: Settings) -> None:
        from src.providers.embedding.nomic_embedding_provider import NomicEmbeddingProvider

        dim = 768
        mock_embedding = MagicMock(embedding=[0.4] * dim)
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_response.usage = MagicMock(total_tokens=10)

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with patch(
            "src.providers.embedding.nomic_embedding_provider.openai.AsyncOpenAI",
            return_value=mock_client,
        ):
            provider = NomicEmbeddingProvider(settings)
            result = await provider.embed_single("test")

        assert len(result) == dim

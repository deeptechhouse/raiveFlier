"""OpenAI embedding provider adapter.

Wraps the ``openai`` async client to implement :class:`IEmbeddingProvider`
using the ``text-embedding-3-small`` model (1536 dimensions).
"""

from __future__ import annotations

import openai
import structlog

from src.config.settings import Settings
from src.interfaces.embedding_provider import IEmbeddingProvider
from src.utils.errors import RAGError

logger = structlog.get_logger(logger_name=__name__)

_OPENAI_BATCH_LIMIT = 2048


class OpenAIEmbeddingProvider(IEmbeddingProvider):
    """Embedding provider backed by the OpenAI embeddings API.

    Uses ``text-embedding-3-small`` which produces 1536-dimensional vectors.
    Handles automatic batching for inputs exceeding the per-call limit of 2048
    texts.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._api_key = settings.openai_api_key
        self._client = openai.AsyncOpenAI(api_key=self._api_key)
        self._model = "text-embedding-3-small"
        self._dimension = 1536

    # ------------------------------------------------------------------
    # IEmbeddingProvider implementation
    # ------------------------------------------------------------------

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a batch of texts.

        Automatically splits into batches of 2048 if the input exceeds the
        OpenAI per-call limit.
        """
        if not texts:
            return []

        try:
            all_embeddings: list[list[float]] = []
            for start in range(0, len(texts), _OPENAI_BATCH_LIMIT):
                batch = texts[start : start + _OPENAI_BATCH_LIMIT]
                response = await self._client.embeddings.create(
                    input=batch,
                    model=self._model,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                logger.info(
                    "openai_embedding_batch",
                    model=self._model,
                    batch_size=len(batch),
                    tokens=response.usage.total_tokens if response.usage else None,
                )
            return all_embeddings
        except openai.APIError as exc:
            raise RAGError(
                message=f"OpenAI embedding API error: {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

    async def embed_single(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text string."""
        result = await self.embed([text])
        return result[0]

    def get_dimension(self) -> int:
        """Return 1536 (text-embedding-3-small dimension)."""
        return self._dimension

    def get_provider_name(self) -> str:
        return "openai_embedding"

    def is_available(self) -> bool:
        """Return ``True`` if an OpenAI API key is configured."""
        return bool(self._api_key)

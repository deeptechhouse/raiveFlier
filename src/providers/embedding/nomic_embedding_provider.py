"""Nomic embedding provider adapter (local/free via Ollama).

Wraps the Ollama OpenAI-compatible endpoint to implement
:class:`IEmbeddingProvider` using ``nomic-embed-text`` (768 dimensions).
Fully free — runs locally with no API key required.
"""

from __future__ import annotations

import httpx
import openai
import structlog

from src.config.settings import Settings
from src.interfaces.embedding_provider import IEmbeddingProvider
from src.utils.errors import RAGError

logger = structlog.get_logger(logger_name=__name__)

_OLLAMA_BATCH_LIMIT = 512


class NomicEmbeddingProvider(IEmbeddingProvider):
    """Embedding provider backed by ``nomic-embed-text`` served via Ollama.

    Communicates through the OpenAI-compatible ``/v1`` endpoint that Ollama
    exposes.  Produces 768-dimensional vectors.  Handles automatic batching
    for inputs exceeding 512 texts per call.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base_url = settings.ollama_base_url
        self._client = openai.AsyncOpenAI(
            base_url=f"{self._base_url}/v1",
            api_key="ollama",  # Ollama doesn't require a real key
        )
        self._model = "nomic-embed-text"
        self._dimension = 768

    # ------------------------------------------------------------------
    # IEmbeddingProvider implementation
    # ------------------------------------------------------------------

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a batch of texts.

        Automatically splits into batches of 512 for the Ollama backend.
        """
        if not texts:
            return []

        try:
            all_embeddings: list[list[float]] = []
            for start in range(0, len(texts), _OLLAMA_BATCH_LIMIT):
                batch = texts[start : start + _OLLAMA_BATCH_LIMIT]
                response = await self._client.embeddings.create(
                    input=batch,
                    model=self._model,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                logger.info(
                    "nomic_embedding_batch",
                    model=self._model,
                    batch_size=len(batch),
                )
            return all_embeddings
        except openai.APIError as exc:
            raise RAGError(
                message=f"Nomic/Ollama embedding API error: {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

    async def embed_single(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text string."""
        result = await self.embed([text])
        return result[0]

    def get_dimension(self) -> int:
        """Return 768 (nomic-embed-text dimension)."""
        return self._dimension

    def get_provider_name(self) -> str:
        return "nomic_embedding"

    def is_available(self) -> bool:
        """Return ``True`` if the Ollama server is reachable."""
        if not self._base_url:
            return False
        try:
            response = httpx.get(f"{self._base_url}/api/tags", timeout=3.0)
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

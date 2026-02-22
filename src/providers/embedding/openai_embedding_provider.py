"""OpenAI-compatible embedding provider adapter.

Wraps the ``openai`` async client to implement :class:`IEmbeddingProvider`.
Supports both real OpenAI and OpenAI-compatible providers (TogetherAI,
Anyscale, Fireworks) via custom ``base_url`` and model name settings.
"""

from __future__ import annotations

import openai
import structlog

from src.config.settings import Settings
from src.interfaces.embedding_provider import IEmbeddingProvider
from src.utils.errors import RAGError

logger = structlog.get_logger(logger_name=__name__)

_OPENAI_BATCH_LIMIT = 2048

# Known embedding model dimensions.
_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "togethercomputer/m2-bert-80M-8k-retrieval": 768,
    "WhereIsAI/UAE-Large-V1": 1024,
    "intfloat/multilingual-e5-large-instruct": 1024,
}

# Maximum input token limits for models with tight context windows.
# Models not listed here are assumed to handle 8192+ tokens.
_MODEL_MAX_TOKENS: dict[str, int] = {
    "BAAI/bge-base-en-v1.5": 512,
    "BAAI/bge-large-en-v1.5": 512,
    "intfloat/multilingual-e5-large-instruct": 512,
    "togethercomputer/m2-bert-80M-8k-retrieval": 8192,
}


class OpenAIEmbeddingProvider(IEmbeddingProvider):
    """Embedding provider backed by an OpenAI-compatible embeddings API.

    Uses ``text-embedding-3-small`` (1536 dims) by default.  When
    ``openai_base_url`` is configured (e.g. TogetherAI), the client
    points at that URL and uses ``openai_embedding_model`` if set.
    Handles automatic batching for inputs exceeding the per-call limit.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._api_key = settings.openai_api_key

        # Build client kwargs — add base_url only when configured.
        client_kwargs: dict = {"api_key": self._api_key}
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url

        self._client = openai.AsyncOpenAI(**client_kwargs)
        self._model = settings.openai_embedding_model or "text-embedding-3-small"
        self._dimension = _MODEL_DIMENSIONS.get(self._model, 768)
        self._max_tokens = _MODEL_MAX_TOKENS.get(self._model, 0)
        self._provider_label = (
            "openai-compatible_embedding" if settings.openai_base_url else "openai_embedding"
        )

    # ------------------------------------------------------------------
    # IEmbeddingProvider implementation
    # ------------------------------------------------------------------

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a batch of texts.

        Automatically splits into batches of 2048 if the input exceeds the
        per-call limit.  Truncates texts that exceed the model's max token
        limit (e.g. 512 tokens for BGE/E5 models on TogetherAI).
        """
        if not texts:
            return []

        # Truncate texts that exceed model token limits.
        if self._max_tokens > 0:
            texts = [self._truncate_to_token_limit(t) for t in texts]

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
                    provider=self._provider_label,
                    batch_size=len(batch),
                    tokens=response.usage.total_tokens if response.usage else None,
                )
            return all_embeddings
        except openai.APIError as exc:
            raise RAGError(
                message=f"{self._provider_label} API error: {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

    async def embed_single(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text string."""
        result = await self.embed([text])
        return result[0]

    def get_dimension(self) -> int:
        return self._dimension

    def get_provider_name(self) -> str:
        return self._provider_label

    def is_available(self) -> bool:
        """Return ``True`` if an API key is configured."""
        return bool(self._api_key)

    def _truncate_to_token_limit(self, text: str) -> str:
        """Truncate text to fit within the model's max token limit.

        Uses a conservative character-based estimate (~3.5 chars per token)
        to safely stay under the limit without requiring a tokenizer.
        """
        max_tokens = self._max_tokens
        if max_tokens <= 0:
            return text

        # Very conservative: ~2.5 characters per token ensures we stay
        # well under the limit even for text heavy with short words, names,
        # and punctuation that tokenize to multiple tokens.
        max_chars = int(max_tokens * 2.5)
        if len(text) <= max_chars:
            return text

        # Truncate at the last word boundary before max_chars.
        truncated = text[:max_chars].rsplit(" ", 1)[0]
        logger.debug(
            "truncating_embedding_input",
            original_chars=len(text),
            truncated_chars=len(truncated),
            model=self._model,
        )
        return truncated

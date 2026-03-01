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
        # Load a fast tokenizer for accurate token-level truncation.
        # Models like intfloat/multilingual-e5-large-instruct have tight
        # 512-token limits; char-based estimates are unreliable for dense
        # text (Discogs data with names, abbreviations, punctuation).
        self._tokenizer = self._load_tokenizer()

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

    @staticmethod
    def _load_tokenizer():  # noqa: ANN205 – optional dep
        """Load a fast HuggingFace tokenizer for accurate token counting.

        Uses ``xlm-roberta-large`` — the same tokenizer family as the
        ``intfloat/multilingual-e5-large-instruct`` and BGE embedding
        models.  Falls back to ``bert-base-uncased`` if xlm-roberta is
        unavailable (requires an initial download).
        """
        try:
            from tokenizers import Tokenizer  # type: ignore[import-untyped]

            # xlm-roberta matches the E5/BGE model family's tokenizer.
            for model_id in ("xlm-roberta-large", "bert-base-uncased"):
                try:
                    return Tokenizer.from_pretrained(model_id)
                except Exception:  # noqa: BLE001
                    continue
            return None
        except Exception:  # noqa: BLE001
            logger.info(
                "embedding_tokenizer_unavailable",
                msg="Falling back to char-based truncation for token limits.",
            )
            return None

    def _truncate_to_token_limit(self, text: str) -> str:
        """Truncate text to fit within the model's max token limit.

        Two strategies, in priority order:
        1. Tokenizer-based (accurate): encode with xlm-roberta tokenizer,
           truncate at 95% of max_tokens (5% margin for edge cases), decode.
        2. Char-based fallback: use 1.5 chars/token estimate (very conservative,
           handles dense text with short words and punctuation).
        """
        max_tokens = self._max_tokens
        if max_tokens <= 0:
            return text

        # Strategy 1: accurate tokenizer-based truncation.
        if self._tokenizer is not None:
            ids = self._tokenizer.encode(text).ids
            # 5% safety margin — minimal because xlm-roberta matches
            # the actual E5/BGE model tokenizer closely.
            safe_limit = int(max_tokens * 0.95)
            if len(ids) <= safe_limit:
                return text
            truncated = self._tokenizer.decode(ids[:safe_limit])
            logger.debug(
                "truncating_embedding_input_tokenizer",
                original_tokens=len(ids),
                truncated_tokens=safe_limit,
                model=self._model,
            )
            return truncated

        # Strategy 2: very conservative char-based fallback.
        # 1.5 chars/token is aggressive but safe for dense text (names,
        # abbreviations, punctuation that tokenize at ~2-3 chars/token).
        max_chars = int(max_tokens * 1.5)
        if len(text) <= max_chars:
            return text

        # Truncate at the last word boundary before max_chars.
        truncated = text[:max_chars].rsplit(" ", 1)[0]
        logger.debug(
            "truncating_embedding_input_chars",
            original_chars=len(text),
            truncated_chars=len(truncated),
            model=self._model,
        )
        return truncated

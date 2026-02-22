"""Local sentence-transformers embedding provider adapter.

Wraps the ``sentence-transformers`` library to implement
:class:`IEmbeddingProvider` using any HuggingFace embedding model locally.
Fully free — runs on CPU/GPU with no API key required.

Default model: ``intfloat/multilingual-e5-large-instruct`` (1024 dimensions),
matching the TogetherAI-hosted model used by
:class:`~src.providers.embedding.openai_embedding_provider.OpenAIEmbeddingProvider`
when configured with ``OPENAI_BASE_URL=https://api.together.xyz/v1``.
"""

from __future__ import annotations

import structlog

from src.interfaces.embedding_provider import IEmbeddingProvider
from src.utils.errors import RAGError

logger = structlog.get_logger(logger_name=__name__)

# Known model dimensions.
_MODEL_DIMENSIONS: dict[str, int] = {
    "intfloat/multilingual-e5-large-instruct": 1024,
    "intfloat/e5-large-v2": 1024,
    "intfloat/e5-base-v2": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
}

_DEFAULT_MODEL = "intfloat/multilingual-e5-large-instruct"
_BATCH_LIMIT = 64  # Conservative batch size for CPU inference


class SentenceTransformerEmbeddingProvider(IEmbeddingProvider):
    """Embedding provider backed by a local sentence-transformers model.

    Loads the model into memory on first use (lazy initialization).
    Produces the same vectors as the API-hosted version of the same model,
    ensuring compatibility with existing ChromaDB collections.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or _DEFAULT_MODEL
        self._dimension = _MODEL_DIMENSIONS.get(self._model_name, 1024)
        self._model = None  # Lazy-loaded

    def _load_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(
                "loading_sentence_transformer",
                model=self._model_name,
                msg="Loading model (first use may download ~1.2GB)...",
            )
            self._model = SentenceTransformer(self._model_name)
            logger.info(
                "sentence_transformer_loaded",
                model=self._model_name,
                dimension=self._dimension,
            )
        except Exception as exc:
            raise RAGError(
                message=f"Failed to load sentence-transformers model '{self._model_name}': {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a batch of texts.

        Automatically splits into batches for memory-safe CPU inference.
        """
        if not texts:
            return []

        self._load_model()

        try:
            all_embeddings: list[list[float]] = []
            for start in range(0, len(texts), _BATCH_LIMIT):
                batch = texts[start : start + _BATCH_LIMIT]
                vectors = self._model.encode(
                    batch,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                all_embeddings.extend(vectors.tolist())
                logger.info(
                    "sentence_transformer_embedding_batch",
                    model=self._model_name,
                    batch_size=len(batch),
                )
            return all_embeddings
        except Exception as exc:
            raise RAGError(
                message=f"Sentence-transformers embedding error: {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

    async def embed_single(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text string."""
        result = await self.embed([text])
        return result[0]

    def get_dimension(self) -> int:
        return self._dimension

    def get_provider_name(self) -> str:
        return f"sentence_transformer_{self._model_name.split('/')[-1]}"

    def is_available(self) -> bool:
        """Return ``True`` if sentence-transformers is installed."""
        try:
            import sentence_transformers  # noqa: F401
            return True
        except ImportError:
            return False

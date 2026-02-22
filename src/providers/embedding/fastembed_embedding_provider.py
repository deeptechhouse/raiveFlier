"""Local ONNX-based embedding provider using fastembed.

Wraps the ``fastembed`` library to implement :class:`IEmbeddingProvider`
using ONNX Runtime — **no PyTorch dependency required**.  Fully free,
runs on CPU with minimal RAM footprint.

Default model: ``intfloat/multilingual-e5-large`` (1024 dimensions),
producing vectors compatible with the corpus built via the TogetherAI API
using ``intfloat/multilingual-e5-large-instruct``.
"""

from __future__ import annotations

import structlog

from src.interfaces.embedding_provider import IEmbeddingProvider
from src.utils.errors import RAGError

logger = structlog.get_logger(logger_name=__name__)

# Known model dimensions for fastembed-supported models.
_MODEL_DIMENSIONS: dict[str, int] = {
    "intfloat/multilingual-e5-large": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}

_DEFAULT_MODEL = "intfloat/multilingual-e5-large"
_BATCH_LIMIT = 64


class FastEmbedEmbeddingProvider(IEmbeddingProvider):
    """Embedding provider backed by fastembed (ONNX Runtime).

    Loads the ONNX model on first use (lazy initialization).
    Downloads model weights (~600MB) on first run, then caches locally.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or _DEFAULT_MODEL
        self._dimension = _MODEL_DIMENSIONS.get(self._model_name, 1024)
        self._model = None  # Lazy-loaded

    def _load_model(self) -> None:
        """Lazy-load the fastembed model."""
        if self._model is not None:
            return
        try:
            from fastembed import TextEmbedding

            logger.info(
                "loading_fastembed_model",
                model=self._model_name,
                msg="Loading ONNX model (first use may download ~600MB)...",
            )
            self._model = TextEmbedding(model_name=self._model_name)
            logger.info(
                "fastembed_model_loaded",
                model=self._model_name,
                dimension=self._dimension,
            )
        except Exception as exc:
            raise RAGError(
                message=f"Failed to load fastembed model '{self._model_name}': {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a batch of texts."""
        if not texts:
            return []

        self._load_model()

        try:
            all_embeddings: list[list[float]] = []
            for start in range(0, len(texts), _BATCH_LIMIT):
                batch = texts[start : start + _BATCH_LIMIT]
                # fastembed returns a generator of numpy arrays
                vectors = list(self._model.embed(batch))
                all_embeddings.extend([v.tolist() for v in vectors])
            return all_embeddings
        except Exception as exc:
            raise RAGError(
                message=f"Fastembed embedding error: {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

    async def embed_single(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text string."""
        result = await self.embed([text])
        return result[0]

    def get_dimension(self) -> int:
        return self._dimension

    def get_provider_name(self) -> str:
        return f"fastembed_{self._model_name.split('/')[-1]}"

    def is_available(self) -> bool:
        """Return ``True`` if fastembed is installed."""
        try:
            import fastembed  # noqa: F401

            return True
        except ImportError:
            return False

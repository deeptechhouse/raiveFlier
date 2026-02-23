"""Abstract base class for text-embedding service providers.

Defines the contract for generating embedding vectors from text.
Implementations may wrap OpenAI ``text-embedding-3-small``, Nomic
``nomic-embed-text`` (local via Ollama), Sentence Transformers, or any
other embedding backend.  The adapter pattern (CLAUDE.md Section 6)
ensures embedding providers are interchangeable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


# Concrete implementations:
#   FastEmbedEmbeddingProvider  — lightweight ONNX (no PyTorch), default for Docker
#   OpenAIEmbeddingProvider     — text-embedding-3-small (requires API key)
#   SentenceTransformerEmbeddingProvider — all-MiniLM-L6-v2 (local, needs PyTorch)
#   NomicEmbeddingProvider      — nomic-embed-text via Ollama (local)
# Located in: src/providers/embedding/
class IEmbeddingProvider(ABC):
    """Contract for text-embedding services used by the RAG pipeline.

    Embeddings are consumed by
    :class:`~src.interfaces.vector_store_provider.IVectorStoreProvider` for
    indexing and query-time similarity search.
    """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a batch of texts.

        Parameters
        ----------
        texts:
            One or more text strings to embed.  Implementations should
            handle batching internally if the underlying API has a per-call
            limit.

        Returns
        -------
        list[list[float]]
            Embedding vectors corresponding positionally to *texts*.  Each
            inner list has length equal to :meth:`get_dimension`.

        Raises
        ------
        src.core.errors.RAGError
            If the embedding API call fails.
        """

    @abstractmethod
    async def embed_single(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text string.

        This is a convenience wrapper around :meth:`embed` for the common
        single-text case (e.g. embedding a search query).

        Parameters
        ----------
        text:
            The text string to embed.

        Returns
        -------
        list[float]
            The embedding vector with length equal to :meth:`get_dimension`.
        """

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the dimensionality of the embedding vectors.

        This value must remain constant for the lifetime of the provider
        instance and must match the dimension configured in the vector store.

        Example values: ``1536`` (OpenAI ``text-embedding-3-small``),
        ``768`` (Nomic ``nomic-embed-text``).
        """

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return a human-readable identifier for this embedding provider.

        Example return values: ``"openai-text-embedding-3-small"``,
        ``"nomic-embed-text"``.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if the provider is configured and reachable.

        Implementations should verify that credentials (if any) are present
        and the model is accessible without generating an actual embedding.
        """

"""Embedding provider implementations.

Embeddings convert text into numeric vectors that capture semantic meaning.
These vectors are stored in ChromaDB and used for similarity search (RAG).

Four implementations of IEmbeddingProvider (listed in typical priority order):
    1. FastEmbedEmbeddingProvider — ONNX-based, no PyTorch needed (~50MB).
       Default for Docker production. Uses all-MiniLM-L6-v2 (384 dims).
    2. OpenAIEmbeddingProvider    — text-embedding-3-small (1536 dims).
       High quality but requires API key and incurs cost per token.
    3. SentenceTransformerEmbeddingProvider — PyTorch-based (~500MB).
       Best quality for local use, but too heavy for Render Starter plan.
    4. NomicEmbeddingProvider      — nomic-embed-text via Ollama (768 dims).
       Free and local, but requires running Ollama server.

Note: Only providers that are importable are re-exported here. FastEmbed and
SentenceTransformer providers are imported directly where needed to avoid
import errors when their dependencies aren't installed.
"""

from src.providers.embedding.nomic_embedding_provider import NomicEmbeddingProvider
from src.providers.embedding.openai_embedding_provider import OpenAIEmbeddingProvider

__all__ = ["OpenAIEmbeddingProvider", "NomicEmbeddingProvider"]

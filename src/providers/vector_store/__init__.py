"""Vector store provider implementations.

ChromaDB is the sole vector store implementation. It stores document chunk
embeddings on disk (persistent) and supports cosine-similarity search with
metadata filtering. Data persists at CHROMADB_PERSIST_DIR (default: /data/chromadb).

To swap ChromaDB for another vector database (Qdrant, Pinecone, Weaviate),
create a new class implementing IVectorStoreProvider and register it in main.py.
"""

from src.providers.vector_store.chromadb_provider import ChromaDBProvider

__all__ = ["ChromaDBProvider"]

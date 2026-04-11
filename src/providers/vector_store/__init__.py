"""Vector store provider implementations.

ChromaDB is the primary vector store for semantic search.  BM25Provider
adds keyword search via SQLite FTS5 for hybrid retrieval.  Data persists
at CHROMADB_PERSIST_DIR (default: /data/chromadb) and the BM25 SQLite DB.

To swap ChromaDB for another vector database (Qdrant, Pinecone, Weaviate),
create a new class implementing IVectorStoreProvider and register it in main.py.

ChromaDB import is wrapped in a try/except because its transitive dependency
chain (opentelemetry) can fail in some environments.  BM25Provider has no
such dependency and is always available.
"""

from src.providers.vector_store.bm25_provider import BM25Provider

try:
    from src.providers.vector_store.chromadb_provider import ChromaDBProvider
except ImportError:
    ChromaDBProvider = None  # type: ignore[assignment, misc]

__all__ = ["BM25Provider", "ChromaDBProvider"]

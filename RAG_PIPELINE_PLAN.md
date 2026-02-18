# raiveFlier — RAG Pipeline Plan

> Current architecture uses no RAG. This document describes the existing approach, its limitations, and a concrete plan for adding RAG to substantially improve research depth and citation quality.

---

## Current Architecture: Search-Then-Synthesize

The app uses a **direct retrieval + context stuffing** approach:

```
Flier Image
  → OCR extraction
  → Entity identification
  → Per-entity API calls (Discogs, MusicBrainz, DuckDuckGo, article scraping)
  → ALL results compiled into a single text blob
  → Entire blob injected into one LLM prompt
  → LLM generates interconnection analysis
```

Each research step is **keyword-based and ephemeral** — results are fetched live, used once, and discarded. There is no vector store, no embedding-based retrieval, no persistent knowledge base. The LLM in Phase 4 receives a raw concatenation of everything found and is asked to synthesize it in a single pass.

---

## Where This Breaks Down

### 1. Context window overflow

A flier with 6 artists, each with 40 releases and 15 articles, produces a context blob that can easily exceed 100K tokens. The current design has no mechanism to select only the most relevant information — it sends everything or truncates arbitrarily.

### 2. No accumulated knowledge

Every flier analysis starts from zero. If you analyze 50 fliers from the same scene/era, the app learns nothing from the previous 49. The same Discogs lookups and web searches run again for the same artists.

### 3. Shallow underground coverage

The app can only find what DuckDuckGo and Discogs surface in real-time. The underground music history that matters most — passages from books like *Energy Flash*, *Last Night a DJ Saved My Life*, *Join the Future*; archived zine articles; oral histories; scene documentation — isn't accessible through web search APIs. It's exactly the kind of material that needs to be pre-indexed.

### 4. Citation quality

When the LLM synthesizes from a compiled text blob, it's one step removed from the actual sources. It can paraphrase what was scraped but can't point to a specific paragraph on a specific page of a specific book.

---

## Proposed RAG Architecture

```
                    ┌─────────────────────────────┐
                    │   VECTOR KNOWLEDGE BASE      │
                    │                               │
                    │  Chunk Store (embeddings)      │
                    │  ├── Book passages             │
                    │  ├── Archived articles          │
                    │  ├── Historical flier metadata   │
                    │  ├── Interview transcripts       │
                    │  ├── Scene documentation         │
                    │  └── Prior analysis results      │
                    │                               │
                    │  Metadata: source, date, tier, │
                    │  entity tags, geographic tags   │
                    └──────────┬────────────────────┘
                               │
Flier Image                    │ semantic retrieval
  → OCR + Entity Extraction    │
  → Live API calls (Discogs,   │
    MusicBrainz, web search)   │
  → MERGE live results ────────┤
                               ▼
                    ┌─────────────────────────────┐
                    │   AUGMENTED LLM CONTEXT       │
                    │                               │
                    │  Top-k relevant chunks from    │
                    │  vector store + live API data  │
                    │  + source metadata per chunk   │
                    └──────────┬────────────────────┘
                               │
                               ▼
                    LLM Synthesis (with grounded citations)
```

---

## Specific Improvements

### 1. Pre-indexed underground music corpus

This is the highest-impact addition. Build a vector store from:

| Source Type | Examples | Ingestion Method |
|---|---|---|
| Published books | *Energy Flash*, *Last Night a DJ Saved My Life*, *Join the Future*, *Bass, Mids, Tops* | Manual OCR/digitization, chunk by section |
| Flier archives | 19hz.info, personal collections, institutional archives | Scrape + metadata extraction |
| Magazine archives | DJ Mag, Mixmag, The Face back issues | Scrape or digitize, chunk by article |
| Interview transcripts | Resident Advisor exchanges, Red Bull Music Academy lectures | Scrape, chunk by Q&A block |
| Scene documentation | Ishkur's Guide, local scene wikis, oral history projects | Scrape, chunk by entry |

Each chunk carries metadata: `source_title`, `author`, `publication_date`, `citation_tier`, `geographic_tags`, `entity_tags` (artist/venue/label names mentioned). This metadata enables **filtered retrieval** — when researching Carl Cox, retrieve only chunks that mention Carl Cox or his known labels/venues.

### 2. Semantic retrieval replaces context stuffing

Instead of dumping all research into one prompt:

```python
# CURRENT (brittle)
context = compile_all_research_to_text(results)  # Could be 200K tokens
llm.complete(system_prompt + context)  # Truncation or failure

# WITH RAG (targeted)
for entity in entities:
    query = f"{entity.name} {entity.type} underground electronic music history"
    relevant_chunks = vector_store.query(
        query=query,
        filter={"date": {"$lte": flier_date}},
        top_k=20
    )
    entity_context = format_chunks_with_citations(relevant_chunks)

# Merge live API data + retrieved chunks, within token budget
augmented_context = merge_and_rank(live_results, retrieved_chunks, max_tokens=30000)
llm.complete(system_prompt + augmented_context)
```

This keeps the LLM context focused and within limits regardless of how many entities are on the flier.

### 3. Grounded citations with page-level precision

Each retrieved chunk carries its source metadata. When the LLM references a chunk, the citation points to a specific source, page, and passage — not a vague "according to web research."

```
"Cox's residency at Shelley's in Stoke-on-Trent established him as..."
  → Citation: Energy Flash, Simon Reynolds, p.142-143, Faber & Faber, 1998 [Tier 1]
```

This is a direct improvement over the current approach where the LLM generates citations from scraped web content with no passage-level granularity.

### 4. Cumulative knowledge from prior analyses

Every completed flier analysis feeds back into the vector store:

- Research results become indexed chunks
- Verified interconnections become retrievable facts
- The 50th flier from the same scene benefits from everything learned in the first 49

This directly addresses the app's core problem: underground music history is obscured and scattered. Each analysis makes the system smarter.

### 5. Hybrid retrieval (structured + semantic)

Combine the existing structured API calls with semantic search:

```
Artist research:
  1. Discogs API → structured discography data (current)
  2. MusicBrainz API → cross-reference (current)
  3. Vector store → book passages, interviews, scene docs mentioning this artist (NEW)
  4. Web search → recent articles (current)
  5. MERGE all with source priority ranking
```

The structured data (Discogs releases, catalog numbers) stays API-driven. The contextual, narrative, historical data comes from the vector store. Each source type plays to its strength.

---

## Implementation Considerations

| Component | Recommendation | Why |
|---|---|---|
| **Embedding model** | `text-embedding-3-small` (OpenAI) or `nomic-embed-text` (local/free) | Wrap behind an adapter interface like all other providers |
| **Vector store** | ChromaDB (local, free, Python-native) or Qdrant (self-hosted) | No vendor lock-in, runs locally for development |
| **Chunk strategy** | 500-token chunks with 100-token overlap, preserve paragraph boundaries | Underground music writing has long contextual passages |
| **Ingestion pipeline** | Separate CLI tool to process and index source material | Decouples corpus building from the analysis pipeline |
| **Retrieval filter** | Filter by date (before flier date), entity tags, geographic region | Prevents anachronistic results |

---

## New Interfaces and Modules

The adapter pattern already in the architecture makes this a clean addition without restructuring existing code.

### New Interface: `IVectorStoreProvider`

```python
# src/interfaces/vector_store_provider.py

class IVectorStoreProvider(ABC):
    """Interface for vector store operations (embedding storage and retrieval)."""

    @abstractmethod
    async def query(
        self,
        query_text: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Semantic search against the vector store."""
        pass

    @abstractmethod
    async def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Add document chunks to the vector store. Returns count added."""
        pass

    @abstractmethod
    async def delete_by_source(self, source_id: str) -> int:
        """Delete all chunks from a specific source."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass
```

### New Interface: `IEmbeddingProvider`

```python
# src/interfaces/embedding_provider.py

class IEmbeddingProvider(ABC):
    """Interface for text embedding generation."""

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return embedding dimension size."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass
```

### New Models

```python
# src/models/rag.py

class DocumentChunk(BaseModel):
    """A chunk of text from a source document, ready for embedding."""
    chunk_id: str
    text: str
    source_id: str
    source_title: str
    source_type: str        # "book", "article", "interview", "flier", "analysis"
    author: Optional[str]
    publication_date: Optional[date]
    citation_tier: int      # 1-6
    page_number: Optional[str]
    entity_tags: List[str]  # Artist/venue/label names mentioned
    geographic_tags: List[str]  # Cities, countries, regions
    genre_tags: List[str]

class RetrievedChunk(BaseModel):
    """A chunk returned from vector store query with similarity score."""
    chunk: DocumentChunk
    similarity_score: float
    formatted_citation: str
```

### New Providers

```
src/providers/vector_store/
  ├── chromadb_provider.py      # ChromaDB (local, free, Python-native)
  ├── qdrant_provider.py        # Qdrant (self-hosted, production-grade)
  └── __init__.py

src/providers/embedding/
  ├── openai_embedding_provider.py    # text-embedding-3-small
  ├── nomic_embedding_provider.py     # nomic-embed-text (local/free via Ollama)
  └── __init__.py
```

### New Service: Ingestion Pipeline

```
src/services/ingestion/
  ├── __init__.py
  ├── ingestion_service.py      # Orchestrates document ingestion
  ├── chunker.py                # Text chunking with overlap and boundary preservation
  ├── metadata_extractor.py     # Extract entity/geographic/genre tags from chunks
  └── source_processors/
      ├── book_processor.py     # Process digitized book text
      ├── article_processor.py  # Process web articles
      ├── flier_processor.py    # Process flier metadata from prior analyses
      └── __init__.py
```

### Modified Service: RAG-Augmented Research

```python
# Modified src/services/artist_researcher.py (additions)

class ArtistResearcher:
    def __init__(
        self,
        music_dbs: List[IMusicDatabaseProvider],
        web_search: IWebSearchProvider,
        article_scraper: IArticleProvider,
        llm: ILLMProvider,
        text_normalizer: TextNormalizer,
        vector_store: Optional[IVectorStoreProvider] = None,  # NEW
        cache: Optional[ICacheProvider] = None,
    ):
        self._vector_store = vector_store

    async def research(self, artist_name: str, before_date: Optional[date] = None) -> ResearchResult:
        # ... existing steps 1-4 ...

        # NEW Step 3.5 — VECTOR STORE RETRIEVAL
        if self._vector_store and self._vector_store.is_available():
            chunks = await self._vector_store.query(
                query_text=f"{artist_name} DJ electronic music history",
                top_k=20,
                filters={
                    "date": {"$lte": before_date.isoformat()} if before_date else None,
                    "entity_tags": {"$contains": artist_name},
                },
            )
            # Merge retrieved chunks into research results
            # Each chunk becomes an ArticleReference with precise citation
```

### New Service: RAG-Augmented Interconnection Analysis

```python
# Modified src/services/interconnection_service.py (additions)

class InterconnectionService:
    def __init__(
        self,
        llm: ILLMProvider,
        citation_service: CitationService,
        vector_store: Optional[IVectorStoreProvider] = None,  # NEW
    ):
        self._vector_store = vector_store

    async def analyze(self, research_results, entities) -> InterconnectionMap:
        # NEW: Before LLM synthesis, retrieve cross-entity context
        if self._vector_store:
            entity_names = [e.text for e in entities.artists] + [entities.venue.text if entities.venue else ""]
            cross_query = " ".join(entity_names) + " connection relationship scene"
            cross_chunks = await self._vector_store.query(
                query_text=cross_query,
                top_k=30,
            )
            # These chunks specifically discuss relationships between entities
            # Much more likely to surface book passages about shared scenes

        # Merge into augmented context (within token budget)
        augmented_context = self._merge_and_rank(
            live_results=research_results,
            retrieved_chunks=cross_chunks,
            max_tokens=30000,
        )
        # ... existing LLM synthesis with augmented_context ...
```

### New CLI: Ingestion Tool

```python
# src/cli/ingest.py
# Standalone CLI for building the vector store corpus

"""
Usage:
  python -m src.cli.ingest book --file /path/to/book.txt --title "Energy Flash" --author "Simon Reynolds" --year 1998
  python -m src.cli.ingest article --url https://example.com/article
  python -m src.cli.ingest directory --path /path/to/articles/ --type article
  python -m src.cli.ingest analysis --session-id abc123  (feed back a completed analysis)
  python -m src.cli.ingest stats  (show corpus statistics)
"""
```

---

## Revised Pipeline with RAG

```
Flier Image
  → Phase 1: OCR + Entity Extraction (unchanged)
  → Phase 3: User Confirmation Gate (unchanged)
  → Phase 2: Per-Entity Research
      ├── Discogs API (structured data)          ← existing
      ├── MusicBrainz API (cross-reference)      ← existing
      ├── Vector Store query (book passages,      ← NEW
      │   interviews, scene docs, prior analyses)
      ├── Web Search (recent articles)            ← existing
      └── MERGE with source priority ranking
  → Phase 4: Interconnection Analysis
      ├── Cross-entity vector retrieval            ← NEW
      ├── Augmented context (within token budget)  ← NEW
      └── LLM synthesis with grounded citations
  → Phase 5: Cited Output
      └── Feed completed analysis back into        ← NEW
          vector store for future queries
```

---

## Additional Dependencies

```
# Added to requirements.txt
chromadb>=0.4.22,<1.0           # Vector store (local, free)
# OR qdrant-client>=1.7.0,<2.0  # Alternative vector store

tokenizers>=0.15.0,<1.0         # Token counting for budget management
```

---

## Impact Summary

| Metric | Without RAG | With RAG |
|---|---|---|
| **Context quality** | Whatever web search finds in real-time | Pre-indexed books, archives, prior analyses |
| **Citation precision** | URL-level ("this article says...") | Page-level ("Energy Flash, p.142") |
| **Context window usage** | Unbounded blob, truncation risk | Token-budgeted, relevance-ranked |
| **Knowledge accumulation** | None — every analysis starts from zero | Cumulative — each analysis enriches the corpus |
| **Underground coverage** | Limited to what's indexed by search engines | Deep — books, zines, oral histories pre-indexed |
| **Cross-flier connections** | None | Automatic — "this artist also appeared on 3 other fliers you analyzed" |

The single most impactful change: **pre-indexing 5-10 key underground music history books** would give the app access to exactly the kind of first-hand, deeply contextual information that web search cannot find — which is the stated goal of the project.

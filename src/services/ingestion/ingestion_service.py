"""Orchestrator for the full document ingestion pipeline.

Pipeline stages: **process -> chunk -> tag -> embed -> store**.

The :class:`IngestionService` implements the **Orchestrator pattern**: it
coordinates five collaborators (source processors, chunker, metadata
extractor, embedding provider, vector store) without any of them knowing
about each other.  Each public ``ingest_*`` method follows the same flow:

    1. Source Processor -- reads the raw format, returns chapter-level chunks
    2. TextChunker -- splits chapters into ~500-token overlapping windows
    3. MetadataExtractor -- LLM-tags each chunk with entities/places/genres
    4. IEmbeddingProvider -- generates dense vector embeddings
    5. IVectorStoreProvider -- persists embedded chunks to Qdrant

All dependencies are injected via constructor (Dependency Injection), so
providers can be swapped (e.g. OpenAI -> Ollama) without changing this class.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from src.models.rag import CorpusStats, DocumentChunk, IngestionResult
from src.services.ingestion.chunker import TextChunker
from src.services.ingestion.metadata_extractor import MetadataExtractor
# Source processors -- one per supported input format.
from src.services.ingestion.source_processors.analysis_processor import (
    AnalysisProcessor,
)
from src.services.ingestion.source_processors.article_processor import (
    ArticleProcessor,
)
from src.services.ingestion.source_processors.book_processor import BookProcessor
from src.services.ingestion.source_processors.epub_processor import EPUBProcessor
from src.services.ingestion.source_processors.pdf_processor import PDFProcessor

if TYPE_CHECKING:
    # TYPE_CHECKING-only imports avoid circular dependencies at runtime;
    # these interfaces are only needed for type annotations.
    from src.interfaces.article_provider import IArticleProvider
    from src.interfaces.embedding_provider import IEmbeddingProvider
    from src.interfaces.vector_store_provider import IVectorStoreProvider
    from src.models.pipeline import PipelineState

logger = structlog.get_logger(logger_name=__name__)


class IngestionService:
    """Orchestrates the full ingestion pipeline: process -> chunk -> tag -> embed -> store.

    This class is the central coordinator (Orchestrator pattern).  It owns
    instances of every source processor and delegates to injected providers
    for embedding and storage, keeping each component decoupled.

    Parameters
    ----------
    chunker:
        Splits raw source text into overlapping token-sized windows.
    metadata_extractor:
        Enriches chunks with entity / geographic / genre tags via LLM.
    embedding_provider:
        Generates embedding vectors for chunk text.
    vector_store:
        Stores embedded chunks for semantic retrieval.
    article_scraper:
        Optional article extraction provider used by :meth:`ingest_article`.
    """

    def __init__(
        self,
        chunker: TextChunker,
        metadata_extractor: MetadataExtractor,
        embedding_provider: IEmbeddingProvider,
        vector_store: IVectorStoreProvider,
        article_scraper: IArticleProvider | None = None,
    ) -> None:
        # Injected collaborators -- all behind interfaces for swappability.
        self._chunker = chunker
        self._metadata_extractor = metadata_extractor
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._article_scraper = article_scraper
        # Source processors -- one per input format, instantiated directly
        # since they have no external dependencies to inject.
        self._book_processor = BookProcessor()
        self._pdf_processor = PDFProcessor()
        self._epub_processor = EPUBProcessor()
        self._article_processor = ArticleProcessor()
        self._analysis_processor = AnalysisProcessor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest_book(
        self,
        file_path: str,
        title: str,
        author: str,
        year: int,
    ) -> IngestionResult:
        """Ingest a plain-text book through the full pipeline.

        1. :class:`BookProcessor` → chapter-level sections
        2. :class:`TextChunker` → embedding-sized chunks
        3. :class:`MetadataExtractor` → entity / geo / genre tags
        4. :class:`IEmbeddingProvider` → embedding vectors
        5. :class:`IVectorStoreProvider` → stored in vector DB

        Returns
        -------
        IngestionResult
            Statistics about the ingestion run.
        """
        start = time.monotonic()

        # Step 1: extract chapter sections.
        sections = self._book_processor.process(file_path, title, author, year)
        if not sections:
            return self._empty_result(title=title, source_id="")

        source_id = sections[0].source_id

        # Step 2: chunk each section.
        all_chunks: list[DocumentChunk] = []
        for section in sections:
            metadata = {
                "source_id": section.source_id,
                "source_title": section.source_title,
                "source_type": section.source_type,
                "citation_tier": section.citation_tier,
                "author": section.author,
                "publication_date": section.publication_date,
                "page_number": section.page_number,
            }
            chunks = self._chunker.chunk(section.text, metadata)
            all_chunks.extend(chunks)

        # Steps 3-5: tag, embed, store.
        return await self._tag_embed_store(
            all_chunks, source_id=source_id, title=title, start=start
        )

    async def ingest_pdf(
        self,
        file_path: str,
        title: str,
        author: str,
        year: int,
    ) -> IngestionResult:
        """Ingest a PDF book or document through the full pipeline.

        1. :class:`PDFProcessor` → page-extracted chapter-level sections
        2. :class:`TextChunker` → embedding-sized chunks
        3. :class:`MetadataExtractor` → entity / geo / genre tags
        4. :class:`IEmbeddingProvider` → embedding vectors
        5. :class:`IVectorStoreProvider` → stored in vector DB

        Returns
        -------
        IngestionResult
            Statistics about the ingestion run.
        """
        start = time.monotonic()

        sections = self._pdf_processor.process(file_path, title, author, year)
        if not sections:
            return self._empty_result(title=title, source_id="")

        source_id = sections[0].source_id

        all_chunks: list[DocumentChunk] = []
        for section in sections:
            metadata = {
                "source_id": section.source_id,
                "source_title": section.source_title,
                "source_type": section.source_type,
                "citation_tier": section.citation_tier,
                "author": section.author,
                "publication_date": section.publication_date,
                "page_number": section.page_number,
            }
            chunks = self._chunker.chunk(section.text, metadata)
            all_chunks.extend(chunks)

        return await self._tag_embed_store(
            all_chunks, source_id=source_id, title=title, start=start
        )

    async def ingest_epub(
        self,
        file_path: str,
        title: str,
        author: str,
        year: int,
    ) -> IngestionResult:
        """Ingest an EPUB book through the full pipeline.

        1. :class:`EPUBProcessor` → chapter-level sections from XHTML
        2. :class:`TextChunker` → embedding-sized chunks
        3. :class:`MetadataExtractor` → entity / geo / genre tags
        4. :class:`IEmbeddingProvider` → embedding vectors
        5. :class:`IVectorStoreProvider` → stored in vector DB

        Returns
        -------
        IngestionResult
            Statistics about the ingestion run.
        """
        start = time.monotonic()

        sections = self._epub_processor.process(file_path, title, author, year)
        if not sections:
            return self._empty_result(title=title, source_id="")

        source_id = sections[0].source_id

        all_chunks: list[DocumentChunk] = []
        for section in sections:
            metadata = {
                "source_id": section.source_id,
                "source_title": section.source_title,
                "source_type": section.source_type,
                "citation_tier": section.citation_tier,
                "author": section.author,
                "publication_date": section.publication_date,
            }
            chunks = self._chunker.chunk(section.text, metadata)
            all_chunks.extend(chunks)

        return await self._tag_embed_store(
            all_chunks, source_id=source_id, title=title, start=start
        )

    async def ingest_article(self, url: str) -> IngestionResult:
        """Ingest a web article through the full pipeline.

        Requires that ``article_scraper`` was provided at construction time.

        Returns
        -------
        IngestionResult
            Statistics about the ingestion run.
        """
        start = time.monotonic()

        if self._article_scraper is None:
            logger.error("ingest_article_no_scraper")
            return self._empty_result(title=url, source_id="")

        raw_chunks = await self._article_processor.process_url(url, self._article_scraper)
        if not raw_chunks:
            return self._empty_result(title=url, source_id="")

        source_id = raw_chunks[0].source_id

        # Re-chunk the article text into embedding-sized windows.
        all_chunks: list[DocumentChunk] = []
        for rc in raw_chunks:
            metadata = {
                "source_id": rc.source_id,
                "source_title": rc.source_title,
                "source_type": rc.source_type,
                "citation_tier": rc.citation_tier,
                "author": rc.author,
                "publication_date": rc.publication_date,
            }
            chunks = self._chunker.chunk(rc.text, metadata)
            all_chunks.extend(chunks)

        return await self._tag_embed_store(all_chunks, source_id=source_id, title=url, start=start)

    async def ingest_analysis(self, pipeline_state: PipelineState) -> IngestionResult:
        """Ingest a completed pipeline analysis for the feedback loop.

        Called automatically after Phase 5 completes (if RAG is enabled).

        Returns
        -------
        IngestionResult
            Statistics about the ingestion run.
        """
        start = time.monotonic()

        raw_chunks = self._analysis_processor.process(pipeline_state)
        if not raw_chunks:
            return self._empty_result(title=f"analysis-{pipeline_state.session_id}", source_id="")

        source_id = raw_chunks[0].source_id

        # Analysis chunks are already logical units — re-chunk for size.
        all_chunks: list[DocumentChunk] = []
        for rc in raw_chunks:
            metadata = {
                "source_id": rc.source_id,
                "source_title": rc.source_title,
                "source_type": rc.source_type,
                "citation_tier": rc.citation_tier,
                "entity_tags": rc.entity_tags,
                "geographic_tags": rc.geographic_tags,
            }
            chunks = self._chunker.chunk(rc.text, metadata)
            all_chunks.extend(chunks)

        return await self._tag_embed_store(
            all_chunks,
            source_id=source_id,
            title=f"analysis-{pipeline_state.session_id}",
            start=start,
        )

    async def ingest_directory(
        self,
        dir_path: str,
        source_type: str,
        skip_source_ids: set[str] | None = None,
        skip_tagging: bool = False,
        concurrency: int = 1,
    ) -> list[IngestionResult]:
        """Ingest all ``.txt`` and ``.html`` files in *dir_path*.

        Parameters
        ----------
        dir_path:
            Path to a directory containing article/text files.
        source_type:
            Source type label applied to all files.
        skip_source_ids:
            Optional set of source IDs already in the vector store.
            Files whose content-based ``source_id`` is in this set are
            skipped, avoiding expensive re-tagging and re-embedding of
            already-ingested documents.
        skip_tagging:
            If ``True``, skip the LLM metadata extraction step (entity,
            genre, and geographic tagging).  Chunks are still embedded
            and stored — semantic search works, but tag-based filters
            won't apply.  Cuts ingestion from hours to minutes for
            large corpora by eliminating thousands of LLM API calls.
        concurrency:
            Maximum number of files to process concurrently.  Defaults
            to 1 (sequential) to stay within the 512 MB RAM budget on
            Render.  Each concurrent file holds its full text, chunks,
            and embeddings in memory simultaneously.  Increase on
            machines with more memory for faster ingestion.

        Returns
        -------
        list[IngestionResult]
            One result per file processed (skipped files are excluded).
        """
        path = Path(dir_path)
        if not path.is_dir():
            logger.error("ingest_directory_not_found", dir_path=dir_path)
            return []

        files = sorted(path.glob("*.txt")) + sorted(path.glob("*.html"))
        # Concurrency limit — defaults to 1 (sequential) to stay within
        # the 512 MB RAM budget on Render.  max(1, ...) guards against
        # deadlock from a zero or negative value.
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def _process_file(fp: Path) -> IngestionResult | None:
            async with semaphore:
                start = time.monotonic()
                raw_chunks = self._article_processor.process_file(
                    str(fp), source_type=source_type
                )
                if not raw_chunks:
                    return None

                source_id = raw_chunks[0].source_id

                # Skip files already in the vector store (incremental ingest)
                if skip_source_ids and source_id in skip_source_ids:
                    logger.debug(
                        "skip_already_ingested",
                        file=fp.name,
                        source_id=source_id[:12],
                    )
                    return None
                _needs_preprocess = source_type in ("transcript", "interview")
                all_chunks: list[DocumentChunk] = []
                for rc in raw_chunks:
                    text = rc.text
                    if _needs_preprocess:
                        from src.utils.text_normalizer import preprocess_transcript

                        text = preprocess_transcript(text)
                    metadata = {
                        "source_id": rc.source_id,
                        "source_title": rc.source_title,
                        "source_type": rc.source_type,
                        "citation_tier": rc.citation_tier,
                    }
                    chunks = self._chunker.chunk(text, metadata)
                    all_chunks.extend(chunks)

                return await self._tag_embed_store(
                    all_chunks,
                    source_id=source_id,
                    title=fp.name,
                    start=start,
                    skip_tagging=skip_tagging,
                )

        raw_results = await asyncio.gather(*[_process_file(f) for f in files])
        results = [r for r in raw_results if r is not None]

        logger.info(
            "directory_ingestion_complete",
            dir_path=dir_path,
            files_processed=len(results),
        )
        return results

    async def get_corpus_stats(self) -> CorpusStats:
        """Return aggregate statistics about the vector-store corpus."""
        return await self._vector_store.get_stats()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    # Maximum number of chunks to embed and store in a single pass.
    # On Render's 512 MB budget, holding thousands of embeddings
    # (each 1024 floats) plus the chunk texts in memory at once can
    # OOM-crash the container.  Processing in slices of 500 keeps
    # peak memory bounded: ~500 chunks * ~2 KB text + ~500 * 4 KB
    # embedding ≈ 3 MB per slice, well within budget.
    _EMBED_STORE_BATCH = 500

    async def _tag_embed_store(
        self,
        chunks: list[DocumentChunk],
        source_id: str,
        title: str,
        start: float,
        skip_tagging: bool = False,
    ) -> IngestionResult:
        """Run the tag -> embed -> store stages and return an :class:`IngestionResult`.

        This is the shared tail of every ``ingest_*`` method.  After format-specific
        processing and chunking, all paths converge here for the final three stages.

        For large files (thousands of chunks), the embed and store steps
        are batched in slices of ``_EMBED_STORE_BATCH`` to keep peak
        memory bounded on the 512 MB Render instance.
        """
        if not chunks:
            return self._empty_result(title=title, source_id=source_id)

        # Step 3: metadata extraction -- LLM tags each chunk with entities,
        # geographic locations, and music genres for filtered retrieval.
        # Skipped when skip_tagging=True (e.g. large reference corpus files
        # where the LLM cost/time isn't justified — semantic search via
        # embeddings still works without tags).
        if skip_tagging:
            logger.info("skip_metadata_tagging", title=title, chunks=len(chunks))
            tagged_chunks = chunks
        else:
            tagged_chunks = await self._metadata_extractor.extract_batch(chunks)

        # Steps 4-5: embed and store in batches to bound peak memory.
        # Each slice embeds its chunk texts, stores the results, then
        # releases the embedding vectors before the next slice starts.
        total_stored = 0
        batch = self._EMBED_STORE_BATCH

        for i in range(0, len(tagged_chunks), batch):
            slice_chunks = tagged_chunks[i : i + batch]
            texts = [c.text for c in slice_chunks]
            embeddings = await self._embedding_provider.embed(texts)
            stored = await self._vector_store.add_chunks(slice_chunks, embeddings)
            total_stored += stored
            # embeddings and texts go out of scope here, allowing GC
            # before the next slice allocates its own.

        elapsed = time.monotonic() - start
        total_tokens = sum(c.token_count for c in tagged_chunks)

        result = IngestionResult(
            source_id=source_id,
            source_title=title,
            chunks_created=total_stored,
            total_tokens=total_tokens,
            ingestion_time=round(elapsed, 2),
        )

        logger.info(
            "ingestion_complete",
            source_title=title,
            chunks=total_stored,
            tokens=total_tokens,
            time_s=result.ingestion_time,
        )
        return result

    @staticmethod
    def _empty_result(title: str, source_id: str) -> IngestionResult:
        """Return an :class:`IngestionResult` with zero counts."""
        return IngestionResult(
            source_id=source_id,
            source_title=title,
            chunks_created=0,
            total_tokens=0,
            ingestion_time=0.0,
        )

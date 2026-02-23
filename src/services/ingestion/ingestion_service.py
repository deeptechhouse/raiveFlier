"""Orchestrator for the full document ingestion pipeline.

Pipeline stages: **process → chunk → tag → embed → store**.

The :class:`IngestionService` coordinates source processors, the text
chunker, the metadata extractor, the embedding provider, and the vector
store to ingest books, articles, completed analyses, and bulk directories
into the RAG knowledge base.
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
    from src.interfaces.article_provider import IArticleProvider
    from src.interfaces.embedding_provider import IEmbeddingProvider
    from src.interfaces.vector_store_provider import IVectorStoreProvider
    from src.models.pipeline import PipelineState

logger = structlog.get_logger(logger_name=__name__)


class IngestionService:
    """Orchestrates the full ingestion pipeline: process → chunk → tag → embed → store.

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
        self._chunker = chunker
        self._metadata_extractor = metadata_extractor
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._article_scraper = article_scraper
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
    ) -> list[IngestionResult]:
        """Ingest all ``.txt`` and ``.html`` files in *dir_path*.

        Parameters
        ----------
        dir_path:
            Path to a directory containing article/text files.
        source_type:
            Source type label applied to all files.

        Returns
        -------
        list[IngestionResult]
            One result per file processed.
        """
        path = Path(dir_path)
        if not path.is_dir():
            logger.error("ingest_directory_not_found", dir_path=dir_path)
            return []

        files = sorted(path.glob("*.txt")) + sorted(path.glob("*.html"))
        semaphore = asyncio.Semaphore(4)

        async def _process_file(fp: Path) -> IngestionResult | None:
            async with semaphore:
                start = time.monotonic()
                raw_chunks = self._article_processor.process_file(
                    str(fp), source_type=source_type
                )
                if not raw_chunks:
                    return None

                source_id = raw_chunks[0].source_id
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

    async def _tag_embed_store(
        self,
        chunks: list[DocumentChunk],
        source_id: str,
        title: str,
        start: float,
    ) -> IngestionResult:
        """Run the tag → embed → store stages and return an :class:`IngestionResult`."""
        if not chunks:
            return self._empty_result(title=title, source_id=source_id)

        # Step 3: metadata extraction (LLM tagging).
        tagged_chunks = await self._metadata_extractor.extract_batch(chunks)

        # Step 4: generate embeddings.
        texts = [c.text for c in tagged_chunks]
        embeddings = await self._embedding_provider.embed(texts)

        # Step 5: store in vector DB.
        stored = await self._vector_store.add_chunks(tagged_chunks, embeddings)

        elapsed = time.monotonic() - start
        total_tokens = sum(c.token_count for c in tagged_chunks)

        result = IngestionResult(
            source_id=source_id,
            source_title=title,
            chunks_created=stored,
            total_tokens=total_tokens,
            ingestion_time=round(elapsed, 2),
        )

        logger.info(
            "ingestion_complete",
            source_title=title,
            chunks=stored,
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

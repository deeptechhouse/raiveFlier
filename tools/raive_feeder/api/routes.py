"""raiveFeeder REST API endpoints.

# ─── ROUTE ARCHITECTURE ───────────────────────────────────────────────
#
# All routes live in a single APIRouter mounted at /api/v1 by main.py.
# Route handlers are thin — they validate input, delegate to services,
# and format responses.  Business logic stays in the services layer.
#
# Dependencies are accessed via request.app.state (set during startup
# by main.py's _build_all function).  This avoids FastAPI's Depends()
# complexity for singleton services.
#
# Endpoint groups:
#   /ingest/*   — Document, audio, image, URL, and batch ingestion
#   /jobs/*     — Batch job lifecycle (status, cancel, pause, resume)
#   /corpus/*   — Corpus management (stats, sources, search, CRUD)
#   /health     — Health check + provider availability
#   /providers  — List available providers
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from tools.raive_feeder.api.schemas import (
    BatchIngestRequest,
    ChunkMetadataUpdate,
    CorpusSearchRequest,
    CorpusSourceSummary,
    CorpusStatsResponse,
    DocumentMetadata,
    HealthResponse,
    ImageIngestRequest,
    ImageIngestResponse,
    ImageMode,
    IngestDocumentResponse,
    JobStatus,
    JobStatusResponse,
    ProviderInfo,
    ScrapedPage,
    ScrapeURLRequest,
    ScrapeURLResponse,
    TranscriptionProvider,
    TranscriptionRequest,
    TranscriptionResponse,
)

logger = structlog.get_logger(logger_name=__name__)

router = APIRouter()


# ─── Helper: get service from app.state ────────────────────────────────

def _get_ingestion(request: Request) -> Any:
    """Retrieve the IngestionService from app.state, raising 503 if unavailable."""
    svc = getattr(request.app.state, "ingestion_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")
    return svc


def _get_vector_store(request: Request) -> Any:
    """Retrieve the vector store from app.state, raising 503 if unavailable."""
    vs = getattr(request.app.state, "vector_store", None)
    if vs is None:
        raise HTTPException(status_code=503, detail="Vector store unavailable")
    return vs


# ═══════════════════════════════════════════════════════════════════════
# INGESTION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════


@router.post("/ingest/document", response_model=IngestDocumentResponse)
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form(""),
    author: str = Form(""),
    year: int = Form(0),
    source_type: str = Form("book"),
    citation_tier: int = Form(2),
) -> IngestDocumentResponse:
    """Upload and ingest a document (PDF, EPUB, TXT, DOCX, RTF, MOBI, DJVU).

    The file is saved to a temp directory, optionally format-converted,
    then passed through the shared ingestion pipeline (chunk → tag → embed → store).
    """
    svc = _get_ingestion(request)
    job_id = str(uuid4())

    # Auto-detect title from filename if not provided.
    if not title:
        title = Path(file.filename or "untitled").stem

    try:
        # Save uploaded file to temp location.
        suffix = Path(file.filename or "").suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Route to appropriate ingestion method based on file extension.
        result = await _ingest_by_extension(
            svc, request, tmp_path, suffix, title, author, year, source_type
        )

        return IngestDocumentResponse(
            job_id=job_id,
            source_id=result.source_id,
            source_title=result.source_title,
            chunks_created=result.chunks_created,
            total_tokens=result.total_tokens,
            ingestion_time=result.ingestion_time,
            status=JobStatus.COMPLETED,
        )

    except Exception as exc:
        logger.error("ingest_document_failed", error=str(exc), filename=file.filename)
        return IngestDocumentResponse(
            job_id=job_id,
            status=JobStatus.FAILED,
            error=str(exc),
        )


async def _ingest_by_extension(
    svc: Any,
    request: Request,
    tmp_path: str,
    suffix: str,
    title: str,
    author: str,
    year: int,
    source_type: str,
) -> Any:
    """Dispatch to the correct ingestion method based on file extension.

    For formats not natively supported by the IngestionService (MOBI, DJVU,
    DOC, DOCX, RTF), delegates to FormatConverter first.
    """
    if suffix == ".pdf":
        return await svc.ingest_pdf(file_path=tmp_path, title=title, author=author, year=year)
    elif suffix == ".epub":
        return await svc.ingest_epub(file_path=tmp_path, title=title, author=author, year=year)
    elif suffix in (".txt", ".html", ".htm"):
        return await svc.ingest_book(file_path=tmp_path, title=title, author=author, year=year)
    elif suffix in (".mobi", ".djvu", ".doc", ".docx", ".rtf"):
        # Convert to a format the ingestion pipeline handles.
        from tools.raive_feeder.services.format_converter import FormatConverter
        converter = FormatConverter()
        converted_path, converted_format = await converter.convert(tmp_path, suffix)
        if converted_format == "epub":
            return await svc.ingest_epub(
                file_path=converted_path, title=title, author=author, year=year
            )
        else:
            return await svc.ingest_book(
                file_path=converted_path, title=title, author=author, year=year
            )
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


@router.post("/ingest/audio", response_model=TranscriptionResponse)
async def ingest_audio(
    request: Request,
    file: UploadFile = File(...),
    provider: str = Form("whisper_local"),
    language: str = Form(None),
    title: str = Form(""),
    source_type: str = Form("interview"),
    citation_tier: int = Form(3),
) -> TranscriptionResponse:
    """Upload audio for transcription.  Returns transcript for review before ingestion."""
    job_id = str(uuid4())

    if not title:
        title = Path(file.filename or "untitled").stem

    try:
        # Save to temp file.
        suffix = Path(file.filename or "").suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Transcribe.
        from tools.raive_feeder.services.audio_transcriber import AudioTranscriber
        settings = request.app.state.settings
        transcriber = AudioTranscriber(settings=settings)
        result = await transcriber.transcribe(
            audio_path=tmp_path,
            provider_name=provider,
            language=language,
        )

        return TranscriptionResponse(
            job_id=job_id,
            transcript=result["text"],
            language=result.get("language", "en"),
            duration_seconds=result.get("duration", 0.0),
            provider_used=result.get("provider", provider),
            status=JobStatus.COMPLETED,
        )

    except Exception as exc:
        logger.error("ingest_audio_failed", error=str(exc))
        return TranscriptionResponse(
            job_id=job_id,
            transcript="",
            status=JobStatus.FAILED,
            error=str(exc),
        )


@router.post("/ingest/images", response_model=ImageIngestResponse)
async def ingest_images(
    request: Request,
    files: list[UploadFile] = File(...),
    mode: str = Form("single_flier"),
    title: str = Form(""),
    author: str = Form(""),
    source_type: str = Form("flier"),
    citation_tier: int = Form(4),
) -> ImageIngestResponse:
    """Upload image(s) for OCR + ingestion."""
    job_id = str(uuid4())

    try:
        # Save all uploaded images to temp files.
        tmp_paths: list[str] = []
        for f in files:
            suffix = Path(f.filename or "").suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await f.read()
                tmp.write(content)
                tmp_paths.append(tmp.name)

        from tools.raive_feeder.services.image_ingester import ImageIngester
        ocr_providers = getattr(request.app.state, "ocr_providers", [])
        ingester = ImageIngester(ocr_providers=ocr_providers)

        if mode == "multi_page_scan":
            result = await ingester.ingest_multi_page(
                image_paths=tmp_paths,
                title=title or "Multi-page scan",
                author=author,
                source_type=source_type,
                ingestion_service=_get_ingestion(request),
            )
        else:
            result = await ingester.ingest_single(
                image_path=tmp_paths[0],
                title=title or Path(files[0].filename or "flier").stem,
                source_type=source_type,
                ingestion_service=_get_ingestion(request),
            )

        return ImageIngestResponse(
            job_id=job_id,
            ocr_text=result.get("ocr_text", ""),
            chunks_created=result.get("chunks_created", 0),
            status=JobStatus.COMPLETED,
        )

    except Exception as exc:
        logger.error("ingest_images_failed", error=str(exc))
        return ImageIngestResponse(
            job_id=job_id,
            status=JobStatus.FAILED,
            error=str(exc),
        )


@router.post("/ingest/url", response_model=ScrapeURLResponse)
async def ingest_url(
    request: Request,
    body: ScrapeURLRequest,
) -> ScrapeURLResponse:
    """Scrape a URL (optionally with recursive depth and NL-guided filtering)."""
    job_id = str(uuid4())

    try:
        from tools.raive_feeder.services.web_crawler import WebCrawler
        llm = getattr(request.app.state, "llm_provider", None)
        settings = request.app.state.settings

        crawler = WebCrawler(
            llm_provider=llm,
            rate_limit_seconds=settings.crawl_rate_limit_seconds,
        )

        pages = await crawler.crawl(
            seed_url=body.url,
            max_depth=body.max_depth,
            max_pages=body.max_pages,
            nl_query=body.nl_query,
        )

        scraped = [
            ScrapedPage(
                url=p["url"],
                title=p.get("title", ""),
                text_preview=p.get("text", "")[:500],
                relevance_score=p.get("relevance_score"),
                word_count=len(p.get("text", "").split()),
            )
            for p in pages
        ]

        # Auto-ingest if requested.
        if body.auto_ingest:
            svc = _get_ingestion(request)
            for p in pages:
                if p.get("text"):
                    try:
                        await svc.ingest_article(url=p["url"])
                    except Exception:
                        pass

        return ScrapeURLResponse(
            job_id=job_id,
            pages=scraped,
            status=JobStatus.COMPLETED,
        )

    except Exception as exc:
        logger.error("ingest_url_failed", error=str(exc))
        return ScrapeURLResponse(
            job_id=job_id,
            status=JobStatus.FAILED,
            error=str(exc),
        )


@router.post("/ingest/batch", response_model=JobStatusResponse)
async def ingest_batch(
    request: Request,
    files: list[UploadFile] = File(...),
    source_type: str = Form("article"),
    citation_tier: int = Form(3),
    skip_tagging: bool = Form(False),
) -> JobStatusResponse:
    """Queue a batch ingestion job for multiple files."""
    batch_processor = getattr(request.app.state, "batch_processor", None)
    if batch_processor is None:
        raise HTTPException(status_code=503, detail="Batch processor unavailable")

    # Save all files to temp locations.
    items: list[dict[str, Any]] = []
    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await f.read()
            tmp.write(content)
            items.append({
                "filename": f.filename or "unknown",
                "tmp_path": tmp.name,
                "suffix": suffix,
                "source_type": source_type,
                "citation_tier": citation_tier,
                "skip_tagging": skip_tagging,
            })

    job_id = batch_processor.create_job(items)
    svc = _get_ingestion(request)

    # Define the per-item processing function.
    async def _process_item(item: dict[str, Any]) -> dict[str, Any]:
        title = Path(item["filename"]).stem
        result = await _ingest_by_extension(
            svc, request, item["tmp_path"], item["suffix"],
            title=title, author="", year=0, source_type=item["source_type"],
        )
        return {"source_id": result.source_id, "chunks": result.chunks_created}

    # Start the batch in the background.
    await batch_processor.start_job(job_id, _process_item)

    status = batch_processor.get_job_status(job_id)
    return status or JobStatusResponse(job_id=job_id, status=JobStatus.QUEUED)


# ═══════════════════════════════════════════════════════════════════════
# JOB MANAGEMENT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════


@router.get("/jobs", response_model=list[JobStatusResponse])
async def list_jobs(request: Request) -> list[JobStatusResponse]:
    """List all active and completed jobs."""
    bp = getattr(request.app.state, "batch_processor", None)
    return bp.list_jobs() if bp else []


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(request: Request, job_id: str) -> JobStatusResponse:
    """Get status of a specific job."""
    bp = getattr(request.app.state, "batch_processor", None)
    if bp is None:
        raise HTTPException(status_code=503, detail="Batch processor unavailable")
    status = bp.get_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(request: Request, job_id: str) -> dict[str, str]:
    """Cancel a running job."""
    bp = getattr(request.app.state, "batch_processor", None)
    if bp and bp.cancel_job(job_id):
        return {"status": "cancelled"}
    raise HTTPException(status_code=404, detail="Job not found")


@router.post("/jobs/{job_id}/pause")
async def pause_job(request: Request, job_id: str) -> dict[str, str]:
    """Pause a running job."""
    bp = getattr(request.app.state, "batch_processor", None)
    if bp and bp.pause_job(job_id):
        return {"status": "paused"}
    raise HTTPException(status_code=404, detail="Job not found")


@router.post("/jobs/{job_id}/resume")
async def resume_job(request: Request, job_id: str) -> dict[str, str]:
    """Resume a paused job."""
    bp = getattr(request.app.state, "batch_processor", None)
    if bp and bp.resume_job(job_id):
        return {"status": "resumed"}
    raise HTTPException(status_code=404, detail="Job not found")


# ═══════════════════════════════════════════════════════════════════════
# CORPUS MANAGEMENT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════


@router.get("/corpus/stats", response_model=CorpusStatsResponse)
async def corpus_stats(request: Request) -> CorpusStatsResponse:
    """Get corpus statistics for the dashboard."""
    vs = _get_vector_store(request)
    stats = await vs.get_stats()
    return CorpusStatsResponse(
        total_chunks=stats.total_chunks,
        total_sources=stats.total_sources,
        sources_by_type=stats.sources_by_type,
        entity_tag_count=stats.entity_tag_count,
        geographic_tag_count=stats.geographic_tag_count,
        genre_tag_count=stats.genre_tag_count,
        genre_tags=stats.genre_tags,
        time_periods=stats.time_periods,
    )


@router.get("/corpus/sources", response_model=list[CorpusSourceSummary])
async def list_sources(request: Request) -> list[CorpusSourceSummary]:
    """List all ingested sources with summary metadata."""
    from tools.raive_feeder.services.corpus_manager import CorpusManager
    vs = _get_vector_store(request)
    manager = CorpusManager(vector_store=vs)
    return await manager.list_sources()


@router.get("/corpus/sources/{source_id}")
async def get_source_detail(request: Request, source_id: str) -> dict[str, Any]:
    """Get all chunks for a specific source."""
    from tools.raive_feeder.services.corpus_manager import CorpusManager
    vs = _get_vector_store(request)
    manager = CorpusManager(vector_store=vs)
    return await manager.get_source_detail(source_id)


@router.delete("/corpus/sources/{source_id}")
async def delete_source(request: Request, source_id: str) -> dict[str, Any]:
    """Delete all chunks from a specific source."""
    vs = _get_vector_store(request)
    deleted = await vs.delete_by_source(source_id)
    return {"source_id": source_id, "deleted_chunks": deleted}


@router.delete("/corpus/type/{source_type}")
async def delete_source_type(request: Request, source_type: str) -> dict[str, Any]:
    """Delete all chunks of a given source type."""
    vs = _get_vector_store(request)
    deleted = await vs.delete_by_source_type(source_type)
    return {"source_type": source_type, "deleted_chunks": deleted}


@router.post("/corpus/search")
async def corpus_search(request: Request, body: CorpusSearchRequest) -> list[dict[str, Any]]:
    """Perform a semantic search against the corpus."""
    vs = _get_vector_store(request)
    filters: dict[str, Any] = {}
    if body.source_type:
        filters["source_type"] = {"$in": [body.source_type]}
    if body.genre:
        filters["genre_tags"] = {"$contains": body.genre}

    results = await vs.query(
        query_text=body.query,
        top_k=body.top_k,
        filters=filters if filters else None,
    )

    return [
        {
            "chunk_id": r.chunk.chunk_id,
            "text": r.chunk.text,
            "source_title": r.chunk.source_title,
            "source_type": r.chunk.source_type,
            "similarity_score": r.similarity_score,
            "formatted_citation": r.formatted_citation,
            "entity_tags": r.chunk.entity_tags,
            "genre_tags": r.chunk.genre_tags,
        }
        for r in results
        if r.similarity_score >= body.min_similarity
    ]


@router.patch("/corpus/chunks/{chunk_id}")
async def update_chunk_metadata(
    request: Request, chunk_id: str, body: ChunkMetadataUpdate
) -> dict[str, Any]:
    """Update metadata on a specific chunk without re-embedding."""
    vs = _get_vector_store(request)
    metadata: dict[str, Any] = {}
    if body.entity_tags is not None:
        metadata["entity_tags"] = ",".join(body.entity_tags)
    if body.geographic_tags is not None:
        metadata["geographic_tags"] = ",".join(body.geographic_tags)
    if body.genre_tags is not None:
        metadata["genre_tags"] = ",".join(body.genre_tags)
    if body.time_period is not None:
        metadata["time_period"] = body.time_period
    if body.citation_tier is not None:
        metadata["citation_tier"] = body.citation_tier

    success = await vs.update_chunk_metadata(chunk_id, metadata)
    return {"chunk_id": chunk_id, "updated": success}


@router.post("/corpus/export")
async def export_corpus(request: Request) -> dict[str, str]:
    """Package the ChromaDB data directory as a tarball for download."""
    from tools.raive_feeder.services.corpus_manager import CorpusManager
    vs = _get_vector_store(request)
    settings = request.app.state.settings
    manager = CorpusManager(vector_store=vs)
    tarball_path = await manager.export_corpus(settings.chromadb_persist_dir)
    return {"tarball_path": tarball_path}


@router.post("/corpus/import")
async def import_corpus(
    request: Request,
    file: UploadFile = File(...),
) -> dict[str, str]:
    """Import a corpus tarball into the ChromaDB data directory."""
    from tools.raive_feeder.services.corpus_manager import CorpusManager
    vs = _get_vector_store(request)
    settings = request.app.state.settings
    manager = CorpusManager(vector_store=vs)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    await manager.import_corpus(tmp_path, settings.chromadb_persist_dir)
    return {"status": "imported"}


@router.post("/corpus/reembed/{source_id}")
async def reembed_source(request: Request, source_id: str) -> dict[str, Any]:
    """Delete and re-ingest a source (re-embeds with current embedding model)."""
    vs = _get_vector_store(request)
    deleted = await vs.delete_by_source(source_id)
    return {"source_id": source_id, "deleted_chunks": deleted, "note": "Re-ingest via document upload"}


# ─── Corpus Publishing (GitHub Release + Render Deploy) ─────────────


@router.post("/corpus/publish")
async def publish_corpus(request: Request) -> dict[str, Any]:
    """Export corpus, upload to GitHub release, optionally trigger Render deploy.

    Accepts JSON body: {"tag": "v1.0.2"}
    """
    from tools.raive_feeder.services.corpus_manager import CorpusManager
    from tools.raive_feeder.services.corpus_publisher import CorpusPublisher

    body = await request.json()
    tag = body.get("tag", "").strip()
    if not tag:
        raise HTTPException(status_code=422, detail="tag is required")

    vs = _get_vector_store(request)
    settings = request.app.state.settings

    publisher = CorpusPublisher(
        corpus_manager=CorpusManager(vector_store=vs),
        github_token=settings.github_token,
        corpus_repo=settings.corpus_repo,
        render_deploy_hook_url=settings.render_deploy_hook_url,
    )

    try:
        result = await publisher.publish(tag, settings.chromadb_persist_dir)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("corpus_publish_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Publish failed: {exc}") from exc


@router.get("/corpus/publish/status")
async def publish_status(request: Request) -> dict[str, Any]:
    """Return publish configuration and latest release tag."""
    from tools.raive_feeder.services.corpus_manager import CorpusManager
    from tools.raive_feeder.services.corpus_publisher import CorpusPublisher

    vs = getattr(request.app.state, "vector_store", None)
    settings = request.app.state.settings

    publisher = CorpusPublisher(
        corpus_manager=CorpusManager(vector_store=vs) if vs else CorpusManager(vector_store=None),
        github_token=settings.github_token,
        corpus_repo=settings.corpus_repo,
        render_deploy_hook_url=settings.render_deploy_hook_url,
    )

    return await publisher.get_publish_status()


# ═══════════════════════════════════════════════════════════════════════
# SYSTEM ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Health check showing provider availability."""
    ingestion = getattr(request.app.state, "ingestion_service", None)
    vs = getattr(request.app.state, "vector_store", None)
    llm = getattr(request.app.state, "llm_provider", None)
    ocr = getattr(request.app.state, "ocr_providers", [])
    status_msg = getattr(request.app.state, "ingestion_status", "")

    return HealthResponse(
        status="ok",
        ingestion_available=ingestion is not None,
        vector_store_available=vs is not None and vs.is_available(),
        llm_available=llm is not None,
        ocr_providers=len(ocr),
        ingestion_status=status_msg,
    )


@router.get("/providers", response_model=list[ProviderInfo])
async def list_providers(request: Request) -> list[ProviderInfo]:
    """List all available providers and their status."""
    providers: list[ProviderInfo] = []

    embedding = getattr(request.app.state, "embedding_provider", None)
    if embedding:
        providers.append(ProviderInfo(
            name=embedding.get_provider_name(),
            provider_type="embedding",
        ))

    llm = getattr(request.app.state, "llm_provider", None)
    if llm:
        providers.append(ProviderInfo(
            name=llm.get_provider_name(),
            provider_type="llm",
        ))

    vs = getattr(request.app.state, "vector_store", None)
    if vs:
        providers.append(ProviderInfo(
            name=vs.get_provider_name(),
            provider_type="vector_store",
            available=vs.is_available(),
        ))

    for ocr in getattr(request.app.state, "ocr_providers", []):
        providers.append(ProviderInfo(
            name=ocr.get_provider_name(),
            provider_type="ocr",
            available=ocr.is_available(),
        ))

    return providers

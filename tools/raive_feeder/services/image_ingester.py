"""Image OCR ingestion service for fliers and multi-page scans.

# ─── DESIGN ────────────────────────────────────────────────────────────
#
# ImageIngester wraps the existing OCR providers (src/providers/ocr/)
# to provide two ingestion modes:
#
#   1. Single flier mode — OCR one image, ingest as source_type="flier"
#      Uses LLM Vision (best for stylized text) falling back to Tesseract.
#
#   2. Multi-page scan mode — OCR multiple images representing pages
#      of a book or magazine, concatenate the text in page order, and
#      ingest as a single document.  Uses Tesseract (best for clean
#      printed text) falling back to LLM Vision.
#
# After OCR, the extracted text is displayed in the UI for manual
# correction before being passed to the IngestionService.
#
# Pattern: Strategy (mode selection), Adapter (wraps OCR providers).
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(logger_name=__name__)


class ImageIngester:
    """OCR-based image ingestion for fliers and multi-page scans.

    Parameters
    ----------
    ocr_providers:
        Ordered list of OCR providers to try (fallback chain).
    """

    def __init__(self, ocr_providers: list[Any] | None = None) -> None:
        self._ocr_providers = ocr_providers or []

    async def ingest_single(
        self,
        image_path: str,
        title: str,
        source_type: str = "flier",
        ingestion_service: Any = None,
    ) -> dict[str, Any]:
        """OCR a single image and ingest it.

        Returns dict with keys: ocr_text, chunks_created.
        """
        # Read image bytes.
        image_bytes = Path(image_path).read_bytes()

        # Run OCR through the fallback chain.
        ocr_text = await self._run_ocr(image_bytes)

        if not ocr_text.strip():
            return {"ocr_text": "", "chunks_created": 0}

        # Ingest via the shared IngestionService.
        chunks_created = 0
        if ingestion_service is not None:
            # Write OCR text to a temp file for the book processor.
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(ocr_text)
                tmp_path = tmp.name

            result = await ingestion_service.ingest_book(
                file_path=tmp_path,
                title=title,
                author="",
                year=0,
            )
            chunks_created = result.chunks_created

        return {"ocr_text": ocr_text, "chunks_created": chunks_created}

    async def ingest_multi_page(
        self,
        image_paths: list[str],
        title: str,
        author: str = "",
        source_type: str = "book",
        ingestion_service: Any = None,
    ) -> dict[str, Any]:
        """OCR multiple images as pages of a document and ingest.

        Images are sorted by filename (assumed page order) and OCR'd
        sequentially.  The resulting texts are concatenated with page
        separators.

        Returns dict with keys: ocr_text, chunks_created.
        """
        # Sort by filename for page ordering.
        sorted_paths = sorted(image_paths, key=lambda p: Path(p).name)

        all_text_parts: list[str] = []
        for idx, img_path in enumerate(sorted_paths, start=1):
            image_bytes = Path(img_path).read_bytes()
            page_text = await self._run_ocr(image_bytes)
            if page_text.strip():
                all_text_parts.append(f"--- Page {idx} ---\n{page_text}")

        combined_text = "\n\n".join(all_text_parts)

        if not combined_text.strip():
            return {"ocr_text": "", "chunks_created": 0}

        # Ingest the combined text.
        chunks_created = 0
        if ingestion_service is not None:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(combined_text)
                tmp_path = tmp.name

            result = await ingestion_service.ingest_book(
                file_path=tmp_path,
                title=title,
                author=author,
                year=0,
            )
            chunks_created = result.chunks_created

        return {"ocr_text": combined_text, "chunks_created": chunks_created}

    async def _run_ocr(self, image_bytes: bytes) -> str:
        """Try each OCR provider in order until one succeeds."""
        for provider in self._ocr_providers:
            try:
                result = await provider.extract_text(image_bytes)
                if result and result.text.strip():
                    logger.info(
                        "ocr_success",
                        provider=provider.get_provider_name(),
                        text_length=len(result.text),
                    )
                    return result.text
            except Exception as exc:
                logger.warning(
                    "ocr_provider_failed",
                    provider=provider.get_provider_name(),
                    error=str(exc),
                )
                continue

        logger.warning("all_ocr_providers_failed")
        return ""

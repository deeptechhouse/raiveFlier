"""EasyOCR provider for rave flier text extraction.

EasyOCR uses deep learning models and handles a wider variety of font styles
than Tesseract, making it a better fallback for the stylised typography
common on rave and electronic-music fliers.
"""

from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from src.interfaces.ocr_provider import IOCRProvider
from src.models.flier import FlierImage, OCRResult, TextRegion
from src.utils.errors import OCRExtractionError
from src.utils.image_preprocessor import ImagePreprocessor
from src.utils.logging import get_logger
from src.utils.ocr_helpers import merge_pass_results
from src.utils.text_normalizer import correct_ocr_errors


class EasyOCRProvider(IOCRProvider):
    """OCR provider backed by EasyOCR (deep-learning-based).

    The EasyOCR reader is initialised lazily on first use to avoid heavy
    model loading at import time.  Multi-pass preprocessing (standard,
    inverted, per-channel, CLAHE, denoised, saturation, Otsu) is applied
    and results are merged across all passes.
    """

    def __init__(self, preprocessor: ImagePreprocessor) -> None:
        self._preprocessor = preprocessor
        self._logger = get_logger(__name__)
        self.__reader = None  # Lazy — loaded on first extract_text call

    # ------------------------------------------------------------------
    # IOCRProvider interface
    # ------------------------------------------------------------------

    async def extract_text(self, image: FlierImage) -> OCRResult:
        """Extract text from a flier image using EasyOCR with multi-pass OCR."""
        start = time.perf_counter()
        try:
            image_bytes = image.image_data
            if image_bytes is None:
                raise OCRExtractionError(
                    "No image data provided",
                    provider_name=self.get_provider_name(),
                )

            original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            passes = self._build_passes(original)
            reader = self._get_reader()

            all_results: list[dict | None] = []

            def _run_pass(args: tuple[str, Image.Image]) -> dict | None:
                pass_name, pass_image = args
                self._logger.debug("running_ocr_pass", pass_name=pass_name, provider="easyocr")
                try:
                    result = self._run_easyocr(reader, pass_image)
                    if result is not None:
                        result["pass_name"] = pass_name
                    return result
                except Exception as exc:
                    self._logger.warning(
                        "ocr_pass_failed",
                        pass_name=pass_name,
                        provider="easyocr",
                        error=str(exc),
                    )
                    return None

            # EasyOCR releases the GIL during C++ inference, so threads
            # give real parallelism on multi-core machines.
            with ThreadPoolExecutor(max_workers=min(4, len(passes))) as pool:
                all_results = list(pool.map(_run_pass, passes))

            elapsed = time.perf_counter() - start

            merged = merge_pass_results(all_results)
            if merged is None:
                raise OCRExtractionError(
                    "All OCR passes returned no usable text",
                    provider_name=self.get_provider_name(),
                )

            self._logger.info(
                "ocr_extraction_complete",
                provider="easyocr",
                passes_run=len(passes),
                confidence=round(merged["confidence"], 4),
                num_regions=len(merged["bounding_boxes"]),
                processing_time=round(elapsed, 3),
            )

            return OCRResult(
                raw_text=correct_ocr_errors(merged["raw_text"]),
                confidence=merged["confidence"],
                provider_used="easyocr",
                processing_time=elapsed,
                bounding_boxes=merged["bounding_boxes"],
            )

        except OCRExtractionError:
            raise
        except Exception as exc:
            elapsed = time.perf_counter() - start
            self._logger.error(
                "ocr_extraction_failed",
                provider="easyocr",
                error=str(exc),
                processing_time=round(elapsed, 3),
            )
            raise OCRExtractionError(
                f"EasyOCR failed: {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

    def get_provider_name(self) -> str:
        return "easyocr"

    def is_available(self) -> bool:
        """Check that the easyocr package is importable."""
        try:
            import easyocr  # noqa: F401

            return True
        except ImportError:
            return False

    def supports_stylized_text(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_reader(self):  # noqa: ANN202
        """Lazily initialise the EasyOCR reader on first use."""
        if self.__reader is None:
            import easyocr

            self._logger.info("initializing_easyocr_reader", languages=["en"])
            self.__reader = easyocr.Reader(["en"], gpu=False)
        return self.__reader

    def _build_passes(self, original: Image.Image) -> list[tuple[str, Image.Image]]:
        """Build (pass_name, preprocessed_image) pairs for multi-pass OCR.

        Delegates preprocessing to :meth:`ImagePreprocessor.build_ocr_passes`
        which caches results so fallback providers skip redundant work.
        """
        return self._preprocessor.build_ocr_passes(original)

    def _run_easyocr(self, reader, pass_image: Image.Image) -> dict | None:  # noqa: ANN001
        """Run EasyOCR on a single image and return results dict or None."""
        img_array = np.array(pass_image)
        results = reader.readtext(
            img_array,
            low_text=0.3,
            text_threshold=0.5,
        )

        if not results:
            return None

        bounding_boxes: list[TextRegion] = []
        confidences: list[float] = []
        text_parts: list[str] = []

        for bbox, text, confidence in results:
            text = text.strip()
            if not text:
                continue

            # EasyOCR bbox: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (quadrilateral)
            # Convert to axis-aligned bounding box (x, y, width, height)
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            x = int(min(xs))
            y = int(min(ys))
            width = int(max(xs) - min(xs))
            height = int(max(ys) - min(ys))

            clamped_conf = min(1.0, max(0.0, float(confidence)))
            confidences.append(clamped_conf)
            text_parts.append(text)
            bounding_boxes.append(
                TextRegion(
                    text=text,
                    confidence=clamped_conf,
                    x=x,
                    y=y,
                    width=max(1, width),
                    height=max(1, height),
                )
            )

        if not text_parts:
            return None

        raw_text = "\n".join(text_parts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "raw_text": raw_text,
            "confidence": min(1.0, avg_confidence),
            "bounding_boxes": bounding_boxes,
        }

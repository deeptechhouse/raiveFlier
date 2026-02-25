"""EasyOCR provider for rave flier text extraction.

EasyOCR uses deep learning models and handles a wider variety of font styles
than Tesseract, making it a better fallback for the stylised typography
common on rave and electronic-music fliers.
"""

from __future__ import annotations

import io
import time

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

    Note: EasyOCR requires PyTorch (~500MB+), which is why it's excluded
    from the Docker production build (Render Starter has 512MB RAM total).
    In development, it's the best traditional OCR option for stylized text.
    """

    def __init__(self, preprocessor: ImagePreprocessor) -> None:
        self._preprocessor = preprocessor
        self._logger = get_logger(__name__)
        # Lazy initialization — the EasyOCR Reader loads ~100MB of neural
        # network weights on first use. We defer this cost until actually needed.
        self.__reader = None  # Lazy — loaded on first extract_text call

    # ------------------------------------------------------------------
    # IOCRProvider interface
    # ------------------------------------------------------------------

    async def extract_text(self, image: FlierImage) -> OCRResult:
        """Extract text from a flier image using EasyOCR with multi-pass OCR.

        Passes are processed **sequentially** via :meth:`iter_ocr_passes` so
        only one preprocessed variant is held in memory at a time.  This
        trades thread-pool parallelism for a ~75% reduction in peak
        preprocessing memory — critical on the 512 MB Render instance.
        """
        start = time.perf_counter()
        try:
            image_bytes = image.image_data
            if image_bytes is None:
                raise OCRExtractionError(
                    "No image data provided",
                    provider_name=self.get_provider_name(),
                )

            original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            reader = self._get_reader()

            all_results: list[dict | None] = []
            pass_count = 0

            # Process each preprocessing variant one at a time.  The
            # generator yields a single (name, image) pair, we run
            # EasyOCR on it, collect the result, and the image becomes
            # eligible for GC before the next variant is created.
            for pass_name, pass_image in self._preprocessor.iter_ocr_passes(original):
                pass_count += 1
                result = self._run_single_pass(reader, pass_name, pass_image)
                all_results.append(result)

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
                passes_run=pass_count,
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

    def _run_single_pass(
        self, reader: object, pass_name: str, pass_image: Image.Image,
    ) -> dict | None:
        """Run EasyOCR on one preprocessing pass and return the result.

        Wraps :meth:`_run_easyocr` with logging and error handling so the
        sequential loop in :meth:`extract_text` stays clean.
        """
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

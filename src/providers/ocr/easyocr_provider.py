"""EasyOCR provider for rave flier text extraction.

EasyOCR uses deep learning models and handles a wider variety of font styles
than Tesseract, making it a better fallback for the stylised typography
common on rave and electronic-music fliers.
"""

from __future__ import annotations

import io
import time

import numpy as np
from PIL import Image, ImageOps

from src.interfaces.ocr_provider import IOCRProvider
from src.models.flier import FlierImage, OCRResult, TextRegion
from src.utils.errors import OCRExtractionError
from src.utils.image_preprocessor import ImagePreprocessor
from src.utils.logging import get_logger


class EasyOCRProvider(IOCRProvider):
    """OCR provider backed by EasyOCR (deep-learning-based).

    The EasyOCR reader is initialised lazily on first use to avoid heavy
    model loading at import time.  Multi-pass preprocessing (standard,
    inverted, per-channel) is applied identically to the Tesseract provider.
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

            best_result: dict | None = None
            best_confidence = -1.0

            for pass_name, pass_image in passes:
                self._logger.debug("running_ocr_pass", pass_name=pass_name, provider="easyocr")
                try:
                    result = self._run_easyocr(reader, pass_image)
                    if result is not None and result["confidence"] > best_confidence:
                        best_confidence = result["confidence"]
                        best_result = result
                        best_result["pass_name"] = pass_name
                except Exception as exc:
                    self._logger.warning(
                        "ocr_pass_failed",
                        pass_name=pass_name,
                        provider="easyocr",
                        error=str(exc),
                    )

            elapsed = time.perf_counter() - start

            if best_result is None:
                raise OCRExtractionError(
                    "All OCR passes returned no usable text",
                    provider_name=self.get_provider_name(),
                )

            self._logger.info(
                "ocr_extraction_complete",
                provider="easyocr",
                best_pass=best_result.get("pass_name", "unknown"),
                confidence=round(best_confidence, 4),
                num_regions=len(best_result["bounding_boxes"]),
                processing_time=round(elapsed, 3),
            )

            return OCRResult(
                raw_text=best_result["raw_text"],
                confidence=best_result["confidence"],
                provider_used="easyocr",
                processing_time=elapsed,
                bounding_boxes=best_result["bounding_boxes"],
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
        """Build (pass_name, preprocessed_image) pairs for multi-pass OCR."""
        passes: list[tuple[str, Image.Image]] = []

        # 1. Standard contrast-enhanced binarized image
        resized = self._preprocessor.resize_for_ocr(original)
        enhanced = self._preprocessor.enhance_contrast(resized)
        binarized = self._preprocessor.binarize(enhanced)
        deskewed = self._preprocessor.deskew(binarized)
        passes.append(("standard", deskewed))

        # 2. Inverted (light text on dark background — common on rave fliers)
        inverted = ImageOps.invert(deskewed.convert("RGB"))
        passes.append(("inverted", inverted))

        # 3. Individual color channels (isolate neon text)
        channels = self._preprocessor.separate_color_channels(resized)
        channel_names = ["red", "green", "blue"]
        for ch_name, channel in zip(channel_names, channels, strict=True):
            passes.append((f"channel_{ch_name}", channel.convert("RGB")))

        return passes

    def _run_easyocr(self, reader, pass_image: Image.Image) -> dict | None:  # noqa: ANN001
        """Run EasyOCR on a single image and return results dict or None."""
        img_array = np.array(pass_image)
        results = reader.readtext(img_array)

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

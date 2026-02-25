"""Tesseract OCR provider for rave flier text extraction.

Wraps pytesseract with multi-pass preprocessing to handle the challenging
typography found on rave fliers: neon text on dark backgrounds, inverted
colors, and channel-specific content.
"""

from __future__ import annotations

import io
import time

from PIL import Image

from src.interfaces.ocr_provider import IOCRProvider
from src.models.flier import FlierImage, OCRResult, TextRegion
from src.utils.errors import OCRExtractionError
from src.utils.image_preprocessor import ImagePreprocessor
from src.utils.logging import get_logger
from src.utils.ocr_helpers import merge_pass_results
from src.utils.text_normalizer import correct_ocr_errors

# Graceful import: pytesseract is an optional dependency. On the Docker
# production image, Tesseract binary is installed via apt but the Python
# wrapper might not be present. This try/except lets the app start even
# without pytesseract — is_available() will return False, and the OCR
# service will skip this provider and use the next one in the priority chain.
try:
    import pytesseract

    _PYTESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None  # type: ignore[assignment]
    _PYTESSERACT_AVAILABLE = False


class TesseractOCRProvider(IOCRProvider):
    """OCR provider backed by Google Tesseract via pytesseract.

    Runs multiple preprocessing passes (standard, inverted, per-channel,
    CLAHE, denoised, saturation, Otsu, sparse) and merges results across
    all passes for maximum text recall.
    """

    def __init__(self, preprocessor: ImagePreprocessor) -> None:
        self._preprocessor = preprocessor
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # IOCRProvider interface
    # ------------------------------------------------------------------

    async def extract_text(self, image: FlierImage) -> OCRResult:
        """Extract text from a flier image using Tesseract with multi-pass OCR.

        Passes are processed **sequentially** via :meth:`iter_ocr_passes` so
        only one preprocessed variant is held in memory at a time.  This
        trades thread-pool parallelism (negligible on Render's 0.5 CPU) for
        a ~75% reduction in peak preprocessing memory.
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

            all_results: list[dict | None] = []
            standard_img: Image.Image | None = None
            pass_count = 0

            # Process each preprocessing variant one at a time.  The
            # generator yields a single (name, image) pair, we run
            # Tesseract on it, collect the result, and the image becomes
            # eligible for GC before the next variant is created.
            for pass_name, pass_image in self._preprocessor.iter_ocr_passes(original):
                pass_count += 1
                result = self._run_single_pass(pass_name, pass_image, config="")
                all_results.append(result)
                # Keep a reference to the standard image for the sparse pass.
                if pass_name == "standard":
                    standard_img = pass_image

            # Tesseract-specific: sparse text mode (PSM 11) reuses the
            # standard deskewed image for a final pass tuned for scattered
            # text layouts common on artistic rave fliers.
            if standard_img is not None:
                pass_count += 1
                result = self._run_single_pass("sparse", standard_img, config="--psm 11")
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
                provider="tesseract",
                passes_run=pass_count,
                confidence=round(merged["confidence"], 4),
                num_regions=len(merged["bounding_boxes"]),
                processing_time=round(elapsed, 3),
            )

            return OCRResult(
                raw_text=correct_ocr_errors(merged["raw_text"]),
                confidence=merged["confidence"],
                provider_used="tesseract",
                processing_time=elapsed,
                bounding_boxes=merged["bounding_boxes"],
            )

        except OCRExtractionError:
            raise
        except Exception as exc:
            elapsed = time.perf_counter() - start
            self._logger.error(
                "ocr_extraction_failed",
                provider="tesseract",
                error=str(exc),
                processing_time=round(elapsed, 3),
            )
            raise OCRExtractionError(
                f"Tesseract OCR failed: {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

    def get_provider_name(self) -> str:
        return "tesseract"

    def is_available(self) -> bool:
        """Check that pytesseract is installed and the Tesseract binary exists."""
        if not _PYTESSERACT_AVAILABLE:
            return False
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def supports_stylized_text(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_single_pass(
        self, pass_name: str, pass_image: Image.Image, config: str,
    ) -> dict | None:
        """Run Tesseract on one preprocessing pass and return the result.

        Wraps :meth:`_run_tesseract` with logging and error handling so the
        sequential loop in :meth:`extract_text` stays clean.
        """
        self._logger.debug("running_ocr_pass", pass_name=pass_name, provider="tesseract")
        try:
            result = self._run_tesseract(pass_image, config=config)
            if result is not None:
                result["pass_name"] = pass_name
            return result
        except Exception as exc:
            self._logger.warning(
                "ocr_pass_failed",
                pass_name=pass_name,
                provider="tesseract",
                error=str(exc),
            )
            return None

    def _run_tesseract(self, image: Image.Image, config: str = "") -> dict | None:
        """Run Tesseract on a single image and return results dict or None.

        Uses only ``image_to_data`` (not ``image_to_string``) to avoid
        running Tesseract twice on the same image.  Raw text is
        reconstructed from the word-level data output.
        """
        data = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT, config=config
        )

        bounding_boxes: list[TextRegion] = []
        confidences: list[float] = []
        text_parts: list[str] = []

        # Track block/paragraph boundaries for line-break reconstruction
        prev_block = -1
        prev_par = -1

        n_boxes = len(data["text"])
        for i in range(n_boxes):
            word = data["text"][i].strip()
            conf = int(data["conf"][i])

            # Skip empty entries and low-confidence noise (conf == -1 means invalid)
            if word and conf > 0:
                # Insert line break on block/paragraph change
                block_num = data["block_num"][i]
                par_num = data["par_num"][i]
                if text_parts and (block_num != prev_block or par_num != prev_par):
                    text_parts.append("\n")
                prev_block = block_num
                prev_par = par_num

                confidences.append(conf)
                text_parts.append(word)
                bounding_boxes.append(
                    TextRegion(
                        text=word,
                        confidence=min(1.0, conf / 100.0),
                        x=data["left"][i],
                        y=data["top"][i],
                        width=data["width"][i],
                        height=data["height"][i],
                    )
                )

        if not text_parts:
            return None

        raw_text = " ".join(text_parts).replace(" \n ", "\n").strip()
        avg_confidence = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0

        return {
            "raw_text": raw_text,
            "confidence": min(1.0, avg_confidence),
            "bounding_boxes": bounding_boxes,
        }

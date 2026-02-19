"""Tesseract OCR provider for rave flier text extraction.

Wraps pytesseract with multi-pass preprocessing to handle the challenging
typography found on rave fliers: neon text on dark backgrounds, inverted
colors, and channel-specific content.
"""

from __future__ import annotations

import io
import time

from PIL import Image, ImageOps

from src.interfaces.ocr_provider import IOCRProvider
from src.models.flier import FlierImage, OCRResult, TextRegion
from src.utils.errors import OCRExtractionError
from src.utils.image_preprocessor import ImagePreprocessor
from src.utils.logging import get_logger
from src.utils.ocr_helpers import merge_pass_results
from src.utils.text_normalizer import correct_ocr_errors

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
        """Extract text from a flier image using Tesseract with multi-pass OCR."""
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

            all_results: list[dict | None] = []

            for pass_name, pass_image, config in passes:
                self._logger.debug("running_ocr_pass", pass_name=pass_name, provider="tesseract")
                try:
                    result = self._run_tesseract(pass_image, config=config)
                    if result is not None:
                        result["pass_name"] = pass_name
                    all_results.append(result)
                except Exception as exc:
                    self._logger.warning(
                        "ocr_pass_failed",
                        pass_name=pass_name,
                        provider="tesseract",
                        error=str(exc),
                    )
                    all_results.append(None)

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
                passes_run=len(passes),
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

    def _build_passes(
        self, original: Image.Image
    ) -> list[tuple[str, Image.Image, str]]:
        """Build (pass_name, preprocessed_image, tesseract_config) triples."""
        pp = self._preprocessor
        passes: list[tuple[str, Image.Image, str]] = []

        # Shared base: resized (with upscaling for small images)
        resized = pp.resize_for_ocr(original)

        # 1. Standard contrast-enhanced binarized image
        enhanced = pp.enhance_contrast(resized)
        binarized = pp.binarize(enhanced)
        deskewed = pp.deskew(binarized)
        passes.append(("standard", deskewed, ""))

        # 2. Inverted (light text on dark background — common on rave fliers)
        inverted = ImageOps.invert(deskewed.convert("RGB"))
        passes.append(("inverted", inverted, ""))

        # 3. Individual color channels (isolate neon text)
        channels = pp.separate_color_channels(resized)
        channel_names = ["red", "green", "blue"]
        for ch_name, channel in zip(channel_names, channels, strict=True):
            passes.append((f"channel_{ch_name}", channel.convert("RGB"), ""))

        # 4. CLAHE — localized adaptive contrast (handles uneven lighting)
        clahe_enhanced = pp.enhance_contrast_clahe(resized)
        clahe_binarized = pp.binarize(clahe_enhanced)
        clahe_deskewed = pp.deskew(clahe_binarized)
        passes.append(("clahe", clahe_deskewed, ""))

        # 5. Denoised — remove camera sensor noise before OCR
        denoised = pp.denoise(resized)
        denoised_enhanced = pp.enhance_contrast(denoised)
        denoised_binarized = pp.binarize(denoised_enhanced)
        denoised_deskewed = pp.deskew(denoised_binarized)
        passes.append(("denoised", denoised_deskewed, ""))

        # 6. HSV saturation channel — isolates neon-colored text
        saturation = pp.extract_saturation_channel(resized)
        sat_binarized = pp.binarize(saturation.convert("RGB"))
        passes.append(("saturation", sat_binarized, ""))

        # 7. Otsu binarization — automatic global threshold
        otsu_enhanced = pp.enhance_contrast(resized)
        otsu_binarized = pp.binarize_otsu(otsu_enhanced)
        otsu_deskewed = pp.deskew(otsu_binarized)
        passes.append(("otsu", otsu_deskewed, ""))

        # 8. Sparse text mode (PSM 11) — finds scattered text at any position
        passes.append(("sparse", deskewed, "--psm 11"))

        return passes

    def _run_tesseract(self, image: Image.Image, config: str = "") -> dict | None:
        """Run Tesseract on a single image and return results dict or None."""
        data = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT, config=config
        )
        raw_text = pytesseract.image_to_string(image, config=config).strip()

        if not raw_text:
            return None

        bounding_boxes: list[TextRegion] = []
        confidences: list[float] = []

        n_boxes = len(data["text"])
        for i in range(n_boxes):
            word = data["text"][i].strip()
            conf = int(data["conf"][i])

            # Skip empty entries and low-confidence noise (conf == -1 means invalid)
            if word and conf > 0:
                confidences.append(conf)
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

        avg_confidence = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0

        return {
            "raw_text": raw_text,
            "confidence": min(1.0, avg_confidence),
            "bounding_boxes": bounding_boxes,
        }

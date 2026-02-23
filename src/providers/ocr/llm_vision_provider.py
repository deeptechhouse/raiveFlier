"""LLM Vision OCR provider for rave flier text extraction.

Uses a vision-capable LLM to extract and interpret text from rave flier
images.  This is the primary OCR strategy because LLMs excel at reading
the heavily stylised, distorted, and layered typography found on rave
and electronic-music fliers — far surpassing traditional OCR engines.
"""

from __future__ import annotations

import io
import re
import time

from PIL import Image

from src.interfaces.llm_provider import ILLMProvider
from src.interfaces.ocr_provider import IOCRProvider
from src.models.flier import FlierImage, OCRResult, TextRegion
from src.utils.errors import OCRExtractionError
from src.utils.image_preprocessor import ImagePreprocessor
from src.utils.logging import get_logger

# Minimum largest-dimension threshold below which we send an enhanced image
# instead of the raw original to the vision model.
_LOW_QUALITY_THRESHOLD = 800

_VISION_PROMPT = """\
You are analyzing a rave/electronic music event flier image.
Extract ALL readable text from this flier. The text may be heavily stylized,
distorted, use unconventional fonts, or be layered over graphics.

## Rave Flier Visual Conventions
- HEADLINER names are the LARGEST text, usually top-center or center of the flier.
- Supporting DJs/artists are listed below the headliner in smaller type.
- DJ names are often fully capitalized: "CARL COX", "RICHIE HAWTIN", "JEFF MILLS".
- Names may include numbers, symbols, or unconventional spelling: "3D", "2Bad",
  "A.N.A.L.", "DJ ?", "LTJ Bukem".
- Common separators between artists: b2b, B2B, vs, VS, &, feat., ft., w/, +
- Dates appear in many formats: "SAT 15th MARCH", "03.15.97", "15/03/1997",
  "15|03|97", "SATURDAY MARCH 15", weekday abbreviations.
- Venue/location is typically at the bottom, often with a street address or city.
- Promoter/event series names are in small text at the top or bottom edges.
- Price info appears as: "$10", "FREE B4 11PM", "10 DM", "£5", "PWYC",
  "$10 ADV / $15 DOOR".
- Text can appear at ANY orientation — rotated 90°, diagonal, curved, or
  following a path. Read text at ALL angles.
- Neon-colored text (pink, cyan, green, yellow) on black backgrounds is standard.
  Read ALL colored text even if it blends with graphics.
- Look for small text in corners, edges, and borders — these often contain
  promoter info, phone numbers, or web addresses.

## What to Extract
1. Artist/DJ names (usually the largest text, often at top)
2. Event date and time
3. Venue/location name and address
4. Promoter or organization name (often smaller text)
5. Ticket price or cover charge
6. Genre tags or event series name
7. Any other readable text (URLs, phone numbers, age restrictions, etc.)

Return your findings as structured text, one item per line, in this format:
ARTIST: [name]
DATE: [date as written on flier]
VENUE: [venue name]
LOCATION: [address or city]
PROMOTER: [promoter/org name]
PRICE: [price if visible]
GENRE: [genre tags]
OTHER: [any other text]

List EACH artist on a separate ARTIST: line. Do not combine multiple artists.
If text is partially illegible, include your best guess with [?] suffix.
If you cannot read certain text at all, note: UNREADABLE: [description of where/what]"""

# Pattern that matches labelled lines produced by the prompt format above.
_FIELD_PATTERN = re.compile(
    r"^(ARTIST|DATE|VENUE|LOCATION|PROMOTER|PRICE|GENRE|OTHER|UNREADABLE):\s*(.+)$",
    re.MULTILINE,
)

# Uncertainty markers placed by the LLM when text is partially illegible.
_UNCERTAINTY_MARKER = re.compile(r"\[\?\]")


class LLMVisionOCRProvider(IOCRProvider):
    """OCR provider that delegates to a vision-capable LLM.

    The LLM is prompted with a rave-flier-specific instruction set and
    returns structured text that is parsed into an :class:`OCRResult`.
    Confidence is derived from the proportion of ``[?]`` uncertainty
    markers in the response.

    This is the PRIMARY OCR provider because LLMs understand context —
    they can infer "DJ SHADOW" even when the text is heavily distorted,
    because they know DJ Shadow is a real artist name. Traditional OCR
    engines like Tesseract have no such contextual knowledge.
    """

    def __init__(
        self,
        llm_provider: ILLMProvider,
        preprocessor: ImagePreprocessor | None = None,
    ) -> None:
        # This provider delegates to any ILLMProvider that supports vision.
        # It's an "adapter of an adapter" — composing two provider interfaces.
        self._llm_provider = llm_provider
        # Optional image preprocessor for enhancing low-quality images
        # before sending to the vision model.
        self._preprocessor = preprocessor
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # IOCRProvider interface
    # ------------------------------------------------------------------

    async def extract_text(self, image: FlierImage) -> OCRResult:
        """Extract text from a flier image using the LLM's vision capability."""
        start = time.perf_counter()
        try:
            image_bytes = image.image_data
            if image_bytes is None:
                raise OCRExtractionError(
                    "No image data provided",
                    provider_name=self.get_provider_name(),
                )

            # Enhance low-quality images before sending to the vision model
            send_bytes = self._prepare_image_bytes(image_bytes)

            self._logger.debug(
                "running_llm_vision_ocr",
                provider=self.get_provider_name(),
                llm=self._llm_provider.get_provider_name(),
            )

            raw_response = await self._llm_provider.vision_extract(
                image_bytes=send_bytes,
                prompt=_VISION_PROMPT,
            )

            bounding_boxes, raw_text = self._parse_response(raw_response)
            confidence = self._compute_confidence(raw_response)
            elapsed = time.perf_counter() - start

            self._logger.info(
                "ocr_extraction_complete",
                provider="llm_vision",
                confidence=round(confidence, 4),
                num_regions=len(bounding_boxes),
                processing_time=round(elapsed, 3),
            )

            return OCRResult(
                raw_text=raw_text,
                confidence=confidence,
                provider_used="llm_vision",
                processing_time=elapsed,
                bounding_boxes=bounding_boxes,
            )

        except OCRExtractionError:
            raise
        except Exception as exc:
            elapsed = time.perf_counter() - start
            self._logger.error(
                "ocr_extraction_failed",
                provider="llm_vision",
                error=str(exc),
                processing_time=round(elapsed, 3),
            )
            raise OCRExtractionError(
                f"LLM Vision OCR failed: {exc}",
                provider_name=self.get_provider_name(),
            ) from exc

    def get_provider_name(self) -> str:
        return "llm_vision"

    def is_available(self) -> bool:
        """Available only if the underlying LLM is reachable and supports vision."""
        return self._llm_provider.is_available() and self._llm_provider.supports_vision()

    def supports_stylized_text(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_image_bytes(self, image_bytes: bytes) -> bytes:
        """Enhance low-quality images before sending to the vision model.

        If the preprocessor is available and the image is below
        *_LOW_QUALITY_THRESHOLD* pixels on its largest dimension, apply
        CLAHE contrast enhancement and upscaling so the LLM has better
        text to read.  High-quality originals are returned unmodified.
        """
        if self._preprocessor is None:
            return image_bytes

        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            largest = max(img.size)
            if largest >= _LOW_QUALITY_THRESHOLD:
                return image_bytes

            self._logger.debug(
                "enhancing_low_quality_image",
                original_size=img.size,
                largest_dim=largest,
            )
            img = self._preprocessor.resize_for_ocr(img)
            img = self._preprocessor.enhance_contrast_clahe(img)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            # If enhancement fails for any reason, fall back to original
            return image_bytes

    @staticmethod
    def _parse_response(response: str) -> tuple[list[TextRegion], str]:
        """Parse structured LLM output into TextRegion objects and raw text.

        Returns a tuple of (bounding_boxes, raw_text).  Bounding boxes use
        ``x=0, y=0, width=0, height=0`` because the LLM does not provide
        spatial coordinates.
        """
        matches = _FIELD_PATTERN.findall(response)
        bounding_boxes: list[TextRegion] = []
        text_lines: list[str] = []

        for label, value in matches:
            value = value.strip()
            if not value:
                continue
            # Strip uncertainty markers for the confidence calc but keep
            # the original text (with [?]) in the TextRegion so downstream
            # consumers can see what was uncertain.
            region_confidence = 1.0 if not _UNCERTAINTY_MARKER.search(value) else 0.5
            bounding_boxes.append(
                TextRegion(
                    text=f"{label}: {value}",
                    confidence=region_confidence,
                    x=0,
                    y=0,
                    width=0,
                    height=0,
                )
            )
            text_lines.append(f"{label}: {value}")

        raw_text = "\n".join(text_lines) if text_lines else response.strip()
        return bounding_boxes, raw_text

    @staticmethod
    def _compute_confidence(response: str) -> float:
        """Derive an overall confidence score from uncertainty markers.

        The confidence starts at 0.95 (LLM vision is inherently strong for
        stylised text) and is penalised by 0.05 for each ``[?]`` marker,
        floored at 0.3.  An empty response yields 0.0.
        """
        if not response or not response.strip():
            return 0.0

        uncertain_count = len(_UNCERTAINTY_MARKER.findall(response))
        confidence = max(0.3, 0.95 - (uncertain_count * 0.05))
        return round(min(1.0, confidence), 4)

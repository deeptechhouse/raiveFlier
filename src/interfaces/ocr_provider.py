"""Abstract base class for OCR service providers.

Defines the contract for any OCR engine used to extract text from rave flier
images.  Implementations may wrap Tesseract, Google Vision, AWS Textract, or
any other OCR backend.  The adapter pattern (CLAUDE.md Section 6) ensures that
swapping providers requires only a new concrete class — no call-site changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.flier import FlierImage, OCRResult


# Concrete implementations: LLMVisionOCRProvider, EasyOCRProvider, TesseractOCRProvider
# Located in: src/providers/ocr/
# The OCR service (src/services/ocr_service.py) tries providers in priority order
# defined in config/config.yaml and uses the first one that meets the confidence threshold.
class IOCRProvider(ABC):
    """Contract for OCR services that extract text from flier images.

    Every concrete provider must be able to:
    * Accept a ``FlierImage`` and return a structured ``OCRResult``.
    * Report its availability (credentials present, service reachable).
    * Declare whether it can handle the stylised/distorted typography
      commonly found on rave fliers.
    """

    @abstractmethod
    async def extract_text(self, image: FlierImage) -> OCRResult:
        """Run OCR on *image* and return the extraction result.

        Parameters
        ----------
        image:
            The flier image to process.  ``image.image_data`` contains the
            raw bytes; ``image.content_type`` indicates the format.

        Returns
        -------
        OCRResult
            Structured result including raw text, per-region bounding boxes,
            overall confidence, and processing time.

        Raises
        ------
        src.core.errors.OCRError
            If the OCR engine fails or returns an unusable result.
        """

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return a human-readable identifier for this OCR provider.

        Example return values: ``"tesseract"``, ``"google-vision"``.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if the provider is configured and reachable.

        Implementations should check for required credentials, binaries, or
        network connectivity without performing a full OCR pass.
        """

    @abstractmethod
    def supports_stylized_text(self) -> bool:
        """Return ``True`` if this provider handles distorted rave-flier fonts.

        Stylised text includes warped, stretched, neon-outlined, or heavily
        stylised typography typical of rave and electronic-music fliers.
        Providers that lack specialised handling should return ``False`` so
        the orchestrator can prefer a more capable backend when available.
        """

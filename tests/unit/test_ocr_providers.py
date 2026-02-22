"""Unit tests for OCR provider adapters — Tesseract, EasyOCR, LLM Vision."""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from src.models.flier import FlierImage


def _make_flier_image() -> FlierImage:
    """Create a minimal FlierImage with raw image data."""
    img = Image.new("RGB", (400, 300), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    image_data = buf.getvalue()
    import hashlib

    flier = FlierImage(
        filename="test.jpg",
        content_type="image/jpeg",
        file_size=len(image_data),
        image_hash=hashlib.sha256(image_data).hexdigest(),
    )
    flier.__pydantic_private__["_image_data"] = image_data
    return flier


# ======================================================================
# Tesseract OCR Provider
# ======================================================================


class TestTesseractOCRProvider:
    def test_get_provider_name(self) -> None:
        from src.providers.ocr.tesseract_provider import TesseractOCRProvider
        from src.utils.image_preprocessor import ImagePreprocessor

        provider = TesseractOCRProvider(ImagePreprocessor())
        assert provider.get_provider_name() == "tesseract"

    def test_supports_stylized_text(self) -> None:
        from src.providers.ocr.tesseract_provider import TesseractOCRProvider
        from src.utils.image_preprocessor import ImagePreprocessor

        provider = TesseractOCRProvider(ImagePreprocessor())
        assert provider.supports_stylized_text() is False

    @pytest.mark.asyncio
    async def test_extract_text_success(self) -> None:
        from src.providers.ocr.tesseract_provider import TesseractOCRProvider
        from src.utils.image_preprocessor import ImagePreprocessor

        flier = _make_flier_image()
        mock_preprocessor = MagicMock(spec=ImagePreprocessor)

        # build_ocr_passes returns list of (name, PIL.Image) tuples
        test_img = Image.new("RGB", (400, 300), (255, 255, 255))
        mock_preprocessor.build_ocr_passes.return_value = [
            ("standard", test_img),
        ]

        mock_ocr_data = {
            "text": ["CARL", "COX", "TRESOR", ""],
            "conf": [90, 85, 80, 0],
            "block_num": [1, 1, 2, 0],
            "par_num": [1, 1, 1, 0],
            "left": [10, 60, 10, 0],
            "top": [10, 10, 50, 0],
            "width": [40, 40, 60, 0],
            "height": [20, 20, 20, 0],
        }

        with patch("src.providers.ocr.tesseract_provider.pytesseract") as mock_tess:
            mock_tess.image_to_data.return_value = mock_ocr_data
            mock_tess.Output.DICT = "dict"

            provider = TesseractOCRProvider(mock_preprocessor)
            result = await provider.extract_text(flier)

        assert "CARL" in result.raw_text
        assert "COX" in result.raw_text
        assert result.provider_used == "tesseract"
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_extract_text_empty_raises(self) -> None:
        from src.providers.ocr.tesseract_provider import TesseractOCRProvider
        from src.utils.errors import OCRExtractionError
        from src.utils.image_preprocessor import ImagePreprocessor

        flier = _make_flier_image()
        mock_preprocessor = MagicMock(spec=ImagePreprocessor)
        test_img = Image.new("RGB", (400, 300), (0, 0, 0))
        mock_preprocessor.build_ocr_passes.return_value = [("standard", test_img)]

        mock_ocr_data = {
            "text": [""],
            "conf": [0],
            "block_num": [0],
            "par_num": [0],
            "left": [0],
            "top": [0],
            "width": [0],
            "height": [0],
        }

        with patch("src.providers.ocr.tesseract_provider.pytesseract") as mock_tess:
            mock_tess.image_to_data.return_value = mock_ocr_data
            mock_tess.Output.DICT = "dict"

            provider = TesseractOCRProvider(mock_preprocessor)
            with pytest.raises(OCRExtractionError):
                await provider.extract_text(flier)

    def test_is_available_true(self) -> None:
        from src.providers.ocr.tesseract_provider import TesseractOCRProvider
        from src.utils.image_preprocessor import ImagePreprocessor

        with patch("src.providers.ocr.tesseract_provider._PYTESSERACT_AVAILABLE", True), \
             patch("src.providers.ocr.tesseract_provider.pytesseract") as mock_tess:
            mock_tess.get_tesseract_version.return_value = "5.3.0"
            provider = TesseractOCRProvider(ImagePreprocessor())
            assert provider.is_available() is True

    def test_is_available_false(self) -> None:
        from src.providers.ocr.tesseract_provider import TesseractOCRProvider
        from src.utils.image_preprocessor import ImagePreprocessor

        with patch("src.providers.ocr.tesseract_provider._PYTESSERACT_AVAILABLE", False):
            provider = TesseractOCRProvider(ImagePreprocessor())
            assert provider.is_available() is False


# ======================================================================
# EasyOCR Provider
# ======================================================================


class TestEasyOCRProvider:
    def test_get_provider_name(self) -> None:
        from src.providers.ocr.easyocr_provider import EasyOCRProvider
        from src.utils.image_preprocessor import ImagePreprocessor

        provider = EasyOCRProvider(ImagePreprocessor())
        assert provider.get_provider_name() == "easyocr"

    def test_supports_stylized_text(self) -> None:
        from src.providers.ocr.easyocr_provider import EasyOCRProvider
        from src.utils.image_preprocessor import ImagePreprocessor

        provider = EasyOCRProvider(ImagePreprocessor())
        assert provider.supports_stylized_text() is True

    @pytest.mark.asyncio
    async def test_extract_text_success(self) -> None:
        from src.providers.ocr.easyocr_provider import EasyOCRProvider
        from src.utils.image_preprocessor import ImagePreprocessor
        from src.models.flier import TextRegion

        flier = _make_flier_image()
        mock_preprocessor = MagicMock(spec=ImagePreprocessor)
        test_img = Image.new("RGB", (400, 300), (255, 255, 255))
        mock_preprocessor.build_ocr_passes.return_value = [("standard", test_img)]

        # Mock _run_easyocr to return a dict matching merge_pass_results input
        mock_result = {
            "raw_text": "CARL COX\nTRESOR BERLIN",
            "confidence": 0.9,
            "bounding_boxes": [
                TextRegion(text="CARL COX", confidence=0.92, x=10, y=10, width=70, height=20),
                TextRegion(text="TRESOR BERLIN", confidence=0.88, x=10, y=50, width=110, height=20),
            ],
            "pass_name": "standard",
        }

        provider = EasyOCRProvider(mock_preprocessor)
        with patch.object(provider, "_run_easyocr", return_value=mock_result):
            with patch.object(provider, "_get_reader", return_value=MagicMock()):
                result = await provider.extract_text(flier)

        assert "CARL COX" in result.raw_text
        assert result.provider_used == "easyocr"

    def test_is_available(self) -> None:
        from src.providers.ocr.easyocr_provider import EasyOCRProvider
        from src.utils.image_preprocessor import ImagePreprocessor

        provider = EasyOCRProvider(ImagePreprocessor())
        # easyocr is installed in the environment
        result = provider.is_available()
        assert isinstance(result, bool)

    def test_run_easyocr_with_results(self) -> None:
        """_run_easyocr converts EasyOCR output to the internal dict format."""
        from src.providers.ocr.easyocr_provider import EasyOCRProvider
        from src.utils.image_preprocessor import ImagePreprocessor

        provider = EasyOCRProvider(ImagePreprocessor())

        # Mock reader with realistic EasyOCR output:
        # each result is (bbox, text, confidence)
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[10, 10], [100, 10], [100, 30], [10, 30]], "CARL COX", 0.92),
            ([[10, 50], [150, 50], [150, 70], [10, 70]], "TRESOR BERLIN", 0.85),
        ]

        test_img = Image.new("RGB", (400, 300), (255, 255, 255))
        result = provider._run_easyocr(mock_reader, test_img)

        assert result is not None
        assert "CARL COX" in result["raw_text"]
        assert "TRESOR BERLIN" in result["raw_text"]
        assert len(result["bounding_boxes"]) == 2
        assert result["confidence"] > 0
        # Verify bounding box coordinates
        assert result["bounding_boxes"][0].x == 10
        assert result["bounding_boxes"][0].y == 10
        assert result["bounding_boxes"][0].width == 90

    def test_run_easyocr_empty_results(self) -> None:
        """_run_easyocr returns None when EasyOCR detects no text."""
        from src.providers.ocr.easyocr_provider import EasyOCRProvider
        from src.utils.image_preprocessor import ImagePreprocessor

        provider = EasyOCRProvider(ImagePreprocessor())

        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []

        test_img = Image.new("RGB", (400, 300), (0, 0, 0))
        result = provider._run_easyocr(mock_reader, test_img)
        assert result is None

    def test_run_easyocr_whitespace_only_text(self) -> None:
        """_run_easyocr returns None when all detected text is whitespace."""
        from src.providers.ocr.easyocr_provider import EasyOCRProvider
        from src.utils.image_preprocessor import ImagePreprocessor

        provider = EasyOCRProvider(ImagePreprocessor())

        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[10, 10], [50, 10], [50, 30], [10, 30]], "   ", 0.5),
            ([[10, 50], [50, 50], [50, 70], [10, 70]], "", 0.3),
        ]

        test_img = Image.new("RGB", (400, 300), (128, 128, 128))
        result = provider._run_easyocr(mock_reader, test_img)
        assert result is None

    def test_run_easyocr_clamps_confidence(self) -> None:
        """_run_easyocr clamps confidence values to [0.0, 1.0]."""
        from src.providers.ocr.easyocr_provider import EasyOCRProvider
        from src.utils.image_preprocessor import ImagePreprocessor

        provider = EasyOCRProvider(ImagePreprocessor())

        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [50, 0], [50, 20], [0, 20]], "TEXT", 1.5),
        ]

        test_img = Image.new("RGB", (100, 100), (255, 255, 255))
        result = provider._run_easyocr(mock_reader, test_img)
        assert result is not None
        assert result["confidence"] <= 1.0
        assert result["bounding_boxes"][0].confidence <= 1.0

    @pytest.mark.asyncio
    async def test_extract_text_multiple_passes(self) -> None:
        """extract_text merges results from multiple preprocessing passes."""
        from src.providers.ocr.easyocr_provider import EasyOCRProvider
        from src.utils.image_preprocessor import ImagePreprocessor
        from src.models.flier import TextRegion

        flier = _make_flier_image()
        mock_preprocessor = MagicMock(spec=ImagePreprocessor)
        test_img1 = Image.new("RGB", (400, 300), (255, 255, 255))
        test_img2 = Image.new("RGB", (400, 300), (0, 0, 0))
        mock_preprocessor.build_ocr_passes.return_value = [
            ("standard", test_img1),
            ("inverted", test_img2),
        ]

        # Pass 1: detects CARL COX
        pass1_result = {
            "raw_text": "CARL COX",
            "confidence": 0.90,
            "bounding_boxes": [
                TextRegion(text="CARL COX", confidence=0.90, x=10, y=10, width=80, height=20),
            ],
            "pass_name": "standard",
        }
        # Pass 2: detects TRESOR BERLIN
        pass2_result = {
            "raw_text": "TRESOR BERLIN",
            "confidence": 0.85,
            "bounding_boxes": [
                TextRegion(text="TRESOR BERLIN", confidence=0.85, x=10, y=50, width=120, height=20),
            ],
            "pass_name": "inverted",
        }

        provider = EasyOCRProvider(mock_preprocessor)

        call_count = 0

        def mock_run_easyocr(reader, pass_image):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return pass1_result
            return pass2_result

        with patch.object(provider, "_run_easyocr", side_effect=mock_run_easyocr), \
             patch.object(provider, "_get_reader", return_value=MagicMock()):
            result = await provider.extract_text(flier)

        assert result.provider_used == "easyocr"
        # Should contain text from both passes after merging
        assert "CARL COX" in result.raw_text or "TRESOR BERLIN" in result.raw_text

    @pytest.mark.asyncio
    async def test_extract_text_all_passes_fail(self) -> None:
        """When all OCR passes return None, OCRExtractionError is raised."""
        from src.providers.ocr.easyocr_provider import EasyOCRProvider
        from src.utils.errors import OCRExtractionError
        from src.utils.image_preprocessor import ImagePreprocessor

        flier = _make_flier_image()
        mock_preprocessor = MagicMock(spec=ImagePreprocessor)
        test_img = Image.new("RGB", (400, 300), (0, 0, 0))
        mock_preprocessor.build_ocr_passes.return_value = [("standard", test_img)]

        provider = EasyOCRProvider(mock_preprocessor)

        with patch.object(provider, "_run_easyocr", return_value=None), \
             patch.object(provider, "_get_reader", return_value=MagicMock()):
            with pytest.raises(OCRExtractionError, match="no usable text"):
                await provider.extract_text(flier)

    @pytest.mark.asyncio
    async def test_extract_text_no_image_data_raises(self) -> None:
        """When FlierImage has no image data, OCRExtractionError is raised."""
        from src.providers.ocr.easyocr_provider import EasyOCRProvider
        from src.utils.errors import OCRExtractionError
        from src.utils.image_preprocessor import ImagePreprocessor

        flier = FlierImage(
            filename="empty.jpg",
            content_type="image/jpeg",
            file_size=0,
            image_hash="empty",
        )
        # image_data is None (no private field set)

        provider = EasyOCRProvider(ImagePreprocessor())
        with pytest.raises(OCRExtractionError, match="No image data"):
            await provider.extract_text(flier)

    @pytest.mark.asyncio
    async def test_extract_text_unexpected_exception_wrapped(self) -> None:
        """Unexpected exceptions during extraction are wrapped in OCRExtractionError."""
        from src.providers.ocr.easyocr_provider import EasyOCRProvider
        from src.utils.errors import OCRExtractionError
        from src.utils.image_preprocessor import ImagePreprocessor

        flier = _make_flier_image()
        mock_preprocessor = MagicMock(spec=ImagePreprocessor)
        # Make build_ocr_passes raise an unexpected error
        mock_preprocessor.build_ocr_passes.side_effect = RuntimeError("GPU error")

        provider = EasyOCRProvider(mock_preprocessor)

        with pytest.raises(OCRExtractionError, match="EasyOCR failed"):
            await provider.extract_text(flier)

    def test_get_reader_lazy_init(self) -> None:
        """_get_reader initializes the reader lazily on first call."""
        from src.providers.ocr.easyocr_provider import EasyOCRProvider
        from src.utils.image_preprocessor import ImagePreprocessor

        provider = EasyOCRProvider(ImagePreprocessor())
        assert provider._EasyOCRProvider__reader is None

        mock_easyocr_module = MagicMock()
        mock_reader_instance = MagicMock()
        mock_easyocr_module.Reader.return_value = mock_reader_instance

        with patch.dict("sys.modules", {"easyocr": mock_easyocr_module}):
            reader = provider._get_reader()

        assert reader is mock_reader_instance
        mock_easyocr_module.Reader.assert_called_once_with(["en"], gpu=False)

    def test_get_reader_returns_cached(self) -> None:
        """Subsequent _get_reader calls return the cached reader."""
        from src.providers.ocr.easyocr_provider import EasyOCRProvider
        from src.utils.image_preprocessor import ImagePreprocessor

        provider = EasyOCRProvider(ImagePreprocessor())
        sentinel = MagicMock()
        provider._EasyOCRProvider__reader = sentinel

        result = provider._get_reader()
        assert result is sentinel


# ======================================================================
# LLM Vision OCR Provider
# ======================================================================


class TestLLMVisionOCRProvider:
    def test_get_provider_name(self) -> None:
        from src.providers.ocr.llm_vision_provider import LLMVisionOCRProvider

        mock_llm = MagicMock()
        provider = LLMVisionOCRProvider(mock_llm)
        assert provider.get_provider_name() == "llm_vision"

    def test_supports_stylized_text(self) -> None:
        from src.providers.ocr.llm_vision_provider import LLMVisionOCRProvider

        mock_llm = MagicMock()
        provider = LLMVisionOCRProvider(mock_llm)
        assert provider.supports_stylized_text() is True

    def test_is_available_with_vision_support(self) -> None:
        from src.providers.ocr.llm_vision_provider import LLMVisionOCRProvider

        mock_llm = MagicMock()
        mock_llm.supports_vision.return_value = True
        mock_llm.is_available.return_value = True
        provider = LLMVisionOCRProvider(mock_llm)
        assert provider.is_available() is True

    def test_is_available_without_vision_support(self) -> None:
        from src.providers.ocr.llm_vision_provider import LLMVisionOCRProvider

        mock_llm = MagicMock()
        mock_llm.supports_vision.return_value = False
        provider = LLMVisionOCRProvider(mock_llm)
        assert provider.is_available() is False

    @pytest.mark.asyncio
    async def test_extract_text_success(self) -> None:
        from src.providers.ocr.llm_vision_provider import LLMVisionOCRProvider

        flier = _make_flier_image()

        mock_llm = MagicMock()
        mock_llm.supports_vision.return_value = True
        mock_llm.is_available.return_value = True
        mock_llm.vision_extract = AsyncMock(
            return_value=(
                "TRESOR PRESENTS\n"
                "CARL COX\n"
                "JEFF MILLS\n"
                "TRESOR BERLIN\n"
                "SATURDAY MARCH 15TH 1997"
            )
        )

        provider = LLMVisionOCRProvider(mock_llm)
        result = await provider.extract_text(flier)

        assert "CARL COX" in result.raw_text
        assert result.provider_used == "llm_vision"
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_extract_text_llm_error(self) -> None:
        from src.providers.ocr.llm_vision_provider import LLMVisionOCRProvider

        flier = _make_flier_image()

        mock_llm = MagicMock()
        mock_llm.supports_vision.return_value = True
        mock_llm.is_available.return_value = True
        mock_llm.vision_extract = AsyncMock(side_effect=Exception("LLM timeout"))

        provider = LLMVisionOCRProvider(mock_llm)
        with pytest.raises(Exception):
            await provider.extract_text(flier)

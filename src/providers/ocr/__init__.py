"""OCR provider implementations for rave flier text extraction.

Three implementations of IOCRProvider, tried in priority order by ocr_service.py:
    1. LLMVisionOCRProvider — sends the image to a vision LLM (GPT-4o / Claude).
       Best at reading stylized rave typography. Primary provider in production.
    2. EasyOCRProvider — deep-learning OCR (PyTorch). Better than Tesseract at
       stylized text, but requires ~500MB+ RAM for PyTorch. Excluded from Docker
       build to fit Render's 512MB Starter plan.
    3. TesseractOCRProvider — traditional OCR (Google Tesseract). Lightweight
       and free, but struggles with distorted/neon fonts. Last-resort fallback.

Each provider uses multi-pass preprocessing (standard, inverted, per-channel,
CLAHE contrast enhancement) to maximize text extraction from challenging images.
"""

from src.providers.ocr.easyocr_provider import EasyOCRProvider
from src.providers.ocr.llm_vision_provider import LLMVisionOCRProvider
from src.providers.ocr.tesseract_provider import TesseractOCRProvider

__all__ = ["EasyOCRProvider", "LLMVisionOCRProvider", "TesseractOCRProvider"]

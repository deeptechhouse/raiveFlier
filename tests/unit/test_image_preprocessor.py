"""Unit tests for ImagePreprocessor — including new OCR accuracy methods."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from src.utils.image_preprocessor import ImagePreprocessor


@pytest.fixture()
def preprocessor() -> ImagePreprocessor:
    return ImagePreprocessor()


def _make_image(width: int = 200, height: int = 150, color: tuple = (128, 128, 128)) -> Image.Image:
    """Create a simple test image."""
    return Image.new("RGB", (width, height), color)


def _make_neon_on_black(width: int = 400, height: int = 300) -> Image.Image:
    """Create a dark image with bright neon-colored regions."""
    img = Image.new("RGB", (width, height), (10, 10, 10))
    arr = np.array(img)
    # Pink neon band
    arr[50:80, 50:350] = [255, 0, 200]
    # Cyan neon band
    arr[120:150, 50:350] = [0, 255, 255]
    return Image.fromarray(arr)


# ======================================================================
# resize_for_ocr (upscaling)
# ======================================================================


class TestResizeForOCR:
    def test_upscales_small_image(self, preprocessor: ImagePreprocessor) -> None:
        small = _make_image(400, 300)
        result = preprocessor.resize_for_ocr(small, min_dim=1200)
        assert max(result.size) >= 1200

    def test_downscales_large_image(self, preprocessor: ImagePreprocessor) -> None:
        large = _make_image(3000, 2000)
        result = preprocessor.resize_for_ocr(large, max_dim=2000)
        assert max(result.size) <= 2000

    def test_preserves_image_in_bounds(self, preprocessor: ImagePreprocessor) -> None:
        medium = _make_image(1500, 1000)
        result = preprocessor.resize_for_ocr(medium, min_dim=1200, max_dim=2000)
        assert result.size == medium.size

    def test_preserves_aspect_ratio(self, preprocessor: ImagePreprocessor) -> None:
        small = _make_image(600, 300)
        result = preprocessor.resize_for_ocr(small, min_dim=1200)
        original_ratio = 600 / 300
        result_ratio = result.size[0] / result.size[1]
        assert abs(original_ratio - result_ratio) < 0.01


# ======================================================================
# enhance_contrast_clahe
# ======================================================================


class TestEnhanceContrastCLAHE:
    def test_returns_rgb_image(self, preprocessor: ImagePreprocessor) -> None:
        img = _make_image()
        result = preprocessor.enhance_contrast_clahe(img)
        assert result.mode == "RGB"

    def test_preserves_dimensions(self, preprocessor: ImagePreprocessor) -> None:
        img = _make_image(300, 200)
        result = preprocessor.enhance_contrast_clahe(img)
        assert result.size == (300, 200)

    def test_modifies_pixel_values(self, preprocessor: ImagePreprocessor) -> None:
        img = _make_neon_on_black()
        result = preprocessor.enhance_contrast_clahe(img)
        orig_arr = np.array(img)
        result_arr = np.array(result)
        assert not np.array_equal(orig_arr, result_arr)


# ======================================================================
# denoise
# ======================================================================


class TestDenoise:
    def test_returns_rgb_image(self, preprocessor: ImagePreprocessor) -> None:
        img = _make_image()
        result = preprocessor.denoise(img)
        assert result.mode == "RGB"

    def test_preserves_dimensions(self, preprocessor: ImagePreprocessor) -> None:
        img = _make_image(300, 200)
        result = preprocessor.denoise(img)
        assert result.size == (300, 200)

    def test_reduces_noise(self, preprocessor: ImagePreprocessor) -> None:
        # Create noisy image
        rng = np.random.default_rng(42)
        noisy = rng.integers(0, 256, (200, 300, 3), dtype=np.uint8)
        img = Image.fromarray(noisy)
        result = preprocessor.denoise(img)
        result_arr = np.array(result)
        # Denoised image should have lower variance (smoother)
        assert result_arr.var() < noisy.var()


# ======================================================================
# extract_saturation_channel
# ======================================================================


class TestExtractSaturationChannel:
    def test_returns_grayscale_image(self, preprocessor: ImagePreprocessor) -> None:
        img = _make_neon_on_black()
        result = preprocessor.extract_saturation_channel(img)
        assert result.mode == "L"

    def test_preserves_dimensions(self, preprocessor: ImagePreprocessor) -> None:
        img = _make_neon_on_black(400, 300)
        result = preprocessor.extract_saturation_channel(img)
        assert result.size == (400, 300)

    def test_neon_regions_are_bright(self, preprocessor: ImagePreprocessor) -> None:
        img = _make_neon_on_black()
        result = preprocessor.extract_saturation_channel(img)
        arr = np.array(result)
        # Neon pink band (rows 50-80) should have high saturation
        neon_region = arr[50:80, 50:350]
        dark_region = arr[0:10, 0:10]
        assert neon_region.mean() > dark_region.mean()


# ======================================================================
# binarize_otsu
# ======================================================================


class TestBinarizeOtsu:
    def test_returns_rgb_image(self, preprocessor: ImagePreprocessor) -> None:
        img = _make_image()
        result = preprocessor.binarize_otsu(img)
        assert result.mode == "RGB"

    def test_produces_binary_values(self, preprocessor: ImagePreprocessor) -> None:
        img = _make_neon_on_black()
        result = preprocessor.binarize_otsu(img)
        arr = np.array(result.convert("L"))
        unique = set(arr.flatten())
        assert unique.issubset({0, 255})

    def test_preserves_dimensions(self, preprocessor: ImagePreprocessor) -> None:
        img = _make_image(300, 200)
        result = preprocessor.binarize_otsu(img)
        assert result.size == (300, 200)


# ======================================================================
# morphological_cleanup
# ======================================================================


class TestMorphologicalCleanup:
    def test_returns_rgb_image(self, preprocessor: ImagePreprocessor) -> None:
        img = _make_image()
        result = preprocessor.morphological_cleanup(img)
        assert result.mode == "RGB"

    def test_preserves_dimensions(self, preprocessor: ImagePreprocessor) -> None:
        img = _make_image(300, 200)
        result = preprocessor.morphological_cleanup(img)
        assert result.size == (300, 200)

    def test_fills_small_gaps(self, preprocessor: ImagePreprocessor) -> None:
        # Create image with a small gap in a white line on black bg
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[50, 20:45] = 255  # left segment
        arr[50, 47:70] = 255  # right segment (2px gap)
        img = Image.fromarray(arr).convert("RGB")
        result = preprocessor.morphological_cleanup(img, kernel_size=3)
        result_arr = np.array(result.convert("L"))
        # The gap should be filled (or at least reduced)
        gap_region = result_arr[50, 45:47]
        assert gap_region.max() > 0

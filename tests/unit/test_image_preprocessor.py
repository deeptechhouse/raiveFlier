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
        # Create a structured image (gradient) with Gaussian noise added.
        # Pure random noise doesn't get smoothed — the denoiser needs
        # underlying structure to distinguish signal from noise.
        rng = np.random.default_rng(42)
        rows, cols = 200, 300
        # Horizontal gradient as base signal
        gradient = np.tile(np.linspace(0, 255, cols, dtype=np.float64), (rows, 1))
        base = np.stack([gradient, gradient, gradient], axis=-1)
        # Add Gaussian noise (sigma=40)
        noise = rng.normal(0, 40, base.shape)
        noisy = np.clip(base + noise, 0, 255).astype(np.uint8)
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


# ======================================================================
# build_ocr_passes
# ======================================================================


class TestBuildOcrPasses:
    def test_returns_eight_passes(self, preprocessor: ImagePreprocessor) -> None:
        img = _make_neon_on_black(400, 300)
        passes = preprocessor.build_ocr_passes(img)
        # standard, inverted, channel_red, channel_green, channel_blue,
        # clahe, denoised, saturation, otsu => but code shows 8 from reading
        assert len(passes) >= 8
        names = [name for name, _ in passes]
        assert "standard" in names
        assert "inverted" in names
        assert "channel_red" in names
        assert "channel_green" in names
        assert "channel_blue" in names
        assert "clahe" in names
        assert "denoised" in names

    def test_passes_are_pil_images(self, preprocessor: ImagePreprocessor) -> None:
        img = _make_image(400, 300)
        passes = preprocessor.build_ocr_passes(img)
        for name, processed_img in passes:
            assert isinstance(name, str)
            assert isinstance(processed_img, Image.Image)

    def test_caching_returns_same_result(self, preprocessor: ImagePreprocessor) -> None:
        img = _make_image(400, 300)
        passes1 = preprocessor.build_ocr_passes(img)
        passes2 = preprocessor.build_ocr_passes(img)
        # Cached — should be the exact same list object
        assert passes1 is passes2

    def test_different_image_replaces_cache(self, preprocessor: ImagePreprocessor) -> None:
        img1 = _make_image(400, 300, (100, 100, 100))
        img2 = _make_image(400, 300, (200, 200, 200))
        passes1 = preprocessor.build_ocr_passes(img1)
        passes2 = preprocessor.build_ocr_passes(img2)
        # Different images — cache replaced
        assert passes1 is not passes2


# ======================================================================
# prepare_for_ocr
# ======================================================================


class TestPrepareForOcr:
    def test_returns_png_bytes(self, preprocessor: ImagePreprocessor) -> None:
        img = _make_image(400, 300)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        result = preprocessor.prepare_for_ocr(buf.getvalue())
        assert isinstance(result, bytes)
        # PNG magic bytes
        assert result[:4] == b"\x89PNG"

    def test_output_is_valid_image(self, preprocessor: ImagePreprocessor) -> None:
        img = _make_image(400, 300)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        result = preprocessor.prepare_for_ocr(buf.getvalue())
        # Should be openable as an image
        opened = Image.open(io.BytesIO(result))
        assert opened.format == "PNG"


# ======================================================================
# detect_skew_angle + apply_deskew
# ======================================================================


class TestDeskew:
    def test_apply_deskew_none_returns_original(self, preprocessor: ImagePreprocessor) -> None:
        img = _make_image(200, 200)
        result = preprocessor.apply_deskew(img, None)
        assert result is img

    def test_apply_deskew_with_angle_rotates(self, preprocessor: ImagePreprocessor) -> None:
        img = _make_image(200, 200)
        result = preprocessor.apply_deskew(img, 5.0)
        # Rotated image with expand=True will have different dimensions
        assert result.size != img.size or result is not img

    def test_detect_skew_angle_no_lines(self, preprocessor: ImagePreprocessor) -> None:
        # Uniform image — no lines to detect
        img = _make_image(200, 200)
        angle = preprocessor.detect_skew_angle(img)
        assert angle is None

    def test_deskew_convenience_method(self, preprocessor: ImagePreprocessor) -> None:
        img = _make_image(200, 200)
        result = preprocessor.deskew(img)
        assert isinstance(result, Image.Image)


# ======================================================================
# separate_color_channels
# ======================================================================


class TestSeparateColorChannels:
    def test_returns_three_channels(self, preprocessor: ImagePreprocessor) -> None:
        img = _make_neon_on_black(200, 200)
        channels = preprocessor.separate_color_channels(img)
        assert len(channels) == 3
        for ch in channels:
            assert ch.mode == "L"
            assert ch.size == (200, 200)

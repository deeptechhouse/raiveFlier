"""Image preprocessing utilities for OCR extraction from rave flier images."""

import io

import cv2
import numpy as np
from PIL import Image, ImageEnhance


class ImagePreprocessor:
    """Prepares rave flier images for OCR by enhancing text readability.

    Rave fliers typically feature neon/bright text on dark backgrounds,
    heavy stylization, and varying orientations — all of which degrade
    standard OCR accuracy.
    """

    def prepare_for_ocr(self, image_bytes: bytes) -> bytes:
        """Run the full preprocessing pipeline on raw image bytes.

        Pipeline order: resize -> enhance contrast -> binarize -> deskew.

        Args:
            image_bytes: Raw image file bytes (JPEG, PNG, etc.).

        Returns:
            Preprocessed image as PNG bytes, optimized for OCR.
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = self.resize_for_ocr(image)
        image = self.enhance_contrast(image)
        image = self.binarize(image)
        image = self.deskew(image)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def enhance_contrast(self, image: Image.Image) -> Image.Image:
        """Boost image contrast to make text stand out from backgrounds.

        Args:
            image: Input PIL Image.

        Returns:
            Contrast-enhanced PIL Image.
        """
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)

        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(1.5)

    def binarize(self, image: Image.Image, threshold: int = 128) -> Image.Image:
        """Convert image to black-and-white using adaptive thresholding.

        Uses OpenCV adaptive Gaussian thresholding for better results on
        images with uneven lighting (common in flier photos).

        Args:
            image: Input PIL Image.
            threshold: Fallback global threshold (0–255) if adaptive fails.

        Returns:
            Binarized PIL Image in RGB mode.
        """
        gray = image.convert("L")
        img_array = np.array(gray)

        binary = cv2.adaptiveThreshold(
            img_array,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

        return Image.fromarray(binary).convert("RGB")

    def deskew(self, image: Image.Image) -> Image.Image:
        """Correct image rotation/skew using Hough line detection.

        Args:
            image: Input PIL Image.

        Returns:
            Deskewed PIL Image, or the original if no significant skew detected.
        """
        gray = np.array(image.convert("L"))
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10
        )

        if lines is None:
            return image

        angles: list[float] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Only consider near-horizontal lines (likely text baselines)
            if abs(angle) < 45:
                angles.append(angle)

        if not angles:
            return image

        median_angle = float(np.median(angles))

        # Only deskew if the angle is significant but not extreme
        if abs(median_angle) < 0.5 or abs(median_angle) > 15:
            return image

        return image.rotate(median_angle, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0))

    def separate_color_channels(self, image: Image.Image) -> list[Image.Image]:
        """Extract individual color channels to isolate neon text from dark backgrounds.

        Useful for fliers with bright colored text (pink, green, cyan) on
        black or very dark backgrounds.

        Args:
            image: Input PIL Image in RGB mode.

        Returns:
            List of three grayscale PIL Images: [Red, Green, Blue] channels.
        """
        rgb = image.convert("RGB")
        return list(rgb.split())

    def resize_for_ocr(self, image: Image.Image, max_dim: int = 2000) -> Image.Image:
        """Resize image so the largest dimension does not exceed max_dim.

        Preserves aspect ratio. Images smaller than max_dim are not upscaled.

        Args:
            image: Input PIL Image.
            max_dim: Maximum allowed dimension in pixels.

        Returns:
            Resized PIL Image (or original if already within bounds).
        """
        width, height = image.size
        if width <= max_dim and height <= max_dim:
            return image

        scale = max_dim / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return image.resize((new_width, new_height), Image.LANCZOS)

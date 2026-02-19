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

    # ------------------------------------------------------------------
    # Extended preprocessing methods (OCR accuracy enhancements)
    # ------------------------------------------------------------------

    def enhance_contrast_clahe(
        self,
        image: Image.Image,
        clip_limit: float = 3.0,
        tile_size: int = 8,
    ) -> Image.Image:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Works in LAB color space so only the lightness channel is
        equalized — color information is preserved.  Handles uneven
        lighting from phone-captured flier photos far better than a
        global contrast boost.

        Args:
            image: Input PIL Image in RGB mode.
            clip_limit: Contrast limiting threshold for CLAHE.
            tile_size: Grid size for the adaptive histogram tiles.

        Returns:
            Contrast-enhanced PIL Image in RGB mode.
        """
        img_array = np.array(image.convert("RGB"))
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=(tile_size, tile_size)
        )
        l_enhanced = clahe.apply(l_channel)

        merged = cv2.merge([l_enhanced, a_channel, b_channel])
        result = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return Image.fromarray(result)

    def denoise(self, image: Image.Image, strength: int = 10) -> Image.Image:
        """Remove sensor noise from phone-captured flier photos.

        Uses OpenCV fast non-local means denoising which preserves edges
        (important for text) while smoothing out camera noise.

        Args:
            image: Input PIL Image in RGB mode.
            strength: Filter strength — higher removes more noise but may
                blur fine text.  10 is a safe default.

        Returns:
            Denoised PIL Image in RGB mode.
        """
        img_array = np.array(image.convert("RGB"))
        denoised = cv2.fastNlMeansDenoisingColored(
            img_array, None, strength, strength, 7, 21
        )
        return Image.fromarray(denoised)

    def extract_saturation_channel(self, image: Image.Image) -> Image.Image:
        """Extract the HSV saturation channel to isolate neon text.

        Neon/bright colored text (pink, cyan, green, yellow) on dark
        backgrounds has very high saturation, while the dark background
        has near-zero saturation.  This produces a much cleaner text
        mask than simple RGB channel splitting.

        Args:
            image: Input PIL Image in RGB mode.

        Returns:
            Grayscale PIL Image representing the saturation channel.
        """
        img_array = np.array(image.convert("RGB"))
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        return Image.fromarray(saturation)

    def binarize_otsu(self, image: Image.Image) -> Image.Image:
        """Binarize using Otsu's automatic thresholding method.

        Otsu's method computes the optimal global threshold by
        minimizing intra-class variance.  Works well when the image
        has a bimodal histogram (bright text on dark background or
        vice versa).

        Args:
            image: Input PIL Image.

        Returns:
            Binarized PIL Image in RGB mode.
        """
        gray = np.array(image.convert("L"))
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return Image.fromarray(binary).convert("RGB")

    def morphological_cleanup(
        self, image: Image.Image, kernel_size: int = 2
    ) -> Image.Image:
        """Apply morphological closing to reconnect broken letter strokes.

        Aggressive binarization or low resolution can leave gaps in
        letter shapes.  Closing (dilate then erode) fills small holes
        without significantly changing letter outlines.

        Args:
            image: Input PIL Image.
            kernel_size: Size of the structuring element.  2 is
                conservative; increase for very fragmented text.

        Returns:
            Cleaned PIL Image in RGB mode.
        """
        gray = np.array(image.convert("L"))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        return Image.fromarray(closed).convert("RGB")

    def resize_for_ocr(
        self, image: Image.Image, max_dim: int = 2000, min_dim: int = 1200
    ) -> Image.Image:
        """Resize image so the largest dimension is between *min_dim* and *max_dim*.

        Preserves aspect ratio.  Small images (e.g. phone-captured flier
        photos) are upscaled so OCR engines have enough pixel data to work
        with — characters should be at least 20-30 px tall for reliable
        recognition.

        Args:
            image: Input PIL Image.
            max_dim: Maximum allowed dimension in pixels.
            min_dim: Minimum target for the largest dimension. Images smaller
                than this are upscaled.

        Returns:
            Resized PIL Image.
        """
        width, height = image.size
        largest = max(width, height)

        if largest < min_dim:
            scale = min_dim / largest
        elif largest > max_dim:
            scale = max_dim / largest
        else:
            return image

        new_width = int(width * scale)
        new_height = int(height * scale)
        return image.resize((new_width, new_height), Image.LANCZOS)

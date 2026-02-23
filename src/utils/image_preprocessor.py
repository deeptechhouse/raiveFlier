"""Image preprocessing utilities for OCR extraction from rave flier images.

Rave fliers present a uniquely challenging OCR problem: neon/bright text on
dark backgrounds, heavy stylization, creative fonts, rotation, and mixed
colour schemes.  Standard OCR engines (Tesseract, EasyOCR) perform poorly
on raw flier photos.

This module implements a **multi-pass preprocessing strategy** that generates
7 different image variants from a single input, each targeting a different
failure mode:

    Pass 1 - "standard"    : Contrast boost + adaptive binarization + deskew
    Pass 2 - "inverted"    : Colour inversion (catches light-on-dark text)
    Pass 3 - "channel_*"   : Individual R/G/B channels (isolates neon text)
    Pass 4 - "clahe"       : CLAHE adaptive contrast (handles uneven lighting)
    Pass 5 - "denoised"    : Bilateral denoising (phone camera noise)
    Pass 6 - "saturation"  : HSV saturation channel (neon text mask)
    Pass 7 - "otsu"        : Otsu global binarization (bimodal histograms)

OCR providers run all passes and merge results via ``ocr_helpers.merge_pass_results``
to maximise text recovery.  Results are cached by image content hash so
multiple providers sharing the same image avoid redundant computation.
"""

import hashlib
import io

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


class ImagePreprocessor:
    """Prepares rave flier images for OCR by enhancing text readability.

    Rave fliers typically feature neon/bright text on dark backgrounds,
    heavy stylization, and varying orientations -- all of which degrade
    standard OCR accuracy.
    """

    def __init__(self) -> None:
        # Cache: maps image content MD5 -> list of preprocessed passes.
        # Only the most recent image is cached to limit memory usage.
        self._pass_cache: dict[str, list[tuple[str, Image.Image]]] = {}

    def build_ocr_passes(self, original: Image.Image) -> list[tuple[str, Image.Image]]:
        """Build all 7 OCR preprocessing passes for an image.

        Results are cached by image content hash so that multiple providers
        (e.g. Tesseract fallback to EasyOCR) share the same preprocessed
        images without recomputing.

        Returns a list of (pass_name, preprocessed_image) tuples.
        """
        # Content-based cache key avoids reprocessing the same image bytes.
        img_hash = hashlib.md5(original.tobytes()).hexdigest()
        if img_hash in self._pass_cache:
            return self._pass_cache[img_hash]

        passes: list[tuple[str, Image.Image]] = []

        # Shared base: resize to OCR-friendly dimensions (upscale small images,
        # downscale huge ones) so all passes work from the same resolution.
        resized = self.resize_for_ocr(original)

        # --- Pass 1: Standard contrast + adaptive binarization + deskew ---
        # This is the baseline pass that works for well-lit, well-oriented fliers.
        enhanced = self.enhance_contrast(resized)
        enhanced_gray = np.array(enhanced.convert("L"))
        binarized = self.binarize(enhanced, gray_array=enhanced_gray)

        # Compute skew angle ONCE on the standard pass and reuse across all
        # passes -- saves expensive Hough transform computation.
        binarized_gray = np.array(binarized.convert("L"))
        skew_angle = self.detect_skew_angle(binarized, gray_array=binarized_gray)

        deskewed = self.apply_deskew(binarized, skew_angle)
        passes.append(("standard", deskewed))

        # --- Pass 2: Inverted (catches light text on dark backgrounds) ---
        # Many rave fliers have white/neon text on black -- inverting makes
        # this look like standard black-on-white for the OCR engine.
        inverted = ImageOps.invert(deskewed.convert("RGB"))
        passes.append(("inverted", inverted))

        # --- Pass 3: Individual R/G/B color channels (isolate neon text) ---
        # Neon pink text shows up strongly in the red channel; neon green in
        # the green channel, etc.  Isolating channels improves contrast for
        # specific colour schemes.
        channels = self.separate_color_channels(resized)
        channel_names = ["red", "green", "blue"]
        for ch_name, channel in zip(channel_names, channels, strict=True):
            passes.append((f"channel_{ch_name}", channel.convert("RGB")))

        # --- Pass 4: CLAHE (Contrast Limited Adaptive Histogram Equalization) ---
        # Handles uneven lighting from phone-camera flash or ambient light.
        # Operates in LAB colour space so only lightness is equalized.
        clahe_enhanced = self.enhance_contrast_clahe(resized)
        clahe_gray = np.array(clahe_enhanced.convert("L"))
        clahe_binarized = self.binarize(clahe_enhanced, gray_array=clahe_gray)
        clahe_deskewed = self.apply_deskew(clahe_binarized, skew_angle)
        passes.append(("clahe", clahe_deskewed))

        # --- Pass 5: Denoised (bilateral filter preserves text edges) ---
        # Phone cameras in dark clubs produce noisy images; bilateral filtering
        # smooths noise while keeping letter edges sharp.
        denoised = self.denoise(resized)
        denoised_enhanced = self.enhance_contrast(denoised)
        denoised_gray = np.array(denoised_enhanced.convert("L"))
        denoised_binarized = self.binarize(denoised_enhanced, gray_array=denoised_gray)
        denoised_deskewed = self.apply_deskew(denoised_binarized, skew_angle)
        passes.append(("denoised", denoised_deskewed))

        # --- Pass 6: HSV saturation channel (neon text mask) ---
        # Neon text has very high saturation; dark backgrounds have near-zero.
        # This produces a cleaner text/background separation than RGB splitting.
        saturation = self.extract_saturation_channel(resized)
        sat_rgb = saturation.convert("RGB")
        sat_gray = np.array(sat_rgb.convert("L"))
        sat_binarized = self.binarize(sat_rgb, gray_array=sat_gray)
        passes.append(("saturation", sat_binarized))

        # --- Pass 7: Otsu binarization (automatic global threshold) ---
        # Best for images with a clear bimodal histogram (bright text on dark
        # background).  Complements the adaptive threshold in Pass 1.
        otsu_enhanced = self.enhance_contrast(resized)
        otsu_gray = np.array(otsu_enhanced.convert("L"))
        otsu_binarized = self.binarize_otsu(otsu_enhanced, gray_array=otsu_gray)
        otsu_deskewed = self.apply_deskew(otsu_binarized, skew_angle)
        passes.append(("otsu", otsu_deskewed))

        # Cache only the most recent image to limit memory footprint.
        self._pass_cache.clear()
        self._pass_cache[img_hash] = passes

        return passes

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

    def binarize(
        self,
        image: Image.Image,
        threshold: int = 128,
        gray_array: np.ndarray | None = None,
    ) -> Image.Image:
        """Convert image to black-and-white using adaptive thresholding.

        Uses OpenCV adaptive Gaussian thresholding for better results on
        images with uneven lighting (common in flier photos).

        Args:
            image: Input PIL Image.
            threshold: Fallback global threshold (0–255) if adaptive fails.
            gray_array: Optional pre-computed grayscale numpy array to avoid
                redundant RGB→L conversions.

        Returns:
            Binarized PIL Image in RGB mode.
        """
        if gray_array is None:
            gray_array = np.array(image.convert("L"))

        binary = cv2.adaptiveThreshold(
            gray_array,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

        return Image.fromarray(binary).convert("RGB")

    def detect_skew_angle(
        self, image: Image.Image, gray_array: np.ndarray | None = None
    ) -> float | None:
        """Detect the skew angle of an image using Hough line detection.

        Args:
            image: Input PIL Image.
            gray_array: Optional pre-computed grayscale numpy array.

        Returns:
            The median skew angle in degrees, or ``None`` if no significant
            skew is detected.
        """
        if gray_array is None:
            gray_array = np.array(image.convert("L"))
        gray = gray_array
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10
        )

        if lines is None:
            return None

        angles: list[float] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:
                angles.append(angle)

        if not angles:
            return None

        median_angle = float(np.median(angles))

        if abs(median_angle) < 0.5 or abs(median_angle) > 15:
            return None

        return median_angle

    @staticmethod
    def apply_deskew(image: Image.Image, angle: float | None) -> Image.Image:
        """Rotate an image by a pre-computed skew angle.

        Args:
            image: Input PIL Image.
            angle: Skew angle in degrees, or ``None`` to return the image unchanged.

        Returns:
            Deskewed PIL Image.
        """
        if angle is None:
            return image
        return image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0))

    def deskew(self, image: Image.Image) -> Image.Image:
        """Correct image rotation/skew using Hough line detection.

        Args:
            image: Input PIL Image.

        Returns:
            Deskewed PIL Image, or the original if no significant skew detected.
        """
        angle = self.detect_skew_angle(image)
        return self.apply_deskew(image, angle)

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

        Uses OpenCV bilateral filter which preserves edges (important for
        text) while smoothing out camera noise.  Bilateral filtering is
        10-50x faster than ``fastNlMeansDenoisingColored`` with comparable
        OCR-relevant results.

        Args:
            image: Input PIL Image in RGB mode.
            strength: Filter strength — higher removes more noise but may
                blur fine text.  10 is a safe default.

        Returns:
            Denoised PIL Image in RGB mode.
        """
        img_array = np.array(image.convert("RGB"))
        denoised = cv2.bilateralFilter(img_array, d=9, sigmaColor=strength * 7.5, sigmaSpace=strength * 7.5)
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

    def binarize_otsu(
        self, image: Image.Image, gray_array: np.ndarray | None = None
    ) -> Image.Image:
        """Binarize using Otsu's automatic thresholding method.

        Otsu's method computes the optimal global threshold by
        minimizing intra-class variance.  Works well when the image
        has a bimodal histogram (bright text on dark background or
        vice versa).

        Args:
            image: Input PIL Image.
            gray_array: Optional pre-computed grayscale numpy array.

        Returns:
            Binarized PIL Image in RGB mode.
        """
        if gray_array is None:
            gray_array = np.array(image.convert("L"))
        _, binary = cv2.threshold(
            gray_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
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

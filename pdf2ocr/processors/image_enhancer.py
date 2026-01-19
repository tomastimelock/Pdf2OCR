"""Image Enhancer - Enhance document images for better OCR quality."""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Union, Callable
import tempfile
import shutil


@dataclass
class QualityAnalysis:
    """Analysis of image quality."""
    blur_score: float  # 0-1, higher = more blur
    noise_level: float  # 0-1, higher = more noise
    contrast_score: float  # 0-1, higher = better contrast
    brightness_score: float  # 0-1, 0.5 = optimal
    overall_score: float  # 0-1, composite quality score
    needs_enhancement: bool
    recommendations: List[str]


@dataclass
class EnhancementResult:
    """Result of image enhancement."""
    original_path: Path
    enhanced_path: Path
    quality_before: float
    quality_after: float
    operations_applied: List[str]
    success: bool
    error: Optional[str] = None


class ImageEnhancer:
    """
    Enhance document images to improve OCR accuracy.

    Uses the image_core vendor module for processing.
    Automatically detects and applies appropriate enhancements
    for document images (scans, photos of documents, etc.)

    Enhancement capabilities:
    - Contrast adjustment
    - Sharpening
    - Noise reduction
    - Brightness normalization
    - Deskewing (rotation correction)
    """

    # Enhancement presets for different document types
    PRESETS = {
        "document": {
            "contrast": 1.2,
            "sharpness": 1.3,
            "noise_reduction": 0.5,
            "brightness": 1.0
        },
        "scan": {
            "contrast": 1.3,
            "sharpness": 1.4,
            "noise_reduction": 0.6,
            "brightness": 1.05
        },
        "photo": {
            "contrast": 1.1,
            "sharpness": 1.2,
            "noise_reduction": 0.4,
            "brightness": 1.0
        },
        "low_quality": {
            "contrast": 1.4,
            "sharpness": 1.5,
            "noise_reduction": 0.7,
            "brightness": 1.1
        }
    }

    # Quality thresholds
    QUALITY_THRESHOLD = 0.6  # Below this, enhancement recommended
    BLUR_THRESHOLD = 0.4  # Above this, image is too blurry
    NOISE_THRESHOLD = 0.5  # Above this, noise reduction needed

    def __init__(
        self,
        preset: str = "document",
        auto_enhance: bool = True,
        quality_threshold: float = QUALITY_THRESHOLD
    ):
        """
        Initialize the image enhancer.

        Args:
            preset: Enhancement preset ("document", "scan", "photo", "low_quality")
            auto_enhance: Automatically determine enhancement needs
            quality_threshold: Quality score below which enhancement is applied
        """
        self.preset = preset
        self.auto_enhance = auto_enhance
        self.quality_threshold = quality_threshold
        self._image_core = None

    @property
    def image_core(self):
        """Lazy load image_core module."""
        if self._image_core is None:
            try:
                from pdf2ocr.vendor.image_core import ImageCore
                self._image_core = ImageCore()
            except ImportError:
                raise ImportError(
                    "image_core vendor module not available. "
                    "Ensure vendor modules are properly installed."
                )
        return self._image_core

    def analyze_quality(self, image_path: Union[str, Path]) -> QualityAnalysis:
        """
        Analyze image quality for OCR suitability.

        Args:
            image_path: Path to the image file

        Returns:
            QualityAnalysis with scores and recommendations
        """
        image_path = Path(image_path)

        try:
            from PIL import Image
            import numpy as np

            with Image.open(image_path) as img:
                # Convert to grayscale for analysis
                if img.mode != 'L':
                    gray = img.convert('L')
                else:
                    gray = img

                pixels = np.array(gray)

                # Calculate metrics
                blur_score = self._estimate_blur(pixels)
                noise_level = self._estimate_noise(pixels)
                contrast_score = self._calculate_contrast(pixels)
                brightness_score = self._calculate_brightness(pixels)

                # Calculate overall score
                overall_score = (
                    (1 - blur_score) * 0.3 +
                    (1 - noise_level) * 0.2 +
                    contrast_score * 0.3 +
                    (1 - abs(brightness_score - 0.5) * 2) * 0.2
                )

                # Generate recommendations
                recommendations = []
                if blur_score > self.BLUR_THRESHOLD:
                    recommendations.append("Apply sharpening")
                if noise_level > self.NOISE_THRESHOLD:
                    recommendations.append("Apply noise reduction")
                if contrast_score < 0.4:
                    recommendations.append("Increase contrast")
                if brightness_score < 0.3:
                    recommendations.append("Increase brightness")
                elif brightness_score > 0.7:
                    recommendations.append("Decrease brightness")

                needs_enhancement = overall_score < self.quality_threshold

                return QualityAnalysis(
                    blur_score=blur_score,
                    noise_level=noise_level,
                    contrast_score=contrast_score,
                    brightness_score=brightness_score,
                    overall_score=overall_score,
                    needs_enhancement=needs_enhancement,
                    recommendations=recommendations
                )

        except Exception as e:
            # Return default analysis on error
            return QualityAnalysis(
                blur_score=0.5,
                noise_level=0.5,
                contrast_score=0.5,
                brightness_score=0.5,
                overall_score=0.5,
                needs_enhancement=True,
                recommendations=[f"Analysis failed: {e}"]
            )

    def enhance_for_ocr(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        preset: Optional[str] = None,
        analyze_first: bool = True
    ) -> EnhancementResult:
        """
        Enhance an image for better OCR results.

        Args:
            image_path: Path to the input image
            output_path: Path for enhanced image (uses temp file if not provided)
            preset: Enhancement preset to use (overrides instance preset)
            analyze_first: Whether to analyze quality before enhancing

        Returns:
            EnhancementResult with before/after quality scores
        """
        image_path = Path(image_path)
        preset = preset or self.preset
        settings = self.PRESETS.get(preset, self.PRESETS["document"])

        # Analyze quality if requested
        quality_before = 0.5
        if analyze_first:
            analysis = self.analyze_quality(image_path)
            quality_before = analysis.overall_score

            # Skip enhancement if quality is already good
            if self.auto_enhance and not analysis.needs_enhancement:
                return EnhancementResult(
                    original_path=image_path,
                    enhanced_path=image_path,
                    quality_before=quality_before,
                    quality_after=quality_before,
                    operations_applied=[],
                    success=True
                )

        # Determine output path
        if output_path is None:
            suffix = image_path.suffix
            output_path = image_path.parent / f"{image_path.stem}_enhanced{suffix}"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        operations = []

        try:
            from PIL import Image, ImageEnhance, ImageFilter

            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')

                # Apply contrast enhancement
                if settings.get("contrast", 1.0) != 1.0:
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(settings["contrast"])
                    operations.append(f"contrast:{settings['contrast']}")

                # Apply brightness adjustment
                if settings.get("brightness", 1.0) != 1.0:
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(settings["brightness"])
                    operations.append(f"brightness:{settings['brightness']}")

                # Apply sharpening
                if settings.get("sharpness", 1.0) > 1.0:
                    enhancer = ImageEnhance.Sharpness(img)
                    img = enhancer.enhance(settings["sharpness"])
                    operations.append(f"sharpness:{settings['sharpness']}")

                # Apply noise reduction (using slight blur + sharpen)
                noise_level = settings.get("noise_reduction", 0)
                if noise_level > 0:
                    # Mild smoothing for noise reduction
                    if noise_level > 0.5:
                        img = img.filter(ImageFilter.MedianFilter(3))
                        operations.append("median_filter")
                    else:
                        img = img.filter(ImageFilter.SMOOTH)
                        operations.append("smooth_filter")

                # Save enhanced image
                if image_path.suffix.lower() in ('.jpg', '.jpeg'):
                    img.save(str(output_path), 'JPEG', quality=95)
                else:
                    img.save(str(output_path))

            # Analyze quality after enhancement
            quality_after = quality_before
            if analyze_first:
                analysis_after = self.analyze_quality(output_path)
                quality_after = analysis_after.overall_score

            return EnhancementResult(
                original_path=image_path,
                enhanced_path=output_path,
                quality_before=quality_before,
                quality_after=quality_after,
                operations_applied=operations,
                success=True
            )

        except Exception as e:
            return EnhancementResult(
                original_path=image_path,
                enhanced_path=image_path,
                quality_before=quality_before,
                quality_after=quality_before,
                operations_applied=[],
                success=False,
                error=str(e)
            )

    def batch_enhance(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        preset: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, EnhancementResult], None]] = None
    ) -> List[EnhancementResult]:
        """
        Enhance multiple images in batch.

        Args:
            image_paths: List of image paths
            output_dir: Directory for enhanced images
            preset: Enhancement preset to use
            progress_callback: Callback for progress updates

        Returns:
            List of EnhancementResult for each image
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        total = len(image_paths)

        for idx, image_path in enumerate(image_paths, 1):
            image_path = Path(image_path)
            output_path = output_dir / f"{image_path.stem}_enhanced{image_path.suffix}"

            result = self.enhance_for_ocr(
                image_path,
                output_path,
                preset=preset
            )
            results.append(result)

            if progress_callback:
                progress_callback(idx, total, result)

        return results

    def _estimate_blur(self, pixels) -> float:
        """Estimate blur level using Laplacian variance."""
        import numpy as np
        from scipy import ndimage

        try:
            laplacian = ndimage.laplace(pixels.astype(float))
            variance = laplacian.var()
            # Normalize to 0-1 range (higher = more blur)
            blur_score = 1.0 - min(variance / 500, 1.0)
            return max(0, min(1, blur_score))
        except Exception:
            return 0.5

    def _estimate_noise(self, pixels) -> float:
        """Estimate noise level."""
        import numpy as np

        try:
            # Calculate local variance
            from scipy import ndimage
            local_var = ndimage.generic_filter(
                pixels.astype(float),
                np.var,
                size=5
            )
            noise_estimate = np.median(local_var) / 1000
            return max(0, min(1, noise_estimate))
        except Exception:
            return 0.5

    def _calculate_contrast(self, pixels) -> float:
        """Calculate contrast score."""
        import numpy as np

        try:
            std = np.std(pixels)
            # Normalize to 0-1 (higher std = better contrast)
            contrast = std / 128
            return max(0, min(1, contrast))
        except Exception:
            return 0.5

    def _calculate_brightness(self, pixels) -> float:
        """Calculate brightness score (0.5 is optimal)."""
        import numpy as np

        try:
            mean = np.mean(pixels)
            # Normalize to 0-1
            return mean / 255
        except Exception:
            return 0.5

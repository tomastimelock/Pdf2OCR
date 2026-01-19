"""
Image quality analysis.

Provides comprehensive quality metrics including blur detection,
noise analysis, and aesthetic scoring.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any, List
import logging

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class BlurMetrics:
    """Blur detection metrics."""
    laplacian_variance: float
    blur_score: float  # 0-1, higher = sharper
    is_blurry: bool
    threshold_used: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'laplacian_variance': self.laplacian_variance,
            'blur_score': self.blur_score,
            'is_blurry': self.is_blurry,
            'threshold_used': self.threshold_used,
        }


@dataclass
class NoiseMetrics:
    """Noise analysis metrics."""
    noise_level: float  # 0-1, higher = more noise
    snr: float  # Signal-to-noise ratio
    is_noisy: bool
    noise_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'noise_level': self.noise_level,
            'snr': self.snr,
            'is_noisy': self.is_noisy,
            'noise_type': self.noise_type,
        }


@dataclass
class AestheticMetrics:
    """Aesthetic quality metrics."""
    overall_score: float  # 0-10
    composition_score: float
    color_harmony_score: float
    exposure_score: float
    sharpness_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'overall_score': self.overall_score,
            'composition_score': self.composition_score,
            'color_harmony_score': self.color_harmony_score,
            'exposure_score': self.exposure_score,
            'sharpness_score': self.sharpness_score,
        }


@dataclass
class QualityAnalysis:
    """Complete quality analysis result."""
    success: bool
    image_size: Tuple[int, int]
    blur: Optional[BlurMetrics] = None
    noise: Optional[NoiseMetrics] = None
    aesthetics: Optional[AestheticMetrics] = None
    overall_quality: Optional[float] = None  # 0-100
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'image_size': self.image_size,
            'blur': self.blur.to_dict() if self.blur else None,
            'noise': self.noise.to_dict() if self.noise else None,
            'aesthetics': self.aesthetics.to_dict() if self.aesthetics else None,
            'overall_quality': self.overall_quality,
            'issues': self.issues,
            'recommendations': self.recommendations,
            'error': self.error,
        }


class QualityAnalyzer:
    """
    Image quality analysis engine.

    Provides comprehensive quality metrics including blur,
    noise, and aesthetic analysis.
    """

    def __init__(
        self,
        blur_threshold: float = 100.0,
        noise_threshold: float = 0.3
    ):
        """
        Initialize the analyzer.

        Args:
            blur_threshold: Laplacian variance threshold for blur detection
            noise_threshold: Noise level threshold
        """
        self.blur_threshold = blur_threshold
        self.noise_threshold = noise_threshold

    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """Load image from path or return if already an Image."""
        if isinstance(image, Image.Image):
            return image
        return Image.open(str(image))

    def _calculate_blur_metrics(self, image: Image.Image) -> BlurMetrics:
        """Calculate blur metrics using Laplacian variance."""
        try:
            import numpy as np
            from scipy import ndimage

            # Convert to grayscale
            gray = image.convert('L')
            arr = np.array(gray, dtype=np.float64)

            # Calculate Laplacian
            laplacian = ndimage.laplace(arr)
            variance = np.var(laplacian)

            # Normalize to score
            # Higher variance = sharper image
            blur_score = min(1.0, variance / (self.blur_threshold * 2))
            is_blurry = variance < self.blur_threshold

            return BlurMetrics(
                laplacian_variance=float(variance),
                blur_score=float(blur_score),
                is_blurry=is_blurry,
                threshold_used=self.blur_threshold
            )

        except ImportError:
            # Fallback without scipy
            logger.warning("scipy not available, using basic blur detection")

            gray = image.convert('L')

            # Simple edge detection using PIL
            from PIL import ImageFilter
            edges = gray.filter(ImageFilter.FIND_EDGES)

            import statistics
            edge_data = list(edges.getdata())
            variance = statistics.variance(edge_data) if len(edge_data) > 1 else 0

            blur_score = min(1.0, variance / 5000)
            is_blurry = variance < 1000

            return BlurMetrics(
                laplacian_variance=float(variance),
                blur_score=float(blur_score),
                is_blurry=is_blurry,
                threshold_used=1000.0
            )

    def _calculate_noise_metrics(self, image: Image.Image) -> NoiseMetrics:
        """Estimate image noise level."""
        try:
            import numpy as np
            from scipy import ndimage

            # Convert to grayscale
            gray = image.convert('L')
            arr = np.array(gray, dtype=np.float64)

            # Estimate noise using median absolute deviation
            # High-pass filter to isolate noise
            smoothed = ndimage.gaussian_filter(arr, sigma=1.0)
            residual = arr - smoothed

            # Calculate noise metrics
            noise_std = np.std(residual)
            signal_std = np.std(arr)

            # Normalize noise level to 0-1
            noise_level = min(1.0, noise_std / 50.0)
            snr = signal_std / (noise_std + 1e-10)
            is_noisy = noise_level > self.noise_threshold

            # Try to determine noise type
            noise_type = None
            if is_noisy:
                # Check for pattern noise (salt and pepper)
                extreme_pixels = np.sum((arr < 10) | (arr > 245)) / arr.size
                if extreme_pixels > 0.01:
                    noise_type = "salt_and_pepper"
                else:
                    noise_type = "gaussian"

            return NoiseMetrics(
                noise_level=float(noise_level),
                snr=float(snr),
                is_noisy=is_noisy,
                noise_type=noise_type
            )

        except ImportError:
            logger.warning("scipy not available, using basic noise estimation")

            gray = image.convert('L')
            data = list(gray.getdata())

            import statistics
            if len(data) > 1:
                std = statistics.stdev(data)
                noise_level = min(1.0, std / 50.0)
            else:
                noise_level = 0.0

            return NoiseMetrics(
                noise_level=float(noise_level),
                snr=10.0,
                is_noisy=noise_level > self.noise_threshold,
                noise_type=None
            )

    def _calculate_aesthetic_metrics(self, image: Image.Image) -> AestheticMetrics:
        """Calculate aesthetic quality metrics."""
        try:
            import numpy as np

            arr = np.array(image)

            # Exposure score (based on histogram distribution)
            gray = image.convert('L')
            hist = gray.histogram()
            hist = np.array(hist, dtype=np.float64)
            hist = hist / hist.sum()

            # Ideal exposure has balanced histogram
            mean_brightness = np.sum(np.arange(256) * hist)
            exposure_score = 1.0 - abs(mean_brightness - 128) / 128
            exposure_score = float(max(0, min(10, exposure_score * 10)))

            # Color harmony (based on color variance in HSV)
            if image.mode == 'RGB':
                hsv = image.convert('HSV')
                h, s, v = hsv.split()
                h_data = np.array(h)
                s_data = np.array(s)

                # Good color harmony has coherent hues
                h_std = np.std(h_data)
                color_harmony = 1.0 - min(1.0, h_std / 100)
                color_harmony_score = float(max(0, min(10, color_harmony * 10)))
            else:
                color_harmony_score = 5.0

            # Composition score (rule of thirds approximation)
            # Check for interesting content at thirds
            w, h = image.size
            third_w, third_h = w // 3, h // 3

            gray_arr = np.array(gray)

            # Calculate variance in each third
            regions = []
            for i in range(3):
                for j in range(3):
                    region = gray_arr[j*third_h:(j+1)*third_h, i*third_w:(i+1)*third_w]
                    regions.append(np.var(region))

            # Good composition has high variance at intersections (indices 0,2,6,8)
            intersection_variance = np.mean([regions[0], regions[2], regions[6], regions[8]])
            center_variance = regions[4]

            if center_variance > 0:
                composition_ratio = intersection_variance / (center_variance + 1)
                composition_score = float(min(10, max(0, composition_ratio * 5)))
            else:
                composition_score = 5.0

            # Sharpness score from blur metrics
            blur_metrics = self._calculate_blur_metrics(image)
            sharpness_score = float(blur_metrics.blur_score * 10)

            # Overall aesthetic score
            overall_score = (
                exposure_score * 0.25 +
                color_harmony_score * 0.25 +
                composition_score * 0.25 +
                sharpness_score * 0.25
            )

            return AestheticMetrics(
                overall_score=round(overall_score, 2),
                composition_score=round(composition_score, 2),
                color_harmony_score=round(color_harmony_score, 2),
                exposure_score=round(exposure_score, 2),
                sharpness_score=round(sharpness_score, 2)
            )

        except ImportError:
            # Return default scores without numpy
            return AestheticMetrics(
                overall_score=5.0,
                composition_score=5.0,
                color_harmony_score=5.0,
                exposure_score=5.0,
                sharpness_score=5.0
            )

    def analyze(
        self,
        image: Union[str, Path, Image.Image],
        include_blur: bool = True,
        include_noise: bool = True,
        include_aesthetics: bool = True
    ) -> QualityAnalysis:
        """
        Analyze image quality.

        Args:
            image: Input image (path or PIL Image)
            include_blur: Whether to analyze blur
            include_noise: Whether to analyze noise
            include_aesthetics: Whether to analyze aesthetics

        Returns:
            QualityAnalysis with all requested metrics
        """
        try:
            img = self._load_image(image)

            blur_metrics = None
            noise_metrics = None
            aesthetic_metrics = None
            issues = []
            recommendations = []

            if include_blur:
                blur_metrics = self._calculate_blur_metrics(img)
                if blur_metrics.is_blurry:
                    issues.append("Image appears blurry")
                    recommendations.append("Consider using a tripod or faster shutter speed")

            if include_noise:
                noise_metrics = self._calculate_noise_metrics(img)
                if noise_metrics.is_noisy:
                    issues.append(f"High noise level detected ({noise_metrics.noise_type or 'unknown type'})")
                    recommendations.append("Consider using noise reduction or lower ISO")

            if include_aesthetics:
                aesthetic_metrics = self._calculate_aesthetic_metrics(img)
                if aesthetic_metrics.exposure_score < 4:
                    issues.append("Poor exposure")
                    recommendations.append("Adjust exposure or use exposure compensation")
                if aesthetic_metrics.color_harmony_score < 4:
                    issues.append("Weak color harmony")
                    recommendations.append("Consider color grading or white balance adjustment")

            # Calculate overall quality score
            scores = []
            if blur_metrics:
                scores.append(blur_metrics.blur_score * 100)
            if noise_metrics:
                scores.append((1 - noise_metrics.noise_level) * 100)
            if aesthetic_metrics:
                scores.append(aesthetic_metrics.overall_score * 10)

            overall_quality = sum(scores) / len(scores) if scores else None

            return QualityAnalysis(
                success=True,
                image_size=img.size,
                blur=blur_metrics,
                noise=noise_metrics,
                aesthetics=aesthetic_metrics,
                overall_quality=round(overall_quality, 2) if overall_quality else None,
                issues=issues,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Quality analysis error: {e}")
            return QualityAnalysis(
                success=False,
                image_size=(0, 0),
                error=str(e)
            )

    def detect_blur(
        self,
        image: Union[str, Path, Image.Image]
    ) -> BlurMetrics:
        """Convenience method for blur detection only."""
        img = self._load_image(image)
        return self._calculate_blur_metrics(img)

    def detect_noise(
        self,
        image: Union[str, Path, Image.Image]
    ) -> NoiseMetrics:
        """Convenience method for noise detection only."""
        img = self._load_image(image)
        return self._calculate_noise_metrics(img)

    def score_aesthetics(
        self,
        image: Union[str, Path, Image.Image]
    ) -> AestheticMetrics:
        """Convenience method for aesthetic scoring only."""
        img = self._load_image(image)
        return self._calculate_aesthetic_metrics(img)

    def batch_analyze(
        self,
        images: List[Union[str, Path]],
        include_blur: bool = True,
        include_noise: bool = True,
        include_aesthetics: bool = True
    ) -> List[QualityAnalysis]:
        """Analyze multiple images."""
        results = []
        for img_path in images:
            result = self.analyze(
                img_path,
                include_blur=include_blur,
                include_noise=include_noise,
                include_aesthetics=include_aesthetics
            )
            results.append(result)
        return results

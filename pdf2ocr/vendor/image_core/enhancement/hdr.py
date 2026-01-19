"""
HDR (High Dynamic Range) merging operations.

Provides HDR merging functionality for combining multiple
exposures into a single image.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any, List
import logging

from PIL import Image

logger = logging.getLogger(__name__)


class ToneMappingMethod(Enum):
    """Tone mapping methods for HDR output."""
    REINHARD = "reinhard"
    DRAGO = "drago"
    MANTIUK = "mantiuk"
    LINEAR = "linear"


@dataclass
class HDRResult:
    """Result of an HDR merge operation."""
    success: bool
    input_count: int
    output_size: Tuple[int, int]
    tone_mapping: str
    output_path: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'success': self.success,
            'input_count': self.input_count,
            'output_size': self.output_size,
            'tone_mapping': self.tone_mapping,
            'output_path': self.output_path,
            'error': self.error,
            'metadata': self.metadata,
        }


class HDRMerger:
    """
    HDR merging engine.

    Combines multiple exposures into a single HDR image
    with tone mapping for display.
    """

    def __init__(self, default_tone_mapping: ToneMappingMethod = ToneMappingMethod.REINHARD):
        """
        Initialize the HDR merger.

        Args:
            default_tone_mapping: Default tone mapping method
        """
        self.default_tone_mapping = default_tone_mapping

    def _load_images(
        self,
        images: List[Union[str, Path, Image.Image]]
    ) -> List[Image.Image]:
        """Load multiple images."""
        loaded = []
        for img in images:
            if isinstance(img, Image.Image):
                loaded.append(img)
            else:
                loaded.append(Image.open(str(img)))
        return loaded

    def _align_images(self, images: List[Image.Image]) -> List[Image.Image]:
        """Align images to the first image size."""
        if not images:
            return []

        reference_size = images[0].size
        aligned = []

        for img in images:
            if img.size != reference_size:
                img = img.resize(reference_size, Image.Resampling.LANCZOS)
            aligned.append(img)

        return aligned

    def _merge_exposures_simple(self, images: List[Image.Image]) -> Image.Image:
        """Simple exposure fusion without OpenCV."""
        try:
            import numpy as np

            # Convert all images to arrays
            arrays = [np.array(img, dtype=np.float64) for img in images]

            # Calculate weights based on well-exposedness
            weights = []
            for arr in arrays:
                # Convert to grayscale for weight calculation
                if len(arr.shape) == 3:
                    gray = np.mean(arr, axis=2)
                else:
                    gray = arr

                # Well-exposed pixels are closer to mid-gray (128)
                weight = np.exp(-((gray - 128) ** 2) / (2 * 64 ** 2))
                weights.append(weight)

            # Normalize weights
            weight_sum = np.sum(weights, axis=0)
            weight_sum[weight_sum == 0] = 1  # Avoid division by zero

            # Weighted blend
            result = np.zeros_like(arrays[0])
            for arr, weight in zip(arrays, weights):
                if len(arr.shape) == 3:
                    for c in range(arr.shape[2]):
                        result[:, :, c] += arr[:, :, c] * weight
                else:
                    result += arr * weight

            if len(result.shape) == 3:
                for c in range(result.shape[2]):
                    result[:, :, c] /= weight_sum
            else:
                result /= weight_sum

            # Clip and convert back
            result = np.clip(result, 0, 255).astype(np.uint8)

            return Image.fromarray(result, mode=images[0].mode)

        except ImportError:
            # Ultra simple fallback - average
            logger.warning("numpy not available, using simple average")
            result = images[0].copy()
            for img in images[1:]:
                result = Image.blend(result, img, 0.5)
            return result

    def _apply_tone_mapping(
        self,
        image: Image.Image,
        method: ToneMappingMethod,
        gamma: float = 1.0
    ) -> Image.Image:
        """Apply tone mapping to HDR result."""
        try:
            import numpy as np

            arr = np.array(image, dtype=np.float64) / 255.0

            if method == ToneMappingMethod.REINHARD:
                # Reinhard tone mapping
                luminance = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
                luminance_mapped = luminance / (1 + luminance)

                # Apply to each channel
                for c in range(3):
                    arr[:, :, c] = arr[:, :, c] * (luminance_mapped / (luminance + 1e-10))

            elif method == ToneMappingMethod.DRAGO:
                # Simplified Drago tone mapping
                luminance = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
                bias = 0.85
                max_lum = np.max(luminance)

                luminance_mapped = np.log(1 + luminance) / np.log(1 + max_lum) * \
                                   (np.log(2 + 8 * ((luminance / max_lum) ** (np.log(bias) / np.log(0.5)))) / np.log(10))

                for c in range(3):
                    arr[:, :, c] = arr[:, :, c] * (luminance_mapped / (luminance + 1e-10))

            elif method == ToneMappingMethod.MANTIUK:
                # Simplified Mantiuk-like contrast mapping
                arr = np.power(arr, 0.8)

            elif method == ToneMappingMethod.LINEAR:
                # Simple linear scaling
                arr = arr / np.max(arr) if np.max(arr) > 0 else arr

            # Apply gamma correction
            if gamma != 1.0:
                arr = np.power(np.clip(arr, 0, 1), 1.0 / gamma)

            # Clip and convert
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)

            return Image.fromarray(arr, mode=image.mode)

        except ImportError:
            logger.warning("numpy not available, returning unprocessed image")
            return image

    def merge(
        self,
        images: List[Union[str, Path, Image.Image]],
        output: Union[str, Path],
        tone_mapping: Optional[ToneMappingMethod] = None,
        gamma: float = 1.0,
        align: bool = True,
        quality: int = 95
    ) -> HDRResult:
        """
        Merge multiple exposures into an HDR image.

        Args:
            images: List of input images (paths or PIL Images)
            output: Output path
            tone_mapping: Tone mapping method
            gamma: Gamma correction value
            align: Whether to align images
            quality: JPEG quality (1-100)

        Returns:
            HDRResult with merge details
        """
        try:
            if len(images) < 2:
                return HDRResult(
                    success=False,
                    input_count=len(images),
                    output_size=(0, 0),
                    tone_mapping='none',
                    error="At least 2 images required for HDR merge"
                )

            # Load images
            loaded_images = self._load_images(images)

            # Align if requested
            if align:
                loaded_images = self._align_images(loaded_images)

            # Try OpenCV HDR if available
            try:
                import cv2
                import numpy as np

                # Convert to OpenCV format
                cv_images = []
                for img in loaded_images:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    cv_images.append(cv_img)

                # Create exposure times (assume even spacing if not known)
                exposure_times = np.array([0.25, 1.0, 4.0][:len(cv_images)], dtype=np.float32)
                if len(exposure_times) < len(cv_images):
                    exposure_times = np.geomspace(0.1, 10.0, len(cv_images), dtype=np.float32)

                # Merge using Mertens fusion (doesn't require exposure times)
                merge_mertens = cv2.createMergeMertens()
                hdr = merge_mertens.process(cv_images)

                # Tone map
                tone_map_method = tone_mapping or self.default_tone_mapping

                if tone_map_method == ToneMappingMethod.REINHARD:
                    tonemap = cv2.createTonemapReinhard(gamma=gamma)
                elif tone_map_method == ToneMappingMethod.DRAGO:
                    tonemap = cv2.createTonemapDrago(gamma=gamma)
                elif tone_map_method == ToneMappingMethod.MANTIUK:
                    tonemap = cv2.createTonemapMantiuk(gamma=gamma)
                else:
                    tonemap = cv2.createTonemap(gamma=gamma)

                ldr = tonemap.process(hdr)
                ldr = np.clip(ldr * 255, 0, 255).astype(np.uint8)

                # Convert back to PIL
                result_img = Image.fromarray(cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB))

            except ImportError:
                # Fall back to simple method
                logger.info("OpenCV not available, using simple HDR merge")
                tone_map_method = tone_mapping or self.default_tone_mapping

                # Merge exposures
                merged = self._merge_exposures_simple(loaded_images)

                # Apply tone mapping
                result_img = self._apply_tone_mapping(merged, tone_map_method, gamma)

            # Save result
            output_path = Path(output)

            save_kwargs = {}
            ext = output_path.suffix.lower()

            if ext in ['.jpg', '.jpeg']:
                if result_img.mode in ('RGBA', 'P'):
                    result_img = result_img.convert('RGB')
                save_kwargs['quality'] = quality
                save_kwargs['optimize'] = True
            elif ext == '.png':
                save_kwargs['optimize'] = True
            elif ext == '.webp':
                save_kwargs['quality'] = quality

            result_img.save(str(output_path), **save_kwargs)

            return HDRResult(
                success=True,
                input_count=len(images),
                output_size=result_img.size,
                tone_mapping=(tone_mapping or self.default_tone_mapping).value,
                output_path=str(output_path),
                metadata={
                    'gamma': gamma,
                    'aligned': align,
                }
            )

        except Exception as e:
            logger.error(f"HDR merge error: {e}")
            return HDRResult(
                success=False,
                input_count=len(images),
                output_size=(0, 0),
                tone_mapping='unknown',
                error=str(e)
            )

    def create_pseudo_hdr(
        self,
        image: Union[str, Path, Image.Image],
        output: Union[str, Path],
        strength: float = 0.5,
        quality: int = 95
    ) -> HDRResult:
        """
        Create a pseudo-HDR effect from a single image.

        Args:
            image: Input image (path or PIL Image)
            output: Output path
            strength: Effect strength (0-1)
            quality: JPEG quality (1-100)

        Returns:
            HDRResult with processing details
        """
        try:
            if isinstance(image, Image.Image):
                img = image
            else:
                img = Image.open(str(image))

            try:
                import numpy as np
                from PIL import ImageEnhance

                arr = np.array(img, dtype=np.float64)

                # Increase local contrast
                from scipy import ndimage

                if len(arr.shape) == 3:
                    gray = np.mean(arr, axis=2)
                else:
                    gray = arr

                # Calculate local mean
                local_mean = ndimage.uniform_filter(gray, size=50)

                # Apply local contrast
                if len(arr.shape) == 3:
                    for c in range(arr.shape[2]):
                        diff = arr[:, :, c] - local_mean
                        arr[:, :, c] = arr[:, :, c] + diff * strength
                else:
                    diff = arr - local_mean
                    arr = arr + diff * strength

                arr = np.clip(arr, 0, 255).astype(np.uint8)
                result_img = Image.fromarray(arr, mode=img.mode)

                # Boost saturation
                enhancer = ImageEnhance.Color(result_img)
                result_img = enhancer.enhance(1.0 + strength * 0.3)

                # Increase contrast
                enhancer = ImageEnhance.Contrast(result_img)
                result_img = enhancer.enhance(1.0 + strength * 0.2)

            except ImportError:
                # Simple fallback
                from PIL import ImageEnhance

                result_img = img

                enhancer = ImageEnhance.Contrast(result_img)
                result_img = enhancer.enhance(1.0 + strength * 0.5)

                enhancer = ImageEnhance.Color(result_img)
                result_img = enhancer.enhance(1.0 + strength * 0.3)

            # Save result
            output_path = Path(output)

            save_kwargs = {}
            ext = output_path.suffix.lower()

            if ext in ['.jpg', '.jpeg']:
                if result_img.mode in ('RGBA', 'P'):
                    result_img = result_img.convert('RGB')
                save_kwargs['quality'] = quality
            elif ext == '.webp':
                save_kwargs['quality'] = quality

            result_img.save(str(output_path), **save_kwargs)

            return HDRResult(
                success=True,
                input_count=1,
                output_size=result_img.size,
                tone_mapping='pseudo_hdr',
                output_path=str(output_path),
                metadata={
                    'strength': strength,
                    'type': 'pseudo_hdr',
                }
            )

        except Exception as e:
            logger.error(f"Pseudo HDR error: {e}")
            return HDRResult(
                success=False,
                input_count=1,
                output_size=(0, 0),
                tone_mapping='pseudo_hdr',
                error=str(e)
            )

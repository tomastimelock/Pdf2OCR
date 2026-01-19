"""
Image enhancement operations.

Provides auto-enhancement, noise reduction, sharpening,
and other enhancement capabilities.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any, List
import logging

from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)


class EnhancePreset(Enum):
    """Preset enhancement profiles."""
    AUTO = "auto"
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    NIGHT = "night"
    HDR_LOOK = "hdr_look"
    VINTAGE = "vintage"
    VIVID = "vivid"
    SOFT = "soft"


@dataclass
class EnhanceResult:
    """Result of an enhancement operation."""
    success: bool
    original_size: Tuple[int, int]
    new_size: Tuple[int, int]
    enhancements_applied: List[str]
    output_path: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'success': self.success,
            'original_size': self.original_size,
            'new_size': self.new_size,
            'enhancements_applied': self.enhancements_applied,
            'output_path': self.output_path,
            'error': self.error,
            'metadata': self.metadata,
        }


class ImageEnhancer:
    """
    Image enhancement engine.

    Provides various enhancement operations including
    auto-enhancement, noise reduction, and sharpening.
    """

    # Preset configurations
    PRESETS = {
        EnhancePreset.AUTO: {
            'brightness': 'auto',
            'contrast': 'auto',
            'saturation': 1.1,
            'sharpness': 1.1,
        },
        EnhancePreset.PORTRAIT: {
            'brightness': 1.05,
            'contrast': 1.0,
            'saturation': 0.95,
            'sharpness': 0.9,
            'smooth': 0.3,
        },
        EnhancePreset.LANDSCAPE: {
            'brightness': 1.0,
            'contrast': 1.15,
            'saturation': 1.2,
            'sharpness': 1.2,
        },
        EnhancePreset.NIGHT: {
            'brightness': 1.3,
            'contrast': 1.1,
            'saturation': 0.9,
            'noise_reduction': 0.5,
        },
        EnhancePreset.HDR_LOOK: {
            'brightness': 1.0,
            'contrast': 1.4,
            'saturation': 1.3,
            'sharpness': 1.3,
            'clarity': 0.5,
        },
        EnhancePreset.VINTAGE: {
            'brightness': 0.95,
            'contrast': 0.9,
            'saturation': 0.7,
            'warmth': 0.3,
        },
        EnhancePreset.VIVID: {
            'brightness': 1.05,
            'contrast': 1.2,
            'saturation': 1.4,
            'sharpness': 1.1,
        },
        EnhancePreset.SOFT: {
            'brightness': 1.05,
            'contrast': 0.9,
            'saturation': 0.9,
            'sharpness': 0.8,
            'smooth': 0.2,
        },
    }

    def __init__(self):
        """Initialize the enhancer."""
        pass

    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """Load image from path or return if already an Image."""
        if isinstance(image, Image.Image):
            return image
        return Image.open(str(image))

    def _calculate_auto_brightness(self, image: Image.Image) -> float:
        """Calculate automatic brightness adjustment."""
        gray = image.convert('L')
        hist = gray.histogram()

        # Calculate mean brightness
        total_pixels = sum(hist)
        mean_brightness = sum(i * count for i, count in enumerate(hist)) / total_pixels

        # Target mean brightness around 128
        if mean_brightness < 100:
            return 1.0 + (128 - mean_brightness) / 256
        elif mean_brightness > 156:
            return 1.0 - (mean_brightness - 128) / 256
        return 1.0

    def _calculate_auto_contrast(self, image: Image.Image) -> float:
        """Calculate automatic contrast adjustment."""
        gray = image.convert('L')
        hist = gray.histogram()

        # Find the range of brightness values
        min_val = 0
        max_val = 255

        for i in range(256):
            if hist[i] > 0:
                min_val = i
                break

        for i in range(255, -1, -1):
            if hist[i] > 0:
                max_val = i
                break

        # Calculate contrast based on range
        current_range = max_val - min_val
        if current_range < 200:
            return 1.0 + (200 - current_range) / 400
        return 1.0

    def _apply_noise_reduction(
        self,
        image: Image.Image,
        strength: float = 0.5
    ) -> Image.Image:
        """Apply noise reduction using blur-based method."""
        try:
            import numpy as np
            from scipy import ndimage

            # Convert to array
            arr = np.array(image, dtype=np.float64)

            # Apply bilateral-like filtering using Gaussian
            sigma = strength * 2.0
            if len(arr.shape) == 3:
                for c in range(arr.shape[2]):
                    arr[:, :, c] = ndimage.gaussian_filter(arr[:, :, c], sigma=sigma)
            else:
                arr = ndimage.gaussian_filter(arr, sigma=sigma)

            # Blend with original
            original = np.array(image, dtype=np.float64)
            result = original * (1 - strength) + arr * strength

            return Image.fromarray(result.astype(np.uint8), mode=image.mode)

        except ImportError:
            # Fallback to PIL blur
            blur_radius = int(strength * 2) + 1
            blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

            # Blend with original
            return Image.blend(image, blurred, strength * 0.5)

    def _apply_sharpening(
        self,
        image: Image.Image,
        amount: float = 1.0
    ) -> Image.Image:
        """Apply sharpening."""
        if amount <= 0:
            return image

        # Use unsharp mask for better results
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(amount)

    def _apply_clarity(
        self,
        image: Image.Image,
        amount: float = 0.5
    ) -> Image.Image:
        """Apply clarity enhancement (local contrast)."""
        try:
            import numpy as np
            from scipy import ndimage

            arr = np.array(image, dtype=np.float64)

            if len(arr.shape) == 3:
                # Convert to grayscale for mask
                gray = np.mean(arr, axis=2)
            else:
                gray = arr

            # Calculate local mean
            local_mean = ndimage.uniform_filter(gray, size=50)

            # Apply local contrast
            if len(arr.shape) == 3:
                for c in range(arr.shape[2]):
                    diff = arr[:, :, c] - local_mean
                    arr[:, :, c] = arr[:, :, c] + diff * amount
            else:
                diff = arr - local_mean
                arr = arr + diff * amount

            # Clip values
            arr = np.clip(arr, 0, 255)

            return Image.fromarray(arr.astype(np.uint8), mode=image.mode)

        except ImportError:
            # Fallback: just increase contrast
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(1.0 + amount * 0.3)

    def _apply_warmth(
        self,
        image: Image.Image,
        amount: float = 0.3
    ) -> Image.Image:
        """Apply warmth adjustment (color temperature shift)."""
        if image.mode != 'RGB':
            image = image.convert('RGB')

        try:
            import numpy as np

            arr = np.array(image, dtype=np.float64)

            # Increase red/yellow, decrease blue
            arr[:, :, 0] = np.clip(arr[:, :, 0] * (1 + amount * 0.1), 0, 255)
            arr[:, :, 1] = np.clip(arr[:, :, 1] * (1 + amount * 0.05), 0, 255)
            arr[:, :, 2] = np.clip(arr[:, :, 2] * (1 - amount * 0.1), 0, 255)

            return Image.fromarray(arr.astype(np.uint8), mode='RGB')

        except ImportError:
            # Fallback: slight saturation increase
            enhancer = ImageEnhance.Color(image)
            return enhancer.enhance(1.0 + amount * 0.2)

    def _apply_smoothing(
        self,
        image: Image.Image,
        amount: float = 0.3
    ) -> Image.Image:
        """Apply skin smoothing effect."""
        # Use slight blur for smoothing
        blur_radius = amount * 3
        blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return Image.blend(image, blurred, amount * 0.5)

    def enhance(
        self,
        image: Union[str, Path, Image.Image],
        brightness: Optional[float] = None,
        contrast: Optional[float] = None,
        saturation: Optional[float] = None,
        sharpness: Optional[float] = None,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[EnhanceResult, Tuple[Image.Image, EnhanceResult]]:
        """
        Enhance an image with specified adjustments.

        Args:
            image: Input image (path or PIL Image)
            brightness: Brightness factor (1.0 = no change)
            contrast: Contrast factor (1.0 = no change)
            saturation: Saturation factor (1.0 = no change)
            sharpness: Sharpness factor (1.0 = no change)
            output: Output path (if None, returns PIL Image)
            quality: JPEG quality (1-100)

        Returns:
            EnhanceResult if output path provided, otherwise (Image, EnhanceResult)
        """
        try:
            img = self._load_image(image)
            original_size = img.size
            enhancements = []

            # Ensure RGB mode for color adjustments
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')

            # Apply brightness
            if brightness is not None and brightness != 1.0:
                if brightness == 'auto':
                    brightness = self._calculate_auto_brightness(img)
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(brightness)
                enhancements.append(f'brightness:{brightness:.2f}')

            # Apply contrast
            if contrast is not None and contrast != 1.0:
                if contrast == 'auto':
                    contrast = self._calculate_auto_contrast(img)
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(contrast)
                enhancements.append(f'contrast:{contrast:.2f}')

            # Apply saturation
            if saturation is not None and saturation != 1.0:
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(saturation)
                enhancements.append(f'saturation:{saturation:.2f}')

            # Apply sharpness
            if sharpness is not None and sharpness != 1.0:
                img = self._apply_sharpening(img, sharpness)
                enhancements.append(f'sharpness:{sharpness:.2f}')

            result = EnhanceResult(
                success=True,
                original_size=original_size,
                new_size=img.size,
                enhancements_applied=enhancements,
            )

            if output:
                output_path = Path(output)

                save_kwargs = {}
                ext = output_path.suffix.lower()

                if ext in ['.jpg', '.jpeg']:
                    if img.mode in ('RGBA', 'P'):
                        img = img.convert('RGB')
                    save_kwargs['quality'] = quality
                    save_kwargs['optimize'] = True
                elif ext == '.png':
                    save_kwargs['optimize'] = True
                elif ext == '.webp':
                    save_kwargs['quality'] = quality

                img.save(str(output_path), **save_kwargs)
                result.output_path = str(output_path)
                return result
            else:
                return (img, result)

        except Exception as e:
            logger.error(f"Enhance error: {e}")
            return EnhanceResult(
                success=False,
                original_size=(0, 0),
                new_size=(0, 0),
                enhancements_applied=[],
                error=str(e)
            )

    def auto_enhance(
        self,
        image: Union[str, Path, Image.Image],
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[EnhanceResult, Tuple[Image.Image, EnhanceResult]]:
        """
        Automatically enhance image based on analysis.

        Args:
            image: Input image (path or PIL Image)
            output: Output path (if None, returns PIL Image)
            quality: JPEG quality (1-100)

        Returns:
            EnhanceResult if output path provided, otherwise (Image, EnhanceResult)
        """
        try:
            img = self._load_image(image)
            original_size = img.size
            enhancements = []

            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')

            # Auto brightness
            brightness = self._calculate_auto_brightness(img)
            if abs(brightness - 1.0) > 0.05:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(brightness)
                enhancements.append(f'auto_brightness:{brightness:.2f}')

            # Auto contrast
            contrast = self._calculate_auto_contrast(img)
            if abs(contrast - 1.0) > 0.05:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(contrast)
                enhancements.append(f'auto_contrast:{contrast:.2f}')

            # Slight saturation boost
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.1)
            enhancements.append('saturation:1.10')

            # Slight sharpening
            img = self._apply_sharpening(img, 1.1)
            enhancements.append('sharpness:1.10')

            result = EnhanceResult(
                success=True,
                original_size=original_size,
                new_size=img.size,
                enhancements_applied=enhancements,
                metadata={'preset': 'auto'}
            )

            if output:
                output_path = Path(output)

                save_kwargs = {}
                ext = output_path.suffix.lower()

                if ext in ['.jpg', '.jpeg']:
                    if img.mode in ('RGBA', 'P'):
                        img = img.convert('RGB')
                    save_kwargs['quality'] = quality
                    save_kwargs['optimize'] = True
                elif ext == '.png':
                    save_kwargs['optimize'] = True
                elif ext == '.webp':
                    save_kwargs['quality'] = quality

                img.save(str(output_path), **save_kwargs)
                result.output_path = str(output_path)
                return result
            else:
                return (img, result)

        except Exception as e:
            logger.error(f"Auto-enhance error: {e}")
            return EnhanceResult(
                success=False,
                original_size=(0, 0),
                new_size=(0, 0),
                enhancements_applied=[],
                error=str(e)
            )

    def apply_preset(
        self,
        image: Union[str, Path, Image.Image],
        preset: EnhancePreset,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[EnhanceResult, Tuple[Image.Image, EnhanceResult]]:
        """
        Apply a preset enhancement profile.

        Args:
            image: Input image (path or PIL Image)
            preset: Enhancement preset
            output: Output path (if None, returns PIL Image)
            quality: JPEG quality (1-100)

        Returns:
            EnhanceResult if output path provided, otherwise (Image, EnhanceResult)
        """
        try:
            img = self._load_image(image)
            original_size = img.size
            enhancements = [f'preset:{preset.value}']

            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')

            config = self.PRESETS.get(preset, {})

            # Apply each enhancement in the preset
            for key, value in config.items():
                if key == 'brightness':
                    if value == 'auto':
                        value = self._calculate_auto_brightness(img)
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(value)
                    enhancements.append(f'brightness:{value:.2f}')

                elif key == 'contrast':
                    if value == 'auto':
                        value = self._calculate_auto_contrast(img)
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(value)
                    enhancements.append(f'contrast:{value:.2f}')

                elif key == 'saturation':
                    enhancer = ImageEnhance.Color(img)
                    img = enhancer.enhance(value)
                    enhancements.append(f'saturation:{value:.2f}')

                elif key == 'sharpness':
                    img = self._apply_sharpening(img, value)
                    enhancements.append(f'sharpness:{value:.2f}')

                elif key == 'noise_reduction':
                    img = self._apply_noise_reduction(img, value)
                    enhancements.append(f'noise_reduction:{value:.2f}')

                elif key == 'clarity':
                    img = self._apply_clarity(img, value)
                    enhancements.append(f'clarity:{value:.2f}')

                elif key == 'warmth':
                    img = self._apply_warmth(img, value)
                    enhancements.append(f'warmth:{value:.2f}')

                elif key == 'smooth':
                    img = self._apply_smoothing(img, value)
                    enhancements.append(f'smooth:{value:.2f}')

            result = EnhanceResult(
                success=True,
                original_size=original_size,
                new_size=img.size,
                enhancements_applied=enhancements,
                metadata={'preset': preset.value}
            )

            if output:
                output_path = Path(output)

                save_kwargs = {}
                ext = output_path.suffix.lower()

                if ext in ['.jpg', '.jpeg']:
                    if img.mode in ('RGBA', 'P'):
                        img = img.convert('RGB')
                    save_kwargs['quality'] = quality
                    save_kwargs['optimize'] = True
                elif ext == '.png':
                    save_kwargs['optimize'] = True
                elif ext == '.webp':
                    save_kwargs['quality'] = quality

                img.save(str(output_path), **save_kwargs)
                result.output_path = str(output_path)
                return result
            else:
                return (img, result)

        except Exception as e:
            logger.error(f"Preset error: {e}")
            return EnhanceResult(
                success=False,
                original_size=(0, 0),
                new_size=(0, 0),
                enhancements_applied=[],
                error=str(e)
            )

    def reduce_noise(
        self,
        image: Union[str, Path, Image.Image],
        strength: float = 0.5,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[EnhanceResult, Tuple[Image.Image, EnhanceResult]]:
        """Apply noise reduction."""
        try:
            img = self._load_image(image)
            original_size = img.size

            img = self._apply_noise_reduction(img, strength)

            result = EnhanceResult(
                success=True,
                original_size=original_size,
                new_size=img.size,
                enhancements_applied=[f'noise_reduction:{strength:.2f}'],
            )

            if output:
                output_path = Path(output)

                save_kwargs = {}
                ext = output_path.suffix.lower()

                if ext in ['.jpg', '.jpeg']:
                    if img.mode in ('RGBA', 'P'):
                        img = img.convert('RGB')
                    save_kwargs['quality'] = quality
                elif ext == '.webp':
                    save_kwargs['quality'] = quality

                img.save(str(output_path), **save_kwargs)
                result.output_path = str(output_path)
                return result
            else:
                return (img, result)

        except Exception as e:
            logger.error(f"Noise reduction error: {e}")
            return EnhanceResult(
                success=False,
                original_size=(0, 0),
                new_size=(0, 0),
                enhancements_applied=[],
                error=str(e)
            )

    def sharpen(
        self,
        image: Union[str, Path, Image.Image],
        amount: float = 1.0,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[EnhanceResult, Tuple[Image.Image, EnhanceResult]]:
        """Apply sharpening."""
        try:
            img = self._load_image(image)
            original_size = img.size

            img = self._apply_sharpening(img, amount)

            result = EnhanceResult(
                success=True,
                original_size=original_size,
                new_size=img.size,
                enhancements_applied=[f'sharpness:{amount:.2f}'],
            )

            if output:
                output_path = Path(output)

                save_kwargs = {}
                ext = output_path.suffix.lower()

                if ext in ['.jpg', '.jpeg']:
                    if img.mode in ('RGBA', 'P'):
                        img = img.convert('RGB')
                    save_kwargs['quality'] = quality
                elif ext == '.webp':
                    save_kwargs['quality'] = quality

                img.save(str(output_path), **save_kwargs)
                result.output_path = str(output_path)
                return result
            else:
                return (img, result)

        except Exception as e:
            logger.error(f"Sharpen error: {e}")
            return EnhanceResult(
                success=False,
                original_size=(0, 0),
                new_size=(0, 0),
                enhancements_applied=[],
                error=str(e)
            )

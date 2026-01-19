"""
Image filter operations.

Provides various image filters including color adjustments,
artistic effects, and stylization.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any, List
import logging

from PIL import Image, ImageFilter, ImageOps, ImageEnhance

logger = logging.getLogger(__name__)


class FilterType(Enum):
    """Available filter types."""
    # Basic adjustments
    GRAYSCALE = "grayscale"
    SEPIA = "sepia"
    INVERT = "invert"

    # Color effects
    WARM = "warm"
    COOL = "cool"
    VIBRANT = "vibrant"
    MUTED = "muted"

    # Artistic effects
    BLUR = "blur"
    SHARPEN = "sharpen"
    EDGE = "edge"
    EMBOSS = "emboss"
    CONTOUR = "contour"

    # Stylization
    VINTAGE = "vintage"
    DRAMATIC = "dramatic"
    DREAMY = "dreamy"
    NOIR = "noir"
    POP_ART = "pop_art"
    POSTERIZE = "posterize"
    SOLARIZE = "solarize"


@dataclass
class FilterResult:
    """Result of a filter operation."""
    success: bool
    filter_applied: str
    original_size: Tuple[int, int]
    new_size: Tuple[int, int]
    intensity: float
    output_path: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'success': self.success,
            'filter_applied': self.filter_applied,
            'original_size': self.original_size,
            'new_size': self.new_size,
            'intensity': self.intensity,
            'output_path': self.output_path,
            'error': self.error,
            'metadata': self.metadata,
        }


class FilterEngine:
    """
    Image filter engine.

    Provides various image filters with adjustable intensity.
    """

    def __init__(self):
        """Initialize the filter engine."""
        pass

    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """Load image from path or return if already an Image."""
        if isinstance(image, Image.Image):
            return image
        return Image.open(str(image))

    def _apply_grayscale(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply grayscale filter."""
        gray = ImageOps.grayscale(image)
        if image.mode == 'RGBA':
            gray = gray.convert('RGBA')
        elif image.mode != 'L':
            gray = gray.convert('RGB')

        if intensity < 1.0:
            return Image.blend(image.convert(gray.mode), gray, intensity)
        return gray

    def _apply_sepia(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply sepia tone filter."""
        try:
            import numpy as np

            if image.mode != 'RGB':
                image = image.convert('RGB')

            arr = np.array(image, dtype=np.float64)

            # Sepia transformation matrix
            sepia_matrix = np.array([
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131]
            ])

            # Apply transformation
            sepia_arr = np.dot(arr, sepia_matrix.T)
            sepia_arr = np.clip(sepia_arr, 0, 255).astype(np.uint8)

            sepia_img = Image.fromarray(sepia_arr, mode='RGB')

            if intensity < 1.0:
                return Image.blend(image, sepia_img, intensity)
            return sepia_img

        except ImportError:
            # Fallback without numpy
            gray = ImageOps.grayscale(image)
            sepia = ImageOps.colorize(gray, '#704214', '#f0e68c')
            if intensity < 1.0:
                return Image.blend(image.convert('RGB'), sepia, intensity)
            return sepia

    def _apply_invert(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply color inversion."""
        if image.mode == 'RGBA':
            r, g, b, a = image.split()
            rgb = Image.merge('RGB', (r, g, b))
            inverted = ImageOps.invert(rgb)
            ir, ig, ib = inverted.split()
            inverted = Image.merge('RGBA', (ir, ig, ib, a))
        else:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            inverted = ImageOps.invert(image)

        if intensity < 1.0:
            return Image.blend(image, inverted, intensity)
        return inverted

    def _apply_warm(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply warm color temperature."""
        try:
            import numpy as np

            if image.mode not in ('RGB', 'RGBA'):
                image = image.convert('RGB')

            arr = np.array(image, dtype=np.float64)

            # Increase red/yellow
            warmth = intensity * 30
            arr[:, :, 0] = np.clip(arr[:, :, 0] + warmth, 0, 255)
            arr[:, :, 2] = np.clip(arr[:, :, 2] - warmth * 0.5, 0, 255)

            return Image.fromarray(arr.astype(np.uint8), mode=image.mode)

        except ImportError:
            # Fallback using color enhancement
            enhancer = ImageEnhance.Color(image)
            return enhancer.enhance(1.0 + intensity * 0.2)

    def _apply_cool(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply cool color temperature."""
        try:
            import numpy as np

            if image.mode not in ('RGB', 'RGBA'):
                image = image.convert('RGB')

            arr = np.array(image, dtype=np.float64)

            # Increase blue
            coolness = intensity * 30
            arr[:, :, 0] = np.clip(arr[:, :, 0] - coolness * 0.5, 0, 255)
            arr[:, :, 2] = np.clip(arr[:, :, 2] + coolness, 0, 255)

            return Image.fromarray(arr.astype(np.uint8), mode=image.mode)

        except ImportError:
            # Fallback
            enhancer = ImageEnhance.Color(image)
            return enhancer.enhance(1.0 - intensity * 0.1)

    def _apply_vibrant(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply vibrant colors."""
        enhancer = ImageEnhance.Color(image)
        enhanced = enhancer.enhance(1.0 + intensity * 0.5)

        enhancer = ImageEnhance.Contrast(enhanced)
        return enhancer.enhance(1.0 + intensity * 0.2)

    def _apply_muted(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply muted colors."""
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(1.0 - intensity * 0.4)

    def _apply_blur(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply blur effect."""
        radius = intensity * 5
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def _apply_sharpen(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply sharpen effect."""
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(1.0 + intensity * 2)

    def _apply_edge(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply edge detection."""
        edges = image.filter(ImageFilter.FIND_EDGES)
        if intensity < 1.0:
            return Image.blend(image, edges.convert(image.mode), intensity)
        return edges

    def _apply_emboss(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply emboss effect."""
        embossed = image.filter(ImageFilter.EMBOSS)
        if intensity < 1.0:
            return Image.blend(image, embossed.convert(image.mode), intensity)
        return embossed

    def _apply_contour(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply contour effect."""
        contour = image.filter(ImageFilter.CONTOUR)
        if intensity < 1.0:
            return Image.blend(image, contour.convert(image.mode), intensity)
        return contour

    def _apply_vintage(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply vintage effect."""
        # Slight sepia
        result = self._apply_sepia(image, intensity * 0.4)

        # Reduce contrast
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(1.0 - intensity * 0.1)

        # Add slight warmth
        result = self._apply_warm(result, intensity * 0.3)

        return result

    def _apply_dramatic(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply dramatic effect."""
        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        result = enhancer.enhance(1.0 + intensity * 0.5)

        # Increase sharpness
        enhancer = ImageEnhance.Sharpness(result)
        result = enhancer.enhance(1.0 + intensity * 0.3)

        # Slightly desaturate
        enhancer = ImageEnhance.Color(result)
        result = enhancer.enhance(1.0 - intensity * 0.2)

        return result

    def _apply_dreamy(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply dreamy soft effect."""
        # Slight blur
        blurred = image.filter(ImageFilter.GaussianBlur(radius=intensity * 3))

        # Blend with original
        result = Image.blend(image, blurred, intensity * 0.3)

        # Increase brightness
        enhancer = ImageEnhance.Brightness(result)
        result = enhancer.enhance(1.0 + intensity * 0.1)

        # Slight desaturation
        enhancer = ImageEnhance.Color(result)
        result = enhancer.enhance(1.0 - intensity * 0.1)

        return result

    def _apply_noir(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply film noir effect."""
        # Convert to grayscale
        result = self._apply_grayscale(image, intensity)

        # Increase contrast
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(1.0 + intensity * 0.4)

        return result

    def _apply_pop_art(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply pop art effect."""
        # Posterize
        if image.mode != 'RGB':
            image = image.convert('RGB')

        bits = max(1, int(8 - intensity * 6))
        result = ImageOps.posterize(image, bits)

        # Increase saturation
        enhancer = ImageEnhance.Color(result)
        result = enhancer.enhance(1.0 + intensity * 0.8)

        return result

    def _apply_posterize(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply posterize effect."""
        if image.mode == 'RGBA':
            r, g, b, a = image.split()
            rgb = Image.merge('RGB', (r, g, b))
            bits = max(1, int(8 - intensity * 6))
            posterized = ImageOps.posterize(rgb, bits)
            pr, pg, pb = posterized.split()
            return Image.merge('RGBA', (pr, pg, pb, a))
        else:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            bits = max(1, int(8 - intensity * 6))
            return ImageOps.posterize(image, bits)

    def _apply_solarize(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply solarize effect."""
        threshold = int(255 * (1 - intensity))
        if image.mode == 'RGBA':
            r, g, b, a = image.split()
            rgb = Image.merge('RGB', (r, g, b))
            solarized = ImageOps.solarize(rgb, threshold=threshold)
            sr, sg, sb = solarized.split()
            return Image.merge('RGBA', (sr, sg, sb, a))
        else:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return ImageOps.solarize(image, threshold=threshold)

    def apply_filter(
        self,
        image: Union[str, Path, Image.Image],
        filter_type: Union[str, FilterType],
        intensity: float = 1.0,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[FilterResult, Tuple[Image.Image, FilterResult]]:
        """
        Apply a filter to an image.

        Args:
            image: Input image (path or PIL Image)
            filter_type: Type of filter to apply
            intensity: Filter intensity (0-1)
            output: Output path (if None, returns PIL Image)
            quality: JPEG quality (1-100)

        Returns:
            FilterResult if output path provided, otherwise (Image, FilterResult)
        """
        try:
            img = self._load_image(image)
            original_size = img.size

            # Parse filter type
            if isinstance(filter_type, str):
                filter_type = FilterType(filter_type.lower())

            # Clamp intensity
            intensity = max(0.0, min(1.0, intensity))

            # Apply filter
            filter_map = {
                FilterType.GRAYSCALE: self._apply_grayscale,
                FilterType.SEPIA: self._apply_sepia,
                FilterType.INVERT: self._apply_invert,
                FilterType.WARM: self._apply_warm,
                FilterType.COOL: self._apply_cool,
                FilterType.VIBRANT: self._apply_vibrant,
                FilterType.MUTED: self._apply_muted,
                FilterType.BLUR: self._apply_blur,
                FilterType.SHARPEN: self._apply_sharpen,
                FilterType.EDGE: self._apply_edge,
                FilterType.EMBOSS: self._apply_emboss,
                FilterType.CONTOUR: self._apply_contour,
                FilterType.VINTAGE: self._apply_vintage,
                FilterType.DRAMATIC: self._apply_dramatic,
                FilterType.DREAMY: self._apply_dreamy,
                FilterType.NOIR: self._apply_noir,
                FilterType.POP_ART: self._apply_pop_art,
                FilterType.POSTERIZE: self._apply_posterize,
                FilterType.SOLARIZE: self._apply_solarize,
            }

            filter_func = filter_map.get(filter_type)
            if filter_func:
                result_img = filter_func(img, intensity)
            else:
                result_img = img

            result = FilterResult(
                success=True,
                filter_applied=filter_type.value,
                original_size=original_size,
                new_size=result_img.size,
                intensity=intensity,
            )

            if output:
                output_path = Path(output)

                save_kwargs = {}
                ext = output_path.suffix.lower()

                if ext in ['.jpg', '.jpeg']:
                    if result_img.mode in ('RGBA', 'P', 'L', 'LA'):
                        result_img = result_img.convert('RGB')
                    save_kwargs['quality'] = quality
                    save_kwargs['optimize'] = True
                elif ext == '.png':
                    save_kwargs['optimize'] = True
                elif ext == '.webp':
                    save_kwargs['quality'] = quality

                result_img.save(str(output_path), **save_kwargs)
                result.output_path = str(output_path)
                return result
            else:
                return (result_img, result)

        except Exception as e:
            logger.error(f"Filter error: {e}")
            return FilterResult(
                success=False,
                filter_applied=str(filter_type),
                original_size=(0, 0),
                new_size=(0, 0),
                intensity=intensity,
                error=str(e)
            )

    def list_filters(self) -> List[str]:
        """Get list of available filters."""
        return [f.value for f in FilterType]

    def apply_multiple(
        self,
        image: Union[str, Path, Image.Image],
        filters: List[Tuple[Union[str, FilterType], float]],
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[FilterResult, Tuple[Image.Image, FilterResult]]:
        """
        Apply multiple filters in sequence.

        Args:
            image: Input image
            filters: List of (filter_type, intensity) tuples
            output: Output path
            quality: JPEG quality

        Returns:
            FilterResult with all applied filters
        """
        try:
            img = self._load_image(image)
            original_size = img.size
            applied_filters = []

            for filter_type, intensity in filters:
                result = self.apply_filter(img, filter_type, intensity)
                if isinstance(result, tuple):
                    img, res = result
                    applied_filters.append(f"{filter_type}:{intensity:.2f}")
                else:
                    if result.success:
                        applied_filters.append(f"{filter_type}:{intensity:.2f}")

            result = FilterResult(
                success=True,
                filter_applied=','.join(applied_filters),
                original_size=original_size,
                new_size=img.size,
                intensity=1.0,
                metadata={'filters': applied_filters}
            )

            if output:
                output_path = Path(output)

                save_kwargs = {}
                ext = output_path.suffix.lower()

                if ext in ['.jpg', '.jpeg']:
                    if img.mode in ('RGBA', 'P', 'L', 'LA'):
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
            logger.error(f"Multiple filter error: {e}")
            return FilterResult(
                success=False,
                filter_applied='multiple',
                original_size=(0, 0),
                new_size=(0, 0),
                intensity=0,
                error=str(e)
            )

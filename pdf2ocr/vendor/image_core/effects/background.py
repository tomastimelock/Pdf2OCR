"""
Background removal operations.

Provides background removal and replacement functionality.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any
import logging

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class BackgroundResult:
    """Result of a background operation."""
    success: bool
    original_size: Tuple[int, int]
    new_size: Tuple[int, int]
    operation: str
    has_transparency: bool
    output_path: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'success': self.success,
            'original_size': self.original_size,
            'new_size': self.new_size,
            'operation': self.operation,
            'has_transparency': self.has_transparency,
            'output_path': self.output_path,
            'error': self.error,
            'metadata': self.metadata,
        }


class BackgroundRemover:
    """
    Background removal engine.

    Provides background removal using various methods
    including AI-based and color-based approaches.
    """

    def __init__(self, use_ai: bool = True):
        """
        Initialize the background remover.

        Args:
            use_ai: Whether to use AI-based removal when available
        """
        self.use_ai = use_ai
        self._rembg_session = None

    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """Load image from path or return if already an Image."""
        if isinstance(image, Image.Image):
            return image
        return Image.open(str(image))

    def _remove_background_rembg(self, image: Image.Image) -> Image.Image:
        """Remove background using rembg library."""
        try:
            from rembg import remove, new_session

            if self._rembg_session is None:
                self._rembg_session = new_session()

            return remove(image, session=self._rembg_session)

        except ImportError:
            raise ImportError("rembg not installed. Install with: pip install rembg")

    def _remove_background_color(
        self,
        image: Image.Image,
        color: Tuple[int, int, int],
        tolerance: int = 30
    ) -> Image.Image:
        """Remove background by color matching."""
        try:
            import numpy as np

            if image.mode != 'RGBA':
                image = image.convert('RGBA')

            arr = np.array(image)

            # Calculate distance from target color
            diff = np.abs(arr[:, :, :3].astype(np.int16) - np.array(color))
            distance = np.sum(diff, axis=2)

            # Create mask
            mask = distance > tolerance * 3
            arr[:, :, 3] = np.where(mask, 255, 0)

            return Image.fromarray(arr, mode='RGBA')

        except ImportError:
            # Fallback without numpy
            if image.mode != 'RGBA':
                image = image.convert('RGBA')

            pixels = image.load()
            width, height = image.size

            for y in range(height):
                for x in range(width):
                    r, g, b, a = pixels[x, y]
                    diff = abs(r - color[0]) + abs(g - color[1]) + abs(b - color[2])
                    if diff <= tolerance * 3:
                        pixels[x, y] = (r, g, b, 0)

            return image

    def _remove_background_grabcut(self, image: Image.Image) -> Image.Image:
        """Remove background using GrabCut algorithm."""
        try:
            import cv2
            import numpy as np

            if image.mode != 'RGB':
                image = image.convert('RGB')

            arr = np.array(image)
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            mask = np.zeros(arr.shape[:2], np.uint8)

            # Define rectangle for GrabCut
            h, w = arr.shape[:2]
            margin = int(min(h, w) * 0.05)
            rect = (margin, margin, w - margin * 2, h - margin * 2)

            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            cv2.grabCut(arr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

            # Create mask
            mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')

            # Apply mask
            result = cv2.cvtColor(arr, cv2.COLOR_BGR2RGBA)
            result[:, :, 3] = mask2

            return Image.fromarray(result, mode='RGBA')

        except ImportError:
            raise ImportError("OpenCV not installed. Install with: pip install opencv-python")

    def remove(
        self,
        image: Union[str, Path, Image.Image],
        method: str = 'auto',
        output: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Union[BackgroundResult, Tuple[Image.Image, BackgroundResult]]:
        """
        Remove background from an image.

        Args:
            image: Input image (path or PIL Image)
            method: Removal method ('auto', 'ai', 'color', 'grabcut')
            output: Output path (if None, returns PIL Image)
            **kwargs: Method-specific options

        Returns:
            BackgroundResult if output path provided, otherwise (Image, BackgroundResult)
        """
        try:
            img = self._load_image(image)
            original_size = img.size
            method_used = method

            if method == 'auto':
                # Try AI first, fall back to GrabCut
                if self.use_ai:
                    try:
                        result_img = self._remove_background_rembg(img)
                        method_used = 'ai'
                    except ImportError:
                        logger.info("rembg not available, trying grabcut")
                        try:
                            result_img = self._remove_background_grabcut(img)
                            method_used = 'grabcut'
                        except ImportError:
                            # Last resort: simple edge-based
                            logger.info("OpenCV not available, using simple method")
                            result_img = self._remove_background_color(
                                img, kwargs.get('color', (255, 255, 255)),
                                kwargs.get('tolerance', 30)
                            )
                            method_used = 'color'
                else:
                    try:
                        result_img = self._remove_background_grabcut(img)
                        method_used = 'grabcut'
                    except ImportError:
                        result_img = self._remove_background_color(
                            img, kwargs.get('color', (255, 255, 255)),
                            kwargs.get('tolerance', 30)
                        )
                        method_used = 'color'

            elif method == 'ai':
                result_img = self._remove_background_rembg(img)
                method_used = 'ai'

            elif method == 'color':
                color = kwargs.get('color', (255, 255, 255))
                tolerance = kwargs.get('tolerance', 30)
                result_img = self._remove_background_color(img, color, tolerance)
                method_used = 'color'

            elif method == 'grabcut':
                result_img = self._remove_background_grabcut(img)
                method_used = 'grabcut'

            else:
                result_img = self._remove_background_rembg(img)
                method_used = 'ai'

            result = BackgroundResult(
                success=True,
                original_size=original_size,
                new_size=result_img.size,
                operation='remove',
                has_transparency=result_img.mode == 'RGBA',
                metadata={'method': method_used}
            )

            if output:
                output_path = Path(output)

                # Force PNG for transparency
                ext = output_path.suffix.lower()
                if ext in ['.jpg', '.jpeg']:
                    logger.warning("JPEG doesn't support transparency, saving as PNG")
                    output_path = output_path.with_suffix('.png')

                result_img.save(str(output_path), optimize=True)
                result.output_path = str(output_path)
                return result
            else:
                return (result_img, result)

        except Exception as e:
            logger.error(f"Background removal error: {e}")
            return BackgroundResult(
                success=False,
                original_size=(0, 0),
                new_size=(0, 0),
                operation='remove',
                has_transparency=False,
                error=str(e)
            )

    def replace(
        self,
        image: Union[str, Path, Image.Image],
        background: Union[str, Path, Image.Image, Tuple[int, int, int]],
        method: str = 'auto',
        output: Optional[Union[str, Path]] = None,
        quality: int = 95,
        **kwargs
    ) -> Union[BackgroundResult, Tuple[Image.Image, BackgroundResult]]:
        """
        Replace background with a new one.

        Args:
            image: Input image
            background: New background (image or color)
            method: Removal method
            output: Output path
            quality: JPEG quality
            **kwargs: Method-specific options

        Returns:
            BackgroundResult or (Image, BackgroundResult)
        """
        try:
            # First remove background
            result = self.remove(image, method, **kwargs)

            if isinstance(result, tuple):
                fg_img, remove_result = result
            else:
                if not result.success:
                    return result
                # This shouldn't happen if no output path
                return result

            # Prepare background
            if isinstance(background, tuple):
                # Solid color background
                bg_img = Image.new('RGBA', fg_img.size, (*background, 255))
            else:
                bg_img = self._load_image(background)
                if bg_img.size != fg_img.size:
                    bg_img = bg_img.resize(fg_img.size, Image.Resampling.LANCZOS)
                if bg_img.mode != 'RGBA':
                    bg_img = bg_img.convert('RGBA')

            # Composite
            result_img = Image.alpha_composite(bg_img, fg_img)

            result = BackgroundResult(
                success=True,
                original_size=remove_result.original_size,
                new_size=result_img.size,
                operation='replace',
                has_transparency=False,
                metadata={'method': remove_result.metadata.get('method', 'unknown')}
            )

            if output:
                output_path = Path(output)

                save_kwargs = {}
                ext = output_path.suffix.lower()

                if ext in ['.jpg', '.jpeg']:
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
            logger.error(f"Background replace error: {e}")
            return BackgroundResult(
                success=False,
                original_size=(0, 0),
                new_size=(0, 0),
                operation='replace',
                has_transparency=False,
                error=str(e)
            )

    def blur_background(
        self,
        image: Union[str, Path, Image.Image],
        blur_radius: int = 20,
        method: str = 'auto',
        output: Optional[Union[str, Path]] = None,
        quality: int = 95,
        **kwargs
    ) -> Union[BackgroundResult, Tuple[Image.Image, BackgroundResult]]:
        """
        Blur the background while keeping foreground sharp.

        Args:
            image: Input image
            blur_radius: Blur radius
            method: Background detection method
            output: Output path
            quality: JPEG quality
            **kwargs: Method-specific options

        Returns:
            BackgroundResult or (Image, BackgroundResult)
        """
        try:
            img = self._load_image(image)

            # Remove background to get mask
            result = self.remove(img, method, **kwargs)

            if isinstance(result, tuple):
                fg_img, remove_result = result
            else:
                if not result.success:
                    return result
                return result

            # Create blurred version
            from PIL import ImageFilter

            if img.mode != 'RGBA':
                img = img.convert('RGBA')

            blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

            # Use alpha from removed background as mask
            if fg_img.mode == 'RGBA':
                mask = fg_img.split()[3]
            else:
                mask = Image.new('L', img.size, 255)

            # Composite
            result_img = Image.composite(img, blurred, mask)

            result = BackgroundResult(
                success=True,
                original_size=remove_result.original_size,
                new_size=result_img.size,
                operation='blur',
                has_transparency=False,
                metadata={
                    'method': remove_result.metadata.get('method', 'unknown'),
                    'blur_radius': blur_radius
                }
            )

            if output:
                output_path = Path(output)

                save_kwargs = {}
                ext = output_path.suffix.lower()

                if ext in ['.jpg', '.jpeg']:
                    result_img = result_img.convert('RGB')
                    save_kwargs['quality'] = quality
                elif ext == '.webp':
                    save_kwargs['quality'] = quality

                result_img.save(str(output_path), **save_kwargs)
                result.output_path = str(output_path)
                return result
            else:
                return (result_img, result)

        except Exception as e:
            logger.error(f"Background blur error: {e}")
            return BackgroundResult(
                success=False,
                original_size=(0, 0),
                new_size=(0, 0),
                operation='blur',
                has_transparency=False,
                error=str(e)
            )

"""
Image resizing operations.

Provides comprehensive resizing functionality with various modes
and quality settings.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any
import logging

from PIL import Image

logger = logging.getLogger(__name__)


class ResizeMode(Enum):
    """Resize mode options."""
    FIT = "fit"  # Fit within dimensions, maintain aspect ratio
    FILL = "fill"  # Fill dimensions, may crop
    EXACT = "exact"  # Exact dimensions, may distort
    WIDTH = "width"  # Resize to width, maintain aspect ratio
    HEIGHT = "height"  # Resize to height, maintain aspect ratio
    THUMBNAIL = "thumbnail"  # Fast thumbnail resize


@dataclass
class ResizeResult:
    """Result of a resize operation."""
    success: bool
    original_size: Tuple[int, int]
    new_size: Tuple[int, int]
    mode_used: str
    output_path: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'success': self.success,
            'original_size': self.original_size,
            'new_size': self.new_size,
            'mode_used': self.mode_used,
            'output_path': self.output_path,
            'error': self.error,
            'metadata': self.metadata,
        }


class ImageResizer:
    """
    Image resizing engine with multiple resize modes.

    Supports various resampling filters and resize strategies.
    """

    # Resampling filter mapping
    RESAMPLE_FILTERS = {
        'lanczos': Image.Resampling.LANCZOS,
        'bicubic': Image.Resampling.BICUBIC,
        'bilinear': Image.Resampling.BILINEAR,
        'nearest': Image.Resampling.NEAREST,
        'box': Image.Resampling.BOX,
        'hamming': Image.Resampling.HAMMING,
    }

    def __init__(self, default_filter: str = 'lanczos'):
        """
        Initialize the resizer.

        Args:
            default_filter: Default resampling filter to use
        """
        self.default_filter = default_filter

    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """Load image from path or return if already an Image."""
        if isinstance(image, Image.Image):
            return image
        return Image.open(str(image))

    def _get_resample_filter(self, filter_name: Optional[str] = None) -> int:
        """Get the resampling filter constant."""
        name = filter_name or self.default_filter
        return self.RESAMPLE_FILTERS.get(name.lower(), Image.Resampling.LANCZOS)

    def _calculate_fit_size(
        self,
        original_size: Tuple[int, int],
        target_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Calculate size to fit within target while maintaining aspect ratio."""
        orig_w, orig_h = original_size
        target_w, target_h = target_size

        # Calculate scaling factors
        scale_w = target_w / orig_w
        scale_h = target_h / orig_h

        # Use the smaller scale to fit within bounds
        scale = min(scale_w, scale_h)

        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        return (new_w, new_h)

    def _calculate_fill_size(
        self,
        original_size: Tuple[int, int],
        target_size: Tuple[int, int]
    ) -> Tuple[Tuple[int, int], Tuple[int, int, int, int]]:
        """
        Calculate size and crop box for fill mode.

        Returns:
            Tuple of (resize_size, crop_box)
        """
        orig_w, orig_h = original_size
        target_w, target_h = target_size

        # Calculate scaling factors
        scale_w = target_w / orig_w
        scale_h = target_h / orig_h

        # Use the larger scale to fill the area
        scale = max(scale_w, scale_h)

        resize_w = int(orig_w * scale)
        resize_h = int(orig_h * scale)

        # Calculate crop box to center the image
        left = (resize_w - target_w) // 2
        top = (resize_h - target_h) // 2
        right = left + target_w
        bottom = top + target_h

        return ((resize_w, resize_h), (left, top, right, bottom))

    def resize(
        self,
        image: Union[str, Path, Image.Image],
        width: Optional[int] = None,
        height: Optional[int] = None,
        mode: ResizeMode = ResizeMode.FIT,
        maintain_aspect: bool = True,
        resample_filter: Optional[str] = None,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[ResizeResult, Tuple[Image.Image, ResizeResult]]:
        """
        Resize an image.

        Args:
            image: Input image (path or PIL Image)
            width: Target width
            height: Target height
            mode: Resize mode
            maintain_aspect: Whether to maintain aspect ratio
            resample_filter: Resampling filter name
            output: Output path (if None, returns PIL Image)
            quality: JPEG quality (1-100)

        Returns:
            ResizeResult if output path provided, otherwise (Image, ResizeResult)
        """
        try:
            img = self._load_image(image)
            original_size = img.size
            resample = self._get_resample_filter(resample_filter)

            # Handle different resize modes
            if mode == ResizeMode.WIDTH and width:
                # Resize by width, calculate height
                ratio = width / original_size[0]
                new_size = (width, int(original_size[1] * ratio))
                result_img = img.resize(new_size, resample)

            elif mode == ResizeMode.HEIGHT and height:
                # Resize by height, calculate width
                ratio = height / original_size[1]
                new_size = (int(original_size[0] * ratio), height)
                result_img = img.resize(new_size, resample)

            elif mode == ResizeMode.EXACT and width and height:
                # Exact resize, may distort
                new_size = (width, height)
                result_img = img.resize(new_size, resample)

            elif mode == ResizeMode.FILL and width and height:
                # Fill and crop
                target_size = (width, height)
                resize_size, crop_box = self._calculate_fill_size(
                    original_size, target_size
                )
                result_img = img.resize(resize_size, resample)
                result_img = result_img.crop(crop_box)
                new_size = result_img.size

            elif mode == ResizeMode.THUMBNAIL and width and height:
                # Fast thumbnail
                img_copy = img.copy()
                img_copy.thumbnail((width, height), resample)
                result_img = img_copy
                new_size = result_img.size

            elif mode == ResizeMode.FIT and width and height:
                # Fit within bounds
                target_size = (width, height)
                new_size = self._calculate_fit_size(original_size, target_size)
                result_img = img.resize(new_size, resample)

            else:
                # Default: use provided dimensions or original
                if width and not height:
                    if maintain_aspect:
                        ratio = width / original_size[0]
                        new_size = (width, int(original_size[1] * ratio))
                    else:
                        new_size = (width, original_size[1])
                elif height and not width:
                    if maintain_aspect:
                        ratio = height / original_size[1]
                        new_size = (int(original_size[0] * ratio), height)
                    else:
                        new_size = (original_size[0], height)
                elif width and height:
                    if maintain_aspect:
                        new_size = self._calculate_fit_size(
                            original_size, (width, height)
                        )
                    else:
                        new_size = (width, height)
                else:
                    new_size = original_size

                result_img = img.resize(new_size, resample)

            result = ResizeResult(
                success=True,
                original_size=original_size,
                new_size=result_img.size,
                mode_used=mode.value,
                metadata={
                    'resample_filter': resample_filter or self.default_filter,
                    'maintain_aspect': maintain_aspect,
                }
            )

            if output:
                output_path = Path(output)

                # Handle format-specific save options
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
                result.output_path = str(output_path)
                return result
            else:
                return (result_img, result)

        except Exception as e:
            logger.error(f"Resize error: {e}")
            return ResizeResult(
                success=False,
                original_size=(0, 0),
                new_size=(0, 0),
                mode_used=mode.value if mode else 'unknown',
                error=str(e)
            )

    def resize_to_width(
        self,
        image: Union[str, Path, Image.Image],
        width: int,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[ResizeResult, Tuple[Image.Image, ResizeResult]]:
        """Convenience method to resize to specific width."""
        return self.resize(
            image, width=width, mode=ResizeMode.WIDTH,
            output=output, quality=quality
        )

    def resize_to_height(
        self,
        image: Union[str, Path, Image.Image],
        height: int,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[ResizeResult, Tuple[Image.Image, ResizeResult]]:
        """Convenience method to resize to specific height."""
        return self.resize(
            image, height=height, mode=ResizeMode.HEIGHT,
            output=output, quality=quality
        )

    def create_thumbnail(
        self,
        image: Union[str, Path, Image.Image],
        size: Tuple[int, int] = (150, 150),
        output: Optional[Union[str, Path]] = None,
        quality: int = 85
    ) -> Union[ResizeResult, Tuple[Image.Image, ResizeResult]]:
        """Create a thumbnail of the image."""
        return self.resize(
            image, width=size[0], height=size[1],
            mode=ResizeMode.THUMBNAIL, output=output, quality=quality
        )

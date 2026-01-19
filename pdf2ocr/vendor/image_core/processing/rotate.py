"""
Image rotation operations.

Provides rotation, flipping, and orientation correction
functionality.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any
import logging

from PIL import Image, ExifTags

logger = logging.getLogger(__name__)


class FlipMode(Enum):
    """Flip mode options."""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    BOTH = "both"


@dataclass
class RotateResult:
    """Result of a rotation operation."""
    success: bool
    original_size: Tuple[int, int]
    new_size: Tuple[int, int]
    rotation_degrees: float
    flipped: Optional[str] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'success': self.success,
            'original_size': self.original_size,
            'new_size': self.new_size,
            'rotation_degrees': self.rotation_degrees,
            'flipped': self.flipped,
            'output_path': self.output_path,
            'error': self.error,
            'metadata': self.metadata,
        }


class ImageRotator:
    """
    Image rotation and orientation engine.

    Supports arbitrary rotation, flipping, and EXIF-based
    orientation correction.
    """

    # EXIF orientation values and their corrections
    EXIF_ORIENTATION = {
        1: (0, None),  # Normal
        2: (0, FlipMode.HORIZONTAL),  # Mirrored
        3: (180, None),  # Upside down
        4: (180, FlipMode.HORIZONTAL),  # Upside down mirrored
        5: (90, FlipMode.HORIZONTAL),  # 90 CW mirrored
        6: (270, None),  # 90 CW
        7: (270, FlipMode.HORIZONTAL),  # 90 CCW mirrored
        8: (90, None),  # 90 CCW
    }

    def __init__(self, default_fill_color: str = 'white'):
        """
        Initialize the rotator.

        Args:
            default_fill_color: Default fill color for expanded areas
        """
        self.default_fill_color = default_fill_color

    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """Load image from path or return if already an Image."""
        if isinstance(image, Image.Image):
            return image
        return Image.open(str(image))

    def _get_exif_orientation(self, image: Image.Image) -> Optional[int]:
        """Get EXIF orientation value from image."""
        try:
            exif = image._getexif()
            if exif:
                for tag, value in exif.items():
                    if ExifTags.TAGS.get(tag) == 'Orientation':
                        return value
        except (AttributeError, KeyError, IndexError):
            pass
        return None

    def _parse_color(self, color: Union[str, Tuple]) -> Union[str, Tuple]:
        """Parse color specification."""
        if isinstance(color, str):
            if color.startswith('#'):
                # Convert hex to RGB
                hex_color = color.lstrip('#')
                if len(hex_color) == 6:
                    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                elif len(hex_color) == 8:
                    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4, 6))
            return color
        return color

    def rotate(
        self,
        image: Union[str, Path, Image.Image],
        degrees: float,
        expand: bool = True,
        fill_color: Optional[Union[str, Tuple]] = None,
        resample: str = 'bicubic',
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[RotateResult, Tuple[Image.Image, RotateResult]]:
        """
        Rotate an image.

        Args:
            image: Input image (path or PIL Image)
            degrees: Rotation angle in degrees (counter-clockwise)
            expand: Whether to expand canvas to fit rotated image
            fill_color: Color to fill expanded areas
            resample: Resampling method
            output: Output path (if None, returns PIL Image)
            quality: JPEG quality (1-100)

        Returns:
            RotateResult if output path provided, otherwise (Image, RotateResult)
        """
        try:
            img = self._load_image(image)
            original_size = img.size

            # Handle color
            color = self._parse_color(fill_color or self.default_fill_color)

            # Map resample method
            resample_map = {
                'nearest': Image.Resampling.NEAREST,
                'bilinear': Image.Resampling.BILINEAR,
                'bicubic': Image.Resampling.BICUBIC,
            }
            resample_method = resample_map.get(resample, Image.Resampling.BICUBIC)

            # Handle RGBA images
            if img.mode == 'RGBA':
                # Create background
                background = Image.new('RGBA', img.size, color if isinstance(color, tuple) else (255, 255, 255, 255))
                # Paste image
                result_img = img.rotate(
                    degrees,
                    resample=resample_method,
                    expand=expand,
                    fillcolor=(0, 0, 0, 0)
                )
                if expand:
                    background = Image.new('RGBA', result_img.size, color if isinstance(color, tuple) else (255, 255, 255, 255))
                result_img = Image.alpha_composite(background, result_img)
            else:
                result_img = img.rotate(
                    degrees,
                    resample=resample_method,
                    expand=expand,
                    fillcolor=color
                )

            result = RotateResult(
                success=True,
                original_size=original_size,
                new_size=result_img.size,
                rotation_degrees=degrees,
                metadata={
                    'expand': expand,
                    'fill_color': str(color),
                    'resample': resample,
                }
            )

            if output:
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
                result.output_path = str(output_path)
                return result
            else:
                return (result_img, result)

        except Exception as e:
            logger.error(f"Rotate error: {e}")
            return RotateResult(
                success=False,
                original_size=(0, 0),
                new_size=(0, 0),
                rotation_degrees=degrees,
                error=str(e)
            )

    def flip(
        self,
        image: Union[str, Path, Image.Image],
        mode: FlipMode = FlipMode.HORIZONTAL,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[RotateResult, Tuple[Image.Image, RotateResult]]:
        """
        Flip an image.

        Args:
            image: Input image (path or PIL Image)
            mode: Flip mode (horizontal, vertical, or both)
            output: Output path (if None, returns PIL Image)
            quality: JPEG quality (1-100)

        Returns:
            RotateResult if output path provided, otherwise (Image, RotateResult)
        """
        try:
            img = self._load_image(image)
            original_size = img.size

            if mode == FlipMode.HORIZONTAL:
                result_img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            elif mode == FlipMode.VERTICAL:
                result_img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            elif mode == FlipMode.BOTH:
                result_img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                result_img = result_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            else:
                result_img = img

            result = RotateResult(
                success=True,
                original_size=original_size,
                new_size=result_img.size,
                rotation_degrees=0,
                flipped=mode.value,
            )

            if output:
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
                result.output_path = str(output_path)
                return result
            else:
                return (result_img, result)

        except Exception as e:
            logger.error(f"Flip error: {e}")
            return RotateResult(
                success=False,
                original_size=(0, 0),
                new_size=(0, 0),
                rotation_degrees=0,
                error=str(e)
            )

    def auto_orient(
        self,
        image: Union[str, Path, Image.Image],
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[RotateResult, Tuple[Image.Image, RotateResult]]:
        """
        Auto-orient image based on EXIF data.

        Args:
            image: Input image (path or PIL Image)
            output: Output path (if None, returns PIL Image)
            quality: JPEG quality (1-100)

        Returns:
            RotateResult if output path provided, otherwise (Image, RotateResult)
        """
        try:
            img = self._load_image(image)
            original_size = img.size

            orientation = self._get_exif_orientation(img)
            rotation = 0
            flip_mode = None

            if orientation and orientation in self.EXIF_ORIENTATION:
                rotation, flip_mode = self.EXIF_ORIENTATION[orientation]

                # Apply rotation
                if rotation == 90:
                    img = img.transpose(Image.Transpose.ROTATE_90)
                elif rotation == 180:
                    img = img.transpose(Image.Transpose.ROTATE_180)
                elif rotation == 270:
                    img = img.transpose(Image.Transpose.ROTATE_270)

                # Apply flip
                if flip_mode == FlipMode.HORIZONTAL:
                    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            result = RotateResult(
                success=True,
                original_size=original_size,
                new_size=img.size,
                rotation_degrees=rotation,
                flipped=flip_mode.value if flip_mode else None,
                metadata={
                    'exif_orientation': orientation,
                    'auto_corrected': orientation is not None and orientation != 1,
                }
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
            logger.error(f"Auto-orient error: {e}")
            return RotateResult(
                success=False,
                original_size=(0, 0),
                new_size=(0, 0),
                rotation_degrees=0,
                error=str(e)
            )

    def rotate_90_cw(
        self,
        image: Union[str, Path, Image.Image],
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[RotateResult, Tuple[Image.Image, RotateResult]]:
        """Rotate image 90 degrees clockwise."""
        return self.rotate(image, -90, expand=True, output=output, quality=quality)

    def rotate_90_ccw(
        self,
        image: Union[str, Path, Image.Image],
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[RotateResult, Tuple[Image.Image, RotateResult]]:
        """Rotate image 90 degrees counter-clockwise."""
        return self.rotate(image, 90, expand=True, output=output, quality=quality)

    def rotate_180(
        self,
        image: Union[str, Path, Image.Image],
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[RotateResult, Tuple[Image.Image, RotateResult]]:
        """Rotate image 180 degrees."""
        return self.rotate(image, 180, expand=True, output=output, quality=quality)

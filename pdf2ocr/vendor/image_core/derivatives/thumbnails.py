"""
Thumbnail generation operations.

Provides various thumbnail generation strategies including
smart cropping and multi-size generation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any, List
import logging

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ThumbnailResult:
    """Result of a thumbnail operation."""
    success: bool
    original_size: Tuple[int, int]
    thumbnail_size: Tuple[int, int]
    method: str
    output_path: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'success': self.success,
            'original_size': self.original_size,
            'thumbnail_size': self.thumbnail_size,
            'method': self.method,
            'output_path': self.output_path,
            'error': self.error,
            'metadata': self.metadata,
        }


class ThumbnailGenerator:
    """
    Thumbnail generation engine.

    Provides various thumbnail generation strategies
    optimized for different use cases.
    """

    # Standard thumbnail sizes
    SIZES = {
        'small': (75, 75),
        'medium': (150, 150),
        'large': (300, 300),
        'xlarge': (600, 600),
        'icon': (32, 32),
        'avatar': (100, 100),
        'preview': (200, 200),
        'gallery': (400, 400),
    }

    def __init__(self, default_size: Tuple[int, int] = (150, 150)):
        """
        Initialize the generator.

        Args:
            default_size: Default thumbnail size
        """
        self.default_size = default_size

    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """Load image from path or return if already an Image."""
        if isinstance(image, Image.Image):
            return image.copy()
        return Image.open(str(image))

    def _parse_size(
        self,
        size: Union[str, Tuple[int, int], int]
    ) -> Tuple[int, int]:
        """Parse size specification."""
        if isinstance(size, str):
            return self.SIZES.get(size, self.default_size)
        elif isinstance(size, int):
            return (size, size)
        else:
            return size

    def create(
        self,
        image: Union[str, Path, Image.Image],
        size: Union[str, Tuple[int, int], int] = None,
        method: str = 'fit',
        output: Optional[Union[str, Path]] = None,
        quality: int = 85
    ) -> Union[ThumbnailResult, Tuple[Image.Image, ThumbnailResult]]:
        """
        Create a thumbnail.

        Args:
            image: Input image (path or PIL Image)
            size: Target size (name, tuple, or int)
            method: Creation method ('fit', 'fill', 'exact')
            output: Output path (if None, returns PIL Image)
            quality: JPEG quality (1-100)

        Returns:
            ThumbnailResult if output path provided, otherwise (Image, ThumbnailResult)
        """
        try:
            img = self._load_image(image)
            original_size = img.size

            # Parse size
            target_size = self._parse_size(size or self.default_size)

            if method == 'fit':
                # Fit within bounds, maintain aspect ratio
                img.thumbnail(target_size, Image.Resampling.LANCZOS)
                thumb = img

            elif method == 'fill':
                # Fill target size, center crop
                orig_ratio = original_size[0] / original_size[1]
                target_ratio = target_size[0] / target_size[1]

                if orig_ratio > target_ratio:
                    # Image is wider
                    new_height = target_size[1]
                    new_width = int(new_height * orig_ratio)
                else:
                    # Image is taller
                    new_width = target_size[0]
                    new_height = int(new_width / orig_ratio)

                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Center crop
                left = (new_width - target_size[0]) // 2
                top = (new_height - target_size[1]) // 2
                thumb = img.crop((left, top, left + target_size[0], top + target_size[1]))

            elif method == 'exact':
                # Force exact size (may distort)
                thumb = img.resize(target_size, Image.Resampling.LANCZOS)

            else:
                # Default to fit
                img.thumbnail(target_size, Image.Resampling.LANCZOS)
                thumb = img

            result = ThumbnailResult(
                success=True,
                original_size=original_size,
                thumbnail_size=thumb.size,
                method=method,
            )

            if output:
                output_path = Path(output)

                save_kwargs = {}
                ext = output_path.suffix.lower()

                if ext in ['.jpg', '.jpeg']:
                    if thumb.mode in ('RGBA', 'P'):
                        thumb = thumb.convert('RGB')
                    save_kwargs['quality'] = quality
                    save_kwargs['optimize'] = True
                elif ext == '.png':
                    save_kwargs['optimize'] = True
                elif ext == '.webp':
                    save_kwargs['quality'] = quality

                thumb.save(str(output_path), **save_kwargs)
                result.output_path = str(output_path)
                return result
            else:
                return (thumb, result)

        except Exception as e:
            logger.error(f"Thumbnail error: {e}")
            return ThumbnailResult(
                success=False,
                original_size=(0, 0),
                thumbnail_size=(0, 0),
                method=method,
                error=str(e)
            )

    def create_smart(
        self,
        image: Union[str, Path, Image.Image],
        size: Union[str, Tuple[int, int], int] = None,
        output: Optional[Union[str, Path]] = None,
        quality: int = 85
    ) -> Union[ThumbnailResult, Tuple[Image.Image, ThumbnailResult]]:
        """
        Create a smart thumbnail using content-aware cropping.

        Args:
            image: Input image
            size: Target size
            output: Output path
            quality: JPEG quality

        Returns:
            ThumbnailResult or (Image, ThumbnailResult)
        """
        try:
            import numpy as np
            from scipy import ndimage

            img = self._load_image(image)
            original_size = img.size
            target_size = self._parse_size(size or self.default_size)

            # Calculate aspect-corrected crop size
            orig_ratio = original_size[0] / original_size[1]
            target_ratio = target_size[0] / target_size[1]

            if orig_ratio > target_ratio:
                crop_h = original_size[1]
                crop_w = int(crop_h * target_ratio)
            else:
                crop_w = original_size[0]
                crop_h = int(crop_w / target_ratio)

            # Find best crop region using edge detection
            gray = img.convert('L')
            arr = np.array(gray, dtype=np.float32)

            # Calculate gradient magnitude
            sobel_x = ndimage.sobel(arr, axis=1)
            sobel_y = ndimage.sobel(arr, axis=0)
            gradient = np.sqrt(sobel_x**2 + sobel_y**2)

            # Find region with highest gradient
            best_score = -1
            best_box = (0, 0, crop_w, crop_h)

            step = max(1, min(crop_w, crop_h) // 20)

            for y in range(0, original_size[1] - crop_h + 1, step):
                for x in range(0, original_size[0] - crop_w + 1, step):
                    region = gradient[y:y + crop_h, x:x + crop_w]
                    score = np.sum(region)

                    if score > best_score:
                        best_score = score
                        best_box = (x, y, x + crop_w, y + crop_h)

            # Crop and resize
            cropped = img.crop(best_box)
            thumb = cropped.resize(target_size, Image.Resampling.LANCZOS)

            result = ThumbnailResult(
                success=True,
                original_size=original_size,
                thumbnail_size=thumb.size,
                method='smart',
                metadata={'crop_box': best_box}
            )

            if output:
                output_path = Path(output)

                save_kwargs = {}
                ext = output_path.suffix.lower()

                if ext in ['.jpg', '.jpeg']:
                    if thumb.mode in ('RGBA', 'P'):
                        thumb = thumb.convert('RGB')
                    save_kwargs['quality'] = quality
                elif ext == '.webp':
                    save_kwargs['quality'] = quality

                thumb.save(str(output_path), **save_kwargs)
                result.output_path = str(output_path)
                return result
            else:
                return (thumb, result)

        except ImportError:
            # Fall back to center crop
            logger.warning("numpy/scipy not available, using center crop")
            return self.create(image, size, method='fill', output=output, quality=quality)

        except Exception as e:
            logger.error(f"Smart thumbnail error: {e}")
            return ThumbnailResult(
                success=False,
                original_size=(0, 0),
                thumbnail_size=(0, 0),
                method='smart',
                error=str(e)
            )

    def create_multiple(
        self,
        image: Union[str, Path, Image.Image],
        sizes: List[Union[str, Tuple[int, int]]],
        output_dir: Union[str, Path],
        method: str = 'fit',
        quality: int = 85,
        format: str = 'jpg'
    ) -> List[ThumbnailResult]:
        """
        Create multiple thumbnails of different sizes.

        Args:
            image: Input image
            sizes: List of sizes
            output_dir: Output directory
            method: Creation method
            quality: JPEG quality
            format: Output format

        Returns:
            List of ThumbnailResult objects
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        img = self._load_image(image)

        # Get base filename
        if isinstance(image, (str, Path)):
            base_name = Path(image).stem
        else:
            base_name = 'image'

        results = []
        for size in sizes:
            parsed_size = self._parse_size(size)
            size_name = f"{parsed_size[0]}x{parsed_size[1]}"
            output_path = output_dir / f"{base_name}_{size_name}.{format}"

            result = self.create(
                img, size=parsed_size, method=method,
                output=output_path, quality=quality
            )
            if isinstance(result, tuple):
                result = result[1]
            results.append(result)

        return results

    def create_responsive_set(
        self,
        image: Union[str, Path, Image.Image],
        output_dir: Union[str, Path],
        quality: int = 85
    ) -> Dict[str, ThumbnailResult]:
        """
        Create a responsive image set for web use.

        Args:
            image: Input image
            output_dir: Output directory
            quality: JPEG quality

        Returns:
            Dictionary of size name to ThumbnailResult
        """
        responsive_sizes = {
            'xs': (320, 320),
            'sm': (640, 640),
            'md': (768, 768),
            'lg': (1024, 1024),
            'xl': (1280, 1280),
            '2xl': (1536, 1536),
        }

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        img = self._load_image(image)

        if isinstance(image, (str, Path)):
            base_name = Path(image).stem
        else:
            base_name = 'image'

        results = {}
        for size_name, size in responsive_sizes.items():
            output_path = output_dir / f"{base_name}_{size_name}.jpg"

            result = self.create(
                img, size=size, method='fit',
                output=output_path, quality=quality
            )
            if isinstance(result, tuple):
                result = result[1]
            results[size_name] = result

        return results

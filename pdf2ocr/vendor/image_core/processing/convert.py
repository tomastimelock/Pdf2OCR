"""
Image format conversion operations.

Provides format conversion with quality settings and
optimization options.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any, List
import logging

from PIL import Image

logger = logging.getLogger(__name__)


class ImageFormat(Enum):
    """Supported image formats."""
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    GIF = "gif"
    BMP = "bmp"
    TIFF = "tiff"
    ICO = "ico"
    PDF = "pdf"

    @classmethod
    def from_extension(cls, ext: str) -> 'ImageFormat':
        """Get format from file extension."""
        ext = ext.lower().lstrip('.')
        mapping = {
            'jpg': cls.JPEG,
            'jpeg': cls.JPEG,
            'png': cls.PNG,
            'webp': cls.WEBP,
            'gif': cls.GIF,
            'bmp': cls.BMP,
            'tif': cls.TIFF,
            'tiff': cls.TIFF,
            'ico': cls.ICO,
            'pdf': cls.PDF,
        }
        return mapping.get(ext, cls.JPEG)


@dataclass
class ConvertResult:
    """Result of a format conversion."""
    success: bool
    original_format: str
    new_format: str
    original_size: Tuple[int, int]
    new_size: Tuple[int, int]
    original_file_size: Optional[int] = None
    new_file_size: Optional[int] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'success': self.success,
            'original_format': self.original_format,
            'new_format': self.new_format,
            'original_size': self.original_size,
            'new_size': self.new_size,
            'original_file_size': self.original_file_size,
            'new_file_size': self.new_file_size,
            'output_path': self.output_path,
            'error': self.error,
            'metadata': self.metadata,
        }

    @property
    def compression_ratio(self) -> Optional[float]:
        """Calculate compression ratio if file sizes available."""
        if self.original_file_size and self.new_file_size:
            return self.new_file_size / self.original_file_size
        return None


class FormatConverter:
    """
    Image format conversion engine.

    Supports various image formats with customizable quality
    and optimization settings.
    """

    # Format-specific settings
    FORMAT_SETTINGS = {
        ImageFormat.JPEG: {
            'quality': 95,
            'optimize': True,
            'progressive': True,
        },
        ImageFormat.PNG: {
            'optimize': True,
            'compress_level': 6,
        },
        ImageFormat.WEBP: {
            'quality': 90,
            'method': 6,
            'lossless': False,
        },
        ImageFormat.GIF: {
            'optimize': True,
        },
        ImageFormat.TIFF: {
            'compression': 'lzw',
        },
    }

    def __init__(self):
        """Initialize the converter."""
        pass

    def _load_image(self, image: Union[str, Path, Image.Image]) -> Tuple[Image.Image, Optional[str]]:
        """
        Load image from path or return if already an Image.

        Returns:
            Tuple of (image, original_path)
        """
        if isinstance(image, Image.Image):
            return image, None
        path = str(image)
        return Image.open(path), path

    def _get_format_string(self, fmt: Union[str, ImageFormat]) -> str:
        """Get PIL format string."""
        if isinstance(fmt, ImageFormat):
            return fmt.value.upper()
        return fmt.upper()

    def _prepare_image_for_format(
        self,
        image: Image.Image,
        target_format: ImageFormat
    ) -> Image.Image:
        """Prepare image mode for target format."""
        if target_format == ImageFormat.JPEG:
            # JPEG doesn't support transparency
            if image.mode in ('RGBA', 'LA', 'P'):
                # Create white background
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                if image.mode in ('RGBA', 'LA'):
                    background.paste(image, mask=image.split()[-1])
                    return background
                return image.convert('RGB')
            elif image.mode != 'RGB':
                return image.convert('RGB')
        elif target_format == ImageFormat.PNG:
            if image.mode not in ('RGB', 'RGBA', 'L', 'LA', 'P'):
                return image.convert('RGBA')
        elif target_format == ImageFormat.GIF:
            if image.mode not in ('P', 'L'):
                return image.convert('P', palette=Image.Palette.ADAPTIVE, colors=256)
        elif target_format == ImageFormat.BMP:
            if image.mode not in ('RGB', 'L', '1'):
                return image.convert('RGB')

        return image

    def convert(
        self,
        image: Union[str, Path, Image.Image],
        target_format: Union[str, ImageFormat],
        output: Optional[Union[str, Path]] = None,
        quality: Optional[int] = None,
        optimize: bool = True,
        **kwargs
    ) -> Union[ConvertResult, Tuple[Image.Image, ConvertResult]]:
        """
        Convert image to a different format.

        Args:
            image: Input image (path or PIL Image)
            target_format: Target format
            output: Output path (if None, returns PIL Image)
            quality: Quality setting (1-100)
            optimize: Whether to optimize output
            **kwargs: Additional format-specific options

        Returns:
            ConvertResult if output path provided, otherwise (Image, ConvertResult)
        """
        try:
            img, original_path = self._load_image(image)
            original_size = img.size
            original_format = img.format or 'unknown'

            # Get original file size if available
            original_file_size = None
            if original_path:
                original_file_size = Path(original_path).stat().st_size

            # Parse target format
            if isinstance(target_format, str):
                target_format = ImageFormat.from_extension(target_format)

            # Prepare image for target format
            result_img = self._prepare_image_for_format(img, target_format)

            # Build save options
            save_options = dict(self.FORMAT_SETTINGS.get(target_format, {}))
            save_options.update(kwargs)

            if quality is not None:
                save_options['quality'] = quality
            if optimize is not None and 'optimize' in save_options:
                save_options['optimize'] = optimize

            result = ConvertResult(
                success=True,
                original_format=original_format,
                new_format=target_format.value,
                original_size=original_size,
                new_size=result_img.size,
                original_file_size=original_file_size,
                metadata={
                    'save_options': {k: str(v) for k, v in save_options.items()},
                }
            )

            if output:
                output_path = Path(output)

                # Save with appropriate options
                format_str = self._get_format_string(target_format)

                # Remove unsupported options for certain formats
                if target_format == ImageFormat.PDF:
                    save_options = {k: v for k, v in save_options.items()
                                   if k in ['resolution', 'title', 'author']}
                elif target_format == ImageFormat.BMP:
                    save_options = {}
                elif target_format == ImageFormat.ICO:
                    save_options = {}

                result_img.save(str(output_path), format=format_str, **save_options)

                result.output_path = str(output_path)
                result.new_file_size = output_path.stat().st_size

                return result
            else:
                return (result_img, result)

        except Exception as e:
            logger.error(f"Convert error: {e}")
            return ConvertResult(
                success=False,
                original_format='unknown',
                new_format=str(target_format),
                original_size=(0, 0),
                new_size=(0, 0),
                error=str(e)
            )

    def to_jpeg(
        self,
        image: Union[str, Path, Image.Image],
        output: Union[str, Path],
        quality: int = 95,
        progressive: bool = True
    ) -> ConvertResult:
        """Convert to JPEG format."""
        result = self.convert(
            image, ImageFormat.JPEG, output,
            quality=quality, progressive=progressive
        )
        return result if isinstance(result, ConvertResult) else result[1]

    def to_png(
        self,
        image: Union[str, Path, Image.Image],
        output: Union[str, Path],
        compress_level: int = 6
    ) -> ConvertResult:
        """Convert to PNG format."""
        result = self.convert(
            image, ImageFormat.PNG, output,
            compress_level=compress_level
        )
        return result if isinstance(result, ConvertResult) else result[1]

    def to_webp(
        self,
        image: Union[str, Path, Image.Image],
        output: Union[str, Path],
        quality: int = 90,
        lossless: bool = False
    ) -> ConvertResult:
        """Convert to WebP format."""
        result = self.convert(
            image, ImageFormat.WEBP, output,
            quality=quality, lossless=lossless
        )
        return result if isinstance(result, ConvertResult) else result[1]

    def batch_convert(
        self,
        images: List[Union[str, Path]],
        target_format: Union[str, ImageFormat],
        output_dir: Union[str, Path],
        quality: Optional[int] = None,
        **kwargs
    ) -> List[ConvertResult]:
        """
        Convert multiple images to the same format.

        Args:
            images: List of input image paths
            target_format: Target format
            output_dir: Output directory
            quality: Quality setting
            **kwargs: Additional format-specific options

        Returns:
            List of ConvertResult objects
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Parse target format
        if isinstance(target_format, str):
            target_format = ImageFormat.from_extension(target_format)

        # Get extension for target format
        ext_map = {
            ImageFormat.JPEG: '.jpg',
            ImageFormat.PNG: '.png',
            ImageFormat.WEBP: '.webp',
            ImageFormat.GIF: '.gif',
            ImageFormat.BMP: '.bmp',
            ImageFormat.TIFF: '.tiff',
            ImageFormat.ICO: '.ico',
            ImageFormat.PDF: '.pdf',
        }
        target_ext = ext_map.get(target_format, '.jpg')

        results = []
        for img_path in images:
            input_path = Path(img_path)
            output_path = output_dir / (input_path.stem + target_ext)

            result = self.convert(
                img_path, target_format, output_path,
                quality=quality, **kwargs
            )
            if isinstance(result, tuple):
                result = result[1]
            results.append(result)

        return results

    def optimize(
        self,
        image: Union[str, Path],
        output: Optional[Union[str, Path]] = None,
        target_size_kb: Optional[int] = None
    ) -> ConvertResult:
        """
        Optimize image file size while maintaining quality.

        Args:
            image: Input image path
            output: Output path (defaults to overwriting input)
            target_size_kb: Target file size in KB

        Returns:
            ConvertResult with optimization details
        """
        try:
            input_path = Path(image)
            output_path = Path(output) if output else input_path

            img = Image.open(str(input_path))
            original_size = input_path.stat().st_size
            original_format = img.format or ImageFormat.from_extension(input_path.suffix).value

            target_format = ImageFormat.from_extension(input_path.suffix)

            if target_size_kb:
                # Binary search for optimal quality
                min_quality = 10
                max_quality = 95
                best_quality = max_quality

                import io

                while min_quality <= max_quality:
                    mid_quality = (min_quality + max_quality) // 2

                    buffer = io.BytesIO()
                    save_img = self._prepare_image_for_format(img.copy(), target_format)
                    save_img.save(buffer, format=target_format.value.upper(),
                                 quality=mid_quality, optimize=True)
                    size_kb = buffer.tell() / 1024

                    if size_kb <= target_size_kb:
                        best_quality = mid_quality
                        min_quality = mid_quality + 1
                    else:
                        max_quality = mid_quality - 1

                result = self.convert(
                    img, target_format, output_path,
                    quality=best_quality, optimize=True
                )
            else:
                result = self.convert(
                    img, target_format, output_path,
                    optimize=True
                )

            if isinstance(result, tuple):
                result = result[1]

            return result

        except Exception as e:
            logger.error(f"Optimize error: {e}")
            return ConvertResult(
                success=False,
                original_format='unknown',
                new_format='unknown',
                original_size=(0, 0),
                new_size=(0, 0),
                error=str(e)
            )

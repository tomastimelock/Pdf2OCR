"""
RAW file processing operations.

Provides RAW image processing and development capabilities
using rawpy and other RAW processing libraries.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any, List
import logging

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class RawSettings:
    """RAW processing settings."""
    # White balance
    use_camera_wb: bool = True
    use_auto_wb: bool = False
    custom_wb: Optional[Tuple[float, float, float, float]] = None

    # Exposure
    exposure_shift: float = 0.0
    highlight_mode: int = 0  # 0=clip, 1=ignore, 2=blend, 3-9=rebuild

    # Color
    output_color_space: str = 'sRGB'  # sRGB, Adobe, Wide, ProPhoto, XYZ
    gamma: Tuple[float, float] = (2.222, 4.5)

    # Noise reduction
    denoise_threshold: Optional[float] = None
    fbdd_noise_reduction: int = 0  # 0=off, 1=light, 2=full

    # Output
    output_bps: int = 8  # 8 or 16 bits per sample
    half_size: bool = False
    four_color_rgb: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'use_camera_wb': self.use_camera_wb,
            'use_auto_wb': self.use_auto_wb,
            'custom_wb': self.custom_wb,
            'exposure_shift': self.exposure_shift,
            'highlight_mode': self.highlight_mode,
            'output_color_space': self.output_color_space,
            'gamma': self.gamma,
            'denoise_threshold': self.denoise_threshold,
            'fbdd_noise_reduction': self.fbdd_noise_reduction,
            'output_bps': self.output_bps,
            'half_size': self.half_size,
            'four_color_rgb': self.four_color_rgb,
        }


@dataclass
class RawResult:
    """Result of a RAW processing operation."""
    success: bool
    raw_file: str
    output_size: Tuple[int, int]
    output_format: str
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'success': self.success,
            'raw_file': self.raw_file,
            'output_size': self.output_size,
            'output_format': self.output_format,
            'camera_make': self.camera_make,
            'camera_model': self.camera_model,
            'output_path': self.output_path,
            'error': self.error,
            'metadata': self.metadata,
        }


class RawProcessor:
    """
    RAW file processing engine.

    Provides RAW file reading and processing capabilities
    using rawpy for professional-quality output.
    """

    # Supported RAW formats
    SUPPORTED_FORMATS = {
        '.arw': 'Sony',
        '.cr2': 'Canon',
        '.cr3': 'Canon',
        '.dng': 'Adobe DNG',
        '.nef': 'Nikon',
        '.nrw': 'Nikon',
        '.orf': 'Olympus',
        '.pef': 'Pentax',
        '.raf': 'Fujifilm',
        '.raw': 'Generic',
        '.rw2': 'Panasonic',
        '.srw': 'Samsung',
        '.x3f': 'Sigma',
    }

    def __init__(self, default_settings: Optional[RawSettings] = None):
        """
        Initialize the RAW processor.

        Args:
            default_settings: Default processing settings
        """
        self.default_settings = default_settings or RawSettings()

    def is_supported(self, file_path: Union[str, Path]) -> bool:
        """Check if a file format is supported."""
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_FORMATS

    def get_camera_info(self, raw_file: Union[str, Path]) -> Dict[str, Any]:
        """Get camera information from RAW file."""
        try:
            import rawpy

            with rawpy.imread(str(raw_file)) as raw:
                return {
                    'make': raw.camera_make if hasattr(raw, 'camera_make') else None,
                    'model': raw.camera_model if hasattr(raw, 'camera_model') else None,
                    'white_balance': raw.camera_whitebalance.tolist() if hasattr(raw, 'camera_whitebalance') else None,
                    'daylight_wb': raw.daylight_whitebalance.tolist() if hasattr(raw, 'daylight_whitebalance') else None,
                    'black_levels': raw.black_level_per_channel.tolist() if hasattr(raw, 'black_level_per_channel') else None,
                    'raw_image_size': raw.raw_image.shape if hasattr(raw, 'raw_image') else None,
                }

        except ImportError:
            logger.error("rawpy not installed")
            return {}
        except Exception as e:
            logger.error(f"Failed to get camera info: {e}")
            return {}

    def process(
        self,
        raw_file: Union[str, Path],
        output: Union[str, Path],
        settings: Optional[RawSettings] = None,
        output_format: str = 'jpg',
        quality: int = 95
    ) -> RawResult:
        """
        Process a RAW file.

        Args:
            raw_file: Input RAW file path
            output: Output file path
            settings: Processing settings
            output_format: Output format (jpg, png, tiff)
            quality: JPEG quality (1-100)

        Returns:
            RawResult with processing details
        """
        raw_path = Path(raw_file)
        output_path = Path(output)

        if not self.is_supported(raw_path):
            return RawResult(
                success=False,
                raw_file=str(raw_path),
                output_size=(0, 0),
                output_format=output_format,
                error=f"Unsupported format: {raw_path.suffix}"
            )

        try:
            import rawpy
            import numpy as np

            settings = settings or self.default_settings

            with rawpy.imread(str(raw_path)) as raw:
                # Get camera info
                camera_make = None
                camera_model = None
                try:
                    camera_make = raw.camera_make if hasattr(raw, 'camera_make') else None
                    camera_model = raw.camera_model if hasattr(raw, 'camera_model') else None
                except Exception:
                    pass

                # Build postprocessing parameters
                params = {}

                # White balance
                if settings.use_camera_wb:
                    params['use_camera_wb'] = True
                elif settings.use_auto_wb:
                    params['use_auto_wb'] = True
                elif settings.custom_wb:
                    params['user_wb'] = settings.custom_wb

                # Exposure
                if settings.exposure_shift != 0:
                    params['exp_shift'] = 2 ** settings.exposure_shift

                params['highlight_mode'] = settings.highlight_mode

                # Color space
                color_space_map = {
                    'sRGB': rawpy.ColorSpace.sRGB,
                    'Adobe': rawpy.ColorSpace.Adobe,
                    'Wide': rawpy.ColorSpace.Wide,
                    'ProPhoto': rawpy.ColorSpace.ProPhoto,
                    'XYZ': rawpy.ColorSpace.XYZ,
                }
                params['output_color'] = color_space_map.get(
                    settings.output_color_space,
                    rawpy.ColorSpace.sRGB
                )

                # Gamma
                params['gamma'] = settings.gamma

                # Noise reduction
                if settings.denoise_threshold is not None:
                    params['dcb_enhance'] = True

                params['fbdd_noise_reduction'] = rawpy.FBDDNoiseReductionMode(
                    settings.fbdd_noise_reduction
                )

                # Output settings
                params['output_bps'] = settings.output_bps
                params['half_size'] = settings.half_size
                params['four_color_rgb'] = settings.four_color_rgb

                # Process RAW
                rgb = raw.postprocess(**params)

                # Convert to PIL Image
                if settings.output_bps == 16:
                    # Convert 16-bit to 8-bit for PIL
                    rgb = (rgb / 256).astype(np.uint8)

                image = Image.fromarray(rgb)

                # Save output
                save_kwargs = {}
                ext = output_path.suffix.lower()

                if ext in ['.jpg', '.jpeg']:
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    save_kwargs['quality'] = quality
                    save_kwargs['optimize'] = True
                elif ext == '.png':
                    save_kwargs['optimize'] = True
                elif ext in ['.tiff', '.tif']:
                    pass  # TIFF handles various bit depths
                elif ext == '.webp':
                    save_kwargs['quality'] = quality

                image.save(str(output_path), **save_kwargs)

                return RawResult(
                    success=True,
                    raw_file=str(raw_path),
                    output_size=image.size,
                    output_format=output_format,
                    camera_make=camera_make,
                    camera_model=camera_model,
                    output_path=str(output_path),
                    metadata={
                        'settings': settings.to_dict(),
                        'original_format': raw_path.suffix.lower(),
                    }
                )

        except ImportError:
            logger.error("rawpy not installed. Install with: pip install rawpy")
            return RawResult(
                success=False,
                raw_file=str(raw_path),
                output_size=(0, 0),
                output_format=output_format,
                error="rawpy not installed. Install with: pip install rawpy"
            )

        except Exception as e:
            logger.error(f"RAW processing error: {e}")
            return RawResult(
                success=False,
                raw_file=str(raw_path),
                output_size=(0, 0),
                output_format=output_format,
                error=str(e)
            )

    def process_to_image(
        self,
        raw_file: Union[str, Path],
        settings: Optional[RawSettings] = None
    ) -> Union[Image.Image, None]:
        """
        Process a RAW file and return PIL Image.

        Args:
            raw_file: Input RAW file path
            settings: Processing settings

        Returns:
            PIL Image or None on error
        """
        try:
            import rawpy
            import numpy as np
            import tempfile

            settings = settings or self.default_settings

            with rawpy.imread(str(raw_file)) as raw:
                params = {
                    'use_camera_wb': settings.use_camera_wb,
                    'output_bps': 8,
                }

                if settings.exposure_shift != 0:
                    params['exp_shift'] = 2 ** settings.exposure_shift

                rgb = raw.postprocess(**params)
                return Image.fromarray(rgb)

        except ImportError:
            logger.error("rawpy not installed")
            return None
        except Exception as e:
            logger.error(f"RAW processing error: {e}")
            return None

    def batch_process(
        self,
        raw_files: List[Union[str, Path]],
        output_dir: Union[str, Path],
        settings: Optional[RawSettings] = None,
        output_format: str = 'jpg',
        quality: int = 95
    ) -> List[RawResult]:
        """
        Process multiple RAW files.

        Args:
            raw_files: List of RAW file paths
            output_dir: Output directory
            settings: Processing settings
            output_format: Output format
            quality: JPEG quality

        Returns:
            List of RawResult objects
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for raw_file in raw_files:
            raw_path = Path(raw_file)
            output_path = output_dir / f"{raw_path.stem}.{output_format}"

            result = self.process(
                raw_file, output_path,
                settings=settings,
                output_format=output_format,
                quality=quality
            )
            results.append(result)

        return results

    def extract_thumbnail(
        self,
        raw_file: Union[str, Path],
        output: Optional[Union[str, Path]] = None
    ) -> Union[RawResult, Tuple[Image.Image, RawResult]]:
        """
        Extract embedded thumbnail from RAW file.

        Args:
            raw_file: Input RAW file path
            output: Output path (if None, returns PIL Image)

        Returns:
            RawResult if output path provided, otherwise (Image, RawResult)
        """
        raw_path = Path(raw_file)

        try:
            import rawpy

            with rawpy.imread(str(raw_path)) as raw:
                # Try to get embedded thumbnail
                try:
                    thumb = raw.extract_thumb()
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        from io import BytesIO
                        image = Image.open(BytesIO(thumb.data))
                    elif thumb.format == rawpy.ThumbFormat.BITMAP:
                        import numpy as np
                        image = Image.fromarray(thumb.data)
                    else:
                        raise ValueError("Unknown thumbnail format")
                except rawpy.LibRawNoThumbnailError:
                    # Fall back to processing at half size
                    rgb = raw.postprocess(half_size=True, use_camera_wb=True)
                    image = Image.fromarray(rgb)

                result = RawResult(
                    success=True,
                    raw_file=str(raw_path),
                    output_size=image.size,
                    output_format='jpg',
                    metadata={'type': 'thumbnail'}
                )

                if output:
                    output_path = Path(output)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    image.save(str(output_path), quality=85)
                    result.output_path = str(output_path)
                    return result
                else:
                    return (image, result)

        except ImportError:
            error = "rawpy not installed"
            logger.error(error)
            return RawResult(
                success=False,
                raw_file=str(raw_path),
                output_size=(0, 0),
                output_format='jpg',
                error=error
            )

        except Exception as e:
            logger.error(f"Thumbnail extraction error: {e}")
            return RawResult(
                success=False,
                raw_file=str(raw_path),
                output_size=(0, 0),
                output_format='jpg',
                error=str(e)
            )

    def list_supported_formats(self) -> Dict[str, str]:
        """Get dictionary of supported formats and their manufacturers."""
        return self.SUPPORTED_FORMATS.copy()

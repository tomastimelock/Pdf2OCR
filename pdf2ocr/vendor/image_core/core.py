"""
Unified ImageCore class.

Provides a single, cohesive API for all image processing operations
by consolidating functionality from multiple specialized modules.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any, List
import logging

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of an image processing operation."""
    path: Optional[str]
    original_size: Tuple[int, int]
    new_size: Tuple[int, int]
    format: str
    operations_applied: List[str]
    quality_score: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if operation was successful."""
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'path': self.path,
            'original_size': self.original_size,
            'new_size': self.new_size,
            'format': self.format,
            'operations_applied': self.operations_applied,
            'quality_score': self.quality_score,
            'success': self.success,
            'error': self.error,
            'metadata': self.metadata,
        }


@dataclass
class QualityAnalysis:
    """Complete quality analysis result."""
    blur_score: float
    noise_level: float
    aesthetic_score: float
    overall_score: float
    is_blurry: bool
    is_noisy: bool
    issues: List[str]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'blur_score': self.blur_score,
            'noise_level': self.noise_level,
            'aesthetic_score': self.aesthetic_score,
            'overall_score': self.overall_score,
            'is_blurry': self.is_blurry,
            'is_noisy': self.is_noisy,
            'issues': self.issues,
            'recommendations': self.recommendations,
        }


class ImagePipeline:
    """
    Chainable image processing pipeline.

    Allows building complex processing workflows with a fluent API.
    """

    def __init__(self, core: 'ImageCore'):
        """
        Initialize the pipeline.

        Args:
            core: ImageCore instance to use for processing
        """
        self._core = core
        self._operations: List[Tuple[str, Dict[str, Any]]] = []

    def resize(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        maintain_aspect: bool = True
    ) -> 'ImagePipeline':
        """Add resize operation to pipeline."""
        self._operations.append(('resize', {
            'width': width,
            'height': height,
            'maintain_aspect': maintain_aspect,
        }))
        return self

    def crop(
        self,
        box: Optional[Tuple[int, int, int, int]] = None,
        aspect_ratio: Optional[str] = None
    ) -> 'ImagePipeline':
        """Add crop operation to pipeline."""
        self._operations.append(('crop', {
            'box': box,
            'aspect_ratio': aspect_ratio,
        }))
        return self

    def rotate(self, degrees: float, expand: bool = True) -> 'ImagePipeline':
        """Add rotate operation to pipeline."""
        self._operations.append(('rotate', {
            'degrees': degrees,
            'expand': expand,
        }))
        return self

    def enhance(
        self,
        brightness: Optional[float] = None,
        contrast: Optional[float] = None,
        saturation: Optional[float] = None,
        sharpness: Optional[float] = None
    ) -> 'ImagePipeline':
        """Add enhance operation to pipeline."""
        self._operations.append(('enhance', {
            'brightness': brightness,
            'contrast': contrast,
            'saturation': saturation,
            'sharpness': sharpness,
        }))
        return self

    def auto_enhance(self) -> 'ImagePipeline':
        """Add auto-enhance operation to pipeline."""
        self._operations.append(('auto_enhance', {}))
        return self

    def filter(self, filter_name: str, intensity: float = 1.0) -> 'ImagePipeline':
        """Add filter operation to pipeline."""
        self._operations.append(('filter', {
            'filter_name': filter_name,
            'intensity': intensity,
        }))
        return self

    def border(
        self,
        width: int,
        color: str = 'white'
    ) -> 'ImagePipeline':
        """Add border operation to pipeline."""
        self._operations.append(('border', {
            'width': width,
            'color': color,
        }))
        return self

    def watermark(
        self,
        text_or_image: Union[str, Path, Image.Image],
        position: str = 'bottom-right',
        opacity: float = 0.5
    ) -> 'ImagePipeline':
        """Add watermark operation to pipeline."""
        self._operations.append(('watermark', {
            'text_or_image': text_or_image,
            'position': position,
            'opacity': opacity,
        }))
        return self

    def remove_background(self) -> 'ImagePipeline':
        """Add background removal operation to pipeline."""
        self._operations.append(('remove_background', {}))
        return self

    def reduce_noise(self, strength: float = 0.5) -> 'ImagePipeline':
        """Add noise reduction operation to pipeline."""
        self._operations.append(('reduce_noise', {
            'strength': strength,
        }))
        return self

    def sharpen(self, amount: float = 1.0) -> 'ImagePipeline':
        """Add sharpen operation to pipeline."""
        self._operations.append(('sharpen', {
            'amount': amount,
        }))
        return self

    def convert(self, format: str, quality: int = 95) -> 'ImagePipeline':
        """Add format conversion operation to pipeline."""
        self._operations.append(('convert', {
            'format': format,
            'quality': quality,
        }))
        return self

    def execute(
        self,
        image: Union[str, Path, Image.Image],
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> ProcessingResult:
        """
        Execute the pipeline on an image.

        Args:
            image: Input image (path or PIL Image)
            output: Output path (if None, operations are applied but not saved)
            quality: JPEG quality for output

        Returns:
            ProcessingResult with all applied operations
        """
        try:
            # Load image
            if isinstance(image, Image.Image):
                img = image.copy()
                original_path = None
            else:
                img = Image.open(str(image))
                original_path = str(image)

            original_size = img.size
            applied_ops = []

            # Execute each operation
            for op_name, op_params in self._operations:
                try:
                    if op_name == 'resize':
                        result = self._core.resize(img, **op_params)
                        if isinstance(result, tuple):
                            img = result[0]
                        applied_ops.append(f"resize:{op_params}")

                    elif op_name == 'crop':
                        result = self._core.crop(img, **op_params)
                        if isinstance(result, tuple):
                            img = result[0]
                        applied_ops.append(f"crop:{op_params}")

                    elif op_name == 'rotate':
                        result = self._core.rotate(img, **op_params)
                        if isinstance(result, tuple):
                            img = result[0]
                        applied_ops.append(f"rotate:{op_params['degrees']}")

                    elif op_name == 'enhance':
                        result = self._core._enhancer.enhance(img, **op_params)
                        if isinstance(result, tuple):
                            img = result[0]
                        applied_ops.append(f"enhance:{op_params}")

                    elif op_name == 'auto_enhance':
                        result = self._core.auto_enhance(img)
                        if isinstance(result, tuple):
                            img = result[0]
                        applied_ops.append("auto_enhance")

                    elif op_name == 'filter':
                        result = self._core.apply_filter(img, **op_params)
                        if isinstance(result, tuple):
                            img = result[0]
                        applied_ops.append(f"filter:{op_params['filter_name']}")

                    elif op_name == 'border':
                        result = self._core.add_border(img, **op_params)
                        if isinstance(result, tuple):
                            img = result[0]
                        applied_ops.append(f"border:{op_params['width']}")

                    elif op_name == 'watermark':
                        result = self._core.add_watermark(img, **op_params)
                        if isinstance(result, tuple):
                            img = result[0]
                        applied_ops.append("watermark")

                    elif op_name == 'remove_background':
                        result = self._core.remove_background(img)
                        if isinstance(result, tuple):
                            img = result[0]
                        applied_ops.append("remove_background")

                    elif op_name == 'reduce_noise':
                        result = self._core.reduce_noise(img, **op_params)
                        if isinstance(result, tuple):
                            img = result[0]
                        applied_ops.append(f"reduce_noise:{op_params['strength']}")

                    elif op_name == 'sharpen':
                        result = self._core.sharpen(img, **op_params)
                        if isinstance(result, tuple):
                            img = result[0]
                        applied_ops.append(f"sharpen:{op_params['amount']}")

                    elif op_name == 'convert':
                        # Convert is handled at save time
                        applied_ops.append(f"convert:{op_params['format']}")

                except Exception as e:
                    logger.warning(f"Pipeline operation {op_name} failed: {e}")

            # Save if output specified
            output_path = None
            output_format = 'unknown'

            if output:
                output_path = Path(output)
                output_format = output_path.suffix.lstrip('.')

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

            return ProcessingResult(
                path=str(output_path) if output_path else None,
                original_size=original_size,
                new_size=img.size,
                format=output_format,
                operations_applied=applied_ops,
                metadata={'pipeline': True}
            )

        except Exception as e:
            logger.error(f"Pipeline execution error: {e}")
            return ProcessingResult(
                path=None,
                original_size=(0, 0),
                new_size=(0, 0),
                format='unknown',
                operations_applied=[],
                error=str(e)
            )

    def clear(self) -> 'ImagePipeline':
        """Clear all operations from pipeline."""
        self._operations = []
        return self


class ImageCore:
    """
    Unified image processing engine.

    Consolidates all image processing functionality into a single,
    cohesive API with lazy-loaded components.
    """

    def __init__(self):
        """Initialize ImageCore with lazy-loaded components."""
        self._resizer = None
        self._cropper = None
        self._rotator = None
        self._converter = None
        self._quality_analyzer = None
        self._enhancer = None
        self._hdr_merger = None
        self._lens_corrector = None
        self._filter_engine = None
        self._border_maker = None
        self._overlay_tool = None
        self._background_remover = None
        self._thumbnail_generator = None
        self._avatar_maker = None
        self._contact_sheet_generator = None
        self._raw_processor = None

    # ========== Lazy-loaded components ==========

    @property
    def resizer(self):
        """Get the resizer component."""
        if self._resizer is None:
            from .processing import ImageResizer
            self._resizer = ImageResizer()
        return self._resizer

    @property
    def cropper(self):
        """Get the cropper component."""
        if self._cropper is None:
            from .processing import ImageCropper
            self._cropper = ImageCropper()
        return self._cropper

    @property
    def rotator(self):
        """Get the rotator component."""
        if self._rotator is None:
            from .processing import ImageRotator
            self._rotator = ImageRotator()
        return self._rotator

    @property
    def converter(self):
        """Get the format converter component."""
        if self._converter is None:
            from .processing import FormatConverter
            self._converter = FormatConverter()
        return self._converter

    @property
    def quality_analyzer(self):
        """Get the quality analyzer component."""
        if self._quality_analyzer is None:
            from .enhancement import QualityAnalyzer
            self._quality_analyzer = QualityAnalyzer()
        return self._quality_analyzer

    @property
    def enhancer(self):
        """Get the enhancer component."""
        if self._enhancer is None:
            from .enhancement import ImageEnhancer
            self._enhancer = ImageEnhancer()
        return self._enhancer

    @property
    def hdr_merger(self):
        """Get the HDR merger component."""
        if self._hdr_merger is None:
            from .enhancement import HDRMerger
            self._hdr_merger = HDRMerger()
        return self._hdr_merger

    @property
    def lens_corrector(self):
        """Get the lens corrector component."""
        if self._lens_corrector is None:
            from .enhancement import LensCorrector
            self._lens_corrector = LensCorrector()
        return self._lens_corrector

    @property
    def filter_engine(self):
        """Get the filter engine component."""
        if self._filter_engine is None:
            from .effects import FilterEngine
            self._filter_engine = FilterEngine()
        return self._filter_engine

    @property
    def border_maker(self):
        """Get the border maker component."""
        if self._border_maker is None:
            from .effects import BorderMaker
            self._border_maker = BorderMaker()
        return self._border_maker

    @property
    def overlay_tool(self):
        """Get the overlay tool component."""
        if self._overlay_tool is None:
            from .effects import OverlayTool
            self._overlay_tool = OverlayTool()
        return self._overlay_tool

    @property
    def background_remover(self):
        """Get the background remover component."""
        if self._background_remover is None:
            from .effects import BackgroundRemover
            self._background_remover = BackgroundRemover()
        return self._background_remover

    @property
    def thumbnail_generator(self):
        """Get the thumbnail generator component."""
        if self._thumbnail_generator is None:
            from .derivatives import ThumbnailGenerator
            self._thumbnail_generator = ThumbnailGenerator()
        return self._thumbnail_generator

    @property
    def avatar_maker(self):
        """Get the avatar maker component."""
        if self._avatar_maker is None:
            from .derivatives import AvatarMaker
            self._avatar_maker = AvatarMaker()
        return self._avatar_maker

    @property
    def contact_sheet_generator(self):
        """Get the contact sheet generator component."""
        if self._contact_sheet_generator is None:
            from .derivatives import ContactSheetGenerator
            self._contact_sheet_generator = ContactSheetGenerator()
        return self._contact_sheet_generator

    @property
    def raw_processor(self):
        """Get the RAW processor component."""
        if self._raw_processor is None:
            from .raw import RawProcessor
            self._raw_processor = RawProcessor()
        return self._raw_processor

    # ========== Helper methods ==========

    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """Load image from path or return if already an Image."""
        if isinstance(image, Image.Image):
            return image
        return Image.open(str(image))

    def _create_result(
        self,
        original_size: Tuple[int, int],
        new_size: Tuple[int, int],
        operation: str,
        output_path: Optional[str] = None,
        format: str = 'unknown',
        error: Optional[str] = None,
        **metadata
    ) -> ProcessingResult:
        """Create a ProcessingResult object."""
        return ProcessingResult(
            path=output_path,
            original_size=original_size,
            new_size=new_size,
            format=format,
            operations_applied=[operation],
            error=error,
            metadata=metadata
        )

    # ========== Processing operations ==========

    def resize(
        self,
        image: Union[str, Path, Image.Image],
        width: Optional[int] = None,
        height: Optional[int] = None,
        maintain_aspect: bool = True,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[ProcessingResult, Tuple[Image.Image, ProcessingResult]]:
        """
        Resize an image.

        Args:
            image: Input image (path or PIL Image)
            width: Target width
            height: Target height
            maintain_aspect: Whether to maintain aspect ratio
            output: Output path (if None, returns PIL Image)
            quality: JPEG quality

        Returns:
            ProcessingResult if output path provided, otherwise (Image, ProcessingResult)
        """
        result = self.resizer.resize(
            image, width=width, height=height,
            maintain_aspect=maintain_aspect,
            output=output, quality=quality
        )

        if output:
            resize_result = result
            return self._create_result(
                resize_result.original_size,
                resize_result.new_size,
                f"resize:{width}x{height}",
                output_path=resize_result.output_path,
                format=Path(output).suffix.lstrip('.') if output else 'unknown',
                error=resize_result.error
            )
        else:
            img, resize_result = result
            proc_result = self._create_result(
                resize_result.original_size,
                resize_result.new_size,
                f"resize:{width}x{height}",
                error=resize_result.error
            )
            return (img, proc_result)

    def crop(
        self,
        image: Union[str, Path, Image.Image],
        box: Optional[Tuple[int, int, int, int]] = None,
        aspect_ratio: Optional[str] = None,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[ProcessingResult, Tuple[Image.Image, ProcessingResult]]:
        """
        Crop an image.

        Args:
            image: Input image
            box: Crop box (left, top, right, bottom)
            aspect_ratio: Target aspect ratio (e.g., "16:9")
            output: Output path
            quality: JPEG quality

        Returns:
            ProcessingResult or (Image, ProcessingResult)
        """
        from .processing import CropMode

        if aspect_ratio:
            result = self.cropper.crop(
                image, aspect_ratio=aspect_ratio,
                mode=CropMode.ASPECT,
                output=output, quality=quality
            )
        elif box:
            result = self.cropper.crop(
                image, box=box,
                mode=CropMode.BOX,
                output=output, quality=quality
            )
        else:
            result = self.cropper.crop(image, output=output, quality=quality)

        if output:
            crop_result = result
            return self._create_result(
                crop_result.original_size,
                crop_result.new_size,
                f"crop:{crop_result.crop_box}",
                output_path=crop_result.output_path,
                format=Path(output).suffix.lstrip('.') if output else 'unknown',
                error=crop_result.error
            )
        else:
            img, crop_result = result
            proc_result = self._create_result(
                crop_result.original_size,
                crop_result.new_size,
                f"crop:{crop_result.crop_box}",
                error=crop_result.error
            )
            return (img, proc_result)

    def rotate(
        self,
        image: Union[str, Path, Image.Image],
        degrees: float,
        expand: bool = True,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[ProcessingResult, Tuple[Image.Image, ProcessingResult]]:
        """
        Rotate an image.

        Args:
            image: Input image
            degrees: Rotation angle (counter-clockwise)
            expand: Whether to expand canvas
            output: Output path
            quality: JPEG quality

        Returns:
            ProcessingResult or (Image, ProcessingResult)
        """
        result = self.rotator.rotate(
            image, degrees, expand=expand,
            output=output, quality=quality
        )

        if output:
            rotate_result = result
            return self._create_result(
                rotate_result.original_size,
                rotate_result.new_size,
                f"rotate:{degrees}",
                output_path=rotate_result.output_path,
                format=Path(output).suffix.lstrip('.') if output else 'unknown',
                error=rotate_result.error
            )
        else:
            img, rotate_result = result
            proc_result = self._create_result(
                rotate_result.original_size,
                rotate_result.new_size,
                f"rotate:{degrees}",
                error=rotate_result.error
            )
            return (img, proc_result)

    def convert(
        self,
        image: Union[str, Path, Image.Image],
        format: str,
        output: Union[str, Path],
        quality: int = 95
    ) -> ProcessingResult:
        """
        Convert image to a different format.

        Args:
            image: Input image
            format: Target format
            output: Output path
            quality: JPEG quality

        Returns:
            ProcessingResult
        """
        result = self.converter.convert(
            image, format, output=output, quality=quality
        )

        if isinstance(result, tuple):
            result = result[1]

        return self._create_result(
            result.original_size,
            result.new_size,
            f"convert:{format}",
            output_path=result.output_path,
            format=format,
            error=result.error
        )

    # ========== Enhancement operations ==========

    def analyze_quality(
        self,
        image: Union[str, Path, Image.Image]
    ) -> QualityAnalysis:
        """
        Analyze image quality.

        Args:
            image: Input image

        Returns:
            QualityAnalysis with quality metrics
        """
        result = self.quality_analyzer.analyze(image)

        return QualityAnalysis(
            blur_score=result.blur.blur_score if result.blur else 0.0,
            noise_level=result.noise.noise_level if result.noise else 0.0,
            aesthetic_score=result.aesthetics.overall_score if result.aesthetics else 0.0,
            overall_score=result.overall_quality or 0.0,
            is_blurry=result.blur.is_blurry if result.blur else False,
            is_noisy=result.noise.is_noisy if result.noise else False,
            issues=result.issues,
            recommendations=result.recommendations
        )

    def auto_enhance(
        self,
        image: Union[str, Path, Image.Image],
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[ProcessingResult, Tuple[Image.Image, ProcessingResult]]:
        """
        Automatically enhance image.

        Args:
            image: Input image
            output: Output path
            quality: JPEG quality

        Returns:
            ProcessingResult or (Image, ProcessingResult)
        """
        result = self.enhancer.auto_enhance(image, output=output, quality=quality)

        if output:
            enhance_result = result
            return self._create_result(
                enhance_result.original_size,
                enhance_result.new_size,
                "auto_enhance",
                output_path=enhance_result.output_path,
                format=Path(output).suffix.lstrip('.') if output else 'unknown',
                error=enhance_result.error
            )
        else:
            img, enhance_result = result
            proc_result = self._create_result(
                enhance_result.original_size,
                enhance_result.new_size,
                "auto_enhance",
                error=enhance_result.error
            )
            return (img, proc_result)

    def reduce_noise(
        self,
        image: Union[str, Path, Image.Image],
        strength: float = 0.5,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[ProcessingResult, Tuple[Image.Image, ProcessingResult]]:
        """
        Reduce noise in image.

        Args:
            image: Input image
            strength: Noise reduction strength (0-1)
            output: Output path
            quality: JPEG quality

        Returns:
            ProcessingResult or (Image, ProcessingResult)
        """
        result = self.enhancer.reduce_noise(
            image, strength=strength, output=output, quality=quality
        )

        if output:
            enhance_result = result
            return self._create_result(
                enhance_result.original_size,
                enhance_result.new_size,
                f"reduce_noise:{strength}",
                output_path=enhance_result.output_path,
                format=Path(output).suffix.lstrip('.') if output else 'unknown',
                error=enhance_result.error
            )
        else:
            img, enhance_result = result
            proc_result = self._create_result(
                enhance_result.original_size,
                enhance_result.new_size,
                f"reduce_noise:{strength}",
                error=enhance_result.error
            )
            return (img, proc_result)

    def sharpen(
        self,
        image: Union[str, Path, Image.Image],
        amount: float = 1.0,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[ProcessingResult, Tuple[Image.Image, ProcessingResult]]:
        """
        Sharpen image.

        Args:
            image: Input image
            amount: Sharpening amount
            output: Output path
            quality: JPEG quality

        Returns:
            ProcessingResult or (Image, ProcessingResult)
        """
        result = self.enhancer.sharpen(
            image, amount=amount, output=output, quality=quality
        )

        if output:
            enhance_result = result
            return self._create_result(
                enhance_result.original_size,
                enhance_result.new_size,
                f"sharpen:{amount}",
                output_path=enhance_result.output_path,
                format=Path(output).suffix.lstrip('.') if output else 'unknown',
                error=enhance_result.error
            )
        else:
            img, enhance_result = result
            proc_result = self._create_result(
                enhance_result.original_size,
                enhance_result.new_size,
                f"sharpen:{amount}",
                error=enhance_result.error
            )
            return (img, proc_result)

    def merge_hdr(
        self,
        images: List[Union[str, Path, Image.Image]],
        output: Union[str, Path],
        quality: int = 95
    ) -> ProcessingResult:
        """
        Merge multiple exposures into HDR image.

        Args:
            images: List of exposure images
            output: Output path
            quality: JPEG quality

        Returns:
            ProcessingResult
        """
        result = self.hdr_merger.merge(images, output, quality=quality)

        return self._create_result(
            (0, 0),  # Multiple inputs
            result.output_size,
            f"merge_hdr:{result.input_count}_images",
            output_path=result.output_path,
            format=Path(output).suffix.lstrip('.'),
            error=result.error
        )

    def correct_lens(
        self,
        image: Union[str, Path, Image.Image],
        profile: Optional[str] = None,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[ProcessingResult, Tuple[Image.Image, ProcessingResult]]:
        """
        Apply lens corrections.

        Args:
            image: Input image
            profile: Lens profile name
            output: Output path
            quality: JPEG quality

        Returns:
            ProcessingResult or (Image, ProcessingResult)
        """
        result = self.lens_corrector.correct(
            image, profile=profile, output=output, quality=quality
        )

        if output:
            lens_result = result
            return self._create_result(
                lens_result.original_size,
                lens_result.new_size,
                f"correct_lens:{profile or 'auto'}",
                output_path=lens_result.output_path,
                format=Path(output).suffix.lstrip('.') if output else 'unknown',
                error=lens_result.error
            )
        else:
            img, lens_result = result
            proc_result = self._create_result(
                lens_result.original_size,
                lens_result.new_size,
                f"correct_lens:{profile or 'auto'}",
                error=lens_result.error
            )
            return (img, proc_result)

    # ========== Effects operations ==========

    def apply_filter(
        self,
        image: Union[str, Path, Image.Image],
        filter_name: str,
        intensity: float = 1.0,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[ProcessingResult, Tuple[Image.Image, ProcessingResult]]:
        """
        Apply a filter to image.

        Args:
            image: Input image
            filter_name: Filter name
            intensity: Filter intensity (0-1)
            output: Output path
            quality: JPEG quality

        Returns:
            ProcessingResult or (Image, ProcessingResult)
        """
        result = self.filter_engine.apply_filter(
            image, filter_name, intensity=intensity,
            output=output, quality=quality
        )

        if output:
            filter_result = result
            return self._create_result(
                filter_result.original_size,
                filter_result.new_size,
                f"filter:{filter_name}:{intensity}",
                output_path=filter_result.output_path,
                format=Path(output).suffix.lstrip('.') if output else 'unknown',
                error=filter_result.error
            )
        else:
            img, filter_result = result
            proc_result = self._create_result(
                filter_result.original_size,
                filter_result.new_size,
                f"filter:{filter_name}:{intensity}",
                error=filter_result.error
            )
            return (img, proc_result)

    def add_border(
        self,
        image: Union[str, Path, Image.Image],
        width: int,
        color: str = 'white',
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[ProcessingResult, Tuple[Image.Image, ProcessingResult]]:
        """
        Add border to image.

        Args:
            image: Input image
            width: Border width
            color: Border color
            output: Output path
            quality: JPEG quality

        Returns:
            ProcessingResult or (Image, ProcessingResult)
        """
        result = self.border_maker.add_border(
            image, width, color=color,
            output=output, quality=quality
        )

        if output:
            border_result = result
            return self._create_result(
                border_result.original_size,
                border_result.new_size,
                f"border:{width}:{color}",
                output_path=border_result.output_path,
                format=Path(output).suffix.lstrip('.') if output else 'unknown',
                error=border_result.error
            )
        else:
            img, border_result = result
            proc_result = self._create_result(
                border_result.original_size,
                border_result.new_size,
                f"border:{width}:{color}",
                error=border_result.error
            )
            return (img, proc_result)

    def add_watermark(
        self,
        image: Union[str, Path, Image.Image],
        text_or_image: Union[str, Path, Image.Image],
        position: str = 'bottom-right',
        opacity: float = 0.5,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[ProcessingResult, Tuple[Image.Image, ProcessingResult]]:
        """
        Add watermark to image.

        Args:
            image: Input image
            text_or_image: Watermark text or image
            position: Watermark position
            opacity: Watermark opacity
            output: Output path
            quality: JPEG quality

        Returns:
            ProcessingResult or (Image, ProcessingResult)
        """
        result = self.overlay_tool.add_watermark(
            image, text_or_image,
            position=position, opacity=opacity,
            output=output, quality=quality
        )

        if output:
            overlay_result = result
            return self._create_result(
                overlay_result.original_size,
                overlay_result.new_size,
                f"watermark:{position}:{opacity}",
                output_path=overlay_result.output_path,
                format=Path(output).suffix.lstrip('.') if output else 'unknown',
                error=overlay_result.error
            )
        else:
            img, overlay_result = result
            proc_result = self._create_result(
                overlay_result.original_size,
                overlay_result.new_size,
                f"watermark:{position}:{opacity}",
                error=overlay_result.error
            )
            return (img, proc_result)

    def remove_background(
        self,
        image: Union[str, Path, Image.Image],
        output: Optional[Union[str, Path]] = None
    ) -> Union[ProcessingResult, Tuple[Image.Image, ProcessingResult]]:
        """
        Remove image background.

        Args:
            image: Input image
            output: Output path

        Returns:
            ProcessingResult or (Image, ProcessingResult)
        """
        result = self.background_remover.remove(image, output=output)

        if output:
            bg_result = result
            return self._create_result(
                bg_result.original_size,
                bg_result.new_size,
                "remove_background",
                output_path=bg_result.output_path,
                format='png',
                error=bg_result.error
            )
        else:
            img, bg_result = result
            proc_result = self._create_result(
                bg_result.original_size,
                bg_result.new_size,
                "remove_background",
                format='png',
                error=bg_result.error
            )
            return (img, proc_result)

    # ========== Derivative operations ==========

    def create_thumbnail(
        self,
        image: Union[str, Path, Image.Image],
        size: Tuple[int, int] = (150, 150),
        output: Optional[Union[str, Path]] = None,
        quality: int = 85
    ) -> Union[ProcessingResult, Tuple[Image.Image, ProcessingResult]]:
        """
        Create a thumbnail.

        Args:
            image: Input image
            size: Thumbnail size
            output: Output path
            quality: JPEG quality

        Returns:
            ProcessingResult or (Image, ProcessingResult)
        """
        result = self.thumbnail_generator.create(
            image, size=size, output=output, quality=quality
        )

        if output:
            thumb_result = result
            return self._create_result(
                thumb_result.original_size,
                thumb_result.thumbnail_size,
                f"thumbnail:{size}",
                output_path=thumb_result.output_path,
                format=Path(output).suffix.lstrip('.') if output else 'unknown',
                error=thumb_result.error
            )
        else:
            img, thumb_result = result
            proc_result = self._create_result(
                thumb_result.original_size,
                thumb_result.thumbnail_size,
                f"thumbnail:{size}",
                error=thumb_result.error
            )
            return (img, proc_result)

    def create_avatar(
        self,
        image: Union[str, Path, Image.Image],
        size: int = 200,
        shape: str = 'circle',
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[ProcessingResult, Tuple[Image.Image, ProcessingResult]]:
        """
        Create an avatar.

        Args:
            image: Input image
            size: Avatar size
            shape: Avatar shape
            output: Output path
            quality: JPEG quality

        Returns:
            ProcessingResult or (Image, ProcessingResult)
        """
        result = self.avatar_maker.create(
            image, size=size, shape=shape,
            output=output, quality=quality
        )

        if output:
            avatar_result = result
            return self._create_result(
                avatar_result.original_size,
                (avatar_result.avatar_size, avatar_result.avatar_size),
                f"avatar:{size}:{shape}",
                output_path=avatar_result.output_path,
                format=Path(output).suffix.lstrip('.') if output else 'unknown',
                error=avatar_result.error
            )
        else:
            img, avatar_result = result
            proc_result = self._create_result(
                avatar_result.original_size,
                (avatar_result.avatar_size, avatar_result.avatar_size),
                f"avatar:{size}:{shape}",
                error=avatar_result.error
            )
            return (img, proc_result)

    def create_contact_sheet(
        self,
        images: List[Union[str, Path, Image.Image]],
        output: Union[str, Path],
        columns: int = 4,
        quality: int = 95
    ) -> ProcessingResult:
        """
        Create a contact sheet from multiple images.

        Args:
            images: List of input images
            output: Output path
            columns: Number of columns
            quality: JPEG quality

        Returns:
            ProcessingResult
        """
        result = self.contact_sheet_generator.create(
            images, output, columns=columns, quality=quality
        )

        return self._create_result(
            (0, 0),  # Multiple inputs
            result.output_size,
            f"contact_sheet:{result.image_count}_images",
            output_path=result.output_path,
            format=Path(output).suffix.lstrip('.'),
            error=result.error
        )

    # ========== RAW operations ==========

    def process_raw(
        self,
        raw_file: Union[str, Path],
        output: Union[str, Path],
        output_format: str = 'jpg',
        quality: int = 95
    ) -> ProcessingResult:
        """
        Process a RAW file.

        Args:
            raw_file: Input RAW file
            output: Output path
            output_format: Output format
            quality: JPEG quality

        Returns:
            ProcessingResult
        """
        result = self.raw_processor.process(
            raw_file, output,
            output_format=output_format, quality=quality
        )

        return self._create_result(
            (0, 0),  # RAW doesn't have simple size
            result.output_size,
            f"process_raw:{output_format}",
            output_path=result.output_path,
            format=output_format,
            error=result.error,
            camera=f"{result.camera_make} {result.camera_model}".strip()
        )

    # ========== Pipeline operations ==========

    def pipeline(self) -> ImagePipeline:
        """
        Create a new processing pipeline.

        Returns:
            ImagePipeline for chaining operations
        """
        return ImagePipeline(self)

    def batch_process(
        self,
        images: List[Union[str, Path]],
        operations: List[Tuple[str, Dict[str, Any]]],
        output_dir: Union[str, Path],
        quality: int = 95
    ) -> List[ProcessingResult]:
        """
        Process multiple images with the same operations.

        Args:
            images: List of input image paths
            operations: List of (operation_name, params) tuples
            output_dir: Output directory
            quality: JPEG quality

        Returns:
            List of ProcessingResult objects
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []

        # Build pipeline from operations
        pipe = self.pipeline()
        for op_name, params in operations:
            if hasattr(pipe, op_name):
                getattr(pipe, op_name)(**params)

        # Process each image
        for img_path in images:
            input_path = Path(img_path)
            output_path = output_dir / input_path.name

            result = pipe.execute(img_path, output=output_path, quality=quality)
            results.append(result)

        return results

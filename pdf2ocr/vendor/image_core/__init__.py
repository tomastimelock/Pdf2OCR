"""
ImageCore - Unified Image Processing Engine.

A comprehensive image processing super-module that consolidates
functionality from multiple specialized modules into a single,
cohesive API.

Consolidates:
- image_quality (blur, noise, aesthetics, enhance)
- image_derivatives (avatars, thumbnails, contact sheets)
- effects_engine (filters, borders, overlays, watermarks)
- image_utils, image_resizer, image_rotator, image_cropper
- format_converter
- hdr_merger
- collage_maker (basic operations)
- background_remover
- lens_corrector
- raw_processor

Usage:
    from image_core import ImageCore

    # Create engine instance
    core = ImageCore()

    # Basic operations
    result = core.resize(image, width=800, output="resized.jpg")
    result = core.crop(image, aspect_ratio="16:9", output="cropped.jpg")
    result = core.rotate(image, 45, output="rotated.jpg")

    # Enhancement
    quality = core.analyze_quality(image)
    result = core.auto_enhance(image, output="enhanced.jpg")

    # Effects
    result = core.apply_filter(image, "sepia", intensity=0.8, output="filtered.jpg")
    result = core.add_watermark(image, "Copyright", output="watermarked.jpg")

    # Derivatives
    result = core.create_thumbnail(image, size=(200, 200), output="thumb.jpg")
    result = core.create_avatar(image, size=100, shape="circle", output="avatar.png")

    # RAW processing
    result = core.process_raw("photo.arw", output="photo.jpg")

    # Pipeline for chained operations
    result = core.pipeline() \\
        .resize(width=1200) \\
        .auto_enhance() \\
        .filter("vibrant", 0.5) \\
        .watermark("Copyright 2024", position="bottom-right") \\
        .execute(image, "output.jpg")

    # Batch processing
    results = core.batch_process(
        images=["img1.jpg", "img2.jpg", "img3.jpg"],
        operations=[
            ("resize", {"width": 800}),
            ("auto_enhance", {}),
            ("filter", {"filter_name": "sepia", "intensity": 0.5}),
        ],
        output_dir="processed/"
    )
"""

__version__ = "1.0.0"
__author__ = "Image Processing Team"
__all__ = [
    # Main classes
    'ImageCore',
    'ImagePipeline',
    'ProcessingResult',
    'QualityAnalysis',

    # Processing submodule
    'ImageResizer',
    'ResizeMode',
    'ImageCropper',
    'CropMode',
    'ImageRotator',
    'FlipMode',
    'FormatConverter',
    'ImageFormat',

    # Enhancement submodule
    'QualityAnalyzer',
    'BlurMetrics',
    'NoiseMetrics',
    'AestheticMetrics',
    'ImageEnhancer',
    'EnhancePreset',
    'HDRMerger',
    'ToneMappingMethod',
    'LensCorrector',
    'LensProfile',

    # Effects submodule
    'FilterEngine',
    'FilterType',
    'BorderMaker',
    'BorderStyle',
    'OverlayTool',
    'WatermarkPosition',
    'BackgroundRemover',

    # Derivatives submodule
    'ThumbnailGenerator',
    'AvatarMaker',
    'AvatarShape',
    'ContactSheetGenerator',

    # RAW submodule
    'RawProcessor',
    'RawSettings',
]

# Main classes
from .core import ImageCore, ImagePipeline, ProcessingResult, QualityAnalysis

# Processing submodule
from .processing import (
    ImageResizer,
    ResizeMode,
    ImageCropper,
    CropMode,
    ImageRotator,
    FormatConverter,
    ImageFormat,
)
from .processing.rotate import FlipMode

# Enhancement submodule
from .enhancement import (
    QualityAnalyzer,
    ImageEnhancer,
    EnhancePreset,
    HDRMerger,
    LensCorrector,
    LensProfile,
)
from .enhancement.quality import BlurMetrics, NoiseMetrics, AestheticMetrics
from .enhancement.hdr import ToneMappingMethod

# Effects submodule
from .effects import (
    FilterEngine,
    FilterType,
    BorderMaker,
    BorderStyle,
    OverlayTool,
    WatermarkPosition,
    BackgroundRemover,
)

# Derivatives submodule
from .derivatives import (
    ThumbnailGenerator,
    AvatarMaker,
    AvatarShape,
    ContactSheetGenerator,
)

# RAW submodule
from .raw import RawProcessor, RawSettings


def create_core() -> ImageCore:
    """
    Factory function to create an ImageCore instance.

    Returns:
        ImageCore instance
    """
    return ImageCore()


# Convenience function for quick operations
def quick_resize(
    image,
    width=None,
    height=None,
    output=None,
    quality=95
):
    """
    Quick resize operation without creating ImageCore instance.

    Args:
        image: Input image path or PIL Image
        width: Target width
        height: Target height
        output: Output path
        quality: JPEG quality

    Returns:
        ProcessingResult or (Image, ProcessingResult)
    """
    core = ImageCore()
    return core.resize(image, width=width, height=height, output=output, quality=quality)


def quick_enhance(image, output=None, quality=95):
    """
    Quick auto-enhance operation without creating ImageCore instance.

    Args:
        image: Input image path or PIL Image
        output: Output path
        quality: JPEG quality

    Returns:
        ProcessingResult or (Image, ProcessingResult)
    """
    core = ImageCore()
    return core.auto_enhance(image, output=output, quality=quality)


def quick_thumbnail(image, size=(150, 150), output=None, quality=85):
    """
    Quick thumbnail creation without creating ImageCore instance.

    Args:
        image: Input image path or PIL Image
        size: Thumbnail size
        output: Output path
        quality: JPEG quality

    Returns:
        ProcessingResult or (Image, ProcessingResult)
    """
    core = ImageCore()
    return core.create_thumbnail(image, size=size, output=output, quality=quality)

# Filepath: code_migration/ai_providers/anthropic_vision/model_config.py
# Description: Vision model configurations for Claude
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/Anthropic/model_config.py

"""
Anthropic Vision Model Configuration

Defines vision-capable Claude models and their parameters.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class VisionModelConfig:
    """Configuration for a vision-capable Claude model."""
    name: str
    description: str
    context_window: int
    max_output: int
    vision_capable: bool = True
    supported_formats: List[str] = field(default_factory=lambda: ["image/jpeg", "image/png", "image/gif", "image/webp"])
    max_images: int = 100
    optimal_dimension: int = 1568
    notes: str = ""


# Vision-capable models (Claude 4.5 generation)
VISION_MODELS = {
    "claude-opus-4-5-20250918": VisionModelConfig(
        name="claude-opus-4-5-20250918",
        description="Maximum intelligence - best for complex visual analysis",
        context_window=200000,
        max_output=64000,
        notes="Highest quality vision understanding, extended thinking support"
    ),

    "claude-sonnet-4-5-20250929": VisionModelConfig(
        name="claude-sonnet-4-5-20250929",
        description="Recommended default - balanced quality and speed",
        context_window=200000,
        max_output=64000,
        notes="Best all-around choice for vision tasks, 1M context beta available"
    ),

    "claude-haiku-4-5-20251001": VisionModelConfig(
        name="claude-haiku-4-5-20251001",
        description="Fastest - good for simple image analysis",
        context_window=200000,
        max_output=64000,
        notes="Best for high-volume vision processing"
    ),

    "claude-sonnet-4-20250514": VisionModelConfig(
        name="claude-sonnet-4-20250514",
        description="Previous generation - balanced performance",
        context_window=200000,
        max_output=64000,
        notes="Claude 4 model with solid vision capabilities"
    ),

    "claude-opus-4-1-20250805": VisionModelConfig(
        name="claude-opus-4-1-20250805",
        description="Previous generation Opus - exceptional intelligence",
        context_window=200000,
        max_output=32000,
        notes="Claude 4.1 with strong vision understanding"
    ),

    "claude-3-7-sonnet-20250219": VisionModelConfig(
        name="claude-3-7-sonnet-20250219",
        description="Legacy model - being deprecated Feb 19, 2026",
        context_window=200000,
        max_output=8000,
        notes="DEPRECATED - Use Claude 4.5 models instead"
    ),
}


# Vision configuration constants
VISION_CONFIG = {
    # Supported image formats
    "supported_formats": ["image/jpeg", "image/png", "image/gif", "image/webp"],
    "file_extensions": {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp"
    },

    # Size limits
    "max_dimensions": {"width": 8000, "height": 8000},
    "max_dimensions_20plus_images": {"width": 2000, "height": 2000},
    "optimal_megapixels": 1.15,
    "optimal_max_dimension": 1568,

    # File size limits
    "max_file_size_api": 5 * 1024 * 1024,  # 5MB
    "max_file_size_claude_ai": 10 * 1024 * 1024,  # 10MB
    "max_request_size": 32 * 1024 * 1024,  # 32MB

    # Image count limits
    "max_images_api": 100,
    "max_images_claude_ai": 20,
    "recommended_max_images": 20,

    # Token calculation
    "tokens_per_pixel": 1 / 750,  # tokens = (width * height) / 750

    # Source types
    "source_types": ["base64", "url"],

    # Best practices
    "best_practices": {
        "resize_threshold_px": 1568,
        "jpeg_quality": 85,
        "prefer_jpeg_for_photos": True,
        "prefer_png_for_text": True
    }
}


def get_vision_models() -> Dict[str, VisionModelConfig]:
    """
    Get all vision-capable models.

    Returns:
        Dictionary of model name to configuration
    """
    return VISION_MODELS.copy()


def get_current_vision_models() -> Dict[str, VisionModelConfig]:
    """
    Get only current (non-deprecated) vision models.

    Returns:
        Dictionary of current model name to configuration
    """
    return {
        k: v for k, v in VISION_MODELS.items()
        if "DEPRECATED" not in v.notes and "deprecated" not in v.description.lower()
    }


def get_vision_model_config(model_name: str) -> Optional[VisionModelConfig]:
    """
    Get configuration for a specific vision model.

    Args:
        model_name: Name of the model

    Returns:
        Model configuration or None if not found
    """
    return VISION_MODELS.get(model_name)


def get_default_vision_model() -> str:
    """
    Get the default recommended vision model.

    Returns:
        Model name (claude-sonnet-4-5-20250929)
    """
    return "claude-sonnet-4-5-20250929"


def get_fastest_vision_model() -> str:
    """
    Get the fastest vision model.

    Returns:
        Model name (claude-haiku-4-5-20251001)
    """
    return "claude-haiku-4-5-20251001"


def get_smartest_vision_model() -> str:
    """
    Get the most capable vision model.

    Returns:
        Model name (claude-opus-4-5-20250918)
    """
    return "claude-opus-4-5-20250918"


def supports_vision(model_name: str) -> bool:
    """
    Check if a model supports vision/image input.

    Args:
        model_name: Name of the model to check

    Returns:
        True if model supports vision, False otherwise
    """
    config = get_vision_model_config(model_name)
    return config is not None and config.vision_capable


def calculate_image_tokens(width: int, height: int) -> int:
    """
    Calculate estimated token cost for an image.

    Formula: tokens = (width × height) / 750

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Estimated token count

    Examples:
        >>> calculate_image_tokens(1092, 1092)
        1590
        >>> calculate_image_tokens(1568, 1568)
        3277
    """
    return int((width * height) / 750)


def is_optimal_size(width: int, height: int) -> bool:
    """
    Check if image dimensions are optimal for cost/quality.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        True if dimensions are optimal, False if should resize
    """
    max_dim = max(width, height)
    megapixels = (width * height) / 1_000_000

    return max_dim <= VISION_CONFIG["optimal_max_dimension"] and \
           megapixels <= VISION_CONFIG["optimal_megapixels"]


def get_optimal_resize_dimensions(width: int, height: int) -> tuple[int, int]:
    """
    Calculate optimal resize dimensions for an image.

    Args:
        width: Original width in pixels
        height: Original height in pixels

    Returns:
        Tuple of (new_width, new_height)
    """
    max_dim = max(width, height)
    optimal_max = VISION_CONFIG["optimal_max_dimension"]

    if max_dim <= optimal_max:
        return width, height

    # Scale to optimal max dimension
    scale = optimal_max / max_dim
    new_width = int(width * scale)
    new_height = int(height * scale)

    return new_width, new_height


def validate_image_format(file_extension: str) -> bool:
    """
    Validate if image format is supported.

    Args:
        file_extension: File extension (e.g., '.jpg', '.png')

    Returns:
        True if format is supported, False otherwise
    """
    return file_extension.lower() in VISION_CONFIG["file_extensions"]


def get_media_type(file_extension: str) -> Optional[str]:
    """
    Get MIME type for image file extension.

    Args:
        file_extension: File extension (e.g., '.jpg', '.png')

    Returns:
        MIME type string or None if unsupported
    """
    return VISION_CONFIG["file_extensions"].get(file_extension.lower())


def get_model_context_window(model_name: str) -> int:
    """
    Get context window size for a model.

    Args:
        model_name: Name of the model

    Returns:
        Context window size in tokens
    """
    config = get_vision_model_config(model_name)
    return config.context_window if config else 200000


def get_model_max_output(model_name: str) -> int:
    """
    Get maximum output tokens for a model.

    Args:
        model_name: Name of the model

    Returns:
        Maximum output tokens
    """
    config = get_vision_model_config(model_name)
    return config.max_output if config else 4096


def get_recommended_model_for_task(task_type: str) -> str:
    """
    Get recommended model for a specific vision task.

    Args:
        task_type: Type of task (ocr, description, analysis, comparison, detection)

    Returns:
        Recommended model name
    """
    recommendations = {
        "ocr": "claude-sonnet-4-5-20250929",  # Good balance for text extraction
        "description": "claude-haiku-4-5-20251001",  # Fast for simple descriptions
        "analysis": "claude-sonnet-4-5-20250929",  # Default for general analysis
        "comparison": "claude-opus-4-5-20250918",  # Best for complex comparisons
        "detection": "claude-sonnet-4-5-20250929",  # Good for object detection
        "chart": "claude-opus-4-5-20250918",  # Best for chart/graph analysis
        "document": "claude-sonnet-4-5-20250929",  # Good for document analysis
    }

    return recommendations.get(task_type.lower(), get_default_vision_model())


def get_model_info_summary() -> str:
    """
    Get a formatted summary of all vision models.

    Returns:
        Formatted string with model information
    """
    lines = ["Available Vision Models", "=" * 50, ""]

    for name, config in VISION_MODELS.items():
        deprecated = "[DEPRECATED] " if "DEPRECATED" in config.notes else ""
        lines.append(f"{deprecated}{name}")
        lines.append(f"  Description: {config.description}")
        lines.append(f"  Context: {config.context_window:,} tokens")
        lines.append(f"  Max Output: {config.max_output:,} tokens")
        lines.append(f"  Notes: {config.notes}")
        lines.append("")

    return "\n".join(lines)

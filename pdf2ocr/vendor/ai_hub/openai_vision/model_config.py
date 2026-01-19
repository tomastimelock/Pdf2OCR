# Filepath: code_migration/ai_providers/openai_vision/model_config.py
# Description: Configuration for OpenAI Vision models
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/openai/model_config.py

"""
OpenAI Vision Model Configuration
Configuration for all vision-capable OpenAI models.
"""

from typing import Dict, Any, List, Optional


# Vision-capable models with their configurations
VISION_MODELS = {
    "gpt-4o": {
        "description": "GPT-4o - fast multimodal model with vision",
        "detail_levels": ["low", "high", "auto"],
        "default_detail": "auto",
        "max_tokens": 4096,
        "supports_vision": True,
        "supports_tools": True,
        "notes": "Best balance of speed and accuracy. Recommended for most use cases.",
        "context_window": 128000,
        "training_data": "Up to Oct 2023"
    },
    "gpt-4o-mini": {
        "description": "GPT-4o Mini - cheaper vision model",
        "detail_levels": ["low", "high", "auto"],
        "default_detail": "auto",
        "max_tokens": 16384,
        "supports_vision": True,
        "supports_tools": True,
        "notes": "More cost-effective for simple vision tasks. Good for batch processing.",
        "context_window": 128000,
        "training_data": "Up to Oct 2023"
    },
    "gpt-4-turbo": {
        "description": "GPT-4 Turbo - high quality vision model",
        "detail_levels": ["low", "high", "auto"],
        "default_detail": "auto",
        "max_tokens": 4096,
        "supports_vision": True,
        "supports_tools": True,
        "notes": "High quality vision analysis with strong reasoning.",
        "context_window": 128000,
        "training_data": "Up to Apr 2023"
    },
    "gpt-4-vision-preview": {
        "description": "GPT-4 Vision Preview - legacy vision model",
        "detail_levels": ["low", "high", "auto"],
        "default_detail": "auto",
        "max_tokens": 4096,
        "supports_vision": True,
        "supports_tools": False,
        "notes": "Legacy model. Use gpt-4o or gpt-4-turbo instead.",
        "context_window": 128000,
        "training_data": "Up to Apr 2023",
        "deprecated": True
    }
}


# Detail level configurations
DETAIL_LEVELS = {
    "low": {
        "description": "Low detail - 512x512 resolution",
        "resolution": "512x512",
        "token_cost": "~85 tokens",
        "use_case": "Fast processing, simple images, lower cost",
        "speed": "Fast"
    },
    "high": {
        "description": "High detail - multiple high-res tiles",
        "resolution": "Variable (tiled)",
        "token_cost": "~129 base + tiles",
        "use_case": "Detailed analysis, OCR, complex images",
        "speed": "Slower"
    },
    "auto": {
        "description": "Auto - model chooses based on image size",
        "resolution": "Automatic",
        "token_cost": "Varies",
        "use_case": "Default, balanced approach",
        "speed": "Variable"
    }
}


# Supported image formats
SUPPORTED_FORMATS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp"
}


# Image size limits
IMAGE_LIMITS = {
    "max_file_size_mb": 20,
    "max_dimension_pixels": 4096,
    "recommended_max_pixels": 2048
}


def get_vision_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get configuration information for a specific vision model.

    Args:
        model_name: Name of the model (e.g., "gpt-4o")

    Returns:
        Model configuration dictionary or None if not found

    Example:
        >>> info = get_vision_model_info("gpt-4o")
        >>> print(info["description"])
        'GPT-4o - fast multimodal model with vision'
    """
    return VISION_MODELS.get(model_name)


def get_detail_level_info(detail: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a detail level.

    Args:
        detail: Detail level ("low", "high", or "auto")

    Returns:
        Detail level configuration dictionary or None if not found

    Example:
        >>> info = get_detail_level_info("high")
        >>> print(info["use_case"])
        'Detailed analysis, OCR, complex images'
    """
    return DETAIL_LEVELS.get(detail)


def list_supported_formats() -> List[str]:
    """
    List all supported image formats.

    Returns:
        List of file extensions (e.g., [".png", ".jpg", ...])

    Example:
        >>> formats = list_supported_formats()
        >>> print(formats)
        ['.png', '.jpg', '.jpeg', '.gif', '.webp']
    """
    return list(SUPPORTED_FORMATS.keys())


def get_mime_type(file_extension: str) -> Optional[str]:
    """
    Get MIME type for a file extension.

    Args:
        file_extension: File extension (e.g., ".png")

    Returns:
        MIME type string or None if not supported

    Example:
        >>> mime = get_mime_type(".png")
        >>> print(mime)
        'image/png'
    """
    return SUPPORTED_FORMATS.get(file_extension.lower())


def is_format_supported(file_extension: str) -> bool:
    """
    Check if a file format is supported.

    Args:
        file_extension: File extension (e.g., ".png")

    Returns:
        True if supported, False otherwise

    Example:
        >>> is_format_supported(".png")
        True
        >>> is_format_supported(".bmp")
        False
    """
    return file_extension.lower() in SUPPORTED_FORMATS


def get_recommended_model(use_case: str = "general") -> str:
    """
    Get recommended model for a specific use case.

    Args:
        use_case: Use case ("general", "fast", "accurate", "ocr", "batch")

    Returns:
        Recommended model name

    Example:
        >>> model = get_recommended_model("ocr")
        >>> print(model)
        'gpt-4o'
    """
    recommendations = {
        "general": "gpt-4o",
        "fast": "gpt-4o-mini",
        "accurate": "gpt-4-turbo",
        "ocr": "gpt-4o",
        "batch": "gpt-4o-mini",
        "cost_effective": "gpt-4o-mini"
    }
    return recommendations.get(use_case, "gpt-4o")


def get_recommended_detail(use_case: str = "general") -> str:
    """
    Get recommended detail level for a specific use case.

    Args:
        use_case: Use case ("general", "ocr", "fast", "detailed")

    Returns:
        Recommended detail level ("low", "high", "auto")

    Example:
        >>> detail = get_recommended_detail("ocr")
        >>> print(detail)
        'high'
    """
    recommendations = {
        "general": "auto",
        "ocr": "high",
        "fast": "low",
        "detailed": "high",
        "simple": "low",
        "batch": "low"
    }
    return recommendations.get(use_case, "auto")


def estimate_token_cost(detail: str, num_images: int = 1) -> Dict[str, Any]:
    """
    Estimate token cost for an API call.

    Args:
        detail: Detail level ("low", "high", "auto")
        num_images: Number of images to process

    Returns:
        Dictionary with estimated token costs

    Example:
        >>> cost = estimate_token_cost("high", num_images=2)
        >>> print(cost)
        {'min': 258, 'max': 516, 'detail': 'high', 'images': 2}
    """
    base_costs = {
        "low": 85,
        "high": 129,  # Minimum, can be higher with tiles
        "auto": 100   # Average estimate
    }

    base = base_costs.get(detail, 100)

    return {
        "min": base * num_images,
        "max": base * num_images * 2 if detail == "high" else base * num_images,
        "detail": detail,
        "images": num_images,
        "note": "Actual cost may vary based on image size and complexity"
    }


def get_image_limits() -> Dict[str, Any]:
    """
    Get image size and format limits.

    Returns:
        Dictionary with image limits

    Example:
        >>> limits = get_image_limits()
        >>> print(limits["max_file_size_mb"])
        20
    """
    return IMAGE_LIMITS.copy()

# Filepath: code_migration/ai_providers/xai_chat/model_config.py
# Description: xAI Grok model configurations with capabilities and parameters
# Layer: AI Processor
# References: reference_codebase/AIMOS/providers/xAI/

"""
xAI Grok Model Configuration

Defines available Grok models, their capabilities, context lengths, and default parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class ModelConfig:
    """Configuration for a Grok model."""
    name: str
    model_id: str
    category: str
    description: str
    context_length: int = 128000
    supported_params: List[str] = field(default_factory=list)
    default_params: Dict[str, Any] = field(default_factory=dict)
    param_options: Dict[str, List[Any]] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    notes: str = ""


# Base URL for xAI API (OpenAI-compatible)
XAI_API_URL = "https://api.x.ai/v1"

# Chat/Text Models
CHAT_MODELS: Dict[str, ModelConfig] = {
    "grok-2-1212": ModelConfig(
        name="Grok 2",
        model_id="grok-2-1212",
        category="chat",
        description="Latest Grok 2 text model with function calling and structured outputs",
        context_length=128000,
        supported_params=[
            "temperature", "max_tokens", "top_p", "stream", "stop",
            "frequency_penalty", "presence_penalty", "tools", "tool_choice",
            "response_format", "logprobs", "search_parameters"
        ],
        default_params={"temperature": 1.0, "top_p": 1.0, "max_tokens": 4096},
        capabilities=["chat", "function_calling", "structured_outputs", "streaming", "live_search"],
        notes="Recommended for most use cases. Supports live web search."
    ),
    "grok-2-012": ModelConfig(
        name="Grok 2 January 2025",
        model_id="grok-2-012",
        category="chat",
        description="Grok 2 January 2025 version",
        context_length=128000,
        supported_params=[
            "temperature", "max_tokens", "top_p", "stream", "stop",
            "frequency_penalty", "presence_penalty", "tools", "tool_choice"
        ],
        default_params={"temperature": 1.0, "top_p": 1.0, "max_tokens": 4096},
        capabilities=["chat", "function_calling", "streaming"],
        notes="Previous stable release"
    ),
    "grok-beta": ModelConfig(
        name="Grok Beta",
        model_id="grok-beta",
        category="chat",
        description="Beta version with experimental features",
        context_length=128000,
        supported_params=["temperature", "max_tokens", "top_p", "stream", "stop"],
        default_params={"temperature": 1.0, "top_p": 1.0, "max_tokens": 4096},
        capabilities=["chat", "streaming", "experimental"],
        notes="May have experimental features. Use with caution."
    ),
    "grok-3": ModelConfig(
        name="Grok 3",
        model_id="grok-3",
        category="chat",
        description="Grok 3 with advanced reasoning and extended context",
        context_length=256000,
        supported_params=[
            "temperature", "max_tokens", "top_p", "stream", "stop",
            "frequency_penalty", "presence_penalty", "reasoning_effort"
        ],
        default_params={"temperature": 1.0, "top_p": 1.0, "max_tokens": 8192},
        capabilities=["chat", "advanced_reasoning", "streaming", "extended_context"],
        notes="Extended 256K context window and enhanced reasoning capabilities"
    ),
}

# Vision Models (defined for completeness, implemented in separate module)
VISION_MODELS: Dict[str, ModelConfig] = {
    "grok-2-vision-1212": ModelConfig(
        name="Grok 2 Vision",
        model_id="grok-2-vision-1212",
        category="vision",
        description="Grok 2 with image understanding capabilities",
        context_length=128000,
        supported_params=[
            "temperature", "max_tokens", "top_p", "stream", "stop",
            "frequency_penalty", "presence_penalty"
        ],
        default_params={"temperature": 1.0, "top_p": 1.0, "max_tokens": 4096},
        param_options={
            "image_formats": ["jpeg", "png", "gif", "webp"],
            "max_image_size": "20MB"
        },
        capabilities=["vision", "image_understanding", "ocr", "streaming"],
        notes="Supports JPEG, PNG, GIF, WebP up to 20MB"
    ),
}

# Image Generation Models (defined for completeness, implemented in separate module)
IMAGE_MODELS: Dict[str, ModelConfig] = {
    "grok-2-image-1212": ModelConfig(
        name="Aurora",
        model_id="grok-2-image-1212",
        category="image",
        description="Aurora image generator for text-to-image generation",
        context_length=0,
        supported_params=["prompt", "n"],
        default_params={"n": 1},
        param_options={
            "n": list(range(1, 11)),  # 1-10 images
            "output_format": ["jpeg"]
        },
        capabilities=["text_to_image", "photorealistic"],
        notes="Max 10 images per request, 5 requests/second rate limit"
    ),
}

# All models combined
ALL_MODELS: Dict[str, ModelConfig] = {
    **CHAT_MODELS,
    **VISION_MODELS,
    **IMAGE_MODELS
}


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """
    Get configuration for a specific model.

    Args:
        model_name: Name/ID of the model

    Returns:
        ModelConfig object or None if not found
    """
    return ALL_MODELS.get(model_name)


def get_default_chat_model() -> str:
    """Get the default chat model ID."""
    return "grok-2-1212"


def get_default_vision_model() -> str:
    """Get the default vision model ID."""
    return "grok-2-vision-1212"


def get_default_image_model() -> str:
    """Get the default image generation model ID."""
    return "grok-2-image-1212"


def get_default_model(category: str) -> str:
    """
    Get the default model for a category.

    Args:
        category: Model category (chat, vision, image)

    Returns:
        Default model ID for the category
    """
    defaults = {
        "chat": get_default_chat_model(),
        "vision": get_default_vision_model(),
        "image": get_default_image_model()
    }
    return defaults.get(category, get_default_chat_model())


def get_models_by_category(category: str) -> Dict[str, ModelConfig]:
    """
    Get all models in a category.

    Args:
        category: Model category (chat, vision, image)

    Returns:
        Dictionary of models in the category
    """
    categories = {
        "chat": CHAT_MODELS,
        "vision": VISION_MODELS,
        "image": IMAGE_MODELS
    }
    return categories.get(category, {})


def list_chat_models() -> List[str]:
    """Get list of available chat model IDs."""
    return list(CHAT_MODELS.keys())


def validate_model(model_name: str, category: str = "chat") -> bool:
    """
    Validate if a model exists in a category.

    Args:
        model_name: Model ID to validate
        category: Expected category

    Returns:
        True if model exists and matches category
    """
    config = get_model_config(model_name)
    return config is not None and config.category == category

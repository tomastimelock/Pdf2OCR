# Filepath: code_migration/ai_providers/anthropic_streaming/model_config.py
# Description: Model configurations for Anthropic Claude streaming
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/Anthropic/model_config.py

"""
Anthropic Model Configuration
==============================

Central configuration for all Claude models and their streaming-compatible parameters.
Based on official Anthropic API documentation (as of January 2025).

All Claude models support streaming - no special configuration required.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    category: str  # text, vision, structured, tools, thinking
    description: str
    context_window: int = 200000  # Default 200K for most Claude models
    max_output: int = 64000  # Default 64K for most Claude 4.5 models
    supported_params: List[str] = field(default_factory=list)
    default_params: Dict[str, Any] = field(default_factory=dict)
    param_options: Dict[str, List[Any]] = field(default_factory=dict)
    notes: str = ""
    knowledge_cutoff: str = ""
    supports_streaming: bool = True  # All models support streaming


# =============================================================================
# CLAUDE 4.5 MODELS (Current Generation) - RECOMMENDED
# =============================================================================

CLAUDE_45_MODELS = {
    # Claude Opus 4.5 - Maximum Intelligence
    "claude-opus-4-5-20250918": ModelConfig(
        name="claude-opus-4-5-20250918",
        category="text",
        description="Maximum intelligence with practical performance - best for complex reasoning",
        context_window=200000,
        max_output=64000,
        supported_params=["messages", "system", "max_tokens", "temperature", "top_p", "top_k",
                         "stop_sequences", "tools", "tool_choice", "thinking", "metadata", "stream"],
        default_params={
            "max_tokens": 4096,
            "temperature": 1.0,
            "stream": False,
        },
        param_options={
            "temperature": [0.0, 0.25, 0.5, 0.75, 1.0],
            "top_p": [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0],
            "tool_choice": ["auto", "any", "none"],
        },
        notes="Supports vision, extended thinking (summarized), tool use, PDF processing, streaming",
        knowledge_cutoff="March 2025",
        supports_streaming=True
    ),

    # Claude Sonnet 4.5 - Recommended for most use cases
    "claude-sonnet-4-5-20250929": ModelConfig(
        name="claude-sonnet-4-5-20250929",
        category="text",
        description="Best model for complex agents and coding - recommended default",
        context_window=200000,  # 1M beta available
        max_output=64000,
        supported_params=["messages", "system", "max_tokens", "temperature", "top_p", "top_k",
                         "stop_sequences", "tools", "tool_choice", "thinking", "metadata", "stream"],
        default_params={
            "max_tokens": 4096,
            "temperature": 1.0,
            "stream": False,
        },
        param_options={
            "temperature": [0.0, 0.25, 0.5, 0.75, 1.0],
            "top_p": [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0],
            "tool_choice": ["auto", "any", "none"],
        },
        notes="Supports vision, extended thinking (summarized), tool use, PDF processing, streaming. 1M context beta available.",
        knowledge_cutoff="January 2025",
        supports_streaming=True
    ),

    # Claude Haiku 4.5 - Fastest
    "claude-haiku-4-5-20251001": ModelConfig(
        name="claude-haiku-4-5-20251001",
        category="text",
        description="Fastest model with near-frontier intelligence - speed-critical applications",
        context_window=200000,
        max_output=64000,
        supported_params=["messages", "system", "max_tokens", "temperature", "top_p", "top_k",
                         "stop_sequences", "tools", "tool_choice", "thinking", "metadata", "stream"],
        default_params={
            "max_tokens": 4096,
            "temperature": 1.0,
            "stream": False,
        },
        param_options={
            "temperature": [0.0, 0.25, 0.5, 0.75, 1.0],
            "top_p": [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0],
            "tool_choice": ["auto", "any", "none"],
        },
        notes="Supports vision, extended thinking (summarized), tool use, streaming. Best for high-volume tasks.",
        knowledge_cutoff="February 2025",
        supports_streaming=True
    ),
}


# =============================================================================
# CLAUDE 4.x MODELS (Previous Generation)
# =============================================================================

CLAUDE_4_MODELS = {
    # Claude Opus 4.1
    "claude-opus-4-1-20250805": ModelConfig(
        name="claude-opus-4-1-20250805",
        category="text",
        description="Exceptional intelligence for specialized complex tasks",
        context_window=200000,
        max_output=32000,
        supported_params=["messages", "system", "max_tokens", "temperature", "top_p", "top_k",
                         "stop_sequences", "tools", "tool_choice", "thinking", "metadata", "stream"],
        default_params={
            "max_tokens": 4096,
            "temperature": 1.0,
            "stream": False,
        },
        param_options={
            "temperature": [0.0, 0.25, 0.5, 0.75, 1.0],
            "tool_choice": ["auto", "any", "none"],
        },
        notes="Supports vision, extended thinking with interleaved thinking beta, tool use, streaming",
        supports_streaming=True
    ),

    # Claude Sonnet 4
    "claude-sonnet-4-20250514": ModelConfig(
        name="claude-sonnet-4-20250514",
        category="text",
        description="Balanced performance and capability",
        context_window=200000,  # 1M beta available
        max_output=64000,
        supported_params=["messages", "system", "max_tokens", "temperature", "top_p", "top_k",
                         "stop_sequences", "tools", "tool_choice", "thinking", "metadata", "stream"],
        default_params={
            "max_tokens": 4096,
            "temperature": 1.0,
            "stream": False,
        },
        param_options={
            "temperature": [0.0, 0.25, 0.5, 0.75, 1.0],
            "tool_choice": ["auto", "any", "none"],
        },
        notes="Supports vision, extended thinking, tool use, streaming. 1M context beta available.",
        supports_streaming=True
    ),
}


# =============================================================================
# CLAUDE 3.x MODELS (Legacy - Some Deprecated)
# =============================================================================

CLAUDE_3_MODELS = {
    # Claude 3.7 Sonnet (Deprecated - Feb 19, 2026)
    "claude-3-7-sonnet-20250219": ModelConfig(
        name="claude-3-7-sonnet-20250219",
        category="text",
        description="[DEPRECATED] Claude 3.7 Sonnet - retiring February 19, 2026",
        context_window=200000,
        max_output=8000,
        supported_params=["messages", "system", "max_tokens", "temperature", "top_p", "top_k",
                         "stop_sequences", "tools", "tool_choice", "thinking", "metadata", "stream"],
        default_params={
            "max_tokens": 4096,
            "temperature": 1.0,
            "stream": False,
        },
        param_options={
            "temperature": [0.0, 0.25, 0.5, 0.75, 1.0],
            "tool_choice": ["auto", "any", "none"],
        },
        notes="DEPRECATED - Returns full thinking output (not summarized). Retire date: Feb 19, 2026. Supports streaming.",
        supports_streaming=True
    ),

    # Claude 3 Opus (Deprecated - Jan 5, 2026)
    "claude-3-opus-20240229": ModelConfig(
        name="claude-3-opus-20240229",
        category="text",
        description="[DEPRECATED] Claude 3 Opus - retiring January 5, 2026",
        context_window=200000,
        max_output=4096,
        supported_params=["messages", "system", "max_tokens", "temperature", "top_p", "top_k",
                         "stop_sequences", "tools", "tool_choice", "metadata", "stream"],
        default_params={
            "max_tokens": 4096,
            "temperature": 1.0,
            "stream": False,
        },
        param_options={
            "temperature": [0.0, 0.25, 0.5, 0.75, 1.0],
            "tool_choice": ["auto", "any", "none"],
        },
        notes="DEPRECATED - Retire date: Jan 5, 2026. Does not support extended thinking. Supports streaming.",
        supports_streaming=True
    ),

    # Claude Haiku 3.5
    "claude-haiku-3-5-20241022": ModelConfig(
        name="claude-haiku-3-5-20241022",
        category="text",
        description="Fast and economical Claude 3.5 Haiku",
        context_window=200000,
        max_output=8000,
        supported_params=["messages", "system", "max_tokens", "temperature", "top_p", "top_k",
                         "stop_sequences", "tools", "tool_choice", "metadata", "stream"],
        default_params={
            "max_tokens": 4096,
            "temperature": 1.0,
            "stream": False,
        },
        param_options={
            "temperature": [0.0, 0.25, 0.5, 0.75, 1.0],
            "tool_choice": ["auto", "any", "none"],
        },
        notes="Fast processing, good for simple tasks. Does not support extended thinking. Supports streaming.",
        supports_streaming=True
    ),
}


# Combined dictionary for convenience
ANTHROPIC_MODELS = {
    **CLAUDE_45_MODELS,
    **CLAUDE_4_MODELS,
    **CLAUDE_3_MODELS,
}


# =============================================================================
# STREAMING CONFIGURATION
# =============================================================================

STREAMING_CONFIG = {
    "supported_models": list(ANTHROPIC_MODELS.keys()),
    "event_types": [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop"
    ],
    "delta_types": [
        "text_delta",
        "thinking_delta",
        "input_json_delta"
    ],
    "notes": "All Claude models support streaming. Use stream=True in API calls."
}


# =============================================================================
# EXTENDED THINKING CONFIGURATION
# =============================================================================

THINKING_CONFIG = {
    "min_budget_tokens": 1024,
    "recommended_moderate": 10000,
    "recommended_complex": 32000,
    "max_budget_tokens": 128000,  # For batch processing
    "supported_models": [
        "claude-opus-4-5-20250918",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-1-20250805",
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-20250219",
    ],
    "interleaved_thinking_models": [
        "claude-opus-4-5-20250918",
        "claude-opus-4-1-20250805",
    ],
    "notes": {
        "summarized": "Claude 4+ models provide summarized thinking output",
        "full": "Claude 3.7 Sonnet returns full thinking output",
        "interleaved": "Use beta header 'interleaved-thinking-2025-05-14' for Claude 4 models",
        "streaming": "Extended thinking fully supports streaming"
    }
}


# =============================================================================
# VISION CONFIGURATION
# =============================================================================

VISION_CONFIG = {
    "supported_formats": ["image/jpeg", "image/png", "image/gif", "image/webp"],
    "max_dimensions": {"width": 8000, "height": 8000},
    "max_dimensions_20plus_images": {"width": 2000, "height": 2000},
    "optimal_megapixels": 1.15,
    "optimal_max_dimension": 1568,
    "max_file_size_api": "5MB",
    "max_file_size_claude_ai": "10MB",
    "max_request_size": "32MB",
    "max_images_api": 100,
    "max_images_claude_ai": 20,
    "token_calculation": "tokens = (width * height) / 750",
    "source_types": ["base64", "url", "file"],
    "supported_models": list(ANTHROPIC_MODELS.keys()),
    "streaming_compatible": True
}


# =============================================================================
# TOOL USE CONFIGURATION
# =============================================================================

TOOL_CONFIG = {
    "tool_choice_options": ["auto", "any", "none"],
    "tool_choice_specific": {"type": "tool", "name": "<tool_name>"},
    "max_tools": 128,
    "tool_name_pattern": r"^[a-zA-Z0-9_-]{1,64}$",
    "supported_models": list(ANTHROPIC_MODELS.keys()),
    "server_tools": ["web_search", "web_fetch", "text_editor", "bash", "computer"],
    "beta_headers": {
        "token_efficient": "token-efficient-tools-2025-02-19",
        "fine_grained_streaming": "fine-grained-tool-streaming-2025-05-14",
    },
    "streaming_compatible": True,
    "streaming_notes": "Tool use fully supports streaming with fine-grained events"
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_all_models() -> Dict[str, ModelConfig]:
    """Get all available models."""
    return ANTHROPIC_MODELS


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """Get configuration for a specific model."""
    return ANTHROPIC_MODELS.get(model_name)


def get_models_by_category(category: str) -> Dict[str, ModelConfig]:
    """Get all models in a category."""
    return {k: v for k, v in ANTHROPIC_MODELS.items() if v.category == category}


def get_current_models() -> Dict[str, ModelConfig]:
    """Get only current (non-deprecated) models."""
    return {**CLAUDE_45_MODELS, **CLAUDE_4_MODELS}


def get_deprecated_models() -> Dict[str, ModelConfig]:
    """Get deprecated models."""
    deprecated = {}
    for name, config in ANTHROPIC_MODELS.items():
        if "[DEPRECATED]" in config.description:
            deprecated[name] = config
    return deprecated


def get_streaming_models() -> Dict[str, ModelConfig]:
    """Get all models that support streaming (all of them)."""
    return ANTHROPIC_MODELS


def get_default_model() -> str:
    """Get the default recommended model."""
    return "claude-sonnet-4-5-20250929"


def get_fastest_model() -> str:
    """Get the fastest model."""
    return "claude-haiku-4-5-20251001"


def get_smartest_model() -> str:
    """Get the most capable model."""
    return "claude-opus-4-5-20250918"


def supports_streaming(model_name: str) -> bool:
    """Check if a model supports streaming (all Claude models do)."""
    config = get_model_config(model_name)
    return config.supports_streaming if config else False


def supports_thinking(model_name: str) -> bool:
    """Check if a model supports extended thinking."""
    return model_name in THINKING_CONFIG["supported_models"]


def supports_vision(model_name: str) -> bool:
    """Check if a model supports vision/image input."""
    return model_name in VISION_CONFIG["supported_models"]


def supports_tools(model_name: str) -> bool:
    """Check if a model supports tool use."""
    return model_name in TOOL_CONFIG["supported_models"]


def get_model_context_window(model_name: str) -> int:
    """Get the context window size for a model."""
    config = get_model_config(model_name)
    return config.context_window if config else 200000


def get_model_max_output(model_name: str) -> int:
    """Get the maximum output tokens for a model."""
    config = get_model_config(model_name)
    return config.max_output if config else 4096


def get_valid_params(model_name: str) -> List[str]:
    """Get list of valid parameters for a model."""
    config = get_model_config(model_name)
    if config:
        return config.supported_params
    return []


def get_param_options(model_name: str, param_name: str) -> List[Any]:
    """Get valid options for a parameter on a specific model."""
    config = get_model_config(model_name)
    if config and param_name in config.param_options:
        return config.param_options[param_name]
    return []


def build_payload_template(model_name: str, streaming: bool = False) -> Dict[str, Any]:
    """
    Build a template payload for a model with default values.

    Args:
        model_name: Name of the model
        streaming: Whether to enable streaming

    Returns:
        Template payload dictionary
    """
    config = get_model_config(model_name)
    if not config:
        return {}

    payload = {"model": model_name}
    payload.update(config.default_params)

    if streaming:
        payload["stream"] = True

    return payload


def calculate_image_tokens(width: int, height: int) -> int:
    """Calculate token cost for an image."""
    return int((width * height) / 750)


def get_streaming_event_types() -> List[str]:
    """Get list of streaming event types."""
    return STREAMING_CONFIG["event_types"]


def get_streaming_delta_types() -> List[str]:
    """Get list of streaming delta types."""
    return STREAMING_CONFIG["delta_types"]

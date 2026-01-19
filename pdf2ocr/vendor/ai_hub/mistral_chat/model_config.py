# Filepath: code_migration/ai_providers/mistral_chat/model_config.py
# Description: Mistral AI model configurations and metadata
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/Mistral/

"""
Mistral AI Model Configuration

Centralized configuration for all Mistral AI chat models and their capabilities.
This module provides model metadata, default parameters, and capability information.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class ModelConfig:
    """Configuration for a Mistral model.

    Attributes:
        name: Human-readable model name
        model_id: API identifier for the model
        category: Model category (chat, code, embed, moderation)
        description: Brief description of model capabilities
        context_length: Maximum context window in tokens
        supported_params: List of supported API parameters
        default_params: Default parameter values
        param_options: Valid options for specific parameters
        capabilities: Dictionary of capability flags
        notes: Additional notes or usage tips
    """
    name: str
    model_id: str
    category: str
    description: str
    context_length: int = 32768
    supported_params: List[str] = field(default_factory=list)
    default_params: Dict[str, Any] = field(default_factory=dict)
    param_options: Dict[str, List[Any]] = field(default_factory=dict)
    capabilities: Dict[str, bool] = field(default_factory=dict)
    notes: str = ""


# Chat Models
CHAT_MODELS: Dict[str, ModelConfig] = {
    "mistral-large-latest": ModelConfig(
        name="Mistral Large",
        model_id="mistral-large-latest",
        category="chat",
        description="Most capable model for complex reasoning and analysis",
        context_length=128000,
        supported_params=[
            "max_tokens", "temperature", "top_p", "stream", "safe_prompt",
            "random_seed", "response_format", "tools", "tool_choice"
        ],
        default_params={"temperature": 0.7, "max_tokens": 4096},
        capabilities={
            "chat": True,
            "function_calling": True,
            "vision": False,
            "json_mode": True,
            "streaming": True
        },
        notes="Best for complex reasoning, analysis, and multi-step tasks"
    ),
    "mistral-small-latest": ModelConfig(
        name="Mistral Small",
        model_id="mistral-small-latest",
        category="chat",
        description="Balanced performance and cost for general tasks",
        context_length=32768,
        supported_params=[
            "max_tokens", "temperature", "top_p", "stream", "safe_prompt",
            "random_seed", "response_format", "tools", "tool_choice"
        ],
        default_params={"temperature": 0.7, "max_tokens": 4096},
        capabilities={
            "chat": True,
            "function_calling": True,
            "vision": False,
            "json_mode": True,
            "streaming": True
        },
        notes="Default model - good balance of performance and cost"
    ),
    "ministral-3b-latest": ModelConfig(
        name="Ministral 3B",
        model_id="ministral-3b-latest",
        category="chat",
        description="Efficient small model for simple tasks and low latency",
        context_length=32768,
        supported_params=["max_tokens", "temperature", "top_p", "stream", "safe_prompt"],
        default_params={"temperature": 0.7, "max_tokens": 2048},
        capabilities={
            "chat": True,
            "function_calling": False,
            "vision": False,
            "json_mode": False,
            "streaming": True
        },
        notes="Fast and efficient for simple tasks"
    ),
    "ministral-8b-latest": ModelConfig(
        name="Ministral 8B",
        model_id="ministral-8b-latest",
        category="chat",
        description="Mid-size model for moderate complexity tasks",
        context_length=32768,
        supported_params=[
            "max_tokens", "temperature", "top_p", "stream", "safe_prompt",
            "random_seed", "response_format"
        ],
        default_params={"temperature": 0.7, "max_tokens": 4096},
        capabilities={
            "chat": True,
            "function_calling": True,
            "vision": False,
            "json_mode": True,
            "streaming": True
        },
        notes="Good for moderate complexity tasks with lower latency"
    ),
    "pixtral-12b-latest": ModelConfig(
        name="Pixtral 12B",
        model_id="pixtral-12b-latest",
        category="chat",
        description="Vision-enabled model for image understanding",
        context_length=32768,
        supported_params=["max_tokens", "temperature", "top_p", "stream", "safe_prompt"],
        default_params={"temperature": 0.7, "max_tokens": 4096},
        capabilities={
            "chat": True,
            "function_calling": False,
            "vision": True,
            "json_mode": False,
            "streaming": True
        },
        notes="Supports image inputs for visual understanding tasks"
    ),
    "codestral-latest": ModelConfig(
        name="Codestral",
        model_id="codestral-latest",
        category="code",
        description="Optimized for code generation and understanding",
        context_length=32768,
        supported_params=["max_tokens", "temperature", "top_p", "stream", "stop"],
        default_params={"temperature": 0.2, "max_tokens": 4096},
        capabilities={
            "chat": True,
            "function_calling": False,
            "vision": False,
            "json_mode": False,
            "streaming": True
        },
        notes="Specialized for code generation - use lower temperature"
    ),
    "mistral-medium-latest": ModelConfig(
        name="Mistral Medium",
        model_id="mistral-medium-latest",
        category="chat",
        description="Mid-tier model for balanced performance",
        context_length=32768,
        supported_params=[
            "max_tokens", "temperature", "top_p", "stream", "safe_prompt",
            "random_seed", "response_format", "tools", "tool_choice"
        ],
        default_params={"temperature": 0.7, "max_tokens": 4096},
        capabilities={
            "chat": True,
            "function_calling": True,
            "vision": False,
            "json_mode": True,
            "streaming": True
        },
        notes="Good middle ground between large and small models"
    ),
}

# All available models
ALL_MODELS: Dict[str, ModelConfig] = CHAT_MODELS


# API Configuration
MISTRAL_API_URL = "https://api.mistral.ai/v1"


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """Get configuration for a specific model.

    Args:
        model_name: Name or ID of the model

    Returns:
        ModelConfig object or None if model not found

    Example:
        >>> config = get_model_config("mistral-large-latest")
        >>> print(config.context_length)
        128000
    """
    return ALL_MODELS.get(model_name)


def get_default_chat_model() -> str:
    """Get the default chat model identifier.

    Returns:
        Model ID string for the default chat model

    Example:
        >>> model = get_default_chat_model()
        >>> print(model)
        'mistral-small-latest'
    """
    return "mistral-small-latest"


def list_chat_models() -> Dict[str, ModelConfig]:
    """List all available chat models with their configurations.

    Returns:
        Dictionary mapping model IDs to ModelConfig objects

    Example:
        >>> models = list_chat_models()
        >>> for name, config in models.items():
        ...     print(f"{name}: {config.description}")
    """
    return CHAT_MODELS


def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific model.

    Args:
        model_name: Name or ID of the model

    Returns:
        Dictionary with model details or None if not found

    Example:
        >>> info = get_model_info("mistral-large-latest")
        >>> print(info['capabilities'])
        {'chat': True, 'function_calling': True, ...}
    """
    config = get_model_config(model_name)
    if config:
        return {
            "name": config.name,
            "model_id": config.model_id,
            "category": config.category,
            "description": config.description,
            "context_length": config.context_length,
            "supported_params": config.supported_params,
            "default_params": config.default_params,
            "capabilities": config.capabilities,
            "notes": config.notes
        }
    return None


def validate_model_params(model_name: str, params: Dict[str, Any]) -> bool:
    """Validate parameters against model's supported parameters.

    Args:
        model_name: Name or ID of the model
        params: Dictionary of parameters to validate

    Returns:
        True if all parameters are supported, False otherwise

    Example:
        >>> params = {"temperature": 0.7, "max_tokens": 1000}
        >>> is_valid = validate_model_params("mistral-small-latest", params)
        >>> print(is_valid)
        True
    """
    config = get_model_config(model_name)
    if not config:
        return False

    for param in params.keys():
        if param not in config.supported_params and param != "model" and param != "messages":
            return False

    return True


def get_help_text() -> str:
    """Get comprehensive help text for Mistral Chat.

    Returns:
        Formatted help text string

    Example:
        >>> help_text = get_help_text()
        >>> print(help_text)
    """
    return """
Mistral Chat Provider - Model Information:

AVAILABLE MODELS:
• mistral-large-latest
  - Most capable model for complex reasoning and analysis
  - Context: 128K tokens
  - Features: Function calling, JSON mode, streaming

• mistral-small-latest (default)
  - Balanced performance and cost for general tasks
  - Context: 32K tokens
  - Features: Function calling, JSON mode, streaming

• mistral-medium-latest
  - Mid-tier model for balanced performance
  - Context: 32K tokens
  - Features: Function calling, JSON mode, streaming

• ministral-3b-latest
  - Efficient small model for simple tasks and low latency
  - Context: 32K tokens
  - Features: Basic chat, streaming

• ministral-8b-latest
  - Mid-size model for moderate complexity tasks
  - Context: 32K tokens
  - Features: Function calling, JSON mode, streaming

• pixtral-12b-latest
  - Vision-enabled model for image understanding
  - Context: 32K tokens
  - Features: Image inputs, streaming

• codestral-latest
  - Optimized for code generation and understanding
  - Context: 32K tokens
  - Features: Code completion, streaming

PARAMETERS:
• temperature: Sampling temperature (0.0-1.0)
• max_tokens: Maximum tokens in response
• top_p: Nucleus sampling parameter
• stream: Enable streaming responses
• safe_prompt: Enable safety prompt injection
• random_seed: Seed for deterministic results
• response_format: JSON mode configuration
• tools: Function/tool definitions for calling
• tool_choice: Tool selection mode (auto, any, none)

CAPABILITIES:
• Chat completion with system messages
• Multi-turn conversations with history
• Function/tool calling
• JSON mode with schema validation
• Streaming responses
• Vision understanding (pixtral model)
• Code generation (codestral model)
"""

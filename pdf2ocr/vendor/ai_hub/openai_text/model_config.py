# Filepath: code_migration/ai_providers/openai_text/model_config.py
# Description: OpenAI model configurations for text completion
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/openai/model_config.py

"""
OpenAI Model Configuration
===========================

Centralized configuration for OpenAI text completion models.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for a specific OpenAI model."""

    name: str
    description: str
    max_tokens: int
    context_window: int
    supports_functions: bool = True
    supports_vision: bool = False
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'description': self.description,
            'max_tokens': self.max_tokens,
            'context_window': self.context_window,
            'supports_functions': self.supports_functions,
            'supports_vision': self.supports_vision,
            'cost_per_1k_input': self.cost_per_1k_input,
            'cost_per_1k_output': self.cost_per_1k_output,
            'notes': self.notes
        }


# =============================================================================
# TEXT COMPLETION MODELS
# =============================================================================

TEXT_MODELS = {
    "gpt-4o": ModelConfig(
        name="gpt-4o",
        description="GPT-4o - Most capable multimodal model, fast and efficient",
        max_tokens=16384,
        context_window=128000,
        supports_functions=True,
        supports_vision=True,
        cost_per_1k_input=2.50,
        cost_per_1k_output=10.00,
        notes="Best for complex reasoning, Swedish language, structured output"
    ),

    "gpt-4o-mini": ModelConfig(
        name="gpt-4o-mini",
        description="GPT-4o Mini - Smaller, faster, cheaper GPT-4o variant",
        max_tokens=16384,
        context_window=128000,
        supports_functions=True,
        supports_vision=True,
        cost_per_1k_input=0.15,
        cost_per_1k_output=0.60,
        notes="Great for most tasks, excellent cost/performance ratio"
    ),

    "gpt-4-turbo": ModelConfig(
        name="gpt-4-turbo",
        description="GPT-4 Turbo - Large context window, high capability",
        max_tokens=4096,
        context_window=128000,
        supports_functions=True,
        supports_vision=True,
        cost_per_1k_input=10.00,
        cost_per_1k_output=30.00,
        notes="Use for long documents requiring deep analysis"
    ),

    "gpt-3.5-turbo": ModelConfig(
        name="gpt-3.5-turbo",
        description="GPT-3.5 Turbo - Fast, cost-effective for simple tasks",
        max_tokens=4096,
        context_window=16385,
        supports_functions=True,
        supports_vision=False,
        cost_per_1k_input=0.50,
        cost_per_1k_output=1.50,
        notes="Good for basic extraction and simple classification"
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """
    Get configuration for a specific model.

    Args:
        model_name: Name of the model (e.g., "gpt-4o")

    Returns:
        ModelConfig object or None if not found
    """
    return TEXT_MODELS.get(model_name)


def get_default_model() -> str:
    """
    Get the default model name.

    Returns:
        Default model name (gpt-4o)
    """
    return "gpt-4o"


def list_all_models() -> Dict[str, Dict[str, Any]]:
    """
    List all available models with their configurations.

    Returns:
        Dictionary mapping model names to configuration dicts
    """
    return {
        name: config.to_dict()
        for name, config in TEXT_MODELS.items()
    }


def get_model_by_capability(
    vision_required: bool = False,
    max_cost_per_1k: Optional[float] = None
) -> List[str]:
    """
    Get models matching specific capability requirements.

    Args:
        vision_required: Whether vision support is required
        max_cost_per_1k: Maximum cost per 1k input tokens

    Returns:
        List of matching model names
    """
    matching = []

    for name, config in TEXT_MODELS.items():
        # Check vision requirement
        if vision_required and not config.supports_vision:
            continue

        # Check cost requirement
        if max_cost_per_1k is not None and config.cost_per_1k_input > max_cost_per_1k:
            continue

        matching.append(name)

    return matching


def get_recommended_model(task_type: str) -> str:
    """
    Get recommended model for a specific task type.

    Args:
        task_type: One of "extraction", "classification", "generation",
                  "analysis", "translation"

    Returns:
        Recommended model name
    """
    recommendations = {
        "extraction": "gpt-4o",  # Accuracy is critical
        "classification": "gpt-4o-mini",  # Balance of speed and accuracy
        "generation": "gpt-4o",  # Quality matters
        "analysis": "gpt-4o",  # Deep reasoning needed
        "translation": "gpt-4o-mini",  # Good multilingual support
        "summarization": "gpt-4o-mini",  # Fast and capable
        "simple": "gpt-3.5-turbo",  # Basic tasks
    }

    return recommendations.get(task_type, "gpt-4o")


def estimate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int
) -> float:
    """
    Estimate cost for a completion.

    Args:
        model_name: Name of the model
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Estimated cost in USD
    """
    config = get_model_config(model_name)
    if not config:
        return 0.0

    input_cost = (input_tokens / 1000) * config.cost_per_1k_input
    output_cost = (output_tokens / 1000) * config.cost_per_1k_output

    return input_cost + output_cost


# =============================================================================
# CONFIGURATION PRESETS
# =============================================================================

EXTRACTION_PRESET = {
    "temperature": 0.1,
    "max_tokens": 2000,
    "top_p": 1.0,
}

CLASSIFICATION_PRESET = {
    "temperature": 0.0,
    "max_tokens": 100,
    "top_p": 1.0,
}

GENERATION_PRESET = {
    "temperature": 0.7,
    "max_tokens": 4000,
    "top_p": 0.9,
}

ANALYSIS_PRESET = {
    "temperature": 0.3,
    "max_tokens": 3000,
    "top_p": 1.0,
}


def get_preset(preset_name: str) -> Dict[str, Any]:
    """
    Get parameter preset for common tasks.

    Args:
        preset_name: One of "extraction", "classification", "generation", "analysis"

    Returns:
        Dictionary of parameters
    """
    presets = {
        "extraction": EXTRACTION_PRESET,
        "classification": CLASSIFICATION_PRESET,
        "generation": GENERATION_PRESET,
        "analysis": ANALYSIS_PRESET,
    }

    return presets.get(preset_name, {}).copy()

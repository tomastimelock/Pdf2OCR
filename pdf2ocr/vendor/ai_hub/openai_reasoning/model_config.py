# Filepath: code_migration/ai_providers/openai_reasoning/model_config.py
# Description: Model configurations for OpenAI reasoning models
# Layer: AI Adapter
# References: reference_codebase/AIMOS/providers/openai/model_config.py

"""
OpenAI Reasoning Model Configuration
====================================

Configurations for all OpenAI models that support reasoning/chain-of-thought.

Model Categories:
-----------------
1. Legacy API (Chat Completions): o1, o1-mini, o1-preview
2. Responses API: o3, o3-mini, gpt-5 family

Key Differences:
----------------
Legacy API:
- Uses max_completion_tokens instead of max_tokens
- No system messages support
- No temperature/top_p control
- Reasoning tokens hidden

Responses API:
- Uses 'reasoning' parameter with effort levels
- Uses 'instructions' instead of system messages
- Supports reasoning summary extraction
- More explicit reasoning control
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class ReasoningModelConfig:
    """Configuration for a reasoning model."""
    name: str
    api_type: str  # 'chat' or 'responses'
    description: str
    max_tokens: int
    supports_reasoning_param: bool
    supported_params: List[str] = field(default_factory=list)
    default_params: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


# =============================================================================
# REASONING MODELS - LEGACY CHAT COMPLETIONS API
# =============================================================================

LEGACY_REASONING_MODELS = {
    "o1": ReasoningModelConfig(
        name="o1",
        api_type="chat",
        description="O1 reasoning model - deep chain-of-thought reasoning",
        max_tokens=128000,  # Context window
        supports_reasoning_param=False,
        supported_params=["messages", "max_completion_tokens"],
        default_params={
            "max_completion_tokens": 32000
        },
        notes=(
            "Original o1 model. No system messages, temperature, or top_p. "
            "Reasoning tokens are billed but not visible in response. "
            "Use max_completion_tokens (not max_tokens)."
        )
    ),
    "o1-mini": ReasoningModelConfig(
        name="o1-mini",
        api_type="chat",
        description="Smaller O1 model - faster, cheaper reasoning",
        max_tokens=128000,
        supports_reasoning_param=False,
        supported_params=["messages", "max_completion_tokens"],
        default_params={
            "max_completion_tokens": 16000
        },
        notes=(
            "Smaller, faster o1 variant. Best for most reasoning tasks. "
            "80% cheaper than o1. No system messages or temperature control."
        )
    ),
    "o1-preview": ReasoningModelConfig(
        name="o1-preview",
        api_type="chat",
        description="O1 preview version",
        max_tokens=128000,
        supports_reasoning_param=False,
        supported_params=["messages", "max_completion_tokens"],
        default_params={
            "max_completion_tokens": 32000
        },
        notes="Preview version of o1. Same limitations as o1."
    ),
}


# =============================================================================
# REASONING MODELS - RESPONSES API
# =============================================================================

RESPONSES_REASONING_MODELS = {
    "o3": ReasoningModelConfig(
        name="o3",
        api_type="responses",
        description="O3 reasoning model - most advanced reasoning",
        max_tokens=200000,
        supports_reasoning_param=True,
        supported_params=["input", "reasoning", "instructions", "max_output_tokens", "tools"],
        default_params={
            "reasoning": {"effort": "medium"},
            "max_output_tokens": 16000
        },
        notes=(
            "Advanced o3 model with explicit reasoning control. "
            "Supports effort levels (low, medium, high) and reasoning summaries. "
            "Can use tools/image generation."
        )
    ),
    "o3-mini": ReasoningModelConfig(
        name="o3-mini",
        api_type="responses",
        description="Smaller O3 model - efficient reasoning",
        max_tokens=200000,
        supports_reasoning_param=True,
        supported_params=["input", "reasoning", "instructions", "max_output_tokens"],
        default_params={
            "reasoning": {"effort": "medium"},
            "max_output_tokens": 8000
        },
        notes="Smaller o3 variant. Good balance of quality and cost."
    ),
    "gpt-5": ReasoningModelConfig(
        name="gpt-5",
        api_type="responses",
        description="GPT-5 with reasoning support - most capable",
        max_tokens=200000,
        supports_reasoning_param=True,
        supported_params=["input", "reasoning", "instructions", "max_output_tokens", "tools"],
        default_params={
            "reasoning": {"effort": "medium"},
            "max_output_tokens": 16000
        },
        notes=(
            "Full GPT-5 model with reasoning. Best for complex tasks, "
            "coding, planning. Supports image generation tool."
        )
    ),
    "gpt-5-mini": ReasoningModelConfig(
        name="gpt-5-mini",
        api_type="responses",
        description="GPT-5 mini with reasoning - balanced performance",
        max_tokens=200000,
        supports_reasoning_param=True,
        supported_params=["input", "reasoning", "instructions", "max_output_tokens", "tools"],
        default_params={
            "reasoning": {"effort": "medium"},
            "max_output_tokens": 8000
        },
        notes="Smaller GPT-5. Good quality at lower cost."
    ),
    "gpt-5-nano": ReasoningModelConfig(
        name="gpt-5-nano",
        api_type="responses",
        description="GPT-5 nano with reasoning - fastest and cheapest",
        max_tokens=200000,
        supports_reasoning_param=True,
        supported_params=["input", "reasoning", "instructions", "max_output_tokens"],
        default_params={
            "reasoning": {"effort": "low"},
            "max_output_tokens": 4000
        },
        notes="Smallest GPT-5. Quick responses for simpler reasoning tasks."
    ),
    "o4-mini": ReasoningModelConfig(
        name="o4-mini",
        api_type="responses",
        description="O4 mini reasoning model",
        max_tokens=200000,
        supports_reasoning_param=True,
        supported_params=["input", "reasoning", "instructions", "max_output_tokens"],
        default_params={
            "reasoning": {"effort": "medium"},
            "max_output_tokens": 8000
        },
        notes="O4 series mini model with reasoning support."
    ),
}


# Combined registry
REASONING_MODELS = {
    **LEGACY_REASONING_MODELS,
    **RESPONSES_REASONING_MODELS
}


# =============================================================================
# REASONING PARAMETERS
# =============================================================================

EFFORT_LEVELS = ["low", "medium", "high"]

SUMMARY_MODES = ["auto", "concise", "detailed"]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_reasoning_model_config(model_name: str) -> Optional[ReasoningModelConfig]:
    """
    Get configuration for a reasoning model.

    Args:
        model_name: Name of the model

    Returns:
        Model configuration or None if not found
    """
    return REASONING_MODELS.get(model_name)


def is_responses_api_model(model_name: str) -> bool:
    """
    Check if model uses the new Responses API.

    Args:
        model_name: Name of the model

    Returns:
        True if uses Responses API, False if uses Chat Completions API
    """
    config = get_reasoning_model_config(model_name)
    if config:
        return config.api_type == "responses"
    return False


def is_legacy_api_model(model_name: str) -> bool:
    """
    Check if model uses the legacy Chat Completions API.

    Args:
        model_name: Name of the model

    Returns:
        True if uses Chat Completions API
    """
    return not is_responses_api_model(model_name)


def get_default_reasoning_model() -> str:
    """
    Get the recommended default reasoning model.

    Returns:
        Model name (currently o1-mini for best balance)
    """
    return "o1-mini"


def get_models_by_api_type(api_type: str) -> Dict[str, ReasoningModelConfig]:
    """
    Get all models for a specific API type.

    Args:
        api_type: 'chat' or 'responses'

    Returns:
        Dictionary of models for that API
    """
    return {
        name: config
        for name, config in REASONING_MODELS.items()
        if config.api_type == api_type
    }


def list_all_models() -> List[str]:
    """
    List all available reasoning model names.

    Returns:
        List of model names
    """
    return list(REASONING_MODELS.keys())


def get_recommended_effort(task_type: str) -> str:
    """
    Get recommended effort level for a task type.

    Args:
        task_type: Type of task (code, math, analysis, planning, simple)

    Returns:
        Recommended effort level
    """
    recommendations = {
        "code": "high",
        "math": "high",
        "planning": "high",
        "analysis": "medium",
        "simple": "low",
        "default": "medium"
    }
    return recommendations.get(task_type, recommendations["default"])


def validate_effort_level(effort: str) -> bool:
    """
    Validate effort level.

    Args:
        effort: Effort level to validate

    Returns:
        True if valid
    """
    return effort in EFFORT_LEVELS


def validate_summary_mode(mode: str) -> bool:
    """
    Validate summary mode.

    Args:
        mode: Summary mode to validate

    Returns:
        True if valid
    """
    return mode in SUMMARY_MODES


# =============================================================================
# MODEL SELECTION HELPERS
# =============================================================================

def recommend_model(
    task_type: str = "general",
    budget: str = "medium",
    speed: str = "medium"
) -> str:
    """
    Recommend a model based on requirements.

    Args:
        task_type: Type of task (code, math, analysis, planning, simple)
        budget: Budget level (low, medium, high)
        speed: Speed requirement (fast, medium, slow)

    Returns:
        Recommended model name
    """
    # Complex tasks need full models
    if task_type in ["code", "math", "planning"]:
        if budget == "high":
            return "o3" if speed != "fast" else "o3-mini"
        else:
            return "o1" if budget == "medium" else "o1-mini"

    # Simple tasks can use smaller models
    if task_type == "simple":
        return "gpt-5-nano" if budget == "low" else "o1-mini"

    # Default: o1-mini for best balance
    return "o1-mini"


if __name__ == "__main__":
    # Test the configuration
    print("=== OpenAI Reasoning Models ===\n")

    print("Legacy API (Chat Completions):")
    for name, config in LEGACY_REASONING_MODELS.items():
        print(f"  {name}: {config.description}")

    print("\nResponses API:")
    for name, config in RESPONSES_REASONING_MODELS.items():
        print(f"  {name}: {config.description}")

    print(f"\nDefault model: {get_default_reasoning_model()}")
    print(f"Effort levels: {EFFORT_LEVELS}")
    print(f"Summary modes: {SUMMARY_MODES}")

    print("\nModel recommendations:")
    print(f"  Code generation: {recommend_model('code', 'high', 'medium')}")
    print(f"  Math problems: {recommend_model('math', 'medium', 'fast')}")
    print(f"  Simple analysis: {recommend_model('simple', 'low', 'fast')}")

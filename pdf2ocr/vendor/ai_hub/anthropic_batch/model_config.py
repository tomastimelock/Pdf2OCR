# Filepath: code_migration/ai_providers/anthropic_batch/model_config.py
# Description: Model configurations for Anthropic Batch API
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/Anthropic/model_config.py

"""
Anthropic Model Configuration for Batch Processing
Central configuration for batch-compatible Claude models.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    name: str
    category: str
    description: str
    context_window: int = 200000
    max_output: int = 64000
    supported_params: List[str] = field(default_factory=list)
    default_params: Dict[str, Any] = field(default_factory=dict)
    param_options: Dict[str, List[Any]] = field(default_factory=dict)
    notes: str = ""
    knowledge_cutoff: str = ""


# =============================================================================
# CLAUDE 4.5 MODELS (Current Generation - Batch Compatible)
# =============================================================================

CLAUDE_45_MODELS = {
    # Claude Opus 4.5 - Maximum Intelligence
    "claude-opus-4-5-20250918": ModelConfig(
        name="claude-opus-4-5-20250918",
        category="text",
        description="Maximum intelligence with practical performance - best for complex reasoning",
        context_window=200000,
        max_output=64000,
        supported_params=[
            "messages",
            "system",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
            "tools",
            "tool_choice",
            "thinking",
            "metadata",
        ],
        default_params={
            "max_tokens": 4096,
            "temperature": 1.0,
        },
        param_options={
            "temperature": [0.0, 0.25, 0.5, 0.75, 1.0],
            "top_p": [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0],
            "tool_choice": ["auto", "any", "none"],
        },
        notes="Supports vision, extended thinking (summarized), tool use, PDF processing. Batch compatible.",
        knowledge_cutoff="March 2025",
    ),
    # Claude Sonnet 4.5 - Recommended for most use cases
    "claude-sonnet-4-5-20250929": ModelConfig(
        name="claude-sonnet-4-5-20250929",
        category="text",
        description="Best model for complex agents and coding - recommended default",
        context_window=200000,
        max_output=64000,
        supported_params=[
            "messages",
            "system",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
            "tools",
            "tool_choice",
            "thinking",
            "metadata",
        ],
        default_params={
            "max_tokens": 4096,
            "temperature": 1.0,
        },
        param_options={
            "temperature": [0.0, 0.25, 0.5, 0.75, 1.0],
            "top_p": [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0],
            "tool_choice": ["auto", "any", "none"],
        },
        notes="Supports vision, extended thinking (summarized), tool use, PDF processing. Batch compatible.",
        knowledge_cutoff="January 2025",
    ),
    # Claude Haiku 4.5 - Fastest
    "claude-haiku-4-5-20251001": ModelConfig(
        name="claude-haiku-4-5-20251001",
        category="text",
        description="Fastest model with near-frontier intelligence - speed-critical applications",
        context_window=200000,
        max_output=64000,
        supported_params=[
            "messages",
            "system",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
            "tools",
            "tool_choice",
            "thinking",
            "metadata",
        ],
        default_params={
            "max_tokens": 4096,
            "temperature": 1.0,
        },
        param_options={
            "temperature": [0.0, 0.25, 0.5, 0.75, 1.0],
            "top_p": [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0],
            "tool_choice": ["auto", "any", "none"],
        },
        notes="Supports vision, extended thinking (summarized), tool use. Best for high-volume batch tasks.",
        knowledge_cutoff="February 2025",
    ),
}


# =============================================================================
# CLAUDE 4.x MODELS (Previous Generation - Batch Compatible)
# =============================================================================

CLAUDE_4_MODELS = {
    # Claude Opus 4.1
    "claude-opus-4-1-20250805": ModelConfig(
        name="claude-opus-4-1-20250805",
        category="text",
        description="Exceptional intelligence for specialized complex tasks",
        context_window=200000,
        max_output=32000,
        supported_params=[
            "messages",
            "system",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
            "tools",
            "tool_choice",
            "thinking",
            "metadata",
        ],
        default_params={
            "max_tokens": 4096,
            "temperature": 1.0,
        },
        param_options={
            "temperature": [0.0, 0.25, 0.5, 0.75, 1.0],
            "tool_choice": ["auto", "any", "none"],
        },
        notes="Supports vision, extended thinking with interleaved thinking beta, tool use. Batch compatible.",
    ),
    # Claude Sonnet 4
    "claude-sonnet-4-20250514": ModelConfig(
        name="claude-sonnet-4-20250514",
        category="text",
        description="Balanced performance and capability",
        context_window=200000,
        max_output=64000,
        supported_params=[
            "messages",
            "system",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
            "tools",
            "tool_choice",
            "thinking",
            "metadata",
        ],
        default_params={
            "max_tokens": 4096,
            "temperature": 1.0,
        },
        param_options={
            "temperature": [0.0, 0.25, 0.5, 0.75, 1.0],
            "tool_choice": ["auto", "any", "none"],
        },
        notes="Supports vision, extended thinking, tool use. Batch compatible.",
    ),
}


# =============================================================================
# BATCH API CONFIGURATION
# =============================================================================

BATCH_CONFIG = {
    "discount": "50%",
    "max_requests_per_batch": 100000,
    "max_batch_size_mb": 256,
    "result_availability_hours": 24,
    "typical_completion_time": "1 hour",
    "endpoint": "https://api.anthropic.com/v1/messages/batches",
    "supported_models": list(CLAUDE_45_MODELS.keys()) + list(CLAUDE_4_MODELS.keys()),
    "best_practices": [
        "Use batches for non-time-sensitive workloads",
        "Group similar requests together",
        "Use custom_id for easy result mapping",
        "Monitor status periodically (60s interval recommended)",
        "Download results within 24 hours",
        "Use extended thinking for complex reasoning",
        "Set appropriate max_tokens per request",
        "Use Haiku for simple tasks (fastest + cheapest)",
    ],
}


# =============================================================================
# EXTENDED THINKING CONFIGURATION (for Batch)
# =============================================================================

THINKING_CONFIG = {
    "min_budget_tokens": 1024,
    "recommended_moderate": 10000,
    "recommended_complex": 32000,
    "max_budget_tokens": 128000,  # Higher limit for batch processing
    "supported_models": [
        "claude-opus-4-5-20250918",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-1-20250805",
        "claude-sonnet-4-20250514",
    ],
    "notes": {
        "summarized": "Claude 4+ models provide summarized thinking output",
        "batch_advantage": "Batch API supports larger thinking budgets for complex analysis",
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_all_models() -> Dict[str, ModelConfig]:
    """Get all batch-compatible models."""
    return {
        **CLAUDE_45_MODELS,
        **CLAUDE_4_MODELS,
    }


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """Get configuration for a specific model."""
    all_models = get_all_models()
    return all_models.get(model_name)


def get_models_by_category(category: str) -> Dict[str, ModelConfig]:
    """Get all models in a category."""
    all_models = get_all_models()
    return {k: v for k, v in all_models.items() if v.category == category}


def get_current_models() -> Dict[str, ModelConfig]:
    """Get current generation models (Claude 4.5 and 4.x)."""
    return {**CLAUDE_45_MODELS, **CLAUDE_4_MODELS}


def get_default_model() -> str:
    """Get the default recommended model for batch processing."""
    return "claude-sonnet-4-5-20250929"


def get_fastest_model() -> str:
    """Get the fastest model for batch processing."""
    return "claude-haiku-4-5-20251001"


def get_smartest_model() -> str:
    """Get the most capable model."""
    return "claude-opus-4-5-20250918"


def get_most_economical_model() -> str:
    """Get the most economical model for batch processing."""
    return "claude-haiku-4-5-20251001"


def supports_thinking(model_name: str) -> bool:
    """Check if a model supports extended thinking."""
    return model_name in THINKING_CONFIG["supported_models"]


def supports_batch(model_name: str) -> bool:
    """Check if a model supports batch processing."""
    return model_name in BATCH_CONFIG["supported_models"]


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


def build_payload_template(model_name: str) -> Dict[str, Any]:
    """Build a template payload for a model with default values."""
    config = get_model_config(model_name)
    if not config:
        return {}

    payload = {"model": model_name}
    payload.update(config.default_params)
    return payload


def get_batch_recommendations(
    num_requests: int, complexity: str = "moderate"
) -> Dict[str, Any]:
    """
    Get model recommendations for batch processing.

    Args:
        num_requests: Number of requests in the batch
        complexity: Task complexity ("simple", "moderate", "complex")

    Returns:
        Dictionary with model recommendations and settings
    """
    if complexity == "simple":
        recommended_model = get_fastest_model()
        max_tokens = 1024
        thinking_budget = None
    elif complexity == "complex":
        recommended_model = get_smartest_model()
        max_tokens = 8192
        thinking_budget = 32000
    else:  # moderate
        recommended_model = get_default_model()
        max_tokens = 4096
        thinking_budget = 10000

    estimated_time_hours = max(1, num_requests // 10000)  # Rough estimate

    return {
        "recommended_model": recommended_model,
        "max_tokens": max_tokens,
        "thinking_budget": thinking_budget,
        "estimated_time_hours": estimated_time_hours,
        "cost_savings": "50% vs standard API",
        "total_requests": num_requests,
        "complexity": complexity,
    }


def validate_batch_request(request: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate a batch request format.

    Args:
        request: Request object to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for custom_id
    if "custom_id" not in request:
        return False, "Missing required field: custom_id"

    # Check for either params or simplified format
    if "params" not in request and "prompt" not in request and "messages" not in request:
        return False, "Missing required field: params, prompt, or messages"

    # If params exists, validate it
    if "params" in request:
        params = request["params"]
        if not isinstance(params, dict):
            return False, "params must be a dictionary"
        if "model" not in params:
            return False, "params.model is required"
        if "messages" not in params:
            return False, "params.messages is required"
        if "max_tokens" not in params:
            return False, "params.max_tokens is required"

    return True, None


def estimate_batch_cost(
    num_requests: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    model: str = "claude-sonnet-4-5-20250929",
) -> Dict[str, Any]:
    """
    Estimate cost for batch processing.

    Args:
        num_requests: Number of requests
        avg_input_tokens: Average input tokens per request
        avg_output_tokens: Average output tokens per request
        model: Model name

    Returns:
        Dictionary with cost estimates
    """
    # Pricing per million tokens (batch API - 50% discount applied)
    batch_pricing = {
        "claude-opus-4-5-20250918": {"input": 7.50, "output": 37.50},
        "claude-sonnet-4-5-20250929": {"input": 1.50, "output": 7.50},
        "claude-haiku-4-5-20251001": {"input": 0.40, "output": 2.00},
        "claude-opus-4-1-20250805": {"input": 7.50, "output": 37.50},
        "claude-sonnet-4-20250514": {"input": 1.50, "output": 7.50},
    }

    if model not in batch_pricing:
        model = get_default_model()

    pricing = batch_pricing[model]

    total_input_tokens = num_requests * avg_input_tokens
    total_output_tokens = num_requests * avg_output_tokens

    input_cost = (total_input_tokens / 1_000_000) * pricing["input"]
    output_cost = (total_output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost

    # Compare to standard API (2x the batch price)
    standard_cost = total_cost * 2
    savings = standard_cost - total_cost

    return {
        "model": model,
        "num_requests": num_requests,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "batch_api_cost": round(total_cost, 2),
        "standard_api_cost": round(standard_cost, 2),
        "savings": round(savings, 2),
        "savings_percent": "50%",
        "currency": "USD",
    }

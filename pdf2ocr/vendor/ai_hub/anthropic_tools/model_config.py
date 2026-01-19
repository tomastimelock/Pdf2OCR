# Filepath: code_migration/ai_providers/anthropic_tools/model_config.py
# Description: Model configuration for Anthropic Tool Use
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/Anthropic/model_config.py

"""
Model configuration for Anthropic Tool Use.
"""

from typing import Dict, Any, Union, List


# Tool use configuration
TOOL_CONFIG = {
    "tool_choice_options": ["auto", "any", "none"],
    "tool_choice_specific": {"type": "tool", "name": "<tool_name>"},
    "max_tools": 128,
    "tool_name_pattern": r"^[a-zA-Z0-9_-]{1,64}$",
    "server_tools": ["web_search", "web_fetch", "text_editor", "bash", "computer"],
    "beta_headers": {
        "token_efficient": "token-efficient-tools-2025-02-19",
        "fine_grained_streaming": "fine-grained-tool-streaming-2025-05-14",
    },
    "strict_mode_available": True,
    "parallel_tool_use_supported": True
}


# Models that support tool use
TOOL_SUPPORTED_MODELS = [
    # Claude 4.5 (Current)
    "claude-opus-4-5-20250918",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",

    # Claude 4.x
    "claude-opus-4-1-20250805",
    "claude-sonnet-4-20250514",

    # Claude 3.x (Legacy)
    "claude-3-7-sonnet-20250219",
    "claude-3-opus-20240229",
    "claude-haiku-3-5-20241022",
]


def get_default_model() -> str:
    """Get the default recommended model for tool use."""
    return "claude-sonnet-4-5-20250929"


def get_tool_choice(choice: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert tool_choice input to proper API format.

    Args:
        choice: Tool choice as string or dict
            - "auto": Let Claude decide
            - "any": Require any tool
            - "none": Disable tools
            - "tool_name": Force specific tool
            - {"type": "tool", "name": "tool_name"}: Explicit format

    Returns:
        Formatted tool_choice dict
    """
    if isinstance(choice, dict):
        return choice

    if choice == "auto":
        return {"type": "auto"}
    elif choice == "any":
        return {"type": "any"}
    elif choice == "none":
        return {"type": "none"}
    else:
        # Assume it's a tool name
        return {"type": "tool", "name": choice}


def supports_tools(model_name: str) -> bool:
    """Check if a model supports tool use."""
    return model_name in TOOL_SUPPORTED_MODELS


def get_max_tools() -> int:
    """Get maximum number of tools per request."""
    return TOOL_CONFIG["max_tools"]


def supports_strict_mode(model_name: str) -> bool:
    """Check if a model supports strict mode for tools."""
    # All current models support strict mode
    return model_name in TOOL_SUPPORTED_MODELS


def supports_parallel_tools(model_name: str) -> bool:
    """Check if a model supports parallel tool execution."""
    # All current models support parallel tools
    return model_name in TOOL_SUPPORTED_MODELS


def get_token_efficient_header() -> str:
    """Get beta header for token-efficient tool use (Claude 3.7 only)."""
    return TOOL_CONFIG["beta_headers"]["token_efficient"]


def get_streaming_header() -> str:
    """Get beta header for fine-grained tool streaming."""
    return TOOL_CONFIG["beta_headers"]["fine_grained_streaming"]


# Model-specific recommendations
MODEL_RECOMMENDATIONS = {
    "simple_tools": {
        "model": "claude-haiku-4-5-20251001",
        "reason": "Fast and economical for straightforward tools"
    },
    "complex_tools": {
        "model": "claude-sonnet-4-5-20250929",
        "reason": "Best balance of capability and speed"
    },
    "many_parameters": {
        "model": "claude-opus-4-5-20250918",
        "reason": "Maximum intelligence for complex tool schemas"
    },
    "structured_extraction": {
        "model": "claude-sonnet-4-5-20250929",
        "reason": "Excellent for structured outputs with strict mode"
    }
}


def get_recommended_model(use_case: str) -> str:
    """
    Get recommended model for a use case.

    Args:
        use_case: One of: simple_tools, complex_tools, many_parameters, structured_extraction

    Returns:
        Recommended model name
    """
    recommendation = MODEL_RECOMMENDATIONS.get(use_case)
    if recommendation:
        return recommendation["model"]
    return get_default_model()


# Example tool configurations
EXAMPLE_TOOL_CONFIGS = {
    "weather": {
        "name": "get_weather",
        "description": "Get current weather for a location. Returns temperature, conditions, humidity, and wind speed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state, e.g. 'San Francisco, CA' or 'Paris, France'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit to use"
                }
            },
            "required": ["location"]
        }
    },

    "calculator": {
        "name": "calculate",
        "description": "Perform mathematical calculations. Supports basic arithmetic, trigonometry, and algebra.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate, e.g. '2 + 2' or 'sin(pi/2)'"
                }
            },
            "required": ["expression"]
        }
    },

    "database": {
        "name": "query_database",
        "description": "Query the customer database. Use this when asked about customer information, orders, or account status.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL-like query string"
                },
                "table": {
                    "type": "string",
                    "enum": ["customers", "orders", "products"],
                    "description": "Database table to query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return (default: 10)"
                }
            },
            "required": ["query", "table"]
        }
    },

    "extraction": {
        "name": "extract_structured_data",
        "description": "Extract structured data from text",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Person's full name"
                },
                "age": {
                    "type": "integer",
                    "description": "Person's age in years"
                },
                "email": {
                    "type": "string",
                    "format": "email",
                    "description": "Email address"
                },
                "phone": {
                    "type": "string",
                    "description": "Phone number"
                }
            },
            "required": ["name"]
        },
        "strict": True
    }
}


def get_example_tool(tool_type: str) -> Dict[str, Any]:
    """
    Get an example tool configuration.

    Args:
        tool_type: One of: weather, calculator, database, extraction

    Returns:
        Tool definition
    """
    return EXAMPLE_TOOL_CONFIGS.get(tool_type, {})

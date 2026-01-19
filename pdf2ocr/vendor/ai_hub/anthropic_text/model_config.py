"""
Anthropic Model Configuration
Central configuration for all Claude models and their supported parameters.
Based on official Anthropic API documentation.
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


# =============================================================================
# CLAUDE 4.5 MODELS (Current Generation)
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
                         "stop_sequences", "tools", "tool_choice", "thinking", "metadata"],
        default_params={
            "max_tokens": 4096,
            "temperature": 1.0,
        },
        param_options={
            "temperature": [0.0, 0.25, 0.5, 0.75, 1.0],
            "top_p": [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0],
            "tool_choice": ["auto", "any", "none"],
        },
        notes="Supports vision, extended thinking (summarized), tool use, PDF processing",
        knowledge_cutoff="March 2025"
    ),

    # Claude Sonnet 4.5 - Recommended for most use cases
    "claude-sonnet-4-5-20250929": ModelConfig(
        name="claude-sonnet-4-5-20250929",
        category="text",
        description="Best model for complex agents and coding - recommended default",
        context_window=200000,  # 1M beta available
        max_output=64000,
        supported_params=["messages", "system", "max_tokens", "temperature", "top_p", "top_k",
                         "stop_sequences", "tools", "tool_choice", "thinking", "metadata"],
        default_params={
            "max_tokens": 4096,
            "temperature": 1.0,
        },
        param_options={
            "temperature": [0.0, 0.25, 0.5, 0.75, 1.0],
            "top_p": [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0],
            "tool_choice": ["auto", "any", "none"],
        },
        notes="Supports vision, extended thinking (summarized), tool use, PDF processing. 1M context beta available.",
        knowledge_cutoff="January 2025"
    ),

    # Claude Haiku 4.5 - Fastest
    "claude-haiku-4-5-20251001": ModelConfig(
        name="claude-haiku-4-5-20251001",
        category="text",
        description="Fastest model with near-frontier intelligence - speed-critical applications",
        context_window=200000,
        max_output=64000,
        supported_params=["messages", "system", "max_tokens", "temperature", "top_p", "top_k",
                         "stop_sequences", "tools", "tool_choice", "thinking", "metadata"],
        default_params={
            "max_tokens": 4096,
            "temperature": 1.0,
        },
        param_options={
            "temperature": [0.0, 0.25, 0.5, 0.75, 1.0],
            "top_p": [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0],
            "tool_choice": ["auto", "any", "none"],
        },
        notes="Supports vision, extended thinking (summarized), tool use. Best for high-volume tasks.",
        knowledge_cutoff="February 2025"
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
                         "stop_sequences", "tools", "tool_choice", "thinking", "metadata"],
        default_params={
            "max_tokens": 4096,
            "temperature": 1.0,
        },
        param_options={
            "temperature": [0.0, 0.25, 0.5, 0.75, 1.0],
            "tool_choice": ["auto", "any", "none"],
        },
        notes="Supports vision, extended thinking with interleaved thinking beta, tool use"
    ),

    # Claude Sonnet 4
    "claude-sonnet-4-20250514": ModelConfig(
        name="claude-sonnet-4-20250514",
        category="text",
        description="Balanced performance and capability",
        context_window=200000,  # 1M beta available
        max_output=64000,
        supported_params=["messages", "system", "max_tokens", "temperature", "top_p", "top_k",
                         "stop_sequences", "tools", "tool_choice", "thinking", "metadata"],
        default_params={
            "max_tokens": 4096,
            "temperature": 1.0,
        },
        param_options={
            "temperature": [0.0, 0.25, 0.5, 0.75, 1.0],
            "tool_choice": ["auto", "any", "none"],
        },
        notes="Supports vision, extended thinking, tool use. 1M context beta available."
    ),
}


# =============================================================================
# CLAUDE 3.x MODELS (Legacy - Deprecated)
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
                         "stop_sequences", "tools", "tool_choice", "thinking", "metadata"],
        default_params={
            "max_tokens": 4096,
            "temperature": 1.0,
        },
        param_options={
            "temperature": [0.0, 0.25, 0.5, 0.75, 1.0],
            "tool_choice": ["auto", "any", "none"],
        },
        notes="DEPRECATED - Returns full thinking output (not summarized). Retire date: Feb 19, 2026"
    ),

    # Claude 3 Opus (Deprecated - Jan 5, 2026)
    "claude-3-opus-20240229": ModelConfig(
        name="claude-3-opus-20240229",
        category="text",
        description="[DEPRECATED] Claude 3 Opus - retiring January 5, 2026",
        context_window=200000,
        max_output=4096,
        supported_params=["messages", "system", "max_tokens", "temperature", "top_p", "top_k",
                         "stop_sequences", "tools", "tool_choice", "metadata"],
        default_params={
            "max_tokens": 4096,
            "temperature": 1.0,
        },
        param_options={
            "temperature": [0.0, 0.25, 0.5, 0.75, 1.0],
            "tool_choice": ["auto", "any", "none"],
        },
        notes="DEPRECATED - Retire date: Jan 5, 2026. Does not support extended thinking."
    ),

    # Claude Haiku 3.5
    "claude-haiku-3-5-20241022": ModelConfig(
        name="claude-haiku-3-5-20241022",
        category="text",
        description="Fast and economical Claude 3.5 Haiku",
        context_window=200000,
        max_output=8000,
        supported_params=["messages", "system", "max_tokens", "temperature", "top_p", "top_k",
                         "stop_sequences", "tools", "tool_choice", "metadata"],
        default_params={
            "max_tokens": 4096,
            "temperature": 1.0,
        },
        param_options={
            "temperature": [0.0, 0.25, 0.5, 0.75, 1.0],
            "tool_choice": ["auto", "any", "none"],
        },
        notes="Fast processing, good for simple tasks. Does not support extended thinking."
    ),
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
        "interleaved": "Use beta header 'interleaved-thinking-2025-05-14' for Claude 4 models"
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
    "supported_models": list(CLAUDE_45_MODELS.keys()) + list(CLAUDE_4_MODELS.keys()) + list(CLAUDE_3_MODELS.keys())
}


# =============================================================================
# TOOL USE CONFIGURATION
# =============================================================================

TOOL_CONFIG = {
    "tool_choice_options": ["auto", "any", "none"],
    "tool_choice_specific": {"type": "tool", "name": "<tool_name>"},
    "max_tools": 128,
    "tool_name_pattern": r"^[a-zA-Z0-9_-]{1,64}$",
    "supported_models": list(CLAUDE_45_MODELS.keys()) + list(CLAUDE_4_MODELS.keys()) + list(CLAUDE_3_MODELS.keys()),
    "server_tools": ["web_search", "web_fetch", "text_editor", "bash", "computer"],
    "beta_headers": {
        "token_efficient": "token-efficient-tools-2025-02-19",
        "fine_grained_streaming": "fine-grained-tool-streaming-2025-05-14",
    }
}


# =============================================================================
# STRUCTURED OUTPUTS CONFIGURATION
# =============================================================================

STRUCTURED_OUTPUT_CONFIG = {
    "supported_json_schema_features": [
        "object", "array", "string", "integer", "number", "boolean", "null",
        "enum", "const", "anyOf", "allOf", "$ref", "$defs",
        "date-time", "date", "email", "uri", "uuid", "ipv4", "ipv6",
        "required", "additionalProperties"
    ],
    "unsupported_features": [
        "recursive definitions",
        "external $ref URLs",
        "minimum/maximum",
        "minLength/maxLength",
        "complex array constraints",
        "regex backreferences"
    ],
    "incompatibilities": ["citations", "message prefilling"],
    "notes": "First request with schema incurs compilation latency; 24-hour cache"
}


# =============================================================================
# BATCH API CONFIGURATION
# =============================================================================

BATCH_CONFIG = {
    "discount": "50%",
    "max_requests_per_batch": 100000,
    "result_availability": "24 hours",
    "endpoint": "https://api.anthropic.com/v1/messages/batches",
    "supported_models": list(CLAUDE_45_MODELS.keys()) + list(CLAUDE_4_MODELS.keys())
}


# =============================================================================
# PROMPT CACHING CONFIGURATION
# =============================================================================

PROMPT_CACHING_CONFIG = {
    "cache_duration": "5 minutes",
    "min_cacheable_tokens": {
        "claude-opus-4-5-20250918": 1024,
        "claude-sonnet-4-5-20250929": 1024,
        "claude-haiku-4-5-20251001": 1024,
        "default": 1024
    },
    "cache_breakpoint": {"type": "ephemeral"},
    "discount_on_hit": "90%",
    "supported_content": ["system prompts", "tools", "large documents", "few-shot examples"]
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_all_models() -> Dict[str, ModelConfig]:
    """Get all available models."""
    return {
        **CLAUDE_45_MODELS,
        **CLAUDE_4_MODELS,
        **CLAUDE_3_MODELS,
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
    """Get only current (non-deprecated) models."""
    return {**CLAUDE_45_MODELS, **CLAUDE_4_MODELS}


def get_deprecated_models() -> Dict[str, ModelConfig]:
    """Get deprecated models."""
    return CLAUDE_3_MODELS


def get_default_model() -> str:
    """Get the default recommended model."""
    return "claude-sonnet-4-5-20250929"


def get_fastest_model() -> str:
    """Get the fastest model."""
    return "claude-haiku-4-5-20251001"


def get_smartest_model() -> str:
    """Get the most capable model."""
    return "claude-opus-4-5-20250918"


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


def build_payload_template(model_name: str) -> Dict[str, Any]:
    """Build a template payload for a model with default values."""
    config = get_model_config(model_name)
    if not config:
        return {}

    payload = {"model": model_name}
    payload.update(config.default_params)
    return payload


def calculate_image_tokens(width: int, height: int) -> int:
    """Calculate token cost for an image."""
    return int((width * height) / 750)


def get_help_text(category: str) -> str:
    """Get help text for a category of commands."""
    help_texts = {
        "text": """
ANTHROPIC TEXT GENERATION COMMANDS
===================================

Commands:
---------
TEXT_GENERATE prompt="..." [output="file.txt"] [model="claude-sonnet-4-5-20250929"] [system="..."]
TEXT_GENERATE -> variable_name  (stores result in variable)
TEXT_MESSAGES messages="..." [output="file.txt"] [model="..."]

Models:
-------
- claude-sonnet-4-5-20250929 (default) - Best for complex agents and coding
- claude-opus-4-5-20250918 - Maximum intelligence
- claude-haiku-4-5-20251001 - Fastest, most economical
- claude-opus-4-1-20250805 - Previous gen, 32K max output
- claude-sonnet-4-20250514 - Previous gen balanced

Parameters:
-----------
- prompt: The text prompt to generate from
- output: Optional file path to save the result
- model: Model to use (default: claude-sonnet-4-5-20250929)
- system: System prompt for context and instructions
- max_tokens: Maximum tokens to generate (default: 4096)
- temperature: Randomness 0.0-1.0 (default: 1.0)
- top_p: Nucleus sampling (default: not set)
- stop_sequences: Stop generation at these strings
""",
        "vision": """
ANTHROPIC VISION COMMANDS
=========================

Commands:
---------
VISION_ANALYZE image="path/to/image.jpg" prompt="What's in this image?"
VISION_ANALYZE_URL url="https://..." prompt="Describe this image"
VISION_COMPARE images="img1.jpg,img2.jpg" prompt="Compare these images"
VISION_OCR image="document.png" [output="text.txt"]

Supported Formats:
------------------
JPEG, PNG, GIF, WebP

Size Limits:
------------
- Max dimensions: 8000x8000 pixels (2000x2000 for 20+ images)
- Optimal: 1568 pixels max in any dimension
- Max file size: 5MB per image (API), 10MB (claude.ai)
- Max images: 100 per request

Token Cost:
-----------
tokens = (width × height) / 750
Example: 1092×1092 ≈ 1,590 tokens

Best Practices:
---------------
- Place images before text in content
- Resize large images for efficiency
- Use specific prompts for better results
""",
        "thinking": """
ANTHROPIC EXTENDED THINKING COMMANDS
=====================================

Commands:
---------
THINK prompt="..." budget_tokens=10000 [model="claude-sonnet-4-5-20250929"]
THINK_COMPLEX prompt="..." budget_tokens=32000

Budget Tokens:
--------------
- Minimum: 1,024
- Moderate tasks: 10,000
- Complex analysis: 32,000+
- Note: budget_tokens must be less than max_tokens

Supported Models:
-----------------
- Claude Opus 4.5 (summarized thinking)
- Claude Sonnet 4.5 (summarized thinking)
- Claude Haiku 4.5 (summarized thinking)
- Claude 3.7 Sonnet (full thinking output)

Best For:
---------
- Complex STEM problems
- Multi-step reasoning
- Analysis requiring multiple approaches
- Tasks benefiting from self-verification

Not Recommended For:
--------------------
- Simple Q&A
- Quick responses needed
- Tasks not requiring deep reasoning
""",
        "tools": """
ANTHROPIC TOOL USE COMMANDS
===========================

Commands:
---------
TOOL_CALL tools_file="tools.json" prompt="..." [tool_choice="auto"]
TOOL_DEFINE name="..." description="..." schema="..."

Tool Choice Options:
--------------------
- auto: Let Claude decide (default)
- any: Require any tool
- none: Disable tools
- {"type": "tool", "name": "specific_tool"}: Force specific tool

Server Tools (Anthropic-hosted):
--------------------------------
- web_search: Search the web
- web_fetch: Fetch URL content
- text_editor: Edit text files
- bash: Execute shell commands
- computer: Computer use capabilities

Tool Definition Example:
------------------------
{
    "name": "get_weather",
    "description": "Get weather for a location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string"}
        },
        "required": ["location"]
    }
}
""",
        "structured": """
ANTHROPIC STRUCTURED OUTPUT COMMANDS
====================================

Commands:
---------
STRUCT_EXTRACT schema_file="schema.json" prompt="..." [output="data.json"]
STRUCT_GENERATE schema="..." prompt="..."

Output Format:
--------------
Use output_format parameter with JSON Schema:
{
    "type": "json_schema",
    "json_schema": {
        "name": "schema_name",
        "schema": { ... JSON Schema ... }
    }
}

Supported JSON Schema Features:
-------------------------------
- Basic types: object, array, string, integer, number, boolean, null
- enum, const
- anyOf, allOf
- $ref and $defs
- String formats: date-time, date, email, uri, uuid, ipv4, ipv6
- required, additionalProperties: false

Not Supported:
--------------
- Recursive definitions
- External $ref URLs
- Numerical constraints (minimum, maximum)
- String length constraints
- Complex array constraints

Strict Tool Use:
----------------
Add "strict": true to tool definition for guaranteed schema compliance
""",
        "batch": """
ANTHROPIC BATCH API COMMANDS
============================

Commands:
---------
BATCH_CREATE requests_file="requests.json" [output_dir="results/"]
BATCH_STATUS batch_id="..."
BATCH_RESULTS batch_id="..." [output_dir="results/"]
BATCH_CANCEL batch_id="..."

Features:
---------
- 50% discount on API costs
- Up to 100,000 requests per batch
- Results available for 24 hours
- Asynchronous processing

Request Format:
---------------
{
    "custom_id": "request-1",
    "params": {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 1024,
        "messages": [...]
    }
}

Best For:
---------
- Large-scale processing
- Cost-sensitive applications
- Non-time-sensitive workloads
- Extended thinking with large budgets
""",
    }
    return help_texts.get(category, f"No help available for category: {category}")

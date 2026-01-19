"""
Anthropic Text Generation Provider

Standalone module for text generation using Claude models via the Anthropic Messages API.
Self-contained with no external dependencies beyond the Anthropic SDK.

Features:
- Simple text generation with Claude models
- Multi-turn conversations with message history
- Response prefilling for guided output
- Streaming support for real-time responses
- Command execution DSL for workflows
- Full access to all Claude 4.5, 4.x, and 3.x models

Usage:
    from anthropic_text import AnthropicTextProvider

    # Initialize
    provider = AnthropicTextProvider(api_key="your-key")

    # Generate text
    response = provider.generate(
        prompt="Write a story about robots",
        system="You are a creative writer"
    )

    # Multi-turn conversation
    response = provider.generate_with_messages([
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ])

    # Stream responses
    for chunk in provider.stream_generate(prompt="Tell me a joke"):
        print(chunk, end="", flush=True)
"""

from .provider import AnthropicTextProvider
from .model_config import (
    ModelConfig,
    get_all_models,
    get_model_config,
    get_default_model,
    get_fastest_model,
    get_smartest_model,
    supports_thinking,
    supports_vision,
    supports_tools,
    get_model_context_window,
    get_model_max_output,
)

__all__ = [
    'AnthropicTextProvider',
    'ModelConfig',
    'get_all_models',
    'get_model_config',
    'get_default_model',
    'get_fastest_model',
    'get_smartest_model',
    'supports_thinking',
    'supports_vision',
    'supports_tools',
    'get_model_context_window',
    'get_model_max_output',
]

__version__ = '1.0.0'

# Filepath: code_migration/ai_providers/mistral_chat/__init__.py
# Description: Mistral Chat module initialization and exports
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/Mistral/chat/

"""
Mistral Chat Provider Module

A self-contained, production-ready module for Mistral AI chat completions.
Supports all major chat patterns including streaming, JSON mode, and function calling.

Quick Start:
    >>> from mistral_chat import MistralChatProvider
    >>> provider = MistralChatProvider(api_key="your-api-key")
    >>> response = provider.chat("What is Python?")
    >>> print(response)

Features:
    • Simple chat completion
    • Multi-turn conversations with history
    • Streaming responses
    • JSON mode with schema validation
    • Function/tool calling
    • Multiple model options (large, small, medium, code, vision)
    • Temperature and sampling controls
    • System message support

Environment Setup:
    Set your API key as an environment variable:

    Linux/Mac:
        export MISTRAL_API_KEY='your-api-key'

    Windows (CMD):
        set MISTRAL_API_KEY=your-api-key

    Windows (PowerShell):
        $env:MISTRAL_API_KEY='your-api-key'

    Or create a .env file:
        MISTRAL_API_KEY=your-api-key
        MISTRAL_MODEL=mistral-small-latest  # optional default

Installation:
    pip install -r requirements.txt

Basic Usage Examples:

    1. Simple Chat:
        >>> provider = MistralChatProvider()
        >>> response = provider.chat("Explain quantum computing")
        >>> print(response)

    2. Chat with System Message:
        >>> response = provider.chat(
        ...     message="Explain decorators",
        ...     system="You are a Python expert. Be concise."
        ... )

    3. Multi-turn Conversation:
        >>> history = []
        >>> msg1 = "Hi, I'm learning Python"
        >>> resp1 = provider.chat(msg1, history=history)
        >>> history.extend([
        ...     {"role": "user", "content": msg1},
        ...     {"role": "assistant", "content": resp1}
        ... ])
        >>> msg2 = "What should I learn first?"
        >>> resp2 = provider.chat(msg2, history=history)

    4. Streaming Response:
        >>> for chunk in provider.stream_chat("Write a story"):
        ...     print(chunk, end='', flush=True)

    5. JSON Mode:
        >>> result = provider.chat_with_json(
        ...     message="List 3 programming languages",
        ...     system="Return as JSON with languages array"
        ... )
        >>> print(result)
        {'languages': ['Python', 'JavaScript', 'Java']}

    6. JSON with Schema:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "languages": {
        ...             "type": "array",
        ...             "items": {
        ...                 "type": "object",
        ...                 "properties": {
        ...                     "name": {"type": "string"},
        ...                     "year": {"type": "integer"}
        ...                 }
        ...             }
        ...         }
        ...     }
        ... }
        >>> result = provider.chat_with_json(
        ...     message="List 3 programming languages with creation years",
        ...     json_schema=schema
        ... )

    7. Function Calling:
        >>> tools = [{
        ...     "type": "function",
        ...     "function": {
        ...         "name": "get_weather",
        ...         "description": "Get current weather",
        ...         "parameters": {
        ...             "type": "object",
        ...             "properties": {
        ...                 "location": {"type": "string"}
        ...             },
        ...             "required": ["location"]
        ...         }
        ...     }
        ... }]
        >>> response = provider.chat_with_tools(
        ...     message="What's the weather in Paris?",
        ...     tools=tools
        ... )
        >>> if "tool_calls" in response:
        ...     for call in response["tool_calls"]:
        ...         print(f"Call {call['function']['name']}")

    8. Different Models:
        >>> # Use large model for complex reasoning
        >>> response = provider.chat(
        ...     "Analyze this complex problem...",
        ...     model="mistral-large-latest"
        ... )
        >>> # Use code model for programming
        >>> code = provider.chat(
        ...     "Write a Python function to sort a list",
        ...     model="codestral-latest"
        ... )
        >>> # Use vision model for images
        >>> response = provider.chat(
        ...     "Describe this image",
        ...     model="pixtral-12b-latest"
        ... )

    9. Temperature Control:
        >>> # More deterministic (factual)
        >>> response = provider.chat(
        ...     "What is the capital of France?",
        ...     temperature=0.1
        ... )
        >>> # More creative
        >>> story = provider.chat(
        ...     "Write a creative story",
        ...     temperature=0.9
        ... )

    10. List Available Models:
        >>> models = MistralChatProvider.list_models()
        >>> for name, info in models.items():
        ...     print(f"{name}: {info['description']}")

Available Models:
    • mistral-large-latest - Most capable, 128K context
    • mistral-small-latest - Default, balanced performance
    • mistral-medium-latest - Mid-tier model
    • ministral-3b-latest - Fast, simple tasks
    • ministral-8b-latest - Moderate complexity
    • pixtral-12b-latest - Vision-enabled
    • codestral-latest - Code generation

Common Parameters:
    • message: User message/prompt (required)
    • model: Model to use (optional, defaults to mistral-small-latest)
    • system: System message for instructions (optional)
    • temperature: 0.0-1.0, controls randomness (optional)
    • max_tokens: Maximum response length (optional)
    • history: Previous messages for context (optional)

Error Handling:
    >>> try:
    ...     provider = MistralChatProvider(api_key="invalid")
    ...     response = provider.chat("Hello")
    ... except ValueError as e:
    ...     print(f"Configuration error: {e}")
    ... except Exception as e:
    ...     print(f"API error: {e}")

Advanced Usage:

    Custom Temperature per Request:
        >>> # Factual response
        >>> fact = provider.chat(
        ...     "What is Python?",
        ...     temperature=0.1
        ... )
        >>> # Creative response
        >>> creative = provider.chat(
        ...     "Write a poem about Python",
        ...     temperature=0.9
        ... )

    Max Tokens Limit:
        >>> response = provider.chat(
        ...     "Explain AI",
        ...     max_tokens=100  # Limit response length
        ... )

    Safe Prompt Mode:
        >>> response = provider.chat(
        ...     message="User input here",
        ...     safe_prompt=True  # Enable safety filtering
        ... )

    Deterministic Results:
        >>> response = provider.chat(
        ...     message="Random story",
        ...     random_seed=42  # Same seed = same output
        ... )

    Full Response Object:
        >>> messages = [{"role": "user", "content": "Hello"}]
        >>> response = provider.get_full_response(messages)
        >>> print(response.usage.total_tokens)
        >>> print(response.choices[0].message.content)

Integration Patterns:

    1. Chatbot with Memory:
        >>> class Chatbot:
        ...     def __init__(self):
        ...         self.provider = MistralChatProvider()
        ...         self.history = []
        ...         self.system = "You are a helpful assistant"
        ...
        ...     def chat(self, message):
        ...         response = self.provider.chat(
        ...             message=message,
        ...             system=self.system,
        ...             history=self.history
        ...         )
        ...         self.history.extend([
        ...             {"role": "user", "content": message},
        ...             {"role": "assistant", "content": response}
        ...         ])
        ...         return response

    2. Code Assistant:
        >>> class CodeAssistant:
        ...     def __init__(self):
        ...         self.provider = MistralChatProvider(
        ...             model="codestral-latest"
        ...         )
        ...
        ...     def generate_code(self, description):
        ...         return self.provider.chat(
        ...             message=f"Write Python code: {description}",
        ...             temperature=0.2,
        ...             system="You are an expert Python developer"
        ...         )

    3. Structured Data Extractor:
        >>> class DataExtractor:
        ...     def __init__(self):
        ...         self.provider = MistralChatProvider()
        ...
        ...     def extract(self, text, schema):
        ...         return self.provider.chat_with_json(
        ...             message=f"Extract data from: {text}",
        ...             json_schema=schema,
        ...             temperature=0.1
        ...         )

Notes:
    • All methods support **kwargs for additional API parameters
    • Streaming returns a generator - iterate to get chunks
    • JSON mode may return error dict if parsing fails
    • Function calling requires models with that capability
    • Vision requires pixtral model
    • Temperature 0.0 = deterministic, 1.0 = creative
    • Safe mode and random seed available for reproducibility

Module Structure:
    mistral_chat/
    ├── __init__.py          # This file (exports and docs)
    ├── provider.py          # MistralChatProvider class
    ├── model_config.py      # Model configurations
    └── requirements.txt     # Dependencies

Dependencies:
    • mistralai>=1.0.0 (required)
    • python-dotenv (optional, for .env files)
    • pydantic (optional, for validation)

License:
    MIT License - Free for commercial and personal use

Support:
    • Mistral API Docs: https://docs.mistral.ai/
    • SDK Reference: https://github.com/mistralai/client-python
"""

from .provider import MistralChatProvider
from .model_config import (
    ModelConfig,
    CHAT_MODELS,
    get_model_config,
    get_default_chat_model,
    list_chat_models,
    get_model_info,
    validate_model_params,
    get_help_text,
    MISTRAL_API_URL,
)

__version__ = "1.0.0"
__author__ = "DocumentHandler Team"
__all__ = [
    # Main provider class
    "MistralChatProvider",
    # Model configuration
    "ModelConfig",
    "CHAT_MODELS",
    # Helper functions
    "get_model_config",
    "get_default_chat_model",
    "list_chat_models",
    "get_model_info",
    "validate_model_params",
    "get_help_text",
    # Constants
    "MISTRAL_API_URL",
]


# Module-level convenience functions
def create_provider(api_key: str = None, model: str = None) -> MistralChatProvider:
    """Create a MistralChatProvider instance.

    Convenience function for quick provider creation.

    Args:
        api_key: Mistral API key (optional, uses env var if not provided)
        model: Default model to use (optional)

    Returns:
        Configured MistralChatProvider instance

    Example:
        >>> provider = create_provider()
        >>> response = provider.chat("Hello")
    """
    return MistralChatProvider(api_key=api_key, model=model)


def quick_chat(message: str, **kwargs) -> str:
    """Quick chat without creating a provider instance.

    Creates a temporary provider and sends a single message.
    Useful for one-off requests.

    Args:
        message: User message
        **kwargs: Additional parameters (model, temperature, etc.)

    Returns:
        Response text

    Example:
        >>> response = quick_chat("What is Python?")
        >>> response = quick_chat(
        ...     "Explain AI",
        ...     model="mistral-large-latest",
        ...     temperature=0.5
        ... )
    """
    provider = MistralChatProvider()
    return provider.chat(message=message, **kwargs)


def quick_json(message: str, schema: dict = None, **kwargs) -> dict:
    """Quick JSON response without creating a provider instance.

    Args:
        message: User message
        schema: Optional JSON schema
        **kwargs: Additional parameters

    Returns:
        Parsed JSON response

    Example:
        >>> result = quick_json(
        ...     "List 3 colors",
        ...     system="Return as JSON with colors array"
        ... )
    """
    provider = MistralChatProvider()
    return provider.chat_with_json(message=message, json_schema=schema, **kwargs)

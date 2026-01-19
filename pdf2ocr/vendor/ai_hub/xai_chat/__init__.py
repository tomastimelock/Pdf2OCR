# Filepath: code_migration/ai_providers/xai_chat/__init__.py
# Description: xAI Grok Chat module exports and comprehensive usage documentation
# Layer: AI Processor
# References: reference_codebase/AIMOS/providers/xAI/

"""
xAI Grok Chat Provider Module

A standalone, self-contained module for interacting with xAI's Grok chat models.
Supports text completion, multi-turn conversations, streaming, function calling,
structured JSON output, and live web search.

FEATURES:
---------
- Multiple Grok models (Grok 2, Grok 3, Beta)
- Up to 256K context window (Grok 3)
- Streaming responses
- Function/tool calling
- Structured JSON output with schema validation
- Live web search integration
- OpenAI-compatible API

INSTALLATION:
-------------
pip install -r requirements.txt

ENVIRONMENT SETUP:
------------------
Set your xAI API key as an environment variable:
    export XAI_API_KEY="your-api-key-here"

Or create a .env file:
    XAI_API_KEY=your-api-key-here
    XAI_MODEL=grok-2-1212  # Optional default model

QUICK START:
------------
>>> from xai_chat import XAIChatProvider
>>>
>>> # Initialize provider
>>> provider = XAIChatProvider()
>>>
>>> # Simple chat
>>> response = provider.chat("What is machine learning?")
>>> print(response)
>>>
>>> # Chat with history
>>> history = [
...     {"role": "user", "content": "My name is Alice"},
...     {"role": "assistant", "content": "Nice to meet you, Alice!"}
... ]
>>> response = provider.chat("What's my name?", history=history)

AVAILABLE MODELS:
-----------------
Chat Models:
  - grok-2-1212 (default): Latest Grok 2, 128K context, function calling, search
  - grok-2-012: January 2025 version, 128K context
  - grok-beta: Beta with experimental features
  - grok-3: Advanced reasoning, 256K context

To list models:
>>> models = provider.list_models()
>>> for name, info in models.items():
...     print(f"{name}: {info['description']}")

BASIC USAGE EXAMPLES:
---------------------

1. Simple Chat Completion:
    >>> provider = XAIChatProvider()
    >>> response = provider.chat("Explain quantum computing in simple terms.")
    >>> print(response)

2. Chat with System Message:
    >>> response = provider.chat(
    ...     message="Translate 'hello' to French",
    ...     system="You are a professional translator."
    ... )

3. Complete (Simple Interface):
    >>> response = provider.complete("The meaning of life is", max_tokens=50)

4. Multi-turn Conversation:
    >>> history = [
    ...     {"role": "user", "content": "I'm learning Python"},
    ...     {"role": "assistant", "content": "That's great! How can I help?"}
    ... ]
    >>> response = provider.chat(
    ...     message="Explain list comprehensions",
    ...     history=history,
    ...     system="You are a Python tutor"
    ... )

5. Temperature and Parameters:
    >>> # More creative (higher temperature)
    >>> creative = provider.chat(
    ...     message="Write a short poem about AI",
    ...     temperature=1.5
    ... )
    >>>
    >>> # More focused (lower temperature)
    >>> factual = provider.chat(
    ...     message="What is the capital of Japan?",
    ...     temperature=0.1
    ... )

STREAMING RESPONSES:
--------------------
>>> for chunk in provider.stream_chat("Tell me a story about a robot"):
...     print(chunk, end='', flush=True)

STRUCTURED JSON OUTPUT:
-----------------------

1. Free-form JSON:
    >>> response = provider.chat_json(
    ...     message="List 3 programming languages with descriptions",
    ...     system="Return as JSON with languages array"
    ... )
    >>> print(response)  # Returns parsed dict

2. JSON with Schema:
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
    ...     },
    ...     "required": ["languages"]
    ... }
    >>> response = provider.chat_json(
    ...     message="List Python, Java, JavaScript with creation years",
    ...     schema=schema
    ... )

FUNCTION/TOOL CALLING:
----------------------
>>> tools = [
...     {
...         "type": "function",
...         "function": {
...             "name": "get_weather",
...             "description": "Get current weather for a location",
...             "parameters": {
...                 "type": "object",
...                 "properties": {
...                     "location": {
...                         "type": "string",
...                         "description": "City name"
...                     },
...                     "unit": {
...                         "type": "string",
...                         "enum": ["celsius", "fahrenheit"]
...                     }
...                 },
...                 "required": ["location"]
...             }
...         }
...     }
... ]
>>>
>>> response = provider.chat_with_tools(
...     message="What's the weather in Paris?",
...     tools=tools
... )
>>>
>>> if "tool_calls" in response:
...     for call in response["tool_calls"]:
...         print(f"Function: {call['function']['name']}")
...         print(f"Arguments: {call['function']['arguments']}")

LIVE WEB SEARCH:
----------------
>>> # Get current information using web search
>>> response = provider.chat_with_search(
...     message="What are the latest developments in quantum computing?",
...     max_results=10
... )
>>>
>>> # Search with date range
>>> response = provider.chat_with_search(
...     message="Recent AI breakthroughs",
...     from_date="2025-01-01",
...     to_date="2025-12-31",
...     max_results=5
... )

ADVANCED PARAMETERS:
--------------------
>>> response = provider.chat(
...     message="Write a creative story",
...     model="grok-2-1212",
...     system="You are a creative writer",
...     temperature=1.2,           # Creativity (0-2)
...     max_tokens=500,            # Response length
...     top_p=0.9,                 # Nucleus sampling
...     frequency_penalty=0.5,     # Reduce repetition
...     presence_penalty=0.3,      # Topic diversity
...     stop=["THE END"]           # Stop sequences
... )

GET FULL API RESPONSE:
----------------------
>>> messages = [
...     {"role": "user", "content": "Hello!"}
... ]
>>> full_response = provider.get_full_response(messages)
>>> print(f"Model used: {full_response['model']}")
>>> print(f"Tokens used: {full_response['usage']['total_tokens']}")
>>> print(f"Content: {full_response['choices'][0]['message']['content']}")

MODEL INFORMATION:
------------------
>>> # Get info about a specific model
>>> info = provider.get_model_info("grok-2-1212")
>>> print(f"Context length: {info['context_length']}")
>>> print(f"Capabilities: {info['capabilities']}")
>>> print(f"Supported params: {info['supported_params']}")
>>>
>>> # List all available models
>>> models = provider.get_available_models()
>>> print(models)  # ['grok-2-1212', 'grok-2-012', 'grok-beta', 'grok-3']

ERROR HANDLING:
---------------
>>> import requests
>>>
>>> try:
...     response = provider.chat("Hello")
... except requests.HTTPError as e:
...     print(f"API error: {e}")
...     print(f"Status code: {e.response.status_code}")
...     print(f"Response: {e.response.text}")
... except ValueError as e:
...     print(f"Configuration error: {e}")

SWEDISH DOCUMENT PROCESSING:
-----------------------------
For DocFlow integration, use low temperature for accuracy:

>>> # Extract structured data from Swedish municipal document
>>> provider = XAIChatProvider()
>>>
>>> schema = {
...     "type": "object",
...     "properties": {
...         "document_type": {"type": "string"},
...         "date": {"type": "string"},
...         "municipality": {"type": "string"},
...         "key_points": {
...             "type": "array",
...             "items": {"type": "string"}
...         }
...     }
... }
>>>
>>> response = provider.chat_json(
...     message=f"Extract information from this Swedish document: {text}",
...     schema=schema,
...     system="You are a Swedish document analyst. Extract key information.",
...     temperature=0.1  # Low temperature for accuracy
... )

RATE LIMITS & BEST PRACTICES:
------------------------------
1. Use appropriate temperature:
   - 0.0-0.3: Factual, deterministic tasks
   - 0.7-1.0: Balanced creativity
   - 1.0-2.0: Creative writing

2. Streaming for long responses:
   - Better UX for users
   - Lower perceived latency

3. Use search sparingly:
   - Only when current information needed
   - Search increases latency

4. Implement retry logic:
   - Handle rate limits (429)
   - Exponential backoff

5. Monitor token usage:
   - Use get_full_response() for metrics
   - Track costs via usage data

MODULE STRUCTURE:
-----------------
xai_chat/
├── __init__.py          # This file (exports and docs)
├── provider.py          # XAIChatProvider class
├── model_config.py      # Model definitions and configurations
└── requirements.txt     # Dependencies

EXPORTS:
--------
- XAIChatProvider: Main provider class
- CHAT_MODELS: Available chat model configurations
- get_model_config: Get config for a specific model
- get_default_chat_model: Get default model ID
- list_chat_models: List available model IDs

API REFERENCE:
--------------
See provider.py for complete method documentation.

Key methods:
- chat(): Basic chat completion
- complete(): Simple completion interface
- stream_chat(): Streaming responses
- chat_json(): Structured JSON output
- chat_with_tools(): Function calling
- chat_with_search(): Live web search
- chat_with_messages(): Direct message list interface
- get_full_response(): Raw API response

NOTES:
------
- xAI API is OpenAI-compatible (same request/response format)
- All methods raise requests.HTTPError on API failures
- JSON parsing errors return dict with "error" key
- Stream methods are generators (use in for loops)
- API key can be passed or use environment variable
- Supports python-dotenv for .env files

VERSION: 1.0.0
LICENSE: Proprietary - DocFlow Document Processing Pipeline
"""

from .provider import XAIChatProvider
from .model_config import (
    CHAT_MODELS,
    ModelConfig,
    get_model_config,
    get_default_chat_model,
    list_chat_models,
    XAI_API_URL
)

__all__ = [
    'XAIChatProvider',
    'CHAT_MODELS',
    'ModelConfig',
    'get_model_config',
    'get_default_chat_model',
    'list_chat_models',
    'XAI_API_URL'
]

__version__ = '1.0.0'
__author__ = 'DocFlow Development Team'

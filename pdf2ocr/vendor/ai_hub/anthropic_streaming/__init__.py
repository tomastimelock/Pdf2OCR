# Filepath: code_migration/ai_providers/anthropic_streaming/__init__.py
# Description: Anthropic Streaming module - Real-time SSE streaming for Claude responses
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/Anthropic/streaming/

"""
Anthropic Streaming Provider
============================

Self-contained module for streaming Claude responses using server-sent events (SSE).
Enables real-time incremental response generation for better user experience.

## Features

- **Text Streaming**: Stream text responses in real-time chunks
- **Callback Support**: Execute callbacks on each chunk for live updates
- **Multi-turn Conversations**: Stream multi-message conversations
- **Extended Thinking**: Stream thinking process + final response
- **Tool Use Streaming**: Stream tool calls and responses
- **File Streaming**: Stream directly to files with progress tracking
- **Error Recovery**: Automatic retry with continuation support
- **Async Support**: Full async/await support for concurrent operations

## Quick Start

### Basic Text Streaming

```python
from anthropic_streaming import AnthropicStreamingProvider

# Initialize provider
provider = AnthropicStreamingProvider(api_key="your_api_key")

# Stream text chunks
for chunk in provider.stream_text("Tell me a story about AI"):
    print(chunk, end="", flush=True)
print()
```

### Stream with Callback

```python
def on_chunk(text):
    # Update UI, log, or process chunk
    print(f"[Received: {len(text)} chars]", end="")

# Stream with callback
full_text = provider.stream_text_full(
    "Write a poem about streaming",
    callback=on_chunk
)
```

### Multi-turn Conversation

```python
messages = [
    {"role": "user", "content": "What is streaming?"},
    {"role": "assistant", "content": "Streaming sends data incrementally..."},
    {"role": "user", "content": "Why is it useful?"}
]

for chunk in provider.stream_messages(messages):
    print(chunk, end="", flush=True)
```

### Extended Thinking

```python
# Stream thinking process + final response
for event in provider.stream_with_thinking(
    prompt="Solve this complex math problem: ...",
    budget_tokens=10000
):
    if event["type"] == "thinking":
        print(f"[Thinking: {event['content']}]", end="")
    else:
        print(event["content"], end="", flush=True)
```

### Tool Use Streaming

```python
tools = [{
    "name": "get_weather",
    "description": "Get weather for a location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string"}
        },
        "required": ["location"]
    }
}]

for event in provider.stream_with_tools(
    prompt="What's the weather in Stockholm?",
    tools=tools
):
    if event["type"] == "text_delta":
        print(event["text"], end="")
    elif event["type"] == "tool_complete":
        print(f"\\n[Tool: {event['tool']['name']}]")
        print(f"[Input: {event['tool']['input']}]")
```

### Stream to File

```python
# Stream response directly to file with progress
text = provider.stream_to_file(
    prompt="Write a comprehensive essay about streaming APIs",
    output_file="essay.txt",
    max_tokens=8000,
    show_progress=True
)
print(f"Wrote {len(text)} characters to essay.txt")
```

### Async Streaming

```python
import asyncio

async def stream_async():
    provider = AnthropicStreamingProvider()

    async for chunk in provider.stream_text_async(
        "Explain async streaming"
    ):
        print(chunk, end="", flush=True)
    print()

asyncio.run(stream_async())
```

### Error Recovery

```python
# Automatic retry with continuation on connection errors
text = provider.stream_with_recovery(
    prompt="Write a long story",
    max_tokens=8000,
    max_retries=3
)
print(f"Generated {len(text)} characters (with recovery)")
```

## Advanced Usage

### Custom Model Selection

```python
from anthropic_streaming.model_config import (
    get_default_model,
    get_fastest_model,
    get_smartest_model
)

# Use fastest model (Haiku)
provider = AnthropicStreamingProvider(model=get_fastest_model())

# Or specify directly
provider = AnthropicStreamingProvider(
    model="claude-opus-4-5-20250918"
)
```

### Stream Events

When streaming with tools or thinking, events have the following structure:

**Text Events:**
- `text_start`: Text block begins
- `text_delta`: Text chunk received (contains "text" field)
- `text_complete`: Text block finished

**Thinking Events:**
- `thinking`: Thinking chunk (contains "type" and "content")
- `text`: Final response chunk (contains "type" and "content")

**Tool Events:**
- `tool_start`: Tool use begins (contains "tool" with id, name)
- `tool_input_delta`: Tool input JSON chunk (contains "partial")
- `tool_complete`: Tool input complete (contains "tool" with input)

**Message Events:**
- `message_delta`: Top-level message update (contains "stop_reason", "usage")

### Temperature Control

```python
# Precise extraction (lower temperature)
provider.stream_text(
    "Extract dates from: Meeting on 2024-01-15",
    temperature=0.1
)

# Creative generation (higher temperature)
provider.stream_text(
    "Write a creative story",
    temperature=0.9
)
```

### System Prompts

```python
provider.stream_text(
    prompt="Translate to Swedish: Hello world",
    system="You are a professional Swedish translator. Provide accurate, natural translations."
)
```

## Supported Models

### Claude 4.5 (Recommended)
- `claude-sonnet-4-5-20250929` (Default) - Best for complex agents, coding
- `claude-opus-4-5-20250918` - Maximum intelligence
- `claude-haiku-4-5-20251001` - Fastest, most economical

### Claude 4.x
- `claude-opus-4-1-20250805` - Previous gen Opus
- `claude-sonnet-4-20250514` - Previous gen Sonnet

### Claude 3.x (Legacy)
- `claude-3-7-sonnet-20250219` (Full thinking output)
- `claude-haiku-3-5-20241022`

## Configuration

All models support:
- **Context**: 200K tokens (1M beta for some models)
- **Max Output**: 4K-64K tokens depending on model
- **Streaming**: All models support real-time streaming
- **Vision**: All models support image input
- **Tools**: All models support tool use
- **Thinking**: Claude 4+ and 3.7 Sonnet

## Best Practices

1. **Always handle stream errors** - Network connections can drop
2. **Use callbacks for UI updates** - Don't block the main thread
3. **Buffer tool JSON** - Parse only after content_block_stop
4. **Implement reconnection** - Use stream_with_recovery for production
5. **Choose appropriate model** - Haiku for speed, Sonnet for balance, Opus for complexity
6. **Set reasonable max_tokens** - Avoid unnecessary costs
7. **Use system prompts** - Guide model behavior consistently

## Error Handling

```python
try:
    for chunk in provider.stream_text("Your prompt"):
        print(chunk, end="")
except anthropic.APIConnectionError:
    print("Network error - connection dropped")
except anthropic.RateLimitError:
    print("Rate limit exceeded - wait before retrying")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Environment Variables

```bash
# Required
export ANTHROPIC_API_KEY="your_api_key_here"

# Optional (overrides in code take precedence)
export ANTHROPIC_MODEL="claude-sonnet-4-5-20250929"
```

## Module Structure

```
anthropic_streaming/
├── __init__.py          # This file - exports and documentation
├── provider.py          # AnthropicStreamingProvider class
├── model_config.py      # Model configurations and helpers
└── requirements.txt     # Dependencies
```

## Dependencies

- anthropic >= 0.25.0 (official Anthropic SDK)
- Python >= 3.8

## Installation

```bash
pip install -r requirements.txt
```

## API Reference

See individual module docstrings for detailed API documentation:
- `provider.py` - Main streaming provider class
- `model_config.py` - Model configurations and helper functions

## Notes

- This module is **self-contained** and copy-paste ready
- All models support streaming - no special configuration needed
- First request with a new schema may have compilation latency
- Prompt caching available for repeated content (5min cache, 90% discount)
- Batch API available for large-scale processing (50% discount)

## Example: Complete Streaming App

```python
import sys
from anthropic_streaming import AnthropicStreamingProvider

def main():
    provider = AnthropicStreamingProvider()

    print("Claude Streaming Chat (type 'quit' to exit)")
    print("-" * 50)

    messages = []

    while True:
        user_input = input("\\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            break

        messages.append({"role": "user", "content": user_input})

        print("Claude: ", end="", flush=True)
        full_response = ""

        try:
            for chunk in provider.stream_messages(messages):
                print(chunk, end="", flush=True)
                full_response += chunk
            print()

            messages.append({"role": "assistant", "content": full_response})

        except KeyboardInterrupt:
            print("\\n[Interrupted]")
            break
        except Exception as e:
            print(f"\\nError: {e}")
            messages.pop()  # Remove failed user message

if __name__ == "__main__":
    main()
```

---

For DocFlow integration, this module provides real-time streaming for:
- Document classification feedback
- Extraction progress updates
- AI-powered analysis with live results
- Multi-step processing with status updates
"""

from .provider import (
    AnthropicStreamingProvider,
    stream_response
)

from .model_config import (
    # Model configurations
    CLAUDE_45_MODELS,
    CLAUDE_4_MODELS,
    CLAUDE_3_MODELS,

    # Configuration dictionaries
    THINKING_CONFIG,
    VISION_CONFIG,
    TOOL_CONFIG,
    STRUCTURED_OUTPUT_CONFIG,
    BATCH_CONFIG,
    PROMPT_CACHING_CONFIG,

    # Helper functions
    get_all_models,
    get_model_config,
    get_models_by_category,
    get_current_models,
    get_deprecated_models,
    get_default_model,
    get_fastest_model,
    get_smartest_model,
    supports_thinking,
    supports_vision,
    supports_tools,
    get_model_context_window,
    get_model_max_output,
    get_valid_params,
    get_param_options,
    build_payload_template,
    calculate_image_tokens,
)

__version__ = "1.0.0"
__author__ = "DocFlow Team"
__all__ = [
    # Main provider
    "AnthropicStreamingProvider",
    "stream_response",

    # Model configurations
    "CLAUDE_45_MODELS",
    "CLAUDE_4_MODELS",
    "CLAUDE_3_MODELS",
    "THINKING_CONFIG",
    "VISION_CONFIG",
    "TOOL_CONFIG",
    "STRUCTURED_OUTPUT_CONFIG",
    "BATCH_CONFIG",
    "PROMPT_CACHING_CONFIG",

    # Helper functions
    "get_all_models",
    "get_model_config",
    "get_models_by_category",
    "get_current_models",
    "get_deprecated_models",
    "get_default_model",
    "get_fastest_model",
    "get_smartest_model",
    "supports_thinking",
    "supports_vision",
    "supports_tools",
    "get_model_context_window",
    "get_model_max_output",
    "get_valid_params",
    "get_param_options",
    "build_payload_template",
    "calculate_image_tokens",
]

# Anthropic Text Provider

Standalone module for text generation using Claude models via the Anthropic Messages API.

## Features

- **Simple text generation** with Claude models
- **Multi-turn conversations** with message history
- **Response prefilling** for guided output formats
- **Streaming support** for real-time responses
- **Command execution DSL** for workflows
- **Full model coverage**: Claude 4.5, 4.x, and 3.x models

## Installation

```bash
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file or set environment variables:

```bash
ANTHROPIC_API_KEY=your-api-key-here
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929  # Optional, defaults to Sonnet 4.5
```

## Quick Start

```python
from anthropic_text import AnthropicTextProvider

# Initialize provider
provider = AnthropicTextProvider()

# Generate text
response = provider.generate(
    prompt="Write a haiku about programming",
    system="You are a poetic assistant"
)
print(response)
```

## Usage Examples

### Basic Generation

```python
provider = AnthropicTextProvider(api_key="your-key")

response = provider.generate(
    prompt="Explain quantum computing in simple terms",
    max_tokens=500,
    temperature=0.7
)
```

### Multi-turn Conversations

```python
response = provider.generate_with_messages([
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."},
    {"role": "user", "content": "What can I build with it?"}
])
```

### Response Prefilling

```python
response = provider.generate_with_prefill(
    prompt="List 3 cities as JSON",
    prefill='{"cities": ['
)
# Output: {"cities": ["New York", "Tokyo", "London"]}
```

### Streaming

```python
for chunk in provider.stream_generate(
    prompt="Tell me a story",
    system="You are a storyteller"
):
    print(chunk, end="", flush=True)
```

### System Prompts

```python
response = provider.generate(
    prompt="How do I bake bread?",
    system="You are a master baker. Provide detailed instructions."
)
```

## Model Selection

### Available Models

```python
# List all models
models = AnthropicTextProvider.list_models()
for name, info in models.items():
    print(f"{name}: {info['description']}")

# Get model info
info = AnthropicTextProvider.get_model_info("claude-sonnet-4-5-20250929")
```

### Model Helpers

```python
from anthropic_text import (
    get_default_model,     # claude-sonnet-4-5-20250929
    get_fastest_model,     # claude-haiku-4-5-20251001
    get_smartest_model,    # claude-opus-4-5-20250918
)

# Use specific model
provider = AnthropicTextProvider(model=get_smartest_model())
```

### Recommended Models

| Model | Best For | Context | Max Output |
|-------|----------|---------|------------|
| `claude-sonnet-4-5-20250929` | General use, coding, agents | 200K | 64K |
| `claude-opus-4-5-20250918` | Complex reasoning | 200K | 64K |
| `claude-haiku-4-5-20251001` | Speed, high-volume | 200K | 64K |

## Parameters

### Core Parameters

- `prompt` (str): The input text prompt
- `model` (str, optional): Model to use (defaults to instance model)
- `system` (str, optional): System prompt for context
- `max_tokens` (int): Maximum tokens to generate (default: 4096)
- `temperature` (float): Randomness 0.0-1.0 (default: 1.0)
- `top_p` (float): Nucleus sampling (mutually exclusive with temperature)
- `top_k` (int): Top-k sampling
- `stop_sequences` (list): Stop generation at these strings

### Temperature vs Top_p

**Note**: `temperature` and `top_p` are mutually exclusive.

```python
# Use temperature for randomness control
response = provider.generate(prompt="...", temperature=0.7)

# OR use top_p for nucleus sampling
response = provider.generate(prompt="...", top_p=0.9)
```

## Command Execution DSL

Execute workflows with a simple command language:

```python
commands = """
TEXT_GENERATE prompt="Write a poem" -> poem
TEXT_GENERATE prompt="Analyze this: ${poem}" system="You are a critic"
"""

result = provider.execute_commands(commands)
print(result['last_output'])
```

### Available Commands

- `TEXT_GENERATE prompt="..." [output="file.txt"] [model="..."]`
- `TEXT_MESSAGES messages="..." [output="file.txt"]`
- `TEXT_STREAM prompt="..."`
- `TEXT_PREFILL prompt="..." prefill="..."`
- `TEXT_MODELS` - List available models
- `TEXT_HELP` - Get help text
- `SET var="value"` - Set variable
- `WAIT seconds=N` - Wait N seconds
- `PRINT message="..."` - Print message
- `SAVE content="..." file="..."` - Save to file

### Variable Substitution

```python
commands = """
SET topic="artificial intelligence"
TEXT_GENERATE prompt="Explain ${topic}" -> explanation
TEXT_GENERATE prompt="Summarize: ${explanation}"
"""
```

## Advanced Usage

### Full Response Object

```python
response = provider.get_full_response(prompt="Hello!")

# Access metadata
print(response.stop_reason)  # end_turn, max_tokens, stop_sequence
print(response.usage)        # {input_tokens: ..., output_tokens: ...}
print(response.model)        # Model used
```

### Error Handling

```python
try:
    response = provider.generate(
        prompt="Test",
        max_tokens=100000  # Exceeds limit
    )
except Exception as e:
    print(f"Error: {e}")
```

## Model Capabilities

### Check Support

```python
from anthropic_text import (
    supports_thinking,
    supports_vision,
    supports_tools
)

model = "claude-sonnet-4-5-20250929"
print(f"Supports thinking: {supports_thinking(model)}")
print(f"Supports vision: {supports_vision(model)}")
print(f"Supports tools: {supports_tools(model)}")
```

### Context Windows

```python
from anthropic_text import get_model_context_window, get_model_max_output

model = "claude-opus-4-5-20250918"
print(f"Context: {get_model_context_window(model):,} tokens")
print(f"Max output: {get_model_max_output(model):,} tokens")
```

## Module Structure

```
anthropic_text/
├── __init__.py          # Exports and module docstring
├── provider.py          # AnthropicTextProvider class
├── model_config.py      # Model configurations and helpers
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Self-Contained Design

This module is completely self-contained:
- Uses relative imports (`.model_config`)
- No external AIMOS dependencies
- Copy-pasteable to any project
- Minimal dependencies (anthropic, python-dotenv)

## API Reference

### AnthropicTextProvider

#### Methods

- `__init__(api_key=None, model=None)` - Initialize provider
- `generate(prompt, **params)` - Generate text
- `generate_with_messages(messages, **params)` - Multi-turn conversation
- `generate_with_prefill(prompt, prefill, **params)` - Guided output
- `stream_generate(prompt, **params)` - Streaming generation
- `get_full_response(prompt, **params)` - Full response object
- `execute_commands(command_string, verbose=True)` - Execute DSL commands

#### Static Methods

- `list_models()` - List all models
- `get_help()` - Get help text
- `get_model_info(model_name)` - Get model details

### Helper Functions

```python
from anthropic_text import (
    get_all_models,            # Dict of all models
    get_model_config,          # Get ModelConfig object
    get_default_model,         # Default model name
    get_fastest_model,         # Fastest model name
    get_smartest_model,        # Most capable model
    supports_thinking,         # Check extended thinking support
    supports_vision,           # Check vision support
    supports_tools,            # Check tool use support
    get_model_context_window,  # Get context window size
    get_model_max_output,      # Get max output tokens
)
```

## Notes

- **Temperature = 0.0** for deterministic, factual outputs
- **Temperature = 1.0** for creative, diverse outputs
- **Max tokens** includes both prompt and response
- **Stop sequences** are case-sensitive
- **Streaming** yields chunks as they arrive (lower latency)

## License

Based on reference code from AIMOS framework.
Adapted for standalone use in DocumentHandler project.

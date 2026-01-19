# Anthropic Text Provider - Quick Reference

## Installation

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
```

## Basic Usage

```python
from anthropic_text import AnthropicTextProvider

provider = AnthropicTextProvider()
response = provider.generate(prompt="Hello!")
```

## Common Patterns

### 1. Simple Generation
```python
response = provider.generate(
    prompt="Explain quantum computing",
    max_tokens=500,
    temperature=0.7
)
```

### 2. With System Prompt
```python
response = provider.generate(
    prompt="How do I bake bread?",
    system="You are a master baker"
)
```

### 3. Conversation
```python
response = provider.generate_with_messages([
    {"role": "user", "content": "Hi!"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "How are you?"}
])
```

### 4. Structured Output (Prefilling)
```python
response = provider.generate_with_prefill(
    prompt="List 3 cities",
    prefill='{"cities": ['
)
# Returns: {"cities": ["New York", "Paris", "Tokyo"]}
```

### 5. Streaming
```python
for chunk in provider.stream_generate(prompt="Tell a story"):
    print(chunk, end="", flush=True)
```

## Model Selection

```python
from anthropic_text import get_default_model, get_fastest_model, get_smartest_model

# Use fastest model
provider = AnthropicTextProvider(model=get_fastest_model())

# Use most capable model
provider = AnthropicTextProvider(model=get_smartest_model())
```

## Temperature Guide

| Temperature | Use Case | Example |
|-------------|----------|---------|
| 0.0 | Deterministic, factual | Data extraction, classification |
| 0.3 | Mostly consistent | Summarization, Q&A |
| 0.7 | Balanced | General conversation |
| 1.0 | Creative | Story writing, brainstorming |

## Parameters

```python
provider.generate(
    prompt="...",              # Required: Input text
    model="...",               # Optional: Model name
    system="...",              # Optional: System prompt
    max_tokens=4096,           # Optional: Max output tokens
    temperature=1.0,           # Optional: 0.0-1.0
    top_p=None,               # Optional: Nucleus sampling
    top_k=None,               # Optional: Top-k sampling
    stop_sequences=None       # Optional: Stop strings
)
```

## Command DSL

```python
commands = """
TEXT_GENERATE prompt="Write a poem" -> poem
TEXT_GENERATE prompt="Analyze: ${poem}" system="You are a critic"
"""
result = provider.execute_commands(commands)
```

## Error Handling

```python
try:
    response = provider.generate(prompt="Test")
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"API error: {e}")
```

## Model Information

```python
# List all models
models = AnthropicTextProvider.list_models()

# Get specific model info
info = AnthropicTextProvider.get_model_info("claude-sonnet-4-5-20250929")
print(info['context_window'])  # 200000
print(info['max_output'])       # 64000
```

## Helper Functions

```python
from anthropic_text import (
    supports_thinking,         # Check extended thinking
    supports_vision,           # Check vision support
    supports_tools,            # Check tool use
    get_model_context_window,  # Get context size
    get_model_max_output,      # Get max output
)

if supports_thinking("claude-sonnet-4-5-20250929"):
    print("Model supports extended thinking!")
```

## Recommended Models

| Model | Speed | Intelligence | Cost | Use Case |
|-------|-------|-------------|------|----------|
| `claude-haiku-4-5-20251001` | ⚡⚡⚡ | ⭐⭐⭐ | $ | High-volume, simple tasks |
| `claude-sonnet-4-5-20250929` | ⚡⚡ | ⭐⭐⭐⭐ | $$ | **Default - Most use cases** |
| `claude-opus-4-5-20250918` | ⚡ | ⭐⭐⭐⭐⭐ | $$$ | Complex reasoning |

## Swedish Language Example

```python
response = provider.generate(
    prompt="Sammanfatta denna svenska text: ...",
    system="Du är en svensk dokumentanalysator",
    temperature=0.1
)
```

## File Structure

```
anthropic_text/
├── __init__.py           # Import: from anthropic_text import ...
├── provider.py           # Core provider class
├── model_config.py       # Model configurations
├── requirements.txt      # Dependencies
├── README.md            # Full documentation
├── example.py           # Usage examples
└── QUICK_REFERENCE.md   # This file
```

## Quick Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'anthropic'`
**Solution**: `pip install anthropic`

**Issue**: `ValueError: Anthropic API key must be provided`
**Solution**: Set `ANTHROPIC_API_KEY` environment variable

**Issue**: Import error
**Solution**: Ensure you're importing from `anthropic_text`, not `AIMOS`

## API Key Setup

### Option 1: Environment Variable
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Option 2: .env File
```
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929
```

### Option 3: Direct Pass
```python
provider = AnthropicTextProvider(api_key="sk-ant-...")
```

## Performance Tips

1. **Use Haiku for bulk**: High-volume simple tasks
2. **Cache system prompts**: Reuse same system prompt for 90% discount
3. **Batch processing**: Group similar requests
4. **Stream for UX**: Use streaming for long responses
5. **Tune temperature**: Lower = faster, higher = more diverse

## Common Use Cases

### Document Classification
```python
response = provider.generate(
    prompt=f"Classify this document: {text}",
    system="Return only: 'legal', 'medical', or 'administrative'",
    temperature=0.0,
    max_tokens=10
)
```

### Data Extraction
```python
response = provider.generate_with_prefill(
    prompt=f"Extract data: {document}",
    prefill='{"name": "',
    temperature=0.1
)
```

### Summarization
```python
response = provider.generate(
    prompt=f"Summarize in 3 sentences: {text}",
    temperature=0.3,
    max_tokens=200
)
```

## Links

- [Full Documentation](README.md)
- [Migration Notes](MIGRATION_NOTES.md)
- [Examples](example.py)
- [Anthropic API Docs](https://docs.anthropic.com)

# Mistral Chat Provider

A complete, self-contained Python module for Mistral AI chat completions. Production-ready with comprehensive documentation, type hints, and examples.

## Features

- ✅ **Simple Chat Completion** - Basic question/answer
- ✅ **Multi-turn Conversations** - Maintain conversation history
- ✅ **Streaming Responses** - Real-time token streaming
- ✅ **JSON Mode** - Structured outputs with schema validation
- ✅ **Function Calling** - Tool use and function execution
- ✅ **Multiple Models** - Large, small, medium, code, vision
- ✅ **Temperature Control** - Adjust creativity/determinism
- ✅ **System Messages** - Control assistant behavior
- ✅ **Type Hints** - Full type annotation
- ✅ **Self-Contained** - No external dependencies (except mistralai SDK)

## Installation

```bash
# Install from requirements file
pip install -r requirements.txt

# Or install directly
pip install mistralai>=1.0.0
```

## Quick Start

```python
from mistral_chat import MistralChatProvider

# Initialize (uses MISTRAL_API_KEY env var)
provider = MistralChatProvider()

# Simple chat
response = provider.chat("What is the capital of France?")
print(response)
# Output: "The capital of France is Paris."
```

## Environment Setup

Set your API key:

```bash
# Linux/Mac
export MISTRAL_API_KEY='your-api-key'

# Windows CMD
set MISTRAL_API_KEY=your-api-key

# Windows PowerShell
$env:MISTRAL_API_KEY='your-api-key'
```

Or use a `.env` file:
```
MISTRAL_API_KEY=your-api-key
MISTRAL_MODEL=mistral-small-latest
```

## Usage Examples

### 1. Basic Chat

```python
provider = MistralChatProvider()

response = provider.chat("Explain quantum computing in simple terms")
print(response)
```

### 2. Chat with System Message

```python
response = provider.chat(
    message="Explain Python decorators",
    system="You are a Python expert. Be concise and use examples."
)
```

### 3. Multi-turn Conversation

```python
provider = MistralChatProvider()
history = []

# First turn
msg1 = "Hi, I'm learning Python"
resp1 = provider.chat(msg1, history=history)
history.extend([
    {"role": "user", "content": msg1},
    {"role": "assistant", "content": resp1}
])

# Second turn (with context)
msg2 = "What should I learn first?"
resp2 = provider.chat(msg2, history=history)
print(resp2)
```

### 4. Streaming Response

```python
print("Response: ", end="")
for chunk in provider.stream_chat("Write a haiku about programming"):
    print(chunk, end="", flush=True)
print()
```

### 5. JSON Mode

```python
# Simple JSON mode
result = provider.chat_with_json(
    message="List 3 French cities with populations",
    system="Return as JSON with cities array"
)
print(result)
# Output: {'cities': [{'name': 'Paris', 'population': 2165000}, ...]}
```

### 6. JSON with Schema Validation

```python
schema = {
    "type": "object",
    "properties": {
        "cities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "population": {"type": "integer"}
                },
                "required": ["name", "population"]
            }
        }
    },
    "required": ["cities"]
}

result = provider.chat_with_json(
    message="List 3 French cities with populations",
    json_schema=schema
)
```

### 7. Function Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        }
    }
}]

response = provider.chat_with_tools(
    message="What's the weather in Paris?",
    tools=tools
)

if "tool_calls" in response:
    for call in response["tool_calls"]:
        func_name = call["function"]["name"]
        func_args = call["function"]["arguments"]
        print(f"Call: {func_name}({func_args})")
```

### 8. Temperature Control

```python
# Deterministic/factual (low temperature)
response = provider.chat(
    "What is the capital of France?",
    temperature=0.1
)

# Creative (high temperature)
story = provider.chat(
    "Write a creative short story",
    temperature=0.9
)
```

### 9. Different Models

```python
# Use large model for complex reasoning
response = provider.chat(
    "Analyze this complex philosophical question...",
    model="mistral-large-latest"
)

# Use code model for programming
code = provider.chat(
    "Write a Python function to merge two sorted lists",
    model="codestral-latest"
)

# Use vision model for images
description = provider.chat(
    "Describe this image in detail",
    model="pixtral-12b-latest"
)
```

### 10. List Available Models

```python
models = MistralChatProvider.list_models()
for name, info in models.items():
    print(f"{name}:")
    print(f"  Description: {info['description']}")
    print(f"  Context: {info['context_length']:,} tokens")
    print()
```

## Available Models

| Model | Context | Best For | Capabilities |
|-------|---------|----------|--------------|
| `mistral-large-latest` | 128K | Complex reasoning, analysis | Function calling, JSON, streaming |
| `mistral-small-latest` | 32K | General tasks (default) | Function calling, JSON, streaming |
| `mistral-medium-latest` | 32K | Balanced performance | Function calling, JSON, streaming |
| `ministral-3b-latest` | 32K | Simple tasks, low latency | Basic chat, streaming |
| `ministral-8b-latest` | 32K | Moderate complexity | Function calling, JSON, streaming |
| `pixtral-12b-latest` | 32K | Image understanding | Vision, streaming |
| `codestral-latest` | 32K | Code generation | Code completion, streaming |

## API Reference

### MistralChatProvider

Main provider class for chat operations.

#### Constructor

```python
MistralChatProvider(api_key: str = None, model: str = None)
```

**Parameters:**
- `api_key` (str, optional): Mistral API key. Uses `MISTRAL_API_KEY` env var if not provided.
- `model` (str, optional): Default model. Defaults to `mistral-small-latest`.

#### Methods

##### chat()

```python
chat(
    message: str,
    model: str = None,
    system: str = None,
    history: List[Dict[str, str]] = None,
    temperature: float = None,
    max_tokens: int = None,
    top_p: float = None,
    safe_prompt: bool = False,
    random_seed: int = None,
    **kwargs
) -> str
```

Generate a chat completion.

**Parameters:**
- `message`: User message content (required)
- `model`: Model to use (optional)
- `system`: System message for instructions (optional)
- `history`: Previous conversation messages (optional)
- `temperature`: Sampling temperature 0.0-1.0 (optional)
- `max_tokens`: Maximum response length (optional)
- `top_p`: Nucleus sampling parameter (optional)
- `safe_prompt`: Enable safety filtering (optional)
- `random_seed`: Seed for deterministic results (optional)

**Returns:** Response text string

##### complete()

```python
complete(
    prompt: str,
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs
) -> str
```

Simple completion interface (single message).

##### stream_chat()

```python
stream_chat(
    message: str,
    model: str = None,
    system: str = None,
    history: List[Dict[str, str]] = None,
    **kwargs
) -> Generator[str, None, None]
```

Stream chat completion response in chunks.

**Yields:** Response text chunks

##### chat_with_json()

```python
chat_with_json(
    message: str,
    json_schema: Dict[str, Any] = None,
    model: str = None,
    system: str = None,
    **kwargs
) -> Dict[str, Any]
```

Generate JSON-formatted response with optional schema validation.

**Returns:** Parsed JSON dictionary

##### chat_with_tools()

```python
chat_with_tools(
    message: str,
    tools: List[Dict[str, Any]],
    model: str = None,
    system: str = None,
    tool_choice: str = "auto",
    **kwargs
) -> Dict[str, Any]
```

Generate chat with function/tool calling support.

**Returns:** Response dict with content and potential tool calls

##### chat_with_messages()

```python
chat_with_messages(
    messages: List[Dict[str, str]],
    model: str = None,
    **kwargs
) -> str
```

Generate chat with explicit message list.

##### get_full_response()

```python
get_full_response(
    messages: List[Dict[str, str]],
    model: str = None,
    **kwargs
) -> Any
```

Get full API response object (for advanced use cases).

#### Static Methods

##### list_models()

```python
@staticmethod
list_models() -> Dict[str, Any]
```

List all available models with configurations.

##### get_model_info()

```python
@staticmethod
get_model_info(model_name: str) -> Optional[Dict[str, Any]]
```

Get detailed info about a specific model.

##### get_help()

```python
@staticmethod
get_help() -> str
```

Get help text for chat functionality.

## Integration Patterns

### Chatbot with Memory

```python
class Chatbot:
    def __init__(self):
        self.provider = MistralChatProvider()
        self.history = []
        self.system = "You are a helpful assistant"

    def chat(self, message):
        response = self.provider.chat(
            message=message,
            system=self.system,
            history=self.history
        )
        self.history.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": response}
        ])
        return response

    def clear_history(self):
        self.history = []

# Usage
bot = Chatbot()
print(bot.chat("Hi, I'm learning Python"))
print(bot.chat("What should I start with?"))
```

### Code Assistant

```python
class CodeAssistant:
    def __init__(self):
        self.provider = MistralChatProvider(model="codestral-latest")

    def generate_code(self, description, language="Python"):
        return self.provider.chat(
            message=f"Write {language} code: {description}",
            temperature=0.2,
            system=f"You are an expert {language} developer. "
                   f"Write clean, well-commented code."
        )

    def explain_code(self, code):
        return self.provider.chat(
            message=f"Explain this code:\n{code}",
            system="You are a code reviewer. Explain clearly."
        )

# Usage
assistant = CodeAssistant()
code = assistant.generate_code("function to calculate fibonacci sequence")
explanation = assistant.explain_code(code)
```

### Structured Data Extractor

```python
class DataExtractor:
    def __init__(self):
        self.provider = MistralChatProvider()

    def extract_entities(self, text):
        schema = {
            "type": "object",
            "properties": {
                "people": {"type": "array", "items": {"type": "string"}},
                "places": {"type": "array", "items": {"type": "string"}},
                "dates": {"type": "array", "items": {"type": "string"}}
            }
        }
        return self.provider.chat_with_json(
            message=f"Extract people, places, and dates from: {text}",
            json_schema=schema,
            temperature=0.1
        )

# Usage
extractor = DataExtractor()
text = "John visited Paris on January 15th and met Marie at the Eiffel Tower."
entities = extractor.extract_entities(text)
print(entities)
```

## Error Handling

```python
from mistral_chat import MistralChatProvider

try:
    provider = MistralChatProvider(api_key="your-api-key")
    response = provider.chat("Hello")
    print(response)

except ValueError as e:
    print(f"Configuration error: {e}")
    # Handle missing API key or invalid config

except ImportError as e:
    print(f"Import error: {e}")
    # Handle missing mistralai package

except Exception as e:
    print(f"API error: {e}")
    # Handle API errors (rate limits, invalid requests, etc.)
```

## Module Structure

```
mistral_chat/
├── __init__.py          # Module exports and documentation
├── provider.py          # MistralChatProvider class
├── model_config.py      # Model configurations and metadata
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Dependencies

- `mistralai>=1.0.0` (required) - Official Mistral AI Python SDK
- `python-dotenv` (optional) - For loading .env files
- `pydantic` (optional) - For data validation

## Best Practices

1. **Use appropriate models**:
   - `mistral-large` for complex reasoning
   - `mistral-small` for general tasks
   - `codestral` for code generation

2. **Control temperature**:
   - `0.0-0.3` for factual/deterministic outputs
   - `0.5-0.7` for balanced responses
   - `0.8-1.0` for creative outputs

3. **Manage context**:
   - Keep history concise (within model's context limit)
   - Use system messages for consistent behavior
   - Clear history when starting new topics

4. **Handle errors**:
   - Wrap API calls in try/except blocks
   - Check for tool_calls before using function calling results
   - Validate JSON schema responses

5. **Optimize costs**:
   - Use smaller models when possible
   - Set max_tokens appropriately
   - Cache common responses

## License

MIT License - Free for commercial and personal use

## Support

- **Mistral API Documentation**: https://docs.mistral.ai/
- **Python SDK**: https://github.com/mistralai/client-python
- **API Reference**: https://docs.mistral.ai/api/

## Changelog

### Version 1.0.0
- Initial release
- Support for all Mistral chat models
- Basic chat, streaming, JSON mode
- Function calling support
- Comprehensive documentation

# OpenAI Text Completion Provider

Self-contained, copy-paste ready module for OpenAI text completion using the Chat Completions API.

## Features

- ✅ Simple text completion from prompts
- ✅ Chat completions with messages
- ✅ System + user message pattern
- ✅ Streaming response generator
- ✅ Model information and recommendations
- ✅ Cost estimation
- ✅ Swedish language support
- ✅ Full type hints and comprehensive documentation
- ✅ Zero external dependencies beyond OpenAI SDK

## Installation

```bash
cd code_migration/ai_providers/openai_text
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in your project root or set environment variables:

```env
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4o  # Optional, defaults to gpt-4o
```

## Quick Start

### Basic Completion

```python
from code_migration.ai_providers.openai_text import OpenAITextProvider

# Initialize
provider = OpenAITextProvider()

# Simple completion
response = provider.complete("Explain machine learning in one sentence")
print(response)
```

### Chat with Messages

```python
response = provider.chat(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"}
    ],
    temperature=0.7,
    max_tokens=500
)
print(response)
```

### System + User Pattern

```python
response = provider.chat_with_system(
    system_prompt="You are a Swedish language expert.",
    user_message="Translate 'hello' to Swedish",
    temperature=0.3
)
print(response)
```

### Streaming Responses

```python
for chunk in provider.stream_chat(
    messages=[{"role": "user", "content": "Write a short story"}]
):
    print(chunk, end='', flush=True)
```

## DocFlow Integration Examples

### Swedish Document Extraction

```python
provider = OpenAITextProvider()

# Extract information from Swedish municipal document
system_prompt = """
Du är en AI-assistent som extraherar strukturerad information från
svenska myndighetsdokument. Svara alltid på svenska med JSON-format.
"""

user_message = f"""
Extrahera följande information från dokumentet:
- Dokumenttyp
- Datum (format: YYYY-MM-DD)
- Ansvarig myndighet
- Sammanfattning (max 200 ord)

Dokument:
{document_text}
"""

response = provider.chat_with_system(
    system_prompt=system_prompt,
    user_message=user_message,
    temperature=0.1,  # Low temperature for accuracy
    max_tokens=2000
)

# Parse JSON response
import json
extracted_data = json.loads(response)
```

### Document Classification

```python
# Classify Swedish legal documents
classification_prompt = """
Klassificera följande dokument i en av dessa kategorier:
- Årsredovisning
- Protokoll
- Detaljplan
- Lagtext
- Medicinsk journal
- Övrigt

Svara med endast kategorinamnet.
"""

response = provider.chat_with_system(
    system_prompt=classification_prompt,
    user_message=document_text[:2000],  # First 2000 chars
    temperature=0.0,  # Deterministic
    max_tokens=50
)

category = response.strip()
```

### Batch Processing with Streaming

```python
def process_documents_with_progress(documents: list):
    """Process multiple documents with real-time progress."""
    results = []

    for i, doc in enumerate(documents, 1):
        print(f"\nProcessing document {i}/{len(documents)}...")

        response = provider.chat_with_system(
            system_prompt="Du är en dokumentanalysator.",
            user_message=f"Sammanfatta: {doc}",
            temperature=0.3
        )

        results.append({
            'document_id': i,
            'summary': response
        })

    return results
```

## API Reference

### OpenAITextProvider

#### `__init__(api_key=None, model=None)`

Initialize the provider.

**Parameters:**
- `api_key` (str, optional): OpenAI API key. Defaults to `OPENAI_API_KEY` env var.
- `model` (str, optional): Default model. Defaults to `OPENAI_MODEL` env var or "gpt-4o".

**Raises:**
- `ValueError`: If API key is not provided or found in environment.

---

#### `chat(messages, model=None, temperature=0.7, max_tokens=None, **kwargs)`

Generate completion from messages.

**Parameters:**
- `messages` (List[Dict[str, str]]): List of message dicts with 'role' and 'content'
- `model` (str, optional): Model to use
- `temperature` (float): Sampling temperature 0.0-2.0 (default: 0.7)
- `max_tokens` (int, optional): Maximum tokens to generate
- `top_p` (float): Nucleus sampling 0.0-1.0 (default: 1.0)
- `frequency_penalty` (float): Penalty for frequency -2.0 to 2.0 (default: 0.0)
- `presence_penalty` (float): Penalty for presence -2.0 to 2.0 (default: 0.0)

**Returns:** str - Generated text

---

#### `complete(prompt, model=None, temperature=0.7, max_tokens=None, **kwargs)`

Simple text completion from prompt.

**Parameters:**
- `prompt` (str): Text prompt
- `model` (str, optional): Model to use
- `temperature` (float): Sampling temperature (default: 0.7)
- `max_tokens` (int, optional): Maximum tokens to generate

**Returns:** str - Generated text

---

#### `chat_with_system(system_prompt, user_message, model=None, temperature=0.7, max_tokens=None, **kwargs)`

Generate completion with system and user messages.

**Parameters:**
- `system_prompt` (str): System message defining AI role/behavior
- `user_message` (str): User's input message
- `model` (str, optional): Model to use
- `temperature` (float): Sampling temperature (default: 0.7)
- `max_tokens` (int, optional): Maximum tokens to generate

**Returns:** str - Generated text

---

#### `stream_chat(messages, model=None, temperature=0.7, max_tokens=None, **kwargs)`

Stream completion from messages.

**Parameters:** Same as `chat()`

**Yields:** str - Text chunks as they are generated

---

#### `get_full_response(messages, model=None, temperature=0.7, max_tokens=None, **kwargs)`

Get full response object (not just text).

**Returns:** ChatCompletion - Full OpenAI response object with metadata

---

#### `get_models()`

Get information about all available models.

**Returns:** Dict[str, Dict] - Model configurations

---

#### `get_model_info(model_name)`

Get detailed information about a specific model.

**Parameters:**
- `model_name` (str): Model name (e.g., "gpt-4o")

**Returns:** Dict or None - Model configuration

---

#### `recommend_model(task_type)`

Get recommended model for a task type.

**Parameters:**
- `task_type` (str): One of "extraction", "classification", "generation", "analysis", "translation", "summarization", "simple"

**Returns:** str - Recommended model name

---

#### `estimate_cost(messages, model=None, max_tokens=1000)`

Estimate cost for a completion (approximate).

**Returns:** Dict with estimated tokens and cost

## Available Models

| Model | Description | Best For | Cost* |
|-------|-------------|----------|-------|
| **gpt-4o** | Most capable, multimodal | Complex reasoning, Swedish, structured output | $2.50/$10 |
| **gpt-4o-mini** | Smaller, faster, cheaper | Most tasks, great cost/performance | $0.15/$0.60 |
| **gpt-4-turbo** | Large context window | Long documents, deep analysis | $10/$30 |
| **gpt-3.5-turbo** | Fast, cost-effective | Simple extraction, classification | $0.50/$1.50 |

*Cost per 1K tokens (input/output) in USD

## Parameter Presets

The module includes presets for common tasks:

```python
from code_migration.ai_providers.openai_text.model_config import get_preset

# Extraction preset
params = get_preset("extraction")  # temperature=0.1, max_tokens=2000

# Classification preset
params = get_preset("classification")  # temperature=0.0, max_tokens=100

# Generation preset
params = get_preset("generation")  # temperature=0.7, max_tokens=4000

# Analysis preset
params = get_preset("analysis")  # temperature=0.3, max_tokens=3000
```

## Best Practices

### For Extraction Tasks
```python
# Use low temperature for deterministic results
response = provider.chat_with_system(
    system_prompt="Extract data as JSON",
    user_message=document,
    temperature=0.1,  # Low for accuracy
    max_tokens=2000
)
```

### For Creative Tasks
```python
# Use higher temperature for creativity
response = provider.complete(
    prompt="Write a creative story",
    temperature=0.9,  # High for creativity
    max_tokens=4000
)
```

### For Cost Optimization
```python
# Use gpt-4o-mini for most tasks
provider = OpenAITextProvider(model="gpt-4o-mini")

# Estimate cost before processing
estimate = provider.estimate_cost(messages, max_tokens=500)
print(f"Estimated cost: ${estimate['estimated_cost_usd']:.4f}")

# Use gpt-3.5-turbo for simple tasks
response = provider.chat(messages, model="gpt-3.5-turbo")
```

### Error Handling
```python
from openai import OpenAIError

try:
    response = provider.complete("Test prompt")
except ValueError as e:
    print(f"Configuration error: {e}")
except OpenAIError as e:
    print(f"API error: {e}")
```

## Running Examples

```bash
# Run built-in examples
python provider.py

# Or import and use
python -c "from code_migration.ai_providers.openai_text import OpenAITextProvider; \
           p = OpenAITextProvider(); \
           print(p.complete('Hello world'))"
```

## Module Structure

```
openai_text/
├── __init__.py          # Module exports and comprehensive docs
├── provider.py          # OpenAITextProvider class
├── model_config.py      # Model configurations and helpers
├── requirements.txt     # Dependencies (openai, python-dotenv)
└── README.md           # This file
```

## Testing

Create a test file `test_provider.py`:

```python
from code_migration.ai_providers.openai_text import OpenAITextProvider

def test_basic_completion():
    provider = OpenAITextProvider()
    response = provider.complete("Say 'test successful'")
    assert len(response) > 0
    print(f"✓ Basic completion: {response}")

def test_swedish_support():
    provider = OpenAITextProvider()
    response = provider.chat_with_system(
        system_prompt="Du är en svensk assistent.",
        user_message="Säg 'Hej!'"
    )
    assert len(response) > 0
    print(f"✓ Swedish support: {response}")

if __name__ == "__main__":
    test_basic_completion()
    test_swedish_support()
    print("\nAll tests passed!")
```

## License

Part of DocumentHandler/DocFlow project.

## References

- Based on: `reference_codebase/AIMOS/providers/openai/text/`
- OpenAI API: https://platform.openai.com/docs/api-reference/chat
- Chat Completions Guide: https://platform.openai.com/docs/guides/text-generation

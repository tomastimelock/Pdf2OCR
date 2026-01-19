# Filepath: code_migration/ai_providers/openai_text/__init__.py
# Description: OpenAI Text completion module - exports and usage documentation
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/openai/text/

"""
OpenAI Text Completion Provider
================================

Self-contained module for OpenAI text completion using Chat Completions API.

Features:
---------
- Basic chat completion with messages
- Simple text completion with single prompt
- System + user message pattern
- Streaming response generator
- Model information and listing
- Support for GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo

Installation:
-------------
pip install -r requirements.txt

Configuration:
--------------
Set environment variables in .env file:
    OPENAI_API_KEY=your-api-key-here
    OPENAI_MODEL=gpt-4o  # Optional, defaults to gpt-4o

Basic Usage:
------------
```python
from code_migration.ai_providers.openai_text import OpenAITextProvider

# Initialize provider
provider = OpenAITextProvider()

# Simple completion
response = provider.complete("Explain machine learning in one sentence")
print(response)

# Chat with messages
response = provider.chat(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"}
    ],
    temperature=0.7
)
print(response)

# System + user pattern
response = provider.chat_with_system(
    system_prompt="You are a Swedish language expert.",
    user_message="Translate 'hello' to Swedish"
)
print(response)

# Streaming
for chunk in provider.stream_chat(
    messages=[{"role": "user", "content": "Write a short story"}]
):
    print(chunk, end='', flush=True)
```

Advanced Usage:
---------------
```python
# Custom model and parameters
response = provider.chat(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=500
)

# List available models
models = provider.get_models()
for model_name, model_info in models.items():
    print(f"{model_name}: {model_info['description']}")

# Get specific model info
info = provider.get_model_info("gpt-4o")
print(info)
```

Swedish Document Processing:
-----------------------------
```python
# Extract information from Swedish document
provider = OpenAITextProvider()

response = provider.chat_with_system(
    system_prompt=\"\"\"Du är en AI-assistent som extraherar information från
    svenska myndighetsdokument. Svara alltid på svenska.\"\"\",
    user_message=f"Extrahera alla datum från: {document_text}",
    temperature=0.1  # Low temperature for accuracy
)

# Structured extraction
extraction_prompt = \"\"\"
Extrahera följande information från dokumentet:
- Dokumenttyp
- Datum
- Ansvarig myndighet
- Sammanfattning

Format: JSON
\"\"\"

response = provider.chat(
    messages=[
        {"role": "system", "content": "Du är expert på svenska dokument."},
        {"role": "user", "content": f"{extraction_prompt}\n\n{document_text}"}
    ],
    temperature=0.1,
    max_tokens=2000
)
```

Error Handling:
---------------
```python
from openai import OpenAIError

try:
    response = provider.complete("Test prompt")
except ValueError as e:
    print(f"Configuration error: {e}")
except OpenAIError as e:
    print(f"API error: {e}")
```

Available Models:
-----------------
- gpt-4o: Most capable GPT-4o model, multimodal, fast
- gpt-4o-mini: Smaller, cheaper GPT-4o variant
- gpt-4-turbo: GPT-4 Turbo, large context window
- gpt-3.5-turbo: Fast, cost-effective for simple tasks

Best Practices:
---------------
1. Use temperature=0.1 for extraction tasks (deterministic)
2. Use temperature=0.7-0.9 for creative tasks
3. Always set max_tokens to prevent runaway costs
4. Use system messages for role definition
5. Use streaming for long responses to improve UX
6. Handle API errors gracefully
7. Cache responses when appropriate

Module Structure:
-----------------
openai_text/
├── __init__.py          # This file - exports and docs
├── provider.py          # OpenAITextProvider class
├── model_config.py      # Model configurations
└── requirements.txt     # Dependencies
"""

from .provider import OpenAITextProvider
from .model_config import (
    TEXT_MODELS,
    get_model_config,
    get_default_model,
    list_all_models
)

__all__ = [
    'OpenAITextProvider',
    'TEXT_MODELS',
    'get_model_config',
    'get_default_model',
    'list_all_models'
]

__version__ = '1.0.0'

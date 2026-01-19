# Filepath: code_migration/ai_providers/anthropic_batch/__init__.py
# Description: Anthropic Batch API module - self-contained copy-paste ready provider
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/Anthropic/batch/

"""
Anthropic Batch API Provider
=============================

Self-contained module for Anthropic Message Batches API with 50% cost savings.

Features:
---------
- 50% discount on API costs compared to standard API
- Up to 100,000 requests per batch
- Asynchronous processing (most batches complete within 1 hour)
- Results available for 24 hours after completion
- Support for all Claude 4.5 and Claude 4 models
- Extended thinking support with large budget tokens
- File-based request input (JSON/JSONL)
- Simple prompt-based batch creation
- Progress polling and result retrieval

Installation:
-------------
pip install anthropic>=0.25.0 python-dotenv

Environment Setup:
------------------
export ANTHROPIC_API_KEY="sk-ant-..."
export ANTHROPIC_MODEL="claude-sonnet-4-5-20250929"  # optional

Quick Start:
------------
from anthropic_batch import AnthropicBatchProvider

# Initialize
provider = AnthropicBatchProvider()

# Create batch from simple prompts
batch = provider.create_batch_from_prompts([
    "What is 2+2?",
    "Explain quantum computing in simple terms",
    "Write a haiku about Python"
])

# Wait for completion
status = provider.wait_for_completion(batch['id'], verbose=True)

# Get results
results = provider.get_results_text(batch['id'])
for custom_id, text in results.items():
    print(f"{custom_id}: {text[:100]}...")

# Save to file
provider.get_results(batch['id'], output_dir="results/")

Advanced Usage:
---------------

1. Create batch from file:
   ```python
   batch = provider.create_batch_from_file("requests.json")
   ```

2. Create batch with custom requests:
   ```python
   requests = [
       {
           "custom_id": "req-1",
           "prompt": "What is AI?",
           "max_tokens": 1024,
           "system": "You are a helpful assistant"
       },
       {
           "custom_id": "req-2",
           "messages": [
               {"role": "user", "content": "Hello!"}
           ],
           "model": "claude-opus-4-5-20250918"
       }
   ]
   batch = provider.create_batch(requests)
   ```

3. List all batches:
   ```python
   batches = provider.list_batches(limit=10)
   for batch in batches:
       print(f"{batch['id']}: {batch['status']}")
   ```

4. Check status:
   ```python
   status = provider.get_status(batch_id)
   print(f"Status: {status['status']}")
   print(f"Completed: {status['request_counts']['succeeded']}/{status['request_counts']['total']}")
   ```

5. Cancel batch:
   ```python
   provider.cancel_batch(batch_id)
   ```

Request File Formats:
---------------------

JSON Format (requests.json):
```json
[
  {
    "custom_id": "request-1",
    "prompt": "What is AI?",
    "max_tokens": 1024
  },
  {
    "custom_id": "request-2",
    "params": {
      "model": "claude-sonnet-4-5-20250929",
      "max_tokens": 2048,
      "messages": [
        {"role": "user", "content": "Hello!"}
      ]
    }
  }
]
```

JSONL Format (requests.jsonl):
```
{"custom_id": "req-1", "prompt": "What is AI?"}
{"custom_id": "req-2", "prompt": "Explain Python"}
```

Command Execution:
------------------
Execute batch workflows using the command string DSL:

```python
workflow = '''
BATCH_CREATE_PROMPTS prompts_file="prompts.txt" system="Be helpful" -> batch
BATCH_WAIT batch_id="${batch.id}" poll_interval=30
BATCH_RESULTS batch_id="${batch.id}" output_dir="results/"
PRINT message="Batch processing complete!"
'''

result = provider.execute_commands(workflow, verbose=True)
print(f"Commands executed: {result['total_commands']}")
print(f"Errors: {result['total_errors']}")
```

Available Commands:
- BATCH_CREATE requests_file="..." [output_dir="..."]
- BATCH_CREATE_PROMPTS prompts_file="..." [system="..."] [prefix="..."]
- BATCH_STATUS batch_id="..."
- BATCH_WAIT batch_id="..." [poll_interval=60] [timeout=86400]
- BATCH_RESULTS batch_id="..." [output_dir="..."]
- BATCH_LIST [limit=20]
- BATCH_CANCEL batch_id="..."
- SET var="value"
- WAIT seconds=N
- PRINT message="..."
- SAVE content="..." file="..."

Use `-> variable_name` to store results in variables.
Use `${variable_name}` or `${LAST_OUTPUT}` for substitution.

Batch Processing Best Practices:
---------------------------------
1. Use batches for non-time-sensitive workloads (50% cost savings)
2. Group similar requests together for efficiency
3. Use custom_id for easy result mapping
4. Monitor status periodically (default: 60s interval)
5. Download results within 24 hours of completion
6. Use extended thinking for complex reasoning tasks
7. Set appropriate max_tokens per request
8. Consider using Haiku model for simple tasks (fastest + cheapest)

Cost Optimization:
------------------
- Standard API: $3/million input tokens (Sonnet 4.5)
- Batch API: $1.50/million input tokens (50% savings!)
- Best for: Data processing, classification, extraction, analysis
- Not ideal for: Interactive chat, real-time responses

Supported Models:
-----------------
All Claude 4.5 and Claude 4 models:
- claude-sonnet-4-5-20250929 (default, recommended)
- claude-opus-4-5-20250918 (max intelligence)
- claude-haiku-4-5-20251001 (fastest, most economical)
- claude-opus-4-1-20250805
- claude-sonnet-4-20250514

Limits:
-------
- Max requests per batch: 100,000
- Max batch size: 256 MB
- Results available: 24 hours after completion
- Processing time: Most batches complete within 1 hour

Error Handling:
---------------
```python
try:
    batch = provider.create_batch_from_file("requests.json")
    status = provider.wait_for_completion(batch['id'], timeout=3600)
    results = provider.get_results(batch['id'])
except ValueError as e:
    print(f"Invalid request format: {e}")
except TimeoutError as e:
    print(f"Batch did not complete in time: {e}")
except Exception as e:
    print(f"Error: {e}")
```

Swedish Document Processing Example:
-------------------------------------
```python
# Process Swedish municipal documents
swedish_prompts = [
    "Sammanfatta denna årsredovisning",
    "Extrahera viktiga beslut från protokollet",
    "Lista alla lagrum som nämns i dokumentet"
]

batch = provider.create_batch_from_prompts(
    prompts=swedish_prompts,
    system="Du är en expert på svenska kommunala dokument.",
    model="claude-sonnet-4-5-20250929",
    max_tokens=2048
)

status = provider.wait_for_completion(batch['id'])
results = provider.get_results_text(batch['id'])
```

Module Structure:
-----------------
anthropic_batch/
├── __init__.py          # This file - documentation and exports
├── provider.py          # AnthropicBatchProvider class
├── model_config.py      # Model configurations and helpers
└── requirements.txt     # Dependencies

Author: DocFlow Pipeline System
License: Copy-paste ready, self-contained module
"""

from .provider import AnthropicBatchProvider
from .model_config import (
    get_model_config,
    get_default_model,
    get_fastest_model,
    get_smartest_model,
    get_all_models,
    get_current_models,
    BATCH_CONFIG,
    CLAUDE_45_MODELS,
    CLAUDE_4_MODELS,
)

__all__ = [
    "AnthropicBatchProvider",
    "get_model_config",
    "get_default_model",
    "get_fastest_model",
    "get_smartest_model",
    "get_all_models",
    "get_current_models",
    "BATCH_CONFIG",
    "CLAUDE_45_MODELS",
    "CLAUDE_4_MODELS",
]

__version__ = "1.0.0"
__author__ = "DocFlow Pipeline System"

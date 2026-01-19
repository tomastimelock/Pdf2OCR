# RAG Pipeline - Standalone Module

A complete, self-contained module for building Retrieval Augmented Generation (RAG) pipelines with OpenAI, Anthropic Claude, and Pinecone vector database.

## Features

- **Unified API** - Consistent interface across providers
- **RAG Pipeline** - Complete end-to-end retrieval augmented generation
- **Structured Output** - Pydantic models and JSON schemas
- **Batch Processing** - Process multiple requests efficiently
- **Vector Search** - Semantic search with metadata filtering
- **Hybrid Search** - Combine semantic and keyword search
- **Self-Contained** - All imports relative, copy-pasteable
- **Production Ready** - Error handling, progress tracking, health checks

## Supported Providers

| Provider | Chat | Embeddings | Structured Output | Vector DB | Web Search |
|----------|------|------------|-------------------|-----------|------------|
| **OpenAI** | ✓ | ✓ | ✓ | - | ✓ |
| **Anthropic** | ✓ | - | ✓ | - | - |
| **Pinecone** | - | - | - | ✓ | - |

## Installation

```bash
cd code_migration/integration/rag_pipeline
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- anthropic >= 0.40.0
- openai >= 1.60.0
- pinecone >= 7.0.0
- pydantic >= 2.0.0 (optional)

## Quick Start

### Environment Setup

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export PINECONE_API_KEY="pcsk_..."
```

### Basic Usage

```python
from rag_pipeline import create_provider

# OpenAI chat
openai = create_provider('openai')
response = openai.chat("Explain quantum computing")
print(response)

# Claude chat
claude = create_provider('anthropic')
response = claude.chat("Explain quantum computing")
print(response)
```

### Complete RAG Pipeline

```python
from rag_pipeline import create_provider

# 1. Initialize providers
openai = create_provider('openai')
pinecone = create_provider('pinecone')
claude = create_provider('anthropic')

# 2. Create vector index
pinecone.create_index('docs', dimension=1536)

# 3. Embed documents
chunks = ["Document chunk 1...", "Document chunk 2..."]
embeddings = openai.embed_batch(chunks)

# 4. Upsert to vector DB
vectors = [
    (f"doc-{i}", emb, {"text": chunk})
    for i, (emb, chunk) in enumerate(zip(embeddings, chunks))
]
pinecone.upsert_batch(vectors)

# 5. Query with context
query = "What are the main topics?"
results = pinecone.search_by_text(query, openai, top_k=5)
context = "\n".join([r['metadata']['text'] for r in results])

# 6. Generate response
response = claude.chat_with_context(query, context)
print(response)
```

## Module Structure

```
rag_pipeline/
├── __init__.py              # Exports and documentation
├── base.py                  # Base provider interface & factory
├── anthropic_provider.py    # Claude API integration
├── openai_provider.py       # OpenAI API integration
├── pinecone_provider.py     # Pinecone vector DB integration
├── requirements.txt         # Dependencies
├── README.md               # This file
├── USAGE.md                # Detailed usage guide
└── example.py              # Runnable examples
```

## Key Classes

### Providers

```python
from rag_pipeline import (
    AnthropicProvider,    # Claude chat & structured output
    OpenAIProvider,       # GPT chat, embeddings, images
    PineconeProvider,     # Vector database operations
)
```

### Factory

```python
from rag_pipeline import create_provider, ProviderFactory

# Convenience function
provider = create_provider('openai', api_key='sk-...')

# Factory pattern
ProviderFactory.register('custom', CustomProvider)
provider = ProviderFactory.create('custom')
```

## Core Methods

### Chat

```python
# Basic chat
response = openai.chat("Hello, world!")

# Chat with context (RAG)
response = claude.chat_with_context(query, context)

# Multi-turn conversation
messages = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "How are you?"}
]
response = openai.chat_conversation(messages)

# Batch processing
responses = claude.chat_batch(prompts, on_progress=callback)
```

### Structured Output

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

# Pydantic model
person = openai.chat_with_pydantic(text, Person)

# JSON schema
result = claude.chat_structured(text, schema, "person")
```

### Embeddings

```python
# Single embedding
vector = openai.embed("Sample text")

# Batch embeddings
vectors = openai.embed_batch(["Text 1", "Text 2"])

# Dimension info
dim = openai.get_embedding_dimension()  # 1536
```

### Vector Search

```python
# Text search (with embedding provider)
results = pinecone.search_by_text(query, openai, top_k=5)

# Vector search
results = pinecone.query(vector, top_k=10)

# Metadata filtering
results = pinecone.query(
    vector,
    filter={"category": "quantum", "year": {"$gte": 2020}}
)

# Hybrid search (semantic + keywords)
results = pinecone.search_hybrid(vector, keywords=["ai", "ml"])
```

### Index Management

```python
# Create index
pinecone.create_index('my-index', dimension=1536)

# Connect to existing index
pinecone.connect('my-index')

# List indexes
indexes = pinecone.list_indexes()

# Get stats
stats = pinecone.get_stats()

# Delete index
pinecone.delete_index('my-index')
```

## Advanced Features

### Web Search (OpenAI)

```python
result = openai.web_search(
    "Latest quantum computing developments",
    allowed_domains=["arxiv.org", "nature.com"],
    include_sources=True
)
print(result.text)
print(result.sources)
```

### Image Generation

```python
urls = openai.generate_image(
    "A futuristic quantum computer",
    size="1024x1024",
    quality="hd"
)
```

### Namespaces (Pinecone)

```python
# Upsert to namespace
pinecone.upsert(vectors, namespace="documents")

# Query from namespace
results = pinecone.query(vector, namespace="documents")

# List namespaces
namespaces = pinecone.list_namespaces()

# Delete namespace
pinecone.delete_namespace("documents")
```

### Progress Tracking

```python
def progress(current, total):
    print(f"{current}/{total} completed")

# Batch chat
results = claude.chat_batch(prompts, on_progress=progress)

# Batch upsert
pinecone.upsert_batch(vectors, on_progress=progress)
```

## Configuration

### Models

```python
# OpenAI
openai = create_provider('openai', model='gpt-4o-mini')

# Anthropic
claude = create_provider('anthropic', model='claude-haiku-4-5-20251001')

# Temperature & tokens
provider = create_provider('openai', temperature=0.1, max_tokens=500)
```

### Pinecone Index

```python
pinecone.create_index(
    name='my-index',
    dimension=1536,
    metric='cosine',
    cloud='aws',
    region='us-east-1'
)
```

## Health Checks

```python
# Check API accessibility
if openai.health_check():
    print("OpenAI API is accessible")

# Get provider info
info = claude.get_info()
print(info['capabilities'])
print(info['context_limit'])
```

## Error Handling

```python
from rag_pipeline import create_provider

try:
    openai = create_provider('openai')
    response = openai.chat("Hello")
except ValueError as e:
    print(f"Configuration error: {e}")
except ImportError as e:
    print(f"Missing dependency: {e}")
except Exception as e:
    print(f"API error: {e}")
```

## Examples

Run the included examples:

```bash
python example.py
```

See `USAGE.md` for detailed examples and patterns.

## Integration with DocFlow

```python
from rag_pipeline import create_provider

# Extract text (using DocFlow extractors)
from extractors import PDFExtractor
text = PDFExtractor().extract("document.pdf")

# Chunk text
chunks = chunk_text(text, size=500)

# Generate embeddings
openai = create_provider('openai')
embeddings = openai.embed_batch(chunks)

# Store in vector DB
pinecone = create_provider('pinecone')
pinecone.connect('docflow-index')
vectors = [(f"doc-{i}", emb, {"text": c}) for i, (emb, c) in enumerate(zip(embeddings, chunks))]
pinecone.upsert_batch(vectors)

# Query with RAG
query = "Sammanfattning av dokumentet"
results = pinecone.search_by_text(query, openai)
context = "\n".join([r['metadata']['text'] for r in results])

# Generate response
claude = create_provider('anthropic')
response = claude.chat_with_context(query, context)
```

## Performance Tips

1. **Batch embeddings** - Use `embed_batch()` for multiple texts
2. **Reuse providers** - Create once, use multiple times
3. **Connection pooling** - Keep provider instances alive
4. **Namespaces** - Use for multi-tenant isolation
5. **Progress callbacks** - Monitor long-running operations
6. **Token counting** - Use `count_tokens_approx()` to estimate costs

## Swedish Document Support

```python
# Swedish system message
system_msg = "Du är en assistent för svenska dokument."

# Process Swedish text
claude = create_provider('anthropic')
response = claude.chat(
    "Sammanfatta detta dokument",
    system_message=system_msg
)
```

## Reference

Extracted from:
- `reference_codebase/RAG/provider/base.py`
- `reference_codebase/RAG/provider/anthropic_provider.py`
- `reference_codebase/RAG/provider/openai_provider.py`
- `reference_codebase/RAG/provider/pinecone_provider.py`

## License

Part of DocumentHandler / DocFlow project.

## Support

For issues or questions, see the main DocFlow documentation.

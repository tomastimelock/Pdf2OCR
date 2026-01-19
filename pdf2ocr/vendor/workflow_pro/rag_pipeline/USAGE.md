# RAG Pipeline - Usage Guide

Complete standalone module for Retrieval Augmented Generation pipelines.

## Installation

```bash
cd code_migration/integration/rag_pipeline
pip install -r requirements.txt
```

## Environment Setup

```bash
# Set API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export PINECONE_API_KEY="pcsk_..."
```

## Quick Start

### 1. Basic Chat

```python
from rag_pipeline import create_provider

# OpenAI
openai = create_provider('openai')
response = openai.chat("Explain quantum computing in simple terms")
print(response)

# Claude
claude = create_provider('anthropic')
response = claude.chat("Explain quantum computing in simple terms")
print(response)
```

### 2. Complete RAG Pipeline

```python
from rag_pipeline import create_provider

# Initialize providers
openai = create_provider('openai')
pinecone = create_provider('pinecone')
claude = create_provider('anthropic')

# Step 1: Create vector index
pinecone.create_index('knowledge-base', dimension=1536)

# Step 2: Prepare documents
documents = [
    "Quantum computing uses qubits which can be in superposition.",
    "Quantum entanglement allows qubits to be correlated.",
    "Quantum algorithms like Shor's can factor large numbers efficiently."
]

# Step 3: Generate embeddings
embeddings = openai.embed_batch(documents)

# Step 4: Upsert to vector DB
vectors = [
    (f"doc-{i}", emb, {"text": doc, "topic": "quantum"})
    for i, (emb, doc) in enumerate(zip(embeddings, documents))
]
pinecone.upsert_batch(vectors)

# Step 5: Query with RAG
query = "What is quantum entanglement?"
results = pinecone.search_by_text(query, openai, top_k=3)

# Step 6: Build context
context = "\n".join([r['metadata']['text'] for r in results])

# Step 7: Generate response with context
response = claude.chat_with_context(query, context)
print(response)
```

### 3. Structured Output (Pydantic)

```python
from pydantic import BaseModel
from rag_pipeline import create_provider

class ExtractedInfo(BaseModel):
    person_name: str
    age: int
    occupation: str
    location: str

openai = create_provider('openai')
result = openai.chat_with_pydantic(
    "John Doe is a 35-year-old software engineer living in San Francisco",
    ExtractedInfo
)

print(result.person_name)  # "John Doe"
print(result.age)          # 35
print(result.occupation)   # "software engineer"
```

### 4. Structured Output (JSON Schema)

```python
from rag_pipeline import create_provider

schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_points": {
            "type": "array",
            "items": {"type": "string"}
        },
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"]
        }
    },
    "required": ["summary", "key_points", "sentiment"],
    "additionalProperties": False
}

claude = create_provider('anthropic')
result = claude.chat_structured(
    "Analyze this review: The product is amazing! Great quality and fast shipping.",
    response_schema=schema,
    schema_name="review_analysis"
)

print(result['summary'])
print(result['key_points'])
print(result['sentiment'])
```

### 5. Batch Processing

```python
from rag_pipeline import create_provider

claude = create_provider('anthropic')

prompts = [
    "What is machine learning?",
    "What is deep learning?",
    "What is neural network?"
]

# Process with progress tracking
results = claude.chat_batch(
    prompts,
    system_message="Provide concise explanations",
    on_progress=lambda curr, total: print(f"Progress: {curr}/{total}")
)

for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}")
    print(f"A: {result}\n")
```

### 6. Multi-turn Conversation

```python
from rag_pipeline import create_provider

openai = create_provider('openai')

messages = [
    {"role": "system", "content": "You are a helpful coding assistant"},
    {"role": "user", "content": "How do I reverse a list in Python?"},
    {"role": "assistant", "content": "Use list.reverse() or [::-1]"},
    {"role": "user", "content": "What's the difference?"}
]

response = openai.chat_conversation(messages)
print(response)
```

### 7. Vector Search with Metadata Filtering

```python
from rag_pipeline import create_provider

openai = create_provider('openai')
pinecone = create_provider('pinecone')
pinecone.connect('knowledge-base')

# Search with metadata filter
query_vector = openai.embed("quantum algorithms")
results = pinecone.query(
    vector=query_vector,
    top_k=5,
    filter={"topic": "quantum", "year": {"$gte": 2020}}
)

for r in results:
    print(f"Score: {r['score']:.3f} - {r['metadata']['text']}")
```

### 8. Hybrid Search (Semantic + Keywords)

```python
from rag_pipeline import create_provider

openai = create_provider('openai')
pinecone = create_provider('pinecone')
pinecone.connect('knowledge-base')

query_vector = openai.embed("quantum computing")
results = pinecone.search_hybrid(
    vector=query_vector,
    keywords=["algorithm", "qubit"],
    keyword_field="tags",
    top_k=10
)
```

## Advanced Features

### Web Search (OpenAI)

```python
from rag_pipeline import create_provider

openai = create_provider('openai')

# Basic web search
result = openai.web_search(
    "Latest developments in quantum computing 2025",
    include_sources=True
)

print(result.text)
print("Sources:", result.sources)
print("Citations:", result.citations)

# Scholarly research
result = openai.web_search_for_research(
    topic="Quantum error correction",
    scholarly_sources=True
)
```

### Image Generation

```python
from rag_pipeline import create_provider

openai = create_provider('openai')

urls = openai.generate_image(
    "A futuristic quantum computer in a lab",
    size="1024x1024",
    quality="hd",
    n=1
)

print(urls[0])
```

### Batch Embeddings

```python
from rag_pipeline import create_provider

openai = create_provider('openai')

texts = ["text 1", "text 2", "text 3"]
embeddings = openai.embed_batch(texts)

# Check dimensions
dim = openai.get_embedding_dimension()
print(f"Embedding dimension: {dim}")  # 1536
```

## Provider Capabilities

```python
from rag_pipeline import PROVIDER_CAPABILITIES

# Check what each provider supports
print(PROVIDER_CAPABILITIES['openai']['models'])
print(PROVIDER_CAPABILITIES['anthropic']['context_window'])
print(PROVIDER_CAPABILITIES['pinecone']['features'])
```

## Factory Pattern

```python
from rag_pipeline import ProviderFactory

# List available providers
providers = ProviderFactory.list_providers()
print(providers)  # ['openai', 'anthropic', 'pinecone']

# Create dynamically
provider_type = 'openai'
provider = ProviderFactory.create(provider_type)
```

## Health Checks

```python
from rag_pipeline import create_provider

openai = create_provider('openai')
if openai.health_check():
    print("OpenAI API is accessible")

claude = create_provider('anthropic')
info = claude.get_info()
print(info)
# {
#   'provider': 'anthropic',
#   'model': 'claude-sonnet-4-5-20250929',
#   'capabilities': ['chat', 'structured_output', 'tools'],
#   'context_limit': 200000
# }
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

## Performance Tips

1. **Batch embeddings** - Use `embed_batch()` instead of multiple `embed()` calls
2. **Connection pooling** - Reuse provider instances
3. **Namespace isolation** - Use Pinecone namespaces for multi-tenant scenarios
4. **Progress callbacks** - Monitor long-running batch operations
5. **Context limits** - Check token counts with `count_tokens_approx()`

## Swedish Document Processing

```python
from rag_pipeline import create_provider

# Setup
openai = create_provider('openai')
claude = create_provider('anthropic')
pinecone = create_provider('pinecone')

# Create Swedish documents index
pinecone.create_index('swedish-docs', dimension=1536)

# Swedish system message
system_msg = "Du är en assistent som hjälper till med svenska dokument."

# Process Swedish text
swedish_text = "Enligt kommunallagen..."
response = claude.chat(
    "Sammanfatta detta stycke",
    system_message=system_msg
)
```

## File Structure

```
rag_pipeline/
├── __init__.py              # Main exports and usage examples
├── base.py                  # Base provider interface
├── anthropic_provider.py    # Claude integration
├── openai_provider.py       # GPT integration
├── pinecone_provider.py     # Vector DB integration
├── requirements.txt         # Dependencies
└── USAGE.md                 # This file
```

## Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Set environment variables with API keys
3. Try the examples above
4. Integrate with DocFlow extractors and processors
5. Build custom RAG pipelines for Swedish documents

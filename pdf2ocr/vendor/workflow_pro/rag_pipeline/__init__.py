# Filepath: code_migration/integration/rag_pipeline/__init__.py
# Description: RAG Pipeline - Unified provider framework for retrieval augmented generation
# Layer: Integration
# References: reference_codebase/RAG/provider/

"""
RAG Pipeline - Unified Provider Framework

A standalone, self-contained module providing unified access to:
- Anthropic (Claude) - Chat completions with RAG context
- OpenAI (GPT-4o) - Chat, embeddings, structured output
- Pinecone - Vector database operations

Usage Examples
--------------

Basic Chat:
    from rag_pipeline import create_provider

    # OpenAI chat
    openai = create_provider('openai', api_key='sk-...')
    response = openai.chat("Explain quantum computing")

    # Claude chat
    claude = create_provider('anthropic', api_key='sk-ant-...')
    response = claude.chat("Explain quantum computing")

RAG Pipeline:
    from rag_pipeline import create_provider

    # Initialize providers
    openai = create_provider('openai')
    pinecone = create_provider('pinecone')
    claude = create_provider('anthropic')

    # 1. Create/connect to vector index
    pinecone.create_index('docs', dimension=1536)

    # 2. Extract and embed documents
    chunks = ["Document chunk 1...", "Document chunk 2..."]
    embeddings = openai.embed_batch(chunks)

    # 3. Upsert to vector DB
    vectors = [
        (f"doc-{i}", emb, {"text": chunk})
        for i, (emb, chunk) in enumerate(zip(embeddings, chunks))
    ]
    pinecone.upsert_batch(vectors)

    # 4. Query with context
    query = "What are the main topics?"
    results = pinecone.search_by_text(query, openai, top_k=5)
    context = "\n".join([r['metadata']['text'] for r in results])

    # 5. Generate response
    response = claude.chat_with_context(query, context)
    print(response)

Structured Output:
    from pydantic import BaseModel
    from rag_pipeline import create_provider

    class Person(BaseModel):
        name: str
        age: int
        occupation: str

    openai = create_provider('openai')
    person = openai.chat_with_pydantic(
        "Extract info: John Doe, 30, software engineer",
        Person
    )
    print(person.name)  # "John Doe"
    print(person.age)   # 30

Batch Processing:
    from rag_pipeline import create_provider

    claude = create_provider('anthropic')

    prompts = ["Question 1", "Question 2", "Question 3"]
    results = claude.chat_batch(
        prompts,
        on_progress=lambda curr, total: print(f"{curr}/{total}")
    )

Environment Variables:
    OPENAI_API_KEY - OpenAI API key
    ANTHROPIC_API_KEY - Anthropic API key
    PINECONE_API_KEY - Pinecone API key
"""

from .base import BaseProvider, ProviderType, ProviderFactory, create_provider
from .anthropic_provider import AnthropicProvider
from .openai_provider import OpenAIProvider, WebSearchResult
from .pinecone_provider import PineconeProvider

__version__ = "1.0.0"

__all__ = [
    # Base
    'BaseProvider',
    'ProviderType',
    'ProviderFactory',
    'create_provider',

    # Providers
    'AnthropicProvider',
    'OpenAIProvider',
    'PineconeProvider',

    # Utilities
    'WebSearchResult',
]


# Quick reference
PROVIDER_CAPABILITIES = {
    'anthropic': {
        'chat': True,
        'structured_output': True,
        'embeddings': False,
        'vector_db': False,
        'web_search': False,
        'context_window': 200000,
        'models': [
            'claude-sonnet-4-5-20250929',
            'claude-opus-4-5-20251101',
            'claude-haiku-4-5-20251001',
        ]
    },
    'openai': {
        'chat': True,
        'structured_output': True,
        'embeddings': True,
        'vector_db': False,
        'web_search': True,
        'context_window': 128000,
        'models': [
            'gpt-4o',
            'gpt-4o-mini',
            'gpt-4-turbo',
            'o1',
            'o3-mini',
        ]
    },
    'pinecone': {
        'chat': False,
        'structured_output': False,
        'embeddings': False,
        'vector_db': True,
        'web_search': False,
        'context_window': None,
        'features': [
            'serverless',
            'namespaces',
            'metadata_filtering',
            'hybrid_search',
        ]
    }
}

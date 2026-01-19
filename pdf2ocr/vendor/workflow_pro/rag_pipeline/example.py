#!/usr/bin/env python3
# Filepath: code_migration/integration/rag_pipeline/example.py
# Description: Example usage of RAG pipeline module
# Layer: Integration
# References: reference_codebase/RAG/provider/

"""
RAG Pipeline Examples

Demonstrates key features of the standalone RAG pipeline module.
"""

import os
from typing import List


def example_basic_chat():
    """Example 1: Basic chat with OpenAI and Claude."""
    print("\n=== Example 1: Basic Chat ===")

    from rag_pipeline import create_provider

    # OpenAI
    print("OpenAI GPT-4o:")
    openai = create_provider('openai')
    response = openai.chat("What is a vector database in one sentence?")
    print(f"Response: {response}\n")

    # Claude
    print("Claude Sonnet 4.5:")
    claude = create_provider('anthropic')
    response = claude.chat("What is a vector database in one sentence?")
    print(f"Response: {response}")


def example_structured_output():
    """Example 2: Structured output with Pydantic."""
    print("\n=== Example 2: Structured Output ===")

    from pydantic import BaseModel
    from rag_pipeline import create_provider

    class PersonInfo(BaseModel):
        name: str
        age: int
        occupation: str
        city: str

    openai = create_provider('openai')

    text = "Alice Johnson is a 28-year-old data scientist working in Stockholm"
    result = openai.chat_with_pydantic(text, PersonInfo)

    print(f"Name: {result.name}")
    print(f"Age: {result.age}")
    print(f"Occupation: {result.occupation}")
    print(f"City: {result.city}")


def example_embeddings():
    """Example 3: Generate embeddings."""
    print("\n=== Example 3: Embeddings ===")

    from rag_pipeline import create_provider

    openai = create_provider('openai')

    # Single embedding
    text = "Quantum computing uses qubits"
    embedding = openai.embed(text)
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")

    # Batch embeddings
    texts = ["Text 1", "Text 2", "Text 3"]
    embeddings = openai.embed_batch(texts)
    print(f"\nBatch embeddings: {len(embeddings)} vectors")


def example_rag_pipeline():
    """Example 4: Complete RAG pipeline."""
    print("\n=== Example 4: RAG Pipeline ===")

    from rag_pipeline import create_provider

    # Initialize providers
    openai = create_provider('openai')
    pinecone = create_provider('pinecone')
    claude = create_provider('anthropic')

    # Check Pinecone connection
    if not pinecone.health_check():
        print("Pinecone not available (API key not set)")
        return

    # Create/connect to index
    index_name = 'test-rag-pipeline'
    try:
        pinecone.create_index(index_name, dimension=1536, delete_if_exists=True)
        print(f"Created index: {index_name}")
    except Exception as e:
        print(f"Error creating index: {e}")
        return

    # Prepare documents
    documents = [
        "Python is a high-level programming language.",
        "JavaScript is commonly used for web development.",
        "Rust is known for memory safety and performance."
    ]

    # Generate embeddings
    embeddings = openai.embed_batch(documents)
    print(f"Generated {len(embeddings)} embeddings")

    # Upsert to vector DB
    vectors = [
        (f"doc-{i}", emb, {"text": doc, "category": "programming"})
        for i, (emb, doc) in enumerate(zip(embeddings, documents))
    ]
    result = pinecone.upsert_batch(vectors)
    print(f"Upserted {result['total_upserted']} vectors")

    # Query with RAG
    query = "What language is good for web development?"
    results = pinecone.search_by_text(query, openai, top_k=2)

    print(f"\nTop {len(results)} matches:")
    for r in results:
        print(f"  Score: {r['score']:.3f} - {r['metadata']['text']}")

    # Generate response with context
    context = "\n".join([r['metadata']['text'] for r in results])
    response = claude.chat_with_context(query, context)
    print(f"\nClaude's response:\n{response}")

    # Cleanup
    pinecone.delete_index(index_name)
    print(f"\nDeleted index: {index_name}")


def example_batch_processing():
    """Example 5: Batch processing with progress."""
    print("\n=== Example 5: Batch Processing ===")

    from rag_pipeline import create_provider

    claude = create_provider('anthropic')

    prompts = [
        "Define AI in one sentence",
        "Define ML in one sentence",
        "Define DL in one sentence"
    ]

    def progress_callback(current: int, total: int):
        print(f"Progress: {current}/{total}")

    results = claude.chat_batch(
        prompts,
        system_message="Be concise",
        on_progress=progress_callback
    )

    print("\nResults:")
    for prompt, result in zip(prompts, results):
        print(f"Q: {prompt}")
        print(f"A: {result}\n")


def example_provider_info():
    """Example 6: Provider information."""
    print("\n=== Example 6: Provider Information ===")

    from rag_pipeline import create_provider, ProviderFactory, PROVIDER_CAPABILITIES

    # List available providers
    providers = ProviderFactory.list_providers()
    print(f"Available providers: {providers}")

    # Get info for each provider
    for provider_name in providers:
        try:
            provider = create_provider(provider_name)
            if provider.health_check():
                info = provider.get_info()
                print(f"\n{provider_name.upper()}:")
                print(f"  Capabilities: {info.get('capabilities', [])}")
                if 'context_limit' in info:
                    print(f"  Context limit: {info['context_limit']:,} tokens")
        except Exception as e:
            print(f"\n{provider_name.upper()}: Not configured ({e})")

    # Show capabilities reference
    print("\n\nProvider Capabilities Reference:")
    for name, caps in PROVIDER_CAPABILITIES.items():
        print(f"\n{name.upper()}:")
        for key, value in caps.items():
            if isinstance(value, list):
                print(f"  {key}: {', '.join(map(str, value[:3]))}")
            else:
                print(f"  {key}: {value}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("RAG Pipeline - Example Usage")
    print("=" * 60)

    # Check API keys
    has_openai = bool(os.getenv('OPENAI_API_KEY'))
    has_anthropic = bool(os.getenv('ANTHROPIC_API_KEY'))
    has_pinecone = bool(os.getenv('PINECONE_API_KEY'))

    print("\nAPI Keys Status:")
    print(f"  OPENAI_API_KEY: {'✓' if has_openai else '✗'}")
    print(f"  ANTHROPIC_API_KEY: {'✓' if has_anthropic else '✗'}")
    print(f"  PINECONE_API_KEY: {'✓' if has_pinecone else '✗'}")

    if not (has_openai or has_anthropic):
        print("\nWarning: No API keys configured. Set environment variables:")
        print("  export OPENAI_API_KEY='sk-...'")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        print("  export PINECONE_API_KEY='pcsk_...'")
        return

    # Run examples
    try:
        if has_openai:
            example_embeddings()

        if has_openai or has_anthropic:
            example_basic_chat()

        if has_openai:
            example_structured_output()

        if has_anthropic:
            example_batch_processing()

        if has_openai and has_pinecone:
            example_rag_pipeline()

        example_provider_info()

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

# Filepath: code_migration/ai_providers/mistral_embeddings/model_config.py
# Description: Mistral Embeddings model configuration
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/Mistral/model_config.py

"""
Mistral AI Embeddings Model Configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class EmbeddingModelConfig:
    """Configuration for a Mistral embedding model."""
    name: str
    description: str
    dimensions: int = 1024
    context_length: int = 8192
    notes: str = ""


# Available embedding models
EMBEDDING_MODELS: Dict[str, EmbeddingModelConfig] = {
    "mistral-embed": EmbeddingModelConfig(
        name="mistral-embed",
        description="Mistral's optimized embedding model for semantic search and clustering",
        dimensions=1024,
        context_length=8192,
        notes="Best for semantic similarity, search, and clustering tasks"
    ),
}


def get_default_model() -> str:
    """Get the default embedding model."""
    return "mistral-embed"


def get_model_config(model_name: str) -> Optional[EmbeddingModelConfig]:
    """Get configuration for a specific model."""
    return EMBEDDING_MODELS.get(model_name)


def list_models() -> List[str]:
    """List all available embedding models."""
    return list(EMBEDDING_MODELS.keys())


def get_help_text() -> str:
    """Get help text for embedding commands."""
    return """
Mistral Embeddings - Available Methods
======================================

embed_text(text, model="mistral-embed")
    Generate embedding for a single text string.
    Returns: {embedding, dimensions, usage}

embed_batch(texts, model="mistral-embed")
    Generate embeddings for multiple texts.
    Returns: {embeddings, count, dimensions, usage}

similarity(text1, text2, model="mistral-embed")
    Calculate cosine similarity between two texts.
    Returns: {similarity, usage}

semantic_search(query, documents, top_k=5)
    Search documents by semantic similarity.
    Returns: [{index, document, score}, ...]

Available Models:
- mistral-embed: 1024 dimensions, 8192 context length
"""

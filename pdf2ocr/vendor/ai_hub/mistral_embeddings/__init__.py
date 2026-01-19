# Filepath: code_migration/ai_providers/mistral_embeddings/__init__.py
# Description: Mistral Embeddings module exports
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/Mistral/embeddings/

"""
Mistral AI Embeddings Module

Provides text embedding capabilities using the Mistral AI API:
- Single text embedding
- Batch embedding
- Semantic similarity
- Semantic search
"""

from .provider import (
    MistralEmbeddingsProvider,
    EmbeddingError,
)

from .model_config import (
    EmbeddingModelConfig,
    EMBEDDING_MODELS,
    get_default_model,
    get_model_config,
)

__all__ = [
    "MistralEmbeddingsProvider",
    "EmbeddingError",
    "EmbeddingModelConfig",
    "EMBEDDING_MODELS",
    "get_default_model",
    "get_model_config",
]

__version__ = "1.0.0"

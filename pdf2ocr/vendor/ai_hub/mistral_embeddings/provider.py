# Filepath: code_migration/ai_providers/mistral_embeddings/provider.py
# Description: Mistral AI Embeddings Provider
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/Mistral/embeddings/provider.py

"""
Mistral AI Embeddings Provider

Provides text embedding capabilities using the Mistral AI API.
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from mistralai import Mistral
except ImportError:
    Mistral = None

from .model_config import (
    EMBEDDING_MODELS,
    get_model_config,
    get_default_model,
    get_help_text
)


class EmbeddingError(Exception):
    """Exception raised for embedding errors."""
    pass


class MistralEmbeddingsProvider:
    """
    Provider for Mistral AI embedding operations.

    Supports:
        - Single text embedding
        - Batch embedding
        - Semantic similarity calculation
        - Semantic search
    """

    DEFAULT_MODEL = "mistral-embed"
    EMBEDDING_DIMENSIONS = 1024

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Mistral Embeddings Provider.

        Args:
            api_key: Mistral API key. If not provided, uses MISTRAL_API_KEY env var.

        Raises:
            EmbeddingError: If API key not provided or mistralai not installed.
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise EmbeddingError(
                "Mistral API key required. Set MISTRAL_API_KEY environment variable."
            )

        if Mistral is None:
            raise ImportError(
                "mistralai package required. Install with: pip install mistralai"
            )

        self.client = Mistral(api_key=self.api_key)

    @staticmethod
    def list_models() -> Dict[str, Any]:
        """List all available embedding models."""
        return {name: {
            "description": config.description,
            "dimensions": config.dimensions,
            "context_length": config.context_length,
            "notes": config.notes
        } for name, config in EMBEDDING_MODELS.items()}

    @staticmethod
    def get_help() -> str:
        """Get help text for embedding commands."""
        return get_help_text()

    def embed_text(
        self,
        text: str,
        model: str = DEFAULT_MODEL,
        encoding_format: str = "float"
    ) -> Dict[str, Any]:
        """
        Generate embedding for a single text string.

        Args:
            text: The text to embed.
            model: Embedding model to use.
            encoding_format: Output format ('float' or 'base64').

        Returns:
            Dict with embedding vector and metadata.

        Examples:
            >>> provider = MistralEmbeddingsProvider()
            >>> result = provider.embed_text("Hello, world!")
            >>> print(f"Dimensions: {result['dimensions']}")
        """
        try:
            response = self.client.embeddings.create(
                model=model,
                inputs=[text],
                encoding_format=encoding_format
            )

            return {
                'embedding': response.data[0].embedding,
                'index': response.data[0].index,
                'model': response.model,
                'dimensions': len(response.data[0].embedding),
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }
        except Exception as e:
            raise EmbeddingError(f"Embedding failed: {str(e)}")

    def embed_batch(
        self,
        texts: List[str],
        model: str = DEFAULT_MODEL,
        encoding_format: str = "float"
    ) -> Dict[str, Any]:
        """
        Generate embeddings for multiple text strings.

        Args:
            texts: List of texts to embed.
            model: Embedding model to use.
            encoding_format: Output format.

        Returns:
            Dict with list of embeddings and metadata.

        Examples:
            >>> provider = MistralEmbeddingsProvider()
            >>> result = provider.embed_batch(["Hello", "World", "Foo"])
            >>> print(f"Count: {result['count']}")
        """
        try:
            response = self.client.embeddings.create(
                model=model,
                inputs=texts,
                encoding_format=encoding_format
            )

            # Sort by index to ensure correct order
            sorted_data = sorted(response.data, key=lambda x: x.index)

            return {
                'embeddings': [d.embedding for d in sorted_data],
                'count': len(sorted_data),
                'model': response.model,
                'dimensions': len(sorted_data[0].embedding) if sorted_data else 0,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }
        except Exception as e:
            raise EmbeddingError(f"Batch embedding failed: {str(e)}")

    def similarity(
        self,
        text1: str,
        text2: str,
        model: str = DEFAULT_MODEL
    ) -> Dict[str, Any]:
        """
        Calculate cosine similarity between two texts.

        Args:
            text1: First text.
            text2: Second text.
            model: Embedding model to use.

        Returns:
            Dict with similarity score and metadata.

        Examples:
            >>> provider = MistralEmbeddingsProvider()
            >>> result = provider.similarity(
            ...     "The weather is nice",
            ...     "It's a beautiful day"
            ... )
            >>> print(f"Similarity: {result['similarity']:.4f}")
        """
        result = self.embed_batch([text1, text2], model=model)

        emb1 = result['embeddings'][0]
        emb2 = result['embeddings'][1]

        similarity = self._cosine_similarity(emb1, emb2)

        return {
            'similarity': similarity,
            'text1_preview': text1[:100] + '...' if len(text1) > 100 else text1,
            'text2_preview': text2[:100] + '...' if len(text2) > 100 else text2,
            'model': result['model'],
            'dimensions': result['dimensions'],
            'usage': result['usage']
        }

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def semantic_search(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
        model: str = DEFAULT_MODEL
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search over documents.

        Args:
            query: Search query.
            documents: List of documents to search.
            top_k: Number of top results to return.
            model: Embedding model to use.

        Returns:
            List of top matching documents with scores.

        Examples:
            >>> provider = MistralEmbeddingsProvider()
            >>> results = provider.semantic_search(
            ...     "What is Python?",
            ...     ["Python is a programming language", "Cats are pets"]
            ... )
        """
        # Embed query and documents together
        all_texts = [query] + documents
        result = self.embed_batch(all_texts, model=model)

        query_embedding = result['embeddings'][0]
        doc_embeddings = result['embeddings'][1:]

        # Calculate similarities
        scores = []
        for i, doc_emb in enumerate(doc_embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_emb)
            scores.append({
                'index': i,
                'document': documents[i],
                'score': similarity
            })

        # Sort by score and return top_k
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores[:top_k]

    def embed_and_save(
        self,
        texts: Union[str, List[str]],
        output_path: Union[str, Path],
        model: str = DEFAULT_MODEL
    ) -> Dict[str, Any]:
        """
        Generate embeddings and save to JSON file.

        Args:
            texts: Text or list of texts to embed.
            output_path: Path to save embeddings.
            model: Embedding model to use.

        Returns:
            Dict with embeddings and file path.
        """
        if isinstance(texts, str):
            result = self.embed_text(texts, model=model)
        else:
            result = self.embed_batch(texts, model=model)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

        result['saved_to'] = str(output_path)
        return result


def main():
    """Example usage of the Mistral Embeddings Provider."""
    try:
        provider = MistralEmbeddingsProvider()

        print("=" * 60)
        print("Mistral Embeddings Provider")
        print("=" * 60)

        print("\nAvailable Models:")
        for name, info in provider.list_models().items():
            print(f"  {name}: {info['description']}")
            print(f"    Dimensions: {info['dimensions']}")

        print("\n" + "=" * 60)
        print("Ready to generate embeddings!")
        print("=" * 60)

    except EmbeddingError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()

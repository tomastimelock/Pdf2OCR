# Filepath: code_migration/integration/rag_pipeline/base.py
# Description: Base Provider Interface and Factory
# Layer: Integration
# References: reference_codebase/RAG/provider/base.py

"""
Base Provider Interface and Factory

Defines the common interface for all providers and a factory for creating them.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Type
from enum import Enum


class ProviderType(str, Enum):
    """Available provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    PINECONE = "pinecone"
    ELEVENLABS = "elevenlabs"
    NOTION = "notion"
    DIGITALOCEAN = "digitalocean"


class BaseProvider(ABC):
    """
    Abstract base class for all providers.

    All providers must implement these methods for consistency.
    """

    @abstractmethod
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the provider with API key and optional config."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the provider is properly configured and accessible."""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Return provider information and capabilities."""
        pass


class ProviderFactory:
    """Factory for creating provider instances."""

    _providers: Dict[str, Type[BaseProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[BaseProvider]):
        """Register a provider class."""
        cls._providers[name.lower()] = provider_class

    @classmethod
    def create(
        cls,
        provider_type: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> BaseProvider:
        """
        Create a provider instance.

        Args:
            provider_type: Type of provider to create
            api_key: API key for the provider
            **kwargs: Additional provider-specific configuration

        Returns:
            Provider instance

        Raises:
            ValueError: If provider type is unknown
        """
        provider_type = provider_type.lower()

        if provider_type not in cls._providers:
            available = list(cls._providers.keys())
            raise ValueError(
                f"Unknown provider type: {provider_type}. "
                f"Available: {available}"
            )

        return cls._providers[provider_type](api_key=api_key, **kwargs)

    @classmethod
    def list_providers(cls) -> list:
        """List all registered providers."""
        return list(cls._providers.keys())


def create_provider(
    provider_type: str,
    api_key: Optional[str] = None,
    **kwargs
) -> BaseProvider:
    """
    Convenience function to create a provider.

    Args:
        provider_type: Type of provider ('openai', 'anthropic', etc.)
        api_key: API key for the provider
        **kwargs: Additional configuration

    Returns:
        Provider instance
    """
    return ProviderFactory.create(provider_type, api_key, **kwargs)

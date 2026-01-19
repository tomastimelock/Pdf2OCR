"""Unified LLM Adapter - Single interface for multiple LLM providers."""

import os
import base64
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal, Generator

ProviderType = Literal["anthropic", "openai", "mistral"]


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    text: str
    provider: str
    model: str
    usage: Dict[str, int]
    metadata: Dict[str, Any]


class LLMAdapter:
    """
    Unified adapter for multiple LLM providers.

    Provides a consistent interface for:
    - Text generation
    - Image analysis (vision)
    - Structured output generation

    Supports: Anthropic (Claude), OpenAI (GPT), Mistral
    """

    # Default models for each provider
    DEFAULT_MODELS = {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
        "mistral": "mistral-large-latest"
    }

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        mistral_api_key: Optional[str] = None,
        default_provider: ProviderType = "anthropic"
    ):
        """
        Initialize the LLM adapter.

        Args:
            anthropic_api_key: Anthropic API key (uses ANTHROPIC_API_KEY env)
            openai_api_key: OpenAI API key (uses OPENAI_API_KEY env)
            mistral_api_key: Mistral API key (uses MISTRAL_API_KEY env)
            default_provider: Default provider to use
        """
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.mistral_api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
        self.default_provider = default_provider

        self._clients: Dict[str, Any] = {}

    def _get_client(self, provider: ProviderType):
        """Get or create a client for the specified provider."""
        if provider in self._clients:
            return self._clients[provider]

        if provider == "anthropic":
            if not self.anthropic_api_key:
                raise ValueError("Anthropic API key not configured")
            from anthropic import Anthropic
            client = Anthropic(api_key=self.anthropic_api_key)

        elif provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)

        elif provider == "mistral":
            if not self.mistral_api_key:
                raise ValueError("Mistral API key not configured")
            from mistralai import Mistral
            client = Mistral(api_key=self.mistral_api_key)

        else:
            raise ValueError(f"Unknown provider: {provider}")

        self._clients[provider] = client
        return client

    def is_available(self, provider: ProviderType) -> bool:
        """Check if a provider is available (has API key)."""
        if provider == "anthropic":
            return bool(self.anthropic_api_key)
        elif provider == "openai":
            return bool(self.openai_api_key)
        elif provider == "mistral":
            return bool(self.mistral_api_key)
        return False

    def get_available_providers(self) -> list[str]:
        """Get list of available providers."""
        providers = []
        if self.anthropic_api_key:
            providers.append("anthropic")
        if self.openai_api_key:
            providers.append("openai")
        if self.mistral_api_key:
            providers.append("mistral")
        return providers

    def generate_text(
        self,
        prompt: str,
        system: Optional[str] = None,
        provider: Optional[ProviderType] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> LLMResponse:
        """
        Generate text using the specified LLM.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            provider: Provider to use (defaults to default_provider)
            model: Model to use (defaults to provider's default)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            LLMResponse with generated text
        """
        provider = provider or self.default_provider
        model = model or self.DEFAULT_MODELS.get(provider, "")
        client = self._get_client(provider)

        if provider == "anthropic":
            messages = [{"role": "user", "content": prompt}]
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system or "",
                messages=messages
            )
            return LLMResponse(
                text=response.content[0].text,
                provider=provider,
                model=model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                metadata={"stop_reason": response.stop_reason}
            )

        elif provider == "openai":
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return LLMResponse(
                text=response.choices[0].message.content or "",
                provider=provider,
                model=model,
                usage={
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens
                },
                metadata={"finish_reason": response.choices[0].finish_reason}
            )

        elif provider == "mistral":
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.complete(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return LLMResponse(
                text=response.choices[0].message.content or "",
                provider=provider,
                model=model,
                usage={
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens
                },
                metadata={"finish_reason": response.choices[0].finish_reason}
            )

        raise ValueError(f"Unknown provider: {provider}")

    def analyze_image(
        self,
        image_path: str | Path,
        prompt: str,
        provider: Optional[ProviderType] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096
    ) -> LLMResponse:
        """
        Analyze an image using vision capabilities.

        Args:
            image_path: Path to the image file
            prompt: Analysis prompt
            provider: Provider to use (anthropic or openai)
            model: Model to use
            max_tokens: Maximum tokens in response

        Returns:
            LLMResponse with analysis
        """
        provider = provider or self.default_provider
        if provider == "mistral":
            provider = "openai"  # Mistral doesn't have vision yet

        model = model or self.DEFAULT_MODELS.get(provider, "")
        client = self._get_client(provider)

        image_path = Path(image_path)
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        suffix = image_path.suffix.lower()
        media_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }.get(suffix, "image/jpeg")

        if provider == "anthropic":
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }]

            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages
            )
            return LLMResponse(
                text=response.content[0].text,
                provider=provider,
                model=model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                metadata={}
            )

        elif provider == "openai":
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_data}"
                        }
                    }
                ]
            }]

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens
            )
            return LLMResponse(
                text=response.choices[0].message.content or "",
                provider=provider,
                model=model,
                usage={
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens
                },
                metadata={}
            )

        raise ValueError(f"Provider {provider} does not support vision")

    def stream_text(
        self,
        prompt: str,
        system: Optional[str] = None,
        provider: Optional[ProviderType] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096
    ) -> Generator[str, None, None]:
        """
        Stream text generation.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            provider: Provider to use
            model: Model to use
            max_tokens: Maximum tokens

        Yields:
            Text chunks as they're generated
        """
        provider = provider or self.default_provider
        model = model or self.DEFAULT_MODELS.get(provider, "")
        client = self._get_client(provider)

        if provider == "anthropic":
            messages = [{"role": "user", "content": prompt}]
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                system=system or "",
                messages=messages
            ) as stream:
                for text in stream.text_stream:
                    yield text

        elif provider == "openai":
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        elif provider == "mistral":
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            stream = client.chat.stream(
                model=model,
                messages=messages,
                max_tokens=max_tokens
            )
            for chunk in stream:
                if chunk.data.choices[0].delta.content:
                    yield chunk.data.choices[0].delta.content

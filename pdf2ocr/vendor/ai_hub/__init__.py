"""AI Hub - Unified Multi-LLM Provider Interface.

A comprehensive super-module for interacting with multiple AI/LLM providers
through a unified interface.

Supported Providers:
- Anthropic (Claude 4.5, 4.x, 3.x models)
- OpenAI (GPT-4o, GPT-4-turbo, reasoning models)
- Mistral (chat and embeddings)
- xAI (Grok models)

Features:
- Text generation and chat completions
- Vision/image analysis
- Streaming responses
- Tool use and function calling
- Structured outputs (JSON schemas)
- PDF analysis
- Batch processing
- Extended thinking (Claude)
- Reasoning models (OpenAI o1, o3)

Example Usage:
    from ai_hub import AIHub, AnthropicProvider, OpenAIProvider

    # Quick usage with unified interface
    hub = AIHub()
    response = hub.chat("What is Python?", provider="anthropic")

    # Direct provider usage
    from ai_hub.anthropic_text import AnthropicTextProvider
    claude = AnthropicTextProvider(api_key="...")
    response = claude.generate("Write a poem about AI")

    # OpenAI
    from ai_hub.openai_text import OpenAITextProvider
    gpt = OpenAITextProvider()
    response = gpt.complete("Explain quantum computing")

    # Streaming
    for chunk in claude.stream_generate("Tell me a story"):
        print(chunk, end="", flush=True)

    # Vision
    from ai_hub.anthropic_vision import AnthropicVisionProvider
    vision = AnthropicVisionProvider()
    response = vision.analyze_image("image.jpg", "Describe this image")
"""

__version__ = '1.0.0'
__author__ = 'AI Hub'

# Anthropic providers
try:
    from .anthropic_text import AnthropicTextProvider
except ImportError:
    AnthropicTextProvider = None

try:
    from .anthropic_vision import AnthropicVisionProvider
except ImportError:
    AnthropicVisionProvider = None

try:
    from .anthropic_tools import AnthropicToolsProvider
except ImportError:
    AnthropicToolsProvider = None

try:
    from .anthropic_streaming import AnthropicStreamingProvider
except ImportError:
    AnthropicStreamingProvider = None

try:
    from .anthropic_structured import AnthropicStructuredProvider
except ImportError:
    AnthropicStructuredProvider = None

try:
    from .anthropic_batch import AnthropicBatchProvider
except ImportError:
    AnthropicBatchProvider = None

try:
    from .anthropic_pdf import AnthropicPDFProvider
except ImportError:
    AnthropicPDFProvider = None

try:
    from .anthropic_extended_thinking import AnthropicExtendedThinkingProvider
except ImportError:
    AnthropicExtendedThinkingProvider = None

# OpenAI providers
try:
    from .openai_text import OpenAITextProvider
except ImportError:
    OpenAITextProvider = None

try:
    from .openai_vision import OpenAIVisionProvider
except ImportError:
    OpenAIVisionProvider = None

try:
    from .openai_reasoning import OpenAIReasoningProvider
except ImportError:
    OpenAIReasoningProvider = None

try:
    from .openai_research import OpenAIResearchProvider
except ImportError:
    OpenAIResearchProvider = None

# Mistral providers
try:
    from .mistral_chat import MistralChatProvider
except ImportError:
    MistralChatProvider = None

try:
    from .mistral_embeddings import MistralEmbeddingsProvider
except ImportError:
    MistralEmbeddingsProvider = None

# xAI providers
try:
    from .xai_chat import XAIChatProvider
except ImportError:
    XAIChatProvider = None


class AIHub:
    """Unified interface to all AI providers."""

    def __init__(
        self,
        anthropic_api_key: str = None,
        openai_api_key: str = None,
        mistral_api_key: str = None,
        xai_api_key: str = None,
    ):
        """Initialize AI Hub with optional API keys."""
        self._anthropic_key = anthropic_api_key
        self._openai_key = openai_api_key
        self._mistral_key = mistral_api_key
        self._xai_key = xai_api_key
        self._providers = {}

    def _get_provider(self, provider: str):
        """Get or create a provider instance."""
        if provider not in self._providers:
            if provider == "anthropic":
                self._providers[provider] = AnthropicTextProvider(
                    api_key=self._anthropic_key
                )
            elif provider == "openai":
                self._providers[provider] = OpenAITextProvider(
                    api_key=self._openai_key
                )
            elif provider == "mistral":
                self._providers[provider] = MistralChatProvider(
                    api_key=self._mistral_key
                )
            elif provider == "xai":
                self._providers[provider] = XAIChatProvider(
                    api_key=self._xai_key
                )
            else:
                raise ValueError(f"Unknown provider: {provider}")
        return self._providers[provider]

    def chat(
        self,
        message: str,
        provider: str = "anthropic",
        system: str = None,
        **kwargs
    ) -> str:
        """Send a chat message to the specified provider."""
        p = self._get_provider(provider)
        if hasattr(p, 'generate'):
            return p.generate(prompt=message, system=system, **kwargs)
        elif hasattr(p, 'complete'):
            return p.complete(message, **kwargs)
        elif hasattr(p, 'chat'):
            messages = [{"role": "user", "content": message}]
            if system:
                messages.insert(0, {"role": "system", "content": system})
            return p.chat(messages=messages, **kwargs)
        else:
            raise NotImplementedError(f"Provider {provider} has no chat method")

    def stream(
        self,
        message: str,
        provider: str = "anthropic",
        **kwargs
    ):
        """Stream a response from the specified provider."""
        p = self._get_provider(provider)
        if hasattr(p, 'stream_generate'):
            yield from p.stream_generate(prompt=message, **kwargs)
        elif hasattr(p, 'stream_chat'):
            messages = [{"role": "user", "content": message}]
            yield from p.stream_chat(messages=messages, **kwargs)
        else:
            # Fallback to non-streaming
            yield self.chat(message, provider=provider, **kwargs)

    @staticmethod
    def list_providers() -> list:
        """List all available providers."""
        return ["anthropic", "openai", "mistral", "xai"]

    @staticmethod
    def list_models(provider: str) -> list:
        """List available models for a provider."""
        models = {
            "anthropic": [
                "claude-sonnet-4-20250514",
                "claude-opus-4-20250514",
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
                "claude-3-opus-20240229",
            ],
            "openai": [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "o1",
                "o3-mini",
            ],
            "mistral": [
                "mistral-large-latest",
                "mistral-medium-latest",
                "mistral-small-latest",
            ],
            "xai": [
                "grok-2",
                "grok-2-mini",
            ],
        }
        return models.get(provider, [])


# Convenience function
def create_hub(**kwargs) -> AIHub:
    """Create an AIHub instance."""
    return AIHub(**kwargs)


__all__ = [
    # Main class
    'AIHub',
    'create_hub',

    # Anthropic
    'AnthropicTextProvider',
    'AnthropicVisionProvider',
    'AnthropicToolsProvider',
    'AnthropicStreamingProvider',
    'AnthropicStructuredProvider',
    'AnthropicBatchProvider',
    'AnthropicPDFProvider',
    'AnthropicExtendedThinkingProvider',

    # OpenAI
    'OpenAITextProvider',
    'OpenAIVisionProvider',
    'OpenAIReasoningProvider',
    'OpenAIResearchProvider',

    # Mistral
    'MistralChatProvider',
    'MistralEmbeddingsProvider',

    # xAI
    'XAIChatProvider',
]

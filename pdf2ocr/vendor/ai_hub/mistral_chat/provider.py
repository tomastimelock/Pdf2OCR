# Filepath: code_migration/ai_providers/mistral_chat/provider.py
# Description: Mistral AI chat completion provider with comprehensive chat capabilities
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/Mistral/chat/

"""
Mistral AI Chat Completion Provider

Provides chat completion capabilities using the Mistral AI API.
Supports multi-turn conversations, function calling, JSON output, and streaming.

Features:
    - Basic chat completion with system messages
    - Multi-turn conversations with history
    - Streaming responses
    - JSON mode with schema validation
    - Function/tool calling
    - Vision support (pixtral model)
    - Code generation (codestral model)

Example:
    >>> provider = MistralChatProvider(api_key="your-api-key")
    >>> response = provider.chat("What is the capital of France?")
    >>> print(response)
    'The capital of France is Paris.'
"""

import os
import json
from typing import Optional, Dict, Any, List, Generator

try:
    from mistralai import Mistral
except ImportError:
    Mistral = None

from .model_config import (
    CHAT_MODELS,
    get_model_config,
    get_default_chat_model,
    get_help_text,
    get_model_info,
    validate_model_params,
)


class MistralChatProvider:
    """Provider for Mistral AI chat completion operations.

    This class provides a comprehensive interface to Mistral AI's chat models,
    supporting various chat patterns including simple completion, multi-turn
    conversations, streaming, JSON mode, and function calling.

    Attributes:
        api_key: Mistral API key
        model: Default model to use for completions
        client: Mistral API client instance

    Example:
        >>> provider = MistralChatProvider()
        >>> response = provider.chat("Explain Python decorators")
        >>> print(response)
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize the Mistral Chat Provider.

        Args:
            api_key: Mistral API key. If not provided, uses MISTRAL_API_KEY env var.
            model: Default model to use. Defaults to mistral-small-latest.

        Raises:
            ValueError: If API key is not provided and not in environment
            ImportError: If mistralai package is not installed

        Example:
            >>> # Using environment variable
            >>> provider = MistralChatProvider()
            >>> # Explicit API key
            >>> provider = MistralChatProvider(api_key="your-key")
            >>> # Custom default model
            >>> provider = MistralChatProvider(model="mistral-large-latest")
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Mistral API key required. Provide api_key parameter or set "
                "MISTRAL_API_KEY environment variable."
            )

        if Mistral is None:
            raise ImportError(
                "mistralai package required. Install with: pip install mistralai"
            )

        self.model = model or os.getenv("MISTRAL_MODEL", get_default_chat_model())
        self.client = Mistral(api_key=self.api_key)

    @staticmethod
    def list_models() -> Dict[str, Any]:
        """List all available chat models with their configurations.

        Returns:
            Dictionary mapping model names to their configuration details

        Example:
            >>> models = MistralChatProvider.list_models()
            >>> for name, info in models.items():
            ...     print(f"{name}: {info['description']}")
        """
        return {
            name: {
                "description": config.description,
                "context_length": config.context_length,
                "capabilities": config.capabilities,
                "supported_params": config.supported_params,
            }
            for name, config in CHAT_MODELS.items()
        }

    @staticmethod
    def get_help() -> str:
        """Get help text for chat functionality.

        Returns:
            Formatted help text with model info and usage examples

        Example:
            >>> help_text = MistralChatProvider.get_help()
            >>> print(help_text)
        """
        return get_help_text()

    @staticmethod
    def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a specific model.

        Args:
            model_name: Name or ID of the model

        Returns:
            Dictionary with model details or None if not found

        Example:
            >>> info = MistralChatProvider.get_model_info("mistral-large-latest")
            >>> print(info['context_length'])
            128000
        """
        return get_model_info(model_name)

    def chat(
        self,
        message: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        safe_prompt: bool = False,
        random_seed: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate a chat completion.

        Args:
            message: User message content
            model: Model to use (defaults to instance model)
            system: System message for context/instructions
            history: Previous conversation messages (list of {role, content} dicts)
            temperature: Sampling temperature (0.0-1.0). Higher = more random
            max_tokens: Maximum tokens in response
            top_p: Nucleus sampling parameter (0.0-1.0)
            safe_prompt: Enable safety prompt injection
            random_seed: Seed for deterministic results
            **kwargs: Additional API parameters

        Returns:
            The generated response text

        Example:
            >>> provider = MistralChatProvider()
            >>> # Simple chat
            >>> response = provider.chat("What is Python?")
            >>> # With system message
            >>> response = provider.chat(
            ...     "Explain decorators",
            ...     system="You are a Python expert. Be concise."
            ... )
            >>> # With conversation history
            >>> history = [
            ...     {"role": "user", "content": "Hi"},
            ...     {"role": "assistant", "content": "Hello! How can I help?"}
            ... ]
            >>> response = provider.chat("Tell me about AI", history=history)
        """
        used_model = model or self.model
        messages = []

        # Add system message if provided
        if system:
            messages.append({"role": "system", "content": system})

        # Add history if provided
        if history:
            messages.extend(history)

        # Add current user message
        messages.append({"role": "user", "content": message})

        # Build request parameters
        params = {"model": used_model, "messages": messages}

        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if top_p is not None:
            params["top_p"] = top_p
        if safe_prompt:
            params["safe_prompt"] = True
        if random_seed is not None:
            params["random_seed"] = random_seed

        params.update(kwargs)

        response = self.client.chat.complete(**params)
        return response.choices[0].message.content

    def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> str:
        """Simple completion interface (single message).

        Args:
            prompt: User prompt/question
            model: Model to use (defaults to instance model)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            **kwargs: Additional API parameters

        Returns:
            The generated response text

        Example:
            >>> provider = MistralChatProvider()
            >>> response = provider.complete("Write a haiku about Python")
            >>> print(response)
        """
        return self.chat(
            message=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def chat_with_messages(
        self, messages: List[Dict[str, str]], model: Optional[str] = None, **kwargs
    ) -> str:
        """Generate chat completion with explicit message list.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model to use (defaults to instance model)
            **kwargs: Additional API parameters

        Returns:
            The generated response text

        Example:
            >>> provider = MistralChatProvider()
            >>> messages = [
            ...     {"role": "system", "content": "You are a helpful assistant"},
            ...     {"role": "user", "content": "Hello"},
            ...     {"role": "assistant", "content": "Hi! How can I help?"},
            ...     {"role": "user", "content": "Tell me about AI"}
            ... ]
            >>> response = provider.chat_with_messages(messages)
        """
        used_model = model or self.model
        params = {"model": used_model, "messages": messages, **kwargs}

        response = self.client.chat.complete(**params)
        return response.choices[0].message.content

    def stream_chat(
        self,
        message: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Stream chat completion response in chunks.

        Args:
            message: User message content
            model: Model to use (defaults to instance model)
            system: System message for context
            history: Previous conversation messages
            **kwargs: Additional API parameters

        Yields:
            Response text chunks as they're generated

        Example:
            >>> provider = MistralChatProvider()
            >>> for chunk in provider.stream_chat("Tell me a story"):
            ...     print(chunk, end='', flush=True)
            >>> # With history
            >>> history = [{"role": "user", "content": "Hi"}]
            >>> for chunk in provider.stream_chat("Continue", history=history):
            ...     print(chunk, end='', flush=True)
        """
        used_model = model or self.model
        messages = []

        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": message})

        params = {"model": used_model, "messages": messages, "stream": True, **kwargs}

        response = self.client.chat.stream(**params)

        for chunk in response:
            if chunk.data.choices[0].delta.content:
                yield chunk.data.choices[0].delta.content

    def chat_with_json(
        self,
        message: str,
        json_schema: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        system: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate JSON-formatted response with optional schema validation.

        Args:
            message: User message content
            json_schema: JSON schema for structured output validation
            model: Model to use (defaults to instance model)
            system: System message for context
            **kwargs: Additional API parameters

        Returns:
            Parsed JSON response as dictionary

        Example:
            >>> provider = MistralChatProvider()
            >>> # Simple JSON mode
            >>> result = provider.chat_with_json(
            ...     "List 3 French cities",
            ...     system="Return as JSON with cities array"
            ... )
            >>> print(result)
            {'cities': ['Paris', 'Lyon', 'Marseille']}
            >>> # With schema validation
            >>> schema = {
            ...     "type": "object",
            ...     "properties": {
            ...         "cities": {
            ...             "type": "array",
            ...             "items": {"type": "string"}
            ...         }
            ...     },
            ...     "required": ["cities"]
            ... }
            >>> result = provider.chat_with_json(
            ...     "List 3 French cities",
            ...     json_schema=schema
            ... )
        """
        used_model = model or self.model
        messages = []

        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        params = {"model": used_model, "messages": messages}

        if json_schema:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema},
            }
        else:
            params["response_format"] = {"type": "json_object"}

        params.update(kwargs)

        response = self.client.chat.complete(**params)
        content = response.choices[0].message.content

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            return {"raw_content": content, "error": f"Failed to parse JSON: {str(e)}"}

    def chat_with_tools(
        self,
        message: str,
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        system: Optional[str] = None,
        tool_choice: str = "auto",
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate chat completion with tool/function calling support.

        Args:
            message: User message content
            tools: List of tool definitions (OpenAI function format)
            model: Model to use (defaults to instance model)
            system: System message for context
            tool_choice: Tool selection mode (auto, none, any, required)
            **kwargs: Additional API parameters

        Returns:
            Response dictionary with content and potential tool calls

        Example:
            >>> provider = MistralChatProvider()
            >>> tools = [{
            ...     "type": "function",
            ...     "function": {
            ...         "name": "get_weather",
            ...         "description": "Get weather for a location",
            ...         "parameters": {
            ...             "type": "object",
            ...             "properties": {
            ...                 "location": {"type": "string"}
            ...             },
            ...             "required": ["location"]
            ...         }
            ...     }
            ... }]
            >>> response = provider.chat_with_tools(
            ...     "What's the weather in Paris?",
            ...     tools=tools
            ... )
            >>> if "tool_calls" in response:
            ...     for call in response["tool_calls"]:
            ...         print(call["function"]["name"])
        """
        used_model = model or self.model
        messages = []

        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        params = {
            "model": used_model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            **kwargs,
        }

        response = self.client.chat.complete(**params)
        choice = response.choices[0]

        result = {
            "content": choice.message.content,
            "finish_reason": choice.finish_reason,
        }

        if choice.message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]

        return result

    def get_full_response(
        self, messages: List[Dict[str, str]], model: Optional[str] = None, **kwargs
    ) -> Any:
        """Get the full response object from the API (for advanced use cases).

        Args:
            messages: List of message dictionaries
            model: Model to use (defaults to instance model)
            **kwargs: Additional API parameters

        Returns:
            Full API response object with all metadata

        Example:
            >>> provider = MistralChatProvider()
            >>> messages = [{"role": "user", "content": "Hello"}]
            >>> response = provider.get_full_response(messages)
            >>> print(response.usage.total_tokens)
        """
        params = {"model": model or self.model, "messages": messages, **kwargs}
        return self.client.chat.complete(**params)


def main():
    """Example usage of the Mistral Chat Provider."""
    try:
        # Initialize provider
        provider = MistralChatProvider()

        print("=" * 60)
        print("Mistral Chat Provider - Examples")
        print("=" * 60)

        # List available models
        print("\n1. Available Models:")
        print("-" * 60)
        models = provider.list_models()
        for name, info in models.items():
            print(f"  • {name}")
            print(f"    {info['description']}")
            print(f"    Context: {info['context_length']:,} tokens")
            print()

        # Simple chat
        print("\n2. Simple Chat:")
        print("-" * 60)
        response = provider.chat(
            message="What is the capital of France? Answer in one sentence.",
            temperature=0.3,
        )
        print(f"Q: What is the capital of France?")
        print(f"A: {response}")

        # Chat with system message
        print("\n3. Chat with System Message:")
        print("-" * 60)
        response = provider.chat(
            message="Explain Python decorators",
            system="You are a Python expert. Be concise and use examples.",
            temperature=0.5,
        )
        print(f"Q: Explain Python decorators")
        print(f"A: {response[:200]}...")

        # Streaming chat
        print("\n4. Streaming Chat:")
        print("-" * 60)
        print("Q: Write a haiku about programming")
        print("A: ", end="", flush=True)
        for chunk in provider.stream_chat(
            message="Write a haiku about programming", temperature=0.8
        ):
            print(chunk, end="", flush=True)
        print("\n")

        # JSON mode
        print("\n5. JSON Mode:")
        print("-" * 60)
        result = provider.chat_with_json(
            message="List 3 French cities with their approximate populations",
            system="Return as JSON with cities array. Each city has name and population fields.",
        )
        print("Q: List 3 French cities with populations")
        print(f"A: {json.dumps(result, indent=2)}")

        # Multi-turn conversation
        print("\n6. Multi-turn Conversation:")
        print("-" * 60)
        history = []

        # Turn 1
        msg1 = "Hi, I'm learning Python"
        response1 = provider.chat(msg1, history=history)
        print(f"User: {msg1}")
        print(f"Assistant: {response1}")
        history.append({"role": "user", "content": msg1})
        history.append({"role": "assistant", "content": response1})

        # Turn 2
        msg2 = "What should I learn first?"
        response2 = provider.chat(msg2, history=history)
        print(f"User: {msg2}")
        print(f"Assistant: {response2}")

        print("\n" + "=" * 60)

    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("\nPlease set MISTRAL_API_KEY environment variable:")
        print("  export MISTRAL_API_KEY='your-api-key'")
    except ImportError as e:
        print(f"Import Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

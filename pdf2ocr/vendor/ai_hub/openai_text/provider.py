# Filepath: code_migration/ai_providers/openai_text/provider.py
# Description: OpenAI text completion provider using Chat Completions API
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/openai/text/provider.py

"""
OpenAI Text Completion Provider
================================

Provides text completion using OpenAI's Chat Completions API.
Supports GPT-4o, GPT-4o-mini, GPT-4-turbo, and GPT-3.5-turbo.
"""

import os
from typing import Optional, Dict, Any, List, Union, Generator
from openai import OpenAI
from dotenv import load_dotenv

from .model_config import (
    TEXT_MODELS,
    get_model_config,
    get_default_model,
    list_all_models,
    get_recommended_model,
    estimate_cost
)


class OpenAITextProvider:
    """
    Provider for OpenAI text completion using Chat Completions API.

    This provider handles:
    - Basic chat completions with messages
    - Simple text completion from prompts
    - System + user message patterns
    - Streaming responses
    - Model information and cost estimation

    Attributes:
        client: OpenAI client instance
        api_key: OpenAI API key
        default_model: Default model to use
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize the OpenAI Text Provider.

        Args:
            api_key: OpenAI API key. If not provided, loads from OPENAI_API_KEY env var.
            model: Default model to use. If not provided, loads from OPENAI_MODEL env
                  var or defaults to "gpt-4o".

        Raises:
            ValueError: If API key is not provided or found in environment
        """
        load_dotenv()

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided or set in OPENAI_API_KEY "
                "environment variable"
            )

        self.default_model = model or os.getenv("OPENAI_MODEL", get_default_model())
        self.client = OpenAI(api_key=self.api_key)

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        **kwargs
    ) -> str:
        """
        Generate text completion from a list of messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                     Roles can be: 'system', 'user', 'assistant'
            model: Model to use (defaults to instance default_model)
            temperature: Sampling temperature 0.0-2.0 (default: 0.7)
                        Lower = more deterministic, higher = more creative
            max_tokens: Maximum tokens to generate (default: model default)
            top_p: Nucleus sampling parameter 0.0-1.0 (default: 1.0)
            frequency_penalty: Penalty for token frequency -2.0 to 2.0 (default: 0.0)
            presence_penalty: Penalty for token presence -2.0 to 2.0 (default: 0.0)
            **kwargs: Additional parameters to pass to the API

        Returns:
            Generated text as string

        Example:
            >>> provider = OpenAITextProvider()
            >>> response = provider.chat(
            ...     messages=[
            ...         {"role": "system", "content": "You are helpful."},
            ...         {"role": "user", "content": "Hello!"}
            ...     ],
            ...     temperature=0.7
            ... )
        """
        used_model = model or self.default_model

        params = {
            "model": used_model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            **kwargs
        }

        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content

    def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Simple text completion from a single prompt.

        This is a convenience method that wraps the chat method with a
        single user message.

        Args:
            prompt: Text prompt to complete
            model: Model to use (defaults to instance default_model)
            temperature: Sampling temperature 0.0-2.0 (default: 0.7)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters to pass to chat method

        Returns:
            Generated text as string

        Example:
            >>> provider = OpenAITextProvider()
            >>> response = provider.complete(
            ...     "Explain quantum computing in one sentence"
            ... )
        """
        messages = [{"role": "user", "content": prompt}]

        return self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    def chat_with_system(
        self,
        system_prompt: str,
        user_message: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate completion with system prompt and user message.

        This is a common pattern for defining the AI's role and behavior.

        Args:
            system_prompt: System message defining AI behavior/role
            user_message: User's input message
            model: Model to use (defaults to instance default_model)
            temperature: Sampling temperature 0.0-2.0 (default: 0.7)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters to pass to chat method

        Returns:
            Generated text as string

        Example:
            >>> provider = OpenAITextProvider()
            >>> response = provider.chat_with_system(
            ...     system_prompt="You are a Swedish translator.",
            ...     user_message="Translate 'hello' to Swedish"
            ... )
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        return self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream text completion from messages.

        Yields text chunks as they are generated, allowing for real-time
        display of responses.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model to use (defaults to instance default_model)
            temperature: Sampling temperature 0.0-2.0 (default: 0.7)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters to pass to the API

        Yields:
            Text chunks as they are generated

        Example:
            >>> provider = OpenAITextProvider()
            >>> for chunk in provider.stream_chat(
            ...     messages=[{"role": "user", "content": "Tell a story"}]
            ... ):
            ...     print(chunk, end='', flush=True)
        """
        used_model = model or self.default_model

        params = {
            "model": used_model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            **kwargs
        }

        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        stream = self.client.chat.completions.create(**params)

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def get_full_response(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Get the full response object from the API (not just text).

        Useful for accessing metadata like token usage, finish reason, etc.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model to use (defaults to instance default_model)
            temperature: Sampling temperature 0.0-2.0 (default: 0.7)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters to pass to the API

        Returns:
            Full ChatCompletion response object from OpenAI

        Example:
            >>> provider = OpenAITextProvider()
            >>> response = provider.get_full_response(
            ...     messages=[{"role": "user", "content": "Hi"}]
            ... )
            >>> print(response.usage.total_tokens)
            >>> print(response.choices[0].finish_reason)
        """
        used_model = model or self.default_model

        params = {
            "model": used_model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }

        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        return self.client.chat.completions.create(**params)

    # =========================================================================
    # Model Information Methods
    # =========================================================================

    def get_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available models.

        Returns:
            Dictionary mapping model names to configuration dicts

        Example:
            >>> provider = OpenAITextProvider()
            >>> models = provider.get_models()
            >>> for name, info in models.items():
            ...     print(f"{name}: {info['description']}")
        """
        return list_all_models()

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific model.

        Args:
            model_name: Name of the model (e.g., "gpt-4o")

        Returns:
            Model configuration dict or None if not found

        Example:
            >>> provider = OpenAITextProvider()
            >>> info = provider.get_model_info("gpt-4o")
            >>> print(info['max_tokens'])
        """
        config = get_model_config(model_name)
        if config:
            return config.to_dict()
        return None

    def recommend_model(self, task_type: str) -> str:
        """
        Get recommended model for a specific task type.

        Args:
            task_type: One of "extraction", "classification", "generation",
                      "analysis", "translation", "summarization", "simple"

        Returns:
            Recommended model name

        Example:
            >>> provider = OpenAITextProvider()
            >>> model = provider.recommend_model("extraction")
            >>> print(model)  # "gpt-4o"
        """
        return get_recommended_model(task_type)

    def estimate_cost(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Estimate cost for a completion (approximate).

        Note: This uses a simple token estimation. For accurate token counts,
        use tiktoken library.

        Args:
            messages: List of message dicts
            model: Model to use (defaults to instance default_model)
            max_tokens: Estimated output tokens

        Returns:
            Dict with estimated input_tokens, output_tokens, and cost

        Example:
            >>> provider = OpenAITextProvider()
            >>> estimate = provider.estimate_cost(
            ...     messages=[{"role": "user", "content": "Hello"}],
            ...     max_tokens=500
            ... )
            >>> print(f"Estimated cost: ${estimate['cost']:.4f}")
        """
        used_model = model or self.default_model

        # Rough estimation: 1 token ≈ 4 characters
        total_chars = sum(len(msg.get('content', '')) for msg in messages)
        estimated_input_tokens = total_chars // 4

        cost = estimate_cost(used_model, estimated_input_tokens, max_tokens)

        return {
            'model': used_model,
            'estimated_input_tokens': estimated_input_tokens,
            'estimated_output_tokens': max_tokens,
            'estimated_cost_usd': round(cost, 4)
        }


def main():
    """Example usage of the OpenAI Text Provider."""
    print("=" * 70)
    print("OpenAI Text Provider - Example Usage")
    print("=" * 70)

    try:
        # Initialize the provider
        provider = OpenAITextProvider()
        print(f"\nInitialized with default model: {provider.default_model}")

        # Example 1: Simple completion
        print("\n" + "=" * 70)
        print("Example 1: Simple Completion")
        print("=" * 70)
        response = provider.complete(
            prompt="Write a one-sentence explanation of machine learning.",
            temperature=0.7
        )
        print(f"\nPrompt: Write a one-sentence explanation of machine learning.")
        print(f"Response: {response}")

        # Example 2: Chat with system message
        print("\n" + "=" * 70)
        print("Example 2: Chat with System Message")
        print("=" * 70)
        response = provider.chat_with_system(
            system_prompt="You are a Swedish language expert.",
            user_message="Translate 'Good morning' to Swedish",
            temperature=0.3
        )
        print(f"\nSystem: You are a Swedish language expert.")
        print(f"User: Translate 'Good morning' to Swedish")
        print(f"Response: {response}")

        # Example 3: Multi-turn conversation
        print("\n" + "=" * 70)
        print("Example 3: Multi-turn Conversation")
        print("=" * 70)
        response = provider.chat(
            messages=[
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "What is a Python decorator?"},
                {"role": "assistant", "content": "A decorator is a function that modifies another function."},
                {"role": "user", "content": "Show me a simple example."}
            ],
            temperature=0.5,
            max_tokens=300
        )
        print(f"\nResponse: {response}")

        # Example 4: Streaming
        print("\n" + "=" * 70)
        print("Example 4: Streaming Response")
        print("=" * 70)
        print("\nPrompt: Write a haiku about coding")
        print("Streaming response: ", end='', flush=True)
        for chunk in provider.stream_chat(
            messages=[{"role": "user", "content": "Write a haiku about coding"}],
            temperature=0.8
        ):
            print(chunk, end='', flush=True)
        print("\n")

        # Example 5: Model information
        print("\n" + "=" * 70)
        print("Example 5: Model Information")
        print("=" * 70)
        models = provider.get_models()
        print("\nAvailable models:")
        for name, info in models.items():
            print(f"  - {name}: {info['description']}")

        # Example 6: Cost estimation
        print("\n" + "=" * 70)
        print("Example 6: Cost Estimation")
        print("=" * 70)
        estimate = provider.estimate_cost(
            messages=[
                {"role": "system", "content": "You are a document analyzer."},
                {"role": "user", "content": "Analyze this document: " + "x" * 1000}
            ],
            max_tokens=500
        )
        print(f"\nEstimated cost: ${estimate['estimated_cost_usd']:.4f}")
        print(f"Input tokens: ~{estimate['estimated_input_tokens']}")
        print(f"Output tokens: ~{estimate['estimated_output_tokens']}")

        # Example 7: Task-specific model recommendation
        print("\n" + "=" * 70)
        print("Example 7: Model Recommendations")
        print("=" * 70)
        tasks = ["extraction", "classification", "generation", "analysis"]
        print("\nRecommended models by task:")
        for task in tasks:
            model = provider.recommend_model(task)
            print(f"  - {task}: {model}")

        print("\n" + "=" * 70)
        print("Examples completed successfully!")
        print("=" * 70 + "\n")

    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
        print("\nPlease set OPENAI_API_KEY environment variable:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("Or create a .env file with:")
        print("  OPENAI_API_KEY=your-api-key-here\n")

    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()

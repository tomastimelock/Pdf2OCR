# Filepath: code_migration/ai_providers/xai_chat/provider.py
# Description: xAI Grok Chat Provider for text completion and conversation
# Layer: AI Processor
# References: reference_codebase/AIMOS/providers/xAI/chat/

"""
xAI Grok Chat Completion Provider

Provides chat completion capabilities using the xAI Grok API.
Supports multi-turn conversations, function calling, JSON output, streaming, and live web search.

The xAI API is OpenAI-compatible, using the same endpoint structure and request/response formats.
"""

import os
import json
import requests
from typing import Optional, Dict, Any, List, Generator, Union
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .model_config import (
    CHAT_MODELS,
    get_model_config,
    get_default_chat_model,
    XAI_API_URL,
    list_chat_models
)


class XAIChatProvider:
    """
    Provider for xAI Grok chat completion operations.

    Uses the OpenAI-compatible xAI API for text generation, conversation,
    function calling, structured outputs, and web search integration.

    Attributes:
        api_key: xAI API key
        model: Default model to use for requests
        base_url: API base URL
        headers: Request headers with authorization
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the xAI Chat Provider.

        Args:
            api_key: xAI API key. If not provided, uses XAI_API_KEY env var.
            model: Default model to use. Defaults to grok-2-1212.

        Raises:
            ValueError: If no API key is provided or found in environment
        """
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "xAI API key required. Set XAI_API_KEY environment variable or pass api_key parameter."
            )

        self.model = model or os.getenv("XAI_MODEL", get_default_chat_model())
        self.base_url = XAI_API_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    @staticmethod
    def list_models() -> Dict[str, Any]:
        """
        List all available chat models with their configurations.

        Returns:
            Dictionary mapping model names to their configuration details
        """
        return {
            name: {
                "description": config.description,
                "context_length": config.context_length,
                "capabilities": config.capabilities,
                "supported_params": config.supported_params,
                "default_params": config.default_params
            }
            for name, config in CHAT_MODELS.items()
        }

    @staticmethod
    def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed info about a specific model.

        Args:
            model_name: Model ID

        Returns:
            Dictionary with model details or None if not found
        """
        config = get_model_config(model_name)
        if config:
            return {
                "name": config.name,
                "model_id": config.model_id,
                "category": config.category,
                "description": config.description,
                "context_length": config.context_length,
                "supported_params": config.supported_params,
                "default_params": config.default_params,
                "capabilities": config.capabilities,
                "notes": config.notes
            }
        return None

    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get list of available chat model IDs.

        Returns:
            List of model IDs
        """
        return list_chat_models()

    def chat(
        self,
        message: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate a chat completion.

        Args:
            message: User message content.
            model: Model to use (defaults to instance model).
            system: System message for context.
            history: Previous conversation messages (list of {"role": "user/assistant", "content": "..."}).
            temperature: Sampling temperature (0-2). Higher = more creative.
            max_tokens: Maximum tokens in response.
            top_p: Nucleus sampling parameter (0-1).
            stop: Stop sequences (string or list of strings).
            frequency_penalty: Reduce word repetition (-2 to 2).
            presence_penalty: Encourage topic diversity (-2 to 2).
            **kwargs: Additional API parameters.

        Returns:
            The generated response text.

        Raises:
            requests.HTTPError: If API request fails
        """
        used_model = model or self.model
        messages = []

        # Build message list
        if system:
            messages.append({"role": "system", "content": system})

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": message})

        # Build request payload
        payload = {
            "model": used_model,
            "messages": messages
        }

        # Add optional parameters
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        if stop is not None:
            payload["stop"] = stop
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty

        payload.update(kwargs)

        # Make API request
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Simple completion interface (single prompt without history).

        Args:
            prompt: Text prompt to complete.
            model: Model to use.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.
            **kwargs: Additional parameters.

        Returns:
            The generated completion text.
        """
        return self.chat(
            message=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    def chat_with_messages(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate chat completion with explicit message list.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            model: Model to use.
            **kwargs: Additional parameters (temperature, max_tokens, etc.).

        Returns:
            The generated response text.
        """
        used_model = model or self.model
        payload = {
            "model": used_model,
            "messages": messages,
            **kwargs
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def chat_json(
        self,
        message: str,
        schema: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        system: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate JSON-formatted response.

        Args:
            message: User message content.
            schema: Optional JSON schema for structured output validation.
            model: Model to use.
            system: System message (include JSON formatting instructions here).
            **kwargs: Additional parameters.

        Returns:
            Parsed JSON response as a dictionary.

        Note:
            If schema is provided, uses json_schema mode.
            Otherwise, uses json_object mode (free-form JSON).
        """
        used_model = model or self.model
        messages = []

        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        payload = {
            "model": used_model,
            "messages": messages
        }

        # Set response format
        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema
                }
            }
        else:
            payload["response_format"] = {"type": "json_object"}

        payload.update(kwargs)

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            return {"raw_content": content, "error": f"Failed to parse JSON: {str(e)}"}

    def stream_chat(
        self,
        message: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream chat completion response (yields chunks as they arrive).

        Args:
            message: User message content.
            model: Model to use.
            system: System message.
            history: Previous conversation messages.
            **kwargs: Additional parameters.

        Yields:
            Response text chunks as they are generated.

        Example:
            >>> provider = XAIChatProvider()
            >>> for chunk in provider.stream_chat("Tell me a story"):
            ...     print(chunk, end='', flush=True)
        """
        used_model = model or self.model
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": message})

        payload = {
            "model": used_model,
            "messages": messages,
            "stream": True,
            **kwargs
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            stream=True,
            timeout=120
        )
        response.raise_for_status()

        # Parse SSE stream
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        if chunk["choices"][0].get("delta", {}).get("content"):
                            yield chunk["choices"][0]["delta"]["content"]
                    except json.JSONDecodeError:
                        continue

    def chat_with_tools(
        self,
        message: str,
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        system: Optional[str] = None,
        tool_choice: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate chat completion with tool/function calling.

        Args:
            message: User message content.
            tools: List of tool definitions (OpenAI function format).
            model: Model to use.
            system: System message.
            tool_choice: Tool selection mode ("auto", "none", "any", "required").
            **kwargs: Additional parameters.

        Returns:
            Dictionary with:
                - content: Response text (if any)
                - finish_reason: How generation ended
                - tool_calls: List of tool calls (if any) with id, function name, arguments

        Example:
            >>> tools = [{
            ...     "type": "function",
            ...     "function": {
            ...         "name": "get_weather",
            ...         "description": "Get weather for a location",
            ...         "parameters": {
            ...             "type": "object",
            ...             "properties": {
            ...                 "location": {"type": "string"}
            ...             }
            ...         }
            ...     }
            ... }]
            >>> response = provider.chat_with_tools("What's the weather in Paris?", tools)
        """
        used_model = model or self.model
        messages = []

        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        payload = {
            "model": used_model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            **kwargs
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()

        result = response.json()
        choice = result["choices"][0]

        output = {
            "content": choice["message"].get("content"),
            "finish_reason": choice.get("finish_reason")
        }

        # Extract tool calls if present
        if choice["message"].get("tool_calls"):
            output["tool_calls"] = [
                {
                    "id": tc["id"],
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"]
                    }
                }
                for tc in choice["message"]["tool_calls"]
            ]

        return output

    def chat_with_search(
        self,
        message: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_results: int = 10,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate chat completion with live web search (Grok-specific feature).

        Args:
            message: User message content.
            model: Model to use (should support search, e.g., grok-2-1212).
            system: System message.
            max_results: Maximum search results to use (1-100).
            from_date: Start date for search results (YYYY-MM-DD).
            to_date: End date for search results (YYYY-MM-DD).
            **kwargs: Additional parameters.

        Returns:
            The generated response text with search-informed content.

        Note:
            Search feature is only available on select models (grok-2-1212).
            Results are integrated into the model's context automatically.
        """
        used_model = model or self.model
        messages = []

        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        search_params = {"enabled": True, "max_results": max_results}
        if from_date:
            search_params["from_date"] = from_date
        if to_date:
            search_params["to_date"] = to_date

        payload = {
            "model": used_model,
            "messages": messages,
            "search_parameters": search_params,
            **kwargs
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=180  # Search may take longer
        )
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def get_full_response(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get the full raw response object from the API.

        Args:
            messages: List of messages.
            model: Model to use.
            **kwargs: Additional parameters.

        Returns:
            Full API response object including usage stats, metadata, etc.
        """
        payload = {
            "model": model or self.model,
            "messages": messages,
            **kwargs
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()

        return response.json()


def main():
    """Example usage of the xAI Chat Provider."""
    try:
        provider = XAIChatProvider()

        print("=== xAI Grok Chat Provider ===\n")

        # List available models
        print("Available Models:")
        for name, info in provider.list_models().items():
            print(f"  {name}: {info['description']}")
            print(f"    Context: {info['context_length']:,} tokens")
            print(f"    Capabilities: {', '.join(info['capabilities'])}")
            print()

        # Example 1: Simple chat
        print("\n=== Example 1: Simple Chat ===")
        response = provider.chat(
            message="What is the capital of France? Answer in one sentence.",
            model="grok-2-1212"
        )
        print(f"Response: {response}")

        # Example 2: Chat with system message
        print("\n=== Example 2: Chat with System Message ===")
        response = provider.chat(
            message="Explain Python decorators briefly.",
            system="You are a helpful programming tutor. Be concise and clear."
        )
        print(f"Response: {response}")

        # Example 3: JSON output
        print("\n=== Example 3: JSON Output ===")
        response = provider.chat_json(
            message="List 3 French cities with their populations.",
            system="Return data as JSON with cities array containing name and population fields."
        )
        print(f"Response: {json.dumps(response, indent=2)}")

        # Example 4: Streaming
        print("\n=== Example 4: Streaming ===")
        print("Response: ", end='', flush=True)
        for chunk in provider.stream_chat(
            message="Count from 1 to 5.",
            temperature=0.1
        ):
            print(chunk, end='', flush=True)
        print()

        # Example 5: Complete (simple interface)
        print("\n=== Example 5: Simple Complete ===")
        response = provider.complete(
            prompt="The capital of Japan is",
            temperature=0.1,
            max_tokens=10
        )
        print(f"Completion: {response}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

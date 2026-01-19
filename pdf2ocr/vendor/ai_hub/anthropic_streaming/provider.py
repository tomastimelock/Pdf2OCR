# Filepath: code_migration/ai_providers/anthropic_streaming/provider.py
# Description: AnthropicStreamingProvider - Server-sent events streaming for Claude responses
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/Anthropic/streaming/provider.py

"""
Anthropic Streaming Provider
============================

Real-time streaming of Claude responses using server-sent events (SSE).
Enables incremental response generation for better user experience and
lower perceived latency.

Event Flow:
-----------
1. message_start - Initiates stream with empty Message object
2. content_block_start - Begins each content block
3. content_block_delta - Incremental content updates
4. content_block_stop - Ends content block
5. message_delta - Top-level message changes (tokens, stop_reason)
6. message_stop - Terminates stream
"""

import os
import re
import anthropic
from typing import Iterator, Optional, Dict, Any, List, Callable, AsyncIterator
import asyncio

# Import model configuration
from .model_config import get_model_config, get_default_model, ANTHROPIC_MODELS


class AnthropicStreamingProvider:
    """
    Provider for Anthropic streaming API.

    Enables incremental response streaming through server-sent events (SSE).
    This allows displaying content as it generates rather than waiting for
    complete responses.

    Attributes:
        api_key (str): Anthropic API key
        model (str): Default model to use
        client (anthropic.Anthropic): Sync client instance
        async_client (anthropic.AsyncAnthropic): Async client instance (lazy)
        results (dict): Storage for command execution results
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the streaming provider.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model to use (defaults to claude-sonnet-4-5-20250929)

        Raises:
            ValueError: If no API key is provided
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable or api_key parameter required"
            )

        self.model = model or get_default_model()
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.async_client = None  # Lazy initialization
        self.results = {}

    def _get_async_client(self):
        """Get or create async client."""
        if self.async_client is None:
            self.async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
        return self.async_client

    def stream_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        system: Optional[str] = None,
        temperature: float = 1.0,
        callback: Optional[Callable[[str], None]] = None
    ) -> Iterator[str]:
        """
        Stream text response in real-time.

        Args:
            prompt: The user prompt
            model: Model to use (defaults to instance model)
            max_tokens: Maximum tokens to generate
            system: Optional system prompt
            temperature: Sampling temperature (0.0-1.0)
            callback: Optional callback function for each text chunk

        Yields:
            Text chunks as they are generated

        Example:
            >>> provider = AnthropicStreamingProvider()
            >>> for chunk in provider.stream_text("Tell me a joke"):
            ...     print(chunk, end="", flush=True)
            Why did the AI cross the road?...
        """
        model = model or self.model

        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature
        }

        if system:
            kwargs["system"] = system

        with self.client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                if callback:
                    callback(text)
                yield text

    def stream_text_full(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        system: Optional[str] = None,
        temperature: float = 1.0,
        callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Stream text and return the complete response.

        Args:
            prompt: The user prompt
            model: Model to use (defaults to instance model)
            max_tokens: Maximum tokens to generate
            system: Optional system prompt
            temperature: Sampling temperature (0.0-1.0)
            callback: Optional callback function for each text chunk

        Returns:
            Complete generated text

        Example:
            >>> provider = AnthropicStreamingProvider()
            >>> text = provider.stream_text_full(
            ...     "What is streaming?",
            ...     callback=lambda t: print(".", end="")
            ... )
            ........
            >>> print(text)
            Streaming is a method of transmitting data...
        """
        full_text = ""
        for chunk in self.stream_text(prompt, model, max_tokens, system, temperature, callback):
            full_text += chunk
        return full_text

    def stream_messages(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 4096,
        system: Optional[str] = None,
        temperature: float = 1.0,
        callback: Optional[Callable[[str], None]] = None
    ) -> Iterator[str]:
        """
        Stream response for a multi-turn conversation.

        Args:
            messages: List of conversation messages
            model: Model to use (defaults to instance model)
            max_tokens: Maximum tokens to generate
            system: Optional system prompt
            temperature: Sampling temperature (0.0-1.0)
            callback: Optional callback function for each text chunk

        Yields:
            Text chunks as they are generated

        Example:
            >>> messages = [
            ...     {"role": "user", "content": "Hello"},
            ...     {"role": "assistant", "content": "Hi there!"},
            ...     {"role": "user", "content": "How are you?"}
            ... ]
            >>> for chunk in provider.stream_messages(messages):
            ...     print(chunk, end="")
            I'm doing well, thank you!
        """
        model = model or self.model

        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature
        }

        if system:
            kwargs["system"] = system

        with self.client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                if callback:
                    callback(text)
                yield text

    def stream_with_thinking(
        self,
        prompt: str,
        budget_tokens: int = 10000,
        model: Optional[str] = None,
        max_tokens: int = 16000,
        callback: Optional[Callable[[str, str], None]] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream extended thinking response.

        Args:
            prompt: The user prompt
            budget_tokens: Token budget for thinking (min 1024)
            model: Model to use (defaults to instance model)
            max_tokens: Maximum total output tokens
            callback: Optional callback(type, content) for each chunk

        Yields:
            Dicts with 'type' ('thinking' or 'text') and 'content' keys

        Example:
            >>> for event in provider.stream_with_thinking(
            ...     "Solve 234 * 567",
            ...     budget_tokens=5000
            ... ):
            ...     if event["type"] == "thinking":
            ...         print(f"[Thinking: {event['content']}]")
            ...     else:
            ...         print(event["content"], end="")
            [Thinking: Let me multiply these numbers step by step...]
            The result is 132,678.
        """
        model = model or self.model
        budget_tokens = max(1024, budget_tokens)  # Minimum 1024

        with self.client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            thinking={"type": "enabled", "budget_tokens": budget_tokens},
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0  # Required for extended thinking
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "thinking"):
                        chunk = {"type": "thinking", "content": event.delta.thinking}
                        if callback:
                            callback("thinking", event.delta.thinking)
                        yield chunk
                    elif hasattr(event.delta, "text"):
                        chunk = {"type": "text", "content": event.delta.text}
                        if callback:
                            callback("text", event.delta.text)
                        yield chunk

    def stream_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 4096,
        system: Optional[str] = None,
        callback: Optional[Callable[[str, Any], None]] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream response with tool use.

        Args:
            prompt: The user prompt
            tools: List of tool definitions
            model: Model to use (defaults to instance model)
            max_tokens: Maximum tokens to generate
            system: Optional system prompt
            callback: Optional callback(event_type, data) for events

        Yields:
            Dicts representing stream events

        Example:
            >>> tools = [{
            ...     "name": "calculator",
            ...     "description": "Perform calculations",
            ...     "input_schema": {
            ...         "type": "object",
            ...         "properties": {"expression": {"type": "string"}},
            ...         "required": ["expression"]
            ...     }
            ... }]
            >>> for event in provider.stream_with_tools("What is 5+3?", tools):
            ...     if event["type"] == "tool_complete":
            ...         print(f"Tool: {event['tool']['name']}")
            ...         print(f"Input: {event['tool']['input']}")
            Tool: calculator
            Input: {'expression': '5+3'}
        """
        model = model or self.model

        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "tools": tools,
            "messages": [{"role": "user", "content": prompt}]
        }

        if system:
            kwargs["system"] = system

        current_tool = None
        tool_input = ""

        with self.client.messages.stream(**kwargs) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    if hasattr(event.content_block, "type"):
                        if event.content_block.type == "tool_use":
                            current_tool = {
                                "id": event.content_block.id,
                                "name": event.content_block.name
                            }
                            tool_input = ""
                            if callback:
                                callback("tool_start", current_tool)
                            yield {"type": "tool_start", "tool": current_tool}
                        elif event.content_block.type == "text":
                            if callback:
                                callback("text_start", None)
                            yield {"type": "text_start"}

                elif event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        if callback:
                            callback("text_delta", event.delta.text)
                        yield {"type": "text_delta", "text": event.delta.text}
                    elif hasattr(event.delta, "partial_json"):
                        tool_input += event.delta.partial_json
                        if callback:
                            callback("tool_input_delta", event.delta.partial_json)
                        yield {"type": "tool_input_delta", "partial": event.delta.partial_json}

                elif event.type == "content_block_stop":
                    if current_tool:
                        # Parse complete tool input
                        import json
                        try:
                            parsed_input = json.loads(tool_input) if tool_input else {}
                        except json.JSONDecodeError:
                            parsed_input = {"raw": tool_input}

                        current_tool["input"] = parsed_input
                        if callback:
                            callback("tool_complete", current_tool)
                        yield {"type": "tool_complete", "tool": current_tool}
                        current_tool = None
                    else:
                        if callback:
                            callback("text_complete", None)
                        yield {"type": "text_complete"}

                elif event.type == "message_delta":
                    result = {
                        "type": "message_delta",
                        "stop_reason": getattr(event.delta, "stop_reason", None)
                    }
                    if hasattr(event, "usage"):
                        result["usage"] = {
                            "output_tokens": event.usage.output_tokens
                        }
                    if callback:
                        callback("message_delta", result)
                    yield result

    def stream_to_file(
        self,
        prompt: str,
        output_file: str,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        system: Optional[str] = None,
        show_progress: bool = True
    ) -> str:
        """
        Stream response directly to a file.

        Args:
            prompt: The user prompt
            output_file: Path to output file
            model: Model to use (defaults to instance model)
            max_tokens: Maximum tokens to generate
            system: Optional system prompt
            show_progress: Print progress dots during streaming

        Returns:
            Complete generated text

        Example:
            >>> text = provider.stream_to_file(
            ...     "Write a long essay",
            ...     "essay.txt",
            ...     max_tokens=8000
            ... )
            ..........
            >>> print(f"Wrote {len(text)} characters")
            Wrote 15432 characters
        """
        full_text = ""
        char_count = 0

        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in self.stream_text(prompt, model, max_tokens, system):
                f.write(chunk)
                full_text += chunk
                char_count += len(chunk)

                # Show progress every 100 characters
                if show_progress and char_count >= 100:
                    print(".", end="", flush=True)
                    char_count = 0

        if show_progress:
            print()  # New line after progress dots

        return full_text

    def collect_stream(
        self,
        stream_iterator: Iterator[Any]
    ) -> Any:
        """
        Collect a stream into a complete response.

        Useful for converting any streaming iterator into a complete result.

        Args:
            stream_iterator: Iterator yielding chunks

        Returns:
            Complete collected result (string or dict depending on stream type)

        Example:
            >>> stream = provider.stream_text("Hello")
            >>> result = provider.collect_stream(stream)
            >>> print(result)
            Hello! How can I help you today?
        """
        result = ""
        for chunk in stream_iterator:
            if isinstance(chunk, str):
                result += chunk
            elif isinstance(chunk, dict) and "content" in chunk:
                result += chunk["content"]
        return result

    async def stream_text_async(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        system: Optional[str] = None,
        temperature: float = 1.0,
        callback: Optional[Callable[[str], None]] = None
    ) -> AsyncIterator[str]:
        """
        Async stream text response.

        Args:
            prompt: The user prompt
            model: Model to use (defaults to instance model)
            max_tokens: Maximum tokens to generate
            system: Optional system prompt
            temperature: Sampling temperature (0.0-1.0)
            callback: Optional callback function for each text chunk

        Yields:
            Text chunks as they are generated

        Example:
            >>> import asyncio
            >>> async def main():
            ...     async for chunk in provider.stream_text_async("Hi"):
            ...         print(chunk, end="")
            >>> asyncio.run(main())
            Hello! How can I assist you?
        """
        model = model or self.model
        client = self._get_async_client()

        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }

        if system:
            kwargs["system"] = system

        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                if callback:
                    callback(text)
                yield text

    async def stream_messages_async(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 4096,
        system: Optional[str] = None,
        temperature: float = 1.0,
        callback: Optional[Callable[[str], None]] = None
    ) -> AsyncIterator[str]:
        """
        Async stream multi-turn conversation.

        Args:
            messages: List of conversation messages
            model: Model to use (defaults to instance model)
            max_tokens: Maximum tokens to generate
            system: Optional system prompt
            temperature: Sampling temperature (0.0-1.0)
            callback: Optional callback function for each text chunk

        Yields:
            Text chunks as they are generated
        """
        model = model or self.model
        client = self._get_async_client()

        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature
        }

        if system:
            kwargs["system"] = system

        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                if callback:
                    callback(text)
                yield text

    def stream_with_recovery(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        system: Optional[str] = None,
        max_retries: int = 3
    ) -> str:
        """
        Stream with automatic error recovery.

        If the stream is interrupted, automatically retries with continuation
        from the last received content.

        Args:
            prompt: The user prompt
            model: Model to use (defaults to instance model)
            max_tokens: Maximum tokens to generate
            system: Optional system prompt
            max_retries: Maximum retry attempts on error

        Returns:
            Complete generated text (may be partial if all retries fail)

        Example:
            >>> # Will automatically retry on network errors
            >>> text = provider.stream_with_recovery(
            ...     "Write a very long story",
            ...     max_tokens=8000,
            ...     max_retries=5
            ... )
        """
        partial_response = ""
        retries = 0

        while retries <= max_retries:
            try:
                # If we have partial content, continue from there
                if partial_response:
                    # Create continuation prompt
                    messages = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": partial_response}
                    ]
                    continuation_prompt = "Continue from where you left off."
                    messages.append({"role": "user", "content": continuation_prompt})

                    for chunk in self.stream_messages(messages, model, max_tokens, system):
                        partial_response += chunk
                else:
                    for chunk in self.stream_text(prompt, model, max_tokens, system):
                        partial_response += chunk

                # Success - return full response
                return partial_response

            except Exception as e:
                retries += 1
                if retries > max_retries:
                    print(f"Stream interrupted after {max_retries} retries: {e}")
                    return partial_response
                print(f"Stream interrupted, retrying ({retries}/{max_retries})...")

        return partial_response


# Convenience functions for simple use cases

def stream_response(
    prompt: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 4096,
    system: Optional[str] = None,
    callback: Optional[Callable[[str], None]] = None
) -> str:
    """
    Simple function to stream a response.

    Convenience function for one-off streaming without creating a provider instance.

    Args:
        prompt: The user prompt
        api_key: Anthropic API key
        model: Model to use
        max_tokens: Maximum tokens to generate
        system: Optional system prompt
        callback: Optional callback for each chunk

    Returns:
        Complete generated text

    Example:
        >>> from anthropic_streaming import stream_response
        >>> text = stream_response(
        ...     "What is AI?",
        ...     callback=lambda t: print(t, end="", flush=True)
        ... )
        AI stands for Artificial Intelligence...
    """
    provider = AnthropicStreamingProvider(api_key=api_key, model=model)
    return provider.stream_text_full(
        prompt=prompt,
        max_tokens=max_tokens,
        system=system,
        callback=callback
    )


def stream_chat(
    messages: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 4096,
    system: Optional[str] = None,
    callback: Optional[Callable[[str], None]] = None
) -> str:
    """
    Simple function to stream a multi-turn conversation.

    Args:
        messages: List of conversation messages
        api_key: Anthropic API key
        model: Model to use
        max_tokens: Maximum tokens to generate
        system: Optional system prompt
        callback: Optional callback for each chunk

    Returns:
        Complete generated text

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": "Hi!"},
        ...     {"role": "user", "content": "How are you?"}
        ... ]
        >>> text = stream_chat(messages)
    """
    provider = AnthropicStreamingProvider(api_key=api_key, model=model)
    full_text = ""
    for chunk in provider.stream_messages(messages, max_tokens=max_tokens, system=system, callback=callback):
        full_text += chunk
    return full_text

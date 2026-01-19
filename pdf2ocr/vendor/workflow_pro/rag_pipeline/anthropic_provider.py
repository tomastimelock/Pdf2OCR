# Filepath: code_migration/integration/rag_pipeline/anthropic_provider.py
# Description: Anthropic Provider - Claude Chat Models
# Layer: Integration
# References: reference_codebase/RAG/provider/anthropic_provider.py

"""
Anthropic Provider - Claude Chat Models

Provides access to Anthropic's Claude models:
- Chat completions (Claude 3.5, Claude 4, etc.)
- Structured outputs via tool use
- RAG context integration
- Multi-turn conversations
"""

import os
import json
from typing import List, Dict, Any, Optional, Type, Callable
from .base import BaseProvider, ProviderFactory


class AnthropicProvider(BaseProvider):
    """
    Anthropic Claude API Provider.

    Features:
    - Text generation with Claude models
    - Structured JSON output via tools
    - RAG context integration (XML tags)
    - Multi-turn conversations
    - 200k context window
    """

    # Default models
    DEFAULT_CHAT_MODEL = "claude-sonnet-4-5-20250929"
    DEFAULT_STRUCTURED_MODEL = "claude-sonnet-4-5-20250929"

    # Model context limits
    MODEL_CONTEXT_LIMITS = {
        'claude-sonnet-4-5-20250929': 200000,
        'claude-opus-4-5-20251101': 200000,
        'claude-haiku-4-5-20251001': 200000,
        'claude-3-5-sonnet-20241022': 200000,
        'claude-3-opus-20240229': 200000,
        'claude-3-sonnet-20240229': 200000,
        'claude-3-haiku-20240307': 200000,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Default model for chat
            temperature: Default temperature (0-1)
            max_tokens: Default max tokens (required by Anthropic)
        """
        try:
            import anthropic
            self._anthropic = anthropic
        except ImportError:
            raise ImportError("Anthropic SDK not installed. Run: pip install anthropic")

        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env var.")

        self.client = self._anthropic.Anthropic(api_key=self.api_key)
        self.model = model or self.DEFAULT_CHAT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens

    def health_check(self) -> bool:
        """Check if Anthropic API is accessible."""
        try:
            self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}]
            )
            return True
        except Exception:
            return False

    def get_info(self) -> Dict[str, Any]:
        """Get provider information."""
        return {
            'provider': 'anthropic',
            'model': self.model,
            'temperature': self.temperature,
            'capabilities': ['chat', 'structured_output', 'tools'],
            'context_limit': self.MODEL_CONTEXT_LIMITS.get(self.model, 200000)
        }

    # =========================================================================
    # CHAT METHODS
    # =========================================================================

    def chat(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: User prompt
            system_message: Optional system instructions
            temperature: Override temperature
            max_tokens: Override max tokens
            model: Override model

        Returns:
            Generated text
        """
        kwargs = {
            "model": model or self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens
        }

        if system_message:
            kwargs["system"] = system_message

        response = self.client.messages.create(**kwargs)

        # Extract text from response
        text_parts = []
        for block in response.content:
            if hasattr(block, 'text'):
                text_parts.append(block.text)
        return ''.join(text_parts)

    def chat_with_context(
        self,
        prompt: str,
        context: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate text with RAG context (uses XML tags).

        Args:
            prompt: User prompt
            context: Retrieved context from vector DB
            system_message: Optional system instructions
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        # Claude prefers XML-style context
        full_prompt = f"""Based on the following context, please respond to the user's request.

<context>
{context}
</context>

<user_request>
{prompt}
</user_request>

Use the context above to inform your response. If the context doesn't contain relevant information, indicate that."""

        return self.chat(full_prompt, system_message=system_message, **kwargs)

    def chat_structured(
        self,
        prompt: str,
        response_schema: Dict[str, Any],
        schema_name: str = "response",
        system_message: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output using tool use.

        Args:
            prompt: User prompt
            response_schema: JSON schema for response
            schema_name: Schema name
            system_message: Optional system instructions
            **kwargs: Additional parameters

        Returns:
            Parsed JSON response
        """
        tool = {
            "name": schema_name,
            "description": f"Output structured data according to the {schema_name} schema",
            "input_schema": response_schema
        }

        enhanced_system = system_message or ""
        enhanced_system += f"\n\nYou MUST use the {schema_name} tool to provide your response."

        api_kwargs = {
            "model": kwargs.get('model', self.DEFAULT_STRUCTURED_MODEL),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get('temperature', self.temperature),
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": schema_name}
        }

        if enhanced_system.strip():
            api_kwargs["system"] = enhanced_system.strip()

        response = self.client.messages.create(**api_kwargs)

        # Extract tool use result
        for block in response.content:
            if hasattr(block, 'type') and block.type == 'tool_use':
                if block.name == schema_name:
                    return block.input

        # Fallback: try JSON from text
        for block in response.content:
            if hasattr(block, 'text'):
                try:
                    return json.loads(block.text)
                except json.JSONDecodeError:
                    pass

        raise ValueError(f"Failed to extract structured response for: {schema_name}")

    def chat_with_pydantic(
        self,
        prompt: str,
        pydantic_model: Type,
        system_message: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Generate structured output using Pydantic model.

        Args:
            prompt: User prompt
            pydantic_model: Pydantic BaseModel class
            system_message: Optional system instructions
            **kwargs: Additional parameters

        Returns:
            Pydantic model instance
        """
        schema = pydantic_model.model_json_schema()
        schema_name = pydantic_model.__name__.lower()

        result = self.chat_structured(
            prompt=prompt,
            response_schema=schema,
            schema_name=schema_name,
            system_message=system_message,
            **kwargs
        )
        return pydantic_model(**result)

    def chat_conversation(
        self,
        messages: List[Dict[str, str]],
        system_message: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate response for multi-turn conversation.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_message: Optional system message
            **kwargs: Additional parameters

        Returns:
            Generated response
        """
        # Convert to Anthropic format (no system in messages)
        anthropic_messages = []
        extracted_system = None

        for msg in messages:
            role = msg.get('role', 'user')
            if role == 'system':
                extracted_system = msg.get('content', '')
                continue
            anthropic_messages.append({
                "role": role,
                "content": msg.get('content', '')
            })

        final_system = system_message or extracted_system

        api_kwargs = {
            "model": kwargs.get('model', self.model),
            "messages": anthropic_messages,
            "temperature": kwargs.get('temperature', self.temperature),
            "max_tokens": kwargs.get('max_tokens', self.max_tokens)
        }

        if final_system:
            api_kwargs["system"] = final_system

        response = self.client.messages.create(**api_kwargs)

        text_parts = []
        for block in response.content:
            if hasattr(block, 'text'):
                text_parts.append(block.text)
        return ''.join(text_parts)

    def count_tokens_approx(self, text: str) -> int:
        """Approximate token count (~4 chars per token)."""
        return len(text) // 4

    def get_context_limit(self, model: Optional[str] = None) -> int:
        """
        Get context limit for model.

        Args:
            model: Model name (uses default if not specified)

        Returns:
            Maximum context tokens
        """
        target_model = model or self.model
        return self.MODEL_CONTEXT_LIMITS.get(target_model, 200000)

    # =========================================================================
    # BATCH PROCESSING
    # =========================================================================

    def chat_batch(
        self,
        prompts: List[str],
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> List[str]:
        """
        Process multiple prompts in batch.

        Args:
            prompts: List of prompts to process
            system_message: Optional system instructions
            temperature: Override temperature
            max_tokens: Override max tokens
            model: Override model
            on_progress: Progress callback (current, total)

        Returns:
            List of generated responses
        """
        results = []
        total = len(prompts)

        for i, prompt in enumerate(prompts):
            result = self.chat(
                prompt=prompt,
                system_message=system_message,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model
            )
            results.append(result)

            if on_progress:
                on_progress(i + 1, total)

        return results

    def chat_batch_with_context(
        self,
        prompts: List[str],
        contexts: List[str],
        system_message: Optional[str] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> List[str]:
        """
        Process multiple prompts with contexts in batch.

        Args:
            prompts: List of prompts
            contexts: List of contexts (one per prompt)
            system_message: Optional system instructions
            on_progress: Progress callback (current, total)
            **kwargs: Additional parameters

        Returns:
            List of generated responses
        """
        if len(prompts) != len(contexts):
            raise ValueError("Number of prompts must match number of contexts")

        results = []
        total = len(prompts)

        for i, (prompt, context) in enumerate(zip(prompts, contexts)):
            result = self.chat_with_context(
                prompt=prompt,
                context=context,
                system_message=system_message,
                **kwargs
            )
            results.append(result)

            if on_progress:
                on_progress(i + 1, total)

        return results


# Register with factory
ProviderFactory.register('anthropic', AnthropicProvider)

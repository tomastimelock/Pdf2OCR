# Filepath: code_migration/ai_providers/openai_reasoning/provider.py
# Description: OpenAI Reasoning Provider implementation
# Layer: AI Adapter
# References: reference_codebase/AIMOS/providers/openai/reasoning/provider.py

"""
OpenAI Reasoning Provider
=========================

Provider for OpenAI reasoning models (o1, o1-mini, o3, o3-mini, gpt-5 family).
Supports both legacy Chat Completions API and new Responses API.
"""

import json
from typing import Optional, Dict, Any, List
from openai import OpenAI

from .model_config import (
    get_reasoning_model_config,
    is_responses_api_model,
    get_default_reasoning_model,
    validate_effort_level,
    validate_summary_mode,
    EFFORT_LEVELS,
    SUMMARY_MODES
)


class OpenAIReasoningProvider:
    """
    Provider for OpenAI Reasoning Models.

    Supports both legacy (o1, o1-mini) and new (o3, gpt-5) reasoning models.
    These models use internal chain-of-thought reasoning before generating responses.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the provider with optional API key.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var
        """
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def reason(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        effort: str = "medium",
        summary: Optional[str] = None,
        instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Use reasoning model to answer a prompt.

        Args:
            prompt: The question or problem to solve
            model: Model to use (auto-detects API type)
            max_tokens: Maximum output tokens
            effort: Reasoning effort level (low, medium, high) - Responses API only
            summary: Include reasoning summary (auto, concise, detailed) - Responses API only
            instructions: System-level instructions (Responses API) or None

        Returns:
            Dict with:
                - content: Generated response text
                - model: Model used
                - usage: Token usage stats (including reasoning_tokens if available)
                - reasoning_summary: Summary of reasoning (if requested and available)
                - finish_reason: Why generation stopped
                - success: True if successful

        Example:
            >>> provider = OpenAIReasoningProvider()
            >>> result = provider.reason(
            ...     prompt="What is the time complexity of quicksort?",
            ...     model="o1-mini"
            ... )
            >>> print(result['content'])
        """
        try:
            model = model or get_default_reasoning_model()
            config = get_reasoning_model_config(model)

            if not config:
                return {
                    "error": f"Unknown reasoning model: {model}",
                    "success": False
                }

            # Validate parameters
            if effort and not validate_effort_level(effort):
                return {
                    "error": f"Invalid effort level: {effort}. Must be one of {EFFORT_LEVELS}",
                    "success": False
                }

            if summary and not validate_summary_mode(summary):
                return {
                    "error": f"Invalid summary mode: {summary}. Must be one of {SUMMARY_MODES}",
                    "success": False
                }

            # Route to appropriate API
            if is_responses_api_model(model):
                return self._reason_responses_api(
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    effort=effort,
                    summary=summary,
                    instructions=instructions
                )
            else:
                return self._reason_chat_api(
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens
                )

        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }

    def reason_with_context(
        self,
        prompt: str,
        context: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        effort: str = "medium"
    ) -> Dict[str, Any]:
        """
        Reason about a prompt with additional context.

        Args:
            prompt: The question or problem
            context: Additional context information
            model: Model to use
            max_tokens: Maximum output tokens
            effort: Reasoning effort level

        Returns:
            Response dict (same as reason())

        Example:
            >>> result = provider.reason_with_context(
            ...     prompt="Is this contract legally binding?",
            ...     context="Contract text: ...",
            ...     model="o1"
            ... )
        """
        # Combine context and prompt
        full_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{prompt}"

        return self.reason(
            prompt=full_prompt,
            model=model,
            max_tokens=max_tokens,
            effort=effort
        )

    def solve_problem(
        self,
        problem_description: str,
        model: Optional[str] = None,
        effort: str = "high",
        include_steps: bool = True,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Solve a complex problem with step-by-step reasoning.

        Args:
            problem_description: Description of the problem
            model: Model to use (defaults to o1-mini)
            effort: Reasoning effort (defaults to high for problems)
            include_steps: Request step-by-step breakdown
            max_tokens: Maximum output tokens

        Returns:
            Response dict with solution

        Example:
            >>> result = provider.solve_problem(
            ...     problem_description="Find the shortest path from A to Z in this graph: ...",
            ...     effort="high"
            ... )
        """
        # Build problem-solving prompt
        prompt = f"PROBLEM:\n{problem_description}\n\n"

        if include_steps:
            prompt += "Please provide:\n1. Analysis of the problem\n2. Step-by-step solution\n3. Final answer\n"

        return self.reason(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            effort=effort,
            summary="auto"  # Request reasoning summary
        )

    def get_reasoning_trace(self, response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract reasoning trace from a response.

        Args:
            response: Response dict from reason() or solve_problem()

        Returns:
            Dict with reasoning information:
                - reasoning_tokens: Number of reasoning tokens used
                - reasoning_summary: Summary text (if available)
                - effort: Effort level used
            Or None if no reasoning trace available

        Example:
            >>> result = provider.reason(prompt="...", summary="detailed")
            >>> trace = provider.get_reasoning_trace(result)
            >>> if trace:
            ...     print(f"Used {trace['reasoning_tokens']} reasoning tokens")
        """
        if not response.get('success'):
            return None

        trace = {}

        # Extract reasoning tokens from usage
        if 'usage' in response and 'reasoning_tokens' in response['usage']:
            trace['reasoning_tokens'] = response['usage']['reasoning_tokens']

        # Extract reasoning summary
        if 'reasoning_summary' in response:
            trace['reasoning_summary'] = response['reasoning_summary']

        # Extract effort level (if known)
        if 'effort' in response:
            trace['effort'] = response['effort']

        return trace if trace else None

    # =========================================================================
    # INTERNAL METHODS - API-SPECIFIC IMPLEMENTATIONS
    # =========================================================================

    def _reason_responses_api(
        self,
        prompt: str,
        model: str,
        max_tokens: Optional[int],
        effort: str,
        summary: Optional[str],
        instructions: Optional[str]
    ) -> Dict[str, Any]:
        """
        Use new Responses API for reasoning.

        Supports: o3, o3-mini, gpt-5 family
        """
        # Build request
        request_params = {
            "model": model,
            "reasoning": {"effort": effort},
            "input": [{"role": "user", "content": prompt}]
        }

        if summary:
            request_params["reasoning"]["summary"] = summary

        if instructions:
            request_params["instructions"] = instructions

        if max_tokens:
            request_params["max_output_tokens"] = max_tokens

        # Make request
        response = self.client.responses.create(**request_params)

        # Process response
        return self._process_responses_api_response(response, effort)

    def _reason_chat_api(
        self,
        prompt: str,
        model: str,
        max_tokens: Optional[int]
    ) -> Dict[str, Any]:
        """
        Use legacy Chat Completions API for reasoning.

        Supports: o1, o1-mini, o1-preview

        Note: These models don't support system messages or temperature.
        """
        # Build messages (user message only, no system)
        messages = [{"role": "user", "content": prompt}]

        # Build request
        request_params = {
            "model": model,
            "messages": messages
        }

        # Use max_completion_tokens for o1 models
        if max_tokens:
            request_params["max_completion_tokens"] = max_tokens

        # Make request
        response = self.client.chat.completions.create(**request_params)

        # Process response
        return self._process_chat_api_response(response)

    def _process_responses_api_response(
        self,
        response,
        effort: str
    ) -> Dict[str, Any]:
        """Process Responses API response."""
        result = {
            "success": True,
            "model": response.model,
            "effort": effort
        }

        # Check for incomplete response
        if hasattr(response, 'status'):
            result["status"] = response.status
            if response.status == "incomplete":
                result["incomplete"] = True
                if hasattr(response, 'incomplete_details'):
                    result["incomplete_reason"] = getattr(
                        response.incomplete_details, 'reason', None
                    )

        # Extract content and reasoning
        text_output = ""
        reasoning_summary = ""

        for item in response.output:
            item_type = getattr(item, 'type', None)

            if item_type == "message":
                # Extract message content
                for content in item.content:
                    if hasattr(content, 'text'):
                        text_output += content.text

            elif item_type == "reasoning":
                # Extract reasoning summary
                if hasattr(item, 'summary'):
                    for summary_item in item.summary:
                        if hasattr(summary_item, 'text'):
                            reasoning_summary += summary_item.text

        result["content"] = text_output

        if reasoning_summary:
            result["reasoning_summary"] = reasoning_summary

        # Extract usage info
        if hasattr(response, 'usage'):
            usage = response.usage
            result["usage"] = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens
            }

            # Check for reasoning tokens
            if hasattr(usage, 'output_tokens_details'):
                details = usage.output_tokens_details
                if hasattr(details, 'reasoning_tokens'):
                    result["usage"]["reasoning_tokens"] = details.reasoning_tokens

        # Extract finish reason (if available)
        if hasattr(response, 'finish_reason'):
            result["finish_reason"] = response.finish_reason

        return result

    def _process_chat_api_response(self, response) -> Dict[str, Any]:
        """Process Chat Completions API response."""
        choice = response.choices[0]
        message = choice.message

        result = {
            "success": True,
            "model": response.model,
            "content": message.content,
            "finish_reason": choice.finish_reason
        }

        # Extract usage info
        if hasattr(response, 'usage'):
            usage = response.usage
            result["usage"] = {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            }

            # O1 models have reasoning tokens in completion_tokens_details
            if hasattr(usage, 'completion_tokens_details'):
                details = usage.completion_tokens_details
                if hasattr(details, 'reasoning_tokens'):
                    result["usage"]["reasoning_tokens"] = details.reasoning_tokens

        return result


# =============================================================================
# CLI / TESTING
# =============================================================================

if __name__ == "__main__":
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        exit(1)

    provider = OpenAIReasoningProvider(api_key=api_key)

    print("=== OpenAI Reasoning Provider Test ===\n")

    # Test 1: Simple reasoning (legacy API)
    print("Test 1: Simple reasoning with o1-mini")
    print("-" * 50)
    result = provider.reason(
        prompt="What is 15% of 240? Show your work.",
        model="o1-mini"
    )

    if result.get('success'):
        print(f"Model: {result['model']}")
        print(f"Content: {result['content']}")
        print(f"Usage: {result.get('usage', {})}")

        trace = provider.get_reasoning_trace(result)
        if trace:
            print(f"Reasoning trace: {trace}")
    else:
        print(f"Error: {result.get('error')}")

    print("\n" + "=" * 50 + "\n")

    # Test 2: Problem solving
    print("Test 2: Problem solving")
    print("-" * 50)
    result = provider.solve_problem(
        problem_description="Explain the time complexity of binary search and why it's O(log n).",
        model="o1-mini",
        effort="medium"
    )

    if result.get('success'):
        print(f"Solution: {result['content'][:200]}...")
        if 'reasoning_summary' in result:
            print(f"Reasoning: {result['reasoning_summary'][:100]}...")
    else:
        print(f"Error: {result.get('error')}")

    print("\n" + "=" * 50 + "\n")

    # Test 3: Reasoning with context
    print("Test 3: Reasoning with context")
    print("-" * 50)
    context = "Swedish municipal law requires all decisions to be documented."
    result = provider.reason_with_context(
        prompt="Is this decision properly documented according to Swedish law?",
        context=context,
        model="o1-mini"
    )

    if result.get('success'):
        print(f"Response: {result['content'][:200]}...")
    else:
        print(f"Error: {result.get('error')}")

    print("\n=== All tests completed ===")

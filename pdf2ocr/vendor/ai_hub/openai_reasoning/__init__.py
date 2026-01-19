# Filepath: code_migration/ai_providers/openai_reasoning/__init__.py
# Description: OpenAI Reasoning Provider - o1/o3 models with chain-of-thought
# Layer: AI Adapter
# References: reference_codebase/AIMOS/providers/openai/reasoning/

"""
OpenAI Reasoning Provider
==========================

Self-contained module for OpenAI reasoning models (o1, o1-mini, o1-preview, o3, o3-mini).
These models use internal chain-of-thought reasoning before generating responses,
making them ideal for complex problem-solving, code generation, and multi-step planning.

Key Features:
-------------
- Chain-of-thought reasoning models
- Support for reasoning effort levels (low, medium, high)
- Reasoning trace extraction
- Problem-solving mode
- Context-aware reasoning
- Both legacy (Chat Completions) and new (Responses) API support

Models:
-------
- o1: Original reasoning model (legacy API)
- o1-mini: Smaller, faster reasoning model (legacy API)
- o1-preview: Preview version (legacy API)
- o3: Advanced reasoning model (Responses API)
- o3-mini: Smaller o3 variant (Responses API)
- gpt-5/gpt-5-mini/gpt-5-nano: With reasoning support (Responses API)

Important Limitations:
---------------------
- o1/o3 models do NOT support system messages
- Use 'instructions' parameter instead of system messages (Responses API)
- Temperature, top_p not supported (reasoning is deterministic)
- Streaming not available for reasoning models
- Reserve at least 25,000 tokens for reasoning + output

Usage Examples:
---------------

Basic Reasoning:
>>> from openai_reasoning import OpenAIReasoningProvider
>>> provider = OpenAIReasoningProvider(api_key="sk-...")
>>>
>>> # Simple reasoning task
>>> result = provider.reason(
...     prompt="What is the optimal strategy for solving a Rubik's cube?",
...     model="o1-mini"
... )
>>> print(result['content'])

With Context:
>>> # Reasoning with additional context
>>> context = "The user is a beginner with no prior experience."
>>> result = provider.reason_with_context(
...     prompt="Explain dynamic programming",
...     context=context,
...     model="o1"
... )
>>> print(result['content'])

Problem Solving:
>>> # Structured problem-solving
>>> result = provider.solve_problem(
...     problem_description="Design an efficient algorithm to find the shortest path in a weighted graph",
...     model="o3-mini",
...     effort="high"
... )
>>> print(result['content'])

Extract Reasoning Trace:
>>> # Get reasoning steps (if available)
>>> trace = provider.get_reasoning_trace(result)
>>> if trace:
...     print("Reasoning tokens used:", trace['reasoning_tokens'])

Advanced (Responses API):
>>> # Using new Responses API with effort control
>>> result = provider.reason(
...     prompt="Prove that the square root of 2 is irrational",
...     model="o3",
...     effort="high",
...     summary="detailed"
... )
>>> if 'reasoning_summary' in result:
...     print("Reasoning:", result['reasoning_summary'])
>>> print("Answer:", result['content'])

Code Generation:
>>> # Generate code with reasoning
>>> result = provider.solve_problem(
...     problem_description='''
...         Write a Python function to implement a LRU cache with O(1)
...         get and put operations. Include detailed comments.
...     ''',
...     model="o3",
...     effort="high"
... )
>>> print(result['content'])

Swedish Document Analysis:
>>> # Analyze Swedish legal document
>>> result = provider.reason_with_context(
...     prompt="Analysera juridiska implikationer av detta avtal",
...     context=swedish_contract_text,
...     model="o1"
... )

Configuration:
--------------
Set API key via environment variable:
    export OPENAI_API_KEY="sk-..."

Or pass directly:
    provider = OpenAIReasoningProvider(api_key="sk-...")

Best Practices:
---------------
1. Use o1-mini for faster, cheaper reasoning
2. Use o1 or o3 for complex, high-stakes problems
3. Set effort="high" for critical tasks
4. Monitor reasoning_tokens in usage stats
5. Avoid system messages (not supported)
6. Use instructions parameter for guidance (Responses API)
7. Provide clear, specific prompts
8. Include examples in context when helpful

Token Usage:
------------
Reasoning models consume tokens in two ways:
1. Input tokens: Your prompt + context
2. Output tokens: Visible response
3. Reasoning tokens: Internal chain-of-thought (billed but hidden)

Total cost = input + output + reasoning tokens

Module Contents:
----------------
- OpenAIReasoningProvider: Main provider class
- REASONING_MODELS: Available model configurations
- EFFORT_LEVELS: Reasoning effort options
- SUMMARY_MODES: Reasoning summary modes
"""

from .provider import OpenAIReasoningProvider
from .model_config import (
    REASONING_MODELS,
    EFFORT_LEVELS,
    SUMMARY_MODES,
    get_reasoning_model_config,
    is_responses_api_model,
    get_default_reasoning_model
)

__all__ = [
    'OpenAIReasoningProvider',
    'REASONING_MODELS',
    'EFFORT_LEVELS',
    'SUMMARY_MODES',
    'get_reasoning_model_config',
    'is_responses_api_model',
    'get_default_reasoning_model'
]

__version__ = '1.0.0'

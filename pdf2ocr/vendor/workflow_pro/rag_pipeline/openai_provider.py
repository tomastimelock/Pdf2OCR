# Filepath: code_migration/integration/rag_pipeline/openai_provider.py
# Description: OpenAI Provider - Chat, Embeddings, Images, Web Search
# Layer: Integration
# References: reference_codebase/RAG/provider/openai_provider.py

"""
OpenAI Provider - Chat, Embeddings, Images, Web Search

Provides unified access to OpenAI's APIs:
- Chat completions (GPT-4o, GPT-4, etc.)
- Structured outputs with JSON schemas
- Text embeddings
- Image generation (DALL-E)
- Web search with citations
- Batch processing
"""

import os
import json
from typing import List, Dict, Any, Optional, Type, Union, Callable
from .base import BaseProvider, ProviderFactory


class WebSearchResult:
    """Structured web search result."""

    def __init__(
        self,
        text: str,
        citations: List[Dict[str, Any]] = None,
        sources: List[Dict[str, Any]] = None,
        search_queries: List[str] = None
    ):
        self.text = text
        self.citations = citations or []
        self.sources = sources or []
        self.search_queries = search_queries or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'citations': self.citations,
            'sources': self.sources,
            'search_queries': self.search_queries
        }


class OpenAIProvider(BaseProvider):
    """
    OpenAI API Provider.

    Features:
    - Text generation with chat models
    - Structured JSON output
    - RAG context integration
    - Multi-turn conversations
    - Text embeddings for vector search
    - Image generation
    - Web search with citations
    """

    # Default models
    DEFAULT_CHAT_MODEL = "gpt-4o"
    DEFAULT_STRUCTURED_MODEL = "gpt-4o-2024-08-06"
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
    DEFAULT_IMAGE_MODEL = "dall-e-3"

    # Model context limits
    MODEL_CONTEXT_LIMITS = {
        'gpt-5.1': 1047576,
        'gpt-5': 128000,
        'gpt-4.1': 1047576,
        'gpt-4o': 128000,
        'gpt-4o-2024-08-06': 128000,
        'gpt-4o-mini': 128000,
        'gpt-4-turbo': 128000,
        'gpt-4': 8192,
        'gpt-3.5-turbo': 16385,
    }

    # Models using max_completion_tokens
    MODELS_USE_COMPLETION_TOKENS = {
        'gpt-5.1', 'gpt-5', 'gpt-5-mini', 'gpt-5-nano',
        'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano',
        'o1', 'o1-mini', 'o1-preview', 'o3', 'o3-mini', 'o4-mini'
    }

    # Models that support agentic/reasoning search
    REASONING_MODELS = {'gpt-5', 'o4-mini', 'o3', 'o3-mini', 'o1', 'o1-mini'}

    # Scholarly domains for research
    SCHOLARLY_DOMAINS = [
        "scholar.google.com", "pubmed.ncbi.nlm.nih.gov", "arxiv.org",
        "researchgate.net", "academia.edu", "jstor.org", "springer.com",
        "nature.com", "sciencedirect.com", "wiley.com", "harvard.edu",
        "mit.edu", "stanford.edu", "wikipedia.org", "britannica.com"
    ]

    # Embedding dimensions
    EMBEDDING_DIMENSIONS = {
        'text-embedding-3-small': 1536,
        'text-embedding-3-large': 3072,
        'text-embedding-ada-002': 1536,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Default model for chat
            temperature: Default temperature (0-2)
            max_tokens: Default max tokens
        """
        try:
            from openai import OpenAI
            self._OpenAI = OpenAI
        except ImportError:
            raise ImportError("OpenAI SDK not installed. Run: pip install openai")

        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")

        self.client = self._OpenAI(api_key=self.api_key)
        self.model = model or self.DEFAULT_CHAT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens

    def health_check(self) -> bool:
        """Check if OpenAI API is accessible."""
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    def get_info(self) -> Dict[str, Any]:
        """Get provider information."""
        return {
            'provider': 'openai',
            'model': self.model,
            'temperature': self.temperature,
            'capabilities': ['chat', 'embeddings', 'images', 'websearch', 'structured_output'],
            'context_limit': self.MODEL_CONTEXT_LIMITS.get(self.model, 8192)
        }

    def _uses_completion_tokens(self, model: str) -> bool:
        """Check if model uses max_completion_tokens."""
        if model in self.MODELS_USE_COMPLETION_TOKENS:
            return True
        return any(model.startswith(p) for p in self.MODELS_USE_COMPLETION_TOKENS)

    def _get_token_param(self, model: str, max_tokens: Optional[int]) -> Dict[str, int]:
        """Get appropriate token limit parameter."""
        if max_tokens is None:
            return {}
        key = "max_completion_tokens" if self._uses_completion_tokens(model) else "max_tokens"
        return {key: max_tokens}

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
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        use_model = model or self.model
        response = self.client.chat.completions.create(
            model=use_model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            **self._get_token_param(use_model, max_tokens or self.max_tokens)
        )
        return response.choices[0].message.content

    def chat_with_context(
        self,
        prompt: str,
        context: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate text with RAG context.

        Args:
            prompt: User prompt
            context: Retrieved context from vector DB
            system_message: Optional system instructions
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        full_prompt = f"""Based on the following context, please respond to the user's request.

CONTEXT:
{context}

USER REQUEST:
{prompt}

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
        Generate structured JSON output.

        Args:
            prompt: User prompt
            response_schema: JSON schema for response
            schema_name: Schema name
            system_message: Optional system instructions
            **kwargs: Additional parameters

        Returns:
            Parsed JSON response
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        model = kwargs.get('model', self.DEFAULT_STRUCTURED_MODEL)
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=kwargs.get('temperature', self.temperature),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": response_schema
                }
            },
            **self._get_token_param(model, kwargs.get('max_tokens'))
        )
        return json.loads(response.choices[0].message.content)

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
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        model = kwargs.get('model', self.DEFAULT_STRUCTURED_MODEL)
        response = self.client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            temperature=kwargs.get('temperature', self.temperature),
            response_format=pydantic_model,
            **self._get_token_param(model, kwargs.get('max_tokens'))
        )
        return response.choices[0].message.parsed

    def chat_conversation(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Generate response for multi-turn conversation.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters

        Returns:
            Generated response
        """
        model = kwargs.get('model', self.model)
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=kwargs.get('temperature', self.temperature),
            **self._get_token_param(model, kwargs.get('max_tokens'))
        )
        return response.choices[0].message.content

    def chat_batch(
        self,
        prompts: List[str],
        system_message: Optional[str] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts sequentially.

        Args:
            prompts: List of prompts
            system_message: Optional system instructions (applied to all)
            on_progress: Progress callback (current, total)
            **kwargs: Additional parameters

        Returns:
            List of generated responses
        """
        responses = []
        total = len(prompts)

        for i, prompt in enumerate(prompts):
            response = self.chat(prompt, system_message=system_message, **kwargs)
            responses.append(response)
            if on_progress:
                on_progress(i + 1, total)

        return responses

    def get_context_limit(self, model: Optional[str] = None) -> int:
        """Get context limit for a model."""
        m = model or self.model
        return self.MODEL_CONTEXT_LIMITS.get(m, 8192)

    def count_tokens_approx(self, text: str) -> int:
        """Approximate token count (~4 chars per token)."""
        return len(text) // 4

    # =========================================================================
    # EMBEDDING METHODS
    # =========================================================================

    def embed(
        self,
        text: str,
        model: Optional[str] = None
    ) -> List[float]:
        """
        Create embedding for text.

        Args:
            text: Text to embed
            model: Embedding model

        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            model=model or self.DEFAULT_EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding

    def embed_batch(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """
        Create embeddings for multiple texts.

        Args:
            texts: List of texts
            model: Embedding model

        Returns:
            List of embedding vectors
        """
        response = self.client.embeddings.create(
            model=model or self.DEFAULT_EMBEDDING_MODEL,
            input=texts
        )
        return [d.embedding for d in sorted(response.data, key=lambda x: x.index)]

    def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """Get dimension of embedding model."""
        m = model or self.DEFAULT_EMBEDDING_MODEL
        return self.EMBEDDING_DIMENSIONS.get(m, 1536)

    # =========================================================================
    # IMAGE METHODS
    # =========================================================================

    def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        model: Optional[str] = None,
        n: int = 1
    ) -> List[str]:
        """
        Generate images from prompt.

        Args:
            prompt: Image description
            size: Image size (1024x1024, 1792x1024, etc.)
            quality: 'standard' or 'hd'
            model: Image model
            n: Number of images

        Returns:
            List of image URLs
        """
        response = self.client.images.generate(
            model=model or self.DEFAULT_IMAGE_MODEL,
            prompt=prompt,
            size=size,
            quality=quality,
            n=n
        )
        return [img.url for img in response.data]

    # =========================================================================
    # WEB SEARCH METHODS
    # =========================================================================

    def web_search(
        self,
        query: str,
        model: str = "gpt-4o",
        allowed_domains: Optional[List[str]] = None,
        blocked_domains: Optional[List[str]] = None,
        user_location: Optional[Dict[str, str]] = None,
        use_reasoning: bool = False,
        reasoning_effort: str = "low",
        include_sources: bool = True
    ) -> WebSearchResult:
        """
        Perform web search with AI summary.

        Args:
            query: Search query
            model: Model to use
            allowed_domains: Only search these domains (max 20)
            blocked_domains: Exclude these domains (max 20)
            user_location: Optional geo-location {"country": "US", "city": "NYC"}
            use_reasoning: Use reasoning model for agentic search
            reasoning_effort: Reasoning effort ("low", "medium", "high")
            include_sources: Include sources in response

        Returns:
            WebSearchResult with text, citations, sources
        """
        # Build web search tool configuration
        web_search_config: Dict[str, Any] = {"type": "web_search"}

        if allowed_domains:
            web_search_config["filters"] = {"allowed_domains": allowed_domains[:20]}
        elif blocked_domains:
            web_search_config["filters"] = {"blocked_domains": blocked_domains[:20]}

        if user_location:
            web_search_config["user_location"] = {"type": "approximate", **user_location}

        params: Dict[str, Any] = {
            "model": model,
            "tools": [web_search_config],
            "input": query
        }

        if use_reasoning and model in self.REASONING_MODELS:
            params["reasoning"] = {"effort": reasoning_effort}

        if include_sources:
            params["include"] = ["web_search_call.action.sources"]

        try:
            response = self.client.responses.create(**params)
            return self._parse_web_search_response(response)
        except Exception as e:
            return WebSearchResult(
                text=f"Web search failed: {str(e)}",
                citations=[],
                sources=[],
                search_queries=[query]
            )

    def _parse_web_search_response(self, response) -> WebSearchResult:
        """Parse OpenAI response into WebSearchResult."""
        text = ""
        citations = []
        sources = []
        search_queries = []

        if hasattr(response, 'output_text'):
            text = response.output_text
        elif hasattr(response, 'output'):
            for item in response.output:
                if hasattr(item, 'type'):
                    if item.type == 'web_search_call':
                        if hasattr(item, 'action') and hasattr(item.action, 'query'):
                            search_queries.append(item.action.query)
                        if hasattr(item, 'action') and hasattr(item.action, 'sources'):
                            for source in item.action.sources:
                                sources.append({
                                    'url': getattr(source, 'url', ''),
                                    'title': getattr(source, 'title', ''),
                                    'type': getattr(source, 'type', 'web')
                                })
                    elif item.type == 'message':
                        if hasattr(item, 'content'):
                            for content in item.content:
                                if hasattr(content, 'type') and content.type == 'output_text':
                                    text = getattr(content, 'text', '')
                                    if hasattr(content, 'annotations'):
                                        for ann in content.annotations:
                                            if hasattr(ann, 'type') and ann.type == 'url_citation':
                                                citations.append({
                                                    'url': getattr(ann, 'url', ''),
                                                    'title': getattr(ann, 'title', ''),
                                                    'start_index': getattr(ann, 'start_index', 0),
                                                    'end_index': getattr(ann, 'end_index', 0)
                                                })

        return WebSearchResult(
            text=text,
            citations=citations,
            sources=sources,
            search_queries=search_queries
        )


# Register with factory
ProviderFactory.register('openai', OpenAIProvider)

# Filepath: code_migration/ai_providers/anthropic_structured/provider.py
# Description: Anthropic Structured Outputs Provider with JSON Schema validation
# Layer: AI Processor
# References: reference_codebase/AIMOS/providers/Anthropic/structured_outputs/

"""
Anthropic Structured Outputs Provider
Uses the Anthropic Messages API with output_format for guaranteed JSON schema compliance.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Type
from anthropic import Anthropic
from pydantic import BaseModel


class AnthropicStructuredProvider:
    """
    Provider for Anthropic structured outputs with JSON Schema validation.

    This provider enables guaranteed schema-compliant responses from Claude,
    eliminating parsing errors and ensuring valid JSON output for extraction,
    classification, and structured generation tasks.

    Attributes:
        client: Anthropic API client
        model: Default model to use (claude-sonnet-4-5-20250929)
        api_key: Anthropic API key
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize the Anthropic Structured Outputs Provider.

        Args:
            api_key: Anthropic API key. If not provided, uses ANTHROPIC_API_KEY env var.
            model: Model to use. If not provided, uses ANTHROPIC_MODEL env var or defaults
                  to claude-sonnet-4-5-20250929.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key must be provided or set in ANTHROPIC_API_KEY environment variable"
            )

        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
        self.client = Anthropic(api_key=self.api_key)

    def _extract_text(self, response) -> str:
        """
        Extract text content from a response object.

        Args:
            response: Anthropic API response object

        Returns:
            Concatenated text from all content blocks
        """
        text_parts = []
        for block in response.content:
            if hasattr(block, 'text'):
                text_parts.append(block.text)
        return ''.join(text_parts)

    def chat_with_schema(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        schema_name: str = "response_data",
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response matching a JSON schema.

        This is the core method for structured outputs. It sends a chat conversation
        and ensures the response conforms to the provided JSON schema.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            schema: JSON Schema defining the output structure
            schema_name: Name for the schema (default: "response_data")
            model: Model to use (defaults to instance model)
            system: System prompt for context
            max_tokens: Maximum tokens to generate (default: 4096)
            temperature: Sampling temperature 0.0-1.0 (default: 0.1 for consistency)
            **kwargs: Additional parameters passed to the API

        Returns:
            Response data as a dictionary conforming to the schema

        Raises:
            ValueError: If request is refused or max_tokens is reached

        Example:
            >>> schema = {
            ...     "type": "object",
            ...     "properties": {
            ...         "name": {"type": "string"},
            ...         "age": {"type": "integer"}
            ...     },
            ...     "required": ["name"]
            ... }
            >>> messages = [{"role": "user", "content": "Extract: John, 30"}]
            >>> result = provider.chat_with_schema(messages, schema)
            >>> print(result)  # {'name': 'John', 'age': 30}
        """
        used_model = model or self.model

        output_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema
            }
        }

        params = {
            "model": used_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
            "output_format": output_format
        }

        if system:
            params["system"] = system

        params.update(kwargs)

        response = self.client.messages.create(**params)

        # Check for refusal or incomplete response
        if response.stop_reason == "refusal":
            raise ValueError("Request was refused by the model")
        if response.stop_reason == "max_tokens":
            raise ValueError("Response may be incomplete (max_tokens reached)")

        text = self._extract_text(response)
        return json.loads(text)

    def chat_with_pydantic(
        self,
        messages: List[Dict[str, str]],
        pydantic_model: Type[BaseModel],
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        **kwargs
    ) -> BaseModel:
        """
        Generate a response as a validated Pydantic model.

        This method converts a Pydantic model to JSON Schema, gets a structured
        response, and validates it back into a Pydantic model instance.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            pydantic_model: Pydantic model class to use for validation
            model: Model to use (defaults to instance model)
            system: System prompt for context
            max_tokens: Maximum tokens to generate (default: 4096)
            temperature: Sampling temperature (default: 0.1)
            **kwargs: Additional parameters

        Returns:
            Validated Pydantic model instance

        Example:
            >>> from pydantic import BaseModel
            >>> class Person(BaseModel):
            ...     name: str
            ...     age: int
            >>> messages = [{"role": "user", "content": "Extract: Jane, 25"}]
            >>> person = provider.chat_with_pydantic(messages, Person)
            >>> print(person.name)  # 'Jane'
        """
        # Convert Pydantic model to JSON Schema
        schema = pydantic_model.model_json_schema()
        schema_name = pydantic_model.__name__

        # Get structured response
        data = self.chat_with_schema(
            messages=messages,
            schema=schema,
            schema_name=schema_name,
            model=model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        # Validate and return as Pydantic model
        return pydantic_model(**data)

    def extract_structured(
        self,
        text: str,
        schema: Dict[str, Any],
        schema_name: str = "extracted_data",
        instruction: Optional[str] = None,
        model: Optional[str] = None,
        system: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract structured data from text using a JSON schema.

        Convenience method for single-text extraction tasks. Automatically
        constructs the appropriate prompt.

        Args:
            text: The input text to extract data from
            schema: JSON Schema defining the output structure
            schema_name: Name for the schema (default: "extracted_data")
            instruction: Optional extraction instruction (e.g., "Extract invoice data")
            model: Model to use (defaults to instance model)
            system: System prompt for context
            **kwargs: Additional parameters

        Returns:
            Extracted data as a dictionary

        Example:
            >>> schema = {"type": "object", "properties": {"email": {"type": "string"}}}
            >>> text = "Contact me at john@example.com"
            >>> data = provider.extract_structured(text, schema)
            >>> print(data)  # {'email': 'john@example.com'}
        """
        if instruction:
            prompt = f"{instruction}\n\nText: {text}"
        else:
            prompt = f"Extract structured data from the following text:\n\n{text}"

        messages = [{"role": "user", "content": prompt}]

        return self.chat_with_schema(
            messages=messages,
            schema=schema,
            schema_name=schema_name,
            model=model,
            system=system,
            **kwargs
        )

    def validate_response(
        self,
        response: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """
        Validate a response against a JSON schema.

        Provides defense-in-depth validation even when using structured outputs.

        Args:
            response: Response data to validate
            schema: JSON Schema to validate against

        Returns:
            Tuple of (is_valid, error_messages)

        Example:
            >>> response = {"name": "John", "age": "thirty"}  # Invalid age type
            >>> schema = {"type": "object", "properties": {"age": {"type": "integer"}}}
            >>> is_valid, errors = provider.validate_response(response, schema)
            >>> print(is_valid)  # False
            >>> print(errors)  # ["'thirty' is not of type 'integer'"]
        """
        try:
            from jsonschema import validate, ValidationError

            validate(instance=response, schema=schema)
            return True, []
        except ValidationError as e:
            return False, [str(e)]
        except ImportError:
            # jsonschema not installed, skip validation
            return True, ["jsonschema not installed, validation skipped"]

    # =========================================================================
    # PRE-BUILT EXTRACTORS
    # =========================================================================

    def extract_person(
        self,
        text: str,
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract person information from text.

        Args:
            text: Text containing person information
            model: Model to use
            **kwargs: Additional parameters

        Returns:
            Dictionary with name, age, occupation, email (if found)

        Example:
            >>> result = provider.extract_person("John Smith, 30 years old, engineer")
            >>> print(result)  # {'name': 'John Smith', 'age': 30, 'occupation': 'engineer'}
        """
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's full name"},
                "age": {"type": "integer", "description": "Person's age"},
                "occupation": {"type": "string", "description": "Person's job or profession"},
                "email": {"type": "string", "format": "email", "description": "Email address if mentioned"}
            },
            "required": ["name"]
        }

        return self.extract_structured(
            text=text,
            schema=schema,
            schema_name="person_info",
            instruction="Extract person information from this text",
            model=model,
            **kwargs
        )

    def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract named entities from text.

        Args:
            text: Text to extract entities from
            entity_types: Types of entities to extract (default: people, places, organizations, dates)
            model: Model to use
            **kwargs: Additional parameters

        Returns:
            Dictionary with extracted entities by type

        Example:
            >>> text = "John visited Paris in 2024 to meet with Acme Corp."
            >>> result = provider.extract_entities(text)
            >>> print(result)  # {'people': ['John'], 'places': ['Paris'], ...}
        """
        if entity_types is None:
            entity_types = ["people", "places", "organizations", "dates"]

        properties = {}
        for entity_type in entity_types:
            properties[entity_type] = {
                "type": "array",
                "items": {"type": "string"},
                "description": f"List of {entity_type} mentioned in the text"
            }

        schema = {
            "type": "object",
            "properties": properties,
            "required": entity_types
        }

        instruction = f"Extract the following entity types from this text: {', '.join(entity_types)}"

        return self.extract_structured(
            text=text,
            schema=schema,
            schema_name="extracted_entities",
            instruction=instruction,
            model=model,
            **kwargs
        )

    def classify(
        self,
        text: str,
        categories: List[str],
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Classify text into one of the provided categories.

        Args:
            text: Text to classify
            categories: List of valid category names
            model: Model to use
            **kwargs: Additional parameters

        Returns:
            Dictionary with category, confidence, and reasoning

        Example:
            >>> result = provider.classify(
            ...     "This product is amazing!",
            ...     ["positive", "negative", "neutral"]
            ... )
            >>> print(result)  # {'category': 'positive', 'confidence': 0.95, ...}
        """
        schema = {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": categories,
                    "description": "The primary category for the text"
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score between 0 and 1"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation for the classification"
                }
            },
            "required": ["category", "confidence"]
        }

        instruction = f"Classify the following text into one of these categories: {', '.join(categories)}"

        return self.extract_structured(
            text=text,
            schema=schema,
            schema_name="classification",
            instruction=instruction,
            model=model,
            **kwargs
        )

    def sentiment_analysis(
        self,
        text: str,
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze
            model: Model to use
            **kwargs: Additional parameters

        Returns:
            Dictionary with sentiment, score, and aspects

        Example:
            >>> result = provider.sentiment_analysis("I love the design but hate the price!")
            >>> print(result)  # {'sentiment': 'mixed', 'score': 0.2, 'aspects': [...]}
        """
        schema = {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral", "mixed"],
                    "description": "Overall sentiment of the text"
                },
                "score": {
                    "type": "number",
                    "description": "Sentiment score from -1 (negative) to 1 (positive)"
                },
                "aspects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "aspect": {"type": "string"},
                            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]}
                        },
                        "required": ["aspect", "sentiment"]
                    },
                    "description": "Specific aspects and their sentiment"
                }
            },
            "required": ["sentiment", "score"]
        }

        return self.extract_structured(
            text=text,
            schema=schema,
            schema_name="sentiment_analysis",
            instruction="Analyze the sentiment of this text",
            model=model,
            **kwargs
        )

    def extract_invoice(
        self,
        text: str,
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract invoice data from text (e.g., OCR output).

        Args:
            text: Invoice text or OCR result
            model: Model to use
            **kwargs: Additional parameters

        Returns:
            Dictionary with invoice details (number, dates, vendor, items, total)

        Example:
            >>> ocr_text = "Invoice #12345\\nDate: 2025-01-15\\nTotal: $599.99"
            >>> invoice = provider.extract_invoice(ocr_text)
            >>> print(invoice)  # {'invoice_number': '12345', 'total': 599.99, ...}
        """
        schema = {
            "type": "object",
            "properties": {
                "invoice_number": {"type": "string", "description": "Invoice number"},
                "date": {"type": "string", "format": "date", "description": "Invoice date"},
                "due_date": {"type": "string", "format": "date", "description": "Payment due date"},
                "vendor": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Vendor name"},
                        "address": {"type": "string", "description": "Vendor address"}
                    }
                },
                "total": {"type": "number", "description": "Total amount"},
                "currency": {"type": "string", "description": "Currency code (USD, EUR, SEK, etc.)"},
                "line_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "quantity": {"type": "integer"},
                            "unit_price": {"type": "number"},
                            "total": {"type": "number"}
                        }
                    }
                }
            },
            "required": ["invoice_number", "total"]
        }

        return self.extract_structured(
            text=text,
            schema=schema,
            schema_name="invoice_data",
            instruction="Extract invoice data from this text",
            model=model,
            **kwargs
        )

    def solve_math(
        self,
        problem: str,
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Solve a math problem with step-by-step solution.

        Args:
            problem: Math problem to solve
            model: Model to use
            **kwargs: Additional parameters

        Returns:
            Dictionary with problem_type, steps, final_answer, and verification

        Example:
            >>> result = provider.solve_math("Solve: 2x + 5 = 13")
            >>> print(result['final_answer'])  # 'x = 4'
            >>> for step in result['steps']:
            ...     print(step['explanation'])
        """
        schema = {
            "type": "object",
            "properties": {
                "problem_type": {
                    "type": "string",
                    "description": "Type of math problem (algebra, calculus, geometry, etc.)"
                },
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_number": {"type": "integer"},
                            "explanation": {"type": "string"},
                            "expression": {"type": "string"}
                        },
                        "required": ["step_number", "explanation"]
                    }
                },
                "final_answer": {"type": "string"},
                "verification": {"type": "string", "description": "Verification of the answer"}
            },
            "required": ["steps", "final_answer"]
        }

        messages = [{"role": "user", "content": f"Solve this math problem step by step:\n\n{problem}"}]

        return self.chat_with_schema(
            messages=messages,
            schema=schema,
            schema_name="math_solution",
            model=model,
            system="You are a math tutor. Solve problems step by step, showing all work.",
            **kwargs
        )

    def extract_from_schema_file(
        self,
        text: str,
        schema_file: str,
        instruction: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract data using a schema from a JSON file.

        Args:
            text: The input text/prompt
            schema_file: Path to JSON schema file
            instruction: Optional extraction instruction
            model: Model to use
            **kwargs: Additional parameters

        Returns:
            Extracted data as dictionary

        Example:
            >>> data = provider.extract_from_schema_file(
            ...     text=ocr_text,
            ...     schema_file="schemas/swedish_invoice.json",
            ...     instruction="Extrahera fakturadata"
            ... )
        """
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema_data = json.load(f)

        # Handle both full schema files and nested schemas
        if "schema" in schema_data:
            schema = schema_data["schema"]
            schema_name = schema_data.get("name", "custom_schema")
        else:
            schema = schema_data
            schema_name = Path(schema_file).stem

        return self.extract_structured(
            text=text,
            schema=schema,
            schema_name=schema_name,
            instruction=instruction,
            model=model,
            **kwargs
        )


def main():
    """Example usage of the Anthropic Structured Outputs Provider."""
    try:
        provider = AnthropicStructuredProvider()

        print("=== Anthropic Structured Outputs Provider ===")
        print(f"Model: {provider.model}")
        print()

        # Example 1: Extract person
        print("Example 1: Extract person information")
        person = provider.extract_person("John Smith, 30 years old, software engineer at Google")
        print(f"Result: {json.dumps(person, indent=2)}")
        print()

        # Example 2: Classification
        print("Example 2: Text classification")
        classification = provider.classify(
            text="This product exceeded my expectations!",
            categories=["positive", "negative", "neutral"]
        )
        print(f"Result: {json.dumps(classification, indent=2)}")
        print()

        # Example 3: Sentiment analysis
        print("Example 3: Sentiment analysis")
        sentiment = provider.sentiment_analysis(
            text="I love the design but the price is too high."
        )
        print(f"Result: {json.dumps(sentiment, indent=2)}")
        print()

        # Example 4: Custom schema
        print("Example 4: Custom schema")
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "summary": {"type": "string"},
                "key_points": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["title", "summary"]
        }

        result = provider.extract_structured(
            text="Article about climate change: Rising temperatures affect ecosystems worldwide.",
            schema=schema,
            instruction="Extract key information from this article preview"
        )
        print(f"Result: {json.dumps(result, indent=2)}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

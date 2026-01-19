"""
Anthropic Structured Outputs Provider
======================================

A standalone module for generating structured outputs from Claude using JSON Schema
validation. This module provides guaranteed schema compliance for data extraction,
classification, and structured generation tasks.

Features
--------
- JSON Schema-based structured outputs with validation
- Pydantic model support for type-safe responses
- Pre-built extractors for common use cases (person, invoice, entities, etc.)
- Schema validation helpers
- Swedish language support for DocFlow
- Self-contained, copy-paste ready

Quick Start
-----------
```python
from anthropic_structured import AnthropicStructuredProvider

# Initialize provider
provider = AnthropicStructuredProvider(api_key="your-api-key")

# Extract person information
person = provider.extract_person("John Smith, 30 years old, engineer")
# Returns: {'name': 'John Smith', 'age': 30, 'occupation': 'engineer'}

# Classify text
result = provider.classify(
    text="This product is amazing!",
    categories=["positive", "negative", "neutral"]
)
# Returns: {'category': 'positive', 'confidence': 0.95, 'reasoning': '...'}

# Use custom schema
schema = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "total": {"type": "number"}
    },
    "required": ["invoice_number", "total"]
}
data = provider.extract_with_schema(
    prompt="Extract invoice data: #12345, Total: $599.99",
    schema=schema
)

# Use Pydantic models
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    occupation: str

person = provider.chat_with_pydantic(
    messages=[{"role": "user", "content": "Extract: Jane Doe, 25, designer"}],
    pydantic_model=Person
)
# Returns: Person(name='Jane Doe', age=25, occupation='designer')
```

Pre-built Extractors
-------------------
- `extract_person()` - Extract person information (name, age, occupation, email)
- `extract_entities()` - Extract named entities (people, places, organizations, dates)
- `extract_invoice()` - Extract invoice data (number, dates, vendor, line items)
- `classify()` - Classify text into categories with confidence scores
- `sentiment_analysis()` - Analyze sentiment with scores and aspects
- `solve_math()` - Solve math problems with step-by-step solutions

Swedish Language Support
-----------------------
```python
# Extract municipal document information
schema = {
    "type": "object",
    "properties": {
        "dokumenttyp": {"type": "string", "enum": ["protokoll", "årsredovisning", "detaljplan"]},
        "kommun": {"type": "string"},
        "datum": {"type": "string", "format": "date"}
    },
    "required": ["dokumenttyp", "kommun"]
}

data = provider.extract_with_schema(
    prompt="Extrahera information från: Protokoll Stockholms kommun 2025-01-15",
    schema=schema,
    system="Du är en expert på svenska kommunala dokument."
)
```

Pydantic Integration
-------------------
```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date

class Invoice(BaseModel):
    invoice_number: str = Field(description="Fakturanummer")
    date: date = Field(description="Fakturadatum")
    total: float = Field(description="Totalsumma")
    vendor_name: Optional[str] = Field(None, description="Leverantörsnamn")

class MunicipalDocument(BaseModel):
    document_type: str = Field(description="Dokumenttyp")
    municipality: str = Field(description="Kommun")
    date: date = Field(description="Datum")
    reference: Optional[str] = Field(None, description="Ärendenummer")

# Extract using Pydantic
invoice = provider.chat_with_pydantic(
    messages=[{"role": "user", "content": ocr_text}],
    pydantic_model=Invoice
)
```

Schema Validation
----------------
```python
# Validate response against schema
is_valid, errors = provider.validate_response(response_data, schema)

if not is_valid:
    print(f"Validation errors: {errors}")
```

Supported JSON Schema Features
-----------------------------
- Basic types: object, array, string, integer, number, boolean, null
- enum, const
- anyOf, allOf
- $ref and $defs (internal only)
- String formats: date, date-time, email, uri, uuid, ipv4, ipv6
- required, additionalProperties
- Array minItems (0 or 1 only)

NOT Supported
-------------
- Recursive definitions
- External $ref URLs
- Numerical constraints (minimum, maximum)
- String length constraints (minLength, maxLength)
- Complex array constraints
- Regex backreferences

Advanced Usage
-------------
```python
# Load schema from file
data = provider.extract_from_schema_file(
    prompt="Extract data from this text...",
    schema_file="schemas/invoice.json"
)

# Batch processing with custom system prompt
results = []
for text in documents:
    result = provider.extract_with_schema(
        prompt=f"Extract information: {text}",
        schema=my_schema,
        system="You are a Swedish legal document expert.",
        temperature=0.1  # Lower temperature for consistency
    )
    results.append(result)

# Error handling
try:
    data = provider.extract_with_schema(prompt, schema)
except ValueError as e:
    if "refusal" in str(e):
        # Model refused the request
        pass
    elif "max_tokens" in str(e):
        # Response was truncated
        pass
```

Models
------
Default: claude-sonnet-4-5-20250929 (best for structured outputs)
Alternative: claude-opus-4-5-20250918 (maximum accuracy)
Fast: claude-haiku-4-5-20251001 (high-volume tasks)

Best Practices
-------------
1. Keep schemas simple and flat when possible
2. Use enums for fixed categories
3. Use temperature=0.1 for extraction tasks (consistency)
4. Validate responses even with structured outputs (defense in depth)
5. Handle refusal and max_tokens errors gracefully
6. Cache schemas for better performance (24-hour automatic cache)

Environment Variables
--------------------
ANTHROPIC_API_KEY - Your Anthropic API key (required)
ANTHROPIC_MODEL - Default model to use (optional, defaults to claude-sonnet-4-5-20250929)

Installation
-----------
pip install anthropic>=0.25.0 pydantic>=2.0.0 jsonschema>=4.0.0

License
-------
MIT License - Copy-paste ready for any project

References
----------
- Anthropic Structured Outputs: https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs
- JSON Schema: https://json-schema.org/
- Pydantic: https://docs.pydantic.dev/
"""

from .provider import AnthropicStructuredProvider
from .schemas import (
    PersonSchema,
    InvoiceSchema,
    EntitySchema,
    ClassificationSchema,
    SentimentSchema,
    MathSolutionSchema,
    pydantic_to_json_schema,
    validate_json_schema,
)

__version__ = "1.0.0"

__all__ = [
    'AnthropicStructuredProvider',
    # Schema helpers
    'PersonSchema',
    'InvoiceSchema',
    'EntitySchema',
    'ClassificationSchema',
    'SentimentSchema',
    'MathSolutionSchema',
    'pydantic_to_json_schema',
    'validate_json_schema',
]

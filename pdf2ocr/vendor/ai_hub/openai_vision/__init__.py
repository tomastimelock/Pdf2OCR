# Filepath: code_migration/ai_providers/openai_vision/__init__.py
# Description: OpenAI Vision API module - Self-contained module for image analysis with GPT-4o Vision
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/openai/vision/

"""
OpenAI Vision Provider Module
==============================

A complete, self-contained module for OpenAI Vision API (GPT-4o Vision) image analysis.

Features:
---------
- Single and multi-image analysis
- Image description generation
- OCR-like text extraction from images
- Question-answering about images
- Support for local files (auto base64 encoding) and URLs
- Multiple vision models (gpt-4o, gpt-4o-mini, gpt-4-turbo)
- Configurable detail levels (low, high, auto)

Quick Start:
------------
```python
from openai_vision import OpenAIVisionProvider

# Initialize provider
provider = OpenAIVisionProvider(api_key="your-api-key")

# Analyze an image
result = provider.analyze_image(
    image="path/to/image.jpg",
    prompt="What is in this image?"
)
print(result)

# Describe an image
description = provider.describe_image("https://example.com/image.jpg")

# Extract text (OCR)
text = provider.extract_text("document.png")

# Answer questions
answer = provider.answer_about_image(
    image="photo.jpg",
    question="How many people are in this photo?"
)

# Compare multiple images
comparison = provider.analyze_images(
    images=["before.jpg", "after.jpg"],
    prompt="What changed between these images?"
)
```

Models:
-------
- **gpt-4o** (default): Fast multimodal model with excellent vision capabilities
- **gpt-4o-mini**: More cost-effective for simpler vision tasks
- **gpt-4-turbo**: GPT-4 Turbo with vision support

Detail Levels:
--------------
- **auto** (default): Model chooses based on image size
- **low**: 512x512 fixed resolution, faster, ~85 tokens
- **high**: Detailed analysis with high-res tiles, ~129+ tokens

Input Formats:
--------------
- Local files: Automatically base64 encoded (PNG, JPG, JPEG, GIF, WebP)
- URLs: Direct HTTP/HTTPS image URLs
- Supports: .png, .jpg, .jpeg, .gif, .webp

Environment Variables:
---------------------
- OPENAI_API_KEY: Your OpenAI API key (required if not passed to constructor)
- OPENAI_VISION_MODEL: Default model to use (optional, defaults to gpt-4o)

API Reference:
--------------

### OpenAIVisionProvider(api_key=None, model=None)
Initialize the provider.
- api_key: OpenAI API key (or set OPENAI_API_KEY env var)
- model: Default model (defaults to "gpt-4o")

### analyze_image(image_path_or_url, prompt, model=None, detail="auto", **kwargs)
Analyze a single image with a custom prompt.
- image_path_or_url: Path to local file or HTTP(S) URL
- prompt: Question or instruction about the image
- model: Override default model
- detail: "low", "high", or "auto"
Returns: Analysis text (str)

### analyze_images(images, prompt, model=None, detail="auto", **kwargs)
Analyze multiple images together.
- images: List of image paths/URLs
- prompt: Question or instruction about the images
Returns: Analysis text (str)

### describe_image(image, detail_level="auto", model=None, **kwargs)
Generate a detailed description of an image.
- image: Path to local file or URL
- detail_level: "low", "high", or "auto"
Returns: Description text (str)

### extract_text(image, model=None, **kwargs)
Extract all visible text from an image (OCR-like).
- image: Path to local file or URL
Returns: Extracted text (str)

### answer_about_image(image, question, model=None, detail="auto", **kwargs)
Answer a specific question about an image.
- image: Path to local file or URL
- question: Question to answer
Returns: Answer text (str)

### list_models()
Get available vision models and their configurations.
Returns: Dict with model info

Examples:
---------

### Example 1: Analyze a local image
```python
provider = OpenAIVisionProvider()
result = provider.analyze_image(
    image="screenshot.png",
    prompt="Describe the UI elements in this screenshot"
)
```

### Example 2: Extract text from a document
```python
text = provider.extract_text("invoice.jpg")
print(text)
```

### Example 3: Compare two images
```python
comparison = provider.analyze_images(
    images=["product_old.jpg", "product_new.jpg"],
    prompt="What are the visual differences between these products?"
)
```

### Example 4: Analyze with high detail
```python
result = provider.analyze_image(
    image="detailed_diagram.png",
    prompt="Explain this technical diagram",
    detail="high"  # Use high-res tiles for better accuracy
)
```

### Example 5: URL-based image analysis
```python
result = provider.describe_image(
    image="https://example.com/photo.jpg",
    detail_level="auto"
)
```

### Example 6: Use different model
```python
provider = OpenAIVisionProvider(model="gpt-4o-mini")
# More cost-effective for simple tasks
result = provider.extract_text("simple_text.png")
```

Error Handling:
---------------
```python
try:
    result = provider.analyze_image("image.jpg", "What is this?")
except FileNotFoundError:
    print("Image file not found")
except ValueError:
    print("Invalid API key or configuration")
except Exception as e:
    print(f"API error: {e}")
```

Notes:
------
- Requires openai>=1.3.0 and Pillow>=10.0.0
- Images are automatically base64 encoded for local files
- Supports images up to 20MB
- API rate limits apply based on your OpenAI plan
- Vision API uses token-based pricing (varies by detail level)

Dependencies:
-------------
- openai (>=1.3.0)
- Pillow (>=10.0.0)
- python-dotenv (optional, for .env file support)
"""

from .provider import OpenAIVisionProvider
from .model_config import VISION_MODELS, get_vision_model_info
from .utils import (
    encode_image_to_base64,
    is_url,
    validate_image_path,
    get_image_mime_type
)

__version__ = "1.0.0"

__all__ = [
    "OpenAIVisionProvider",
    "VISION_MODELS",
    "get_vision_model_info",
    "encode_image_to_base64",
    "is_url",
    "validate_image_path",
    "get_image_mime_type",
]

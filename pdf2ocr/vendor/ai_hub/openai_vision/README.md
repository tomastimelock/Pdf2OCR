# OpenAI Vision Provider

A complete, self-contained Python module for OpenAI Vision API (GPT-4o Vision) image analysis.

## Features

- ✅ Single and multi-image analysis
- ✅ Detailed image descriptions
- ✅ OCR-like text extraction
- ✅ Question-answering about images
- ✅ Image comparison
- ✅ Local files (auto base64 encoding) and URLs
- ✅ Multiple vision models (gpt-4o, gpt-4o-mini, gpt-4-turbo)
- ✅ Configurable detail levels (low, high, auto)
- ✅ Comprehensive type hints and docstrings

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- `openai>=1.3.0`
- `Pillow>=10.0.0`
- `python-dotenv>=1.0.0` (optional)

## Quick Start

```python
from openai_vision import OpenAIVisionProvider

# Initialize provider
provider = OpenAIVisionProvider(api_key="sk-...")

# Analyze an image
result = provider.analyze_image(
    image_path_or_url="photo.jpg",
    prompt="What is in this image?"
)
print(result)
```

## Configuration

### Environment Variables

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_VISION_MODEL="gpt-4o"  # Optional, defaults to gpt-4o
```

Or use a `.env` file:

```env
OPENAI_API_KEY=sk-...
OPENAI_VISION_MODEL=gpt-4o
```

### Initialization Options

```python
# Using environment variable
provider = OpenAIVisionProvider()

# Explicit API key
provider = OpenAIVisionProvider(api_key="sk-...")

# Custom model
provider = OpenAIVisionProvider(
    api_key="sk-...",
    model="gpt-4o-mini"
)

# Custom default detail level
provider = OpenAIVisionProvider(
    api_key="sk-...",
    default_detail="high"
)
```

## Usage Examples

### 1. Analyze a Single Image

```python
provider = OpenAIVisionProvider()

# Local file
result = provider.analyze_image(
    image_path_or_url="screenshot.png",
    prompt="Describe the UI elements in this screenshot"
)

# URL
result = provider.analyze_image(
    image_path_or_url="https://example.com/photo.jpg",
    prompt="What's happening in this photo?"
)

# With custom settings
result = provider.analyze_image(
    image_path_or_url="diagram.png",
    prompt="Explain this technical diagram",
    detail="high",  # Use high-res tiles
    model="gpt-4o",
    max_tokens=2048
)
```

### 2. Describe an Image

```python
description = provider.describe_image(
    image="landscape.jpg",
    detail_level="auto"
)
print(description)
# Output: "This image shows a stunning mountain landscape at sunset..."
```

### 3. Extract Text (OCR)

```python
# Extract all visible text
text = provider.extract_text("document.png")
print(text)

# With custom model
text = provider.extract_text(
    image="invoice.jpg",
    model="gpt-4o"
)
```

### 4. Answer Questions About Images

```python
answer = provider.answer_about_image(
    image="crowd.jpg",
    question="How many people are visible in this photo?"
)
print(answer)

answer = provider.answer_about_image(
    image="chart.png",
    question="What is the trend shown in this chart?",
    detail="high"
)
```

### 5. Compare Images

```python
# Compare two images
comparison = provider.compare_images(
    image1="version1.png",
    image2="version2.png",
    aspect="UI layout"
)

# General comparison
comparison = provider.compare_images(
    image1="before.jpg",
    image2="after.jpg"
)
```

### 6. Analyze Multiple Images

```python
# Analyze multiple images together
analysis = provider.analyze_images(
    images=["img1.jpg", "img2.jpg", "img3.jpg"],
    prompt="What's the common theme across these images?"
)

# Compare a series
analysis = provider.analyze_images(
    images=["step1.png", "step2.png", "step3.png"],
    prompt="Describe the progression shown in these steps"
)
```

## Available Models

```python
# List all available models
models = OpenAIVisionProvider.list_models()
for name, info in models.items():
    print(f"{name}: {info['description']}")
```

### Model Comparison

| Model | Speed | Quality | Cost | Use Case |
|-------|-------|---------|------|----------|
| `gpt-4o` | Fast | High | Medium | General purpose (recommended) |
| `gpt-4o-mini` | Very Fast | Good | Low | Batch processing, simple tasks |
| `gpt-4-turbo` | Medium | Very High | High | Complex analysis |

### Detail Levels

| Level | Resolution | Tokens | Speed | Use Case |
|-------|-----------|--------|-------|----------|
| `auto` | Automatic | Varies | Medium | General (recommended) |
| `low` | 512x512 | ~85 | Fast | Simple images, lower cost |
| `high` | Tiled | ~129+ | Slow | OCR, detailed analysis |

## API Reference

### OpenAIVisionProvider

#### `__init__(api_key=None, model=None, default_detail="auto")`

Initialize the provider.

**Parameters:**
- `api_key` (str, optional): OpenAI API key or use `OPENAI_API_KEY` env var
- `model` (str, optional): Default model (defaults to "gpt-4o")
- `default_detail` (str, optional): Default detail level (defaults to "auto")

#### `analyze_image(image_path_or_url, prompt, model=None, detail="auto", max_tokens=1024, **kwargs)`

Analyze a single image with a custom prompt.

**Parameters:**
- `image_path_or_url` (str): Path to local file or URL
- `prompt` (str): Question or instruction
- `model` (str, optional): Override default model
- `detail` (str, optional): Detail level ("low", "high", "auto")
- `max_tokens` (int, optional): Max response tokens
- `**kwargs`: Additional API parameters

**Returns:** str - Analysis text

#### `analyze_images(images, prompt, model=None, detail="auto", max_tokens=1024, **kwargs)`

Analyze multiple images together.

**Parameters:**
- `images` (List[str]): List of image paths/URLs
- `prompt` (str): Question or instruction
- Other params same as `analyze_image`

**Returns:** str - Analysis text

#### `describe_image(image, detail_level="auto", model=None, max_tokens=1024, **kwargs)`

Generate a detailed description.

**Parameters:**
- `image` (str): Path to local file or URL
- `detail_level` (str, optional): Detail level
- Other params same as `analyze_image`

**Returns:** str - Description text

#### `extract_text(image, model=None, max_tokens=2048, **kwargs)`

Extract all visible text (OCR).

**Parameters:**
- `image` (str): Path to local file or URL
- `model` (str, optional): Override default model
- `max_tokens` (int, optional): Max response tokens

**Returns:** str - Extracted text

#### `answer_about_image(image, question, model=None, detail="auto", max_tokens=1024, **kwargs)`

Answer a specific question.

**Parameters:**
- `image` (str): Path to local file or URL
- `question` (str): Question to answer
- Other params same as `analyze_image`

**Returns:** str - Answer text

#### `compare_images(image1, image2, aspect=None, model=None, detail="auto", max_tokens=1024, **kwargs)`

Compare two images.

**Parameters:**
- `image1` (str): Path to first image or URL
- `image2` (str): Path to second image or URL
- `aspect` (str, optional): Specific aspect to compare
- Other params same as `analyze_image`

**Returns:** str - Comparison analysis

#### `list_models()` (static)

List all available vision models.

**Returns:** Dict[str, Any] - Model configurations

#### `get_model_info(model_name)` (static)

Get info about a specific model.

**Parameters:**
- `model_name` (str): Model name

**Returns:** Dict[str, Any] or None

## Utility Functions

```python
from openai_vision.utils import (
    is_url,
    validate_image_path,
    get_image_mime_type,
    encode_image_to_base64,
    get_image_dimensions,
    get_image_info
)

# Check if string is URL
if is_url("https://example.com/image.jpg"):
    print("It's a URL")

# Validate image
validate_image_path("photo.jpg")  # Raises error if invalid

# Get MIME type
mime = get_image_mime_type("photo.jpg")  # "image/jpeg"

# Encode to base64
encoded = encode_image_to_base64("photo.jpg")

# Get dimensions
width, height = get_image_dimensions("photo.jpg")

# Get comprehensive info
info = get_image_info("photo.jpg")
print(info)
# {
#     'path': 'photo.jpg',
#     'name': 'photo.jpg',
#     'format': '.jpg',
#     'mime_type': 'image/jpeg',
#     'dimensions': (1920, 1080),
#     'width': 1920,
#     'height': 1080,
#     'file_size': 2560000,
#     'file_size_formatted': '2.44 MB',
#     'within_limits': True
# }
```

## Error Handling

```python
from openai_vision import OpenAIVisionProvider

try:
    provider = OpenAIVisionProvider()
    result = provider.analyze_image("photo.jpg", "What is this?")
    print(result)

except FileNotFoundError as e:
    print(f"Image not found: {e}")

except ValueError as e:
    print(f"Invalid configuration or format: {e}")

except Exception as e:
    print(f"API error: {e}")
```

## Supported Formats

- PNG (`.png`)
- JPEG (`.jpg`, `.jpeg`)
- GIF (`.gif`)
- WebP (`.webp`)

## Limits

- **Max file size:** 20 MB
- **Max dimension:** 4096 pixels
- **Recommended max:** 2048 pixels

## Best Practices

### 1. Choose the Right Model

```python
# General use
provider = OpenAIVisionProvider(model="gpt-4o")

# Cost-effective batch processing
provider = OpenAIVisionProvider(model="gpt-4o-mini")

# Complex reasoning
provider = OpenAIVisionProvider(model="gpt-4-turbo")
```

### 2. Use Appropriate Detail Levels

```python
# OCR - use high detail
text = provider.extract_text("document.png")  # Auto uses "high"

# Simple tasks - use low detail
result = provider.analyze_image(
    "icon.png",
    "What color is this icon?",
    detail="low"  # Faster and cheaper
)

# Let the model decide
result = provider.analyze_image(
    "photo.jpg",
    "Describe this photo",
    detail="auto"  # Default
)
```

### 3. Batch Processing

```python
import os

provider = OpenAIVisionProvider(model="gpt-4o-mini")  # Cost-effective

results = []
for filename in os.listdir("images/"):
    if filename.endswith(('.jpg', '.png')):
        result = provider.describe_image(
            f"images/{filename}",
            detail_level="low"  # Fast processing
        )
        results.append({
            'file': filename,
            'description': result
        })
```

### 4. Structured Prompts

```python
# Good: Specific, clear instructions
result = provider.analyze_image(
    "chart.png",
    "Identify: 1) Chart type, 2) X-axis label, 3) Y-axis label, 4) Main trend"
)

# Better: Request structured output
result = provider.analyze_image(
    "chart.png",
    """Analyze this chart and provide:
    - Chart type: [type]
    - X-axis: [label]
    - Y-axis: [label]
    - Trend: [description]
    """
)
```

## Module Structure

```
openai_vision/
├── __init__.py          # Module exports and documentation
├── provider.py          # OpenAIVisionProvider class
├── model_config.py      # Model configurations
├── utils.py             # Utility functions
├── requirements.txt     # Dependencies
├── README.md           # This file
└── examples.py         # Usage examples
```

## Integration with DocFlow

This module is designed for the DocFlow document processing pipeline:

```python
# In DocFlow extraction pipeline
from openai_vision import OpenAIVisionProvider

vision_provider = OpenAIVisionProvider()

# Extract text from scanned document
extracted_text = vision_provider.extract_text(
    image="scanned_document.png"
)

# Classify document type
doc_type = vision_provider.analyze_image(
    image="document.png",
    prompt="Is this a: invoice, receipt, contract, or letter?"
)

# Extract structured data
data = vision_provider.analyze_image(
    image="invoice.png",
    prompt="Extract: invoice number, date, total amount, vendor name",
    detail="high"
)
```

## License

This module is part of the DocFlow project. See main project for license details.

## References

- Reference implementation: `reference_codebase/AIMOS/providers/openai/vision/`
- OpenAI Vision API: https://platform.openai.com/docs/guides/vision

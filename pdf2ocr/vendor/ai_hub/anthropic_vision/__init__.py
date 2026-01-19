# Filepath: code_migration/ai_providers/anthropic_vision/__init__.py
# Description: Anthropic Vision module - Image analysis with Claude
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/Anthropic/vision/

"""
Anthropic Vision Provider
==========================

Self-contained module for image analysis using Claude's vision capabilities.

Features
--------
- Single and multiple image analysis
- OCR-like text extraction from images
- Image comparison
- Object detection
- Document and chart analysis
- Streaming support
- Token estimation

Supported Models
----------------
- claude-opus-4-5-20250918 (Maximum intelligence)
- claude-sonnet-4-5-20250929 (Recommended default)
- claude-haiku-4-5-20251001 (Fastest)
- claude-sonnet-4-20250514 (Previous gen)
- claude-opus-4-1-20250805 (Previous gen)
- claude-3-7-sonnet-20250219 (Legacy)

Supported Image Formats
-----------------------
- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- WebP (.webp)

Image Size Limits
-----------------
- Max dimensions: 8000x8000 pixels (2000x2000 for 20+ images)
- Optimal: 1568 pixels max dimension (1.15 megapixels)
- Max file size: 5MB per image (API), 10MB (claude.ai)
- Max request size: 32MB
- Max images: 100 per request (20 recommended for claude.ai)

Token Calculation
-----------------
tokens = (width × height) / 750
Example: 1092×1092 ≈ 1,590 tokens

Quick Start
-----------
```python
from anthropic_vision import AnthropicVisionProvider

# Initialize with API key
provider = AnthropicVisionProvider(api_key="your-api-key")

# Or use environment variable ANTHROPIC_API_KEY
provider = AnthropicVisionProvider()

# Analyze single image
result = provider.analyze_image(
    image="path/to/image.jpg",
    prompt="What's in this image?"
)
print(result)

# Extract text (OCR)
text = provider.extract_text("document.png")
print(text)

# Compare two images
comparison = provider.compare_images(
    image1="photo1.jpg",
    image2="photo2.jpg"
)
print(comparison)

# Analyze multiple images
analysis = provider.analyze_images(
    images=["img1.jpg", "img2.jpg", "img3.jpg"],
    prompt="Describe the sequence of events"
)
print(analysis)

# Describe image with detail level
description = provider.describe_image(
    image="photo.jpg",
    detail_level="comprehensive"  # brief, detailed, comprehensive
)
print(description)

# Detect objects
objects = provider.detect_objects("scene.jpg")
print(objects)

# Analyze chart/graph
chart_analysis = provider.analyze_chart("chart.png")
print(chart_analysis)

# Analyze document
doc_analysis = provider.analyze_document("scan.jpg")
print(doc_analysis)

# Estimate token cost
token_info = provider.estimate_tokens("large_image.jpg")
print(f"Estimated tokens: {token_info['estimated_tokens']}")
print(f"Optimal size: {token_info['optimal']}")
```

Advanced Usage
--------------

### Using URLs
```python
# Analyze image from URL
result = provider.analyze_image(
    image="https://example.com/image.jpg",
    prompt="Describe this image"
)
```

### Custom model selection
```python
# Use faster model
result = provider.analyze_image(
    image="photo.jpg",
    prompt="Quick description",
    model="claude-haiku-4-5-20251001"
)

# Use most powerful model
result = provider.analyze_image(
    image="complex_scene.jpg",
    prompt="Detailed analysis",
    model="claude-opus-4-5-20250918"
)
```

### System prompts
```python
result = provider.analyze_image(
    image="medical_scan.jpg",
    prompt="Identify any abnormalities",
    system="You are an expert radiologist. Provide detailed medical analysis."
)
```

### Streaming responses (for long analyses)
```python
# Stream response for real-time output
for chunk in provider.analyze_image_stream(
    image="complex_diagram.jpg",
    prompt="Explain this diagram in detail"
):
    print(chunk, end="", flush=True)
```

### Batch processing
```python
# Process multiple images efficiently
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = []

for img in images:
    result = provider.analyze_image(
        image=img,
        prompt="Describe this image",
        model="claude-haiku-4-5-20251001"  # Fast model for batch
    )
    results.append(result)
```

Swedish Language Support
------------------------
```python
# Swedish prompts work seamlessly
result = provider.analyze_image(
    image="dokument.jpg",
    prompt="Extrahera all text från detta dokument",
    system="Du är en expert på svenska dokument. Bevara originalformatering."
)

# OCR for Swedish documents
text = provider.extract_text(
    image="protokoll.pdf",
    language="swedish"  # Hint for better accuracy
)
```

Error Handling
--------------
```python
from anthropic_vision import AnthropicVisionProvider
from anthropic_vision.provider import VisionError

try:
    provider = AnthropicVisionProvider()
    result = provider.analyze_image(
        image="photo.jpg",
        prompt="Describe this"
    )
except VisionError as e:
    print(f"Vision error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

Environment Variables
---------------------
```bash
# Set API key
export ANTHROPIC_API_KEY="your-api-key"

# Set default model (optional)
export ANTHROPIC_MODEL="claude-sonnet-4-5-20250929"
```

Token Estimation
----------------
```python
# Check token cost before making API call
info = provider.estimate_tokens("large_image.jpg")
print(f"Width: {info['width']}px")
print(f"Height: {info['height']}px")
print(f"Megapixels: {info['megapixels']:.2f}")
print(f"Estimated tokens: {info['estimated_tokens']}")
print(f"Optimal size: {info['optimal']}")

# Resize if needed
if not info['optimal']:
    print("Consider resizing to 1568x1568 or smaller for optimal cost")
```

Best Practices
--------------

1. **Image Size**: Resize large images to ~1568px max dimension for optimal cost/quality
2. **Format**: Use JPEG for photos (smaller), PNG for text/diagrams (clarity)
3. **Multiple Images**: Place all images before text in prompt for best results
4. **Model Selection**:
   - Use Haiku for simple descriptions
   - Use Sonnet (default) for balanced quality
   - Use Opus for complex analysis requiring maximum intelligence
5. **Prompts**: Be specific about what you want to extract/analyze
6. **OCR**: For pure text extraction, use extract_text() method
7. **Streaming**: Use streaming for long analyses to see results in real-time

Module Structure
----------------
- provider.py - Main AnthropicVisionProvider class
- model_config.py - Model configurations and capabilities
- requirements.txt - Package dependencies

Dependencies
------------
- anthropic>=0.25.0
- Pillow>=10.0.0
- python-dotenv>=1.0.0

License
-------
Self-contained, copy-paste ready module for DocFlow project.

Notes
-----
- This module is completely self-contained with no external project dependencies
- Uses relative imports only
- All configuration is embedded in model_config.py
- Can be copied to any Python project and used immediately
"""

from .provider import AnthropicVisionProvider, VisionError
from .model_config import (
    get_vision_models,
    get_default_vision_model,
    supports_vision,
    calculate_image_tokens,
    VISION_CONFIG
)

__all__ = [
    'AnthropicVisionProvider',
    'VisionError',
    'get_vision_models',
    'get_default_vision_model',
    'supports_vision',
    'calculate_image_tokens',
    'VISION_CONFIG'
]

__version__ = '1.0.0'
__author__ = 'DocFlow Project'
__description__ = 'Anthropic Vision Provider - Image analysis with Claude'

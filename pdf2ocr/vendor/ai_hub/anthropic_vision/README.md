# Anthropic Vision Provider

Self-contained module for image analysis using Claude's vision capabilities.

## Features

- **Single & Multiple Image Analysis** - Analyze one or many images with natural language prompts
- **OCR Text Extraction** - Extract text from images with high accuracy
- **Image Comparison** - Compare two or more images and describe differences
- **Object Detection** - Identify and describe objects in images
- **Document Analysis** - Analyze document structure, extract key information
- **Chart/Graph Analysis** - Understand data visualizations
- **Streaming Support** - Real-time response streaming for long analyses
- **Token Estimation** - Calculate costs before making API calls
- **Swedish Language Support** - Full support for Swedish prompts and documents

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install anthropic>=0.25.0 Pillow>=10.0.0 python-dotenv>=1.0.0
```

## Quick Start

### Setup

```python
# Set API key as environment variable
export ANTHROPIC_API_KEY="your-api-key-here"

# Or pass directly to provider
from anthropic_vision import AnthropicVisionProvider

provider = AnthropicVisionProvider(api_key="your-api-key")
```

### Basic Usage

```python
from anthropic_vision import AnthropicVisionProvider

# Initialize
provider = AnthropicVisionProvider()

# Analyze an image
result = provider.analyze_image(
    image="photo.jpg",
    prompt="What's in this image?"
)
print(result)

# Extract text (OCR)
text = provider.extract_text("document.png")
print(text)

# Compare two images
comparison = provider.compare_images(
    image1="before.jpg",
    image2="after.jpg"
)
print(comparison)
```

## Supported Models

### Current Generation (Recommended)

- **claude-sonnet-4-5-20250929** (Default) - Best balance of quality and speed
- **claude-opus-4-5-20250918** - Maximum intelligence for complex analysis
- **claude-haiku-4-5-20251001** - Fastest for high-volume processing

### Previous Generation

- **claude-sonnet-4-20250514** - Previous gen balanced model
- **claude-opus-4-1-20250805** - Previous gen high intelligence

### Legacy (Deprecated)

- **claude-3-7-sonnet-20250219** - Being deprecated Feb 19, 2026

## Supported Image Formats

- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- GIF (`.gif`)
- WebP (`.webp`)

## Image Size Limits

| Limit | Value |
|-------|-------|
| Max dimensions | 8000×8000 px (2000×2000 for 20+ images) |
| Optimal dimension | 1568 px max |
| Max file size (API) | 5 MB |
| Max file size (claude.ai) | 10 MB |
| Max request size | 32 MB |
| Max images per request | 100 (20 recommended) |

## Token Calculation

```
tokens = (width × height) / 750
```

Example:
- 1092×1092 image ≈ 1,590 tokens
- 1568×1568 image ≈ 3,277 tokens

## Usage Examples

### Analyze Single Image

```python
from anthropic_vision import AnthropicVisionProvider

provider = AnthropicVisionProvider()

# Basic analysis
result = provider.analyze_image(
    image="photo.jpg",
    prompt="Describe this image in detail"
)

# With specific model
result = provider.analyze_image(
    image="complex_scene.jpg",
    prompt="What's happening here?",
    model="claude-opus-4-5-20250918"  # Use most powerful model
)

# With system prompt
result = provider.analyze_image(
    image="medical_scan.jpg",
    prompt="Identify any abnormalities",
    system="You are an expert radiologist. Provide detailed medical analysis."
)
```

### Analyze Multiple Images

```python
# Analyze sequence
result = provider.analyze_images(
    images=["step1.jpg", "step2.jpg", "step3.jpg"],
    prompt="Describe the sequence of events shown in these images"
)

# Compare multiple images
result = provider.analyze_images(
    images=["option_a.jpg", "option_b.jpg", "option_c.jpg"],
    prompt="Compare and contrast these design options"
)
```

### Extract Text (OCR)

```python
# English text
text = provider.extract_text("english_doc.png")

# Swedish text
text = provider.extract_text(
    image="svenskt_dokument.pdf",
    language="swedish"
)

# With specific instructions
text = provider.extract_text(
    image="form.jpg",
    prompt="Extract all text, preserving the table structure"
)
```

### Compare Images

```python
# Basic comparison
comparison = provider.compare_images(
    image1="before.jpg",
    image2="after.jpg"
)

# Custom comparison prompt
comparison = provider.compare_images(
    image1="design_v1.png",
    image2="design_v2.png",
    prompt="List the specific design changes between these versions"
)
```

### Describe Image with Detail Levels

```python
# Brief description (one sentence)
desc = provider.describe_image("scene.jpg", detail_level="brief")

# Detailed description (default)
desc = provider.describe_image("scene.jpg", detail_level="detailed")

# Comprehensive analysis
desc = provider.describe_image("scene.jpg", detail_level="comprehensive")
```

### Detect Objects

```python
objects = provider.detect_objects("scene.jpg")
# Returns: List of objects with descriptions and locations
```

### Analyze Charts and Graphs

```python
analysis = provider.analyze_chart("sales_chart.png")
# Returns: Chart type, trends, key insights
```

### Analyze Documents

```python
analysis = provider.analyze_document("contract.pdf")
# Returns: Document type, summary, key information, tables, notable elements
```

### Streaming Responses

```python
# Stream for real-time output
for chunk in provider.analyze_image_stream(
    image="complex_diagram.jpg",
    prompt="Explain this diagram in detail"
):
    print(chunk, end="", flush=True)
print()  # New line at end
```

### Estimate Token Cost

```python
# Check cost before API call
info = provider.estimate_tokens("large_image.jpg")

print(f"Dimensions: {info['width']}×{info['height']}")
print(f"Megapixels: {info['megapixels']:.2f}")
print(f"Estimated tokens: {info['estimated_tokens']}")
print(f"Optimal size: {info['optimal']}")
print(f"Recommendation: {info['recommendation']}")
```

### Use with URLs

```python
# Analyze image from URL (no download needed)
result = provider.analyze_image(
    image="https://example.com/photo.jpg",
    prompt="What's in this image?"
)
```

## Swedish Language Examples

```python
from anthropic_vision import AnthropicVisionProvider

provider = AnthropicVisionProvider()

# Swedish prompt
resultat = provider.analyze_image(
    image="bild.jpg",
    prompt="Beskriv denna bild i detalj"
)

# Extract Swedish text
text = provider.extract_text(
    image="protokoll.pdf",
    language="swedish"
)

# Analyze Swedish document
analys = provider.analyze_document(
    image="årsredovisning.pdf",
    system="Du är en expert på svenska företagsdokument."
)
```

## Model Selection Guide

Choose the right model for your task:

| Task | Recommended Model | Reason |
|------|------------------|--------|
| Simple descriptions | claude-haiku-4-5-20251001 | Fast, economical |
| General analysis | claude-sonnet-4-5-20250929 | Best balance |
| Complex reasoning | claude-opus-4-5-20250918 | Maximum intelligence |
| OCR/Text extraction | claude-sonnet-4-5-20250929 | Good accuracy |
| Chart analysis | claude-opus-4-5-20250918 | Best data understanding |
| Batch processing | claude-haiku-4-5-20251001 | Speed & cost efficient |

## Error Handling

```python
from anthropic_vision import AnthropicVisionProvider, VisionError

try:
    provider = AnthropicVisionProvider()
    result = provider.analyze_image("photo.jpg", "Describe this")

except VisionError as e:
    print(f"Vision error: {e}")

except FileNotFoundError:
    print("Image file not found")

except Exception as e:
    print(f"Unexpected error: {e}")
```

## Best Practices

### 1. Image Size Optimization

```python
# Check if image needs resizing
info = provider.estimate_tokens("large_image.jpg")

if not info['optimal']:
    # Resize to optimal dimensions before processing
    from PIL import Image

    img = Image.open("large_image.jpg")
    img.thumbnail((1568, 1568))
    img.save("optimized_image.jpg")

    result = provider.analyze_image("optimized_image.jpg", "Describe this")
```

### 2. Format Selection

- Use **JPEG** for photographs (smaller file size)
- Use **PNG** for text, diagrams, screenshots (better clarity)

### 3. Batch Processing

```python
# Process multiple images efficiently
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = []

for img in images:
    result = provider.analyze_image(
        image=img,
        prompt="Quick description",
        model="claude-haiku-4-5-20251001",  # Fast model
        max_tokens=200  # Shorter responses
    )
    results.append(result)
```

### 4. Prompt Engineering

```python
# Good: Specific and clear
result = provider.analyze_image(
    image="product.jpg",
    prompt="List all visible product features and their conditions"
)

# Better: With context and format
result = provider.analyze_image(
    image="product.jpg",
    prompt="""Analyze this product image and provide:
    1. Product type and model
    2. Visible features
    3. Condition assessment
    4. Notable defects or damage
    Format as a structured list."""
)
```

### 5. Temperature Control

```python
# For factual extraction (OCR, data extraction) - use low temperature
text = provider.extract_text("document.png", temperature=0.1)

# For creative descriptions - use higher temperature
desc = provider.describe_image("art.jpg", temperature=0.9)
```

## Module Structure

```
anthropic_vision/
├── __init__.py          # Module exports and documentation
├── provider.py          # AnthropicVisionProvider class
├── model_config.py      # Model configurations
├── requirements.txt     # Dependencies
├── README.md           # This file
└── examples.py         # Usage examples
```

## Dependencies

- **anthropic** ≥ 0.25.0 - Anthropic Python SDK
- **Pillow** ≥ 10.0.0 - Image processing and token estimation
- **python-dotenv** ≥ 1.0.0 - Environment variable management (optional)

## Environment Variables

```bash
# Required
export ANTHROPIC_API_KEY="your-api-key"

# Optional
export ANTHROPIC_MODEL="claude-sonnet-4-5-20250929"
```

## API Reference

### AnthropicVisionProvider

Main class for vision operations.

#### Methods

- `analyze_image(image, prompt, **kwargs)` - Analyze single image
- `analyze_images(images, prompt, **kwargs)` - Analyze multiple images
- `compare_images(image1, image2, **kwargs)` - Compare two images
- `describe_image(image, detail_level, **kwargs)` - Get image description
- `extract_text(image, **kwargs)` - Extract text (OCR)
- `detect_objects(image, **kwargs)` - Detect objects in image
- `analyze_chart(image, **kwargs)` - Analyze chart/graph
- `analyze_document(image, **kwargs)` - Analyze document
- `analyze_image_stream(image, prompt, **kwargs)` - Streaming analysis
- `estimate_tokens(image_path)` - Estimate token cost
- `get_model_info()` - Get current model information

### Model Configuration Functions

- `get_vision_models()` - Get all vision-capable models
- `get_default_vision_model()` - Get default model name
- `supports_vision(model_name)` - Check if model supports vision
- `calculate_image_tokens(width, height)` - Calculate token cost

## License

Self-contained module for DocFlow project. Copy-paste ready with no external dependencies beyond listed packages.

## Support

For issues or questions, refer to:
- [Anthropic API Documentation](https://docs.anthropic.com)
- [Claude Vision Guide](https://docs.anthropic.com/en/docs/vision)

## Version

1.0.0 - Initial release

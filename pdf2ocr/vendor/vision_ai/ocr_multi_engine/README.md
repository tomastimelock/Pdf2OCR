# OCR Multi-Engine Module

A standalone, self-contained OCR extraction library with intelligent multi-engine fallback support.

## Features

- **Multi-Engine OCR**: Tesseract (local), OpenAI Vision (GPT-4o), Mistral OCR
- **Automatic Fallback**: Quality-based engine selection with configurable thresholds
- **Document Processing**: Unified interface for images and PDFs
- **Quality Scoring**: Automatic text quality assessment (0.0-1.0)
- **Statistics Tracking**: Comprehensive processing metrics
- **Swedish Language Support**: Optimized for Swedish documents (å, ä, ö)
- **Self-Contained**: Copy-pasteable module with relative imports

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

**Required:**
- `pytesseract>=0.3.10` - Tesseract OCR wrapper
- `Pillow>=10.0.0` - Image processing
- `PyMuPDF>=1.23.0` - PDF text extraction

**Optional (for cloud OCR):**
- `openai>=1.0.0` - OpenAI Vision API
- `mistralai>=0.0.7` - Mistral OCR API

**System Requirements:**
- Tesseract OCR (for local processing): https://github.com/tesseract-ocr/tesseract

## Quick Start

### Basic Usage

```python
from ocr_multi_engine import create_default_ocr, create_document_processor

# Create multi-engine OCR (tries OpenAI → Mistral → Tesseract)
ocr = create_default_ocr()

# Extract text from image
text = ocr.extract_text("document.jpg")
print(text)

# Process any document (image or PDF)
processor = create_document_processor()
text = processor.process("document.pdf")
print(text)
```

### Swedish Language Support

```python
from ocr_multi_engine import OCRConfig, OCRProviderFactory, DocumentProcessor

# Configure for Swedish
config = OCRConfig(
    tesseract_lang="swe+eng",  # Swedish + English
    engine_order=["openai", "mistral", "tesseract"],
    min_quality=0.7
)

# Create multi-engine OCR
ocr = OCRProviderFactory.create_multi_engine(
    config=config.to_dict(),
    engine_order=config.engine_order
)

# Process Swedish documents
processor = DocumentProcessor(ocr)
text = processor.process("årsredovisning.pdf")
```

### Individual Engine Usage

```python
from ocr_multi_engine import OCRProviderFactory

# OpenAI Vision (highest quality)
openai_ocr = OCRProviderFactory.create("openai", config={
    "openai_api_key": "sk-...",
    "openai_model": "gpt-4o"
})

# Mistral OCR (good for handwriting)
mistral_ocr = OCRProviderFactory.create("mistral", config={
    "mistral_api_key": "your-key"
})

# Tesseract (local, offline)
tesseract_ocr = OCRProviderFactory.create("tesseract", config={
    "tesseract_lang": "swe+eng"
})

text = openai_ocr.extract_text("document.jpg")
```

## Configuration

### Environment Variables

```bash
# API Keys
export OPENAI_API_KEY="sk-..."
export MISTRAL_API_KEY="your-key"

# OCR Settings
export OCR_ENGINE_ORDER="openai,mistral,tesseract"
export OCR_MIN_QUALITY="0.7"
export OCR_TESSERACT_LANG="swe+eng"
export OCR_DEFAULT_ENGINE="openai"
```

### Programmatic Configuration

```python
from ocr_multi_engine import OCRConfig, set_config

config = OCRConfig(
    openai_api_key="sk-...",
    mistral_api_key="your-key",
    openai_model="gpt-4o",
    tesseract_lang="swe+eng",
    engine_order=["openai", "mistral", "tesseract"],
    min_quality=0.7,
    fallback_enabled=True
)

# Validate configuration
errors = config.validate()
if errors:
    print("Configuration errors:", errors)

# Set as global config
set_config(config)
```

## OCR Engine Comparison

| Engine | Quality Score | Speed | Cost | Best For |
|--------|--------------|-------|------|----------|
| **OpenAI Vision** | 0.90 | Medium | $$ | Highest accuracy, complex layouts |
| **Mistral OCR** | 0.85 | Fast | $ | Handwriting, mixed content |
| **Tesseract** | 0.60 | Very Fast | Free | Offline, bulk processing |

## Quality Scoring

The module automatically scores extracted text quality (0.0-1.0 scale):

```python
text = ocr.extract_text("document.jpg")
quality = ocr.get_quality_score(text)

if quality >= 0.9:
    print("Excellent quality")
elif quality >= 0.7:
    print("Good quality")
elif quality >= 0.5:
    print("Acceptable quality")
else:
    print("Poor quality - may need manual review")
```

**Quality factors:**
- Text length and word count
- Character distribution (alpha/digit/special ratio)
- OCR artifact detection (|||, ###, ???)
- Average word length
- Structured content indicators

## Statistics Tracking

```python
from ocr_multi_engine import create_document_processor

processor = create_document_processor()

# Process multiple documents
for doc in documents:
    text = processor.process(doc)

# Get statistics
stats = processor.get_stats()
print(f"Total processed: {stats['total_processed']}")
print(f"Success rate: {stats['success_rate']}%")
print(f"By processor: {stats['by_processor']}")

# Reset stats
processor.reset_stats()
```

## Document Processors

### ImageProcessor

Processes image files using OCR:

```python
from ocr_multi_engine import ImageProcessor, OCRProviderFactory

ocr = OCRProviderFactory.create_multi_engine()
processor = ImageProcessor(ocr)

# Supported formats
supported = processor.get_supported_extensions()
# ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp']

text = processor.process("scan.jpg")
```

### PDFProcessor

Processes PDFs with direct text extraction + OCR fallback:

```python
from ocr_multi_engine import PDFProcessor, OCRProviderFactory

ocr = OCRProviderFactory.create_multi_engine()
processor = PDFProcessor(ocr, config={
    "min_text_length": 100,  # Min chars for direct extraction
    "use_ocr_fallback": True,  # Enable OCR for scanned PDFs
    "ocr_dpi": 200  # DPI for OCR conversion
})

text = processor.process("document.pdf")
```

### DocumentProcessor

Unified processor for all document types:

```python
from ocr_multi_engine import DocumentProcessor, OCRProviderFactory

ocr = OCRProviderFactory.create_multi_engine()
processor = DocumentProcessor(ocr)

# Automatically detects file type and uses appropriate processor
text = processor.process("document.pdf")  # Uses PDFProcessor
text = processor.process("scan.jpg")      # Uses ImageProcessor

# Check supported formats
print(processor.get_supported_extensions())
# ['.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp']
```

## Error Handling

```python
from ocr_multi_engine import (
    OCRError,
    OCRProviderUnavailableError,
    ProcessingError,
    UnsupportedFileTypeError
)

try:
    text = processor.process("document.pdf")
except UnsupportedFileTypeError as e:
    print(f"Unsupported file type: {e}")
except ProcessingError as e:
    print(f"Processing failed: {e}")
except OCRError as e:
    print(f"OCR error: {e}")
```

## Advanced Usage

### Custom OCR Prompt (OpenAI Vision)

```python
from ocr_multi_engine import OpenAIVision

custom_prompt = """
Extract all Swedish text from this municipal report.
Pay special attention to:
- Legal references (e.g., "1 kap. 1 § Regeringsformen")
- Dates in YYYY-MM-DD format
- Budget figures and amounts
- Administrative terminology
"""

ocr = OpenAIVision(
    api_key="sk-...",
    model="gpt-4o",
    prompt=custom_prompt
)

text = ocr.extract_text("årsredovisning.pdf")
```

### Multi-Engine with Custom Order

```python
from ocr_multi_engine import OCRProviderFactory

# Try Mistral first (better for handwriting), then others
ocr = OCRProviderFactory.create_multi_engine(
    engine_order=["mistral", "openai", "tesseract"],
    config={
        "mistral_api_key": "your-key",
        "openai_api_key": "sk-...",
        "tesseract_lang": "swe+eng",
        "min_quality": 0.75  # Higher threshold
    }
)
```

### Image Preprocessing

```python
from ocr_multi_engine import enhance_image_for_ocr, convert_to_rgb
from PIL import Image

# Load image
image = Image.open("scan.jpg")

# Convert to RGB
image = convert_to_rgb(image)

# Enhance for better OCR
image = enhance_image_for_ocr(
    image,
    contrast=1.5,
    sharpness=1.5,
    brightness=1.1
)

# Save enhanced image
image.save("scan_enhanced.jpg")
```

## Module Structure

```
ocr_multi_engine/
├── __init__.py           # Public API and convenience functions
├── base.py               # BaseOCRProvider abstract class
├── config.py             # OCRConfig dataclass
├── factory.py            # OCRProviderFactory and MultiEngineOCR
├── processors.py         # Document processors (Image, PDF, unified)
├── utils.py              # Image preprocessing utilities
├── requirements.txt      # Dependencies
├── engines/
│   ├── __init__.py
│   ├── tesseract.py      # Tesseract OCR implementation
│   ├── openai_vision.py  # OpenAI Vision API
│   └── mistral.py        # Mistral OCR API
└── README.md             # This file
```

## Integration with DocFlow

This module is designed to integrate seamlessly with the DocFlow pipeline:

```python
# In DocFlow extractor nodes
from ocr_multi_engine import create_document_processor

class OCRExtractorNode:
    def __init__(self):
        self.processor = create_document_processor()

    def process(self, input_data):
        file_path = input_data['file_path']
        text = self.processor.process(file_path)

        return {
            'text': text,
            'stats': self.processor.get_stats()
        }
```

## Testing

```python
# Run basic tests
from ocr_multi_engine import OCRProviderFactory

# Check available engines
providers = OCRProviderFactory.list_available_providers()
print("Available providers:", providers)

# Test each engine
for provider_name in providers:
    try:
        ocr = OCRProviderFactory.create(provider_name)
        if ocr.is_available():
            print(f"{provider_name}: Available")
        else:
            print(f"{provider_name}: Not configured")
    except Exception as e:
        print(f"{provider_name}: {e}")
```

## License

Part of the DocumentHandler/DocFlow project.

## References

Extracted from: `reference_codebase/OCR_extractor/provider/`

Based on production-tested OCR extraction patterns with multi-engine fallback.

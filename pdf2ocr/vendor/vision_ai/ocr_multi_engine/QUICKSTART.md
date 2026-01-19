# OCR Multi-Engine - Quick Start Guide

Get started with the OCR Multi-Engine module in 5 minutes.

## Installation

```bash
cd code_migration/extraction/ocr_multi_engine
pip install -r requirements.txt
```

## System Requirements

Install Tesseract OCR (for offline processing):
- **Windows**: https://github.com/UB-Mannheim/tesseract/wiki
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

For Swedish language support:
```bash
# Linux/macOS
sudo apt-get install tesseract-ocr-swe
# or
brew install tesseract-lang
```

## Setup API Keys

Create a `.env` file or set environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export MISTRAL_API_KEY="your-key"
export OCR_ENGINE_ORDER="openai,mistral,tesseract"
export OCR_TESSERACT_LANG="swe+eng"
```

## Basic Usage (3 lines)

```python
from ocr_multi_engine import create_document_processor

processor = create_document_processor()
text = processor.process("document.pdf")
print(text)
```

## Multi-Engine with Fallback

```python
from ocr_multi_engine import create_default_ocr

# Tries engines in order: OpenAI → Mistral → Tesseract
ocr = create_default_ocr()

# Extract text from image
text = ocr.extract_text("scan.jpg")

# Check quality
quality = ocr.get_quality_score(text)
print(f"Quality: {quality:.2f}")
```

## Swedish Documents

```python
from ocr_multi_engine import OCRConfig, OCRProviderFactory

config = OCRConfig(
    tesseract_lang="swe+eng",
    engine_order=["openai", "mistral", "tesseract"]
)

ocr = OCRProviderFactory.create_multi_engine(
    config=config.to_dict(),
    engine_order=config.engine_order
)

text = ocr.extract_text("årsredovisning.jpg")
```

## Single Engine

### OpenAI Vision (Highest Quality)
```python
from ocr_multi_engine import OCRProviderFactory

ocr = OCRProviderFactory.create("openai", config={
    "openai_api_key": "sk-...",
    "openai_model": "gpt-4o"
})

text = ocr.extract_text("document.jpg")
```

### Tesseract (Offline)
```python
from ocr_multi_engine import OCRProviderFactory

ocr = OCRProviderFactory.create("tesseract", config={
    "tesseract_lang": "swe+eng"
})

text = ocr.extract_text("scan.png")
```

### Mistral OCR (Handwriting)
```python
from ocr_multi_engine import OCRProviderFactory

ocr = OCRProviderFactory.create("mistral", config={
    "mistral_api_key": "your-key"
})

text = ocr.extract_text("handwritten.jpg")
```

## Processing PDFs

```python
from ocr_multi_engine import PDFProcessor, OCRProviderFactory

ocr = OCRProviderFactory.create_multi_engine()
processor = PDFProcessor(ocr, config={
    "min_text_length": 100,    # Use OCR if direct extraction < 100 chars
    "use_ocr_fallback": True,  # Enable OCR for scanned PDFs
    "ocr_dpi": 200             # DPI for OCR conversion
})

# Tries text extraction first, falls back to OCR
text = processor.process("document.pdf")
```

## Batch Processing

```python
from ocr_multi_engine import create_document_processor
from pathlib import Path

processor = create_document_processor()

# Process all images in directory
for image_file in Path("documents/").glob("*.jpg"):
    text = processor.process(image_file)
    print(f"{image_file.name}: {len(text)} characters")

# Get statistics
stats = processor.get_stats()
print(f"Success rate: {stats['success_rate']}%")
```

## Custom Configuration

```python
from ocr_multi_engine import OCRConfig, OCRProviderFactory

config = OCRConfig(
    openai_api_key="sk-...",
    mistral_api_key="your-key",
    openai_model="gpt-4o",
    tesseract_lang="swe+eng",
    engine_order=["openai", "mistral", "tesseract"],
    min_quality=0.75,  # Higher threshold
    fallback_enabled=True
)

# Validate configuration
errors = config.validate()
if errors:
    print("Errors:", errors)
else:
    ocr = OCRProviderFactory.create_multi_engine(
        config=config.to_dict(),
        engine_order=config.engine_order
    )
```

## Error Handling

```python
from ocr_multi_engine import (
    create_document_processor,
    OCRError,
    ProcessingError,
    UnsupportedFileTypeError
)

processor = create_document_processor()

try:
    text = processor.process("document.pdf")
except UnsupportedFileTypeError:
    print("File type not supported")
except ProcessingError as e:
    print(f"Processing failed: {e}")
except OCRError as e:
    print(f"OCR error: {e}")
```

## Statistics Tracking

```python
from ocr_multi_engine import create_document_processor

processor = create_document_processor()

# Process documents
for doc in documents:
    text = processor.process(doc)

# Get detailed statistics
stats = processor.get_stats()
print(f"""
Total processed: {stats['total_processed']}
Successful: {stats['successful']}
Failed: {stats['failed']}
Success rate: {stats['success_rate']}%

By processor:
  Images: {stats['by_processor']['image']}
  PDFs: {stats['by_processor']['pdf']}
""")
```

## Common Use Cases

### 1. Municipal Documents (Swedish)
```python
from ocr_multi_engine import create_default_ocr

ocr = create_default_ocr({
    "tesseract_lang": "swe+eng",
    "engine_order": ["openai", "tesseract"]
})

text = ocr.extract_text("protokoll.pdf")
```

### 2. Handwritten Notes
```python
from ocr_multi_engine import OCRProviderFactory

# Mistral is best for handwriting
ocr = OCRProviderFactory.create("mistral", config={
    "mistral_api_key": "your-key"
})

text = ocr.extract_text("handwritten_note.jpg")
```

### 3. Scanned PDFs
```python
from ocr_multi_engine import PDFProcessor, OCRProviderFactory

ocr = OCRProviderFactory.create_multi_engine()
processor = PDFProcessor(ocr, config={
    "use_ocr_fallback": True,
    "ocr_dpi": 300  # Higher DPI for better quality
})

text = processor.process("scanned_document.pdf")
```

### 4. Bulk Processing (Offline)
```python
from ocr_multi_engine import OCRProviderFactory, DocumentProcessor

# Use only Tesseract (no API costs)
ocr = OCRProviderFactory.create("tesseract", config={
    "tesseract_lang": "eng"
})

processor = DocumentProcessor(ocr)

for doc in documents:
    text = processor.process(doc)
```

## Testing Your Setup

```python
from ocr_multi_engine import OCRProviderFactory

# Check which engines are available
for provider in ["tesseract", "openai", "mistral"]:
    try:
        ocr = OCRProviderFactory.create(provider)
        if ocr.is_available():
            print(f"✅ {provider}: Available")
        else:
            print(f"❌ {provider}: Not configured")
    except Exception as e:
        print(f"❌ {provider}: {e}")
```

## Next Steps

1. **Read the full documentation**: `README.md`
2. **Browse examples**: `EXAMPLES.py` (12 examples)
3. **Check the manifest**: `MODULE_MANIFEST.md`
4. **See migration notes**: `MIGRATION_SUMMARY.md`

## Troubleshooting

### "pytesseract not available"
Install pytesseract and Tesseract:
```bash
pip install pytesseract
# Then install Tesseract system package (see Installation above)
```

### "OpenAI library not available"
```bash
pip install openai
export OPENAI_API_KEY="sk-..."
```

### "Mistral client not available"
```bash
pip install mistralai
export MISTRAL_API_KEY="your-key"
```

### "No OCR engines available"
At least one engine must be configured. Install Tesseract for offline processing:
```bash
# Linux
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

## Support

For issues or questions:
1. Check `README.md` for detailed documentation
2. Review `EXAMPLES.py` for usage patterns
3. See `MODULE_MANIFEST.md` for complete specification

---

**Happy OCR processing!** 🚀

# OCR Multi-Engine Module - Manifest

**Version:** 1.0.0
**Created:** 2025-12-21
**Source:** reference_codebase/OCR_extractor/provider/
**Type:** Standalone, Self-Contained Module
**Layer:** Extractor

## Module Overview

Complete OCR extraction library with multi-engine fallback support, extracted and adapted from the OCR_extractor reference codebase.

## File Structure

```
ocr_multi_engine/
├── __init__.py              # Public API, exports, convenience functions
├── base.py                  # BaseOCRProvider abstract class
├── config.py                # OCRConfig dataclass with validation
├── factory.py               # OCRProviderFactory + MultiEngineOCR
├── processors.py            # Document processors (Image, PDF, unified)
├── utils.py                 # Image preprocessing utilities
├── requirements.txt         # Python dependencies
├── README.md                # Comprehensive documentation
├── EXAMPLES.py              # 12 usage examples
├── MODULE_MANIFEST.md       # This file
└── engines/
    ├── __init__.py          # Engine exports
    ├── tesseract.py         # Tesseract OCR (local)
    ├── openai_vision.py     # OpenAI Vision API (cloud)
    └── mistral.py           # Mistral OCR API (cloud)
```

## Core Components

### 1. Base Classes (base.py)

- **BaseOCRProvider**: Abstract base class for all OCR engines
  - Methods: `extract_text()`, `get_provider_name()`, `is_available()`, `get_quality_score()`, `get_stats()`, `reset_stats()`
  - Stats tracking: total_processed, successful, failed, total_characters_extracted

- **Exceptions**:
  - `OCRError`: Base OCR exception
  - `OCRProviderUnavailableError`: Provider not available/configured

### 2. Configuration (config.py)

- **OCRConfig**: Dataclass for configuration
  - API keys: mistral_api_key, openai_api_key
  - OpenAI settings: model, custom prompt
  - Tesseract settings: language
  - Engine settings: default_engine, engine_order, min_quality, fallback_enabled
  - Methods: `from_env()`, `to_dict()`, `validate()`

- **Global Config Functions**:
  - `get_config()`: Get singleton instance
  - `set_config()`: Set global config
  - `reset_config()`: Reset to environment defaults

### 3. Factory & Multi-Engine (factory.py)

- **OCRProviderFactory**: Creates OCR provider instances
  - `create(provider_name, logger, config)`: Create single provider
  - `create_multi_engine(logger, config, engine_order)`: Create multi-engine OCR
  - `list_available_providers()`: List all provider types

- **MultiEngineOCR**: Multi-engine with quality-based fallback
  - Tries engines in order until quality threshold met
  - Tracks best result across all engines
  - Returns best result if all below threshold

### 4. Document Processors (processors.py)

- **BaseProcessor**: Abstract processor class
  - Methods: `process()`, `can_process()`, `get_supported_extensions()`, `get_stats()`, `reset_stats()`

- **ImageProcessor**: Process images with OCR
  - Supported: JPG, JPEG, PNG, BMP, TIFF, TIF, GIF, WebP

- **PDFProcessor**: Process PDFs with text extraction + OCR fallback
  - Direct extraction via PyMuPDF
  - OCR fallback for scanned PDFs
  - Configurable DPI for OCR conversion

- **DocumentProcessor**: Unified processor for all document types
  - Auto-detects file type
  - Routes to appropriate processor
  - Combined statistics

- **Exceptions**:
  - `ProcessingError`: Processing failed
  - `UnsupportedFileTypeError`: File type not supported

### 5. OCR Engines (engines/)

#### TesseractOCR (engines/tesseract.py)
- **Type**: Local OCR engine
- **Quality Score**: 0.6 (base)
- **Dependencies**: pytesseract, Pillow
- **Features**: Image enhancement (contrast, sharpness, brightness)
- **Best For**: Offline processing, bulk operations

#### OpenAIVision (engines/openai_vision.py)
- **Type**: Cloud OCR (GPT-4 Vision)
- **Quality Score**: 0.9 (base)
- **Dependencies**: openai
- **Features**: Default prompt for structured extraction, custom prompts
- **Best For**: Highest accuracy, complex layouts

#### MistralOCR (engines/mistral.py)
- **Type**: Cloud OCR
- **Quality Score**: 0.85 (base)
- **Dependencies**: mistralai (optional), requests
- **Features**: Official client + direct API fallback
- **Best For**: Handwriting, mixed content

### 6. Utilities (utils.py)

- **Image Processing**:
  - `enhance_image_for_ocr()`: Enhance contrast, sharpness, brightness
  - `convert_to_rgb()`: Convert to RGB mode
  - `get_image_mime_type()`: Get MIME type from extension

- **File Type Detection**:
  - `is_image_file()`: Check if image
  - `is_pdf_file()`: Check if PDF
  - `get_file_type()`: Get type (image/pdf/office/text/unknown)

- **Quality Assessment**:
  - `calculate_text_quality()`: Comprehensive quality metrics
  - Returns: (quality_score, metrics_dict)

## Public API

### Exports from __init__.py

```python
# Base
BaseOCRProvider
OCRError
OCRProviderUnavailableError

# Config
OCRConfig
get_config()
set_config()
reset_config()

# Factory
OCRProviderFactory
MultiEngineOCR

# Processors
DocumentProcessor
ImageProcessor
PDFProcessor
ProcessingError
UnsupportedFileTypeError

# Engines
TesseractOCR
OpenAIVision
MistralOCR

# Utils
enhance_image_for_ocr()
convert_to_rgb()
get_image_mime_type()
is_image_file()
is_pdf_file()
get_file_type()
calculate_text_quality()
```

### Convenience Functions

```python
create_default_ocr(config_dict=None)
create_document_processor(config_dict=None)
```

## Dependencies

### Required
- `pytesseract>=0.3.10` - Tesseract wrapper
- `Pillow>=10.0.0` - Image processing
- `PyMuPDF>=1.23.0` - PDF processing

### Optional
- `openai>=1.0.0` - OpenAI Vision API
- `mistralai>=0.0.7` - Mistral OCR API
- `requests>=2.31.0` - Direct API calls

### System
- Tesseract OCR (for local OCR)

## Quality Scoring

### Score Ranges
- **0.9-1.0**: Excellent (OpenAI Vision typical)
- **0.8-0.9**: Very Good (Mistral OCR typical)
- **0.7-0.8**: Good (acceptable threshold)
- **0.6-0.7**: Fair (Tesseract typical)
- **0.0-0.6**: Poor (manual review recommended)

### Quality Factors
- Text length and word count
- Average word length (2-15 chars)
- Character distribution (alpha/digit/special ratios)
- OCR artifact detection (|||, ###, ???, etc.)
- Structured content indicators (paragraphs, lists, labels)

## Engine Priority Order

### Default Order
1. **Mistral** (0.85 base quality) - Good balance
2. **OpenAI** (0.90 base quality) - Highest accuracy
3. **Tesseract** (0.60 base quality) - Offline fallback

### Recommended Orders

**Highest Accuracy:**
```python
["openai", "mistral", "tesseract"]
```

**Cost Optimization:**
```python
["tesseract", "mistral", "openai"]
```

**Handwriting Focus:**
```python
["mistral", "openai", "tesseract"]
```

**Offline Only:**
```python
["tesseract"]
```

## Swedish Language Support

### Configuration
```python
OCRConfig(
    tesseract_lang="swe+eng",  # Swedish + English
    engine_order=["openai", "mistral", "tesseract"]
)
```

### Supported Characters
- å, ä, ö, Å, Ä, Ö (preserved by all engines)

### Best Engines for Swedish
1. **OpenAI Vision** - Excellent Swedish support
2. **Mistral OCR** - Good Swedish support
3. **Tesseract** - Good with swe+eng language pack

## Integration Patterns

### DocFlow Pipeline Node
```python
class OCRExtractorNode(BaseNode):
    def __init__(self):
        self.processor = create_document_processor()

    def process(self, input_data):
        text = self.processor.process(input_data['file_path'])
        return {'text': text, 'stats': self.processor.get_stats()}
```

### Batch Processing
```python
processor = create_document_processor()
for file in files:
    text = processor.process(file)
stats = processor.get_stats()
```

### Custom Engine Configuration
```python
ocr = OCRProviderFactory.create("openai", config={
    "openai_api_key": "sk-...",
    "openai_model": "gpt-4o",
    "openai_prompt": "Custom extraction prompt..."
})
```

## Design Patterns

### Self-Contained
- All imports are relative
- No external dependencies on DocFlow
- Copy-pasteable to other projects

### Factory Pattern
- Dynamic provider creation
- Runtime engine selection
- Configuration-driven instantiation

### Strategy Pattern
- Swappable OCR engines
- Unified interface
- Quality-based selection

### Adapter Pattern
- Unified OCR interface for different APIs
- Consistent error handling
- Standardized quality scoring

## Testing Checklist

- [ ] Tesseract availability check
- [ ] OpenAI API key validation
- [ ] Mistral API key validation
- [ ] Image file processing
- [ ] PDF text extraction
- [ ] PDF OCR fallback
- [ ] Multi-engine fallback logic
- [ ] Quality scoring accuracy
- [ ] Swedish character preservation
- [ ] Statistics tracking
- [ ] Error handling
- [ ] Configuration validation

## Migration Notes

### Changes from Reference Code
1. **Relative Imports**: Changed all imports to relative (.)
2. **Self-Contained**: Removed external DocFlow dependencies
3. **Enhanced Documentation**: Added comprehensive docstrings
4. **Examples**: Created 12 usage examples
5. **Manifest**: Added this documentation file

### Preserved Features
- All OCR engine implementations
- Multi-engine fallback logic
- Quality scoring algorithms
- PDF processing with OCR fallback
- Statistics tracking
- Swedish language support

## Performance Characteristics

### Speed Comparison (approximate)
- **Tesseract**: 0.5-2s per page
- **OpenAI Vision**: 2-5s per page (API latency)
- **Mistral OCR**: 1-3s per page (API latency)

### Accuracy Comparison
- **OpenAI Vision**: 95-99% (best for complex layouts)
- **Mistral OCR**: 90-95% (excellent for handwriting)
- **Tesseract**: 70-85% (depends on image quality)

## Future Enhancements

Potential additions:
- [ ] EasyOCR engine support
- [ ] Batch processing optimization
- [ ] Async/await support
- [ ] Result caching
- [ ] Language detection
- [ ] Table extraction
- [ ] Layout analysis
- [ ] Confidence scores per word
- [ ] Custom quality metrics

## References

**Source Files:**
- reference_codebase/OCR_extractor/provider/base.py
- reference_codebase/OCR_extractor/provider/config.py
- reference_codebase/OCR_extractor/provider/factory.py
- reference_codebase/OCR_extractor/provider/processors.py
- reference_codebase/OCR_extractor/provider/tesseract.py
- reference_codebase/OCR_extractor/provider/openai_vision.py
- reference_codebase/OCR_extractor/provider/mistral.py
- reference_codebase/OCR_extractor/provider/utils.py

**Related Modules:**
- pdf_toolkit/ - PDF processing utilities
- image_iterator/ - Image metadata extraction
- forensic/ - Case management patterns

## Version History

### 1.0.0 (2025-12-21)
- Initial extraction from reference codebase
- Complete standalone module
- All 3 OCR engines implemented
- Full documentation and examples
- Swedish language support
- Multi-engine fallback
- Quality scoring
- Statistics tracking

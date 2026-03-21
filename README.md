# PDF2OCR

An advanced PDF document processing package that extracts, analyzes, and regenerates content from PDF documents using multiple AI services. Transform static PDFs into searchable, structured, AI-enhanced documents with extracted tables, regenerated charts, and improved images.

## Features

### Core Processing
- **PDF Splitting** - Convert PDF pages to high-quality images (JPG/PNG) with configurable DPI
- **Metadata Extraction** - Extract document metadata (title, author, creation date)
- **Embedded Image Extraction** - Extract all images embedded in the PDF

### OCR (Text Extraction)
- **Multi-Engine Support** - Mistral AI (primary), OpenAI Vision, and Tesseract OCR
- **Auto-Fallback Mode** - Automatic fallback with quality thresholds
- **Quality Scoring** - Automatic quality assessment of extracted text
- **Batch Processing** - Process single PDFs or entire directories

### Information Extraction
- Key-value pairs (e.g., "Name: John Doe")
- Lists (bulleted and numbered)
- Tables in markdown format
- Email addresses and URLs
- Phone numbers, dates, and currency amounts

### Table Extraction
- Table detection using pdfplumber
- Export to JSON format with headers and rows
- SVG visualization of tables

### Chart Detection and Regeneration
- Detect charts using OpenAI GPT-4o Vision
- Regenerate charts as clean SVG using Anthropic Claude
- Quality validation with retry logic

### Image Regeneration
- Regenerate embedded images using OpenAI DALL-E
- Quality-based filtering
- Support for various image types

### Export Capabilities
- **Word Export (.docx)** - Formatted documents with text, tables, images, and SVGs
- **PDF Export** - Processed PDFs with extracted content
- **Text Export** - Combined text from all pages
- **JSON Export** - Structured document representation

## Installation

### Basic Installation

```bash
pip install -e "."
```

### Install with Specific Features

```bash
pip install -e ".[charts]"      # Chart regeneration (OpenAI + Anthropic)
pip install -e ".[images]"      # Image regeneration (DALL-E)
pip install -e ".[export]"      # Word/PDF export
pip install -e ".[tesseract]"   # Tesseract OCR fallback
pip install -e ".[all]"         # All features
```

### System Requirements

- Python 3.9+ (tested on 3.9-3.12)
- **Windows**: Cairo DLL included (`libcairo-2.dll`) for SVG operations
- **Linux/macOS**: Install Cairo via package manager if using SVG features

## Configuration

### API Keys

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_key        # Required for OCR
OPENAI_API_KEY=your_openai_key          # Optional: chart detection, image regeneration
ANTHROPIC_API_KEY=your_anthropic_key    # Optional: SVG chart generation
```

### Windows Cairo Setup

For Windows users, if you encounter Cairo-related errors:

```env
CAIROCFFI_DLL_DIRECTORIES=C:\path\to\pdf2ocr
```

## Usage

### Command-Line Interface

```bash
# Full pipeline - run all processing steps
pdf2ocr input.pdf -o output/ --full

# Split PDF to images only
pdf2ocr input.pdf --split-only

# OCR only with information extraction
pdf2ocr input.pdf --ocr-only --extract-info

# Extract tables and charts
pdf2ocr input.pdf --extract-tables --extract-charts

# Export to Word and PDF
pdf2ocr input.pdf --export-word output.docx --export-pdf output.pdf

# Batch processing - process all PDFs in a folder
pdf2ocr /path/to/pdf/folder -o output/

# Custom DPI and format
pdf2ocr input.pdf --dpi 300 --format png -o output/
```

#### CLI Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Output directory |
| `--dpi` | Resolution for image conversion (default: 200) |
| `--format` | Image format: jpg or png (default: jpg) |
| `--split-only` | Only split PDF to images |
| `--ocr-only` | Only run OCR on existing images |
| `--extract-images` | Extract embedded images from PDF |
| `--extract-info` | Extract structured information from text |
| `--extract-tables` | Extract tables to JSON and SVG |
| `--extract-charts` | Detect and regenerate charts |
| `--regenerate-images` | Regenerate images with DALL-E |
| `--structure-output` | Create structured JSON output |
| `--export-word` | Export to Word document |
| `--export-pdf` | Export to PDF document |
| `--full` | Run complete pipeline |

### Programmatic API

```python
from pdf2ocr import PDF2OCR

# Initialize processor
processor = PDF2OCR(
    api_key="your_mistral_api_key",
    dpi=200,
    image_format="jpg"
)

# Process a PDF
result = processor.process(
    pdf_path="document.pdf",
    output_dir="output/",
    extract_info=True,
    extract_images=True
)

# Access results
print(result.combined_text)
print(result.extracted_info.emails)
print(result.extracted_info.key_value_pairs)
```

### Full Pipeline Script

```bash
python run_full_pipeline.py
```

Edit `run_full_pipeline.py` to configure:
- `PDF_PATH` - Input PDF file
- `OUTPUT_DIR` - Output directory
- `DPI` - Image resolution (default: 200)

### Multi-Engine OCR

```python
from pdf2ocr.processors import OCRProcessor

# Auto-fallback mode
processor = OCRProcessor(
    api_key="your_key",
    engine="auto",
    engine_order=["mistral", "openai", "tesseract"],
    quality_threshold=0.7
)

# Process with automatic fallback
result = processor.process_image("page.jpg")
print(f"Text: {result.text}")
print(f"Quality: {result.quality_score}")
print(f"Engine used: {result.provider}")
```

## Output Structure

```
output/
└── document_name/
    ├── pages/              # Page images (JPG/PNG)
    ├── txt/                # OCR text per page
    ├── images/             # Extracted embedded images
    ├── json/               # Extracted tables (JSON)
    ├── svg/                # Charts and tables (SVG)
    ├── regenerated/        # AI-regenerated images
    ├── combined.txt        # Full document text
    ├── extracted_info.json # Structured information
    ├── summary.txt         # Information summary
    ├── document.json       # Complete structured output
    ├── document.docx       # Word export
    └── document_processed.pdf # PDF export
```

## Architecture

```
Input PDF
    │
    ▼
[PDFSplitter] ─────────────────► Page Images
    │
    ▼
[OCRProcessor] ────────────────► Extracted Text
    │
    ├──► [InformationExtractor] ► Key-values, emails, dates
    ├──► [TableExtractor] ──────► Tables (JSON & SVG)
    ├──► [ChartRegenerator] ────► Charts (SVG)
    └──► [ImageRegenerator] ────► Enhanced Images
    │
    ▼
[DocumentStructurer] ──────────► Unified JSON
    │
    ▼
[Exporters] ───────────────────► DOCX, PDF, TXT
```

### Project Structure

```
pdf2ocr/
├── api.py                  # High-level API (PDF2OCR class)
├── cli.py                  # Command-line interface
├── processors/             # Core processing modules
│   ├── pdf_splitter.py     # PDF to image conversion
│   ├── ocr_processor.py    # OCR orchestration
│   ├── chart_regenerator.py # Chart detection & SVG
│   ├── image_regenerator.py # DALL-E image regeneration
│   └── ...
├── providers/              # LLM/OCR service providers
│   ├── mistral_provider.py # Mistral AI
│   ├── openai_provider.py  # OpenAI
│   ├── anthropic_provider.py # Anthropic
│   └── ...
├── extractors/             # Information extraction
│   ├── information_extractor.py
│   ├── table_extractor.py
│   └── document_structurer.py
├── exporters/              # Document export
│   ├── word_exporter.py
│   └── pdf_exporter.py
└── utils/                  # Utilities
    └── svg_validator.py
```

## Use Cases

- **Scanned Documents** - Convert scanned PDFs to searchable, editable documents
- **Data Extraction** - Extract structured data (tables, key-values, emails) from PDFs
- **Document Digitization** - Convert paper documents to digital with enhanced quality
- **Batch Processing** - Process entire PDF libraries automatically
- **Report Generation** - Generate Word/PDF reports from processed content
- **Chart Modernization** - Regenerate blurry/low-quality charts as clean SVG
- **Image Enhancement** - Improve embedded images using AI

## Requirements

| Dependency | Purpose |
|------------|---------|
| PyMuPDF | PDF reading and page-to-image conversion |
| Pillow | Image processing |
| mistralai | Mistral AI for OCR |
| pdfplumber | Table extraction |
| python-dotenv | Environment variable management |
| openai | Chart detection, image regeneration (optional) |
| anthropic | SVG generation (optional) |
| python-docx | Word export (optional) |
| reportlab | PDF export (optional) |
| cairosvg | SVG to PNG conversion (optional) |
| pytesseract | Tesseract OCR fallback (optional) |

## Troubleshooting

### Cairo DLL not found (Windows)

Ensure the `libcairo-2.dll` is in the project directory or set the environment variable:

```env
CAIROCFFI_DLL_DIRECTORIES=C:\path\to\dll\directory
```

### Low OCR Quality

- Increase DPI (try 300 instead of 200)
- Use PNG format instead of JPG for better quality
- Enable multi-engine fallback mode

### API Rate Limits

- Add delays between requests for batch processing
- Use lower DPI to reduce image sizes
- Process in smaller batches

## Documentation

### Additional Guides

| Document | Description |
|----------|-------------|
| [DigitalOcean Claude Module](docs/DIGITALOCEAN_CLAUDE_MODULE.md) | Guide for creating Claude Code integrations with MCP servers, skills, and sub-agents |

## License

MIT

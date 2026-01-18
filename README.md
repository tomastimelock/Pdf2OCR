# PDF2OCR

PDF to OCR processing package with Mistral AI integration. Extracts text, tables, charts, and images from PDF documents with AI-powered regeneration capabilities.

## Features

- **PDF Splitting** - Convert PDF pages to images
- **OCR Processing** - Extract text using Mistral AI vision
- **Table Extraction** - Detect and extract tables to JSON/SVG
- **Chart Detection** - Identify charts and regenerate as clean SVG
- **Image Regeneration** - Recreate images using OpenAI DALL-E
- **Document Structuring** - Create structured JSON output
- **Export** - Generate Word (.docx) and PDF documents

## Installation

```bash
pip install -e ".[all]"
```

Or install with specific features:

```bash
pip install -e "."              # Core only
pip install -e ".[charts]"      # + Chart regeneration
pip install -e ".[images]"      # + Image regeneration
pip install -e ".[export]"      # + Word/PDF export
```

## Configuration

Create a `.env` file with your API keys:

```env
MISTRAL_API_KEY=your_mistral_key
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

## Usage

### Full Pipeline

```bash
python run_full_pipeline.py
```

Edit `run_full_pipeline.py` to configure:
- `PDF_PATH` - Input PDF file
- `OUTPUT_DIR` - Output directory
- `DPI` - Image resolution (default: 200)

### Programmatic API

```python
from pdf2ocr import PDF2OCR

processor = PDF2OCR(
    mistral_api_key="your_key",
    openai_api_key="your_key"  # Optional
)

result = processor.process(
    pdf_path="document.pdf",
    output_dir="output/"
)
```

## Output Structure

```
output/
  document_name/
    pages/          # Page images
    txt/            # OCR text per page
    combined.txt    # Full document text
    json/           # Extracted tables
    svg/            # Charts and tables as SVG
    regenerated/    # AI-regenerated images
    document.json   # Structured data
    document.docx   # Word export
    document.pdf    # PDF export
```

## Requirements

- Python 3.9+
- Mistral API key (required for OCR)
- OpenAI API key (optional, for image/chart regeneration)
- Anthropic API key (optional, for chart analysis)

## License

MIT

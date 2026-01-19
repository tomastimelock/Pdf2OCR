# Filepath: code_migration/extraction/pdf_toolkit/__init__.py
# Description: PDF Toolkit - Self-contained PDF processing module
# Layer: Extraction
# Version: 1.0.0

"""
PDF Toolkit Module
==================

A self-contained, copy-paste ready module for comprehensive PDF processing.

Features:
- PDF validation and metadata extraction
- Text extraction (full document or specific pages)
- Table extraction with export to CSV/JSON/Excel
- PDF splitting, merging, and page manipulation
- OCR support (optional, requires pytesseract)
- Embedding generation (optional, requires sentence-transformers)
- Document comparison and semantic search
- Password protection (encrypt/decrypt)

Installation:
    pip install -r requirements.txt

Quick Start:
    from pdf_toolkit import PDFToolkitProvider

    toolkit = PDFToolkitProvider()

    # Extract text
    text = toolkit.extract_text("document.pdf")

    # Get document info
    doc = toolkit.process_document("document.pdf")
    print(f"Pages: {doc.page_count}, Title: {doc.title}")

    # Extract tables
    tables = toolkit.extract_tables("document.pdf")

    # Split PDF
    files = toolkit.split_pdf("document.pdf", "output_dir/", pages_per_split=5)

    # Merge PDFs
    toolkit.merge_pdfs(["file1.pdf", "file2.pdf"], "merged.pdf")

Dependencies:
    Required: PyPDF2, pdfplumber, Pillow
    Optional: pdf2image, pytesseract, sentence-transformers, pandas, openpyxl
"""

from .provider import PDFToolkitProvider
from .config import ProviderConfig
from .dto import (
    DocumentInfo,
    PageInfo,
    TableInfo,
    ExtractionResult,
    ProcessingResult,
    EmbeddingResult,
    ProcessingStatus,
    OutlineItem,
    FormField,
    LayoutInfo,
    CompressionResult,
    ImageInfo,
    BatchResult,
)
from .exceptions import (
    PDFToolkitError,
    ValidationError,
    ExtractionError,
    ProcessingError,
    EmbeddingError,
    FileOperationError,
)

__all__ = [
    # Main provider
    "PDFToolkitProvider",
    # Configuration
    "ProviderConfig",
    # Data transfer objects
    "DocumentInfo",
    "PageInfo",
    "TableInfo",
    "ExtractionResult",
    "ProcessingResult",
    "EmbeddingResult",
    "ProcessingStatus",
    "OutlineItem",
    "FormField",
    "LayoutInfo",
    "CompressionResult",
    "ImageInfo",
    "BatchResult",
    # Exceptions
    "PDFToolkitError",
    "ValidationError",
    "ExtractionError",
    "ProcessingError",
    "EmbeddingError",
    "FileOperationError",
]

__version__ = "1.0.0"
__author__ = "DocumentHandler Project"

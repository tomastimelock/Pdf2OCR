"""
PDF2OCR - PDF to OCR processing package with Mistral AI integration.

This package provides tools for:
- Splitting PDFs into individual pages
- Converting PDF pages to JPG images
- OCR processing using Mistral AI
- Extracting structured information from OCR text
- Detecting and regenerating charts as SVG
- Extracting tables and converting to JSON/SVG
- Creating structured JSON document output
- Regenerating images (photos, illustrations) using OpenAI
- Exporting to Word (.docx) and PDF formats
"""

from pdf2ocr.api import PDF2OCR, process_pdf, process_directory
from pdf2ocr.processors.pdf_splitter import PDFSplitter
from pdf2ocr.processors.ocr_processor import OCRProcessor
from pdf2ocr.processors.chart_regenerator import ChartRegenerator, ChartData
from pdf2ocr.processors.image_regenerator import ImageRegenerator, RegeneratedImage, ImageRegenerationResult
from pdf2ocr.extractors.information_extractor import InformationExtractor
from pdf2ocr.extractors.table_extractor import TableExtractor, TableData
from pdf2ocr.extractors.table_to_svg import TableToSVG
from pdf2ocr.extractors.document_structurer import DocumentStructurer, StructuredDocument
from pdf2ocr.exporters import WordExporter, PDFExporter

__version__ = "0.1.0"
__all__ = [
    "PDF2OCR",
    "process_pdf",
    "process_directory",
    "PDFSplitter",
    "OCRProcessor",
    "ChartRegenerator",
    "ChartData",
    "ImageRegenerator",
    "RegeneratedImage",
    "ImageRegenerationResult",
    "InformationExtractor",
    "TableExtractor",
    "TableData",
    "TableToSVG",
    "DocumentStructurer",
    "StructuredDocument",
    "WordExporter",
    "PDFExporter",
]

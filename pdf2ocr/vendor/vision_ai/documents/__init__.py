"""Documents submodule for VisionAI.

Provides document processing capabilities including OCR, document scanning,
and structured data extraction from receipts and forms.
"""

from .ocr import (
    OCREngine,
    OCRResult,
    OCRLine,
    OCRWord,
    extract_text,
    read_text,
)
from .scanner import (
    DocumentScanner,
    ScannedDocument,
    ScanConfig,
    scan_document,
    enhance_document,
)
from .parser import (
    DocumentParser,
    ReceiptData,
    FormData,
    FieldValue,
    parse_receipt,
    parse_form,
    extract_fields,
)

__all__ = [
    # OCR
    "OCREngine",
    "OCRResult",
    "OCRLine",
    "OCRWord",
    "extract_text",
    "read_text",
    # Scanner
    "DocumentScanner",
    "ScannedDocument",
    "ScanConfig",
    "scan_document",
    "enhance_document",
    # Parser
    "DocumentParser",
    "ReceiptData",
    "FormData",
    "FieldValue",
    "parse_receipt",
    "parse_form",
    "extract_fields",
]

"""Document processors for PDF2OCR."""

from pdf2ocr.processors.pdf_splitter import PDFSplitter, PDFMetadata, PageInfo
from pdf2ocr.processors.ocr_processor import OCRProcessor, ProcessingResult
from pdf2ocr.processors.chart_regenerator import ChartRegenerator, ChartData, ChartProcessingResult
from pdf2ocr.processors.image_regenerator import (
    ImageRegenerator,
    ImageDescription,
    RegeneratedImage,
    ImageRegenerationResult,
)
from pdf2ocr.processors.pdf_toolkit import (
    PDFToolkit,
    PDFDocument,
    MergeResult,
    SplitResult,
)
from pdf2ocr.processors.llm_enhancer import (
    LLMEnhancer,
    EnhancedText,
    ExtractedStructure,
)
from pdf2ocr.processors.image_enhancer import (
    ImageEnhancer,
    QualityAnalysis,
    EnhancementResult,
)

__all__ = [
    # PDF Processing
    "PDFSplitter",
    "PDFMetadata",
    "PageInfo",
    "PDFToolkit",
    "PDFDocument",
    "MergeResult",
    "SplitResult",
    # OCR
    "OCRProcessor",
    "ProcessingResult",
    # Charts
    "ChartRegenerator",
    "ChartData",
    "ChartProcessingResult",
    # Images
    "ImageRegenerator",
    "ImageDescription",
    "RegeneratedImage",
    "ImageRegenerationResult",
    # LLM Enhancement
    "LLMEnhancer",
    "EnhancedText",
    "ExtractedStructure",
    # Image Enhancement
    "ImageEnhancer",
    "QualityAnalysis",
    "EnhancementResult",
]

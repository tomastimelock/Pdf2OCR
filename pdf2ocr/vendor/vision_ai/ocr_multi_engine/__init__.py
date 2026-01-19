# Filepath: code_migration/extraction/ocr_multi_engine/__init__.py
# Description: OCR Multi-Engine Module - Standalone OCR extraction library
# Layer: Extractor
# References: reference_codebase/OCR_extractor/provider/

"""
OCR Multi-Engine Module

A standalone, self-contained OCR extraction library with multi-engine fallback support.

This module provides:
- Multiple OCR engines (Tesseract, OpenAI Vision, Mistral OCR)
- Automatic quality-based fallback between engines
- Image and PDF processing with OCR
- Configurable quality thresholds
- Statistics tracking

Quick Start:
    >>> from ocr_multi_engine import OCRProviderFactory, DocumentProcessor
    >>>
    >>> # Create multi-engine OCR with fallback
    >>> ocr = OCRProviderFactory.create_multi_engine()
    >>>
    >>> # Process documents (images or PDFs)
    >>> processor = DocumentProcessor(ocr)
    >>> text = processor.process("document.pdf")
    >>> print(text)

Swedish OCR Example:
    >>> from ocr_multi_engine import OCRConfig, OCRProviderFactory
    >>>
    >>> # Configure for Swedish language
    >>> config = OCRConfig(
    ...     tesseract_lang="swe+eng",
    ...     engine_order=["openai", "mistral", "tesseract"]
    ... )
    >>>
    >>> # Create multi-engine OCR
    >>> ocr = OCRProviderFactory.create_multi_engine(
    ...     config=config.to_dict(),
    ...     engine_order=config.engine_order
    ... )
    >>>
    >>> text = ocr.extract_text("swedish_document.jpg")

Individual Engine Usage:
    >>> from ocr_multi_engine import OCRProviderFactory
    >>>
    >>> # Use only OpenAI Vision
    >>> ocr = OCRProviderFactory.create("openai", config={
    ...     "openai_api_key": "sk-...",
    ...     "openai_model": "gpt-4o"
    ... })
    >>> text = ocr.extract_text("image.png")
    >>>
    >>> # Check quality
    >>> quality = ocr.get_quality_score(text)
    >>> print(f"Quality: {quality:.2f}")
    >>>
    >>> # Get statistics
    >>> stats = ocr.get_stats()
    >>> print(stats)

Environment Configuration:
    Set these environment variables for automatic configuration:
    - OPENAI_API_KEY: OpenAI API key
    - MISTRAL_API_KEY: Mistral API key
    - OCR_ENGINE_ORDER: Comma-separated engine order (e.g., "openai,mistral,tesseract")
    - OCR_MIN_QUALITY: Minimum quality threshold (0.0-1.0)
    - OCR_TESSERACT_LANG: Tesseract language code (e.g., "eng", "swe+eng")
"""

# Base classes and exceptions
from .base import (
    BaseOCRProvider,
    OCRError,
    OCRProviderUnavailableError
)

# Configuration
from .config import (
    OCRConfig,
    get_config,
    set_config,
    reset_config
)

# Factory and multi-engine
from .factory import (
    OCRProviderFactory,
    MultiEngineOCR
)

# Document processors
from .processors import (
    DocumentProcessor,
    ImageProcessor,
    PDFProcessor,
    ProcessingError,
    UnsupportedFileTypeError
)

# Individual engines
from .engines import (
    TesseractOCR,
    OpenAIVision,
    MistralOCR
)

# Utilities
from .utils import (
    enhance_image_for_ocr,
    convert_to_rgb,
    get_image_mime_type,
    is_image_file,
    is_pdf_file,
    get_file_type,
    calculate_text_quality
)

__version__ = "1.0.0"

__all__ = [
    # Base
    "BaseOCRProvider",
    "OCRError",
    "OCRProviderUnavailableError",

    # Config
    "OCRConfig",
    "get_config",
    "set_config",
    "reset_config",

    # Factory
    "OCRProviderFactory",
    "MultiEngineOCR",

    # Processors
    "DocumentProcessor",
    "ImageProcessor",
    "PDFProcessor",
    "ProcessingError",
    "UnsupportedFileTypeError",

    # Engines
    "TesseractOCR",
    "OpenAIVision",
    "MistralOCR",

    # Utils
    "enhance_image_for_ocr",
    "convert_to_rgb",
    "get_image_mime_type",
    "is_image_file",
    "is_pdf_file",
    "get_file_type",
    "calculate_text_quality",
]


def create_default_ocr(config_dict=None):
    """
    Create a default multi-engine OCR instance with sensible defaults.

    Args:
        config_dict: Optional configuration dictionary

    Returns:
        MultiEngineOCR instance ready to use

    Example:
        >>> ocr = create_default_ocr()
        >>> text = ocr.extract_text("document.jpg")
    """
    import logging
    logger = logging.getLogger("OCR")

    # Load config from environment if not provided
    if config_dict is None:
        config = OCRConfig.from_env()
        config_dict = config.to_dict()
        engine_order = config.engine_order
    else:
        engine_order = config_dict.get("engine_order", ["openai", "mistral", "tesseract"])

    return OCRProviderFactory.create_multi_engine(
        logger=logger,
        config=config_dict,
        engine_order=engine_order
    )


def create_document_processor(config_dict=None):
    """
    Create a default document processor with multi-engine OCR.

    Args:
        config_dict: Optional configuration dictionary

    Returns:
        DocumentProcessor instance ready to process images and PDFs

    Example:
        >>> processor = create_document_processor()
        >>> text = processor.process("document.pdf")
        >>> stats = processor.get_stats()
    """
    ocr = create_default_ocr(config_dict)
    return DocumentProcessor(ocr)

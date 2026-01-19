# Filepath: code_migration/extraction/ocr_multi_engine/processors.py
# Description: Document processors for images and PDFs with OCR
# Layer: Extractor
# References: reference_codebase/OCR_extractor/provider/processors.py

"""
Document Processors

Processors for extracting text from various document formats using OCR.
"""

from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any, List
from pathlib import Path
import logging
import tempfile
import os

from .base import BaseOCRProvider, OCRError


class ProcessingError(Exception):
    """Exception raised when document processing fails."""
    pass


class UnsupportedFileTypeError(ProcessingError):
    """Exception raised when file type is not supported."""
    pass


class BaseProcessor(ABC):
    """
    Abstract base class for document processors.
    """

    def __init__(self, logger: Optional[logging.Logger] = None, config: Optional[Dict] = None):
        """
        Initialize the processor.

        Args:
            logger: Logger instance
            config: Optional configuration dictionary
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.config = config or {}
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0
        }

    @abstractmethod
    def process(self, file_path: Union[str, Path]) -> str:
        """
        Extract text from a document.

        Args:
            file_path: Path to the document

        Returns:
            Extracted text content
        """
        pass

    @abstractmethod
    def can_process(self, file_path: Union[str, Path]) -> bool:
        """
        Check if this processor can handle the given file.

        Args:
            file_path: Path to the file

        Returns:
            True if this processor can handle the file
        """
        pass

    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.

        Returns:
            List of extensions (e.g., ['.jpg', '.png'])
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        total = self.stats["total_processed"]
        success_rate = (
            self.stats["successful"] / total * 100
            if total > 0 else 0
        )
        return {
            **self.stats,
            "success_rate": round(success_rate, 2)
        }

    def reset_stats(self):
        """Reset processing statistics."""
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0
        }


class ImageProcessor(BaseProcessor):
    """
    Processor for image files using OCR.
    """

    SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp']

    def __init__(
        self,
        ocr_provider: BaseOCRProvider,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize image processor.

        Args:
            ocr_provider: OCR provider to use for text extraction
            logger: Optional logger instance
            config: Optional configuration dictionary
        """
        super().__init__(logger, config)
        self.ocr_provider = ocr_provider

    def process(self, file_path: Union[str, Path]) -> str:
        """
        Extract text from an image using OCR.

        Args:
            file_path: Path to the image file

        Returns:
            Extracted text content

        Raises:
            ProcessingError: If processing fails
            UnsupportedFileTypeError: If file type not supported
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.can_process(file_path):
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {file_path.suffix}. "
                f"Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        self.stats["total_processed"] += 1
        self.logger.info(f"Processing image: {file_path.name}")

        try:
            text = self.ocr_provider.extract_text(file_path)

            if len(text.strip()) < 5:
                self.logger.warning(f"Very little text extracted from {file_path.name}")

            self.stats["successful"] += 1
            return text

        except Exception as e:
            self.stats["failed"] += 1
            self.logger.error(f"Failed to process image {file_path.name}: {str(e)}")
            raise ProcessingError(f"Image processing failed: {str(e)}") from e

    def can_process(self, file_path: Union[str, Path]) -> bool:
        """Check if file is a supported image format."""
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS

    def get_supported_extensions(self) -> List[str]:
        """Get supported image extensions."""
        return self.SUPPORTED_EXTENSIONS.copy()


class PDFProcessor(BaseProcessor):
    """
    Processor for PDF files with OCR fallback.
    """

    def __init__(
        self,
        ocr_provider: Optional[BaseOCRProvider] = None,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize PDF processor.

        Args:
            ocr_provider: Optional OCR provider for fallback
            logger: Optional logger instance
            config: Optional configuration dictionary
                - min_text_length: Minimum text to consider extraction successful (default 100)
                - use_ocr_fallback: Enable OCR fallback (default True if ocr_provider provided)
                - ocr_dpi: DPI for OCR conversion (default 200)
        """
        super().__init__(logger, config)
        self.ocr_provider = ocr_provider
        self.min_text_length = config.get("min_text_length", 100) if config else 100
        self.use_ocr_fallback = config.get("use_ocr_fallback", ocr_provider is not None) if config else (ocr_provider is not None)
        self.ocr_dpi = config.get("ocr_dpi", 200) if config else 200

        # Check for PyMuPDF
        self._has_fitz = False
        try:
            import fitz
            self._fitz = fitz
            self._has_fitz = True
        except ImportError:
            self.logger.warning("PyMuPDF (fitz) not available - PDF text extraction disabled")

    def process(self, file_path: Union[str, Path]) -> str:
        """
        Extract text from a PDF file.

        First attempts direct text extraction with PyMuPDF.
        Falls back to OCR if text extraction is insufficient.

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text content

        Raises:
            ProcessingError: If processing fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.can_process(file_path):
            raise UnsupportedFileTypeError(f"Not a PDF file: {file_path}")

        self.stats["total_processed"] += 1
        self.logger.info(f"Processing PDF: {file_path.name}")

        try:
            # Try direct text extraction first
            if self._has_fitz:
                text = self._extract_text_with_pymupdf(file_path)

                # Check if extraction was sufficient
                if len(text.strip()) >= self.min_text_length:
                    self.logger.info(f"Direct extraction successful: {len(text)} characters")
                    self.stats["successful"] += 1
                    return text

                self.logger.info(
                    f"Direct extraction insufficient ({len(text)} chars < {self.min_text_length}), "
                    f"trying OCR fallback..."
                )

            # Fall back to OCR
            if self.use_ocr_fallback and self.ocr_provider:
                text = self._extract_with_ocr_fallback(file_path)
                self.stats["successful"] += 1
                return text

            # If no OCR fallback and direct extraction failed
            if self._has_fitz:
                self.stats["successful"] += 1
                return text  # Return whatever we got

            raise ProcessingError("No PDF extraction method available")

        except Exception as e:
            self.stats["failed"] += 1
            self.logger.error(f"Failed to process PDF {file_path.name}: {str(e)}")
            raise ProcessingError(f"PDF processing failed: {str(e)}") from e

    def _extract_text_with_pymupdf(self, file_path: Path) -> str:
        """
        Extract text directly from PDF using PyMuPDF.

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text
        """
        doc = self._fitz.open(file_path)
        text_parts = []

        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
        finally:
            doc.close()

        return "\n\n".join(text_parts)

    def _extract_with_ocr_fallback(self, file_path: Path) -> str:
        """
        Extract text from PDF by converting pages to images and running OCR.

        Args:
            file_path: Path to PDF file

        Returns:
            OCR-extracted text
        """
        if not self._has_fitz:
            raise ProcessingError("PyMuPDF required for OCR fallback")

        doc = self._fitz.open(file_path)
        text_parts = []
        temp_files = []

        try:
            zoom = self.ocr_dpi / 72  # 72 is default PDF DPI
            matrix = self._fitz.Matrix(zoom, zoom)

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Convert page to image
                pix = page.get_pixmap(matrix=matrix)

                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                temp_files.append(temp_file.name)
                pix.save(temp_file.name)

                # OCR the image
                self.logger.info(f"OCR processing page {page_num + 1}...")
                page_text = self.ocr_provider.extract_text(temp_file.name)

                if page_text.strip():
                    text_parts.append(f"--- Page {page_num + 1} (OCR) ---\n{page_text}")

        finally:
            doc.close()
            # Clean up temp files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass

        return "\n\n".join(text_parts)

    def can_process(self, file_path: Union[str, Path]) -> bool:
        """Check if file is a PDF."""
        return Path(file_path).suffix.lower() == '.pdf'

    def get_supported_extensions(self) -> List[str]:
        """Get supported extensions."""
        return ['.pdf']


class DocumentProcessor:
    """
    Unified document processor that handles multiple file types.

    Combines image and PDF processors with automatic type detection.
    """

    def __init__(
        self,
        ocr_provider: BaseOCRProvider,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize document processor.

        Args:
            ocr_provider: OCR provider for text extraction
            logger: Optional logger instance
            config: Optional configuration dictionary
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.config = config or {}
        self.ocr_provider = ocr_provider

        # Initialize processors
        self.image_processor = ImageProcessor(ocr_provider, logger, config)
        self.pdf_processor = PDFProcessor(ocr_provider, logger, config)

        self._processors = [self.image_processor, self.pdf_processor]

    def process(self, file_path: Union[str, Path]) -> str:
        """
        Process a document and extract text.

        Automatically selects the appropriate processor based on file type.

        Args:
            file_path: Path to the document

        Returns:
            Extracted text content

        Raises:
            UnsupportedFileTypeError: If no processor can handle the file
            ProcessingError: If processing fails
        """
        file_path = Path(file_path)

        for processor in self._processors:
            if processor.can_process(file_path):
                return processor.process(file_path)

        raise UnsupportedFileTypeError(
            f"No processor available for file type: {file_path.suffix}. "
            f"Supported: {', '.join(self.get_supported_extensions())}"
        )

    def can_process(self, file_path: Union[str, Path]) -> bool:
        """Check if any processor can handle the file."""
        return any(p.can_process(file_path) for p in self._processors)

    def get_supported_extensions(self) -> List[str]:
        """Get all supported file extensions."""
        extensions = []
        for processor in self._processors:
            extensions.extend(processor.get_supported_extensions())
        return list(set(extensions))

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics from all processors."""
        total_processed = sum(p.stats["total_processed"] for p in self._processors)
        successful = sum(p.stats["successful"] for p in self._processors)
        failed = sum(p.stats["failed"] for p in self._processors)

        return {
            "total_processed": total_processed,
            "successful": successful,
            "failed": failed,
            "success_rate": round(successful / total_processed * 100, 2) if total_processed > 0 else 0,
            "by_processor": {
                "image": self.image_processor.get_stats(),
                "pdf": self.pdf_processor.get_stats()
            }
        }

    def reset_stats(self):
        """Reset statistics for all processors."""
        for processor in self._processors:
            processor.reset_stats()

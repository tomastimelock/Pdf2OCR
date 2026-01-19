"""Tesseract OCR Provider - Local OCR fallback using Tesseract."""

import os
from pathlib import Path
from typing import Optional

from pdf2ocr.providers.base import BaseOCRProvider, OCRResult


class TesseractOCRProvider(BaseOCRProvider):
    """
    OCR provider using Tesseract for local text extraction.

    Provides a free, offline fallback when cloud APIs are unavailable.
    Requires tesseract-ocr to be installed on the system.
    """

    name: str = "tesseract"
    quality_score: float = 0.7

    def __init__(self, language: str = "eng"):
        """
        Initialize Tesseract provider.

        Args:
            language: Tesseract language code (default: "eng")
        """
        self.language = language
        self._pytesseract = None

    @property
    def pytesseract(self):
        """Lazy load pytesseract module."""
        if self._pytesseract is None:
            try:
                import pytesseract
                self._pytesseract = pytesseract
            except ImportError:
                raise ImportError(
                    "pytesseract is required for Tesseract OCR. "
                    "Install with: pip install pytesseract"
                )
        return self._pytesseract

    def is_available(self) -> bool:
        """Check if Tesseract is available."""
        try:
            self.pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def extract_text(self, image_path: str | Path, page_number: int = 1) -> OCRResult:
        """
        Extract text from an image using Tesseract.

        Args:
            image_path: Path to the image file
            page_number: Page number for tracking

        Returns:
            OCRResult with extracted text
        """
        image_path = Path(image_path)

        try:
            from PIL import Image

            with Image.open(image_path) as img:
                text = self.pytesseract.image_to_string(
                    img,
                    lang=self.language
                )

            quality = self.calculate_quality_score(text)

            return OCRResult(
                text=text,
                page_number=page_number,
                provider=self.name,
                quality_score=quality,
                confidence=quality,
                language=self.language,
                metadata={
                    "engine": "tesseract",
                    "language": self.language
                }
            )

        except Exception as e:
            return OCRResult(
                text="",
                page_number=page_number,
                provider=self.name,
                quality_score=0.0,
                confidence=0.0,
                metadata={"error": str(e)}
            )

    def extract_text_from_pdf(self, pdf_path: str | Path, page_number: int = 1) -> OCRResult:
        """
        Extract text from a PDF page using Tesseract.

        Converts PDF page to image first, then runs OCR.

        Args:
            pdf_path: Path to the PDF file
            page_number: Page number to extract (1-indexed)

        Returns:
            OCRResult with extracted text
        """
        import tempfile
        import fitz

        pdf_path = Path(pdf_path)

        try:
            with fitz.open(pdf_path) as doc:
                if page_number < 1 or page_number > len(doc):
                    return OCRResult(
                        text="",
                        page_number=page_number,
                        provider=self.name,
                        quality_score=0.0,
                        confidence=0.0,
                        metadata={"error": f"Invalid page number: {page_number}"}
                    )

                page = doc[page_number - 1]
                pix = page.get_pixmap(dpi=200)

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                    pix.save(tmp_path)

                try:
                    result = self.extract_text(tmp_path, page_number)
                finally:
                    os.unlink(tmp_path)

                return result

        except Exception as e:
            return OCRResult(
                text="",
                page_number=page_number,
                provider=self.name,
                quality_score=0.0,
                confidence=0.0,
                metadata={"error": str(e)}
            )

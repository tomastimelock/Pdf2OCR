"""Base OCR provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class OCRResult:
    """Result from OCR processing."""
    text: str
    page_number: int
    provider: str
    quality_score: float = 0.0
    confidence: float = 0.0
    language: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_successful(self) -> bool:
        """Check if OCR extraction was successful."""
        return bool(self.text.strip()) and self.quality_score >= 0.5


class BaseOCRProvider(ABC):
    """Abstract base class for OCR providers."""

    name: str = "base"
    quality_score: float = 0.5

    @abstractmethod
    def extract_text(self, image_path: str | Path, page_number: int = 1) -> OCRResult:
        """
        Extract text from an image using OCR.

        Args:
            image_path: Path to the image file
            page_number: Page number for tracking (default: 1)

        Returns:
            OCRResult with extracted text and metadata
        """
        pass

    @abstractmethod
    def extract_text_from_pdf(self, pdf_path: str | Path, page_number: int = 1) -> OCRResult:
        """
        Extract text directly from a PDF page.

        Args:
            pdf_path: Path to the PDF file
            page_number: Page number to extract (1-indexed)

        Returns:
            OCRResult with extracted text and metadata
        """
        pass

    def calculate_quality_score(self, text: str) -> float:
        """
        Calculate a quality score for the extracted text.

        Args:
            text: Extracted text

        Returns:
            Quality score between 0.0 and 1.0
        """
        if not text or not text.strip():
            return 0.0

        text = text.strip()
        char_count = len(text)
        word_count = len(text.split())

        if char_count < 10:
            return 0.1
        elif char_count < 50:
            return 0.3
        elif char_count < 200:
            return 0.5
        elif char_count < 1000:
            return 0.7
        else:
            return min(0.95, 0.7 + (word_count / 10000))

    def is_available(self) -> bool:
        """
        Check if this provider is available and properly configured.

        Returns:
            True if provider can be used
        """
        return True

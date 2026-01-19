# Filepath: code_migration/extraction/ocr_multi_engine/base.py
# Description: Base OCR Provider Abstract Class - Interface for all OCR engines
# Layer: Extractor
# References: reference_codebase/OCR_extractor/provider/base.py

"""
Base OCR Provider Abstract Class

Provides interface for all OCR engines (Mistral, OpenAI Vision, Tesseract).
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pathlib import Path
import logging


class OCRError(Exception):
    """Exception raised when OCR processing fails."""
    pass


class OCRProviderUnavailableError(OCRError):
    """Exception raised when OCR provider is not available."""
    pass


class BaseOCRProvider(ABC):
    """
    Abstract base class for OCR providers.

    Provides a unified interface for different OCR engines.
    """

    def __init__(self, logger: Optional[logging.Logger] = None, config: Optional[Dict] = None):
        """
        Initialize the OCR provider.

        Args:
            logger: Logger instance (creates a default one if not provided)
            config: Optional configuration dictionary
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.config = config or {}
        self.ocr_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_characters_extracted": 0
        }

    @abstractmethod
    def extract_text(self, image_path: Path) -> str:
        """
        Extract text from an image.

        Args:
            image_path: Path to the image file

        Returns:
            Extracted text content

        Raises:
            OCRError: If OCR processing fails
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Get the name of this OCR provider.

        Returns:
            Provider name (e.g., "Mistral OCR", "OpenAI Vision")
        """
        pass

    def is_available(self) -> bool:
        """
        Check if this OCR provider is available.

        Returns:
            True if provider is configured and ready to use
        """
        return True

    def get_quality_score(self, text: str) -> float:
        """
        Calculate a quality score for extracted text.

        Args:
            text: Extracted text

        Returns:
            Quality score between 0.0 and 1.0
        """
        if not text or len(text.strip()) < 10:
            return 0.0

        # Simple heuristics for quality
        char_count = len(text)
        word_count = len(text.split())
        avg_word_length = char_count / word_count if word_count > 0 else 0

        # Check for reasonable text characteristics
        quality = 1.0

        # Penalize if too short
        if char_count < 50:
            quality -= 0.3

        # Penalize if average word length is unrealistic
        if avg_word_length < 2 or avg_word_length > 15:
            quality -= 0.2

        # Penalize if too many non-alphanumeric characters
        non_alnum = sum(1 for c in text if not c.isalnum() and not c.isspace())
        non_alnum_ratio = non_alnum / char_count if char_count > 0 else 0
        if non_alnum_ratio > 0.3:
            quality -= 0.2

        return max(0.0, min(1.0, quality))

    def get_stats(self) -> Dict[str, Any]:
        """
        Get OCR statistics.

        Returns:
            Dictionary with OCR statistics
        """
        total = self.ocr_stats["total_processed"]
        success_rate = (
            self.ocr_stats["successful"] / total * 100
            if total > 0 else 0
        )

        return {
            **self.ocr_stats,
            "success_rate": round(success_rate, 2),
            "provider": self.get_provider_name()
        }

    def reset_stats(self):
        """Reset OCR statistics."""
        self.ocr_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_characters_extracted": 0
        }

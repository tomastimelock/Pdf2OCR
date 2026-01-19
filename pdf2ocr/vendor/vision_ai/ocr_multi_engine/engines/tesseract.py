# Filepath: code_migration/extraction/ocr_multi_engine/engines/tesseract.py
# Description: Tesseract OCR Provider - Local OCR engine fallback
# Layer: Extractor
# References: reference_codebase/OCR_extractor/provider/tesseract.py

"""
Tesseract OCR Provider

Provides OCR functionality using Tesseract OCR engine.
This is typically used as a fallback when API-based OCR is unavailable.
"""

from typing import Union, Optional
from pathlib import Path
import logging
import tempfile
import os

from ..base import BaseOCRProvider, OCRError


class TesseractOCR(BaseOCRProvider):
    """OCR provider using Tesseract OCR engine."""

    def __init__(
        self,
        lang: str = "eng",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Tesseract OCR provider.

        Args:
            lang: Tesseract language code (default: 'eng' for English)
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.lang = lang
        self._has_tesseract = False
        self._has_pil = False
        self.pytesseract = None
        self.Image = None
        self.ImageEnhance = None

        # Check for Tesseract
        try:
            import pytesseract
            self.pytesseract = pytesseract
            self._has_tesseract = True
        except ImportError:
            self.logger.warning("pytesseract not available")

        # Check for PIL
        try:
            from PIL import Image, ImageEnhance
            self.Image = Image
            self.ImageEnhance = ImageEnhance
            self._has_pil = True
        except ImportError:
            self.logger.warning("PIL/Pillow not available")

    def extract_text(self, image_path: Union[str, Path]) -> str:
        """
        Extract text from image using Tesseract OCR.

        Args:
            image_path: Path to the image file

        Returns:
            Extracted text content

        Raises:
            OCRError: If OCR extraction fails or dependencies are missing
        """
        if not self._has_tesseract:
            raise OCRError("pytesseract library not available")

        if not self._has_pil:
            raise OCRError("PIL/Pillow library not available")

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        self.logger.info(f"Extracting text from {image_path.name} using Tesseract OCR")
        self.ocr_stats["total_processed"] += 1

        try:
            # Open and preprocess the image
            image = self.Image.open(image_path)

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Enhance the image for better OCR results
            image = self._enhance_image(image)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                temp_image_path = tmp.name

            image.save(temp_image_path)

            try:
                # Try with specified language first
                extracted_text = self.pytesseract.image_to_string(
                    temp_image_path,
                    lang=self.lang
                )
            except Exception as lang_error:
                self.logger.warning(
                    f"Failed with language '{self.lang}', trying default: {str(lang_error)}"
                )
                # Fall back to default language (usually English)
                extracted_text = self.pytesseract.image_to_string(temp_image_path)

            # Clean up temporary file
            try:
                os.unlink(temp_image_path)
            except OSError:
                pass

            text = extracted_text.strip()
            self.ocr_stats["successful"] += 1
            self.ocr_stats["total_characters_extracted"] += len(text)
            return text

        except Exception as e:
            self.ocr_stats["failed"] += 1
            self.logger.error(f"Tesseract OCR failed for {image_path.name}: {str(e)}")
            raise OCRError(f"Tesseract OCR failed: {str(e)}") from e

    def _enhance_image(self, image):
        """
        Enhance image for better OCR results.

        Args:
            image: PIL Image object

        Returns:
            Enhanced PIL Image object
        """
        try:
            # Increase contrast
            enhancer = self.ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)

            # Increase sharpness
            enhancer = self.ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)

            # Increase brightness slightly
            enhancer = self.ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.1)

            return image
        except Exception as e:
            self.logger.warning(f"Image enhancement failed: {str(e)}")
            return image

    def get_provider_name(self) -> str:
        """Get the provider name."""
        return f"Tesseract OCR (lang: {self.lang})"

    def is_available(self) -> bool:
        """
        Check if the provider is available.

        Returns:
            True if pytesseract and PIL are both available
        """
        return self._has_tesseract and self._has_pil

    def get_quality_score(self, text: str) -> float:
        """
        Calculate quality score for extracted text.

        Tesseract quality can vary significantly based on image quality.

        Args:
            text: Extracted text to score

        Returns:
            Quality score between 0 and 1
        """
        if not text or not text.strip():
            return 0.0

        score = 0.6  # Base score for Tesseract (lower than API-based OCR)

        # Length check
        if len(text) < 10:
            score -= 0.2
        elif len(text) > 100:
            score += 0.1

        # Check for common Tesseract artifacts and errors
        tesseract_artifacts = [
            "|||",     # Common line artifact
            "---",     # Line artifacts
            "~~~",     # Wavy lines
            "...",     # Dot artifacts
            "???",     # Question marks (unrecognized characters)
        ]

        artifact_count = sum(1 for artifact in tesseract_artifacts if artifact in text)
        score -= artifact_count * 0.05

        # Check for reasonable character distribution
        if len(text) > 0:
            alpha_ratio = sum(c.isalpha() for c in text) / len(text)
            if alpha_ratio > 0.5:  # At least 50% alphabetic characters
                score += 0.1
            elif alpha_ratio < 0.3:  # Too few alphabetic characters
                score -= 0.1

        # Check for excessive whitespace (often indicates poor OCR)
        if text.count('\n\n\n') > 2:  # Multiple triple newlines
            score -= 0.1

        # Check for word-like structures (spaces between groups of characters)
        words = text.split()
        if len(words) > 5:  # Has multiple words
            avg_word_length = sum(len(w) for w in words) / len(words)
            if 2 <= avg_word_length <= 12:  # Reasonable word length
                score += 0.1

        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))

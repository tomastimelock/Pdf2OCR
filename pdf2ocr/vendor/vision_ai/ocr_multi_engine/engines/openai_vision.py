# Filepath: code_migration/extraction/ocr_multi_engine/engines/openai_vision.py
# Description: OpenAI Vision OCR Provider - High-quality cloud OCR
# Layer: Extractor
# References: reference_codebase/OCR_extractor/provider/openai_vision.py

"""
OpenAI Vision OCR Provider

Provides OCR functionality using OpenAI's Vision API (GPT-4 Vision).
"""

from typing import Union, Optional
from pathlib import Path
import base64
import logging

from ..base import BaseOCRProvider, OCRError


class OpenAIVision(BaseOCRProvider):
    """OCR provider using OpenAI Vision API."""

    DEFAULT_PROMPT = """
Extract all text from this image with high accuracy. Follow these guidelines:

1. **Text Extraction**: Extract all visible text exactly as it appears
2. **Structure Preservation**: Maintain the original layout, formatting, and structure
3. **Tables and Schedules**: If there are tables, calendars, or schedules, describe their structure in detail
4. **Special Elements**: Pay attention to:
   - Dates and times
   - Names and personal information
   - Travel plans, vacations, or custody arrangements
   - Colored days or highlighted sections
   - Any patterns that might indicate schedules

5. **Context**: If the image contains a schedule or calendar, describe:
   - What the colored or marked days represent
   - Any legends or keys visible
   - Patterns in the schedule

Return the text in a clear, structured format that preserves the original organization.
"""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        prompt: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize OpenAI Vision provider.

        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o)
            prompt: Custom extraction prompt (optional)
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.api_key = api_key
        self.model = model
        self.prompt = prompt or self.DEFAULT_PROMPT
        self.client = None
        self._has_client = False

        # Try to import and initialize OpenAI client
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            self._has_client = True
        except ImportError:
            self.logger.warning("OpenAI library not available")

    def extract_text(self, image_path: Union[str, Path]) -> str:
        """
        Extract text from image using OpenAI Vision.

        Args:
            image_path: Path to the image file

        Returns:
            Extracted text content

        Raises:
            OCRError: If OCR extraction fails
        """
        if not self._has_client:
            raise OCRError("OpenAI library not available")

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        self.logger.info(f"Extracting text from {image_path.name} using OpenAI Vision")
        self.ocr_stats["total_processed"] += 1

        try:
            # Read and encode the image
            with open(image_path, 'rb') as f:
                image_data = f.read()

            # Determine MIME type
            suffix = image_path.suffix.lower()
            mime_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
            }
            mime_type = mime_types.get(suffix, 'image/jpeg')

            # Encode for OpenAI
            base64_encoded = f"data:{mime_type};base64,{base64.b64encode(image_data).decode('utf-8')}"

            # Make API request
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": base64_encoded}
                            }
                        ]
                    }
                ],
                temperature=0.1,  # Low temperature for accuracy
                max_tokens=2000   # Enough for most documents
            )

            text_content = response.choices[0].message.content
            self.ocr_stats["successful"] += 1
            self.ocr_stats["total_characters_extracted"] += len(text_content)
            return text_content

        except Exception as e:
            self.ocr_stats["failed"] += 1
            self.logger.error(f"OpenAI Vision failed for {image_path.name}: {str(e)}")
            raise OCRError(f"OpenAI Vision failed: {str(e)}") from e

    def get_provider_name(self) -> str:
        """Get the provider name."""
        return f"OpenAI Vision ({self.model})"

    def is_available(self) -> bool:
        """
        Check if the provider is available.

        Returns:
            True if OpenAI library and API key are configured
        """
        return self._has_client and bool(self.api_key)

    def get_quality_score(self, text: str) -> float:
        """
        Calculate quality score for extracted text.

        OpenAI Vision generally produces high-quality results.

        Args:
            text: Extracted text to score

        Returns:
            Quality score between 0 and 1
        """
        if not text or not text.strip():
            return 0.0

        score = 0.9  # Base score for OpenAI Vision (very high quality)

        # Length check
        if len(text) < 10:
            score -= 0.3
        elif len(text) > 50:
            score += 0.05

        # Content quality checks
        text_lower = text.lower()

        # Check for OCR error indicators
        ocr_errors = ["###", "|||", "[unreadable]", "[?]"]
        for error in ocr_errors:
            if error in text_lower:
                score -= 0.1

        # Check for structured content (good indicator)
        has_structure_indicators = [
            "\n\n" in text,  # Paragraphs
            ":" in text,     # Labels
            "-" in text,     # Lists
            "•" in text      # Bullet points
        ]
        if sum(has_structure_indicators) >= 2:
            score += 0.05

        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))

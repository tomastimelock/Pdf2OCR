# Filepath: code_migration/extraction/ocr_multi_engine/engines/mistral.py
# Description: Mistral OCR Provider - Cloud OCR with handwriting support
# Layer: Extractor
# References: reference_codebase/OCR_extractor/provider/mistral.py

"""
Mistral OCR Provider

Provides OCR functionality using Mistral AI's OCR API.
"""

from typing import Union, Optional
from pathlib import Path
import base64
import logging

from ..base import BaseOCRProvider, OCRError


class MistralOCR(BaseOCRProvider):
    """OCR provider using Mistral AI's OCR API."""

    def __init__(self, api_key: str, logger: Optional[logging.Logger] = None):
        """
        Initialize Mistral OCR provider.

        Args:
            api_key: Mistral API key
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.api_key = api_key
        self.client = None
        self._has_client = False

        # Try to import Mistral client
        try:
            from mistralai import Mistral
            self.client = Mistral(api_key=self.api_key)
            self._has_client = True
        except ImportError:
            self.logger.warning("Mistral client not available, using direct API calls")

    def extract_text(self, image_path: Union[str, Path]) -> str:
        """
        Extract text from image using Mistral OCR.

        Args:
            image_path: Path to the image file

        Returns:
            Extracted text content

        Raises:
            OCRError: If OCR extraction fails
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        self.logger.info(f"Extracting text from {image_path.name} using Mistral OCR")
        self.ocr_stats["total_processed"] += 1

        try:
            # Read and encode the image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            # Use official client if available
            if self._has_client and self.client:
                text = self._extract_with_client(base64_image)
            else:
                text = self._extract_with_api(base64_image)

            self.ocr_stats["successful"] += 1
            self.ocr_stats["total_characters_extracted"] += len(text)
            return text

        except Exception as e:
            self.ocr_stats["failed"] += 1
            self.logger.error(f"Mistral OCR failed for {image_path.name}: {str(e)}")
            raise OCRError(f"Mistral OCR failed: {str(e)}") from e

    def _extract_with_client(self, base64_image: str) -> str:
        """
        Extract text using official Mistral client.

        Args:
            base64_image: Base64-encoded image

        Returns:
            Extracted text
        """
        ocr_response = self.client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{base64_image}"
            }
        )

        return ocr_response.text

    def _extract_with_api(self, base64_image: str) -> str:
        """
        Extract text using direct API calls.

        Args:
            base64_image: Base64-encoded image

        Returns:
            Extracted text
        """
        import requests

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": "mistral-ocr-latest",
            "document": {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{base64_image}"
            }
        }

        response = requests.post(
            "https://api.mistral.ai/v1/ocr",
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            raise OCRError(f"Mistral OCR API error: {response.status_code} - {response.text}")

        result = response.json()
        return result.get("text", "")

    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "Mistral OCR"

    def is_available(self) -> bool:
        """
        Check if the provider is available.

        Returns:
            True if API key is configured
        """
        return bool(self.api_key)

    def get_quality_score(self, text: str) -> float:
        """
        Calculate quality score for extracted text.

        Mistral OCR is generally high quality, so base score is higher.

        Args:
            text: Extracted text to score

        Returns:
            Quality score between 0 and 1
        """
        if not text or not text.strip():
            return 0.0

        score = 0.85  # Base score for Mistral OCR

        # Length check
        if len(text) < 10:
            score -= 0.2
        elif len(text) > 50:
            score += 0.1

        # Check for common OCR artifacts
        if "|||" in text or "###" in text:
            score -= 0.1

        # Check for reasonable text structure
        if "\n" in text:  # Has line breaks (structured)
            score += 0.05

        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))

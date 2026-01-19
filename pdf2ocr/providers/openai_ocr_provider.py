"""OpenAI Vision OCR Provider - OCR using GPT-4 Vision."""

import os
import base64
from pathlib import Path
from typing import Optional

from pdf2ocr.providers.base import BaseOCRProvider, OCRResult


class OpenAIOCRProvider(BaseOCRProvider):
    """
    OCR provider using OpenAI's GPT-4 Vision for text extraction.

    Uses the vision capabilities of GPT-4o to extract text from images.
    Provides high-quality OCR especially for complex documents.
    """

    name: str = "openai_vision"
    quality_score: float = 0.85

    DEFAULT_PROMPT = """Extract all text from this image exactly as it appears.
Maintain the original formatting, including:
- Paragraphs and line breaks
- Bullet points and numbered lists
- Table structures (represent as plain text)
- Headers and titles

Return ONLY the extracted text, no explanations or commentary."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        prompt: Optional[str] = None
    ):
        """
        Initialize OpenAI Vision provider.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            model: Model to use (default: gpt-4o)
            prompt: Custom OCR prompt (uses default if not provided)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.prompt = prompt or self.DEFAULT_PROMPT
        self._client = None

    @property
    def client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                    "or pass api_key to constructor."
                )
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package is required. Install with: pip install openai"
                )
        return self._client

    def is_available(self) -> bool:
        """Check if OpenAI Vision is available."""
        return bool(self.api_key)

    def _encode_image(self, image_path: Path) -> tuple[str, str]:
        """Encode image to base64 and determine MIME type."""
        suffix = image_path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        mime_type = mime_types.get(suffix, "image/jpeg")

        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        return encoded, mime_type

    def extract_text(self, image_path: str | Path, page_number: int = 1) -> OCRResult:
        """
        Extract text from an image using GPT-4 Vision.

        Args:
            image_path: Path to the image file
            page_number: Page number for tracking

        Returns:
            OCRResult with extracted text
        """
        image_path = Path(image_path)

        try:
            encoded_image, mime_type = self._encode_image(image_path)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{encoded_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4096
            )

            text = response.choices[0].message.content or ""
            quality = self.calculate_quality_score(text)

            return OCRResult(
                text=text,
                page_number=page_number,
                provider=self.name,
                quality_score=quality,
                confidence=quality,
                metadata={
                    "model": self.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens
                    }
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
        Extract text from a PDF page using GPT-4 Vision.

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

"""Mistral AI OCR provider."""

import os
import base64
from pathlib import Path
from typing import Optional

from pdf2ocr.providers.base import BaseOCRProvider, OCRResult


class MistralOCRProvider(BaseOCRProvider):
    """
    OCR provider using Mistral AI's vision capabilities.

    Uses the Mistral OCR API (mistral-ocr-latest model) for high-quality
    text extraction from images and PDFs.
    """

    name = "mistral"
    quality_score = 0.85

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Mistral OCR provider.

        Args:
            api_key: Mistral API key. If not provided, reads from MISTRAL_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self._client = None

    @property
    def client(self):
        """Lazy initialization of Mistral client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "Mistral API key not provided. "
                    "Set MISTRAL_API_KEY environment variable or pass api_key parameter."
                )
            from mistralai import Mistral
            self._client = Mistral(api_key=self.api_key)
        return self._client

    def is_available(self) -> bool:
        """Check if Mistral API is available."""
        return bool(self.api_key)

    def _get_mime_type(self, file_path: Path) -> str:
        """Determine MIME type from file extension."""
        ext = file_path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".pdf": "application/pdf",
        }
        return mime_types.get(ext, "application/octet-stream")

    def _encode_file(self, file_path: Path) -> str:
        """Read and base64 encode a file."""
        with open(file_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def extract_text(self, image_path: str | Path, page_number: int = 1) -> OCRResult:
        """
        Extract text from an image using Mistral OCR.

        Args:
            image_path: Path to the image file
            page_number: Page number for tracking

        Returns:
            OCRResult with extracted text
        """
        image_path = Path(image_path)

        if not image_path.exists():
            return OCRResult(
                text="",
                page_number=page_number,
                provider=self.name,
                quality_score=0.0,
                metadata={"error": f"File not found: {image_path}"}
            )

        try:
            mime_type = self._get_mime_type(image_path)
            encoded_data = self._encode_file(image_path)
            data_url = f"data:{mime_type};base64,{encoded_data}"

            response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "image_url",
                    "image_url": data_url
                }
            )

            text = self._extract_markdown_from_response(response)
            quality = self.calculate_quality_score(text)

            return OCRResult(
                text=text,
                page_number=page_number,
                provider=self.name,
                quality_score=quality,
                confidence=quality,
                metadata={
                    "source_file": str(image_path),
                    "model": "mistral-ocr-latest"
                }
            )

        except Exception as e:
            return OCRResult(
                text="",
                page_number=page_number,
                provider=self.name,
                quality_score=0.0,
                metadata={"error": str(e)}
            )

    def extract_text_from_pdf(self, pdf_path: str | Path, page_number: int = 1) -> OCRResult:
        """
        Extract text from a PDF page using Mistral OCR.

        Args:
            pdf_path: Path to the PDF file
            page_number: Page number to extract (1-indexed)

        Returns:
            OCRResult with extracted text
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            return OCRResult(
                text="",
                page_number=page_number,
                provider=self.name,
                quality_score=0.0,
                metadata={"error": f"File not found: {pdf_path}"}
            )

        try:
            mime_type = "application/pdf"
            encoded_data = self._encode_file(pdf_path)
            data_url = f"data:{mime_type};base64,{encoded_data}"

            response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": data_url
                }
            )

            all_text = self._extract_markdown_from_response(response)

            if hasattr(response, "pages") and response.pages:
                if page_number <= len(response.pages):
                    page_data = response.pages[page_number - 1]
                    if hasattr(page_data, "markdown"):
                        text = page_data.markdown
                    else:
                        text = all_text
                else:
                    text = all_text
            else:
                text = all_text

            quality = self.calculate_quality_score(text)

            return OCRResult(
                text=text,
                page_number=page_number,
                provider=self.name,
                quality_score=quality,
                confidence=quality,
                metadata={
                    "source_file": str(pdf_path),
                    "model": "mistral-ocr-latest",
                    "total_pages": len(response.pages) if hasattr(response, "pages") else 1
                }
            )

        except Exception as e:
            return OCRResult(
                text="",
                page_number=page_number,
                provider=self.name,
                quality_score=0.0,
                metadata={"error": str(e)}
            )

    def extract_all_pages(self, pdf_path: str | Path) -> list[OCRResult]:
        """
        Extract text from all pages of a PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of OCRResult objects, one per page
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            return [OCRResult(
                text="",
                page_number=1,
                provider=self.name,
                quality_score=0.0,
                metadata={"error": f"File not found: {pdf_path}"}
            )]

        try:
            mime_type = "application/pdf"
            encoded_data = self._encode_file(pdf_path)
            data_url = f"data:{mime_type};base64,{encoded_data}"

            response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": data_url
                }
            )

            results = []

            if hasattr(response, "pages") and response.pages:
                for idx, page_data in enumerate(response.pages, start=1):
                    text = page_data.markdown if hasattr(page_data, "markdown") else ""
                    quality = self.calculate_quality_score(text)

                    results.append(OCRResult(
                        text=text,
                        page_number=idx,
                        provider=self.name,
                        quality_score=quality,
                        confidence=quality,
                        metadata={
                            "source_file": str(pdf_path),
                            "model": "mistral-ocr-latest"
                        }
                    ))
            else:
                all_text = self._extract_markdown_from_response(response)
                quality = self.calculate_quality_score(all_text)
                results.append(OCRResult(
                    text=all_text,
                    page_number=1,
                    provider=self.name,
                    quality_score=quality,
                    confidence=quality,
                    metadata={
                        "source_file": str(pdf_path),
                        "model": "mistral-ocr-latest"
                    }
                ))

            return results

        except Exception as e:
            return [OCRResult(
                text="",
                page_number=1,
                provider=self.name,
                quality_score=0.0,
                metadata={"error": str(e)}
            )]

    def _extract_markdown_from_response(self, response) -> str:
        """Extract markdown text from Mistral OCR response."""
        if hasattr(response, "pages") and response.pages:
            texts = []
            for page in response.pages:
                if hasattr(page, "markdown") and page.markdown:
                    texts.append(page.markdown)
            return "\n\n".join(texts)
        elif hasattr(response, "text"):
            return response.text
        elif hasattr(response, "content"):
            return response.content
        return ""

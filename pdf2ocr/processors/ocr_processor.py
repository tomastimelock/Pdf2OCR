"""OCR Processor - Orchestrates OCR processing with multi-provider support."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable

from pdf2ocr.providers.base import BaseOCRProvider, OCRResult
from pdf2ocr.providers.mistral_provider import MistralOCRProvider


@dataclass
class ProcessingResult:
    """Result of processing a document or page."""
    pages: list[OCRResult] = field(default_factory=list)
    total_pages: int = 0
    successful_pages: int = 0
    failed_pages: int = 0
    average_quality: float = 0.0
    provider_used: str = ""

    @property
    def is_successful(self) -> bool:
        """Check if processing was mostly successful."""
        return self.successful_pages > 0 and self.failed_pages < self.total_pages / 2


class OCRProcessor:
    """
    Orchestrates OCR processing with support for multiple providers.

    Primary provider is Mistral, with optional fallback support for
    Tesseract (local OCR).
    """

    QUALITY_THRESHOLD = 0.7

    def __init__(
        self,
        provider: Optional[BaseOCRProvider] = None,
        fallback_providers: Optional[list[BaseOCRProvider]] = None,
        quality_threshold: float = QUALITY_THRESHOLD
    ):
        """
        Initialize the OCR processor.

        Args:
            provider: Primary OCR provider (default: MistralOCRProvider)
            fallback_providers: List of fallback providers to try if primary fails
            quality_threshold: Minimum quality score to accept results
        """
        self.provider = provider or MistralOCRProvider()
        self.fallback_providers = fallback_providers or []
        self.quality_threshold = quality_threshold

    def process_image(self, image_path: str | Path, page_number: int = 1) -> OCRResult:
        """
        Process a single image with OCR.

        Args:
            image_path: Path to the image file
            page_number: Page number for tracking

        Returns:
            OCRResult with extracted text
        """
        image_path = Path(image_path)

        result = self.provider.extract_text(image_path, page_number)

        if result.quality_score >= self.quality_threshold:
            return result

        best_result = result
        for fallback in self.fallback_providers:
            if not fallback.is_available():
                continue

            fallback_result = fallback.extract_text(image_path, page_number)

            if fallback_result.quality_score > best_result.quality_score:
                best_result = fallback_result

            if fallback_result.quality_score >= self.quality_threshold:
                return fallback_result

        return best_result

    def process_pdf_page(
        self,
        pdf_path: str | Path,
        page_number: int
    ) -> OCRResult:
        """
        Process a single PDF page with OCR.

        Args:
            pdf_path: Path to the PDF file
            page_number: Page number to process (1-indexed)

        Returns:
            OCRResult with extracted text
        """
        pdf_path = Path(pdf_path)

        result = self.provider.extract_text_from_pdf(pdf_path, page_number)

        if result.quality_score >= self.quality_threshold:
            return result

        best_result = result
        for fallback in self.fallback_providers:
            if not fallback.is_available():
                continue

            fallback_result = fallback.extract_text_from_pdf(pdf_path, page_number)

            if fallback_result.quality_score > best_result.quality_score:
                best_result = fallback_result

            if fallback_result.quality_score >= self.quality_threshold:
                return fallback_result

        return best_result

    def process_images_directory(
        self,
        images_dir: str | Path,
        output_dir: Optional[str | Path] = None,
        progress_callback: Optional[Callable[[int, int, OCRResult], None]] = None
    ) -> ProcessingResult:
        """
        Process all images in a directory with OCR.

        Args:
            images_dir: Directory containing images
            output_dir: Directory to save text files (optional)
            progress_callback: Callback for progress updates (current, total, result)

        Returns:
            ProcessingResult with all OCR results
        """
        images_dir = Path(images_dir)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".tiff", ".bmp"}
        image_files = sorted([
            f for f in images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ])

        results = []
        successful = 0
        failed = 0

        for idx, image_path in enumerate(image_files, start=1):
            result = self.process_image(image_path, page_number=idx)
            results.append(result)

            if result.is_successful:
                successful += 1

                if output_dir:
                    txt_path = output_dir / f"page_{idx:03d}.txt"
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(result.text)
            else:
                failed += 1

            if progress_callback:
                progress_callback(idx, len(image_files), result)

        total_quality = sum(r.quality_score for r in results)
        avg_quality = total_quality / len(results) if results else 0.0

        return ProcessingResult(
            pages=results,
            total_pages=len(image_files),
            successful_pages=successful,
            failed_pages=failed,
            average_quality=avg_quality,
            provider_used=self.provider.name
        )

    def process_pdf(
        self,
        pdf_path: str | Path,
        output_dir: Optional[str | Path] = None,
        progress_callback: Optional[Callable[[int, int, OCRResult], None]] = None
    ) -> ProcessingResult:
        """
        Process all pages of a PDF with OCR.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save text files (optional)
            progress_callback: Callback for progress updates (current, total, result)

        Returns:
            ProcessingResult with all OCR results
        """
        pdf_path = Path(pdf_path)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(self.provider, MistralOCRProvider):
            all_results = self.provider.extract_all_pages(pdf_path)

            for idx, result in enumerate(all_results, start=1):
                if output_dir:
                    txt_path = output_dir / f"page_{idx:03d}.txt"
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(result.text)

                if progress_callback:
                    progress_callback(idx, len(all_results), result)

            successful = sum(1 for r in all_results if r.is_successful)
            failed = len(all_results) - successful
            total_quality = sum(r.quality_score for r in all_results)
            avg_quality = total_quality / len(all_results) if all_results else 0.0

            return ProcessingResult(
                pages=all_results,
                total_pages=len(all_results),
                successful_pages=successful,
                failed_pages=failed,
                average_quality=avg_quality,
                provider_used=self.provider.name
            )
        else:
            import fitz
            with fitz.open(pdf_path) as doc:
                page_count = len(doc)

            results = []
            successful = 0
            failed = 0

            for page_num in range(1, page_count + 1):
                result = self.process_pdf_page(pdf_path, page_num)
                results.append(result)

                if result.is_successful:
                    successful += 1

                    if output_dir:
                        txt_path = output_dir / f"page_{page_num:03d}.txt"
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(result.text)
                else:
                    failed += 1

                if progress_callback:
                    progress_callback(page_num, page_count, result)

            total_quality = sum(r.quality_score for r in results)
            avg_quality = total_quality / len(results) if results else 0.0

            return ProcessingResult(
                pages=results,
                total_pages=page_count,
                successful_pages=successful,
                failed_pages=failed,
                average_quality=avg_quality,
                provider_used=self.provider.name
            )

    def get_combined_text(self, result: ProcessingResult) -> str:
        """
        Combine all page texts into a single document.

        Args:
            result: ProcessingResult from processing

        Returns:
            Combined text from all pages
        """
        texts = []
        for page_result in result.pages:
            if page_result.text.strip():
                texts.append(f"--- Page {page_result.page_number} ---")
                texts.append(page_result.text)
        return "\n\n".join(texts)

    def save_combined_text(
        self,
        result: ProcessingResult,
        output_path: str | Path
    ) -> None:
        """
        Save combined text from all pages to a single file.

        Args:
            result: ProcessingResult from processing
            output_path: Path for the output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        combined = self.get_combined_text(result)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(combined)

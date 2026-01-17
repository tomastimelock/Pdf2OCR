"""High-level API for PDF2OCR processing."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable

from pdf2ocr.processors.pdf_splitter import PDFSplitter, PDFMetadata, PageInfo
from pdf2ocr.processors.ocr_processor import OCRProcessor, ProcessingResult
from pdf2ocr.extractors.information_extractor import InformationExtractor, ExtractedInformation
from pdf2ocr.providers.mistral_provider import MistralOCRProvider
from pdf2ocr.providers.base import OCRResult


@dataclass
class DocumentResult:
    """Complete result of processing a PDF document."""
    pdf_path: str
    metadata: PDFMetadata
    pages: list[PageInfo] = field(default_factory=list)
    ocr_result: Optional[ProcessingResult] = None
    extracted_info: Optional[ExtractedInformation] = None
    output_dir: Optional[str] = None
    embedded_images: list[dict] = field(default_factory=list)

    @property
    def is_successful(self) -> bool:
        """Check if processing was successful."""
        return self.ocr_result is not None and self.ocr_result.is_successful

    @property
    def combined_text(self) -> str:
        """Get combined text from all pages."""
        if self.extracted_info:
            return self.extracted_info.raw_text
        elif self.ocr_result:
            return "\n\n".join(
                r.text for r in self.ocr_result.pages if r.text.strip()
            )
        return ""


class PDF2OCR:
    """
    High-level interface for PDF to OCR processing.

    Example usage:
        ```python
        from pdf2ocr import PDF2OCR

        # Initialize with API key
        processor = PDF2OCR(api_key="your-mistral-api-key")

        # Process a PDF
        result = processor.process("document.pdf", output_dir="./output")

        # Access results
        print(result.combined_text)
        print(result.extracted_info.emails)
        ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        dpi: int = 200,
        image_format: str = "jpg"
    ):
        """
        Initialize PDF2OCR processor.

        Args:
            api_key: Mistral API key. If not provided, reads from MISTRAL_API_KEY env var.
            dpi: Resolution for image conversion (default: 200)
            image_format: Output image format - 'jpg' or 'png' (default: 'jpg')
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.dpi = dpi
        self.image_format = image_format

        self.splitter = PDFSplitter(dpi=dpi, image_format=image_format)
        self.extractor = InformationExtractor()

        if self.api_key:
            provider = MistralOCRProvider(api_key=self.api_key)
            self.ocr_processor = OCRProcessor(provider=provider)
        else:
            self.ocr_processor = None

    def process(
        self,
        pdf_path: str | Path,
        output_dir: Optional[str | Path] = None,
        extract_info: bool = True,
        extract_images: bool = False,
        split_pages: bool = True,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> DocumentResult:
        """
        Process a PDF document with full pipeline.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save outputs (optional)
            extract_info: Extract structured information from text
            extract_images: Extract embedded images
            split_pages: Split PDF into page images
            progress_callback: Callback for progress (stage, current, total)

        Returns:
            DocumentResult with all processing outputs
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        metadata = self.splitter.get_metadata(pdf_path)

        result = DocumentResult(
            pdf_path=str(pdf_path),
            metadata=metadata,
            output_dir=str(output_dir) if output_dir else None
        )

        if split_pages and output_dir:
            pages_dir = output_dir / "pages"
            if progress_callback:
                progress_callback("splitting", 0, metadata.page_count)

            result.pages = self.splitter.split_to_images(pdf_path, pages_dir)

            if progress_callback:
                progress_callback("splitting", metadata.page_count, metadata.page_count)

        if extract_images and output_dir:
            images_dir = output_dir / "images"
            result.embedded_images = self.splitter.extract_images(pdf_path, images_dir)

        if self.ocr_processor:
            txt_dir = output_dir / "txt" if output_dir else None

            def ocr_progress(current, total, ocr_result):
                if progress_callback:
                    progress_callback("ocr", current, total)

            result.ocr_result = self.ocr_processor.process_pdf(
                pdf_path,
                txt_dir,
                progress_callback=ocr_progress
            )

            if output_dir:
                combined_path = output_dir / "combined.txt"
                self.ocr_processor.save_combined_text(result.ocr_result, combined_path)

            if extract_info and result.ocr_result:
                result.extracted_info = self.extractor.extract_from_result(result.ocr_result)

                if output_dir:
                    json_path = output_dir / "extracted.json"
                    self.extractor.save_to_json(result.extracted_info, json_path)

                    summary_path = output_dir / "summary.txt"
                    self.extractor.save_summary(result.extracted_info, summary_path)

        return result

    def split_pdf(
        self,
        pdf_path: str | Path,
        output_dir: str | Path,
        to_images: bool = True
    ) -> list[PageInfo]:
        """
        Split a PDF into individual pages.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save page files
            to_images: If True, convert to images; if False, create single-page PDFs

        Returns:
            List of PageInfo objects
        """
        if to_images:
            return self.splitter.split_to_images(pdf_path, output_dir)
        else:
            return self.splitter.split_to_pages(pdf_path, output_dir)

    def ocr_image(self, image_path: str | Path, page_number: int = 1) -> OCRResult:
        """
        Perform OCR on a single image.

        Args:
            image_path: Path to the image file
            page_number: Page number for tracking

        Returns:
            OCRResult with extracted text
        """
        if not self.ocr_processor:
            raise RuntimeError("OCR not available. Provide API key to enable OCR.")

        return self.ocr_processor.process_image(image_path, page_number)

    def ocr_pdf(
        self,
        pdf_path: str | Path,
        output_dir: Optional[str | Path] = None
    ) -> ProcessingResult:
        """
        Perform OCR on a PDF document.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save text files (optional)

        Returns:
            ProcessingResult with OCR results
        """
        if not self.ocr_processor:
            raise RuntimeError("OCR not available. Provide API key to enable OCR.")

        return self.ocr_processor.process_pdf(pdf_path, output_dir)

    def extract_information(self, ocr_result: ProcessingResult) -> ExtractedInformation:
        """
        Extract structured information from OCR results.

        Args:
            ocr_result: ProcessingResult from OCR processing

        Returns:
            ExtractedInformation with extracted data
        """
        return self.extractor.extract_from_result(ocr_result)

    def get_metadata(self, pdf_path: str | Path) -> PDFMetadata:
        """
        Get metadata from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            PDFMetadata object
        """
        return self.splitter.get_metadata(pdf_path)


def process_pdf(
    pdf_path: str | Path,
    output_dir: Optional[str | Path] = None,
    api_key: Optional[str] = None,
    extract_info: bool = True,
    dpi: int = 200
) -> DocumentResult:
    """
    Convenience function to process a PDF with default settings.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save outputs
        api_key: Mistral API key
        extract_info: Extract structured information
        dpi: Resolution for image conversion

    Returns:
        DocumentResult with processing results
    """
    processor = PDF2OCR(api_key=api_key, dpi=dpi)
    return processor.process(pdf_path, output_dir, extract_info=extract_info)


def process_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    api_key: Optional[str] = None,
    extract_info: bool = True,
    dpi: int = 200,
    progress_callback: Optional[Callable[[str, int, int], None]] = None
) -> list[DocumentResult]:
    """
    Process all PDF files in a directory.

    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save outputs
        api_key: Mistral API key
        extract_info: Extract structured information
        dpi: Resolution for image conversion
        progress_callback: Callback for progress (pdf_name, current_pdf, total_pdfs)

    Returns:
        List of DocumentResult objects
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    pdf_files = list(input_dir.glob("*.pdf"))

    if not pdf_files:
        return []

    processor = PDF2OCR(api_key=api_key, dpi=dpi)
    results = []

    for idx, pdf_path in enumerate(pdf_files, start=1):
        if progress_callback:
            progress_callback(pdf_path.name, idx, len(pdf_files))

        doc_output = output_dir / pdf_path.stem
        result = processor.process(pdf_path, doc_output, extract_info=extract_info)
        results.append(result)

    return results

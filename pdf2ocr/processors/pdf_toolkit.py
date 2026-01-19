"""PDF Toolkit - High-level PDF manipulation utilities."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Union

from pdf2ocr.processors.pdf_splitter import PDFSplitter, PDFMetadata, PageInfo


@dataclass
class PDFDocument:
    """Represents a PDF document with its metadata and content."""
    path: Path
    metadata: PDFMetadata
    has_text: bool
    pages: List[PageInfo] = field(default_factory=list)


@dataclass
class MergeResult:
    """Result of merging multiple PDFs."""
    output_path: Path
    metadata: PDFMetadata
    source_count: int
    total_pages: int
    success: bool
    error: Optional[str] = None


@dataclass
class SplitResult:
    """Result of splitting a PDF."""
    source_path: Path
    output_dir: Path
    pages: List[PageInfo]
    total_pages: int
    success: bool
    error: Optional[str] = None


class PDFToolkit:
    """
    High-level PDF toolkit for document manipulation.

    Provides convenient methods for common PDF operations:
    - Document analysis and metadata extraction
    - Splitting PDFs to images or pages
    - Merging multiple PDFs
    - Batch processing
    - Table extraction
    """

    def __init__(self, dpi: int = 200, image_format: str = "jpg"):
        """
        Initialize the PDF toolkit.

        Args:
            dpi: Resolution for image conversion
            image_format: Output format for images (jpg, png)
        """
        self.splitter = PDFSplitter(dpi=dpi, image_format=image_format)
        self.dpi = dpi
        self.image_format = image_format

    def analyze(self, pdf_path: Union[str, Path]) -> PDFDocument:
        """
        Analyze a PDF document and return its information.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            PDFDocument with metadata and page info
        """
        pdf_path = Path(pdf_path)
        metadata = self.splitter.get_metadata(pdf_path)
        has_text = self.splitter.has_embedded_text(pdf_path)

        return PDFDocument(
            path=pdf_path,
            metadata=metadata,
            has_text=has_text
        )

    def split_to_images(
        self,
        pdf_path: Union[str, Path],
        output_dir: Union[str, Path]
    ) -> SplitResult:
        """
        Split a PDF into image files.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory for output images

        Returns:
            SplitResult with page information
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)

        try:
            pages = self.splitter.split_to_images(pdf_path, output_dir)
            return SplitResult(
                source_path=pdf_path,
                output_dir=output_dir,
                pages=pages,
                total_pages=len(pages),
                success=True
            )
        except Exception as e:
            return SplitResult(
                source_path=pdf_path,
                output_dir=output_dir,
                pages=[],
                total_pages=0,
                success=False,
                error=str(e)
            )

    def split_to_pdfs(
        self,
        pdf_path: Union[str, Path],
        output_dir: Union[str, Path]
    ) -> SplitResult:
        """
        Split a PDF into individual single-page PDFs.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory for output PDFs

        Returns:
            SplitResult with page information
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)

        try:
            pages = self.splitter.split_to_pages(pdf_path, output_dir)
            return SplitResult(
                source_path=pdf_path,
                output_dir=output_dir,
                pages=pages,
                total_pages=len(pages),
                success=True
            )
        except Exception as e:
            return SplitResult(
                source_path=pdf_path,
                output_dir=output_dir,
                pages=[],
                total_pages=0,
                success=False,
                error=str(e)
            )

    def merge(
        self,
        pdf_paths: List[Union[str, Path]],
        output_path: Union[str, Path]
    ) -> MergeResult:
        """
        Merge multiple PDFs into a single document.

        Args:
            pdf_paths: List of PDF files to merge (in order)
            output_path: Path for the merged output

        Returns:
            MergeResult with merged document info
        """
        output_path = Path(output_path)
        pdf_paths = [Path(p) for p in pdf_paths]

        try:
            metadata = self.splitter.merge_pdfs(pdf_paths, output_path)
            return MergeResult(
                output_path=output_path,
                metadata=metadata,
                source_count=len(pdf_paths),
                total_pages=metadata.page_count,
                success=True
            )
        except Exception as e:
            return MergeResult(
                output_path=output_path,
                metadata=PDFMetadata(),
                source_count=len(pdf_paths),
                total_pages=0,
                success=False,
                error=str(e)
            )

    def batch_split_to_images(
        self,
        pdf_paths: List[Union[str, Path]],
        output_base_dir: Union[str, Path]
    ) -> dict[str, SplitResult]:
        """
        Split multiple PDFs to images in batch.

        Args:
            pdf_paths: List of PDF files
            output_base_dir: Base directory (subdirs created per PDF)

        Returns:
            Dict mapping PDF name to SplitResult
        """
        output_base_dir = Path(output_base_dir)
        results = {}

        for pdf_path in pdf_paths:
            pdf_path = Path(pdf_path)
            output_dir = output_base_dir / pdf_path.stem
            result = self.split_to_images(pdf_path, output_dir)
            results[pdf_path.stem] = result

        return results

    def extract_text(
        self,
        pdf_path: Union[str, Path],
        page_number: Optional[int] = None
    ) -> str:
        """
        Extract embedded text from a PDF.

        Args:
            pdf_path: Path to the PDF file
            page_number: Specific page (1-indexed), or None for all

        Returns:
            Extracted text
        """
        return self.splitter.extract_text_direct(pdf_path, page_number)

    def extract_images(
        self,
        pdf_path: Union[str, Path],
        output_dir: Union[str, Path],
        min_size: int = 50
    ) -> List[dict]:
        """
        Extract embedded images from a PDF.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory for extracted images
            min_size: Minimum image dimension to extract

        Returns:
            List of image metadata dicts
        """
        return self.splitter.extract_images(pdf_path, output_dir, min_size)

    def extract_tables(
        self,
        pdf_path: Union[str, Path],
        page_numbers: Optional[List[int]] = None
    ) -> List[dict]:
        """
        Extract tables from a PDF.

        Args:
            pdf_path: Path to the PDF file
            page_numbers: Specific pages (1-indexed), or None for all

        Returns:
            List of table dicts with headers and rows
        """
        return self.splitter.extract_tables_raw(pdf_path, page_numbers)

    def get_page_count(self, pdf_path: Union[str, Path]) -> int:
        """Get the number of pages in a PDF."""
        return self.splitter.get_page_count(pdf_path)

    def get_metadata(self, pdf_path: Union[str, Path]) -> PDFMetadata:
        """Get PDF metadata."""
        return self.splitter.get_metadata(pdf_path)

    def is_scanned(self, pdf_path: Union[str, Path]) -> bool:
        """Check if PDF is scanned (no embedded text)."""
        return not self.splitter.has_embedded_text(pdf_path)

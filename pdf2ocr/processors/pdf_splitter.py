"""PDF Splitter - Split PDFs into pages and convert to images."""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image


@dataclass
class PDFMetadata:
    """Metadata extracted from a PDF document."""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    page_count: int = 0


@dataclass
class PageInfo:
    """Information about a single PDF page."""
    page_number: int
    width: float
    height: float
    has_text: bool
    image_path: Optional[str] = None
    pdf_path: Optional[str] = None


class PDFSplitter:
    """
    Split PDF documents into individual pages and convert to images.

    Uses PyMuPDF (fitz) for PDF processing and PIL for image handling.
    """

    DEFAULT_DPI = 200
    DEFAULT_FORMAT = "jpg"

    def __init__(self, dpi: int = DEFAULT_DPI, image_format: str = DEFAULT_FORMAT):
        """
        Initialize the PDF splitter.

        Args:
            dpi: Resolution for image conversion (default: 200)
            image_format: Output image format - 'jpg', 'png', etc. (default: 'jpg')
        """
        self.dpi = dpi
        self.image_format = image_format.lower()

    def get_metadata(self, pdf_path: str | Path) -> PDFMetadata:
        """
        Extract metadata from a PDF document.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            PDFMetadata object with document information
        """
        pdf_path = Path(pdf_path)

        with fitz.open(pdf_path) as doc:
            metadata = doc.metadata or {}
            return PDFMetadata(
                title=metadata.get("title"),
                author=metadata.get("author"),
                subject=metadata.get("subject"),
                creator=metadata.get("creator"),
                producer=metadata.get("producer"),
                creation_date=metadata.get("creationDate"),
                modification_date=metadata.get("modDate"),
                page_count=len(doc)
            )

    def get_page_count(self, pdf_path: str | Path) -> int:
        """
        Get the number of pages in a PDF document.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Number of pages
        """
        with fitz.open(pdf_path) as doc:
            return len(doc)

    def has_embedded_text(self, pdf_path: str | Path) -> bool:
        """
        Check if PDF has embedded (selectable) text or is scanned.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            True if PDF has embedded text, False if scanned/image-only
        """
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text = page.get_text().strip()
                if text:
                    return True
        return False

    def extract_text_direct(self, pdf_path: str | Path, page_number: Optional[int] = None) -> str:
        """
        Extract text directly from PDF (works for PDFs with embedded text).

        Args:
            pdf_path: Path to the PDF file
            page_number: Specific page to extract (1-indexed), or None for all pages

        Returns:
            Extracted text
        """
        texts = []
        with fitz.open(pdf_path) as doc:
            if page_number is not None:
                page_idx = page_number - 1
                if 0 <= page_idx < len(doc):
                    texts.append(doc[page_idx].get_text())
            else:
                for page in doc:
                    texts.append(page.get_text())
        return "\n\n".join(texts)

    def split_to_pages(self, pdf_path: str | Path, output_dir: str | Path) -> list[PageInfo]:
        """
        Split a PDF into individual single-page PDF files.

        Args:
            pdf_path: Path to the input PDF file
            output_dir: Directory to save the page PDF files

        Returns:
            List of PageInfo objects for each page
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pages = []

        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                page_pdf_path = output_dir / f"page_{page_num:03d}.pdf"

                new_doc = fitz.open()
                new_doc.insert_pdf(doc, from_page=page_num-1, to_page=page_num-1)
                new_doc.save(str(page_pdf_path))
                new_doc.close()

                rect = page.rect
                has_text = bool(page.get_text().strip())

                pages.append(PageInfo(
                    page_number=page_num,
                    width=rect.width,
                    height=rect.height,
                    has_text=has_text,
                    pdf_path=str(page_pdf_path)
                ))

        return pages

    def split_to_images(self, pdf_path: str | Path, output_dir: str | Path) -> list[PageInfo]:
        """
        Convert PDF pages to images (JPG/PNG).

        Args:
            pdf_path: Path to the input PDF file
            output_dir: Directory to save the image files

        Returns:
            List of PageInfo objects for each page with image paths
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pages = []
        zoom_factor = self.dpi / 72  # 72 is PDF base DPI
        matrix = fitz.Matrix(zoom_factor, zoom_factor)

        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                pixmap = page.get_pixmap(matrix=matrix)

                ext = self.image_format
                if ext == "jpg":
                    ext = "jpeg"

                image_path = output_dir / f"page_{page_num:03d}.{self.image_format}"

                if self.image_format in ("jpg", "jpeg"):
                    img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                    img.save(str(image_path), "JPEG", quality=95)
                else:
                    pixmap.save(str(image_path))

                rect = page.rect
                has_text = bool(page.get_text().strip())

                pages.append(PageInfo(
                    page_number=page_num,
                    width=rect.width,
                    height=rect.height,
                    has_text=has_text,
                    image_path=str(image_path)
                ))

        return pages

    def convert_page_to_image(
        self,
        pdf_path: str | Path,
        page_number: int,
        output_path: str | Path
    ) -> PageInfo:
        """
        Convert a single PDF page to an image.

        Args:
            pdf_path: Path to the PDF file
            page_number: Page number to convert (1-indexed)
            output_path: Path for the output image

        Returns:
            PageInfo object with image details
        """
        pdf_path = Path(pdf_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        zoom_factor = self.dpi / 72
        matrix = fitz.Matrix(zoom_factor, zoom_factor)

        with fitz.open(pdf_path) as doc:
            page_idx = page_number - 1
            if page_idx < 0 or page_idx >= len(doc):
                raise ValueError(f"Page {page_number} does not exist. PDF has {len(doc)} pages.")

            page = doc[page_idx]
            pixmap = page.get_pixmap(matrix=matrix)

            output_format = output_path.suffix.lower().lstrip(".")
            if output_format in ("jpg", "jpeg"):
                img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                img.save(str(output_path), "JPEG", quality=95)
            else:
                pixmap.save(str(output_path))

            rect = page.rect
            has_text = bool(page.get_text().strip())

            return PageInfo(
                page_number=page_number,
                width=rect.width,
                height=rect.height,
                has_text=has_text,
                image_path=str(output_path)
            )

    def extract_images(
        self,
        pdf_path: str | Path,
        output_dir: str | Path,
        min_size: int = 50
    ) -> list[dict]:
        """
        Extract embedded images from a PDF.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted images
            min_size: Minimum width/height to extract (default: 50px)

        Returns:
            List of dicts with image metadata
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        extracted = []

        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                image_list = page.get_images()

                for img_idx, img_info in enumerate(image_list, start=1):
                    xref = img_info[0]

                    try:
                        base_image = doc.extract_image(xref)
                        if not base_image:
                            continue

                        width = base_image["width"]
                        height = base_image["height"]

                        if width < min_size or height < min_size:
                            continue

                        ext = base_image["ext"]
                        image_bytes = base_image["image"]

                        img_filename = f"page_{page_num:03d}_img_{img_idx:03d}.{ext}"
                        img_path = output_dir / img_filename

                        with open(img_path, "wb") as f:
                            f.write(image_bytes)

                        extracted.append({
                            "page_number": page_num,
                            "image_index": img_idx,
                            "width": width,
                            "height": height,
                            "format": ext,
                            "path": str(img_path),
                            "size_bytes": len(image_bytes)
                        })
                    except Exception:
                        continue

        return extracted

    def merge_pdfs(
        self,
        pdf_paths: list[str | Path],
        output_path: str | Path
    ) -> PDFMetadata:
        """
        Merge multiple PDF files into a single PDF.

        Args:
            pdf_paths: List of paths to PDF files to merge
            output_path: Path for the merged output PDF

        Returns:
            PDFMetadata for the merged document
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        merged_doc = fitz.open()
        total_pages = 0

        for pdf_path in pdf_paths:
            pdf_path = Path(pdf_path)
            with fitz.open(pdf_path) as doc:
                merged_doc.insert_pdf(doc)
                total_pages += len(doc)

        merged_doc.save(str(output_path))
        merged_doc.close()

        return self.get_metadata(output_path)

    def batch_split(
        self,
        pdf_paths: list[str | Path],
        output_base_dir: str | Path,
        to_images: bool = True
    ) -> dict[str, list[PageInfo]]:
        """
        Split multiple PDFs to images or pages in batch.

        Args:
            pdf_paths: List of PDF file paths
            output_base_dir: Base directory for outputs (subdirs created per PDF)
            to_images: If True, convert to images; if False, split to PDFs

        Returns:
            Dict mapping PDF filename to list of PageInfo
        """
        output_base_dir = Path(output_base_dir)
        results = {}

        for pdf_path in pdf_paths:
            pdf_path = Path(pdf_path)
            pdf_name = pdf_path.stem
            output_dir = output_base_dir / pdf_name

            if to_images:
                pages = self.split_to_images(pdf_path, output_dir)
            else:
                pages = self.split_to_pages(pdf_path, output_dir)

            results[pdf_name] = pages

        return results

    def extract_tables_raw(
        self,
        pdf_path: str | Path,
        page_numbers: Optional[list[int]] = None
    ) -> list[dict]:
        """
        Extract raw table data from PDF using pdfplumber.

        Args:
            pdf_path: Path to the PDF file
            page_numbers: Specific pages to extract from (1-indexed), or None for all

        Returns:
            List of table dicts with page_number, table_index, headers, and rows
        """
        try:
            import pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber is required for table extraction. "
                "Install with: pip install pdfplumber"
            )

        pdf_path = Path(pdf_path)
        tables = []

        with pdfplumber.open(pdf_path) as pdf:
            pages_to_process = range(len(pdf.pages))
            if page_numbers:
                pages_to_process = [p - 1 for p in page_numbers if 0 < p <= len(pdf.pages)]

            for page_idx in pages_to_process:
                page = pdf.pages[page_idx]
                page_tables = page.extract_tables()

                for table_idx, table_data in enumerate(page_tables, start=1):
                    if not table_data or len(table_data) < 2:
                        continue

                    headers = table_data[0] if table_data else []
                    rows = table_data[1:] if len(table_data) > 1 else []

                    tables.append({
                        "page_number": page_idx + 1,
                        "table_index": table_idx,
                        "headers": headers,
                        "rows": rows,
                        "row_count": len(rows),
                        "column_count": len(headers) if headers else 0
                    })

        return tables

    def get_page_info(self, pdf_path: str | Path, page_number: int) -> PageInfo:
        """
        Get information about a specific page.

        Args:
            pdf_path: Path to the PDF file
            page_number: Page number (1-indexed)

        Returns:
            PageInfo for the specified page
        """
        with fitz.open(pdf_path) as doc:
            page_idx = page_number - 1
            if page_idx < 0 or page_idx >= len(doc):
                raise ValueError(f"Page {page_number} does not exist")

            page = doc[page_idx]
            rect = page.rect

            return PageInfo(
                page_number=page_number,
                width=rect.width,
                height=rect.height,
                has_text=bool(page.get_text().strip())
            )

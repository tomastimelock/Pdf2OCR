"""
Document Structurer - Create comprehensive JSON output from processed documents.

Combines OCR text, extracted tables, and regenerated charts into a single
structured JSON document.
"""

import json
import hashlib
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Content from a single page."""
    page_number: int
    text: str
    word_count: int = 0
    has_tables: bool = False
    has_charts: bool = False
    has_images: bool = False
    table_count: int = 0
    chart_count: int = 0

    def __post_init__(self):
        if self.text:
            self.word_count = len(self.text.split())


@dataclass
class DocumentSection:
    """A detected section in the document."""
    title: str
    level: int  # 1 = main heading, 2 = subheading, etc.
    start_page: int
    end_page: Optional[int] = None


@dataclass
class DocumentMetadata:
    """Metadata about the document."""
    filename: str
    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[str] = None
    modified_date: Optional[str] = None
    page_count: int = 0
    file_hash: Optional[str] = None
    processed_at: Optional[str] = None


@dataclass
class StructuredDocument:
    """Complete structured document output."""
    document_id: str
    metadata: DocumentMetadata
    content: Dict[str, Any] = field(default_factory=dict)
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    processing_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "metadata": asdict(self.metadata),
            "content": self.content,
            "extracted_data": self.extracted_data,
            "processing_summary": self.processing_summary
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)


class DocumentStructurer:
    """
    Create structured JSON output from processed document components.

    Combines:
    - OCR text from all pages
    - Extracted tables (as JSON data)
    - Regenerated charts (with SVG paths and descriptions)
    - Extracted images metadata
    - Document sections and structure
    """

    # Heading patterns for section detection
    HEADING_PATTERNS = [
        (r'^#{1,6}\s+(.+)$', 'markdown'),  # Markdown headings
        (r'^([A-Z][A-Z\s]{2,50})$', 'allcaps'),  # ALL CAPS headings
        (r'^(\d+\.?\s+[A-Z].{5,80})$', 'numbered'),  # Numbered headings
        (r'^([A-Z][a-z].{5,80}):?\s*$', 'title_case'),  # Title Case headings
    ]

    def __init__(self):
        """Initialize the document structurer."""
        pass

    def structure_document(
        self,
        document_dir: str | Path,
        pdf_path: Optional[str | Path] = None,
        output_path: Optional[str | Path] = None
    ) -> StructuredDocument:
        """
        Create structured JSON from processed document directory.

        Expected directory structure:
        document_dir/
            pages/          # Page images (optional)
            txt/            # OCR text files (page_001.txt, etc.)
            json/           # Table JSON files
            svg/            # Chart/table SVG files
            images/         # Extracted images
            combined.txt    # Combined OCR text (optional)

        Args:
            document_dir: Directory containing processed document files
            pdf_path: Original PDF path for metadata extraction
            output_path: Path to save structured JSON (optional)

        Returns:
            StructuredDocument object
        """
        document_dir = Path(document_dir)

        # Generate document ID
        doc_id = self._generate_document_id(document_dir, pdf_path)

        # Build metadata
        metadata = self._build_metadata(document_dir, pdf_path)

        # Load page texts
        pages = self._load_pages(document_dir)

        # Load tables
        tables = self._load_tables(document_dir)

        # Load charts
        charts = self._load_charts(document_dir)

        # Load images metadata
        images = self._load_images(document_dir)

        # Detect sections
        full_text = "\n\n".join(p.text for p in pages if p.text)
        sections = self._detect_sections(full_text, pages)

        # Mark pages with content types
        self._mark_page_content(pages, tables, charts, images)

        # Build content structure
        content = {
            "full_text": full_text,
            "pages": [asdict(p) for p in pages],
            "sections": [asdict(s) for s in sections]
        }

        # Build extracted data structure
        extracted_data = {
            "tables": tables,
            "charts": charts,
            "images": images
        }

        # Build processing summary
        summary = {
            "total_pages": len(pages),
            "total_words": sum(p.word_count for p in pages),
            "total_tables": len(tables),
            "total_charts": len(charts),
            "total_images": len(images),
            "pages_with_tables": len(set(t.get("page_number") for t in tables)),
            "pages_with_charts": len(set(c.get("page_number") for c in charts)),
            "sections_detected": len(sections)
        }

        # Create structured document
        doc = StructuredDocument(
            document_id=doc_id,
            metadata=metadata,
            content=content,
            extracted_data=extracted_data,
            processing_summary=summary
        )

        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(doc.to_json())
            logger.info(f"Saved structured document: {output_path}")

        return doc

    def _generate_document_id(
        self,
        document_dir: Path,
        pdf_path: Optional[Path]
    ) -> str:
        """Generate a unique document ID."""
        if pdf_path:
            pdf_path = Path(pdf_path)
            if pdf_path.exists():
                with open(pdf_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()[:12]
                return f"{pdf_path.stem}_{file_hash}"

        return f"{document_dir.name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    def _build_metadata(
        self,
        document_dir: Path,
        pdf_path: Optional[Path]
    ) -> DocumentMetadata:
        """Build document metadata."""
        metadata = DocumentMetadata(
            filename=document_dir.name,
            processed_at=datetime.now().isoformat()
        )

        if pdf_path:
            pdf_path = Path(pdf_path)
            metadata.filename = pdf_path.name

            if pdf_path.exists():
                # Calculate file hash
                with open(pdf_path, "rb") as f:
                    metadata.file_hash = hashlib.md5(f.read()).hexdigest()

                # Extract PDF metadata using PyMuPDF
                try:
                    import fitz
                    with fitz.open(pdf_path) as doc:
                        pdf_meta = doc.metadata or {}
                        metadata.title = pdf_meta.get("title")
                        metadata.author = pdf_meta.get("author")
                        metadata.created_date = pdf_meta.get("creationDate")
                        metadata.modified_date = pdf_meta.get("modDate")
                        metadata.page_count = len(doc)
                except Exception as e:
                    logger.warning(f"Could not extract PDF metadata: {e}")

        return metadata

    def _load_pages(self, document_dir: Path) -> List[PageContent]:
        """Load OCR text from page files."""
        pages = []

        # Try txt subdirectory first, then direct txt files
        txt_dirs = [
            document_dir / "txt" / "combined",
            document_dir / "txt",
            document_dir
        ]

        txt_files = []
        for txt_dir in txt_dirs:
            if txt_dir.exists():
                txt_files = sorted(txt_dir.glob("page_*.txt"))
                if txt_files:
                    break

        for txt_file in txt_files:
            try:
                # Extract page number from filename
                match = re.search(r'page_(\d+)', txt_file.stem)
                page_num = int(match.group(1)) if match else len(pages) + 1

                with open(txt_file, "r", encoding="utf-8") as f:
                    text = f.read()

                pages.append(PageContent(
                    page_number=page_num,
                    text=text
                ))

            except Exception as e:
                logger.warning(f"Failed to load {txt_file}: {e}")

        # Sort by page number
        pages.sort(key=lambda p: p.page_number)

        return pages

    def _load_tables(self, document_dir: Path) -> List[Dict[str, Any]]:
        """Load table data from JSON files."""
        tables = []

        json_dir = document_dir / "json"
        if not json_dir.exists():
            return tables

        for json_file in sorted(json_dir.glob("*_table_*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    table_data = json.load(f)

                # Add SVG path if exists
                svg_dir = document_dir / "svg"
                svg_name = json_file.stem + ".svg"
                svg_path = svg_dir / svg_name
                if svg_path.exists():
                    table_data["svg_path"] = str(svg_path)

                tables.append(table_data)

            except Exception as e:
                logger.warning(f"Failed to load table {json_file}: {e}")

        return tables

    def _load_charts(self, document_dir: Path) -> List[Dict[str, Any]]:
        """Load chart data from SVG files."""
        charts = []

        svg_dir = document_dir / "svg"
        if not svg_dir.exists():
            return charts

        for svg_file in sorted(svg_dir.glob("*_chart_*.svg")):
            try:
                # Extract page number and chart index from filename
                match = re.search(r'page_(\d+)_chart_(\d+)', svg_file.stem)
                if match:
                    page_num = int(match.group(1))
                    chart_idx = int(match.group(2))
                else:
                    page_num = 0
                    chart_idx = len(charts) + 1

                # Read SVG content
                with open(svg_file, "r", encoding="utf-8") as f:
                    svg_content = f.read()

                # Extract viewBox dimensions
                width, height = 400, 300
                viewbox_match = re.search(
                    r'viewBox="[\d.\-]+\s+[\d.\-]+\s+([\d.]+)\s+([\d.]+)"',
                    svg_content
                )
                if viewbox_match:
                    width = int(float(viewbox_match.group(1)))
                    height = int(float(viewbox_match.group(2)))

                charts.append({
                    "chart_id": chart_idx,
                    "page_number": page_num,
                    "svg_path": str(svg_file),
                    "width": width,
                    "height": height,
                    "svg_content": svg_content
                })

            except Exception as e:
                logger.warning(f"Failed to load chart {svg_file}: {e}")

        return charts

    def _load_images(self, document_dir: Path) -> List[Dict[str, Any]]:
        """Load image metadata."""
        images = []

        images_dir = document_dir / "images"
        if not images_dir.exists():
            return images

        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

        for img_file in sorted(images_dir.iterdir()):
            if img_file.suffix.lower() not in image_extensions:
                continue

            try:
                # Extract page number from filename
                match = re.search(r'page_(\d+)', img_file.stem)
                page_num = int(match.group(1)) if match else 0

                # Get image dimensions
                try:
                    from PIL import Image
                    with Image.open(img_file) as img:
                        width, height = img.size
                except Exception:
                    width, height = 0, 0

                images.append({
                    "filename": img_file.name,
                    "page_number": page_num,
                    "file_path": str(img_file),
                    "width": width,
                    "height": height,
                    "size_bytes": img_file.stat().st_size
                })

            except Exception as e:
                logger.warning(f"Failed to load image metadata {img_file}: {e}")

        return images

    def _detect_sections(
        self,
        full_text: str,
        pages: List[PageContent]
    ) -> List[DocumentSection]:
        """Detect document sections from headings."""
        sections = []

        if not full_text:
            return sections

        lines = full_text.split('\n')
        current_page = 1
        chars_processed = 0

        for line in lines:
            line = line.strip()
            if not line:
                chars_processed += 1
                continue

            # Check against heading patterns
            for pattern, pattern_type in self.HEADING_PATTERNS:
                match = re.match(pattern, line)
                if match:
                    title = match.group(1).strip()

                    # Determine heading level
                    if pattern_type == 'markdown':
                        level = len(line) - len(line.lstrip('#'))
                    elif pattern_type == 'allcaps':
                        level = 1
                    elif pattern_type == 'numbered':
                        level = line.count('.') + 1
                    else:
                        level = 2

                    # Estimate page number
                    page_num = self._estimate_page_number(chars_processed, pages)

                    sections.append(DocumentSection(
                        title=title,
                        level=level,
                        start_page=page_num
                    ))
                    break

            chars_processed += len(line) + 1

        # Set end pages
        for i, section in enumerate(sections[:-1]):
            section.end_page = sections[i + 1].start_page

        if sections:
            sections[-1].end_page = len(pages)

        return sections

    def _estimate_page_number(
        self,
        char_position: int,
        pages: List[PageContent]
    ) -> int:
        """Estimate which page a character position falls on."""
        total_chars = 0
        for page in pages:
            total_chars += len(page.text) + 2  # +2 for newlines between pages
            if char_position < total_chars:
                return page.page_number
        return pages[-1].page_number if pages else 1

    def _mark_page_content(
        self,
        pages: List[PageContent],
        tables: List[Dict],
        charts: List[Dict],
        images: List[Dict]
    ) -> None:
        """Mark pages with their content types."""
        # Create page lookup
        page_lookup = {p.page_number: p for p in pages}

        # Mark tables
        for table in tables:
            page_num = table.get("page_number", 0)
            if page_num in page_lookup:
                page_lookup[page_num].has_tables = True
                page_lookup[page_num].table_count += 1

        # Mark charts
        for chart in charts:
            page_num = chart.get("page_number", 0)
            if page_num in page_lookup:
                page_lookup[page_num].has_charts = True
                page_lookup[page_num].chart_count += 1

        # Mark images
        for image in images:
            page_num = image.get("page_number", 0)
            if page_num in page_lookup:
                page_lookup[page_num].has_images = True

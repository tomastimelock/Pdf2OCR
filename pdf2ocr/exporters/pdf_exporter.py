"""
PDF Document Exporter - Export processed documents to PDF format.

Creates PDF documents with text, tables, images, and SVG content.
"""

import io
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class PDFExporter:
    """
    Export processed document content to PDF format.

    Creates a replica of the original document with:
    - OCR text content
    - Tables (formatted)
    - Images (original or regenerated)
    - Charts (SVG rendered to images)

    Uses ReportLab for PDF generation.
    """

    # Page settings
    PAGE_WIDTH = 612  # Letter size
    PAGE_HEIGHT = 792
    MARGIN = 72  # 1 inch

    # Font settings
    TITLE_SIZE = 18
    HEADING1_SIZE = 14
    HEADING2_SIZE = 12
    BODY_SIZE = 10
    TABLE_SIZE = 9

    def __init__(self):
        """Initialize the PDF exporter."""
        pass

    def export(
        self,
        document_dir: str | Path,
        output_path: str | Path,
        structured_data: Optional[Dict[str, Any]] = None,
        include_regenerated_images: bool = True,
        include_original_images: bool = False
    ) -> str:
        """
        Export document content to PDF format.

        Args:
            document_dir: Directory containing processed document files
            output_path: Path for output .pdf file
            structured_data: Optional pre-loaded structured document data
            include_regenerated_images: Include AI-regenerated images
            include_original_images: Include original extracted images

        Returns:
            Path to the created PDF document
        """
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Image,
            Table, TableStyle, PageBreak
        )
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        document_dir = Path(document_dir)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load structured data if not provided
        if structured_data is None:
            structured_data = self._load_structured_data(document_dir)

        # Create PDF document
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=self.MARGIN,
            leftMargin=self.MARGIN,
            topMargin=self.MARGIN,
            bottomMargin=self.MARGIN
        )

        # Create styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name='DocTitle',
            parent=styles['Heading1'],
            fontSize=self.TITLE_SIZE,
            alignment=TA_CENTER,
            spaceAfter=20
        ))
        styles.add(ParagraphStyle(
            name='PageHeading',
            parent=styles['Heading2'],
            fontSize=self.HEADING1_SIZE,
            spaceBefore=12,
            spaceAfter=6
        ))
        styles.add(ParagraphStyle(
            name='DocBody',
            parent=styles['Normal'],
            fontSize=self.BODY_SIZE,
            spaceBefore=3,
            spaceAfter=3
        ))
        styles.add(ParagraphStyle(
            name='Caption',
            parent=styles['Italic'],
            fontSize=self.TABLE_SIZE,
            spaceBefore=6,
            spaceAfter=3
        ))

        # Build content
        story = []

        # Add title
        title = structured_data.get("metadata", {}).get("title") or document_dir.name
        story.append(Paragraph(self._escape_text(title), styles['DocTitle']))
        story.append(Spacer(1, 0.25 * inch))

        # Add metadata
        self._add_metadata(story, styles, structured_data)

        # Process content page by page
        pages_content = structured_data.get("content", {}).get("pages", [])
        tables_data = structured_data.get("extracted_data", {}).get("tables", [])
        charts_data = structured_data.get("extracted_data", {}).get("charts", [])
        images_data = structured_data.get("extracted_data", {}).get("images", [])

        for page in pages_content:
            page_num = page.get("page_number", 0)

            # Add page break (except for first page)
            if page_num > 1:
                story.append(PageBreak())

            # Add page heading
            story.append(Paragraph(f"Page {page_num}", styles['PageHeading']))

            # Add page text
            text = page.get("text", "")
            if text:
                self._add_text_content(story, styles, text)

            # Add tables for this page
            page_tables = [t for t in tables_data if t.get("page_number") == page_num]
            for table in page_tables:
                self._add_table(story, styles, table)

            # Add charts for this page
            page_charts = [c for c in charts_data if c.get("page_number") == page_num]
            for chart in page_charts:
                self._add_chart(story, styles, chart, document_dir)

            # Add images
            if include_regenerated_images:
                self._add_regenerated_images(story, styles, page_num, document_dir)

            if include_original_images:
                page_images = [i for i in images_data if i.get("page_number") == page_num]
                for image in page_images:
                    self._add_image(story, styles, image, document_dir)

        # Build PDF
        doc.build(story)
        logger.info(f"PDF document saved: {output_path}")

        return str(output_path)

    def _escape_text(self, text: str) -> str:
        """Escape text for ReportLab."""
        if not text:
            return ""
        # Escape XML special characters
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        return text

    def _load_structured_data(self, document_dir: Path) -> Dict[str, Any]:
        """Load structured document data from JSON file."""
        import json

        json_path = document_dir / "document.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)

        return self._build_basic_structure(document_dir)

    def _build_basic_structure(self, document_dir: Path) -> Dict[str, Any]:
        """Build basic structure from available files."""
        import json

        structure = {
            "metadata": {"title": document_dir.name},
            "content": {"pages": []},
            "extracted_data": {"tables": [], "charts": [], "images": []}
        }

        # Load text files
        txt_dir = document_dir / "txt"
        if txt_dir.exists():
            for txt_file in sorted(txt_dir.glob("page_*.txt")):
                match = re.search(r'page_(\d+)', txt_file.stem)
                page_num = int(match.group(1)) if match else len(structure["content"]["pages"]) + 1

                with open(txt_file, "r", encoding="utf-8") as f:
                    text = f.read()

                structure["content"]["pages"].append({
                    "page_number": page_num,
                    "text": text
                })

        # Load tables
        json_dir = document_dir / "json"
        if json_dir.exists():
            for json_file in sorted(json_dir.glob("*_table_*.json")):
                with open(json_file, "r", encoding="utf-8") as f:
                    table_data = json.load(f)
                structure["extracted_data"]["tables"].append(table_data)

        # Load charts
        svg_dir = document_dir / "svg"
        if svg_dir.exists():
            for svg_file in sorted(svg_dir.glob("*_chart_*.svg")):
                match = re.search(r'page_(\d+)_chart_(\d+)', svg_file.stem)
                if match:
                    structure["extracted_data"]["charts"].append({
                        "page_number": int(match.group(1)),
                        "svg_path": str(svg_file)
                    })

        return structure

    def _add_metadata(self, story: List, styles, structured_data: Dict):
        """Add metadata section."""
        from reportlab.platypus import Spacer
        from reportlab.lib.units import inch

        metadata = structured_data.get("metadata", {})

        info_parts = []
        if metadata.get("author"):
            info_parts.append(f"Author: {metadata['author']}")
        if metadata.get("created_date"):
            info_parts.append(f"Created: {metadata['created_date']}")
        if metadata.get("page_count"):
            info_parts.append(f"Pages: {metadata['page_count']}")

        if info_parts:
            from reportlab.platypus import Paragraph
            info_text = " | ".join(info_parts)
            story.append(Paragraph(self._escape_text(info_text), styles['Caption']))
            story.append(Spacer(1, 0.25 * inch))

    def _add_text_content(self, story: List, styles, text: str):
        """Add text content to the story."""
        from reportlab.platypus import Paragraph, Spacer
        from reportlab.lib.units import inch

        # Split into paragraphs
        paragraphs = text.split('\n\n')

        for para_text in paragraphs:
            para_text = para_text.strip()
            if not para_text:
                continue

            # Check for headings
            if self._is_heading(para_text):
                clean_text = para_text.lstrip('#').strip()
                story.append(Paragraph(self._escape_text(clean_text), styles['PageHeading']))
            else:
                # Handle line breaks within paragraph
                para_text = para_text.replace('\n', '<br/>')
                story.append(Paragraph(self._escape_text(para_text), styles['DocBody']))

    def _is_heading(self, text: str) -> bool:
        """Check if text appears to be a heading."""
        if text.startswith('#'):
            return True
        if text.isupper() and len(text) < 100:
            return True
        if re.match(r'^\d+\.?\s+[A-Z]', text) and len(text) < 100:
            return True
        return False

    def _add_table(self, story: List, styles, table_data: Dict):
        """Add a table to the story."""
        from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch

        headers = table_data.get("headers", [])
        data = table_data.get("data", [])

        if not headers or not data:
            return

        # Caption
        story.append(Paragraph(
            f"Table (Page {table_data.get('page_number', '?')})",
            styles['Caption']
        ))

        # Build table data
        table_rows = [headers]  # Header row

        for row_data in data:
            row = [str(row_data.get(h, ""))[:50] for h in headers]  # Truncate long values
            table_rows.append(row)

        # Calculate column widths
        available_width = self.PAGE_WIDTH - 2 * self.MARGIN
        col_width = available_width / len(headers)

        # Create table
        table = Table(table_rows, colWidths=[col_width] * len(headers))

        # Style table
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), self.TABLE_SIZE),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))

        story.append(table)
        story.append(Spacer(1, 0.2 * inch))

    def _add_chart(self, story: List, styles, chart_data: Dict, document_dir: Path):
        """Add a chart (SVG as image) to the story."""
        from reportlab.platypus import Paragraph, Spacer, Image
        from reportlab.lib.units import inch

        svg_path = chart_data.get("svg_path")
        if not svg_path:
            return

        svg_path = Path(svg_path)
        if not svg_path.is_absolute():
            svg_path = document_dir / svg_path

        if not svg_path.exists():
            return

        # Convert SVG to PNG
        png_data = self._svg_to_png(svg_path)
        if not png_data:
            return

        # Caption
        story.append(Paragraph(
            f"Chart (Page {chart_data.get('page_number', '?')})",
            styles['Caption']
        ))

        # Add image from bytes
        img_buffer = io.BytesIO(png_data)
        img = Image(img_buffer, width=4 * inch, height=3 * inch)
        story.append(img)
        story.append(Spacer(1, 0.2 * inch))

    def _svg_to_png(self, svg_path: Path) -> Optional[bytes]:
        """Convert SVG to PNG bytes."""
        try:
            import cairosvg
            return cairosvg.svg2png(url=str(svg_path))
        except ImportError:
            logger.warning("cairosvg not installed, SVG charts will be skipped")
            return None
        except Exception as e:
            logger.warning(f"SVG to PNG conversion failed: {e}")
            return None

    def _add_regenerated_images(self, story: List, styles, page_num: int, document_dir: Path):
        """Add regenerated images for a page."""
        from reportlab.platypus import Paragraph, Spacer, Image
        from reportlab.lib.units import inch

        regen_dir = document_dir / "regenerated"
        if not regen_dir.exists():
            return

        pattern = f"regen_page_{page_num:03d}_*.png"
        for img_path in sorted(regen_dir.glob(pattern)):
            story.append(Paragraph(
                f"Image (Page {page_num}, Regenerated)",
                styles['Caption']
            ))

            try:
                img = Image(str(img_path), width=4 * inch, height=3 * inch)
                story.append(img)
            except Exception as e:
                logger.warning(f"Failed to add image {img_path}: {e}")

            story.append(Spacer(1, 0.2 * inch))

    def _add_image(self, story: List, styles, image_data: Dict, document_dir: Path):
        """Add an original extracted image."""
        from reportlab.platypus import Paragraph, Spacer, Image
        from reportlab.lib.units import inch

        file_path = image_data.get("file_path")
        if not file_path:
            return

        file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = document_dir / file_path

        if not file_path.exists():
            return

        story.append(Paragraph(
            f"Image (Page {image_data.get('page_number', '?')})",
            styles['Caption']
        ))

        try:
            img = Image(str(file_path), width=4 * inch, height=3 * inch)
            story.append(img)
        except Exception as e:
            logger.warning(f"Failed to add image {file_path}: {e}")

        story.append(Spacer(1, 0.2 * inch))

    def export_from_structured_document(
        self,
        structured_doc,
        output_path: str | Path,
        document_dir: Optional[str | Path] = None
    ) -> str:
        """
        Export from a StructuredDocument object.

        Args:
            structured_doc: StructuredDocument object
            output_path: Path for output .pdf file
            document_dir: Optional directory for resolving relative paths

        Returns:
            Path to the created PDF document
        """
        data = structured_doc.to_dict()

        if document_dir:
            return self.export(document_dir, output_path, structured_data=data)
        else:
            import tempfile
            with tempfile.TemporaryDirectory() as tmp_dir:
                return self.export(tmp_dir, output_path, structured_data=data)

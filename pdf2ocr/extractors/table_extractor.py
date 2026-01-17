"""
Table Extractor - Extract tables from PDFs and convert to JSON/SVG.

Uses pdfplumber for table detection and extraction.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import logging

import pdfplumber

logger = logging.getLogger(__name__)


@dataclass
class TableData:
    """Data for an extracted table."""
    table_id: int
    page_number: int
    table_index: int
    headers: List[str]
    rows: List[List[Any]]
    row_count: int = 0
    column_count: int = 0

    def __post_init__(self):
        self.row_count = len(self.rows)
        self.column_count = len(self.headers) if self.headers else 0

    @property
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with headers as keys."""
        return {
            "table_id": self.table_id,
            "page_number": self.page_number,
            "headers": self.headers,
            "data": [
                dict(zip(self.headers, row))
                for row in self.rows
            ],
            "row_count": self.row_count,
            "column_count": self.column_count
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.as_dict, indent=indent, ensure_ascii=False)


@dataclass
class TableExtractionResult:
    """Result of table extraction from a document."""
    tables: List[TableData] = field(default_factory=list)
    total_tables: int = 0
    pages_with_tables: int = 0
    json_paths: List[str] = field(default_factory=list)
    svg_paths: List[str] = field(default_factory=list)


class TableExtractor:
    """
    Extract tables from PDF documents.

    Uses pdfplumber for table detection and extraction.
    Converts tables to JSON and optionally to SVG format.
    """

    def __init__(self):
        """Initialize the table extractor."""
        pass

    def extract_tables(
        self,
        pdf_path: str | Path,
        output_dir: Optional[str | Path] = None,
        page_number: Optional[int] = None
    ) -> List[TableData]:
        """
        Extract tables from a PDF document.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save JSON files (optional)
            page_number: Specific page to extract from (1-indexed), or None for all

        Returns:
            List of TableData objects
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return []

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        tables = []
        table_id = 0

        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages_to_process = []

                if page_number is not None:
                    if 1 <= page_number <= len(pdf.pages):
                        pages_to_process = [(page_number, pdf.pages[page_number - 1])]
                else:
                    pages_to_process = [(i + 1, page) for i, page in enumerate(pdf.pages)]

                for page_num, page in pages_to_process:
                    page_tables = page.extract_tables()

                    if not page_tables:
                        continue

                    for table_idx, table in enumerate(page_tables):
                        if not table or len(table) < 2:
                            continue

                        # First row is headers
                        headers = [
                            str(cell).strip() if cell else f"Column_{i+1}"
                            for i, cell in enumerate(table[0])
                        ]

                        # Remaining rows are data
                        rows = []
                        for row in table[1:]:
                            cleaned_row = [
                                str(cell).strip() if cell else ""
                                for cell in row
                            ]
                            rows.append(cleaned_row)

                        if not rows:
                            continue

                        table_data = TableData(
                            table_id=table_id,
                            page_number=page_num,
                            table_index=table_idx,
                            headers=headers,
                            rows=rows
                        )

                        tables.append(table_data)

                        # Save JSON if output directory provided
                        if output_dir:
                            json_path = output_dir / f"page_{page_num:03d}_table_{table_idx + 1:03d}.json"
                            with open(json_path, "w", encoding="utf-8") as f:
                                f.write(table_data.to_json())
                            logger.debug(f"Saved: {json_path}")

                        table_id += 1

                logger.info(f"Extracted {len(tables)} tables from {pdf_path.name}")

        except Exception as e:
            logger.error(f"Table extraction failed: {e}")

        return tables

    def get_table_count(self, pdf_path: str | Path) -> int:
        """
        Get the total number of tables in a PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Number of tables
        """
        count = 0
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    if page_tables:
                        count += len([t for t in page_tables if t and len(t) >= 2])
        except Exception as e:
            logger.error(f"Error counting tables: {e}")
        return count

    def has_tables(self, pdf_path: str | Path) -> bool:
        """
        Check if a PDF contains any tables.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            True if PDF has tables
        """
        return self.get_table_count(pdf_path) > 0

    def extract_and_save(
        self,
        pdf_path: str | Path,
        output_dir: str | Path,
        save_svg: bool = True
    ) -> TableExtractionResult:
        """
        Extract tables and save as JSON (and optionally SVG).

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save output files
            save_svg: Whether to also generate SVG versions

        Returns:
            TableExtractionResult with extraction summary
        """
        output_dir = Path(output_dir)
        json_dir = output_dir / "json"
        svg_dir = output_dir / "svg" if save_svg else None

        json_dir.mkdir(parents=True, exist_ok=True)
        if svg_dir:
            svg_dir.mkdir(parents=True, exist_ok=True)

        tables = self.extract_tables(pdf_path, json_dir)

        result = TableExtractionResult(
            tables=tables,
            total_tables=len(tables)
        )

        # Count pages with tables
        pages_with_tables = set(t.page_number for t in tables)
        result.pages_with_tables = len(pages_with_tables)

        # Collect JSON paths
        for table in tables:
            json_path = json_dir / f"page_{table.page_number:03d}_table_{table.table_index + 1:03d}.json"
            result.json_paths.append(str(json_path))

        # Generate SVG if requested
        if save_svg and tables:
            from pdf2ocr.extractors.table_to_svg import TableToSVG
            svg_converter = TableToSVG()

            for table in tables:
                svg_path = svg_dir / f"page_{table.page_number:03d}_table_{table.table_index + 1:03d}.svg"
                svg_converter.convert(table, svg_path)
                result.svg_paths.append(str(svg_path))

        return result


def load_tables_from_json(json_dir: str | Path) -> List[TableData]:
    """
    Load TableData objects from JSON files in a directory.

    Args:
        json_dir: Directory containing table JSON files

    Returns:
        List of TableData objects
    """
    json_dir = Path(json_dir)
    tables = []

    if not json_dir.exists():
        return tables

    for json_file in sorted(json_dir.glob("*_table_*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Convert dict-format data back to rows
            rows = []
            if "data" in data and data["headers"]:
                for row_dict in data["data"]:
                    row = [row_dict.get(h, "") for h in data["headers"]]
                    rows.append(row)

            table = TableData(
                table_id=data.get("table_id", 0),
                page_number=data.get("page_number", 0),
                table_index=data.get("table_id", 0),  # Use table_id as index
                headers=data.get("headers", []),
                rows=rows
            )
            tables.append(table)

        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")

    return tables

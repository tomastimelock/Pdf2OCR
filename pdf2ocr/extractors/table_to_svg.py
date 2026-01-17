"""
Table to SVG Converter - Convert extracted tables to SVG format.

Creates clean, readable SVG representations of tables.
"""

from pathlib import Path
from typing import Optional, List, Any
import logging
import html

logger = logging.getLogger(__name__)


class TableToSVG:
    """
    Convert table data to SVG format.

    Creates professional-looking SVG tables with proper styling.
    """

    # Default styling
    DEFAULT_CELL_PADDING = 10
    DEFAULT_CELL_HEIGHT = 30
    DEFAULT_MIN_CELL_WIDTH = 80
    DEFAULT_MAX_CELL_WIDTH = 200
    DEFAULT_FONT_SIZE = 12
    DEFAULT_HEADER_FONT_SIZE = 13

    # Colors
    HEADER_BG = "#4a5568"
    HEADER_TEXT = "#ffffff"
    ROW_BG_EVEN = "#ffffff"
    ROW_BG_ODD = "#f7fafc"
    BORDER_COLOR = "#e2e8f0"
    TEXT_COLOR = "#2d3748"

    def __init__(
        self,
        cell_padding: int = DEFAULT_CELL_PADDING,
        cell_height: int = DEFAULT_CELL_HEIGHT,
        min_cell_width: int = DEFAULT_MIN_CELL_WIDTH,
        max_cell_width: int = DEFAULT_MAX_CELL_WIDTH,
        font_size: int = DEFAULT_FONT_SIZE
    ):
        """
        Initialize the SVG converter.

        Args:
            cell_padding: Padding inside cells
            cell_height: Height of each row
            min_cell_width: Minimum column width
            max_cell_width: Maximum column width
            font_size: Font size for cell text
        """
        self.cell_padding = cell_padding
        self.cell_height = cell_height
        self.min_cell_width = min_cell_width
        self.max_cell_width = max_cell_width
        self.font_size = font_size

    def _escape_text(self, text: Any) -> str:
        """Escape text for SVG/XML."""
        if text is None:
            return ""
        return html.escape(str(text))

    def _calculate_column_widths(
        self,
        headers: List[str],
        rows: List[List[Any]]
    ) -> List[int]:
        """Calculate optimal column widths based on content."""
        widths = []
        char_width = self.font_size * 0.6  # Approximate character width

        for col_idx in range(len(headers)):
            # Start with header width
            max_len = len(str(headers[col_idx]))

            # Check all rows
            for row in rows:
                if col_idx < len(row):
                    cell_len = len(str(row[col_idx]) if row[col_idx] else "")
                    max_len = max(max_len, cell_len)

            # Calculate width with padding
            width = int(max_len * char_width) + (self.cell_padding * 2)

            # Clamp to min/max
            width = max(self.min_cell_width, min(self.max_cell_width, width))
            widths.append(width)

        return widths

    def _truncate_text(self, text: str, max_width: int) -> str:
        """Truncate text if it exceeds max width."""
        char_width = self.font_size * 0.6
        max_chars = int((max_width - self.cell_padding * 2) / char_width)

        if len(text) > max_chars:
            return text[:max_chars - 3] + "..."
        return text

    def convert(
        self,
        table_data,
        output_path: Optional[str | Path] = None
    ) -> str:
        """
        Convert a TableData object to SVG.

        Args:
            table_data: TableData object with headers and rows
            output_path: Path to save SVG file (optional)

        Returns:
            SVG code as string
        """
        headers = table_data.headers
        rows = table_data.rows

        if not headers:
            return ""

        # Calculate dimensions
        col_widths = self._calculate_column_widths(headers, rows)
        total_width = sum(col_widths)
        total_height = (len(rows) + 1) * self.cell_height  # +1 for header

        # Build SVG
        svg_parts = []

        # SVG header
        svg_parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {total_width} {total_height}" '
            f'width="{total_width}" height="{total_height}">'
        )

        # Style definitions
        svg_parts.append('''
<defs>
    <style>
        .header-cell { font-weight: bold; }
        .cell-text { font-family: Arial, sans-serif; }
    </style>
</defs>''')

        # Background
        svg_parts.append(
            f'<rect width="{total_width}" height="{total_height}" fill="{self.ROW_BG_EVEN}"/>'
        )

        # Draw header row
        x = 0
        y = 0

        # Header background
        svg_parts.append(
            f'<rect x="0" y="0" width="{total_width}" height="{self.cell_height}" '
            f'fill="{self.HEADER_BG}"/>'
        )

        # Header cells
        for col_idx, header in enumerate(headers):
            cell_width = col_widths[col_idx]
            text = self._truncate_text(self._escape_text(header), cell_width)

            # Cell border
            svg_parts.append(
                f'<rect x="{x}" y="0" width="{cell_width}" height="{self.cell_height}" '
                f'fill="none" stroke="{self.BORDER_COLOR}" stroke-width="1"/>'
            )

            # Header text
            text_x = x + self.cell_padding
            text_y = self.cell_height / 2 + self.font_size / 3

            svg_parts.append(
                f'<text x="{text_x}" y="{text_y}" '
                f'font-size="{self.DEFAULT_HEADER_FONT_SIZE}" '
                f'fill="{self.HEADER_TEXT}" '
                f'class="cell-text header-cell">{text}</text>'
            )

            x += cell_width

        # Draw data rows
        for row_idx, row in enumerate(rows):
            y = (row_idx + 1) * self.cell_height
            x = 0

            # Alternate row background
            bg_color = self.ROW_BG_ODD if row_idx % 2 == 0 else self.ROW_BG_EVEN
            svg_parts.append(
                f'<rect x="0" y="{y}" width="{total_width}" height="{self.cell_height}" '
                f'fill="{bg_color}"/>'
            )

            # Data cells
            for col_idx in range(len(headers)):
                cell_width = col_widths[col_idx]
                cell_value = row[col_idx] if col_idx < len(row) else ""
                text = self._truncate_text(self._escape_text(cell_value), cell_width)

                # Cell border
                svg_parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell_width}" height="{self.cell_height}" '
                    f'fill="none" stroke="{self.BORDER_COLOR}" stroke-width="1"/>'
                )

                # Cell text
                text_x = x + self.cell_padding
                text_y = y + self.cell_height / 2 + self.font_size / 3

                svg_parts.append(
                    f'<text x="{text_x}" y="{text_y}" '
                    f'font-size="{self.font_size}" '
                    f'fill="{self.TEXT_COLOR}" '
                    f'class="cell-text">{text}</text>'
                )

                x += cell_width

        # Close SVG
        svg_parts.append('</svg>')

        svg_code = '\n'.join(svg_parts)

        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(svg_code)
            logger.debug(f"Saved SVG table: {output_path}")

        return svg_code

    def convert_from_dict(
        self,
        table_dict: dict,
        output_path: Optional[str | Path] = None
    ) -> str:
        """
        Convert a table dictionary to SVG.

        Args:
            table_dict: Dictionary with 'headers' and 'data' keys
            output_path: Path to save SVG file (optional)

        Returns:
            SVG code as string
        """
        from pdf2ocr.extractors.table_extractor import TableData

        headers = table_dict.get("headers", [])
        data = table_dict.get("data", [])

        # Convert data dicts back to rows
        rows = []
        for row_dict in data:
            row = [row_dict.get(h, "") for h in headers]
            rows.append(row)

        table = TableData(
            table_id=table_dict.get("table_id", 0),
            page_number=table_dict.get("page_number", 0),
            table_index=0,
            headers=headers,
            rows=rows
        )

        return self.convert(table, output_path)

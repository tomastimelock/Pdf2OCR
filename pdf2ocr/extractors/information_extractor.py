"""Information Extractor - Extract structured information from OCR text."""

import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from pdf2ocr.processors.ocr_processor import ProcessingResult


@dataclass
class KeyValuePair:
    """A key-value pair extracted from text."""
    key: str
    value: str
    page_number: int
    confidence: float = 0.0


@dataclass
class ListItem:
    """A list extracted from text."""
    items: list[str]
    list_type: str  # 'bulleted', 'numbered'
    page_number: int


@dataclass
class TableData:
    """A table extracted from text."""
    headers: list[str]
    rows: list[list[str]]
    page_number: int


@dataclass
class ExtractedInformation:
    """All information extracted from a document."""
    title: Optional[str] = None
    key_values: list[KeyValuePair] = field(default_factory=list)
    lists: list[ListItem] = field(default_factory=list)
    tables: list[TableData] = field(default_factory=list)
    emails: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    phone_numbers: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    currencies: list[str] = field(default_factory=list)
    raw_text: str = ""
    page_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "key_values": [
                {"key": kv.key, "value": kv.value, "page": kv.page_number}
                for kv in self.key_values
            ],
            "lists": [
                {"items": lst.items, "type": lst.list_type, "page": lst.page_number}
                for lst in self.lists
            ],
            "tables": [
                {"headers": tbl.headers, "rows": tbl.rows, "page": tbl.page_number}
                for tbl in self.tables
            ],
            "emails": self.emails,
            "urls": self.urls,
            "phone_numbers": self.phone_numbers,
            "dates": self.dates,
            "currencies": self.currencies,
            "page_count": self.page_count
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class InformationExtractor:
    """
    Extract structured information from OCR text.

    Extracts:
    - Key-value pairs (e.g., "Name: John Doe")
    - Lists (bulleted and numbered)
    - Tables (markdown format)
    - Emails, URLs, phone numbers
    - Dates and currency amounts
    """

    EMAIL_PATTERN = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    )

    URL_PATTERN = re.compile(
        r'https?://[^\s<>"{}|\\^`\[\]]+'
    )

    PHONE_PATTERN = re.compile(
        r'(?:\+\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}'
    )

    DATE_PATTERNS = [
        re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),  # ISO format
        re.compile(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'),  # US/EU format
        re.compile(r'\b\d{1,2}\.\d{1,2}\.\d{2,4}\b'),  # EU dot format
        re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', re.I),
        re.compile(r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b', re.I),
    ]

    CURRENCY_PATTERN = re.compile(
        r'(?:[$€£¥]\s*\d+(?:[.,]\d{2,3})*(?:[.,]\d{2})?|\d+(?:[.,]\d{2,3})*(?:[.,]\d{2})?\s*(?:USD|EUR|GBP|SEK|NOK|DKK|kr))'
    )

    KEY_VALUE_PATTERNS = [
        re.compile(r'^([^:\n]{1,50}):\s*(.+)$', re.MULTILINE),
        re.compile(r'^([^=\n]{1,50})\s*=\s*(.+)$', re.MULTILINE),
        re.compile(r'^([^\-\n]{1,50})\s+-\s+(.+)$', re.MULTILINE),
    ]

    def __init__(self):
        """Initialize the information extractor."""
        pass

    def extract_from_result(self, result: ProcessingResult) -> ExtractedInformation:
        """
        Extract information from an OCR processing result.

        Args:
            result: ProcessingResult from OCR processing

        Returns:
            ExtractedInformation with all extracted data
        """
        info = ExtractedInformation(page_count=result.total_pages)

        all_text_parts = []
        for page_result in result.pages:
            page_num = page_result.page_number
            text = page_result.text

            if text.strip():
                all_text_parts.append(text)

            info.key_values.extend(self.extract_key_values(text, page_num))
            info.lists.extend(self.extract_lists(text, page_num))
            info.tables.extend(self.extract_tables(text, page_num))

        info.raw_text = "\n\n".join(all_text_parts)

        info.emails = list(set(self.extract_emails(info.raw_text)))
        info.urls = list(set(self.extract_urls(info.raw_text)))
        info.phone_numbers = list(set(self.extract_phone_numbers(info.raw_text)))
        info.dates = list(set(self.extract_dates(info.raw_text)))
        info.currencies = list(set(self.extract_currencies(info.raw_text)))

        info.title = self.extract_title(info.raw_text)

        return info

    def extract_from_text(self, text: str, page_number: int = 1) -> ExtractedInformation:
        """
        Extract information from raw text.

        Args:
            text: Raw text to extract from
            page_number: Page number for single-page extraction

        Returns:
            ExtractedInformation with all extracted data
        """
        info = ExtractedInformation(page_count=1, raw_text=text)

        info.key_values = self.extract_key_values(text, page_number)
        info.lists = self.extract_lists(text, page_number)
        info.tables = self.extract_tables(text, page_number)
        info.emails = list(set(self.extract_emails(text)))
        info.urls = list(set(self.extract_urls(text)))
        info.phone_numbers = list(set(self.extract_phone_numbers(text)))
        info.dates = list(set(self.extract_dates(text)))
        info.currencies = list(set(self.extract_currencies(text)))
        info.title = self.extract_title(text)

        return info

    def extract_key_values(self, text: str, page_number: int) -> list[KeyValuePair]:
        """
        Extract key-value pairs from text.

        Args:
            text: Text to extract from
            page_number: Page number for tracking

        Returns:
            List of KeyValuePair objects
        """
        results = []
        seen_keys = set()

        for pattern in self.KEY_VALUE_PATTERNS:
            matches = pattern.findall(text)
            for key, value in matches:
                key = key.strip()
                value = value.strip()

                if not key or not value:
                    continue
                if len(key) > 50 or len(value) > 200:
                    continue
                if key.lower() in seen_keys:
                    continue
                if not re.search(r'[a-zA-Z]', key):
                    continue

                seen_keys.add(key.lower())
                results.append(KeyValuePair(
                    key=key,
                    value=value,
                    page_number=page_number,
                    confidence=0.8
                ))

        return results

    def extract_lists(self, text: str, page_number: int) -> list[ListItem]:
        """
        Extract lists from text.

        Args:
            text: Text to extract from
            page_number: Page number for tracking

        Returns:
            List of ListItem objects
        """
        results = []

        bulleted_pattern = re.compile(r'^[\s]*[-*•]\s+(.+)$', re.MULTILINE)
        bulleted_items = bulleted_pattern.findall(text)
        if len(bulleted_items) >= 2:
            results.append(ListItem(
                items=[item.strip() for item in bulleted_items],
                list_type="bulleted",
                page_number=page_number
            ))

        numbered_pattern = re.compile(r'^[\s]*(?:\d+\.|[a-zA-Z]\.)\s+(.+)$', re.MULTILINE)
        numbered_items = numbered_pattern.findall(text)
        if len(numbered_items) >= 2:
            results.append(ListItem(
                items=[item.strip() for item in numbered_items],
                list_type="numbered",
                page_number=page_number
            ))

        return results

    def extract_tables(self, text: str, page_number: int) -> list[TableData]:
        """
        Extract markdown tables from text.

        Args:
            text: Text to extract from
            page_number: Page number for tracking

        Returns:
            List of TableData objects
        """
        results = []

        table_pattern = re.compile(
            r'^\|(.+)\|\s*\n\|[-:\s|]+\|\s*\n((?:\|.+\|\s*\n?)+)',
            re.MULTILINE
        )

        matches = table_pattern.findall(text)
        for header_row, body in matches:
            headers = [h.strip() for h in header_row.split('|') if h.strip()]

            rows = []
            for row_line in body.strip().split('\n'):
                cells = [c.strip() for c in row_line.split('|') if c.strip()]
                if cells:
                    rows.append(cells)

            if headers and rows:
                results.append(TableData(
                    headers=headers,
                    rows=rows,
                    page_number=page_number
                ))

        return results

    def extract_emails(self, text: str) -> list[str]:
        """Extract email addresses from text."""
        return self.EMAIL_PATTERN.findall(text)

    def extract_urls(self, text: str) -> list[str]:
        """Extract URLs from text."""
        return self.URL_PATTERN.findall(text)

    def extract_phone_numbers(self, text: str) -> list[str]:
        """Extract phone numbers from text."""
        matches = self.PHONE_PATTERN.findall(text)
        valid = []
        for m in matches:
            digits = re.sub(r'\D', '', m)
            if 7 <= len(digits) <= 15:
                valid.append(m)
        return valid

    def extract_dates(self, text: str) -> list[str]:
        """Extract dates from text."""
        dates = []
        for pattern in self.DATE_PATTERNS:
            dates.extend(pattern.findall(text))
        return dates

    def extract_currencies(self, text: str) -> list[str]:
        """Extract currency amounts from text."""
        return self.CURRENCY_PATTERN.findall(text)

    def extract_title(self, text: str) -> Optional[str]:
        """
        Attempt to extract document title from text.

        Args:
            text: Text to extract from

        Returns:
            Extracted title or None
        """
        lines = text.strip().split('\n')

        for line in lines[:10]:
            line = line.strip()
            if not line:
                continue

            if line.startswith('#'):
                return line.lstrip('#').strip()

            if 10 < len(line) < 100 and line[0].isupper():
                if not any(c in line for c in [':', '=', '|', '-']):
                    return line

        return None

    def save_to_json(self, info: ExtractedInformation, output_path: str | Path) -> None:
        """
        Save extracted information to a JSON file.

        Args:
            info: ExtractedInformation to save
            output_path: Path for the output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(info.to_json())

    def save_summary(self, info: ExtractedInformation, output_path: str | Path) -> None:
        """
        Save a human-readable summary to a text file.

        Args:
            info: ExtractedInformation to summarize
            output_path: Path for the output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        lines.append("=" * 60)
        lines.append("DOCUMENT SUMMARY")
        lines.append("=" * 60)

        if info.title:
            lines.append(f"\nTitle: {info.title}")

        lines.append(f"Total Pages: {info.page_count}")

        if info.key_values:
            lines.append(f"\n--- Key-Value Pairs ({len(info.key_values)}) ---")
            for kv in info.key_values[:20]:
                lines.append(f"  {kv.key}: {kv.value}")
            if len(info.key_values) > 20:
                lines.append(f"  ... and {len(info.key_values) - 20} more")

        if info.lists:
            lines.append(f"\n--- Lists ({len(info.lists)}) ---")
            for lst in info.lists[:5]:
                lines.append(f"  [{lst.list_type}] {len(lst.items)} items")
            if len(info.lists) > 5:
                lines.append(f"  ... and {len(info.lists) - 5} more")

        if info.tables:
            lines.append(f"\n--- Tables ({len(info.tables)}) ---")
            for tbl in info.tables:
                lines.append(f"  Page {tbl.page_number}: {len(tbl.headers)} columns, {len(tbl.rows)} rows")

        if info.emails:
            lines.append(f"\n--- Emails ({len(info.emails)}) ---")
            for email in info.emails[:10]:
                lines.append(f"  {email}")

        if info.urls:
            lines.append(f"\n--- URLs ({len(info.urls)}) ---")
            for url in info.urls[:10]:
                lines.append(f"  {url}")

        if info.phone_numbers:
            lines.append(f"\n--- Phone Numbers ({len(info.phone_numbers)}) ---")
            for phone in info.phone_numbers[:10]:
                lines.append(f"  {phone}")

        if info.dates:
            lines.append(f"\n--- Dates ({len(info.dates)}) ---")
            for date in info.dates[:10]:
                lines.append(f"  {date}")

        if info.currencies:
            lines.append(f"\n--- Currency Amounts ({len(info.currencies)}) ---")
            for curr in info.currencies[:10]:
                lines.append(f"  {curr}")

        lines.append("\n" + "=" * 60)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

"""Document Parser Module.

Provides structured data extraction from receipts, forms, invoices,
and other document types using OCR and pattern matching.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class ParserError(Exception):
    """Error during document parsing."""
    pass


@dataclass
class FieldValue:
    """An extracted field value."""
    name: str
    value: Any
    raw_text: str = ""
    confidence: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'raw_text': self.raw_text,
            'confidence': self.confidence,
        }


@dataclass
class LineItem:
    """A line item from a receipt or invoice."""
    description: str
    quantity: float = 1.0
    unit_price: float = 0.0
    total: float = 0.0
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'description': self.description,
            'quantity': self.quantity,
            'unit_price': self.unit_price,
            'total': self.total,
        }


@dataclass
class ReceiptData:
    """Parsed receipt data."""
    path: str
    merchant_name: str = ""
    merchant_address: str = ""
    merchant_phone: str = ""
    date: Optional[datetime] = None
    time: str = ""
    items: List[LineItem] = field(default_factory=list)
    subtotal: float = 0.0
    tax: float = 0.0
    total: float = 0.0
    payment_method: str = ""
    currency: str = "USD"
    raw_text: str = ""
    confidence: float = 0.0

    @property
    def item_count(self) -> int:
        return len(self.items)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path,
            'merchant_name': self.merchant_name,
            'merchant_address': self.merchant_address,
            'date': self.date.isoformat() if self.date else None,
            'time': self.time,
            'items': [item.to_dict() for item in self.items],
            'subtotal': self.subtotal,
            'tax': self.tax,
            'total': self.total,
            'payment_method': self.payment_method,
            'currency': self.currency,
            'confidence': self.confidence,
        }


@dataclass
class FormData:
    """Parsed form data."""
    path: str
    form_type: str = ""
    fields: List[FieldValue] = field(default_factory=list)
    tables: List[List[List[str]]] = field(default_factory=list)
    raw_text: str = ""
    confidence: float = 0.0

    def get_field(self, name: str) -> Optional[FieldValue]:
        """Get field by name."""
        for f in self.fields:
            if f.name.lower() == name.lower():
                return f
        return None

    def get_value(self, name: str, default: Any = None) -> Any:
        """Get field value by name."""
        field = self.get_field(name)
        return field.value if field else default

    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path,
            'form_type': self.form_type,
            'fields': [f.to_dict() for f in self.fields],
            'tables': self.tables,
            'confidence': self.confidence,
        }


class DocumentParser:
    """Parse structured data from documents."""

    # Common patterns
    PATTERNS = {
        'date': [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}',
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}',
        ],
        'time': [
            r'\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?',
        ],
        'money': [
            r'\$?\d+[.,]\d{2}',
            r'\d+[.,]\d{2}\s*(?:USD|EUR|GBP|CAD)',
        ],
        'phone': [
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\+\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',
        ],
        'email': [
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        ],
    }

    def __init__(self):
        """Initialize document parser."""
        self._ocr_engine = None

    def _get_ocr_engine(self):
        """Get OCR engine."""
        if self._ocr_engine is None:
            from .ocr import OCREngine
            self._ocr_engine = OCREngine()
        return self._ocr_engine

    def _load_image(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load image as numpy array."""
        if isinstance(image, np.ndarray):
            return image

        try:
            from PIL import Image
            img = Image.open(image).convert('RGB')
            return np.array(img)
        except ImportError:
            import cv2
            img = cv2.imread(str(image))
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _extract_text(self, image: Union[str, Path, np.ndarray]) -> Tuple[str, float]:
        """Extract text from image using OCR."""
        engine = self._get_ocr_engine()
        result = engine.extract(image)
        return result.text, result.confidence

    def _find_pattern(self, text: str, pattern_name: str) -> List[str]:
        """Find all matches for a pattern."""
        matches = []
        for pattern in self.PATTERNS.get(pattern_name, []):
            matches.extend(re.findall(pattern, text, re.IGNORECASE))
        return matches

    def _parse_money(self, text: str) -> Optional[float]:
        """Parse money value from text."""
        # Remove currency symbols and whitespace
        cleaned = re.sub(r'[^\d.,]', '', text)
        if not cleaned:
            return None

        # Handle different decimal separators
        if ',' in cleaned and '.' in cleaned:
            # Assume last separator is decimal
            if cleaned.rfind(',') > cleaned.rfind('.'):
                cleaned = cleaned.replace('.', '').replace(',', '.')
            else:
                cleaned = cleaned.replace(',', '')
        elif ',' in cleaned:
            # Could be decimal or thousands separator
            if len(cleaned.split(',')[-1]) == 2:
                cleaned = cleaned.replace(',', '.')
            else:
                cleaned = cleaned.replace(',', '')

        try:
            return float(cleaned)
        except ValueError:
            return None

    def _parse_date(self, text: str) -> Optional[datetime]:
        """Parse date from text."""
        formats = [
            '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',
            '%m-%d-%Y', '%d-%m-%Y', '%Y-%m-%d',
            '%m/%d/%y', '%d/%m/%y',
            '%B %d, %Y', '%d %B %Y', '%b %d, %Y', '%d %b %Y',
        ]

        for fmt in formats:
            try:
                return datetime.strptime(text.strip(), fmt)
            except ValueError:
                continue

        return None

    def parse_receipt(
        self,
        image: Union[str, Path, np.ndarray],
    ) -> ReceiptData:
        """
        Parse receipt from image.

        Args:
            image: Image path or numpy array

        Returns:
            ReceiptData with extracted information
        """
        path = str(image) if not isinstance(image, np.ndarray) else "array"
        text, confidence = self._extract_text(image)

        receipt = ReceiptData(
            path=path,
            raw_text=text,
            confidence=confidence,
        )

        lines = text.split('\n')

        # Parse merchant name (usually first non-empty line)
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) > 3:
                receipt.merchant_name = line
                break

        # Find date
        dates = self._find_pattern(text, 'date')
        if dates:
            parsed = self._parse_date(dates[0])
            if parsed:
                receipt.date = parsed

        # Find time
        times = self._find_pattern(text, 'time')
        if times:
            receipt.time = times[0]

        # Find phone
        phones = self._find_pattern(text, 'phone')
        if phones:
            receipt.merchant_phone = phones[0]

        # Parse line items and totals
        items = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for price patterns
            price_match = re.search(r'\$?\d+[.,]\d{2}', line)
            if price_match:
                price = self._parse_money(price_match.group())

                # Check for total indicators
                line_lower = line.lower()
                if 'total' in line_lower and 'sub' not in line_lower:
                    if price:
                        receipt.total = price
                elif 'subtotal' in line_lower or 'sub-total' in line_lower:
                    if price:
                        receipt.subtotal = price
                elif 'tax' in line_lower:
                    if price:
                        receipt.tax = price
                elif price:
                    # Regular item
                    description = line[:price_match.start()].strip()
                    if description:
                        items.append(LineItem(
                            description=description,
                            total=price,
                            raw_text=line,
                        ))

        receipt.items = items

        # Detect payment method
        text_lower = text.lower()
        if 'visa' in text_lower:
            receipt.payment_method = 'Visa'
        elif 'mastercard' in text_lower or 'mc' in text_lower:
            receipt.payment_method = 'Mastercard'
        elif 'amex' in text_lower or 'american express' in text_lower:
            receipt.payment_method = 'Amex'
        elif 'cash' in text_lower:
            receipt.payment_method = 'Cash'
        elif 'debit' in text_lower:
            receipt.payment_method = 'Debit'

        return receipt

    def parse_form(
        self,
        image: Union[str, Path, np.ndarray],
        template: Optional[Dict[str, Any]] = None,
    ) -> FormData:
        """
        Parse form from image.

        Args:
            image: Image path or numpy array
            template: Optional template with expected fields

        Returns:
            FormData with extracted fields
        """
        path = str(image) if not isinstance(image, np.ndarray) else "array"
        text, confidence = self._extract_text(image)

        form = FormData(
            path=path,
            raw_text=text,
            confidence=confidence,
        )

        lines = text.split('\n')

        # Extract key-value pairs
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for common patterns: "Label: Value", "Label - Value", etc.
            patterns = [
                r'^([^:]+):\s*(.+)$',
                r'^([^-]+)\s*-\s*(.+)$',
                r'^([^=]+)\s*=\s*(.+)$',
            ]

            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    label = match.group(1).strip()
                    value = match.group(2).strip()

                    if label and value:
                        form.fields.append(FieldValue(
                            name=label,
                            value=value,
                            raw_text=line,
                            confidence=confidence,
                        ))
                    break

        # If template provided, try to match expected fields
        if template and 'fields' in template:
            for expected in template['fields']:
                name = expected.get('name', '')
                pattern = expected.get('pattern', '')

                if pattern:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        form.fields.append(FieldValue(
                            name=name,
                            value=matches[0],
                            raw_text=matches[0],
                            confidence=0.8,
                        ))

        # Extract common fields
        # Dates
        for date_str in self._find_pattern(text, 'date'):
            parsed = self._parse_date(date_str)
            if parsed:
                form.fields.append(FieldValue(
                    name='date',
                    value=parsed,
                    raw_text=date_str,
                ))

        # Emails
        for email in self._find_pattern(text, 'email'):
            form.fields.append(FieldValue(
                name='email',
                value=email,
                raw_text=email,
            ))

        # Phones
        for phone in self._find_pattern(text, 'phone'):
            form.fields.append(FieldValue(
                name='phone',
                value=phone,
                raw_text=phone,
            ))

        return form

    def extract_fields(
        self,
        image: Union[str, Path, np.ndarray],
        field_patterns: Dict[str, str],
    ) -> Dict[str, FieldValue]:
        """
        Extract specific fields using patterns.

        Args:
            image: Image path or numpy array
            field_patterns: Dict of field name to regex pattern

        Returns:
            Dict of field name to extracted FieldValue
        """
        text, confidence = self._extract_text(image)
        results = {}

        for name, pattern in field_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                value = matches[0] if isinstance(matches[0], str) else matches[0][0]
                results[name] = FieldValue(
                    name=name,
                    value=value,
                    raw_text=value,
                    confidence=confidence,
                )

        return results

    def parse_invoice(
        self,
        image: Union[str, Path, np.ndarray],
    ) -> FormData:
        """
        Parse invoice from image.

        Args:
            image: Image path or numpy array

        Returns:
            FormData with invoice fields
        """
        # Use form parser with invoice-specific patterns
        template = {
            'fields': [
                {'name': 'invoice_number', 'pattern': r'(?:Invoice|INV)[#:\s]*(\S+)'},
                {'name': 'po_number', 'pattern': r'(?:PO|P\.O\.|Purchase Order)[#:\s]*(\S+)'},
                {'name': 'due_date', 'pattern': r'(?:Due|Payment Due)[:\s]*([^\n]+)'},
                {'name': 'total', 'pattern': r'(?:Total|Amount Due)[:\s]*\$?([\d,]+\.?\d*)'},
            ]
        }

        form = self.parse_form(image, template)
        form.form_type = 'invoice'
        return form

    def parse_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        doc_type: str = 'receipt',
    ) -> List[Union[ReceiptData, FormData]]:
        """Parse multiple documents."""
        results = []
        for img in images:
            if doc_type == 'receipt':
                results.append(self.parse_receipt(img))
            else:
                results.append(self.parse_form(img))
        return results


# Convenience functions
def parse_receipt(image: Union[str, Path, np.ndarray]) -> ReceiptData:
    """Parse receipt from image."""
    parser = DocumentParser()
    return parser.parse_receipt(image)


def parse_form(
    image: Union[str, Path, np.ndarray],
    template: Optional[Dict[str, Any]] = None,
) -> FormData:
    """Parse form from image."""
    parser = DocumentParser()
    return parser.parse_form(image, template)


def extract_fields(
    image: Union[str, Path, np.ndarray],
    patterns: Dict[str, str],
) -> Dict[str, FieldValue]:
    """Extract specific fields using patterns."""
    parser = DocumentParser()
    return parser.extract_fields(image, patterns)

# Filepath: code_migration/extraction/pdf_toolkit/dto.py
# Description: Data Transfer Objects for PDF Toolkit Provider
# Layer: Extraction

"""Data Transfer Objects for PDF Toolkit Provider."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum


class ProcessingStatus(Enum):
    """Document processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DocumentInfo:
    """
    Information about a processed PDF document.

    Attributes:
        file_name: Name of the PDF file
        file_path: Full path to the file
        file_hash: SHA256 hash for deduplication
        file_size: Size in bytes
        page_count: Number of pages
        title: Document title (from metadata)
        author: Document author (from metadata)
        subject: Document subject (from metadata)
        keywords: Document keywords (from metadata)
        creator: PDF creator application
        producer: PDF producer application
        creation_date: Document creation date
        modification_date: Document modification date
        metadata: Additional metadata dictionary
        status: Processing status
    """

    file_name: str
    file_path: str
    file_hash: str = ""
    file_size: int = 0
    page_count: int = 0
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: ProcessingStatus = ProcessingStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_name": self.file_name,
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "file_size": self.file_size,
            "page_count": self.page_count,
            "title": self.title,
            "author": self.author,
            "subject": self.subject,
            "keywords": self.keywords,
            "creator": self.creator,
            "producer": self.producer,
            "creation_date": self.creation_date,
            "modification_date": self.modification_date,
            "metadata": self.metadata,
            "status": self.status.value,
        }


@dataclass
class PageInfo:
    """
    Information about a single PDF page.

    Attributes:
        page_number: Page number (1-indexed)
        content_text: Extracted text content
        content_hash: Hash of content for change detection
        width: Page width in points
        height: Page height in points
        rotation: Page rotation in degrees
        has_tables: Whether page contains tables
        has_images: Whether page contains images
        word_count: Approximate word count
    """

    page_number: int
    content_text: str = ""
    content_hash: str = ""
    width: int = 0
    height: int = 0
    rotation: int = 0
    has_tables: bool = False
    has_images: bool = False
    word_count: int = 0

    def __post_init__(self):
        if self.content_text and not self.word_count:
            self.word_count = len(self.content_text.split())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "page_number": self.page_number,
            "content_text": self.content_text,
            "content_hash": self.content_hash,
            "width": self.width,
            "height": self.height,
            "rotation": self.rotation,
            "has_tables": self.has_tables,
            "has_images": self.has_images,
            "word_count": self.word_count,
        }


@dataclass
class TableInfo:
    """
    Information about an extracted table.

    Attributes:
        page_number: Page where table was found
        table_index: Index of table on the page
        data: Table data as list of rows
        rows: Number of rows
        columns: Number of columns
        headers: Optional detected headers
    """

    page_number: int
    table_index: int
    data: List[List[Any]]
    rows: int = 0
    columns: int = 0
    headers: Optional[List[str]] = None

    def __post_init__(self):
        if self.data and not self.rows:
            self.rows = len(self.data)
        if self.data and self.data[0] and not self.columns:
            self.columns = len(self.data[0])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "page_number": self.page_number,
            "table_index": self.table_index,
            "data": self.data,
            "rows": self.rows,
            "columns": self.columns,
            "headers": self.headers,
        }

    def to_csv_string(self) -> str:
        """Convert table to CSV string."""
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)
        for row in self.data:
            writer.writerow(row)
        return output.getvalue()


@dataclass
class ExtractionResult:
    """
    Result of text extraction operation.

    Attributes:
        file_path: Source file path
        total_pages: Total number of pages
        pages: List of PageInfo objects
        full_text: Complete extracted text
        extraction_time: Time taken for extraction (seconds)
        success: Whether extraction was successful
        error_message: Error message if failed
    """

    file_path: str
    total_pages: int = 0
    pages: List[PageInfo] = field(default_factory=list)
    full_text: str = ""
    extraction_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "total_pages": self.total_pages,
            "pages": [p.to_dict() for p in self.pages],
            "full_text": self.full_text,
            "extraction_time": self.extraction_time,
            "success": self.success,
            "error_message": self.error_message,
        }


@dataclass
class ProcessingResult:
    """
    Result of document processing operation.

    Attributes:
        document_info: Processed document information
        operation: Type of operation performed
        output_files: List of output file paths (for split/merge)
        processing_time: Time taken (seconds)
        success: Whether operation was successful
        error_message: Error message if failed
    """

    document_info: Optional[DocumentInfo] = None
    operation: str = "process"
    output_files: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_info": self.document_info.to_dict() if self.document_info else None,
            "operation": self.operation,
            "output_files": self.output_files,
            "processing_time": self.processing_time,
            "success": self.success,
            "error_message": self.error_message,
        }


@dataclass
class EmbeddingResult:
    """
    Result of embedding generation.

    Attributes:
        text: Source text
        embedding: Generated embedding vector
        model_name: Name of model used
        dimension: Embedding dimension
        chunk_index: Index if part of chunked text
        success: Whether generation was successful
        error_message: Error message if failed
    """

    text: str = ""
    embedding: List[float] = field(default_factory=list)
    model_name: str = ""
    dimension: int = 0
    chunk_index: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "embedding_length": len(self.embedding),
            "model_name": self.model_name,
            "dimension": self.dimension,
            "chunk_index": self.chunk_index,
            "success": self.success,
            "error_message": self.error_message,
        }


@dataclass
class OutlineItem:
    """Document outline/bookmark item."""

    title: str
    page: Optional[int] = None
    level: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "page": self.page,
            "level": self.level,
        }


@dataclass
class FormField:
    """PDF form field information."""

    name: str
    field_type: str = "unknown"
    value: Any = None
    default_value: Any = None
    page: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "field_type": self.field_type,
            "value": self.value,
            "default_value": self.default_value,
            "page": self.page,
        }


@dataclass
class LayoutInfo:
    """Page layout analysis information."""

    page_number: int
    page_size: Dict[str, float] = field(default_factory=dict)
    margins: Dict[str, float] = field(default_factory=dict)
    columns: Dict[str, Any] = field(default_factory=dict)
    text_blocks: int = 0
    word_count: int = 0
    char_count: int = 0
    has_tables: bool = False
    has_images: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "page_number": self.page_number,
            "page_size": self.page_size,
            "margins": self.margins,
            "columns": self.columns,
            "text_blocks": self.text_blocks,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "has_tables": self.has_tables,
            "has_images": self.has_images,
        }


@dataclass
class CompressionResult:
    """Result of PDF compression operation."""

    input_path: str
    output_path: str
    original_size: int = 0
    compressed_size: int = 0
    reduction_percent: float = 0.0
    quality: str = "medium"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "original_size": self.original_size,
            "compressed_size": self.compressed_size,
            "reduction_percent": self.reduction_percent,
            "quality": self.quality,
        }


@dataclass
class ImageInfo:
    """Information about an extracted image."""

    path: str
    page: int
    index: int
    width: Optional[int] = None
    height: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "page": self.page,
            "index": self.index,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class BatchResult:
    """Result of batch processing operation."""

    total_files: int = 0
    successful: int = 0
    failed: int = 0
    results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_files": self.total_files,
            "successful": self.successful,
            "failed": self.failed,
            "results": self.results,
        }

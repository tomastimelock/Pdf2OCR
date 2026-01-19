# Filepath: code_migration/extraction/pdf_toolkit/exceptions.py
# Description: Custom exceptions for PDF Toolkit Provider
# Layer: Extraction

"""Custom exceptions for PDF Toolkit Provider."""


class PDFToolkitError(Exception):
    """Base exception for PDF Toolkit operations."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class ValidationError(PDFToolkitError):
    """Raised when PDF validation fails."""

    pass


class ExtractionError(PDFToolkitError):
    """Raised when text/table extraction fails."""

    pass


class ProcessingError(PDFToolkitError):
    """Raised when document processing fails."""

    pass


class EmbeddingError(PDFToolkitError):
    """Raised when embedding generation fails."""

    pass


class FileOperationError(PDFToolkitError):
    """Raised when file operations (split, merge) fail."""

    pass

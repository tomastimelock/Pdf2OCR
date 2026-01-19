# Filepath: code_migration/extraction/pdf_toolkit/utils.py
# Description: Utility functions for PDF Toolkit Provider
# Layer: Extraction

"""Utility functions for PDF Toolkit Provider."""

import os
import hashlib
from typing import List, Optional


def ensure_directory_exists(path: str) -> None:
    """Create directory if it doesn't exist."""
    if path:
        os.makedirs(path, exist_ok=True)


def get_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """
    Calculate hash of a file.

    Args:
        file_path: Path to the file
        algorithm: Hash algorithm (sha256, md5, etc.)

    Returns:
        Hash string
    """
    hash_func = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def get_file_size_formatted(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def validate_page_range(
    total_pages: int, pages: List[int]
) -> List[int]:
    """
    Validate and filter page numbers.

    Args:
        total_pages: Total number of pages in document
        pages: List of requested page numbers (1-indexed)

    Returns:
        Filtered list of valid page numbers
    """
    return [p for p in pages if 1 <= p <= total_pages]


def chunk_list(lst: list, chunk_size: int) -> List[list]:
    """
    Split a list into chunks.

    Args:
        lst: List to split
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")
    return filename


def get_temp_path(base_dir: Optional[str] = None, prefix: str = "pdftoolkit_") -> str:
    """
    Get a temporary file path.

    Args:
        base_dir: Base directory for temp file
        prefix: Prefix for temp filename

    Returns:
        Temporary file path
    """
    import tempfile
    import uuid

    if base_dir:
        ensure_directory_exists(base_dir)
        return os.path.join(base_dir, f"{prefix}{uuid.uuid4().hex[:8]}")

    return os.path.join(tempfile.gettempdir(), f"{prefix}{uuid.uuid4().hex[:8]}")

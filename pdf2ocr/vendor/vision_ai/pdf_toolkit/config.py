# Filepath: code_migration/extraction/pdf_toolkit/config.py
# Description: Configuration for PDF Toolkit Provider
# Layer: Extraction

"""Configuration for PDF Toolkit Provider."""

from dataclasses import dataclass, field
from typing import Optional
import logging


@dataclass
class ProviderConfig:
    """
    Configuration options for PDF Toolkit Provider.

    Attributes:
        embedding_model: Name of sentence-transformers model for embeddings
        embedding_enabled: Whether to enable embedding functionality
        ocr_enabled: Whether to enable OCR functionality
        ocr_language: Language code for OCR (default: 'eng')
        chunk_size: Default chunk size for text chunking
        chunk_overlap: Default overlap between chunks
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        temp_dir: Temporary directory for intermediate files
    """

    # Embedding settings
    embedding_model: str = "all-mpnet-base-v2"
    embedding_enabled: bool = True

    # OCR settings
    ocr_enabled: bool = True
    ocr_language: str = "eng"

    # Text chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Logging
    log_level: str = "INFO"

    # File handling
    temp_dir: Optional[str] = None

    # Processing options
    extract_images: bool = False
    extract_tables: bool = True

    def get_log_level(self) -> int:
        """Convert log level string to logging constant."""
        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        return levels.get(self.log_level.upper(), logging.INFO)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ProviderConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "embedding_model": self.embedding_model,
            "embedding_enabled": self.embedding_enabled,
            "ocr_enabled": self.ocr_enabled,
            "ocr_language": self.ocr_language,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "log_level": self.log_level,
            "temp_dir": self.temp_dir,
            "extract_images": self.extract_images,
            "extract_tables": self.extract_tables,
        }

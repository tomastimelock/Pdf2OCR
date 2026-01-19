# Filepath: code_migration/extraction/pdf_toolkit/provider.py
# Description: PDF Toolkit Provider - Main provider class
# Layer: Extraction
# References: reference_codebase/pdf_toolkit/

"""
PDF Toolkit Provider - Main provider class.

A self-contained provider for PDF processing that can be copied to other projects.
"""

import os
import hashlib
import logging
import time
from typing import List, Dict, Any, Optional, Union

from PyPDF2 import PdfReader, PdfWriter
import pdfplumber

from .config import ProviderConfig
from .dto import (
    DocumentInfo,
    PageInfo,
    TableInfo,
    ExtractionResult,
    ProcessingResult,
    EmbeddingResult,
    ProcessingStatus,
)
from .exceptions import (
    ValidationError,
    ExtractionError,
    ProcessingError,
    EmbeddingError,
    FileOperationError,
)


class PDFToolkitProvider:
    """
    Main provider class for PDF toolkit operations.

    Provides a clean, self-contained API for:
    - PDF validation and processing
    - Text extraction
    - Table extraction
    - PDF splitting and merging
    - Embedding generation (optional, requires sentence-transformers)

    Example:
        toolkit = PDFToolkitProvider()
        doc = toolkit.process_document("path/to/file.pdf")
        text = toolkit.extract_text("path/to/file.pdf")
    """

    def __init__(self, config: Optional[ProviderConfig] = None):
        """
        Initialize PDF Toolkit Provider.

        Args:
            config: Optional configuration object
        """
        self.config = config or ProviderConfig()
        self._setup_logging()
        self._embedding_model = None
        self.logger.info("PDFToolkitProvider initialized")

    def _setup_logging(self) -> None:
        """Set up logging for the provider."""
        self.logger = logging.getLogger("PDFToolkitProvider")
        self.logger.setLevel(self.config.get_log_level())

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(handler)

    # ==================== Validation ====================

    def validate_pdf(self, file_path: str) -> bool:
        """
        Validate if file is a valid PDF.

        Args:
            file_path: Path to the PDF file

        Returns:
            True if valid PDF, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                return False

            if not file_path.lower().endswith(".pdf"):
                return False

            with open(file_path, "rb") as f:
                PdfReader(f)

            return True
        except Exception as e:
            self.logger.warning(f"PDF validation failed for {file_path}: {e}")
            return False

    # ==================== Document Processing ====================

    def process_document(self, file_path: str) -> DocumentInfo:
        """
        Process a PDF document and extract metadata.

        Args:
            file_path: Path to the PDF file

        Returns:
            DocumentInfo with document metadata

        Raises:
            ValidationError: If file is not a valid PDF
            ProcessingError: If processing fails
        """
        start_time = time.time()

        if not self.validate_pdf(file_path):
            raise ValidationError(f"Invalid PDF file: {file_path}")

        try:
            metadata = self.extract_metadata(file_path)
            file_hash = self.calculate_hash(file_path)
            file_size = os.path.getsize(file_path)

            doc_info = DocumentInfo(
                file_name=os.path.basename(file_path),
                file_path=file_path,
                file_hash=file_hash,
                file_size=file_size,
                page_count=metadata.get("page_count", 0),
                title=metadata.get("title"),
                author=metadata.get("author"),
                subject=metadata.get("subject"),
                keywords=metadata.get("keywords"),
                creator=metadata.get("creator"),
                producer=metadata.get("producer"),
                creation_date=metadata.get("creation_date"),
                modification_date=metadata.get("modification_date"),
                metadata=metadata,
                status=ProcessingStatus.COMPLETED,
            )

            self.logger.info(
                f"Processed document: {doc_info.file_name} "
                f"({doc_info.page_count} pages) in {time.time() - start_time:.2f}s"
            )
            return doc_info

        except Exception as e:
            raise ProcessingError(f"Failed to process document: {e}")

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary containing document metadata
        """
        try:
            with open(file_path, "rb") as f:
                reader = PdfReader(f)

                metadata = {
                    "page_count": len(reader.pages),
                    "file_name": os.path.basename(file_path),
                }

                if reader.metadata:
                    metadata["title"] = reader.metadata.get("/Title")
                    metadata["author"] = reader.metadata.get("/Author")
                    metadata["subject"] = reader.metadata.get("/Subject")
                    metadata["creator"] = reader.metadata.get("/Creator")
                    metadata["producer"] = reader.metadata.get("/Producer")
                    metadata["keywords"] = reader.metadata.get("/Keywords")

                    if "/CreationDate" in reader.metadata:
                        metadata["creation_date"] = str(reader.metadata["/CreationDate"])
                    if "/ModDate" in reader.metadata:
                        metadata["modification_date"] = str(reader.metadata["/ModDate"])

                return metadata

        except Exception as e:
            self.logger.error(f"Failed to extract metadata from {file_path}: {e}")
            return {"page_count": 0}

    def calculate_hash(self, file_path: str) -> str:
        """
        Calculate SHA256 hash of file.

        Args:
            file_path: Path to the file

        Returns:
            SHA256 hash string
        """
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    # ==================== Text Extraction ====================

    def extract_text(
        self, file_path: str, pages: Optional[List[int]] = None
    ) -> str:
        """
        Extract text from PDF file.

        Args:
            file_path: Path to the PDF file
            pages: Optional list of page numbers (1-indexed) to extract

        Returns:
            Extracted text content

        Raises:
            ExtractionError: If extraction fails
        """
        try:
            text_parts = []

            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                total_pages = len(reader.pages)

                pages_to_extract = pages or list(range(1, total_pages + 1))

                for page_num in pages_to_extract:
                    if 1 <= page_num <= total_pages:
                        page = reader.pages[page_num - 1]
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)

            full_text = "\n\n".join(text_parts)
            self.logger.info(f"Extracted {len(full_text)} characters from {file_path}")
            return full_text

        except Exception as e:
            raise ExtractionError(f"Failed to extract text: {e}")

    def extract_text_from_page(self, file_path: str, page_number: int) -> str:
        """
        Extract text from a specific page.

        Args:
            file_path: Path to the PDF file
            page_number: Page number (1-indexed)

        Returns:
            Extracted text from the page
        """
        try:
            with open(file_path, "rb") as f:
                reader = PdfReader(f)

                if page_number < 1 or page_number > len(reader.pages):
                    raise ValueError(f"Invalid page number: {page_number}")

                page = reader.pages[page_number - 1]
                return page.extract_text() or ""

        except Exception as e:
            raise ExtractionError(f"Failed to extract text from page {page_number}: {e}")

    def extract_all_pages(self, file_path: str) -> ExtractionResult:
        """
        Extract text from all pages with detailed page information.

        Args:
            file_path: Path to the PDF file

        Returns:
            ExtractionResult with page-by-page extraction
        """
        start_time = time.time()

        try:
            pages = []
            full_text_parts = []

            with open(file_path, "rb") as f:
                reader = PdfReader(f)

                for page_num, pdf_page in enumerate(reader.pages, start=1):
                    text = pdf_page.extract_text() or ""
                    content_hash = hashlib.sha256(text.encode()).hexdigest()

                    mediabox = pdf_page.mediabox
                    width = int(float(mediabox.width))
                    height = int(float(mediabox.height))

                    page_info = PageInfo(
                        page_number=page_num,
                        content_text=text,
                        content_hash=content_hash,
                        width=width,
                        height=height,
                        rotation=int(pdf_page.get("/Rotate", 0) or 0),
                    )

                    pages.append(page_info)
                    if text:
                        full_text_parts.append(text)

            return ExtractionResult(
                file_path=file_path,
                total_pages=len(pages),
                pages=pages,
                full_text="\n\n".join(full_text_parts),
                extraction_time=time.time() - start_time,
                success=True,
            )

        except Exception as e:
            return ExtractionResult(
                file_path=file_path,
                success=False,
                error_message=str(e),
                extraction_time=time.time() - start_time,
            )

    # ==================== Table Extraction ====================

    def extract_tables(
        self, file_path: str, page_number: Optional[int] = None
    ) -> List[TableInfo]:
        """
        Extract tables from PDF.

        Args:
            file_path: Path to the PDF file
            page_number: Optional specific page number (1-indexed)

        Returns:
            List of TableInfo objects
        """
        try:
            tables = []

            with pdfplumber.open(file_path) as pdf:
                if page_number:
                    pages_to_process = [(page_number, pdf.pages[page_number - 1])]
                else:
                    pages_to_process = [
                        (idx + 1, page) for idx, page in enumerate(pdf.pages)
                    ]

                for page_idx, page in pages_to_process:
                    page_tables = page.extract_tables()

                    for table_idx, table in enumerate(page_tables):
                        if table:
                            table_info = TableInfo(
                                page_number=page_idx,
                                table_index=table_idx,
                                data=table,
                                headers=table[0] if table else None,
                            )
                            tables.append(table_info)

            self.logger.info(f"Extracted {len(tables)} tables from {file_path}")
            return tables

        except Exception as e:
            raise ExtractionError(f"Failed to extract tables: {e}")

    # ==================== PDF Operations ====================

    def split_pdf(
        self, file_path: str, output_dir: str, pages_per_split: int = 1
    ) -> List[str]:
        """
        Split PDF into multiple files.

        Args:
            file_path: Path to the PDF file
            output_dir: Output directory for split files
            pages_per_split: Number of pages per split file

        Returns:
            List of output file paths

        Raises:
            FileOperationError: If split fails
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            output_files = []

            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                total_pages = len(reader.pages)
                base_name = os.path.splitext(os.path.basename(file_path))[0]

                for i in range(0, total_pages, pages_per_split):
                    writer = PdfWriter()

                    for j in range(i, min(i + pages_per_split, total_pages)):
                        writer.add_page(reader.pages[j])

                    output_file = os.path.join(
                        output_dir,
                        f"{base_name}_pages_{i + 1}-{min(i + pages_per_split, total_pages)}.pdf",
                    )

                    with open(output_file, "wb") as output_f:
                        writer.write(output_f)

                    output_files.append(output_file)

            self.logger.info(f"Split {file_path} into {len(output_files)} files")
            return output_files

        except Exception as e:
            raise FileOperationError(f"Failed to split PDF: {e}")

    def merge_pdfs(self, file_paths: List[str], output_path: str) -> str:
        """
        Merge multiple PDFs into one file.

        Args:
            file_paths: List of PDF file paths to merge
            output_path: Output file path

        Returns:
            Path to merged PDF

        Raises:
            FileOperationError: If merge fails
        """
        try:
            writer = PdfWriter()

            for pdf_path in file_paths:
                if not os.path.exists(pdf_path):
                    raise FileOperationError(f"File not found: {pdf_path}")

                with open(pdf_path, "rb") as f:
                    reader = PdfReader(f)
                    for page in reader.pages:
                        writer.add_page(page)

            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(output_path, "wb") as output_f:
                writer.write(output_f)

            self.logger.info(f"Merged {len(file_paths)} PDFs into {output_path}")
            return output_path

        except FileOperationError:
            raise
        except Exception as e:
            raise FileOperationError(f"Failed to merge PDFs: {e}")

    def extract_pages_to_pdf(
        self, file_path: str, pages: List[int], output_path: str
    ) -> str:
        """
        Extract specific pages to a new PDF.

        Args:
            file_path: Source PDF file path
            pages: List of page numbers to extract (1-indexed)
            output_path: Output file path

        Returns:
            Path to output PDF
        """
        try:
            writer = PdfWriter()

            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                total_pages = len(reader.pages)

                for page_num in pages:
                    if 1 <= page_num <= total_pages:
                        writer.add_page(reader.pages[page_num - 1])

            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(output_path, "wb") as output_f:
                writer.write(output_f)

            self.logger.info(f"Extracted pages {pages} to {output_path}")
            return output_path

        except Exception as e:
            raise FileOperationError(f"Failed to extract pages: {e}")

    # ==================== OCR ====================

    def ocr_page(self, file_path: str, page_number: int) -> str:
        """
        Perform OCR on a specific page.

        Requires: pytesseract, pdf2image

        Args:
            file_path: Path to the PDF file
            page_number: Page number (1-indexed)

        Returns:
            OCR extracted text
        """
        if not self.config.ocr_enabled:
            self.logger.warning("OCR is disabled in configuration")
            return self.extract_text_from_page(file_path, page_number)

        try:
            from pdf2image import convert_from_path
            import pytesseract

            images = convert_from_path(
                file_path,
                first_page=page_number,
                last_page=page_number,
            )

            if images:
                text = pytesseract.image_to_string(
                    images[0], lang=self.config.ocr_language
                )
                return text

            return ""

        except ImportError:
            self.logger.warning(
                "OCR dependencies not installed. Install with: pip install pytesseract pdf2image"
            )
            return self.extract_text_from_page(file_path, page_number)
        except Exception as e:
            raise ExtractionError(f"OCR failed: {e}")

    # ==================== Embeddings ====================

    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            if not self.config.embedding_enabled:
                raise EmbeddingError("Embedding generation is disabled in configuration")

            try:
                from sentence_transformers import SentenceTransformer

                self._embedding_model = SentenceTransformer(self.config.embedding_model)
                self.logger.info(f"Loaded embedding model: {self.config.embedding_model}")
            except ImportError:
                raise EmbeddingError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )

        return self._embedding_model

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.

        Requires: sentence-transformers

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        model = self._get_embedding_model()

        if not text or not text.strip():
            return [0.0] * model.get_sentence_embedding_dimension()

        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        model = self._get_embedding_model()

        if not texts:
            return []

        non_empty_indices = [i for i, text in enumerate(texts) if text and text.strip()]
        non_empty_texts = [texts[i] for i in non_empty_indices]

        if not non_empty_texts:
            dim = model.get_sentence_embedding_dimension()
            return [[0.0] * dim] * len(texts)

        embeddings = model.encode(
            non_empty_texts,
            convert_to_numpy=True,
            show_progress_bar=len(non_empty_texts) > 10,
        )

        result = []
        non_empty_idx = 0
        dim = model.get_sentence_embedding_dimension()

        for i in range(len(texts)):
            if i in non_empty_indices:
                result.append(embeddings[non_empty_idx].tolist())
                non_empty_idx += 1
            else:
                result.append([0.0] * dim)

        return result

    def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> List[str]:
        """
        Split text into chunks for embedding.

        Args:
            text: Input text
            chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks in characters

        Returns:
            List of text chunks
        """
        if not text:
            return []

        chunk_size = chunk_size or self.config.chunk_size
        overlap = overlap or self.config.chunk_overlap

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            start = end - overlap if start > 0 else end

        return chunks

    # ==================== Document Analysis ====================

    def get_document_structure(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze document structure.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary with document structure information
        """
        try:
            with pdfplumber.open(file_path) as pdf:
                pages_structure = []

                for page in pdf.pages:
                    words = page.extract_words() or []
                    tables = page.extract_tables() or []
                    images = page.images or []

                    content_types = []
                    if words:
                        content_types.append("text")
                    if tables:
                        content_types.append("table")
                    if images:
                        content_types.append("image")

                    pages_structure.append(
                        {
                            "number": page.page_number,
                            "size": {"width": page.width, "height": page.height},
                            "content_types": content_types,
                            "word_count": len(words),
                            "table_count": len(tables),
                            "image_count": len(images),
                        }
                    )

                return {
                    "file_path": file_path,
                    "page_count": len(pdf.pages),
                    "pages": pages_structure,
                    "metadata": self.extract_metadata(file_path),
                }

        except Exception as e:
            raise ProcessingError(f"Failed to analyze document structure: {e}")

    # ==================== Compression ====================

    def compress_pdf(
        self,
        file_path: str,
        output_path: str,
        quality: str = "medium",
    ) -> Dict[str, Any]:
        """
        Compress a PDF file.

        Args:
            file_path: Path to the PDF file
            output_path: Output file path
            quality: Compression quality ('low', 'medium', 'high')

        Returns:
            Dictionary with compression results
        """
        if quality not in ["low", "medium", "high"]:
            raise ValueError(f"Invalid quality: {quality}. Use 'low', 'medium', or 'high'")

        try:
            original_size = os.path.getsize(file_path)

            reader = PdfReader(file_path)
            writer = PdfWriter()

            for page in reader.pages:
                writer.add_page(page)

            if reader.metadata:
                writer.add_metadata(reader.metadata)

            for page in writer.pages:
                page.compress_content_streams()

            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(output_path, "wb") as output_f:
                writer.write(output_f)

            compressed_size = os.path.getsize(output_path)
            reduction = ((original_size - compressed_size) / original_size) * 100

            self.logger.info(
                f"Compressed PDF: {original_size} -> {compressed_size} bytes "
                f"({reduction:.1f}% reduction)"
            )

            return {
                "input_path": file_path,
                "output_path": output_path,
                "original_size": original_size,
                "compressed_size": compressed_size,
                "reduction_percent": round(reduction, 2),
                "quality": quality,
            }

        except Exception as e:
            raise FileOperationError(f"Failed to compress PDF: {e}")

    # ==================== Encryption ====================

    def encrypt_pdf(
        self,
        file_path: str,
        output_path: str,
        user_password: str,
        owner_password: Optional[str] = None,
    ) -> str:
        """
        Encrypt a PDF with password protection.

        Args:
            file_path: Path to the PDF file
            output_path: Output file path
            user_password: Password required to open the PDF
            owner_password: Password for owner permissions (defaults to user_password)

        Returns:
            Path to encrypted PDF
        """
        try:
            reader = PdfReader(file_path)
            writer = PdfWriter()

            for page in reader.pages:
                writer.add_page(page)

            if reader.metadata:
                writer.add_metadata(reader.metadata)

            writer.encrypt(
                user_password=user_password,
                owner_password=owner_password or user_password,
            )

            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(output_path, "wb") as output_f:
                writer.write(output_f)

            self.logger.info(f"Encrypted PDF saved to {output_path}")
            return output_path

        except Exception as e:
            raise FileOperationError(f"Failed to encrypt PDF: {e}")

    def decrypt_pdf(
        self,
        file_path: str,
        output_path: str,
        password: str,
    ) -> str:
        """
        Decrypt a password-protected PDF.

        Args:
            file_path: Path to the encrypted PDF file
            output_path: Output file path
            password: Password to decrypt the PDF

        Returns:
            Path to decrypted PDF
        """
        try:
            reader = PdfReader(file_path)

            if reader.is_encrypted:
                reader.decrypt(password)

            writer = PdfWriter()

            for page in reader.pages:
                writer.add_page(page)

            if reader.metadata:
                writer.add_metadata(reader.metadata)

            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(output_path, "wb") as output_f:
                writer.write(output_f)

            self.logger.info(f"Decrypted PDF saved to {output_path}")
            return output_path

        except Exception as e:
            raise FileOperationError(f"Failed to decrypt PDF: {e}")

    # ==================== Utility Methods ====================

    def get_page_count(self, file_path: str) -> int:
        """Get the number of pages in a PDF."""
        try:
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                return len(reader.pages)
        except Exception as e:
            raise ProcessingError(f"Failed to get page count: {e}")

    def is_encrypted(self, file_path: str) -> bool:
        """Check if PDF is encrypted."""
        try:
            reader = PdfReader(file_path)
            return reader.is_encrypted
        except Exception:
            return False

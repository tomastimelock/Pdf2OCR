"""Document Processing Workflow Preset.

Pre-configured workflow for document digitization and archiving.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from enum import Enum


class DocumentCategory(Enum):
    """Document category types."""
    RECEIPT = "receipt"
    INVOICE = "invoice"
    CONTRACT = "contract"
    LETTER = "letter"
    FORM = "form"
    ID = "id"
    STATEMENT = "statement"
    GENERAL = "general"


@dataclass
class ScanConfig:
    """Scanning configuration."""
    enhance: bool = True
    deskew: bool = True
    remove_noise: bool = True
    auto_crop: bool = True
    output_format: str = "pdf"


@dataclass
class OcrConfig:
    """OCR configuration."""
    enabled: bool = True
    languages: List[str] = field(default_factory=lambda: ['eng'])
    detect_orientation: bool = True
    preserve_layout: bool = False


@dataclass
class ClassificationConfig:
    """Document classification configuration."""
    auto_classify: bool = True
    parse_receipts: bool = True
    extract_dates: bool = True
    extract_amounts: bool = True


@dataclass
class ArchiveConfig:
    """Archive configuration."""
    organize_by_type: bool = True
    organize_by_date: bool = True
    generate_checksum: bool = True
    create_searchable: bool = True
    backup: bool = False


@dataclass
class DocumentPresetConfig:
    """Complete document workflow configuration."""
    categories: List[DocumentCategory] = field(default_factory=lambda: list(DocumentCategory))
    scan: ScanConfig = field(default_factory=ScanConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)

    def to_dict(self) -> Dict:
        return {
            'categories': [c.value for c in self.categories],
            'scan': {
                'enhance': self.scan.enhance,
                'output_format': self.scan.output_format,
            },
            'ocr': {
                'enabled': self.ocr.enabled,
                'languages': self.ocr.languages,
            },
            'classification': {
                'auto_classify': self.classification.auto_classify,
                'parse_receipts': self.classification.parse_receipts,
            },
            'archive': {
                'organize_by_type': self.archive.organize_by_type,
                'create_searchable': self.archive.create_searchable,
            },
        }


class DocumentPreset:
    """Document processing workflow preset.

    Provides pre-configured workflows for document processing including:
    - Receipt scanning and expense tracking
    - Contract management
    - General document archiving
    - ID document processing
    """

    RECEIPT_TRACKING = DocumentPresetConfig(
        categories=[DocumentCategory.RECEIPT],
        scan=ScanConfig(enhance=True, auto_crop=True),
        ocr=OcrConfig(enabled=True),
        classification=ClassificationConfig(
            auto_classify=True,
            parse_receipts=True,
            extract_amounts=True,
        ),
        archive=ArchiveConfig(organize_by_date=True),
    )

    CONTRACT_MANAGEMENT = DocumentPresetConfig(
        categories=[DocumentCategory.CONTRACT],
        scan=ScanConfig(enhance=True, output_format='pdf'),
        ocr=OcrConfig(enabled=True, preserve_layout=True),
        classification=ClassificationConfig(
            auto_classify=True,
            extract_dates=True,
        ),
        archive=ArchiveConfig(
            generate_checksum=True,
            backup=True,
        ),
    )

    GENERAL_ARCHIVE = DocumentPresetConfig(
        scan=ScanConfig(enhance=True),
        ocr=OcrConfig(enabled=True),
        classification=ClassificationConfig(auto_classify=True),
        archive=ArchiveConfig(
            organize_by_type=True,
            organize_by_date=True,
            create_searchable=True,
        ),
    )

    ID_PROCESSING = DocumentPresetConfig(
        categories=[DocumentCategory.ID],
        scan=ScanConfig(enhance=True, auto_crop=True),
        ocr=OcrConfig(enabled=True),
        classification=ClassificationConfig(auto_classify=False),
        archive=ArchiveConfig(
            generate_checksum=True,
            backup=True,
        ),
    )

    def __init__(self, config: Optional[DocumentPresetConfig] = None):
        """
        Initialize document preset.

        Args:
            config: Custom configuration or use one of the preset constants
        """
        self.config = config or DocumentPresetConfig()
        self._vault = None

    def _get_vault(self, vault_path: str):
        """Lazy load DocumentVault."""
        if self._vault is None:
            try:
                from document_vault import DocumentVault
                self._vault = DocumentVault(vault_path=vault_path)
            except ImportError:
                pass
        return self._vault

    def process(
        self,
        document: Union[str, Path],
        vault_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Process document using preset configuration.

        Args:
            document: Document image path
            vault_path: Path to document vault

        Returns:
            Processing result dictionary
        """
        doc_path = Path(document)
        vault_dir = Path(vault_path)

        result = {
            'source': str(doc_path),
            'vault': str(vault_dir),
            'success': False,
        }

        vault = self._get_vault(str(vault_dir))

        if vault:
            try:
                ingest_result = vault.ingest(
                    document=doc_path,
                    enhance=self.config.scan.enhance,
                )

                result.update({
                    'doc_id': ingest_result.doc_id,
                    'doc_type': ingest_result.doc_type.value,
                    'archived_path': ingest_result.archived_path,
                    'text_extracted': ingest_result.text_extracted,
                    'ocr_confidence': ingest_result.ocr_confidence,
                    'is_receipt': ingest_result.is_receipt,
                    'success': ingest_result.success,
                })

                if ingest_result.receipt_data:
                    result['receipt'] = ingest_result.receipt_data.to_dict()

            except Exception as e:
                result['error'] = str(e)
        else:
            result['error'] = "DocumentVault module not available"

        return result

    def batch_process(
        self,
        directory: Union[str, Path],
        vault_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Process multiple documents.

        Args:
            directory: Directory containing documents
            vault_path: Path to document vault

        Returns:
            Batch processing result
        """
        source = Path(directory)
        vault_dir = Path(vault_path)

        result = {
            'source': str(source),
            'vault': str(vault_dir),
            'total': 0,
            'succeeded': 0,
            'failed': 0,
            'receipts_found': 0,
            'success': False,
        }

        vault = self._get_vault(str(vault_dir))

        if vault:
            try:
                ingest_results = vault.batch_ingest(source)
                result['total'] = len(ingest_results)

                for ir in ingest_results:
                    if ir.success:
                        result['succeeded'] += 1
                        if ir.is_receipt:
                            result['receipts_found'] += 1
                    else:
                        result['failed'] += 1

                result['success'] = result['failed'] == 0

            except Exception as e:
                result['error'] = str(e)
        else:
            result['error'] = "DocumentVault module not available"

        return result

    def search(
        self,
        vault_path: Union[str, Path],
        query: str,
        doc_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search documents in vault.

        Args:
            vault_path: Path to document vault
            query: Search query
            doc_type: Optional document type filter

        Returns:
            List of matching documents
        """
        vault = self._get_vault(str(vault_path))

        if vault:
            try:
                from document_vault import DocumentType

                dt = DocumentType(doc_type) if doc_type else None
                results = vault.search(query, doc_type=dt)

                return [r.to_dict() for r in results]

            except Exception:
                pass

        return []

    def get_spending_summary(
        self,
        vault_path: Union[str, Path],
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get spending summary from receipts.

        Args:
            vault_path: Path to document vault
            date_from: Start date filter
            date_to: End date filter

        Returns:
            Spending summary
        """
        vault = self._get_vault(str(vault_path))

        if vault:
            try:
                return vault.get_spending_summary(date_from, date_to)
            except Exception:
                pass

        return {}

    @staticmethod
    def list_presets() -> List[str]:
        """List available preset names."""
        return ['RECEIPT_TRACKING', 'CONTRACT_MANAGEMENT', 'GENERAL_ARCHIVE', 'ID_PROCESSING']

    @classmethod
    def get_preset(cls, name: str) -> DocumentPresetConfig:
        """Get a preset by name."""
        presets = {
            'RECEIPT_TRACKING': cls.RECEIPT_TRACKING,
            'CONTRACT_MANAGEMENT': cls.CONTRACT_MANAGEMENT,
            'GENERAL_ARCHIVE': cls.GENERAL_ARCHIVE,
            'ID_PROCESSING': cls.ID_PROCESSING,
        }
        return presets.get(name.upper(), DocumentPresetConfig())

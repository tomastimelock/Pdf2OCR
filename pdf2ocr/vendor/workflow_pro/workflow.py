"""WorkflowPro - Unified Automated Workflow Engine.

Consolidates all automated workflow functionality into a single
powerful interface for image and document processing pipelines.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime
from enum import Enum
import json


class WorkflowError(Exception):
    """Error during workflow operations."""
    pass


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepType(Enum):
    """Pipeline step types."""
    ORGANIZE = "organize"
    MODERATE = "moderate"
    ENHANCE = "enhance"
    RESIZE = "resize"
    CONVERT = "convert"
    WATERMARK = "watermark"
    EXPORT = "export"
    BACKUP = "backup"
    CUSTOM = "custom"


@dataclass
class WorkflowConfig:
    """Workflow configuration."""
    # General
    name: str = "default"
    description: str = ""

    # Processing
    parallel: bool = False
    max_workers: int = 4
    fail_fast: bool = True

    # Output
    output_dir: Optional[str] = None
    preserve_originals: bool = True

    # Logging
    log_level: str = "info"
    log_file: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'parallel': self.parallel,
            'max_workers': self.max_workers,
            'output_dir': self.output_dir,
        }


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""
    workflow_name: str
    source: str
    output: str
    status: WorkflowStatus
    start_time: str
    end_time: str
    duration: float
    files_processed: int
    files_succeeded: int
    files_failed: int
    steps_completed: int
    steps_total: int
    outputs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True

    def to_dict(self) -> Dict:
        return {
            'workflow': self.workflow_name,
            'source': self.source,
            'output': self.output,
            'status': self.status.value,
            'duration': self.duration,
            'files_processed': self.files_processed,
            'files_succeeded': self.files_succeeded,
            'files_failed': self.files_failed,
            'steps': f"{self.steps_completed}/{self.steps_total}",
            'success': self.success,
        }


@dataclass
class OrganizeResult:
    """Result of smart organization."""
    source: str
    total_files: int
    organized: int
    duplicates_found: int
    collections_created: int
    ai_analyzed: bool
    categories: Dict[str, int] = field(default_factory=dict)
    success: bool = True
    error: str = ""

    def to_dict(self) -> Dict:
        return {
            'source': self.source,
            'total_files': self.total_files,
            'organized': self.organized,
            'duplicates': self.duplicates_found,
            'collections': self.collections_created,
            'ai_analyzed': self.ai_analyzed,
            'success': self.success,
        }


@dataclass
class ModerationResult:
    """Result of content moderation."""
    total_checked: int
    safe: int
    flagged: int
    blocked: int
    actions_taken: List[str] = field(default_factory=list)
    threat_summary: Dict[str, int] = field(default_factory=dict)
    success: bool = True

    def to_dict(self) -> Dict:
        return {
            'total': self.total_checked,
            'safe': self.safe,
            'flagged': self.flagged,
            'blocked': self.blocked,
            'success': self.success,
        }


@dataclass
class ComplianceResult:
    """Result of brand compliance check."""
    total_checked: int
    compliant: int
    warnings: int
    violations: int
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    success: bool = True

    def to_dict(self) -> Dict:
        return {
            'total': self.total_checked,
            'compliant': self.compliant,
            'warnings': self.warnings,
            'violations': self.violations,
            'success': self.success,
        }


@dataclass
class PipelineStep:
    """A single step in a pipeline."""
    step_id: str
    step_type: StepType
    operation: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    on_error: str = "stop"  # stop, skip, retry
    condition: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'step_id': self.step_id,
            'type': self.step_type.value,
            'operation': self.operation,
            'on_error': self.on_error,
        }


@dataclass
class Pipeline:
    """A complete processing pipeline."""
    name: str
    description: str = ""
    steps: List[PipelineStep] = field(default_factory=list)
    config: Optional[WorkflowConfig] = None
    created_at: str = ""

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'steps': [s.to_dict() for s in self.steps],
            'created_at': self.created_at,
        }


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    pipeline_name: str
    source: str
    status: WorkflowStatus
    steps_completed: int
    steps_total: int
    outputs: List[str] = field(default_factory=list)
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    duration: float = 0.0
    success: bool = True
    error: str = ""

    def to_dict(self) -> Dict:
        return {
            'pipeline': self.pipeline_name,
            'status': self.status.value,
            'steps': f"{self.steps_completed}/{self.steps_total}",
            'duration': self.duration,
            'success': self.success,
        }


@dataclass
class ScheduleResult:
    """Result of scheduling a pipeline."""
    schedule_id: str
    pipeline_name: str
    schedule: str
    next_run: str
    active: bool = True
    success: bool = True

    def to_dict(self) -> Dict:
        return {
            'schedule_id': self.schedule_id,
            'pipeline': self.pipeline_name,
            'schedule': self.schedule,
            'next_run': self.next_run,
            'active': self.active,
        }


class PipelineBuilder:
    """Builder for creating custom pipelines."""

    def __init__(self, name: str):
        """
        Initialize pipeline builder.

        Args:
            name: Pipeline name
        """
        self.name = name
        self.description = ""
        self._steps: List[PipelineStep] = []
        self._step_counter = 0
        self._on_error = "stop"
        self._config = WorkflowConfig()

    def describe(self, description: str) -> 'PipelineBuilder':
        """Add pipeline description."""
        self.description = description
        return self

    def add_step(
        self,
        operation: str,
        step_type: StepType = StepType.CUSTOM,
        **kwargs,
    ) -> 'PipelineBuilder':
        """
        Add a step to the pipeline.

        Args:
            operation: Operation name
            step_type: Type of step
            **kwargs: Operation parameters
        """
        self._step_counter += 1
        step = PipelineStep(
            step_id=f"step_{self._step_counter}",
            step_type=step_type,
            operation=operation,
            kwargs=kwargs,
            on_error=self._on_error,
        )
        self._steps.append(step)
        return self

    def organize(self, use_ai: bool = True, **kwargs) -> 'PipelineBuilder':
        """Add organization step."""
        return self.add_step("organize", StepType.ORGANIZE, use_ai=use_ai, **kwargs)

    def moderate(self, policy: Optional[str] = None, **kwargs) -> 'PipelineBuilder':
        """Add moderation step."""
        return self.add_step("moderate", StepType.MODERATE, policy=policy, **kwargs)

    def enhance(self, quality: str = "auto", **kwargs) -> 'PipelineBuilder':
        """Add enhancement step."""
        return self.add_step("enhance", StepType.ENHANCE, quality=quality, **kwargs)

    def resize(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        **kwargs,
    ) -> 'PipelineBuilder':
        """Add resize step."""
        return self.add_step("resize", StepType.RESIZE, width=width, height=height, **kwargs)

    def convert(self, format: str, **kwargs) -> 'PipelineBuilder':
        """Add format conversion step."""
        return self.add_step("convert", StepType.CONVERT, format=format, **kwargs)

    def watermark(self, text: str, **kwargs) -> 'PipelineBuilder':
        """Add watermark step."""
        return self.add_step("watermark", StepType.WATERMARK, text=text, **kwargs)

    def export(self, destination: str, **kwargs) -> 'PipelineBuilder':
        """Add export step."""
        return self.add_step("export", StepType.EXPORT, destination=destination, **kwargs)

    def backup(self, location: str, **kwargs) -> 'PipelineBuilder':
        """Add backup step."""
        return self.add_step("backup", StepType.BACKUP, location=location, **kwargs)

    def on_error(self, action: str) -> 'PipelineBuilder':
        """
        Set error handling behavior.

        Args:
            action: "stop", "skip", or "retry"
        """
        if action not in ["stop", "skip", "retry"]:
            raise WorkflowError(f"Invalid error action: {action}")
        self._on_error = action
        return self

    def configure(self, **kwargs) -> 'PipelineBuilder':
        """Set pipeline configuration."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        return self

    def build(self) -> Pipeline:
        """Build and return the pipeline."""
        return Pipeline(
            name=self.name,
            description=self.description,
            steps=self._steps.copy(),
            config=self._config,
            created_at=datetime.now().isoformat(),
        )


class WorkflowPro:
    """Unified automated workflow engine.

    Consolidates smart_organizer, content_shield, pro_print_workflow,
    document_vault, event_photographer, brand_guardian, product_studio,
    and surveillance_ai into a single powerful interface.
    """

    def __init__(
        self,
        config: Optional[WorkflowConfig] = None,
        database_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize WorkflowPro.

        Args:
            config: Workflow configuration
            database_path: Path to workflow database
        """
        self.config = config or WorkflowConfig()
        self.database_path = Path(database_path) if database_path else None

        # Registered pipelines
        self._pipelines: Dict[str, Pipeline] = {}

        # Scheduled pipelines
        self._schedules: Dict[str, ScheduleResult] = {}

        # Lazy-loaded modules
        self._smart_organizer = None
        self._content_shield = None
        self._pro_print_workflow = None
        self._document_vault = None
        self._event_photographer = None
        self._brand_guardian = None
        self._product_studio = None
        self._surveillance_ai = None

        # Super-module references
        self._image_core = None
        self._vision_ai = None
        self._media_vault = None
        self._export_hub = None

    # =========================================================================
    # Lazy-loaded module getters
    # =========================================================================

    def _get_smart_organizer(self):
        """Lazy load SmartOrganizer."""
        if self._smart_organizer is None:
            try:
                from smart_organizer import SmartOrganizer
                self._smart_organizer = SmartOrganizer()
            except ImportError:
                pass
        return self._smart_organizer

    def _get_content_shield(self):
        """Lazy load ContentShield."""
        if self._content_shield is None:
            try:
                from content_shield import ContentShield
                self._content_shield = ContentShield()
            except ImportError:
                pass
        return self._content_shield

    def _get_pro_print_workflow(self):
        """Lazy load ProPrintWorkflow."""
        if self._pro_print_workflow is None:
            try:
                from pro_print_workflow import ProPrintWorkflow
                self._pro_print_workflow = ProPrintWorkflow()
            except ImportError:
                pass
        return self._pro_print_workflow

    def _get_document_vault(self, vault_path: Optional[str] = None):
        """Lazy load DocumentVault."""
        if self._document_vault is None:
            try:
                from document_vault import DocumentVault
                path = vault_path or (str(self.database_path.parent / 'vault') if self.database_path else './vault')
                self._document_vault = DocumentVault(vault_path=path)
            except ImportError:
                pass
        return self._document_vault

    def _get_event_photographer(self):
        """Lazy load EventPhotographer."""
        if self._event_photographer is None:
            try:
                from event_photographer import EventPhotographer
                self._event_photographer = EventPhotographer()
            except ImportError:
                pass
        return self._event_photographer

    def _get_brand_guardian(self):
        """Lazy load BrandGuardian."""
        if self._brand_guardian is None:
            try:
                from brand_guardian import BrandGuardian
                self._brand_guardian = BrandGuardian()
            except ImportError:
                pass
        return self._brand_guardian

    def _get_product_studio(self):
        """Lazy load ProductStudio."""
        if self._product_studio is None:
            try:
                from product_studio import ProductStudio
                self._product_studio = ProductStudio()
            except ImportError:
                pass
        return self._product_studio

    def _get_surveillance_ai(self):
        """Lazy load SurveillanceAI."""
        if self._surveillance_ai is None:
            try:
                from surveillance_ai import SurveillanceAI
                self._surveillance_ai = SurveillanceAI()
            except ImportError:
                pass
        return self._surveillance_ai

    def _get_image_core(self):
        """Lazy load ImageCore super-module."""
        if self._image_core is None:
            try:
                from image_core import ImageCore
                self._image_core = ImageCore()
            except ImportError:
                pass
        return self._image_core

    def _get_vision_ai(self):
        """Lazy load VisionAI super-module."""
        if self._vision_ai is None:
            try:
                from vision_ai import VisionAI
                self._vision_ai = VisionAI()
            except ImportError:
                pass
        return self._vision_ai

    def _get_media_vault(self):
        """Lazy load MediaVault super-module."""
        if self._media_vault is None:
            try:
                from media_vault import MediaVault
                self._media_vault = MediaVault()
            except ImportError:
                pass
        return self._media_vault

    def _get_export_hub(self):
        """Lazy load ExportHub super-module."""
        if self._export_hub is None:
            try:
                from export_hub import ExportHub
                self._export_hub = ExportHub()
            except ImportError:
                pass
        return self._export_hub

    # =========================================================================
    # Preset Workflows
    # =========================================================================

    def event_workflow(
        self,
        source: Union[str, Path],
        output: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
    ) -> WorkflowResult:
        """
        Run event photography workflow.

        Pipeline:
        1. Import and organize photos
        2. Quality assessment and culling
        3. Face detection and grouping
        4. Auto-create albums
        5. Generate slideshow
        6. Build web gallery

        Args:
            source: Source directory with event photos
            output: Output directory
            config: Optional workflow configuration
        """
        import time
        start_time = time.time()

        source_path = Path(source)
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)

        cfg = config or {}

        result = WorkflowResult(
            workflow_name="event_photography",
            source=str(source_path),
            output=str(output_path),
            status=WorkflowStatus.RUNNING,
            start_time=datetime.now().isoformat(),
            end_time="",
            duration=0,
            files_processed=0,
            files_succeeded=0,
            files_failed=0,
            steps_completed=0,
            steps_total=6,
        )

        try:
            photographer = self._get_event_photographer()

            if photographer:
                event_name = cfg.get('event_name', source_path.name)
                event_result = photographer.process_event(
                    source_dir=source_path,
                    event_name=event_name,
                    output_dir=output_path,
                    cull=cfg.get('cull', True),
                    detect_people=cfg.get('detect_people', True),
                    create_albums=cfg.get('create_albums', True),
                    create_slideshow=cfg.get('create_slideshow', True),
                    create_gallery=cfg.get('create_gallery', True),
                )

                result.files_processed = event_result.total_photos
                result.files_succeeded = event_result.selected_photos
                result.files_failed = event_result.total_photos - event_result.selected_photos
                result.steps_completed = 6 if event_result.success else 3
                result.metadata = {
                    'albums_created': event_result.albums_created,
                    'people_detected': event_result.people_detected,
                    'slideshow': event_result.slideshow_path,
                    'gallery': event_result.gallery_path,
                }

                if event_result.slideshow_path:
                    result.outputs.append(event_result.slideshow_path)
                if event_result.gallery_path:
                    result.outputs.append(event_result.gallery_path)

                result.success = event_result.success
                if not event_result.success:
                    result.errors.append(event_result.error)
            else:
                # Fallback: basic organization
                result = self._basic_event_workflow(source_path, output_path, cfg)

            result.status = WorkflowStatus.COMPLETED if result.success else WorkflowStatus.FAILED

        except Exception as e:
            result.success = False
            result.status = WorkflowStatus.FAILED
            result.errors.append(str(e))

        result.end_time = datetime.now().isoformat()
        result.duration = time.time() - start_time

        return result

    def _basic_event_workflow(
        self,
        source: Path,
        output: Path,
        config: Dict[str, Any],
    ) -> WorkflowResult:
        """Fallback basic event workflow."""
        from PIL import Image
        import shutil

        result = WorkflowResult(
            workflow_name="event_photography",
            source=str(source),
            output=str(output),
            status=WorkflowStatus.RUNNING,
            start_time=datetime.now().isoformat(),
            end_time="",
            duration=0,
            files_processed=0,
            files_succeeded=0,
            files_failed=0,
            steps_completed=0,
            steps_total=6,
        )

        # Find images
        extensions = ['*.jpg', '*.jpeg', '*.png']
        images = []
        for ext in extensions:
            images.extend(source.rglob(ext))
            images.extend(source.rglob(ext.upper()))

        result.files_processed = len(images)

        # Copy to output
        for img_path in images:
            try:
                dest = output / img_path.name
                shutil.copy2(img_path, dest)
                result.files_succeeded += 1
                result.outputs.append(str(dest))
            except Exception:
                result.files_failed += 1

        result.steps_completed = 2
        result.success = result.files_failed == 0

        return result

    def product_workflow(
        self,
        source: Union[str, Path],
        output: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
    ) -> WorkflowResult:
        """
        Run product photography workflow.

        Pipeline:
        1. Remove backgrounds
        2. Extract product colors
        3. Generate platform-specific images
        4. Create mockups
        5. Generate thumbnails

        Args:
            source: Source product image(s)
            output: Output directory
            config: Optional workflow configuration
        """
        import time
        start_time = time.time()

        source_path = Path(source)
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)

        cfg = config or {}

        result = WorkflowResult(
            workflow_name="product_studio",
            source=str(source_path),
            output=str(output_path),
            status=WorkflowStatus.RUNNING,
            start_time=datetime.now().isoformat(),
            end_time="",
            duration=0,
            files_processed=0,
            files_succeeded=0,
            files_failed=0,
            steps_completed=0,
            steps_total=5,
        )

        try:
            studio = self._get_product_studio()

            if studio:
                # Handle single image or directory
                if source_path.is_file():
                    package = studio.process_product(
                        image=source_path,
                        output_dir=output_path,
                        remove_background=cfg.get('remove_background', True),
                        generate_mockups=cfg.get('generate_mockups', True),
                        extract_colors=cfg.get('extract_colors', True),
                        add_watermark=cfg.get('add_watermark', False),
                    )

                    result.files_processed = 1
                    result.files_succeeded = 1 if package.success else 0
                    result.files_failed = 0 if package.success else 1
                    result.outputs = [a.path for a in package.assets]
                    result.metadata = {
                        'colors': package.colors.to_dict() if package.colors else None,
                        'transparent_png': package.transparent_png,
                        'total_assets': package.total_assets,
                    }
                    result.success = package.success

                else:
                    packages = studio.batch_process(source_path, output_path)
                    result.files_processed = len(packages)

                    for package in packages:
                        if package.success:
                            result.files_succeeded += 1
                            result.outputs.extend([a.path for a in package.assets])
                        else:
                            result.files_failed += 1

                    result.success = result.files_failed == 0

                result.steps_completed = 5 if result.success else 2
            else:
                result.errors.append("ProductStudio module not available")
                result.success = False

            result.status = WorkflowStatus.COMPLETED if result.success else WorkflowStatus.FAILED

        except Exception as e:
            result.success = False
            result.status = WorkflowStatus.FAILED
            result.errors.append(str(e))

        result.end_time = datetime.now().isoformat()
        result.duration = time.time() - start_time

        return result

    def document_workflow(
        self,
        source: Union[str, Path],
        output: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
    ) -> WorkflowResult:
        """
        Run document processing workflow.

        Pipeline:
        1. Scan/enhance documents
        2. OCR text extraction
        3. Document classification
        4. Receipt parsing (if applicable)
        5. Archive with metadata
        6. Index for search

        Args:
            source: Source document(s)
            output: Output vault directory
            config: Optional workflow configuration
        """
        import time
        start_time = time.time()

        source_path = Path(source)
        output_path = Path(output)

        cfg = config or {}

        result = WorkflowResult(
            workflow_name="document_vault",
            source=str(source_path),
            output=str(output_path),
            status=WorkflowStatus.RUNNING,
            start_time=datetime.now().isoformat(),
            end_time="",
            duration=0,
            files_processed=0,
            files_succeeded=0,
            files_failed=0,
            steps_completed=0,
            steps_total=6,
        )

        try:
            vault = self._get_document_vault(str(output_path))

            if vault:
                if source_path.is_file():
                    ingest_result = vault.ingest(
                        document=source_path,
                        enhance=cfg.get('enhance', True),
                    )

                    result.files_processed = 1
                    result.files_succeeded = 1 if ingest_result.success else 0
                    result.files_failed = 0 if ingest_result.success else 1

                    if ingest_result.success:
                        result.outputs.append(ingest_result.archived_path)

                    result.metadata = {
                        'doc_id': ingest_result.doc_id,
                        'doc_type': ingest_result.doc_type.value,
                        'text_extracted': ingest_result.text_extracted,
                        'is_receipt': ingest_result.is_receipt,
                    }
                    result.success = ingest_result.success

                else:
                    ingest_results = vault.batch_ingest(source_path)
                    result.files_processed = len(ingest_results)

                    receipts = 0
                    for ir in ingest_results:
                        if ir.success:
                            result.files_succeeded += 1
                            result.outputs.append(ir.archived_path)
                            if ir.is_receipt:
                                receipts += 1
                        else:
                            result.files_failed += 1

                    result.metadata = {
                        'receipts_found': receipts,
                    }
                    result.success = result.files_failed == 0

                result.steps_completed = 6 if result.success else 3
            else:
                result.errors.append("DocumentVault module not available")
                result.success = False

            result.status = WorkflowStatus.COMPLETED if result.success else WorkflowStatus.FAILED

        except Exception as e:
            result.success = False
            result.status = WorkflowStatus.FAILED
            result.errors.append(str(e))

        result.end_time = datetime.now().isoformat()
        result.duration = time.time() - start_time

        return result

    def print_workflow(
        self,
        source: Union[str, Path],
        output: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
    ) -> WorkflowResult:
        """
        Run print preparation workflow.

        Pipeline:
        1. RAW development (if applicable)
        2. Lens correction
        3. Noise reduction
        4. Color analysis
        5. Image enhancement
        6. Print preparation

        Args:
            source: Source image(s)
            output: Output directory
            config: Optional workflow configuration
        """
        import time
        start_time = time.time()

        source_path = Path(source)
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)

        cfg = config or {}

        result = WorkflowResult(
            workflow_name="print_preparation",
            source=str(source_path),
            output=str(output_path),
            status=WorkflowStatus.RUNNING,
            start_time=datetime.now().isoformat(),
            end_time="",
            duration=0,
            files_processed=0,
            files_succeeded=0,
            files_failed=0,
            steps_completed=0,
            steps_total=6,
        )

        try:
            workflow = self._get_pro_print_workflow()

            if workflow:
                if source_path.is_file():
                    wf_result = workflow.process(
                        image=source_path,
                        output=output_path / f"{source_path.stem}_print_ready.tiff",
                    )

                    result.files_processed = 1
                    result.files_succeeded = 1 if wf_result.success else 0
                    result.files_failed = 0 if wf_result.success else 1
                    result.outputs.append(wf_result.final_output)
                    result.metadata = {
                        'color_info': wf_result.color_info,
                        'print_ready': wf_result.print_ready,
                        'steps': [s.to_dict() for s in wf_result.steps],
                    }
                    result.success = wf_result.success
                    result.steps_completed = len(wf_result.steps)

                else:
                    wf_results = workflow.batch_process(source_path, output_path)
                    result.files_processed = len(wf_results)

                    for wf in wf_results:
                        if wf.success:
                            result.files_succeeded += 1
                            result.outputs.append(wf.final_output)
                        else:
                            result.files_failed += 1

                    result.success = result.files_failed == 0
                    result.steps_completed = 6 if result.success else 3
            else:
                result.errors.append("ProPrintWorkflow module not available")
                result.success = False

            result.status = WorkflowStatus.COMPLETED if result.success else WorkflowStatus.FAILED

        except Exception as e:
            result.success = False
            result.status = WorkflowStatus.FAILED
            result.errors.append(str(e))

        result.end_time = datetime.now().isoformat()
        result.duration = time.time() - start_time

        return result

    # =========================================================================
    # Smart Operations
    # =========================================================================

    def smart_organize(
        self,
        source: Union[str, Path],
        output: Union[str, Path],
        use_ai: bool = True,
    ) -> OrganizeResult:
        """
        Smart organize photos using AI.

        Features:
        - Scene classification
        - Face detection and grouping
        - Date/location organization
        - Duplicate detection
        - Auto-collection creation

        Args:
            source: Source directory
            output: Output directory
            use_ai: Enable AI-powered features
        """
        source_path = Path(source)
        output_path = Path(output)

        result = OrganizeResult(
            source=str(source_path),
            total_files=0,
            organized=0,
            duplicates_found=0,
            collections_created=0,
            ai_analyzed=use_ai,
        )

        try:
            organizer = self._get_smart_organizer()

            if organizer:
                org_result = organizer.organize_library(
                    source_dir=source_path,
                    recursive=True,
                    detect_duplicates=True,
                    create_collections=True,
                )

                result.total_files = org_result.total_photos
                result.organized = org_result.photos_organized
                result.duplicates_found = org_result.duplicates_found
                result.collections_created = org_result.collections_created
                result.success = org_result.success

            else:
                # Fallback: basic organization by date
                result = self._basic_organize(source_path, output_path)

        except Exception as e:
            result.success = False
            result.error = str(e)

        return result

    def _basic_organize(self, source: Path, output: Path) -> OrganizeResult:
        """Basic file organization by date."""
        from PIL import Image
        from PIL.ExifTags import TAGS
        import shutil

        result = OrganizeResult(
            source=str(source),
            total_files=0,
            organized=0,
            duplicates_found=0,
            collections_created=0,
            ai_analyzed=False,
        )

        extensions = ['*.jpg', '*.jpeg', '*.png']
        images = []
        for ext in extensions:
            images.extend(source.rglob(ext))
            images.extend(source.rglob(ext.upper()))

        result.total_files = len(images)

        for img_path in images:
            try:
                # Try to get date from EXIF
                date_folder = "Unknown"
                try:
                    img = Image.open(img_path)
                    exif = img._getexif()
                    if exif:
                        for tag_id, value in exif.items():
                            tag = TAGS.get(tag_id, tag_id)
                            if tag == 'DateTimeOriginal':
                                date_folder = value[:7].replace(':', '-')
                                break
                except Exception:
                    pass

                dest_dir = output / date_folder
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / img_path.name

                shutil.copy2(img_path, dest)
                result.organized += 1

                if date_folder not in result.categories:
                    result.categories[date_folder] = 0
                result.categories[date_folder] += 1

            except Exception:
                pass

        result.collections_created = len(result.categories)
        result.success = result.organized > 0

        return result

    def moderate_content(
        self,
        images: Union[str, Path, List[Union[str, Path]]],
        actions: Optional[Dict[str, str]] = None,
    ) -> ModerationResult:
        """
        Moderate content for safety and privacy.

        Features:
        - NSFW detection
        - Face detection
        - PII detection
        - Automatic protective actions

        Args:
            images: Image path(s) or directory
            actions: Action configuration per threat type
        """
        if isinstance(images, (str, Path)):
            images = Path(images)
            if images.is_dir():
                image_list = list(images.rglob("*.jpg")) + list(images.rglob("*.png"))
            else:
                image_list = [images]
        else:
            image_list = [Path(i) for i in images]

        result = ModerationResult(
            total_checked=len(image_list),
            safe=0,
            flagged=0,
            blocked=0,
        )

        try:
            shield = self._get_content_shield()

            if shield:
                for img_path in image_list:
                    try:
                        mod_result = shield.moderate(img_path, apply_actions=True)

                        threat = mod_result.threat_level.value
                        if threat == 'safe':
                            result.safe += 1
                        elif threat in ['low', 'medium']:
                            result.flagged += 1
                            result.threat_summary[threat] = result.threat_summary.get(threat, 0) + 1
                        else:
                            result.blocked += 1
                            result.threat_summary[threat] = result.threat_summary.get(threat, 0) + 1

                        for action in mod_result.actions_taken:
                            result.actions_taken.append(f"{img_path.name}: {action.value}")

                    except Exception:
                        pass

                result.success = True

            else:
                result.success = False
                result.safe = len(image_list)  # Assume safe if no moderation available

        except Exception as e:
            result.success = False

        return result

    def check_brand_compliance(
        self,
        images: Union[str, Path, List[Union[str, Path]]],
        guidelines: Dict[str, Any],
    ) -> ComplianceResult:
        """
        Check images for brand compliance.

        Features:
        - Logo presence verification
        - Color palette compliance
        - Resolution requirements
        - Format validation

        Args:
            images: Image path(s) or directory
            guidelines: Brand guidelines configuration
        """
        if isinstance(images, (str, Path)):
            images = Path(images)
            if images.is_dir():
                image_list = list(images.rglob("*.jpg")) + list(images.rglob("*.png"))
            else:
                image_list = [images]
        else:
            image_list = [Path(i) for i in images]

        result = ComplianceResult(
            total_checked=len(image_list),
            compliant=0,
            warnings=0,
            violations=0,
        )

        try:
            guardian = self._get_brand_guardian()

            if guardian:
                # Set guidelines
                from brand_guardian import BrandGuidelines
                bg = BrandGuidelines(
                    name=guidelines.get('name', 'Brand'),
                    primary_colors=guidelines.get('primary_colors', []),
                    logo_files=guidelines.get('logo_files', []),
                    min_resolution=guidelines.get('min_resolution', (800, 600)),
                )
                guardian.set_guidelines(bg)

                for img_path in image_list:
                    try:
                        comp_result = guardian.check_compliance(img_path)

                        status = comp_result.status.value
                        if status == 'compliant':
                            result.compliant += 1
                        elif status == 'warning':
                            result.warnings += 1
                            result.issues.extend(comp_result.warnings)
                        else:
                            result.violations += 1
                            result.issues.extend(comp_result.issues)

                    except Exception:
                        pass

                result.success = True

            else:
                result.success = False
                result.compliant = len(image_list)  # Assume compliant if no guardian available

        except Exception as e:
            result.success = False

        return result

    # =========================================================================
    # Custom Pipelines
    # =========================================================================

    def create_pipeline(self, name: str) -> PipelineBuilder:
        """
        Create a new pipeline.

        Args:
            name: Pipeline name

        Returns:
            PipelineBuilder for fluent configuration
        """
        return PipelineBuilder(name)

    def register_pipeline(self, pipeline: Pipeline):
        """
        Register a pipeline for later use.

        Args:
            pipeline: Pipeline to register
        """
        self._pipelines[pipeline.name] = pipeline

    def get_pipeline(self, name: str) -> Optional[Pipeline]:
        """Get a registered pipeline by name."""
        return self._pipelines.get(name)

    def list_pipelines(self) -> List[str]:
        """List all registered pipeline names."""
        return list(self._pipelines.keys())

    def run_pipeline(
        self,
        pipeline: Union[str, Pipeline],
        source: Union[str, Path],
    ) -> PipelineResult:
        """
        Execute a pipeline.

        Args:
            pipeline: Pipeline name or Pipeline object
            source: Source path for processing
        """
        import time
        start_time = time.time()

        if isinstance(pipeline, str):
            pipeline = self._pipelines.get(pipeline)
            if not pipeline:
                raise WorkflowError(f"Pipeline not found: {pipeline}")

        source_path = Path(source)

        result = PipelineResult(
            pipeline_name=pipeline.name,
            source=str(source_path),
            status=WorkflowStatus.RUNNING,
            steps_completed=0,
            steps_total=len(pipeline.steps),
        )

        try:
            current_input = source_path

            for step in pipeline.steps:
                step_result = self._execute_step(step, current_input)
                result.step_results.append(step_result)

                if step_result.get('success', False):
                    result.steps_completed += 1
                    if step_result.get('output'):
                        current_input = Path(step_result['output'])
                        result.outputs.append(step_result['output'])
                else:
                    if step.on_error == 'stop':
                        result.error = step_result.get('error', 'Step failed')
                        result.success = False
                        break
                    elif step.on_error == 'skip':
                        continue

            if result.steps_completed == result.steps_total:
                result.success = True
                result.status = WorkflowStatus.COMPLETED
            else:
                result.status = WorkflowStatus.FAILED

        except Exception as e:
            result.success = False
            result.status = WorkflowStatus.FAILED
            result.error = str(e)

        result.duration = time.time() - start_time

        return result

    def _execute_step(self, step: PipelineStep, input_path: Path) -> Dict[str, Any]:
        """Execute a single pipeline step."""
        result = {
            'step_id': step.step_id,
            'operation': step.operation,
            'success': False,
        }

        try:
            if step.step_type == StepType.ORGANIZE:
                org_result = self.smart_organize(
                    input_path,
                    input_path.parent / 'organized',
                    use_ai=step.kwargs.get('use_ai', True),
                )
                result['success'] = org_result.success
                result['output'] = str(input_path.parent / 'organized')

            elif step.step_type == StepType.MODERATE:
                mod_result = self.moderate_content(input_path)
                result['success'] = mod_result.success
                result['blocked'] = mod_result.blocked

            elif step.step_type == StepType.ENHANCE:
                image_core = self._get_image_core()
                if image_core:
                    enhanced = image_core.enhance(input_path)
                    result['success'] = enhanced.success
                    result['output'] = enhanced.output_path

            elif step.step_type == StepType.RESIZE:
                image_core = self._get_image_core()
                if image_core:
                    resized = image_core.resize(
                        input_path,
                        width=step.kwargs.get('width'),
                        height=step.kwargs.get('height'),
                    )
                    result['success'] = resized.success
                    result['output'] = resized.output_path

            elif step.step_type == StepType.CONVERT:
                image_core = self._get_image_core()
                if image_core:
                    converted = image_core.convert(
                        input_path,
                        format=step.kwargs.get('format', 'jpeg'),
                    )
                    result['success'] = converted.success
                    result['output'] = converted.output_path

            elif step.step_type == StepType.WATERMARK:
                export_hub = self._get_export_hub()
                if export_hub:
                    watermarked = export_hub.add_watermark(
                        input_path,
                        text=step.kwargs.get('text', ''),
                    )
                    result['success'] = watermarked.success
                    result['output'] = watermarked.output_path

            elif step.step_type == StepType.EXPORT:
                export_hub = self._get_export_hub()
                if export_hub:
                    exported = export_hub.export(
                        input_path,
                        destination=step.kwargs.get('destination'),
                    )
                    result['success'] = exported.success
                    result['output'] = exported.output_path

            elif step.step_type == StepType.BACKUP:
                media_vault = self._get_media_vault()
                if media_vault:
                    backed_up = media_vault.backup(
                        input_path,
                        location=step.kwargs.get('location'),
                    )
                    result['success'] = backed_up.success

            else:
                # Custom step - just pass through
                result['success'] = True
                result['output'] = str(input_path)

        except Exception as e:
            result['error'] = str(e)

        return result

    def schedule_pipeline(
        self,
        pipeline: Union[str, Pipeline],
        schedule: str,
    ) -> ScheduleResult:
        """
        Schedule a pipeline for recurring execution.

        Args:
            pipeline: Pipeline name or object
            schedule: Cron-like schedule string (e.g., "0 0 * * *" for daily)

        Returns:
            ScheduleResult with schedule ID
        """
        import uuid

        if isinstance(pipeline, str):
            pipeline_name = pipeline
        else:
            pipeline_name = pipeline.name
            self.register_pipeline(pipeline)

        schedule_id = str(uuid.uuid4())[:8]

        # Parse schedule to get next run time (simplified)
        next_run = datetime.now().isoformat()  # Placeholder

        result = ScheduleResult(
            schedule_id=schedule_id,
            pipeline_name=pipeline_name,
            schedule=schedule,
            next_run=next_run,
            active=True,
            success=True,
        )

        self._schedules[schedule_id] = result

        return result

    def cancel_schedule(self, schedule_id: str) -> bool:
        """Cancel a scheduled pipeline."""
        if schedule_id in self._schedules:
            self._schedules[schedule_id].active = False
            return True
        return False

    def list_schedules(self) -> List[ScheduleResult]:
        """List all scheduled pipelines."""
        return list(self._schedules.values())

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def save_workflow_state(self, path: Union[str, Path]):
        """Save workflow state to file."""
        state = {
            'config': self.config.to_dict(),
            'pipelines': {name: p.to_dict() for name, p in self._pipelines.items()},
            'schedules': {sid: s.to_dict() for sid, s in self._schedules.items()},
            'saved_at': datetime.now().isoformat(),
        }

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)

    def load_workflow_state(self, path: Union[str, Path]):
        """Load workflow state from file."""
        load_path = Path(path)

        if not load_path.exists():
            return

        with open(load_path, 'r', encoding='utf-8') as f:
            state = json.load(f)

        # Restore configuration
        if 'config' in state:
            for key, value in state['config'].items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)


# Convenience functions
def create_workflow(config: Optional[Dict[str, Any]] = None) -> WorkflowPro:
    """Create a new workflow instance."""
    cfg = WorkflowConfig(**config) if config else None
    return WorkflowPro(config=cfg)


def run_event_workflow(
    source: Union[str, Path],
    output: Union[str, Path],
) -> WorkflowResult:
    """Quick event workflow execution."""
    return WorkflowPro().event_workflow(source, output)


def run_product_workflow(
    source: Union[str, Path],
    output: Union[str, Path],
) -> WorkflowResult:
    """Quick product workflow execution."""
    return WorkflowPro().product_workflow(source, output)

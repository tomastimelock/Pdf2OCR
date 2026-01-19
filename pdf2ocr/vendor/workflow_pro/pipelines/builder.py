"""Pipeline Builder - Create custom processing pipelines.

Provides a fluent API for building image processing pipelines.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime
from enum import Enum
import json


class StepType(Enum):
    """Pipeline step types."""
    # Input/Output
    LOAD = "load"
    SAVE = "save"
    EXPORT = "export"

    # Processing
    RESIZE = "resize"
    CROP = "crop"
    ROTATE = "rotate"
    CONVERT = "convert"
    ENHANCE = "enhance"
    FILTER = "filter"

    # Analysis
    ANALYZE = "analyze"
    DETECT = "detect"
    CLASSIFY = "classify"
    OCR = "ocr"

    # Automation
    ORGANIZE = "organize"
    MODERATE = "moderate"
    WATERMARK = "watermark"
    BACKUP = "backup"

    # Control flow
    BRANCH = "branch"
    PARALLEL = "parallel"
    CUSTOM = "custom"


@dataclass
class StepCondition:
    """Condition for conditional step execution."""
    field: str
    operator: str  # eq, ne, gt, lt, ge, le, contains, matches
    value: Any

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate condition against context."""
        actual = context.get(self.field)
        if actual is None:
            return False

        if self.operator == 'eq':
            return actual == self.value
        elif self.operator == 'ne':
            return actual != self.value
        elif self.operator == 'gt':
            return actual > self.value
        elif self.operator == 'lt':
            return actual < self.value
        elif self.operator == 'ge':
            return actual >= self.value
        elif self.operator == 'le':
            return actual <= self.value
        elif self.operator == 'contains':
            return self.value in actual
        elif self.operator == 'matches':
            import re
            return bool(re.match(self.value, str(actual)))

        return False


@dataclass
class PipelineStep:
    """A single step in a pipeline."""
    step_id: str
    step_type: StepType
    operation: str
    params: Dict[str, Any] = field(default_factory=dict)
    on_error: str = "stop"  # stop, skip, retry
    condition: Optional[StepCondition] = None
    timeout: Optional[float] = None  # seconds
    retries: int = 0

    def to_dict(self) -> Dict:
        return {
            'step_id': self.step_id,
            'type': self.step_type.value,
            'operation': self.operation,
            'params': self.params,
            'on_error': self.on_error,
            'timeout': self.timeout,
            'retries': self.retries,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PipelineStep':
        """Create step from dictionary."""
        return cls(
            step_id=data['step_id'],
            step_type=StepType(data['type']),
            operation=data['operation'],
            params=data.get('params', {}),
            on_error=data.get('on_error', 'stop'),
            timeout=data.get('timeout'),
            retries=data.get('retries', 0),
        )


@dataclass
class Pipeline:
    """A complete processing pipeline."""
    name: str
    description: str = ""
    version: str = "1.0.0"
    steps: List[PipelineStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'steps': [s.to_dict() for s in self.steps],
            'metadata': self.metadata,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Pipeline':
        """Create pipeline from dictionary."""
        steps = [PipelineStep.from_dict(s) for s in data.get('steps', [])]
        return cls(
            name=data['name'],
            description=data.get('description', ''),
            version=data.get('version', '1.0.0'),
            steps=steps,
            metadata=data.get('metadata', {}),
            created_at=data.get('created_at', ''),
            updated_at=data.get('updated_at', ''),
        )

    def save(self, path: Union[str, Path]):
        """Save pipeline to JSON file."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Pipeline':
        """Load pipeline from JSON file."""
        load_path = Path(path)

        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return cls.from_dict(data)


class PipelineBuilder:
    """Builder for creating custom pipelines.

    Provides a fluent API for building processing pipelines with
    various operations, conditions, and error handling.
    """

    def __init__(self, name: str):
        """
        Initialize pipeline builder.

        Args:
            name: Pipeline name
        """
        self.name = name
        self.description = ""
        self.version = "1.0.0"
        self._steps: List[PipelineStep] = []
        self._step_counter = 0
        self._current_on_error = "stop"
        self._metadata: Dict[str, Any] = {}

    def describe(self, description: str) -> 'PipelineBuilder':
        """Add pipeline description."""
        self.description = description
        return self

    def set_version(self, version: str) -> 'PipelineBuilder':
        """Set pipeline version."""
        self.version = version
        return self

    def set_metadata(self, key: str, value: Any) -> 'PipelineBuilder':
        """Set metadata value."""
        self._metadata[key] = value
        return self

    def _create_step(
        self,
        step_type: StepType,
        operation: str,
        condition: Optional[StepCondition] = None,
        **params,
    ) -> 'PipelineBuilder':
        """Create and add a step."""
        self._step_counter += 1
        step = PipelineStep(
            step_id=f"step_{self._step_counter}",
            step_type=step_type,
            operation=operation,
            params=params,
            on_error=self._current_on_error,
            condition=condition,
        )
        self._steps.append(step)
        return self

    # Input/Output operations

    def load(self, source: str, **params) -> 'PipelineBuilder':
        """Add load step."""
        return self._create_step(StepType.LOAD, "load", source=source, **params)

    def save(self, destination: str, **params) -> 'PipelineBuilder':
        """Add save step."""
        return self._create_step(StepType.SAVE, "save", destination=destination, **params)

    def export(self, destination: str, format: str = "auto", **params) -> 'PipelineBuilder':
        """Add export step."""
        return self._create_step(
            StepType.EXPORT, "export",
            destination=destination, format=format, **params
        )

    # Processing operations

    def resize(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        mode: str = "fit",
        **params,
    ) -> 'PipelineBuilder':
        """Add resize step."""
        return self._create_step(
            StepType.RESIZE, "resize",
            width=width, height=height, mode=mode, **params
        )

    def crop(
        self,
        x: int = 0,
        y: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        mode: str = "absolute",
        **params,
    ) -> 'PipelineBuilder':
        """Add crop step."""
        return self._create_step(
            StepType.CROP, "crop",
            x=x, y=y, width=width, height=height, mode=mode, **params
        )

    def rotate(self, angle: float, expand: bool = True, **params) -> 'PipelineBuilder':
        """Add rotate step."""
        return self._create_step(StepType.ROTATE, "rotate", angle=angle, expand=expand, **params)

    def convert(self, format: str, quality: int = 90, **params) -> 'PipelineBuilder':
        """Add format conversion step."""
        return self._create_step(
            StepType.CONVERT, "convert",
            format=format, quality=quality, **params
        )

    def enhance(self, mode: str = "auto", strength: float = 1.0, **params) -> 'PipelineBuilder':
        """Add enhancement step."""
        return self._create_step(
            StepType.ENHANCE, "enhance",
            mode=mode, strength=strength, **params
        )

    def filter(self, filter_name: str, **params) -> 'PipelineBuilder':
        """Add filter step."""
        return self._create_step(StepType.FILTER, "filter", filter_name=filter_name, **params)

    # Analysis operations

    def analyze(self, analysis_type: str = "full", **params) -> 'PipelineBuilder':
        """Add analysis step."""
        return self._create_step(StepType.ANALYZE, "analyze", analysis_type=analysis_type, **params)

    def detect(self, detection_type: str, **params) -> 'PipelineBuilder':
        """Add detection step (faces, objects, text, etc.)."""
        return self._create_step(StepType.DETECT, "detect", detection_type=detection_type, **params)

    def classify(self, **params) -> 'PipelineBuilder':
        """Add classification step."""
        return self._create_step(StepType.CLASSIFY, "classify", **params)

    def ocr(self, language: str = "eng", **params) -> 'PipelineBuilder':
        """Add OCR step."""
        return self._create_step(StepType.OCR, "ocr", language=language, **params)

    # Automation operations

    def organize(self, use_ai: bool = True, **params) -> 'PipelineBuilder':
        """Add organization step."""
        return self._create_step(StepType.ORGANIZE, "organize", use_ai=use_ai, **params)

    def moderate(self, policy: str = "default", **params) -> 'PipelineBuilder':
        """Add moderation step."""
        return self._create_step(StepType.MODERATE, "moderate", policy=policy, **params)

    def watermark(self, text: str, **params) -> 'PipelineBuilder':
        """Add watermark step."""
        return self._create_step(StepType.WATERMARK, "watermark", text=text, **params)

    def backup(self, location: str, **params) -> 'PipelineBuilder':
        """Add backup step."""
        return self._create_step(StepType.BACKUP, "backup", location=location, **params)

    # Control flow

    def when(self, field: str, operator: str, value: Any) -> 'PipelineBuilder':
        """Add condition for next step."""
        # Store condition for next step
        self._pending_condition = StepCondition(field, operator, value)
        return self

    def branch(
        self,
        condition: StepCondition,
        if_true: 'Pipeline',
        if_false: Optional['Pipeline'] = None,
    ) -> 'PipelineBuilder':
        """Add branch step."""
        return self._create_step(
            StepType.BRANCH, "branch",
            condition=condition,
            if_true=if_true.to_dict() if if_true else None,
            if_false=if_false.to_dict() if if_false else None,
        )

    def parallel(self, pipelines: List['Pipeline']) -> 'PipelineBuilder':
        """Add parallel execution step."""
        return self._create_step(
            StepType.PARALLEL, "parallel",
            pipelines=[p.to_dict() for p in pipelines],
        )

    def custom(self, operation: str, **params) -> 'PipelineBuilder':
        """Add custom step."""
        return self._create_step(StepType.CUSTOM, operation, **params)

    # Error handling

    def on_error(self, action: str) -> 'PipelineBuilder':
        """
        Set error handling for subsequent steps.

        Args:
            action: "stop", "skip", or "retry"
        """
        if action not in ["stop", "skip", "retry"]:
            raise ValueError(f"Invalid error action: {action}")
        self._current_on_error = action
        return self

    def with_retry(self, retries: int = 3) -> 'PipelineBuilder':
        """Configure retry for last step."""
        if self._steps:
            self._steps[-1].retries = retries
        return self

    def with_timeout(self, seconds: float) -> 'PipelineBuilder':
        """Configure timeout for last step."""
        if self._steps:
            self._steps[-1].timeout = seconds
        return self

    # Building

    def build(self) -> Pipeline:
        """Build and return the pipeline."""
        now = datetime.now().isoformat()

        return Pipeline(
            name=self.name,
            description=self.description,
            version=self.version,
            steps=self._steps.copy(),
            metadata=self._metadata.copy(),
            created_at=now,
            updated_at=now,
        )

    def validate(self) -> List[str]:
        """Validate pipeline configuration."""
        errors = []

        if not self.name:
            errors.append("Pipeline name is required")

        if not self._steps:
            errors.append("Pipeline must have at least one step")

        # Check for step dependencies
        step_ids = set()
        for step in self._steps:
            if step.step_id in step_ids:
                errors.append(f"Duplicate step ID: {step.step_id}")
            step_ids.add(step.step_id)

        return errors


# Pre-built pipeline templates
class PipelineTemplates:
    """Pre-built pipeline templates."""

    @staticmethod
    def photo_processing() -> PipelineBuilder:
        """Standard photo processing pipeline."""
        return (
            PipelineBuilder("photo_processing")
            .describe("Standard photo enhancement and optimization")
            .enhance(mode="auto")
            .resize(width=1920, mode="fit")
            .convert(format="jpeg", quality=85)
        )

    @staticmethod
    def web_optimization() -> PipelineBuilder:
        """Web image optimization pipeline."""
        return (
            PipelineBuilder("web_optimization")
            .describe("Optimize images for web delivery")
            .resize(width=1200, mode="fit")
            .convert(format="webp", quality=80)
        )

    @staticmethod
    def print_preparation() -> PipelineBuilder:
        """Print preparation pipeline."""
        return (
            PipelineBuilder("print_preparation")
            .describe("Prepare images for print output")
            .enhance(mode="print")
            .resize(width=4000, mode="fit")
            .convert(format="tiff", quality=100)
        )

    @staticmethod
    def content_moderation() -> PipelineBuilder:
        """Content moderation pipeline."""
        return (
            PipelineBuilder("content_moderation")
            .describe("Moderate content for safety")
            .moderate(policy="strict")
            .detect(detection_type="faces")
            .on_error("skip")
        )

    @staticmethod
    def document_processing() -> PipelineBuilder:
        """Document processing pipeline."""
        return (
            PipelineBuilder("document_processing")
            .describe("Process and archive documents")
            .enhance(mode="document")
            .ocr()
            .organize(use_ai=True)
        )


# Convenience function
def build_pipeline(name: str) -> PipelineBuilder:
    """Create a new pipeline builder."""
    return PipelineBuilder(name)

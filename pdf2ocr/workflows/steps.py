"""Workflow step definitions for PDF2OCR pipeline."""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable


class StepType(Enum):
    """Types of steps in a PDF2OCR workflow."""
    SPLIT_PDF = "split_pdf"
    EXTRACT_IMAGES = "extract_images"
    ENHANCE_IMAGES = "enhance_images"
    OCR = "ocr"
    LLM_ENHANCE = "llm_enhance"
    EXTRACT_TABLES = "extract_tables"
    DETECT_CHARTS = "detect_charts"
    REGENERATE_CHARTS = "regenerate_charts"
    REGENERATE_IMAGES = "regenerate_images"
    STRUCTURE_OUTPUT = "structure_output"
    EXPORT_WORD = "export_word"
    EXPORT_PDF = "export_pdf"
    CUSTOM = "custom"


class ErrorAction(Enum):
    """Actions to take when a step fails."""
    STOP = "stop"  # Stop the workflow
    SKIP = "skip"  # Skip to next step
    RETRY = "retry"  # Retry the step


@dataclass
class WorkflowStep:
    """
    A single step in a PDF2OCR workflow.

    Represents one operation in the processing pipeline.
    """
    step_id: str
    step_type: StepType
    name: str
    description: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    on_error: ErrorAction = ErrorAction.STOP
    max_retries: int = 3
    condition: Optional[Callable[[Dict], bool]] = None
    enabled: bool = True

    def should_run(self, context: Dict[str, Any]) -> bool:
        """Check if this step should run based on condition."""
        if not self.enabled:
            return False
        if self.condition is None:
            return True
        return self.condition(context)


@dataclass
class StepResult:
    """Result of executing a workflow step."""
    step_id: str
    step_type: StepType
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# Step configuration templates
STEP_CONFIGS = {
    StepType.SPLIT_PDF: {
        "dpi": 200,
        "image_format": "jpg",
        "output_subdir": "pages"
    },
    StepType.EXTRACT_IMAGES: {
        "min_size": 50,
        "output_subdir": "images"
    },
    StepType.ENHANCE_IMAGES: {
        "preset": "document",
        "auto_enhance": True,
        "quality_threshold": 0.6
    },
    StepType.OCR: {
        "engine": "auto",
        "quality_threshold": 0.7,
        "output_subdir": "txt"
    },
    StepType.LLM_ENHANCE: {
        "provider": "anthropic",
        "correct_errors": True,
        "extract_structure": False
    },
    StepType.EXTRACT_TABLES: {
        "output_subdir": "json",
        "save_svg": True
    },
    StepType.DETECT_CHARTS: {
        "output_subdir": "svg"
    },
    StepType.REGENERATE_CHARTS: {
        "output_subdir": "svg"
    },
    StepType.REGENERATE_IMAGES: {
        "output_subdir": "regenerated"
    },
    StepType.STRUCTURE_OUTPUT: {
        "output_file": "document.json"
    },
    StepType.EXPORT_WORD: {
        "include_regenerated": True,
        "include_original": False
    },
    StepType.EXPORT_PDF: {
        "include_regenerated": True,
        "include_original": False
    }
}


def create_step(
    step_type: StepType,
    step_id: Optional[str] = None,
    name: Optional[str] = None,
    **config
) -> WorkflowStep:
    """
    Factory function to create a workflow step.

    Args:
        step_type: Type of step
        step_id: Unique ID (auto-generated if not provided)
        name: Human-readable name
        **config: Step-specific configuration

    Returns:
        Configured WorkflowStep
    """
    step_id = step_id or f"{step_type.value}_{id(step_type)}"
    name = name or step_type.value.replace("_", " ").title()

    # Merge with default config
    default_config = STEP_CONFIGS.get(step_type, {}).copy()
    default_config.update(config)

    return WorkflowStep(
        step_id=step_id,
        step_type=step_type,
        name=name,
        config=default_config
    )

"""Workflow Builder - Fluent API for building PDF2OCR workflows."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from pdf2ocr.workflows.steps import (
    WorkflowStep, StepType, ErrorAction, create_step
)


@dataclass
class Workflow:
    """
    A complete PDF2OCR workflow definition.

    Contains ordered steps to process a PDF document.
    """
    name: str
    description: str = ""
    steps: List[WorkflowStep] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """Get a step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_steps_by_type(self, step_type: StepType) -> List[WorkflowStep]:
        """Get all steps of a specific type."""
        return [s for s in self.steps if s.step_type == step_type]


class WorkflowBuilder:
    """
    Fluent builder for creating PDF2OCR workflows.

    Example:
        workflow = (WorkflowBuilder("my_workflow")
            .describe("Process PDF documents")
            .split_pdf(dpi=200)
            .enhance_images(preset="document")
            .ocr(engine="auto")
            .extract_tables()
            .export_word()
            .build()
        )
    """

    def __init__(self, name: str):
        """
        Initialize a new workflow builder.

        Args:
            name: Name of the workflow
        """
        self.name = name
        self.description = ""
        self.steps: List[WorkflowStep] = []
        self.config: Dict[str, Any] = {}
        self._step_counter = 0

    def _next_step_id(self, prefix: str) -> str:
        """Generate unique step ID."""
        self._step_counter += 1
        return f"{prefix}_{self._step_counter}"

    def describe(self, description: str) -> "WorkflowBuilder":
        """Set workflow description."""
        self.description = description
        return self

    def configure(self, **config) -> "WorkflowBuilder":
        """Set global workflow configuration."""
        self.config.update(config)
        return self

    def add_step(self, step: WorkflowStep) -> "WorkflowBuilder":
        """Add a pre-configured step."""
        self.steps.append(step)
        return self

    def split_pdf(
        self,
        dpi: int = 200,
        image_format: str = "jpg",
        **kwargs
    ) -> "WorkflowBuilder":
        """Add PDF splitting step."""
        step = create_step(
            StepType.SPLIT_PDF,
            step_id=self._next_step_id("split"),
            dpi=dpi,
            image_format=image_format,
            **kwargs
        )
        self.steps.append(step)
        return self

    def extract_images(
        self,
        min_size: int = 50,
        **kwargs
    ) -> "WorkflowBuilder":
        """Add embedded image extraction step."""
        step = create_step(
            StepType.EXTRACT_IMAGES,
            step_id=self._next_step_id("extract_img"),
            min_size=min_size,
            **kwargs
        )
        self.steps.append(step)
        return self

    def enhance_images(
        self,
        preset: str = "document",
        auto_enhance: bool = True,
        **kwargs
    ) -> "WorkflowBuilder":
        """Add image enhancement step."""
        step = create_step(
            StepType.ENHANCE_IMAGES,
            step_id=self._next_step_id("enhance"),
            preset=preset,
            auto_enhance=auto_enhance,
            **kwargs
        )
        self.steps.append(step)
        return self

    def ocr(
        self,
        engine: str = "auto",
        quality_threshold: float = 0.7,
        **kwargs
    ) -> "WorkflowBuilder":
        """Add OCR processing step."""
        step = create_step(
            StepType.OCR,
            step_id=self._next_step_id("ocr"),
            engine=engine,
            quality_threshold=quality_threshold,
            **kwargs
        )
        self.steps.append(step)
        return self

    def llm_enhance(
        self,
        provider: str = "anthropic",
        correct_errors: bool = True,
        extract_structure: bool = False,
        **kwargs
    ) -> "WorkflowBuilder":
        """Add LLM text enhancement step."""
        step = create_step(
            StepType.LLM_ENHANCE,
            step_id=self._next_step_id("llm"),
            provider=provider,
            correct_errors=correct_errors,
            extract_structure=extract_structure,
            **kwargs
        )
        self.steps.append(step)
        return self

    def extract_tables(
        self,
        save_svg: bool = True,
        **kwargs
    ) -> "WorkflowBuilder":
        """Add table extraction step."""
        step = create_step(
            StepType.EXTRACT_TABLES,
            step_id=self._next_step_id("tables"),
            save_svg=save_svg,
            **kwargs
        )
        self.steps.append(step)
        return self

    def detect_charts(self, **kwargs) -> "WorkflowBuilder":
        """Add chart detection step."""
        step = create_step(
            StepType.DETECT_CHARTS,
            step_id=self._next_step_id("charts"),
            **kwargs
        )
        self.steps.append(step)
        return self

    def regenerate_charts(self, **kwargs) -> "WorkflowBuilder":
        """Add chart regeneration step."""
        step = create_step(
            StepType.REGENERATE_CHARTS,
            step_id=self._next_step_id("regen_charts"),
            **kwargs
        )
        self.steps.append(step)
        return self

    def regenerate_images(self, **kwargs) -> "WorkflowBuilder":
        """Add image regeneration step."""
        step = create_step(
            StepType.REGENERATE_IMAGES,
            step_id=self._next_step_id("regen_img"),
            **kwargs
        )
        self.steps.append(step)
        return self

    def structure_output(self, **kwargs) -> "WorkflowBuilder":
        """Add structured output generation step."""
        step = create_step(
            StepType.STRUCTURE_OUTPUT,
            step_id=self._next_step_id("structure"),
            **kwargs
        )
        self.steps.append(step)
        return self

    def export_word(
        self,
        include_regenerated: bool = True,
        include_original: bool = False,
        **kwargs
    ) -> "WorkflowBuilder":
        """Add Word export step."""
        step = create_step(
            StepType.EXPORT_WORD,
            step_id=self._next_step_id("word"),
            include_regenerated=include_regenerated,
            include_original=include_original,
            **kwargs
        )
        self.steps.append(step)
        return self

    def export_pdf(
        self,
        include_regenerated: bool = True,
        include_original: bool = False,
        **kwargs
    ) -> "WorkflowBuilder":
        """Add PDF export step."""
        step = create_step(
            StepType.EXPORT_PDF,
            step_id=self._next_step_id("pdf"),
            include_regenerated=include_regenerated,
            include_original=include_original,
            **kwargs
        )
        self.steps.append(step)
        return self

    def on_error(self, action: str) -> "WorkflowBuilder":
        """Set error action for the last step."""
        if self.steps:
            self.steps[-1].on_error = ErrorAction(action)
        return self

    def with_retry(self, max_retries: int) -> "WorkflowBuilder":
        """Set max retries for the last step."""
        if self.steps:
            self.steps[-1].max_retries = max_retries
        return self

    def build(self) -> Workflow:
        """Build and return the workflow."""
        return Workflow(
            name=self.name,
            description=self.description,
            steps=self.steps,
            config=self.config
        )

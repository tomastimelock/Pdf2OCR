"""Workflow orchestration for PDF2OCR."""

from pdf2ocr.workflows.steps import (
    StepType,
    ErrorAction,
    WorkflowStep,
    StepResult,
    create_step,
)
from pdf2ocr.workflows.builder import (
    Workflow,
    WorkflowBuilder,
)
from pdf2ocr.workflows.presets import (
    get_preset,
    list_presets,
    basic_workflow,
    standard_workflow,
    full_workflow,
    tables_only_workflow,
    text_only_workflow,
    high_quality_workflow,
    scanned_document_workflow,
)
from pdf2ocr.workflows.runner import (
    WorkflowRunner,
    WorkflowResult,
)

__all__ = [
    # Steps
    "StepType",
    "ErrorAction",
    "WorkflowStep",
    "StepResult",
    "create_step",
    # Builder
    "Workflow",
    "WorkflowBuilder",
    # Presets
    "get_preset",
    "list_presets",
    "basic_workflow",
    "standard_workflow",
    "full_workflow",
    "tables_only_workflow",
    "text_only_workflow",
    "high_quality_workflow",
    "scanned_document_workflow",
    # Runner
    "WorkflowRunner",
    "WorkflowResult",
]

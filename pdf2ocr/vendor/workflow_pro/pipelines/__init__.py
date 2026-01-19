"""WorkflowPro Pipelines - Custom pipeline building and execution.

Provides tools for creating and running custom processing pipelines.
"""

from .builder import (
    PipelineBuilder,
    PipelineStep,
    Pipeline,
    StepType,
    build_pipeline,
)

from .executor import (
    PipelineExecutor,
    ExecutionResult,
    StepResult,
    execute_pipeline,
)

__all__ = [
    # Builder
    'PipelineBuilder',
    'PipelineStep',
    'Pipeline',
    'StepType',
    'build_pipeline',

    # Executor
    'PipelineExecutor',
    'ExecutionResult',
    'StepResult',
    'execute_pipeline',
]

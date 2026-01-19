"""WorkflowPro - Unified Automated Workflow Engine.

Consolidates: smart_organizer, content_shield, pro_print_workflow,
document_vault, event_photographer, brand_guardian, product_studio,
surveillance_ai

Provides comprehensive workflow automation for image processing pipelines.
"""

from .workflow import (
    WorkflowPro,
    WorkflowResult,
    OrganizeResult,
    ModerationResult,
    ComplianceResult,
    PipelineBuilder,
    Pipeline,
    PipelineResult,
    ScheduleResult,
    WorkflowError,
    WorkflowConfig,
)

from .presets import (
    EventPreset,
    ProductPreset,
    DocumentPreset,
    PrintPreset,
    SurveillancePreset,
)

from .automation import (
    SmartOrganization,
    ContentModeration,
    BrandManagement,
)

from .pipelines import (
    build_pipeline,
    execute_pipeline,
)

__all__ = [
    # Main class
    'WorkflowPro',

    # Result types
    'WorkflowResult',
    'OrganizeResult',
    'ModerationResult',
    'ComplianceResult',
    'PipelineResult',
    'ScheduleResult',

    # Pipeline
    'PipelineBuilder',
    'Pipeline',

    # Configuration
    'WorkflowConfig',
    'WorkflowError',

    # Presets
    'EventPreset',
    'ProductPreset',
    'DocumentPreset',
    'PrintPreset',
    'SurveillancePreset',

    # Automation
    'SmartOrganization',
    'ContentModeration',
    'BrandManagement',

    # Functions
    'build_pipeline',
    'execute_pipeline',
]


# RAG Pipeline (from integration module)
from .rag_pipeline import ProviderFactory, create_provider

__version__ = '1.0.0'

"""WorkflowPro Presets - Pre-configured workflow presets.

Provides ready-to-use workflow configurations for common use cases.
"""

from .event import EventPreset
from .product import ProductPreset
from .document import DocumentPreset
from .print import PrintPreset
from .surveillance import SurveillancePreset

__all__ = [
    'EventPreset',
    'ProductPreset',
    'DocumentPreset',
    'PrintPreset',
    'SurveillancePreset',
]

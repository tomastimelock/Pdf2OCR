"""
Effects submodule for image_core.

Provides image effects and visual modifications:
- Filters (color, artistic, stylization)
- Borders and frames
- Overlays and watermarks
- Background removal
"""

from .filters import FilterEngine, FilterType
from .borders import BorderMaker, BorderStyle
from .overlays import OverlayTool, WatermarkPosition
from .background import BackgroundRemover

__all__ = [
    'FilterEngine',
    'FilterType',
    'BorderMaker',
    'BorderStyle',
    'OverlayTool',
    'WatermarkPosition',
    'BackgroundRemover',
]

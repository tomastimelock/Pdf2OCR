"""
Derivatives submodule for image_core.

Provides image derivative generation:
- Thumbnails
- Avatars
- Contact sheets
"""

from .thumbnails import ThumbnailGenerator, ThumbnailResult
from .avatars import AvatarMaker, AvatarShape, AvatarResult
from .sheets import ContactSheetGenerator, ContactSheetResult

__all__ = [
    'ThumbnailGenerator',
    'ThumbnailResult',
    'AvatarMaker',
    'AvatarShape',
    'AvatarResult',
    'ContactSheetGenerator',
    'ContactSheetResult',
]

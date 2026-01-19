"""
Avatar creation operations.

Provides avatar generation with various shapes and styles.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any
import logging

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


class AvatarShape(Enum):
    """Avatar shape options."""
    CIRCLE = "circle"
    SQUARE = "square"
    ROUNDED = "rounded"
    HEXAGON = "hexagon"
    DIAMOND = "diamond"


@dataclass
class AvatarResult:
    """Result of an avatar creation."""
    success: bool
    original_size: Tuple[int, int]
    avatar_size: int
    shape: str
    output_path: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'success': self.success,
            'original_size': self.original_size,
            'avatar_size': self.avatar_size,
            'shape': self.shape,
            'output_path': self.output_path,
            'error': self.error,
            'metadata': self.metadata,
        }


class AvatarMaker:
    """
    Avatar creation engine.

    Provides avatar generation with various shapes,
    including face detection for automatic cropping.
    """

    def __init__(self, default_size: int = 200):
        """
        Initialize the avatar maker.

        Args:
            default_size: Default avatar size
        """
        self.default_size = default_size

    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """Load image from path or return if already an Image."""
        if isinstance(image, Image.Image):
            return image
        return Image.open(str(image))

    def _detect_face(self, image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        """Detect the primary face in an image."""
        try:
            import cv2
            import numpy as np

            # Convert PIL to OpenCV
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Load face cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)

            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) > 0:
                # Return largest face
                largest = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest
                return (x, y, x + w, y + h)

            return None

        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return None

    def _create_circle_mask(self, size: int) -> Image.Image:
        """Create a circular mask."""
        mask = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([0, 0, size - 1, size - 1], fill=255)
        return mask

    def _create_rounded_mask(self, size: int, radius: int) -> Image.Image:
        """Create a rounded rectangle mask."""
        mask = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
        return mask

    def _create_hexagon_mask(self, size: int) -> Image.Image:
        """Create a hexagon mask."""
        mask = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(mask)

        # Calculate hexagon points
        cx, cy = size // 2, size // 2
        r = size // 2

        import math
        points = []
        for i in range(6):
            angle = math.radians(60 * i - 30)
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            points.append((x, y))

        draw.polygon(points, fill=255)
        return mask

    def _create_diamond_mask(self, size: int) -> Image.Image:
        """Create a diamond mask."""
        mask = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(mask)

        cx, cy = size // 2, size // 2
        points = [
            (cx, 0),
            (size - 1, cy),
            (cx, size - 1),
            (0, cy)
        ]
        draw.polygon(points, fill=255)
        return mask

    def _get_mask(self, shape: AvatarShape, size: int) -> Image.Image:
        """Get mask for the specified shape."""
        if shape == AvatarShape.CIRCLE:
            return self._create_circle_mask(size)
        elif shape == AvatarShape.ROUNDED:
            return self._create_rounded_mask(size, size // 8)
        elif shape == AvatarShape.HEXAGON:
            return self._create_hexagon_mask(size)
        elif shape == AvatarShape.DIAMOND:
            return self._create_diamond_mask(size)
        else:
            return None  # Square, no mask needed

    def create(
        self,
        image: Union[str, Path, Image.Image],
        size: int = None,
        shape: Union[str, AvatarShape] = AvatarShape.CIRCLE,
        auto_crop: bool = True,
        border_width: int = 0,
        border_color: Tuple[int, int, int] = (255, 255, 255),
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[AvatarResult, Tuple[Image.Image, AvatarResult]]:
        """
        Create an avatar from an image.

        Args:
            image: Input image (path or PIL Image)
            size: Avatar size in pixels
            shape: Avatar shape
            auto_crop: Whether to use face detection for cropping
            border_width: Border width in pixels
            border_color: Border color (RGB)
            output: Output path (if None, returns PIL Image)
            quality: JPEG quality (1-100)

        Returns:
            AvatarResult if output path provided, otherwise (Image, AvatarResult)
        """
        try:
            img = self._load_image(image)
            original_size = img.size
            avatar_size = size or self.default_size

            # Parse shape
            if isinstance(shape, str):
                shape = AvatarShape(shape.lower())

            # Determine crop region
            crop_box = None

            if auto_crop:
                face_box = self._detect_face(img)
                if face_box:
                    # Expand face box to include more context
                    x1, y1, x2, y2 = face_box
                    face_w = x2 - x1
                    face_h = y2 - y1
                    face_cx = (x1 + x2) // 2
                    face_cy = (y1 + y2) // 2

                    # Make square and expand
                    side = int(max(face_w, face_h) * 1.5)
                    half_side = side // 2

                    crop_x1 = max(0, face_cx - half_side)
                    crop_y1 = max(0, face_cy - half_side)
                    crop_x2 = min(original_size[0], face_cx + half_side)
                    crop_y2 = min(original_size[1], face_cy + half_side)

                    crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)

            if crop_box is None:
                # Center crop to square
                min_dim = min(original_size)
                left = (original_size[0] - min_dim) // 2
                top = (original_size[1] - min_dim) // 2
                crop_box = (left, top, left + min_dim, top + min_dim)

            # Crop and resize
            cropped = img.crop(crop_box)
            avatar = cropped.resize((avatar_size, avatar_size), Image.Resampling.LANCZOS)

            # Convert to RGBA for masking
            if avatar.mode != 'RGBA':
                avatar = avatar.convert('RGBA')

            # Apply shape mask
            mask = self._get_mask(shape, avatar_size)
            if mask:
                # Create transparent background
                result = Image.new('RGBA', (avatar_size, avatar_size), (0, 0, 0, 0))
                result.paste(avatar, (0, 0), mask)
                avatar = result

            # Add border if requested
            if border_width > 0:
                bordered_size = avatar_size + border_width * 2
                bordered = Image.new('RGBA', (bordered_size, bordered_size), (0, 0, 0, 0))

                # Create border mask
                border_mask = self._get_mask(shape, bordered_size)
                if border_mask:
                    # Fill border area
                    border_fill = Image.new('RGBA', (bordered_size, bordered_size), (*border_color, 255))
                    bordered.paste(border_fill, (0, 0), border_mask)

                # Paste avatar on top
                avatar_with_border_mask = self._get_mask(shape, avatar_size)
                if avatar_with_border_mask:
                    bordered.paste(avatar, (border_width, border_width), avatar_with_border_mask)
                else:
                    bordered.paste(avatar, (border_width, border_width))

                avatar = bordered
                avatar_size = bordered_size

            result = AvatarResult(
                success=True,
                original_size=original_size,
                avatar_size=avatar_size,
                shape=shape.value,
                metadata={
                    'crop_box': crop_box,
                    'face_detected': auto_crop and self._detect_face(img) is not None,
                    'border_width': border_width,
                }
            )

            if output:
                output_path = Path(output)

                # Use PNG for transparency
                ext = output_path.suffix.lower()
                if ext in ['.jpg', '.jpeg']:
                    # Fill transparency with white for JPEG
                    background = Image.new('RGB', avatar.size, (255, 255, 255))
                    background.paste(avatar, mask=avatar.split()[3])
                    avatar = background

                save_kwargs = {}
                if ext in ['.jpg', '.jpeg']:
                    save_kwargs['quality'] = quality
                    save_kwargs['optimize'] = True
                elif ext == '.png':
                    save_kwargs['optimize'] = True
                elif ext == '.webp':
                    save_kwargs['quality'] = quality

                avatar.save(str(output_path), **save_kwargs)
                result.output_path = str(output_path)
                return result
            else:
                return (avatar, result)

        except Exception as e:
            logger.error(f"Avatar creation error: {e}")
            return AvatarResult(
                success=False,
                original_size=(0, 0),
                avatar_size=0,
                shape=str(shape),
                error=str(e)
            )

    def create_circle(
        self,
        image: Union[str, Path, Image.Image],
        size: int = None,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[AvatarResult, Tuple[Image.Image, AvatarResult]]:
        """Create a circular avatar."""
        return self.create(
            image, size=size, shape=AvatarShape.CIRCLE,
            output=output, quality=quality
        )

    def create_square(
        self,
        image: Union[str, Path, Image.Image],
        size: int = None,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[AvatarResult, Tuple[Image.Image, AvatarResult]]:
        """Create a square avatar."""
        return self.create(
            image, size=size, shape=AvatarShape.SQUARE,
            output=output, quality=quality
        )

    def create_rounded(
        self,
        image: Union[str, Path, Image.Image],
        size: int = None,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[AvatarResult, Tuple[Image.Image, AvatarResult]]:
        """Create a rounded square avatar."""
        return self.create(
            image, size=size, shape=AvatarShape.ROUNDED,
            output=output, quality=quality
        )

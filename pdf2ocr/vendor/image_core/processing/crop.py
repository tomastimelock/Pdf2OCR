"""
Image cropping operations.

Provides various cropping modes including smart cropping,
aspect ratio cropping, and face-aware cropping.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any, List
import logging

from PIL import Image

logger = logging.getLogger(__name__)


class CropMode(Enum):
    """Crop mode options."""
    BOX = "box"  # Crop to specified box coordinates
    CENTER = "center"  # Crop from center
    ASPECT = "aspect"  # Crop to aspect ratio
    SMART = "smart"  # Smart crop (content-aware)
    FACE = "face"  # Face-aware crop


@dataclass
class CropResult:
    """Result of a crop operation."""
    success: bool
    original_size: Tuple[int, int]
    new_size: Tuple[int, int]
    crop_box: Tuple[int, int, int, int]
    mode_used: str
    output_path: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'success': self.success,
            'original_size': self.original_size,
            'new_size': self.new_size,
            'crop_box': self.crop_box,
            'mode_used': self.mode_used,
            'output_path': self.output_path,
            'error': self.error,
            'metadata': self.metadata,
        }


class ImageCropper:
    """
    Image cropping engine with multiple crop modes.

    Supports various cropping strategies including smart
    and content-aware cropping.
    """

    # Common aspect ratios
    ASPECT_RATIOS = {
        '1:1': (1, 1),
        '4:3': (4, 3),
        '3:4': (3, 4),
        '16:9': (16, 9),
        '9:16': (9, 16),
        '3:2': (3, 2),
        '2:3': (2, 3),
        '5:4': (5, 4),
        '4:5': (4, 5),
    }

    def __init__(self):
        """Initialize the cropper."""
        self._face_detector = None

    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """Load image from path or return if already an Image."""
        if isinstance(image, Image.Image):
            return image
        return Image.open(str(image))

    def _parse_aspect_ratio(
        self,
        aspect_ratio: Union[str, Tuple[int, int], float]
    ) -> float:
        """Parse aspect ratio to a float value (width/height)."""
        if isinstance(aspect_ratio, str):
            if aspect_ratio in self.ASPECT_RATIOS:
                w, h = self.ASPECT_RATIOS[aspect_ratio]
                return w / h
            elif ':' in aspect_ratio:
                parts = aspect_ratio.split(':')
                return float(parts[0]) / float(parts[1])
            else:
                return float(aspect_ratio)
        elif isinstance(aspect_ratio, tuple):
            return aspect_ratio[0] / aspect_ratio[1]
        else:
            return float(aspect_ratio)

    def _calculate_center_crop_box(
        self,
        original_size: Tuple[int, int],
        crop_size: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """Calculate crop box for center crop."""
        orig_w, orig_h = original_size
        crop_w, crop_h = crop_size

        left = (orig_w - crop_w) // 2
        top = (orig_h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h

        return (left, top, right, bottom)

    def _calculate_aspect_crop_box(
        self,
        original_size: Tuple[int, int],
        target_ratio: float,
        anchor: str = 'center'
    ) -> Tuple[int, int, int, int]:
        """Calculate crop box to achieve target aspect ratio."""
        orig_w, orig_h = original_size
        orig_ratio = orig_w / orig_h

        if orig_ratio > target_ratio:
            # Image is wider than target, crop width
            new_w = int(orig_h * target_ratio)
            new_h = orig_h
        else:
            # Image is taller than target, crop height
            new_w = orig_w
            new_h = int(orig_w / target_ratio)

        # Calculate position based on anchor
        if anchor == 'center':
            left = (orig_w - new_w) // 2
            top = (orig_h - new_h) // 2
        elif anchor == 'top':
            left = (orig_w - new_w) // 2
            top = 0
        elif anchor == 'bottom':
            left = (orig_w - new_w) // 2
            top = orig_h - new_h
        elif anchor == 'left':
            left = 0
            top = (orig_h - new_h) // 2
        elif anchor == 'right':
            left = orig_w - new_w
            top = (orig_h - new_h) // 2
        else:
            left = (orig_w - new_w) // 2
            top = (orig_h - new_h) // 2

        return (left, top, left + new_w, top + new_h)

    def _calculate_smart_crop_box(
        self,
        image: Image.Image,
        target_size: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Calculate smart crop box based on image content.

        Uses edge detection and entropy to find the most
        interesting region.
        """
        try:
            import numpy as np
            from scipy import ndimage

            # Convert to grayscale array
            gray = image.convert('L')
            arr = np.array(gray, dtype=np.float32)

            # Calculate gradient magnitude (edge detection)
            sobel_x = ndimage.sobel(arr, axis=1)
            sobel_y = ndimage.sobel(arr, axis=0)
            gradient = np.sqrt(sobel_x**2 + sobel_y**2)

            # Find region with highest gradient sum (most edges/detail)
            orig_w, orig_h = image.size
            target_w, target_h = target_size

            if target_w >= orig_w and target_h >= orig_h:
                return (0, 0, orig_w, orig_h)

            best_score = -1
            best_box = (0, 0, target_w, target_h)

            # Slide window to find best region
            step = max(1, min(target_w, target_h) // 10)

            for y in range(0, orig_h - target_h + 1, step):
                for x in range(0, orig_w - target_w + 1, step):
                    region = gradient[y:y + target_h, x:x + target_w]
                    score = np.sum(region)

                    if score > best_score:
                        best_score = score
                        best_box = (x, y, x + target_w, y + target_h)

            return best_box

        except ImportError:
            # Fall back to center crop
            logger.warning("scipy not available, falling back to center crop")
            return self._calculate_center_crop_box(image.size, target_size)

    def _detect_faces(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """Detect faces in the image."""
        try:
            import cv2
            import numpy as np

            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Load face cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)

            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            return [(x, y, x + w, y + h) for (x, y, w, h) in faces]

        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return []

    def _calculate_face_crop_box(
        self,
        image: Image.Image,
        target_size: Tuple[int, int],
        padding: float = 0.2
    ) -> Tuple[int, int, int, int]:
        """Calculate crop box centered on detected faces."""
        faces = self._detect_faces(image)

        if not faces:
            # No faces found, use center crop
            return self._calculate_center_crop_box(image.size, target_size)

        # Calculate bounding box of all faces
        all_left = min(f[0] for f in faces)
        all_top = min(f[1] for f in faces)
        all_right = max(f[2] for f in faces)
        all_bottom = max(f[3] for f in faces)

        # Calculate center of faces
        face_center_x = (all_left + all_right) // 2
        face_center_y = (all_top + all_bottom) // 2

        orig_w, orig_h = image.size
        target_w, target_h = target_size

        # Position crop box centered on faces
        left = face_center_x - target_w // 2
        top = face_center_y - target_h // 2

        # Clamp to image bounds
        left = max(0, min(left, orig_w - target_w))
        top = max(0, min(top, orig_h - target_h))

        return (left, top, left + target_w, top + target_h)

    def crop(
        self,
        image: Union[str, Path, Image.Image],
        box: Optional[Tuple[int, int, int, int]] = None,
        aspect_ratio: Optional[Union[str, Tuple[int, int], float]] = None,
        size: Optional[Tuple[int, int]] = None,
        mode: CropMode = CropMode.BOX,
        anchor: str = 'center',
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[CropResult, Tuple[Image.Image, CropResult]]:
        """
        Crop an image.

        Args:
            image: Input image (path or PIL Image)
            box: Crop box (left, top, right, bottom)
            aspect_ratio: Target aspect ratio
            size: Target size for smart/face crop
            mode: Crop mode
            anchor: Anchor point for aspect crop
            output: Output path (if None, returns PIL Image)
            quality: JPEG quality (1-100)

        Returns:
            CropResult if output path provided, otherwise (Image, CropResult)
        """
        try:
            img = self._load_image(image)
            original_size = img.size
            crop_box = None

            if mode == CropMode.BOX and box:
                crop_box = box

            elif mode == CropMode.CENTER and size:
                crop_box = self._calculate_center_crop_box(original_size, size)

            elif mode == CropMode.ASPECT and aspect_ratio:
                target_ratio = self._parse_aspect_ratio(aspect_ratio)
                crop_box = self._calculate_aspect_crop_box(
                    original_size, target_ratio, anchor
                )

            elif mode == CropMode.SMART and size:
                crop_box = self._calculate_smart_crop_box(img, size)

            elif mode == CropMode.FACE and size:
                crop_box = self._calculate_face_crop_box(img, size)

            elif box:
                # Default to box if provided
                crop_box = box

            else:
                # No valid crop parameters
                return CropResult(
                    success=False,
                    original_size=original_size,
                    new_size=original_size,
                    crop_box=(0, 0, original_size[0], original_size[1]),
                    mode_used=mode.value,
                    error="No valid crop parameters provided"
                )

            # Validate crop box
            left, top, right, bottom = crop_box
            left = max(0, left)
            top = max(0, top)
            right = min(original_size[0], right)
            bottom = min(original_size[1], bottom)
            crop_box = (left, top, right, bottom)

            # Perform crop
            result_img = img.crop(crop_box)

            result = CropResult(
                success=True,
                original_size=original_size,
                new_size=result_img.size,
                crop_box=crop_box,
                mode_used=mode.value,
                metadata={'anchor': anchor}
            )

            if output:
                output_path = Path(output)

                save_kwargs = {}
                ext = output_path.suffix.lower()

                if ext in ['.jpg', '.jpeg']:
                    if result_img.mode in ('RGBA', 'P'):
                        result_img = result_img.convert('RGB')
                    save_kwargs['quality'] = quality
                    save_kwargs['optimize'] = True
                elif ext == '.png':
                    save_kwargs['optimize'] = True
                elif ext == '.webp':
                    save_kwargs['quality'] = quality

                result_img.save(str(output_path), **save_kwargs)
                result.output_path = str(output_path)
                return result
            else:
                return (result_img, result)

        except Exception as e:
            logger.error(f"Crop error: {e}")
            return CropResult(
                success=False,
                original_size=(0, 0),
                new_size=(0, 0),
                crop_box=(0, 0, 0, 0),
                mode_used=mode.value if mode else 'unknown',
                error=str(e)
            )

    def crop_to_aspect(
        self,
        image: Union[str, Path, Image.Image],
        aspect_ratio: Union[str, Tuple[int, int], float],
        anchor: str = 'center',
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[CropResult, Tuple[Image.Image, CropResult]]:
        """Convenience method to crop to aspect ratio."""
        return self.crop(
            image, aspect_ratio=aspect_ratio, mode=CropMode.ASPECT,
            anchor=anchor, output=output, quality=quality
        )

    def smart_crop(
        self,
        image: Union[str, Path, Image.Image],
        size: Tuple[int, int],
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[CropResult, Tuple[Image.Image, CropResult]]:
        """Convenience method for smart crop."""
        return self.crop(
            image, size=size, mode=CropMode.SMART,
            output=output, quality=quality
        )

    def face_crop(
        self,
        image: Union[str, Path, Image.Image],
        size: Tuple[int, int],
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[CropResult, Tuple[Image.Image, CropResult]]:
        """Convenience method for face-aware crop."""
        return self.crop(
            image, size=size, mode=CropMode.FACE,
            output=output, quality=quality
        )

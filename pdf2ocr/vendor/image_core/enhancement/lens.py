"""
Lens correction operations.

Provides lens distortion correction, chromatic aberration
removal, and vignette correction.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any, List
import logging

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class LensProfile:
    """Lens correction profile."""
    name: str
    distortion_k1: float = 0.0
    distortion_k2: float = 0.0
    distortion_k3: float = 0.0
    vignette_amount: float = 0.0
    chromatic_aberration_red: float = 0.0
    chromatic_aberration_blue: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'distortion_k1': self.distortion_k1,
            'distortion_k2': self.distortion_k2,
            'distortion_k3': self.distortion_k3,
            'vignette_amount': self.vignette_amount,
            'chromatic_aberration_red': self.chromatic_aberration_red,
            'chromatic_aberration_blue': self.chromatic_aberration_blue,
        }


@dataclass
class LensCorrectionResult:
    """Result of a lens correction operation."""
    success: bool
    original_size: Tuple[int, int]
    new_size: Tuple[int, int]
    corrections_applied: List[str]
    profile_used: Optional[str] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'success': self.success,
            'original_size': self.original_size,
            'new_size': self.new_size,
            'corrections_applied': self.corrections_applied,
            'profile_used': self.profile_used,
            'output_path': self.output_path,
            'error': self.error,
            'metadata': self.metadata,
        }


class LensCorrector:
    """
    Lens correction engine.

    Provides distortion correction, chromatic aberration
    removal, and vignette correction.
    """

    # Common lens profiles
    PROFILES = {
        'wide_angle': LensProfile(
            name='wide_angle',
            distortion_k1=-0.3,
            distortion_k2=0.1,
            vignette_amount=0.2,
            chromatic_aberration_red=0.002,
            chromatic_aberration_blue=-0.002,
        ),
        'fisheye': LensProfile(
            name='fisheye',
            distortion_k1=-0.6,
            distortion_k2=0.2,
            distortion_k3=-0.05,
            vignette_amount=0.3,
        ),
        'telephoto': LensProfile(
            name='telephoto',
            distortion_k1=0.05,
            vignette_amount=0.1,
        ),
        'kit_lens': LensProfile(
            name='kit_lens',
            distortion_k1=-0.15,
            distortion_k2=0.05,
            vignette_amount=0.15,
            chromatic_aberration_red=0.001,
            chromatic_aberration_blue=-0.001,
        ),
        'smartphone': LensProfile(
            name='smartphone',
            distortion_k1=-0.2,
            distortion_k2=0.08,
            vignette_amount=0.1,
        ),
    }

    def __init__(self):
        """Initialize the lens corrector."""
        pass

    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """Load image from path or return if already an Image."""
        if isinstance(image, Image.Image):
            return image
        return Image.open(str(image))

    def _correct_distortion(
        self,
        image: Image.Image,
        k1: float,
        k2: float = 0.0,
        k3: float = 0.0
    ) -> Image.Image:
        """
        Correct radial distortion using Brown-Conrady model.

        Args:
            image: Input image
            k1, k2, k3: Radial distortion coefficients
        """
        try:
            import numpy as np

            arr = np.array(image)
            h, w = arr.shape[:2]

            # Create coordinate grids
            cx, cy = w / 2, h / 2
            x, y = np.meshgrid(np.arange(w), np.arange(h))

            # Normalize coordinates
            x_norm = (x - cx) / cx
            y_norm = (y - cy) / cy

            # Calculate radial distance
            r2 = x_norm ** 2 + y_norm ** 2
            r4 = r2 ** 2
            r6 = r2 ** 3

            # Calculate distortion factor
            factor = 1 + k1 * r2 + k2 * r4 + k3 * r6

            # Apply correction (inverse mapping)
            x_corrected = cx + (x - cx) / factor
            y_corrected = cy + (y - cy) / factor

            # Clip to valid range
            x_corrected = np.clip(x_corrected, 0, w - 1).astype(np.float32)
            y_corrected = np.clip(y_corrected, 0, h - 1).astype(np.float32)

            # Use OpenCV for remapping if available
            try:
                import cv2

                if len(arr.shape) == 3:
                    result = cv2.remap(arr, x_corrected, y_corrected, cv2.INTER_LINEAR)
                else:
                    result = cv2.remap(arr, x_corrected, y_corrected, cv2.INTER_LINEAR)

            except ImportError:
                # Manual bilinear interpolation
                from scipy import ndimage

                if len(arr.shape) == 3:
                    result = np.zeros_like(arr)
                    for c in range(arr.shape[2]):
                        result[:, :, c] = ndimage.map_coordinates(
                            arr[:, :, c],
                            [y_corrected, x_corrected],
                            order=1
                        )
                else:
                    result = ndimage.map_coordinates(
                        arr,
                        [y_corrected, x_corrected],
                        order=1
                    )

            return Image.fromarray(result.astype(np.uint8), mode=image.mode)

        except ImportError:
            logger.warning("numpy/scipy not available, skipping distortion correction")
            return image

    def _correct_vignette(
        self,
        image: Image.Image,
        amount: float
    ) -> Image.Image:
        """
        Correct vignetting.

        Args:
            image: Input image
            amount: Vignette correction amount
        """
        try:
            import numpy as np

            arr = np.array(image, dtype=np.float64)
            h, w = arr.shape[:2]

            # Create radial gradient
            cx, cy = w / 2, h / 2
            x, y = np.meshgrid(np.arange(w), np.arange(h))

            # Calculate normalized radial distance
            max_dist = np.sqrt(cx ** 2 + cy ** 2)
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_dist

            # Calculate correction factor
            # Correction is inverse of vignette falloff
            correction = 1 + amount * (dist ** 2)

            # Apply correction
            if len(arr.shape) == 3:
                for c in range(arr.shape[2]):
                    arr[:, :, c] = arr[:, :, c] * correction
            else:
                arr = arr * correction

            # Clip values
            arr = np.clip(arr, 0, 255).astype(np.uint8)

            return Image.fromarray(arr, mode=image.mode)

        except ImportError:
            logger.warning("numpy not available, skipping vignette correction")
            return image

    def _correct_chromatic_aberration(
        self,
        image: Image.Image,
        red_shift: float,
        blue_shift: float
    ) -> Image.Image:
        """
        Correct chromatic aberration by scaling color channels.

        Args:
            image: Input image (RGB)
            red_shift: Scale factor for red channel
            blue_shift: Scale factor for blue channel
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')

        try:
            import numpy as np
            from scipy import ndimage

            arr = np.array(image)
            h, w = arr.shape[:2]

            # Create coordinate grids
            cx, cy = w / 2, h / 2

            # Correct red channel (scale from center)
            if red_shift != 0:
                x, y = np.meshgrid(np.arange(w), np.arange(h))
                scale = 1 + red_shift
                x_scaled = cx + (x - cx) * scale
                y_scaled = cy + (y - cy) * scale

                arr[:, :, 0] = ndimage.map_coordinates(
                    arr[:, :, 0],
                    [np.clip(y_scaled, 0, h-1), np.clip(x_scaled, 0, w-1)],
                    order=1
                )

            # Correct blue channel
            if blue_shift != 0:
                x, y = np.meshgrid(np.arange(w), np.arange(h))
                scale = 1 + blue_shift
                x_scaled = cx + (x - cx) * scale
                y_scaled = cy + (y - cy) * scale

                arr[:, :, 2] = ndimage.map_coordinates(
                    arr[:, :, 2],
                    [np.clip(y_scaled, 0, h-1), np.clip(x_scaled, 0, w-1)],
                    order=1
                )

            return Image.fromarray(arr, mode='RGB')

        except ImportError:
            logger.warning("numpy/scipy not available, skipping CA correction")
            return image

    def correct(
        self,
        image: Union[str, Path, Image.Image],
        profile: Optional[Union[str, LensProfile]] = None,
        distortion: Optional[float] = None,
        vignette: Optional[float] = None,
        chromatic_aberration: bool = False,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[LensCorrectionResult, Tuple[Image.Image, LensCorrectionResult]]:
        """
        Apply lens corrections.

        Args:
            image: Input image (path or PIL Image)
            profile: Lens profile name or LensProfile object
            distortion: Manual distortion correction (overrides profile)
            vignette: Manual vignette correction (overrides profile)
            chromatic_aberration: Whether to correct CA (uses profile)
            output: Output path (if None, returns PIL Image)
            quality: JPEG quality (1-100)

        Returns:
            LensCorrectionResult if output path provided, otherwise (Image, LensCorrectionResult)
        """
        try:
            img = self._load_image(image)
            original_size = img.size
            corrections = []
            profile_name = None

            # Get profile if specified
            lens_profile = None
            if profile:
                if isinstance(profile, str):
                    lens_profile = self.PROFILES.get(profile)
                    profile_name = profile
                else:
                    lens_profile = profile
                    profile_name = lens_profile.name

            # Apply distortion correction
            k1 = distortion
            k2, k3 = 0.0, 0.0

            if k1 is None and lens_profile:
                k1 = lens_profile.distortion_k1
                k2 = lens_profile.distortion_k2
                k3 = lens_profile.distortion_k3

            if k1 and k1 != 0:
                img = self._correct_distortion(img, k1, k2, k3)
                corrections.append(f'distortion:k1={k1:.3f}')

            # Apply vignette correction
            vig_amount = vignette
            if vig_amount is None and lens_profile:
                vig_amount = lens_profile.vignette_amount

            if vig_amount and vig_amount != 0:
                img = self._correct_vignette(img, vig_amount)
                corrections.append(f'vignette:{vig_amount:.2f}')

            # Apply chromatic aberration correction
            if chromatic_aberration and lens_profile:
                red_shift = lens_profile.chromatic_aberration_red
                blue_shift = lens_profile.chromatic_aberration_blue

                if red_shift != 0 or blue_shift != 0:
                    img = self._correct_chromatic_aberration(img, red_shift, blue_shift)
                    corrections.append(f'chromatic_aberration:r={red_shift:.4f},b={blue_shift:.4f}')

            result = LensCorrectionResult(
                success=True,
                original_size=original_size,
                new_size=img.size,
                corrections_applied=corrections,
                profile_used=profile_name,
            )

            if output:
                output_path = Path(output)

                save_kwargs = {}
                ext = output_path.suffix.lower()

                if ext in ['.jpg', '.jpeg']:
                    if img.mode in ('RGBA', 'P'):
                        img = img.convert('RGB')
                    save_kwargs['quality'] = quality
                    save_kwargs['optimize'] = True
                elif ext == '.png':
                    save_kwargs['optimize'] = True
                elif ext == '.webp':
                    save_kwargs['quality'] = quality

                img.save(str(output_path), **save_kwargs)
                result.output_path = str(output_path)
                return result
            else:
                return (img, result)

        except Exception as e:
            logger.error(f"Lens correction error: {e}")
            return LensCorrectionResult(
                success=False,
                original_size=(0, 0),
                new_size=(0, 0),
                corrections_applied=[],
                error=str(e)
            )

    def auto_correct(
        self,
        image: Union[str, Path, Image.Image],
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[LensCorrectionResult, Tuple[Image.Image, LensCorrectionResult]]:
        """
        Automatically detect and correct lens issues.

        Args:
            image: Input image (path or PIL Image)
            output: Output path (if None, returns PIL Image)
            quality: JPEG quality (1-100)

        Returns:
            LensCorrectionResult if output path provided, otherwise (Image, LensCorrectionResult)
        """
        # Use a generic correction profile
        return self.correct(
            image,
            profile='kit_lens',
            chromatic_aberration=True,
            output=output,
            quality=quality
        )

    def correct_distortion_only(
        self,
        image: Union[str, Path, Image.Image],
        amount: float,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[LensCorrectionResult, Tuple[Image.Image, LensCorrectionResult]]:
        """Convenience method for distortion correction only."""
        return self.correct(
            image,
            distortion=amount,
            output=output,
            quality=quality
        )

    def correct_vignette_only(
        self,
        image: Union[str, Path, Image.Image],
        amount: float,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[LensCorrectionResult, Tuple[Image.Image, LensCorrectionResult]]:
        """Convenience method for vignette correction only."""
        return self.correct(
            image,
            vignette=amount,
            output=output,
            quality=quality
        )

    def list_profiles(self) -> List[str]:
        """Get list of available lens profiles."""
        return list(self.PROFILES.keys())

    def get_profile(self, name: str) -> Optional[LensProfile]:
        """Get a specific lens profile."""
        return self.PROFILES.get(name)

    def add_profile(self, profile: LensProfile) -> None:
        """Add a custom lens profile."""
        self.PROFILES[profile.name] = profile

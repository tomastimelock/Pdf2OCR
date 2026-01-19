"""Document Scanner Module.

Provides document scanning and enhancement capabilities including
perspective correction, deskewing, and image enhancement for scanned documents.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class ScannerError(Exception):
    """Error during document scanning."""
    pass


@dataclass
class ScanConfig:
    """Document scanning configuration."""
    # Enhancement options
    enhance: bool = True
    deskew: bool = True
    remove_shadows: bool = False
    sharpen: bool = True
    denoise: bool = True

    # Output options
    output_dpi: int = 300
    output_format: str = 'png'
    color_mode: str = 'color'  # 'color', 'grayscale', 'binary'

    # Detection options
    auto_crop: bool = True
    margin: int = 10  # Margin in pixels

    def to_dict(self) -> Dict[str, Any]:
        return {
            'enhance': self.enhance,
            'deskew': self.deskew,
            'remove_shadows': self.remove_shadows,
            'sharpen': self.sharpen,
            'denoise': self.denoise,
            'output_dpi': self.output_dpi,
            'color_mode': self.color_mode,
            'auto_crop': self.auto_crop,
        }


@dataclass
class ScannedDocument:
    """Result of document scanning."""
    path: str
    image: np.ndarray
    original_size: Tuple[int, int]
    final_size: Tuple[int, int]
    corners: Optional[List[Tuple[int, int]]] = None
    rotation_angle: float = 0.0
    config: ScanConfig = field(default_factory=ScanConfig)
    processing_time: float = 0.0

    @property
    def width(self) -> int:
        return self.final_size[0]

    @property
    def height(self) -> int:
        return self.final_size[1]

    @property
    def was_rotated(self) -> bool:
        return abs(self.rotation_angle) > 0.5

    @property
    def was_cropped(self) -> bool:
        return self.original_size != self.final_size

    def save(self, output_path: Union[str, Path]) -> str:
        """Save scanned document to file."""
        try:
            from PIL import Image
            img = Image.fromarray(self.image)
            img.save(str(output_path), dpi=(self.config.output_dpi, self.config.output_dpi))
            return str(output_path)
        except ImportError:
            import cv2
            cv2.imwrite(str(output_path), cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
            return str(output_path)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path,
            'original_size': self.original_size,
            'final_size': self.final_size,
            'rotation_angle': self.rotation_angle,
            'was_rotated': self.was_rotated,
            'was_cropped': self.was_cropped,
            'config': self.config.to_dict(),
        }


class DocumentScanner:
    """Document scanning and enhancement."""

    def __init__(self, config: Optional[ScanConfig] = None):
        """
        Initialize document scanner.

        Args:
            config: Scanning configuration
        """
        self.config = config or ScanConfig()

    def _load_image(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load image as numpy array (RGB)."""
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                return np.stack([image, image, image], axis=-1)
            return image

        try:
            from PIL import Image
            img = Image.open(image).convert('RGB')
            return np.array(img)
        except ImportError:
            import cv2
            img = cv2.imread(str(image))
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def scan(
        self,
        image: Union[str, Path, np.ndarray],
        config: Optional[ScanConfig] = None,
    ) -> ScannedDocument:
        """
        Scan and process a document image.

        Args:
            image: Image path or numpy array
            config: Override default configuration

        Returns:
            ScannedDocument with processed image
        """
        import time
        start_time = time.time()

        cfg = config or self.config
        img = self._load_image(image)
        path = str(image) if not isinstance(image, np.ndarray) else "array"

        original_size = (img.shape[1], img.shape[0])
        corners = None
        rotation = 0.0

        # Detect document corners and apply perspective transform
        if cfg.auto_crop:
            img, corners = self._detect_and_crop(img)

        # Deskew
        if cfg.deskew:
            img, rotation = self._deskew(img)

        # Enhance
        if cfg.enhance:
            img = self._enhance(img, cfg)

        # Apply color mode
        if cfg.color_mode == 'grayscale':
            img = self._to_grayscale(img)
        elif cfg.color_mode == 'binary':
            img = self._to_binary(img)

        final_size = (img.shape[1], img.shape[0])
        processing_time = time.time() - start_time

        return ScannedDocument(
            path=path,
            image=img,
            original_size=original_size,
            final_size=final_size,
            corners=corners,
            rotation_angle=rotation,
            config=cfg,
            processing_time=processing_time,
        )

    def _detect_and_crop(
        self,
        img: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[List[Tuple[int, int]]]]:
        """Detect document corners and apply perspective transform."""
        try:
            import cv2

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Apply blur and edge detection
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 75, 200)

            # Dilate edges to close gaps
            kernel = np.ones((5, 5), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # Sort by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            # Find document contour (largest 4-point polygon)
            doc_contour = None
            for contour in contours[:5]:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                if len(approx) == 4:
                    doc_contour = approx
                    break

            if doc_contour is None:
                return img, None

            # Order corners: top-left, top-right, bottom-right, bottom-left
            corners = self._order_corners(doc_contour.reshape(4, 2))

            # Apply perspective transform
            warped = self._four_point_transform(img, corners)

            corner_list = [(int(c[0]), int(c[1])) for c in corners]
            return warped, corner_list

        except ImportError:
            return img, None

    def _order_corners(self, pts: np.ndarray) -> np.ndarray:
        """Order corners: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)

        # Top-left has smallest sum, bottom-right has largest
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Top-right has smallest difference, bottom-left has largest
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def _four_point_transform(
        self,
        img: np.ndarray,
        corners: np.ndarray,
    ) -> np.ndarray:
        """Apply perspective transform using four corners."""
        import cv2

        # Get dimensions of destination rectangle
        tl, tr, br, bl = corners

        width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        max_width = max(int(width_top), int(width_bottom))

        height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        max_height = max(int(height_left), int(height_right))

        # Destination points
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ], dtype=np.float32)

        # Get transform matrix and apply
        M = cv2.getPerspectiveTransform(corners, dst)
        warped = cv2.warpPerspective(img, M, (max_width, max_height))

        return warped

    def _deskew(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """Deskew image by detecting and correcting rotation."""
        try:
            import cv2

            # Convert to grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img

            # Edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            # Detect lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

            if lines is None:
                return img, 0.0

            # Calculate angles
            angles = []
            for line in lines[:20]:  # Only use first 20 lines
                rho, theta = line[0]
                angle = (theta * 180 / np.pi) - 90

                # Only consider near-horizontal/vertical lines
                if abs(angle) < 45:
                    angles.append(angle)

            if not angles:
                return img, 0.0

            # Get median angle
            median_angle = np.median(angles)

            if abs(median_angle) < 0.5:
                return img, 0.0

            # Rotate image
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            return rotated, median_angle

        except ImportError:
            return img, 0.0

    def _enhance(self, img: np.ndarray, config: ScanConfig) -> np.ndarray:
        """Enhance document image."""
        try:
            import cv2

            result = img.copy()

            # Remove shadows
            if config.remove_shadows:
                result = self._remove_shadows(result)

            # Denoise
            if config.denoise:
                if len(result.shape) == 3:
                    result = cv2.fastNlMeansDenoisingColored(result, None, 10, 10, 7, 21)
                else:
                    result = cv2.fastNlMeansDenoising(result, None, 10, 7, 21)

            # Sharpen
            if config.sharpen:
                kernel = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
                result = cv2.filter2D(result, -1, kernel)

            # Increase contrast
            if len(result.shape) == 3:
                lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                lab = cv2.merge([l, a, b])
                result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                result = clahe.apply(result)

            return result

        except ImportError:
            return img

    def _remove_shadows(self, img: np.ndarray) -> np.ndarray:
        """Remove shadows from document image."""
        try:
            import cv2

            if len(img.shape) == 3:
                rgb_planes = cv2.split(img)
                result_planes = []

                for plane in rgb_planes:
                    dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
                    bg = cv2.medianBlur(dilated, 21)
                    diff = 255 - cv2.absdiff(plane, bg)
                    result_planes.append(cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX))

                return cv2.merge(result_planes)
            else:
                dilated = cv2.dilate(img, np.ones((7, 7), np.uint8))
                bg = cv2.medianBlur(dilated, 21)
                diff = 255 - cv2.absdiff(img, bg)
                return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

        except ImportError:
            return img

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Convert to grayscale."""
        if len(img.shape) == 2:
            return img

        try:
            import cv2
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return np.stack([gray, gray, gray], axis=-1)
        except ImportError:
            gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            return np.stack([gray, gray, gray], axis=-1)

    def _to_binary(self, img: np.ndarray) -> np.ndarray:
        """Convert to binary (black and white)."""
        try:
            import cv2

            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img

            # Adaptive thresholding for better results
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            return np.stack([binary, binary, binary], axis=-1)

        except ImportError:
            gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            binary = (gray > 128).astype(np.uint8) * 255
            return np.stack([binary, binary, binary], axis=-1)

    def scan_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        config: Optional[ScanConfig] = None,
    ) -> List[ScannedDocument]:
        """Scan multiple documents."""
        return [self.scan(img, config) for img in images]


# Convenience functions
def scan_document(
    image: Union[str, Path, np.ndarray],
    enhance: bool = True,
    deskew: bool = True,
) -> ScannedDocument:
    """Scan a document image."""
    config = ScanConfig(enhance=enhance, deskew=deskew)
    scanner = DocumentScanner(config)
    return scanner.scan(image)


def enhance_document(
    image: Union[str, Path, np.ndarray],
    remove_shadows: bool = False,
) -> np.ndarray:
    """Enhance a document image."""
    config = ScanConfig(
        enhance=True,
        deskew=False,
        auto_crop=False,
        remove_shadows=remove_shadows,
    )
    scanner = DocumentScanner(config)
    result = scanner.scan(image)
    return result.image

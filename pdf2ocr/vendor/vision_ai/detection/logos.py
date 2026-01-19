"""Logo Detection Module.

Provides logo detection and matching capabilities using template matching,
feature detection, and deep learning approaches.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class LogoDetectionError(Exception):
    """Error during logo detection."""
    pass


@dataclass
class LogoMatch:
    """A detected logo match."""
    name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    template_path: Optional[str] = None
    match_method: str = "template"

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def width(self) -> int:
        return self.bbox[2]

    @property
    def height(self) -> int:
        return self.bbox[3]

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'center': self.center,
            'match_method': self.match_method,
        }


@dataclass
class LogoDetectionResult:
    """Logo detection result."""
    path: str
    logos: List[LogoMatch] = field(default_factory=list)
    processing_time: float = 0.0

    @property
    def count(self) -> int:
        return len(self.logos)

    @property
    def detected_names(self) -> List[str]:
        return list(set(logo.name for logo in self.logos))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path,
            'count': self.count,
            'logos': [logo.to_dict() for logo in self.logos],
            'detected_names': self.detected_names,
        }


@dataclass
class LogoTemplate:
    """A logo template for matching."""
    name: str
    image_path: str
    image: Optional[np.ndarray] = None
    features: Optional[Any] = None
    keypoints: Optional[Any] = None
    descriptors: Optional[np.ndarray] = None
    min_scale: float = 0.5
    max_scale: float = 2.0


class LogoDetector:
    """Logo detection using multiple methods."""

    METHODS = ['template', 'feature', 'deep', 'hybrid']

    def __init__(
        self,
        method: str = 'hybrid',
        templates_dir: Optional[Union[str, Path]] = None,
        min_confidence: float = 0.7,
        scales: Optional[List[float]] = None,
    ):
        """
        Initialize logo detector.

        Args:
            method: Detection method ('template', 'feature', 'deep', 'hybrid')
            templates_dir: Directory containing logo templates
            min_confidence: Minimum match confidence
            scales: List of scales for multi-scale matching
        """
        self.method = method
        self.templates_dir = Path(templates_dir) if templates_dir else None
        self.min_confidence = min_confidence
        self.scales = scales or [0.5, 0.75, 1.0, 1.25, 1.5]

        # Template cache
        self.templates: Dict[str, LogoTemplate] = {}

        # Lazy-loaded components
        self._orb = None
        self._sift = None
        self._matcher = None
        self._deep_model = None

        # Load templates if directory provided
        if self.templates_dir and self.templates_dir.exists():
            self._load_templates()

    def _load_templates(self):
        """Load logo templates from directory."""
        if not self.templates_dir:
            return

        extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

        for img_path in self.templates_dir.iterdir():
            if img_path.suffix.lower() in extensions:
                name = img_path.stem
                self.add_template(name, img_path)

    def add_template(
        self,
        name: str,
        image: Union[str, Path, np.ndarray],
        min_scale: float = 0.5,
        max_scale: float = 2.0,
    ):
        """Add a logo template."""
        if isinstance(image, (str, Path)):
            img = self._load_image(image)
            path = str(image)
        else:
            img = image
            path = ""

        template = LogoTemplate(
            name=name,
            image_path=path,
            image=img,
            min_scale=min_scale,
            max_scale=max_scale,
        )

        # Extract features for feature matching
        if self.method in ('feature', 'hybrid'):
            template.keypoints, template.descriptors = self._extract_features(img)

        self.templates[name] = template

    def _load_image(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load image as numpy array."""
        if isinstance(image, np.ndarray):
            return image

        try:
            from PIL import Image
            img = Image.open(image).convert('RGB')
            return np.array(img)
        except ImportError:
            import cv2
            img = cv2.imread(str(image))
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if len(img.shape) == 2:
            return img
        try:
            import cv2
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        except ImportError:
            # Simple grayscale conversion
            return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    def _get_orb(self):
        """Get ORB feature detector."""
        if self._orb is None:
            try:
                import cv2
                self._orb = cv2.ORB_create(nfeatures=500)
            except ImportError:
                pass
        return self._orb

    def _get_sift(self):
        """Get SIFT feature detector."""
        if self._sift is None:
            try:
                import cv2
                self._sift = cv2.SIFT_create()
            except (ImportError, AttributeError):
                pass
        return self._sift

    def _get_matcher(self):
        """Get feature matcher."""
        if self._matcher is None:
            try:
                import cv2
                self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            except ImportError:
                pass
        return self._matcher

    def _extract_features(self, img: np.ndarray) -> Tuple[Any, Optional[np.ndarray]]:
        """Extract features from image."""
        gray = self._to_gray(img)

        # Try SIFT first
        sift = self._get_sift()
        if sift:
            return sift.detectAndCompute(gray, None)

        # Fall back to ORB
        orb = self._get_orb()
        if orb:
            return orb.detectAndCompute(gray, None)

        return None, None

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        templates: Optional[List[str]] = None,
    ) -> LogoDetectionResult:
        """
        Detect logos in an image.

        Args:
            image: Image to search
            templates: Specific template names to search for (None = all)

        Returns:
            LogoDetectionResult with detected logos
        """
        import time
        start_time = time.time()

        img = self._load_image(image)
        path = str(image) if not isinstance(image, np.ndarray) else "array"

        logos = []

        # Get templates to search
        search_templates = self.templates
        if templates:
            search_templates = {k: v for k, v in self.templates.items() if k in templates}

        for name, template in search_templates.items():
            matches = self._find_logo(img, template)
            logos.extend(matches)

        # Non-maximum suppression
        logos = self._nms(logos)

        processing_time = time.time() - start_time

        return LogoDetectionResult(
            path=path,
            logos=logos,
            processing_time=processing_time,
        )

    def _find_logo(self, image: np.ndarray, template: LogoTemplate) -> List[LogoMatch]:
        """Find logo in image using configured method."""
        matches = []

        if self.method == 'template':
            matches = self._template_match(image, template)
        elif self.method == 'feature':
            matches = self._feature_match(image, template)
        elif self.method == 'hybrid':
            # Try both methods
            matches = self._template_match(image, template)
            if not matches:
                matches = self._feature_match(image, template)
        elif self.method == 'deep':
            matches = self._deep_match(image, template)

        return matches

    def _template_match(self, image: np.ndarray, template: LogoTemplate) -> List[LogoMatch]:
        """Multi-scale template matching."""
        try:
            import cv2
        except ImportError:
            return []

        if template.image is None:
            return []

        matches = []
        gray = self._to_gray(image)
        template_gray = self._to_gray(template.image)
        th, tw = template_gray.shape[:2]

        for scale in self.scales:
            # Resize template
            new_w = int(tw * scale)
            new_h = int(th * scale)

            if new_w < 10 or new_h < 10:
                continue
            if new_w > gray.shape[1] or new_h > gray.shape[0]:
                continue

            resized = cv2.resize(template_gray, (new_w, new_h))

            # Template matching
            result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= self.min_confidence)

            for pt in zip(*locations[::-1]):
                confidence = float(result[pt[1], pt[0]])

                matches.append(LogoMatch(
                    name=template.name,
                    confidence=confidence,
                    bbox=(pt[0], pt[1], new_w, new_h),
                    template_path=template.image_path,
                    match_method='template',
                ))

        return matches

    def _feature_match(self, image: np.ndarray, template: LogoTemplate) -> List[LogoMatch]:
        """Feature-based logo matching."""
        try:
            import cv2
        except ImportError:
            return []

        if template.descriptors is None:
            return []

        # Extract features from target image
        kp, desc = self._extract_features(image)
        if desc is None or len(desc) == 0:
            return []

        # Match features
        matcher = self._get_matcher()
        if matcher is None:
            return []

        try:
            matches = matcher.match(template.descriptors, desc)
        except Exception:
            return []

        if len(matches) < 4:
            return []

        # Sort by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Calculate confidence based on match quality
        good_matches = [m for m in matches if m.distance < 50]
        confidence = len(good_matches) / max(len(template.descriptors), 1)

        if confidence < self.min_confidence:
            return []

        # Get bounding box from matched keypoints
        src_pts = np.float32([template.keypoints[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches])

        if len(dst_pts) == 0:
            return []

        # Calculate bounding box
        x_min = int(np.min(dst_pts[:, 0]))
        y_min = int(np.min(dst_pts[:, 1]))
        x_max = int(np.max(dst_pts[:, 0]))
        y_max = int(np.max(dst_pts[:, 1]))

        return [LogoMatch(
            name=template.name,
            confidence=confidence,
            bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
            template_path=template.image_path,
            match_method='feature',
        )]

    def _deep_match(self, image: np.ndarray, template: LogoTemplate) -> List[LogoMatch]:
        """Deep learning-based logo detection."""
        # Placeholder for deep learning detection
        # Would use models like YOLO trained on logo datasets
        return []

    def _nms(self, logos: List[LogoMatch], iou_threshold: float = 0.5) -> List[LogoMatch]:
        """Non-maximum suppression for overlapping detections."""
        if len(logos) <= 1:
            return logos

        # Sort by confidence
        logos = sorted(logos, key=lambda x: x.confidence, reverse=True)

        keep = []
        for logo in logos:
            should_keep = True
            for kept in keep:
                if self._iou(logo.bbox, kept.bbox) > iou_threshold:
                    should_keep = False
                    break
            if should_keep:
                keep.append(logo)

        return keep

    def _iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def match(
        self,
        image: Union[str, Path, np.ndarray],
        template: Union[str, Path, np.ndarray],
        name: str = "query",
    ) -> List[LogoMatch]:
        """
        Match a single template against an image.

        Args:
            image: Target image
            template: Template image
            name: Name for the template

        Returns:
            List of matches
        """
        # Add template temporarily
        temp_name = f"_temp_{name}"
        self.add_template(temp_name, template)

        # Detect
        result = self.detect(image, templates=[temp_name])

        # Remove temporary template
        del self.templates[temp_name]

        # Rename matches
        for match in result.logos:
            match.name = name

        return result.logos


# Convenience functions
def detect_logos(
    image: Union[str, Path, np.ndarray],
    templates_dir: Optional[Union[str, Path]] = None,
    min_confidence: float = 0.7,
) -> LogoDetectionResult:
    """Detect logos in an image."""
    detector = LogoDetector(templates_dir=templates_dir, min_confidence=min_confidence)
    return detector.detect(image)


def match_logo(
    image: Union[str, Path, np.ndarray],
    template: Union[str, Path, np.ndarray],
    name: str = "logo",
) -> List[LogoMatch]:
    """Match a template logo against an image."""
    detector = LogoDetector()
    return detector.match(image, template, name)

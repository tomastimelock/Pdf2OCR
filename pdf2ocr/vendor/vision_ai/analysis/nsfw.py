"""NSFW Content Detection Module.

Provides content moderation and safety checking capabilities
using machine learning models for detecting inappropriate content.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import numpy as np


class NSFWDetectionError(Exception):
    """Error during NSFW detection."""
    pass


class SafetyCategory(str, Enum):
    """Content safety categories."""
    SAFE = "safe"
    SUGGESTIVE = "suggestive"
    NSFW = "nsfw"
    EXPLICIT = "explicit"
    VIOLENCE = "violence"
    GORE = "gore"
    UNKNOWN = "unknown"


# Category severity levels (higher = more severe)
CATEGORY_SEVERITY = {
    SafetyCategory.SAFE: 0,
    SafetyCategory.SUGGESTIVE: 1,
    SafetyCategory.NSFW: 2,
    SafetyCategory.VIOLENCE: 2,
    SafetyCategory.EXPLICIT: 3,
    SafetyCategory.GORE: 3,
}


@dataclass
class SafetyResult:
    """Content safety analysis result."""
    path: str
    is_safe: bool
    confidence: float
    category: str = "safe"
    scores: Dict[str, float] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    model: str = ""

    @property
    def severity(self) -> int:
        """Get severity level (0-3)."""
        try:
            cat = SafetyCategory(self.category)
            return CATEGORY_SEVERITY.get(cat, 0)
        except ValueError:
            return 0

    @property
    def is_explicit(self) -> bool:
        return self.category in ('explicit', 'nsfw')

    @property
    def is_suggestive(self) -> bool:
        return self.category == 'suggestive'

    @property
    def needs_review(self) -> bool:
        """Content needs human review."""
        return self.severity >= 1 or self.confidence < 0.7

    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path,
            'is_safe': self.is_safe,
            'confidence': self.confidence,
            'category': self.category,
            'severity': self.severity,
            'scores': self.scores,
            'flags': self.flags,
            'needs_review': self.needs_review,
        }


class NSFWDetector:
    """Content safety detection with multiple backend support."""

    BACKENDS = ['nsfw_detector', 'nudenet', 'opennsfw', 'basic']
    CATEGORIES = ['safe', 'suggestive', 'nsfw', 'explicit']

    def __init__(
        self,
        backend: str = 'auto',
        threshold: float = 0.5,
        model_path: Optional[str] = None,
    ):
        """
        Initialize NSFW detector.

        Args:
            backend: Detection backend ('auto', 'nsfw_detector', 'nudenet', 'opennsfw', 'basic')
            threshold: Threshold for unsafe classification
            model_path: Path to custom model file
        """
        self.threshold = threshold
        self.model_path = model_path
        self.backend = backend

        # Lazy-loaded models
        self._nsfw_model = None
        self._nudenet = None
        self._opennsfw = None

        if backend == 'auto':
            self.backend = self._find_best_backend()

    def _find_best_backend(self) -> str:
        """Find the best available backend."""
        # Try nsfw_detector
        try:
            from nsfw_detector import predict
            return 'nsfw_detector'
        except ImportError:
            pass

        # Try NudeNet
        try:
            from nudenet import NudeDetector
            return 'nudenet'
        except ImportError:
            pass

        # Try OpenNSFW
        try:
            import opennsfw2
            return 'opennsfw'
        except ImportError:
            pass

        return 'basic'

    def _get_nsfw_model(self):
        """Get nsfw_detector model."""
        if self._nsfw_model is None:
            try:
                from nsfw_detector import predict
                self._nsfw_model = predict
            except ImportError:
                pass
        return self._nsfw_model

    def _get_nudenet(self):
        """Get NudeNet detector."""
        if self._nudenet is None:
            try:
                from nudenet import NudeDetector
                self._nudenet = NudeDetector()
            except ImportError:
                pass
        return self._nudenet

    def _get_opennsfw(self):
        """Get OpenNSFW model."""
        if self._opennsfw is None:
            try:
                import opennsfw2
                self._opennsfw = opennsfw2
            except ImportError:
                pass
        return self._opennsfw

    def _load_image(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load image as numpy array (RGB)."""
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

    def check(
        self,
        image: Union[str, Path, np.ndarray],
        detailed: bool = False,
    ) -> SafetyResult:
        """
        Check image for NSFW content.

        Args:
            image: Image path or numpy array
            detailed: Return detailed scores

        Returns:
            SafetyResult with safety assessment
        """
        path = str(image) if not isinstance(image, np.ndarray) else "array"

        if self.backend == 'nsfw_detector':
            return self._check_nsfw_detector(image, path, detailed)
        elif self.backend == 'nudenet':
            return self._check_nudenet(image, path, detailed)
        elif self.backend == 'opennsfw':
            return self._check_opennsfw(image, path, detailed)
        else:
            return self._check_basic(image, path)

    def _check_nsfw_detector(
        self,
        image: Union[str, Path, np.ndarray],
        path: str,
        detailed: bool = False,
    ) -> SafetyResult:
        """Check using nsfw_detector."""
        model = self._get_nsfw_model()
        if model is None:
            return self._check_basic(image, path)

        try:
            # nsfw_detector requires file path
            if isinstance(image, np.ndarray):
                from PIL import Image
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                    Image.fromarray(image).save(f.name)
                    result = model.classify(self.model_path or model.make_open_nsfw_model(), f.name)
            else:
                result = model.classify(self.model_path or model.make_open_nsfw_model(), str(image))

            # Parse results
            scores = result.get(path, result.get(list(result.keys())[0], {}))

            nsfw_score = scores.get('porn', 0) + scores.get('sexy', 0) + scores.get('hentai', 0)
            safe_score = scores.get('neutral', 0) + scores.get('drawings', 0)

            is_safe = nsfw_score < self.threshold
            category = 'safe' if is_safe else ('explicit' if nsfw_score > 0.8 else 'nsfw')

            return SafetyResult(
                path=path,
                is_safe=is_safe,
                confidence=max(nsfw_score, safe_score),
                category=category,
                scores=scores if detailed else {},
                model='nsfw_detector',
            )

        except Exception:
            return self._check_basic(image, path)

    def _check_nudenet(
        self,
        image: Union[str, Path, np.ndarray],
        path: str,
        detailed: bool = False,
    ) -> SafetyResult:
        """Check using NudeNet."""
        nudenet = self._get_nudenet()
        if nudenet is None:
            return self._check_basic(image, path)

        try:
            if isinstance(image, np.ndarray):
                from PIL import Image
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                    Image.fromarray(image).save(f.name)
                    detections = nudenet.detect(f.name)
            else:
                detections = nudenet.detect(str(image))

            # Analyze detections
            unsafe_labels = [
                'EXPOSED_BREAST_F', 'EXPOSED_GENITALIA_F', 'EXPOSED_GENITALIA_M',
                'EXPOSED_BUTTOCKS', 'EXPOSED_ANUS', 'EXPOSED_BREAST_M'
            ]
            suggestive_labels = [
                'COVERED_BREAST_F', 'COVERED_GENITALIA_F', 'COVERED_GENITALIA_M',
                'COVERED_BUTTOCKS', 'FEMALE_BREAST_COVERED', 'MALE_BREAST_EXPOSED'
            ]

            flags = []
            max_unsafe_score = 0
            max_suggestive_score = 0

            for det in detections:
                label = det.get('label', det.get('class', ''))
                score = det.get('score', det.get('confidence', 0))

                if label in unsafe_labels:
                    max_unsafe_score = max(max_unsafe_score, score)
                    flags.append(label)
                elif label in suggestive_labels:
                    max_suggestive_score = max(max_suggestive_score, score)

            # Determine category
            if max_unsafe_score >= self.threshold:
                category = 'explicit' if max_unsafe_score > 0.8 else 'nsfw'
                is_safe = False
                confidence = max_unsafe_score
            elif max_suggestive_score >= self.threshold:
                category = 'suggestive'
                is_safe = True
                confidence = max_suggestive_score
            else:
                category = 'safe'
                is_safe = True
                confidence = 1 - max(max_unsafe_score, max_suggestive_score)

            scores = {
                'unsafe': max_unsafe_score,
                'suggestive': max_suggestive_score,
                'safe': 1 - max(max_unsafe_score, max_suggestive_score),
            }

            return SafetyResult(
                path=path,
                is_safe=is_safe,
                confidence=confidence,
                category=category,
                scores=scores if detailed else {},
                flags=list(set(flags)),
                model='nudenet',
            )

        except Exception:
            return self._check_basic(image, path)

    def _check_opennsfw(
        self,
        image: Union[str, Path, np.ndarray],
        path: str,
        detailed: bool = False,
    ) -> SafetyResult:
        """Check using OpenNSFW2."""
        opennsfw = self._get_opennsfw()
        if opennsfw is None:
            return self._check_basic(image, path)

        try:
            img = self._load_image(image)
            from PIL import Image
            pil_img = Image.fromarray(img)

            nsfw_prob = opennsfw.predict_image(pil_img)

            is_safe = nsfw_prob < self.threshold
            if nsfw_prob > 0.8:
                category = 'explicit'
            elif nsfw_prob > self.threshold:
                category = 'nsfw'
            elif nsfw_prob > 0.3:
                category = 'suggestive'
            else:
                category = 'safe'

            return SafetyResult(
                path=path,
                is_safe=is_safe,
                confidence=max(nsfw_prob, 1 - nsfw_prob),
                category=category,
                scores={'nsfw': nsfw_prob, 'safe': 1 - nsfw_prob} if detailed else {},
                model='opennsfw2',
            )

        except Exception:
            return self._check_basic(image, path)

    def _check_basic(
        self,
        image: Union[str, Path, np.ndarray],
        path: str,
    ) -> SafetyResult:
        """Basic NSFW check using skin tone heuristics (very rough)."""
        img = self._load_image(image)
        pixels = img.reshape(-1, 3)
        total = len(pixels)

        # Very rough skin tone detection
        skin_count = 0
        for r, g, b in pixels[::max(1, total // 1000)]:  # Sample
            # Rough skin tone detection
            if (r > 95 and g > 40 and b > 20 and
                max(r, g, b) - min(r, g, b) > 15 and
                abs(r - g) > 15 and r > g and r > b):
                skin_count += 1

        skin_ratio = skin_count / min(total, 1000)

        # Very conservative - just flag for review if high skin ratio
        nsfw_score = min(0.4, skin_ratio * 0.5)  # Cap at 0.4 for basic detector

        return SafetyResult(
            path=path,
            is_safe=True,  # Basic detector always says safe but suggests review
            confidence=0.5,  # Low confidence
            category='safe',
            scores={'nsfw_estimate': nsfw_score},
            flags=['basic_detector', 'needs_review'],
            model='basic_heuristic',
        )

    def check_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        detailed: bool = False,
    ) -> List[SafetyResult]:
        """Check multiple images."""
        return [self.check(img, detailed) for img in images]

    def check_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = False,
        extensions: Optional[List[str]] = None,
    ) -> List[SafetyResult]:
        """Check all images in a directory."""
        directory = Path(directory)
        ext_set = set(extensions or ['.jpg', '.jpeg', '.png', '.webp'])

        results = []
        pattern = '**/*' if recursive else '*'

        for img_path in directory.glob(pattern):
            if img_path.suffix.lower() in ext_set:
                try:
                    result = self.check(img_path)
                    results.append(result)
                except Exception:
                    pass

        return results

    def get_unsafe_images(
        self,
        directory: Union[str, Path],
        recursive: bool = False,
    ) -> List[str]:
        """Get list of unsafe image paths in directory."""
        results = self.check_directory(directory, recursive)
        return [r.path for r in results if not r.is_safe]


# Convenience functions
def check_nsfw(
    image: Union[str, Path, np.ndarray],
    threshold: float = 0.5,
) -> SafetyResult:
    """Check image for NSFW content."""
    detector = NSFWDetector(threshold=threshold)
    return detector.check(image)


def is_safe(
    image: Union[str, Path, np.ndarray],
    threshold: float = 0.5,
) -> bool:
    """Quick check if image is safe."""
    result = check_nsfw(image, threshold)
    return result.is_safe


def moderate_content(
    images: List[Union[str, Path, np.ndarray]],
    threshold: float = 0.5,
) -> Dict[str, List[str]]:
    """Moderate multiple images, return categorized paths."""
    detector = NSFWDetector(threshold=threshold)
    results = detector.check_batch(images)

    categorized = {
        'safe': [],
        'suggestive': [],
        'nsfw': [],
        'explicit': [],
    }

    for result in results:
        category = result.category
        if category in categorized:
            categorized[category].append(result.path)
        else:
            categorized['safe'].append(result.path)

    return categorized

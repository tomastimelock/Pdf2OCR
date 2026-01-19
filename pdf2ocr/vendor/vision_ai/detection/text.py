"""Text Detection Module.

Provides text region detection capabilities for OCR preprocessing,
using multiple detection backends (EAST, CRAFT, OpenCV).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class TextDetectionError(Exception):
    """Error during text detection."""
    pass


@dataclass
class TextRegion:
    """A detected text region in an image."""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float = 0.0
    text: str = ""  # Populated if OCR is applied
    polygon: List[Tuple[int, int]] = field(default_factory=list)
    orientation: float = 0.0  # Angle in degrees
    language: Optional[str] = None

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

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'center': self.center,
            'area': self.area,
            'orientation': self.orientation,
        }
        if self.text:
            result['text'] = self.text
        if self.polygon:
            result['polygon'] = self.polygon
        return result


@dataclass
class TextDetectionResult:
    """Text detection result for an image."""
    path: str
    regions: List[TextRegion] = field(default_factory=list)
    full_text: str = ""
    processing_time: float = 0.0
    detector: str = ""

    @property
    def count(self) -> int:
        return len(self.regions)

    @property
    def has_text(self) -> bool:
        return self.count > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path,
            'count': self.count,
            'has_text': self.has_text,
            'regions': [r.to_dict() for r in self.regions],
            'full_text': self.full_text,
            'detector': self.detector,
        }


class TextDetector:
    """Text region detection with multiple backend support."""

    BACKENDS = ['east', 'craft', 'easyocr', 'opencv', 'basic']

    def __init__(
        self,
        backend: str = 'auto',
        min_confidence: float = 0.5,
        min_size: int = 10,
        model_path: Optional[str] = None,
    ):
        """
        Initialize text detector.

        Args:
            backend: Detection backend ('auto', 'east', 'craft', 'easyocr', 'opencv', 'basic')
            min_confidence: Minimum detection confidence
            min_size: Minimum text region size in pixels
            model_path: Path to custom model file
        """
        self.min_confidence = min_confidence
        self.min_size = min_size
        self.model_path = model_path
        self.backend = backend

        # Lazy-loaded components
        self._east_net = None
        self._craft_model = None
        self._easyocr_reader = None

        if backend == 'auto':
            self.backend = self._find_best_backend()

    def _find_best_backend(self) -> str:
        """Find the best available backend."""
        # Try EasyOCR (includes detection)
        try:
            import easyocr
            return 'easyocr'
        except ImportError:
            pass

        # Try OpenCV with EAST
        try:
            import cv2
            # Check if EAST model exists
            east_path = self.model_path or 'frozen_east_text_detection.pb'
            if Path(east_path).exists():
                return 'east'
            return 'opencv'
        except ImportError:
            pass

        return 'basic'

    def _get_easyocr_reader(self):
        """Get EasyOCR reader."""
        if self._easyocr_reader is None:
            try:
                import easyocr
                self._easyocr_reader = easyocr.Reader(['en'], gpu=False)
            except ImportError:
                pass
        return self._easyocr_reader

    def _get_east_net(self):
        """Get EAST text detection network."""
        if self._east_net is None:
            try:
                import cv2
                model_path = self.model_path or 'frozen_east_text_detection.pb'
                if Path(model_path).exists():
                    self._east_net = cv2.dnn.readNet(model_path)
            except (ImportError, Exception):
                pass
        return self._east_net

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

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        with_text: bool = False,
    ) -> TextDetectionResult:
        """
        Detect text regions in an image.

        Args:
            image: Image path or numpy array
            with_text: Also extract text content (OCR)

        Returns:
            TextDetectionResult with detected regions
        """
        import time
        start_time = time.time()

        img = self._load_image(image)
        path = str(image) if not isinstance(image, np.ndarray) else "array"

        if self.backend == 'easyocr':
            regions = self._detect_easyocr(img, with_text)
            detector_name = 'easyocr'
        elif self.backend == 'east':
            regions = self._detect_east(img)
            detector_name = 'east'
        elif self.backend == 'opencv':
            regions = self._detect_opencv(img)
            detector_name = 'opencv_mser'
        else:
            regions = self._detect_basic(img)
            detector_name = 'basic'

        # Filter by size
        regions = [r for r in regions if r.width >= self.min_size and r.height >= self.min_size]

        # Build full text
        full_text = ' '.join(r.text for r in regions if r.text)

        processing_time = time.time() - start_time

        return TextDetectionResult(
            path=path,
            regions=regions,
            full_text=full_text,
            processing_time=processing_time,
            detector=detector_name,
        )

    def _detect_easyocr(self, img: np.ndarray, with_text: bool = False) -> List[TextRegion]:
        """Detect text using EasyOCR."""
        reader = self._get_easyocr_reader()
        if reader is None:
            return self._detect_basic(img)

        # Run detection
        results = reader.readtext(img)
        regions = []

        for bbox, text, confidence in results:
            if confidence < self.min_confidence:
                continue

            # Convert polygon to bbox
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x = int(min(xs))
            y = int(min(ys))
            w = int(max(xs) - x)
            h = int(max(ys) - y)

            regions.append(TextRegion(
                bbox=(x, y, w, h),
                confidence=confidence,
                text=text if with_text else "",
                polygon=[(int(p[0]), int(p[1])) for p in bbox],
            ))

        return regions

    def _detect_east(self, img: np.ndarray) -> List[TextRegion]:
        """Detect text using EAST detector."""
        net = self._get_east_net()
        if net is None:
            return self._detect_opencv(img)

        import cv2

        h, w = img.shape[:2]

        # EAST requires dimensions to be multiples of 32
        new_h = (h // 32) * 32
        new_w = (w // 32) * 32

        ratio_h = h / new_h
        ratio_w = w / new_w

        # Resize and prepare blob
        resized = cv2.resize(img, (new_w, new_h))
        blob = cv2.dnn.blobFromImage(resized, 1.0, (new_w, new_h), (123.68, 116.78, 103.94), True, False)

        # Run detection
        net.setInput(blob)
        output_layers = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']
        scores, geometry = net.forward(output_layers)

        # Decode predictions
        regions = self._decode_east(scores, geometry, self.min_confidence, ratio_w, ratio_h)

        return regions

    def _decode_east(
        self,
        scores: np.ndarray,
        geometry: np.ndarray,
        min_confidence: float,
        ratio_w: float,
        ratio_h: float,
    ) -> List[TextRegion]:
        """Decode EAST detector output."""
        import cv2

        num_rows, num_cols = scores.shape[2:4]
        rects = []
        confidences = []

        for y in range(num_rows):
            scores_data = scores[0, 0, y]
            x_data0 = geometry[0, 0, y]
            x_data1 = geometry[0, 1, y]
            x_data2 = geometry[0, 2, y]
            x_data3 = geometry[0, 3, y]
            angles_data = geometry[0, 4, y]

            for x in range(num_cols):
                score = scores_data[x]

                if score < min_confidence:
                    continue

                offset_x = x * 4.0
                offset_y = y * 4.0

                angle = angles_data[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                h = x_data0[x] + x_data2[x]
                w = x_data1[x] + x_data3[x]

                end_x = int(offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
                end_y = int(offset_y - (sin * x_data1[x]) + (cos * x_data2[x]))
                start_x = int(end_x - w)
                start_y = int(end_y - h)

                rects.append((start_x, start_y, end_x, end_y))
                confidences.append(float(score))

        # Apply NMS
        if len(rects) == 0:
            return []

        boxes = np.array(rects)
        confidences_array = np.array(confidences)

        indices = cv2.dnn.NMSBoxesRotated(
            [(x, y, w - x, h - y, 0) for x, y, w, h in rects],
            confidences,
            min_confidence,
            0.4
        )

        regions = []
        for i in indices.flatten() if len(indices) > 0 else []:
            x1, y1, x2, y2 = rects[i]

            # Scale back to original size
            x1 = int(x1 * ratio_w)
            y1 = int(y1 * ratio_h)
            x2 = int(x2 * ratio_w)
            y2 = int(y2 * ratio_h)

            regions.append(TextRegion(
                bbox=(x1, y1, x2 - x1, y2 - y1),
                confidence=confidences[i],
            ))

        return regions

    def _detect_opencv(self, img: np.ndarray) -> List[TextRegion]:
        """Detect text using OpenCV MSER."""
        try:
            import cv2
        except ImportError:
            return self._detect_basic(img)

        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        # Create MSER detector
        mser = cv2.MSER_create()
        mser.setMinArea(100)
        mser.setMaxArea(5000)

        # Detect regions
        regions_list, _ = mser.detectRegions(gray)

        # Convert to TextRegions
        regions = []
        for region in regions_list:
            x, y, w, h = cv2.boundingRect(region)

            if w < self.min_size or h < self.min_size:
                continue

            # Calculate aspect ratio (text regions are usually wider than tall)
            aspect_ratio = w / max(h, 1)
            if aspect_ratio < 0.1 or aspect_ratio > 20:
                continue

            regions.append(TextRegion(
                bbox=(x, y, w, h),
                confidence=0.7,  # MSER doesn't provide confidence
            ))

        # Merge overlapping regions
        regions = self._merge_regions(regions)

        return regions

    def _detect_basic(self, img: np.ndarray) -> List[TextRegion]:
        """Basic text detection fallback using edge detection."""
        try:
            import cv2
        except ImportError:
            return []

        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Dilate to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w < self.min_size or h < self.min_size:
                continue

            # Filter by aspect ratio
            aspect_ratio = w / max(h, 1)
            if aspect_ratio < 0.5:
                continue

            regions.append(TextRegion(
                bbox=(x, y, w, h),
                confidence=0.5,
            ))

        return regions

    def _merge_regions(self, regions: List[TextRegion], overlap_threshold: float = 0.5) -> List[TextRegion]:
        """Merge overlapping text regions."""
        if len(regions) <= 1:
            return regions

        # Sort by y, then x
        regions = sorted(regions, key=lambda r: (r.y, r.x))

        merged = []
        used = set()

        for i, r1 in enumerate(regions):
            if i in used:
                continue

            current_bbox = list(r1.bbox)
            current_conf = r1.confidence

            for j, r2 in enumerate(regions[i+1:], start=i+1):
                if j in used:
                    continue

                # Check overlap
                iou = self._calculate_iou(tuple(current_bbox), r2.bbox)
                if iou > overlap_threshold or self._is_adjacent(tuple(current_bbox), r2.bbox):
                    # Merge
                    x1 = min(current_bbox[0], r2.x)
                    y1 = min(current_bbox[1], r2.y)
                    x2 = max(current_bbox[0] + current_bbox[2], r2.x + r2.width)
                    y2 = max(current_bbox[1] + current_bbox[3], r2.y + r2.height)
                    current_bbox = [x1, y1, x2 - x1, y2 - y1]
                    current_conf = max(current_conf, r2.confidence)
                    used.add(j)

            merged.append(TextRegion(
                bbox=tuple(current_bbox),
                confidence=current_conf,
            ))
            used.add(i)

        return merged

    def _calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
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

    def _is_adjacent(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int], threshold: int = 20) -> bool:
        """Check if two boxes are adjacent (close enough to merge)."""
        # Check horizontal adjacency
        h_gap = max(box2[0] - (box1[0] + box1[2]), box1[0] - (box2[0] + box2[2]))
        # Check vertical overlap
        v_overlap = min(box1[1] + box1[3], box2[1] + box2[3]) - max(box1[1], box2[1])

        return h_gap < threshold and h_gap > -threshold and v_overlap > 0


# Convenience functions
def detect_text(
    image: Union[str, Path, np.ndarray],
    with_content: bool = False,
) -> TextDetectionResult:
    """Detect text regions in an image."""
    detector = TextDetector()
    return detector.detect(image, with_text=with_content)


def find_text_regions(
    image: Union[str, Path, np.ndarray],
    min_confidence: float = 0.5,
) -> List[TextRegion]:
    """Find text regions in an image."""
    detector = TextDetector(min_confidence=min_confidence)
    result = detector.detect(image)
    return result.regions

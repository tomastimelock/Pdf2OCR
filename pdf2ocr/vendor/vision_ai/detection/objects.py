"""Object Detection Module.

Provides object detection capabilities with multiple backend support
(YOLO, TensorFlow, PyTorch, OpenCV DNN).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class ObjectDetectionError(Exception):
    """Error during object detection."""
    pass


# COCO class names for common models
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


@dataclass
class DetectedObject:
    """A detected object in an image."""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    class_id: int = -1
    mask: Optional[np.ndarray] = None
    keypoints: List[Tuple[int, int]] = field(default_factory=list)

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

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    def to_dict(self) -> Dict[str, Any]:
        return {
            'label': self.label,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'center': self.center,
            'area': self.area,
            'class_id': self.class_id,
        }

    def iou(self, other: 'DetectedObject') -> float:
        """Calculate Intersection over Union with another object."""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0


@dataclass
class ObjectDetectionResult:
    """Object detection result for an image."""
    path: str
    objects: List[DetectedObject] = field(default_factory=list)
    inference_time: float = 0.0
    model: str = ""

    @property
    def count(self) -> int:
        return len(self.objects)

    @property
    def labels(self) -> List[str]:
        return list(set(obj.label for obj in self.objects))

    def filter_by_label(self, label: str) -> List[DetectedObject]:
        """Get objects with specific label."""
        return [obj for obj in self.objects if obj.label.lower() == label.lower()]

    def filter_by_confidence(self, min_confidence: float) -> List[DetectedObject]:
        """Get objects above confidence threshold."""
        return [obj for obj in self.objects if obj.confidence >= min_confidence]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path,
            'count': self.count,
            'labels': self.labels,
            'objects': [obj.to_dict() for obj in self.objects],
            'inference_time': self.inference_time,
            'model': self.model,
        }


class ObjectDetector:
    """Object detection with multiple backend support."""

    BACKENDS = ['yolov8', 'yolov5', 'tensorflow', 'opencv', 'basic']

    def __init__(
        self,
        backend: str = 'auto',
        model_path: Optional[str] = None,
        min_confidence: float = 0.5,
        nms_threshold: float = 0.4,
        classes: Optional[List[str]] = None,
    ):
        """
        Initialize object detector.

        Args:
            backend: Detection backend ('auto', 'yolov8', 'yolov5', 'tensorflow', 'opencv', 'basic')
            model_path: Path to custom model file
            min_confidence: Minimum detection confidence
            nms_threshold: Non-maximum suppression threshold
            classes: Filter to specific class names
        """
        self.model_path = model_path
        self.min_confidence = min_confidence
        self.nms_threshold = nms_threshold
        self.filter_classes = classes
        self.backend = backend

        # Lazy-loaded models
        self._yolo_model = None
        self._tf_model = None
        self._opencv_net = None

        # Determine best available backend
        if backend == 'auto':
            self.backend = self._find_best_backend()

    def _find_best_backend(self) -> str:
        """Find the best available backend."""
        # Try YOLOv8 (ultralytics)
        try:
            from ultralytics import YOLO
            return 'yolov8'
        except ImportError:
            pass

        # Try YOLOv5
        try:
            import torch
            return 'yolov5'
        except ImportError:
            pass

        # Try TensorFlow
        try:
            import tensorflow as tf
            return 'tensorflow'
        except ImportError:
            pass

        # Try OpenCV DNN
        try:
            import cv2
            return 'opencv'
        except ImportError:
            pass

        return 'basic'

    def _get_yolo_model(self):
        """Get YOLOv8 model."""
        if self._yolo_model is None:
            try:
                from ultralytics import YOLO
                model_path = self.model_path or 'yolov8n.pt'
                self._yolo_model = YOLO(model_path)
            except (ImportError, Exception):
                pass
        return self._yolo_model

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

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        classes: Optional[List[str]] = None,
    ) -> ObjectDetectionResult:
        """
        Detect objects in an image.

        Args:
            image: Image path or numpy array
            classes: Filter to specific classes (overrides init setting)

        Returns:
            ObjectDetectionResult with detected objects
        """
        import time
        start_time = time.time()

        img = self._load_image(image)
        path = str(image) if not isinstance(image, np.ndarray) else "array"

        filter_classes = classes or self.filter_classes

        if self.backend == 'yolov8':
            objects = self._detect_yolov8(img, filter_classes)
            model_name = 'yolov8'
        elif self.backend == 'yolov5':
            objects = self._detect_yolov5(img, filter_classes)
            model_name = 'yolov5'
        elif self.backend == 'tensorflow':
            objects = self._detect_tensorflow(img, filter_classes)
            model_name = 'tensorflow'
        elif self.backend == 'opencv':
            objects = self._detect_opencv(img, filter_classes)
            model_name = 'opencv_dnn'
        else:
            objects = self._detect_basic(img, filter_classes)
            model_name = 'basic'

        inference_time = time.time() - start_time

        return ObjectDetectionResult(
            path=path,
            objects=objects,
            inference_time=inference_time,
            model=model_name,
        )

    def _detect_yolov8(
        self,
        img: np.ndarray,
        filter_classes: Optional[List[str]] = None,
    ) -> List[DetectedObject]:
        """Detect objects using YOLOv8."""
        model = self._get_yolo_model()
        if model is None:
            return self._detect_basic(img, filter_classes)

        results = model(img, conf=self.min_confidence, verbose=False)
        objects = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i])
                confidence = float(boxes.conf[i])
                label = model.names[class_id]

                if filter_classes and label.lower() not in [c.lower() for c in filter_classes]:
                    continue

                # Get bbox (xyxy format to xywh)
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

                objects.append(DetectedObject(
                    label=label,
                    confidence=confidence,
                    bbox=bbox,
                    class_id=class_id,
                ))

        return objects

    def _detect_yolov5(
        self,
        img: np.ndarray,
        filter_classes: Optional[List[str]] = None,
    ) -> List[DetectedObject]:
        """Detect objects using YOLOv5."""
        try:
            import torch

            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            model.conf = self.min_confidence

            results = model(img)
            detections = results.xyxy[0].cpu().numpy()

            objects = []
            for det in detections:
                x1, y1, x2, y2, conf, class_id = det
                class_id = int(class_id)
                label = model.names[class_id]

                if filter_classes and label.lower() not in [c.lower() for c in filter_classes]:
                    continue

                bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

                objects.append(DetectedObject(
                    label=label,
                    confidence=float(conf),
                    bbox=bbox,
                    class_id=class_id,
                ))

            return objects
        except Exception:
            return self._detect_basic(img, filter_classes)

    def _detect_tensorflow(
        self,
        img: np.ndarray,
        filter_classes: Optional[List[str]] = None,
    ) -> List[DetectedObject]:
        """Detect objects using TensorFlow Hub model."""
        try:
            import tensorflow as tf
            import tensorflow_hub as hub

            # Load SSD MobileNet model
            model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

            # Prepare input
            input_tensor = tf.convert_to_tensor(img)
            input_tensor = input_tensor[tf.newaxis, ...]

            # Run detection
            output = model(input_tensor)

            boxes = output['detection_boxes'][0].numpy()
            classes = output['detection_classes'][0].numpy().astype(int)
            scores = output['detection_scores'][0].numpy()

            h, w = img.shape[:2]
            objects = []

            for i in range(len(scores)):
                if scores[i] < self.min_confidence:
                    continue

                class_id = classes[i]
                label = COCO_CLASSES[class_id - 1] if class_id <= len(COCO_CLASSES) else f"class_{class_id}"

                if filter_classes and label.lower() not in [c.lower() for c in filter_classes]:
                    continue

                y1, x1, y2, x2 = boxes[i]
                bbox = (int(x1 * w), int(y1 * h), int((x2 - x1) * w), int((y2 - y1) * h))

                objects.append(DetectedObject(
                    label=label,
                    confidence=float(scores[i]),
                    bbox=bbox,
                    class_id=class_id,
                ))

            return objects
        except Exception:
            return self._detect_basic(img, filter_classes)

    def _detect_opencv(
        self,
        img: np.ndarray,
        filter_classes: Optional[List[str]] = None,
    ) -> List[DetectedObject]:
        """Detect objects using OpenCV DNN."""
        try:
            import cv2

            # Try to load YOLO weights
            config_path = self.model_path or 'yolov3.cfg'
            weights_path = 'yolov3.weights'

            if not Path(config_path).exists() or not Path(weights_path).exists():
                return self._detect_basic(img, filter_classes)

            net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

            blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)

            layer_names = net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

            outputs = net.forward(output_layers)

            h, w = img.shape[:2]
            boxes = []
            confidences = []
            class_ids = []

            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > self.min_confidence:
                        center_x = int(detection[0] * w)
                        center_y = int(detection[1] * h)
                        width = int(detection[2] * w)
                        height = int(detection[3] * h)
                        x = center_x - width // 2
                        y = center_y - height // 2

                        boxes.append([x, y, width, height])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply NMS
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.min_confidence, self.nms_threshold)

            objects = []
            for i in indices.flatten() if len(indices) > 0 else []:
                label = COCO_CLASSES[class_ids[i]] if class_ids[i] < len(COCO_CLASSES) else f"class_{class_ids[i]}"

                if filter_classes and label.lower() not in [c.lower() for c in filter_classes]:
                    continue

                objects.append(DetectedObject(
                    label=label,
                    confidence=confidences[i],
                    bbox=tuple(boxes[i]),
                    class_id=class_ids[i],
                ))

            return objects
        except Exception:
            return self._detect_basic(img, filter_classes)

    def _detect_basic(
        self,
        img: np.ndarray,
        filter_classes: Optional[List[str]] = None,
    ) -> List[DetectedObject]:
        """Basic detection fallback - returns empty."""
        return []

    def detect_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        classes: Optional[List[str]] = None,
    ) -> List[ObjectDetectionResult]:
        """Detect objects in multiple images."""
        return [self.detect(img, classes) for img in images]

    def count_objects(
        self,
        image: Union[str, Path, np.ndarray],
        label: Optional[str] = None,
    ) -> int:
        """Count objects in image, optionally filtered by label."""
        result = self.detect(image)
        if label:
            return len(result.filter_by_label(label))
        return result.count


# Convenience functions
def detect_objects(
    image: Union[str, Path, np.ndarray],
    min_confidence: float = 0.5,
    classes: Optional[List[str]] = None,
) -> ObjectDetectionResult:
    """Detect objects in an image."""
    detector = ObjectDetector(min_confidence=min_confidence)
    return detector.detect(image, classes=classes)


def find_objects(
    image: Union[str, Path, np.ndarray],
    label: str,
) -> List[DetectedObject]:
    """Find specific objects in an image."""
    detector = ObjectDetector()
    result = detector.detect(image)
    return result.filter_by_label(label)

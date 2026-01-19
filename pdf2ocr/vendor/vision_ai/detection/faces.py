"""Face Detection and Recognition Module.

Provides face detection, landmark extraction, and face recognition capabilities
with multiple backend support (OpenCV, dlib, face_recognition).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class FaceDetectionError(Exception):
    """Error during face detection."""
    pass


@dataclass
class FaceLandmarks:
    """Facial landmarks/keypoints."""
    left_eye: Optional[Tuple[int, int]] = None
    right_eye: Optional[Tuple[int, int]] = None
    nose: Optional[Tuple[int, int]] = None
    left_mouth: Optional[Tuple[int, int]] = None
    right_mouth: Optional[Tuple[int, int]] = None
    # Extended landmarks (68-point model)
    jaw: List[Tuple[int, int]] = field(default_factory=list)
    left_eyebrow: List[Tuple[int, int]] = field(default_factory=list)
    right_eyebrow: List[Tuple[int, int]] = field(default_factory=list)
    nose_bridge: List[Tuple[int, int]] = field(default_factory=list)
    nose_tip: List[Tuple[int, int]] = field(default_factory=list)
    left_eye_points: List[Tuple[int, int]] = field(default_factory=list)
    right_eye_points: List[Tuple[int, int]] = field(default_factory=list)
    top_lip: List[Tuple[int, int]] = field(default_factory=list)
    bottom_lip: List[Tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'left_eye': self.left_eye,
            'right_eye': self.right_eye,
            'nose': self.nose,
            'left_mouth': self.left_mouth,
            'right_mouth': self.right_mouth,
        }


@dataclass
class FaceInfo:
    """Detected face information."""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float = 0.0
    landmarks: Optional[FaceLandmarks] = None
    embedding: Optional[np.ndarray] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    face_id: Optional[str] = None

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
        }
        if self.landmarks:
            result['landmarks'] = self.landmarks.to_dict()
        if self.age is not None:
            result['age'] = self.age
        if self.gender is not None:
            result['gender'] = self.gender
        return result


@dataclass
class FaceMatch:
    """Face matching result."""
    face1: FaceInfo
    face2: FaceInfo
    similarity: float
    is_match: bool
    distance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'face1_bbox': self.face1.bbox,
            'face2_bbox': self.face2.bbox,
            'similarity': self.similarity,
            'is_match': self.is_match,
            'distance': self.distance,
        }


class FaceDetector:
    """Face detection and recognition with multiple backend support."""

    BACKENDS = ['opencv', 'dlib', 'face_recognition', 'mediapipe', 'basic']

    def __init__(
        self,
        backend: str = 'auto',
        min_confidence: float = 0.5,
        min_face_size: int = 20,
    ):
        """
        Initialize face detector.

        Args:
            backend: Detection backend ('auto', 'opencv', 'dlib', 'face_recognition', 'mediapipe', 'basic')
            min_confidence: Minimum detection confidence
            min_face_size: Minimum face size in pixels
        """
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size
        self.backend = backend

        # Lazy-loaded backends
        self._opencv_cascade = None
        self._opencv_dnn = None
        self._dlib_detector = None
        self._dlib_predictor = None
        self._face_recognition = None
        self._mediapipe = None

        # Determine best available backend
        if backend == 'auto':
            self.backend = self._find_best_backend()
        else:
            self.backend = backend

    def _find_best_backend(self) -> str:
        """Find the best available backend."""
        # Try face_recognition first (best quality)
        try:
            import face_recognition
            return 'face_recognition'
        except ImportError:
            pass

        # Try dlib
        try:
            import dlib
            return 'dlib'
        except ImportError:
            pass

        # Try mediapipe
        try:
            import mediapipe
            return 'mediapipe'
        except ImportError:
            pass

        # Try OpenCV DNN
        try:
            import cv2
            return 'opencv'
        except ImportError:
            pass

        return 'basic'

    def _get_opencv_cascade(self):
        """Get OpenCV Haar cascade detector."""
        if self._opencv_cascade is None:
            try:
                import cv2
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self._opencv_cascade = cv2.CascadeClassifier(cascade_path)
            except Exception:
                pass
        return self._opencv_cascade

    def _get_dlib_detector(self):
        """Get dlib face detector."""
        if self._dlib_detector is None:
            try:
                import dlib
                self._dlib_detector = dlib.get_frontal_face_detector()
            except ImportError:
                pass
        return self._dlib_detector

    def _get_dlib_predictor(self):
        """Get dlib shape predictor for landmarks."""
        if self._dlib_predictor is None:
            try:
                import dlib
                # Try to load 68-point model
                model_paths = [
                    'shape_predictor_68_face_landmarks.dat',
                    'models/shape_predictor_68_face_landmarks.dat',
                ]
                for path in model_paths:
                    if Path(path).exists():
                        self._dlib_predictor = dlib.shape_predictor(path)
                        break
            except (ImportError, RuntimeError):
                pass
        return self._dlib_predictor

    def _get_face_recognition(self):
        """Get face_recognition module."""
        if self._face_recognition is None:
            try:
                import face_recognition
                self._face_recognition = face_recognition
            except ImportError:
                pass
        return self._face_recognition

    def _get_mediapipe(self):
        """Get MediaPipe face detection."""
        if self._mediapipe is None:
            try:
                import mediapipe as mp
                self._mediapipe = mp.solutions.face_detection.FaceDetection(
                    min_detection_confidence=self.min_confidence
                )
            except ImportError:
                pass
        return self._mediapipe

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
        return_landmarks: bool = False,
        return_embeddings: bool = False,
    ) -> List[FaceInfo]:
        """
        Detect faces in an image.

        Args:
            image: Image path or numpy array
            return_landmarks: Include facial landmarks
            return_embeddings: Include face embeddings for recognition

        Returns:
            List of FaceInfo objects
        """
        img = self._load_image(image)

        if self.backend == 'face_recognition':
            return self._detect_face_recognition(img, return_landmarks, return_embeddings)
        elif self.backend == 'dlib':
            return self._detect_dlib(img, return_landmarks, return_embeddings)
        elif self.backend == 'mediapipe':
            return self._detect_mediapipe(img, return_landmarks)
        elif self.backend == 'opencv':
            return self._detect_opencv(img, return_landmarks)
        else:
            return self._detect_basic(img)

    def _detect_face_recognition(
        self,
        img: np.ndarray,
        return_landmarks: bool = False,
        return_embeddings: bool = False,
    ) -> List[FaceInfo]:
        """Detect faces using face_recognition library."""
        fr = self._get_face_recognition()
        if fr is None:
            return self._detect_basic(img)

        # Detect face locations
        face_locations = fr.face_locations(img, model='hog')
        faces = []

        # Get landmarks if requested
        landmarks_list = None
        if return_landmarks:
            landmarks_list = fr.face_landmarks(img)

        # Get embeddings if requested
        embeddings_list = None
        if return_embeddings:
            embeddings_list = fr.face_encodings(img, face_locations)

        for i, (top, right, bottom, left) in enumerate(face_locations):
            bbox = (left, top, right - left, bottom - top)

            if bbox[2] < self.min_face_size or bbox[3] < self.min_face_size:
                continue

            face = FaceInfo(
                bbox=bbox,
                confidence=0.95,  # face_recognition doesn't provide confidence
            )

            if landmarks_list and i < len(landmarks_list):
                lm = landmarks_list[i]
                face.landmarks = FaceLandmarks(
                    left_eye=lm.get('left_eye', [None])[0] if lm.get('left_eye') else None,
                    right_eye=lm.get('right_eye', [None])[0] if lm.get('right_eye') else None,
                    nose=lm.get('nose_tip', [None])[0] if lm.get('nose_tip') else None,
                    left_eye_points=lm.get('left_eye', []),
                    right_eye_points=lm.get('right_eye', []),
                    left_eyebrow=lm.get('left_eyebrow', []),
                    right_eyebrow=lm.get('right_eyebrow', []),
                    nose_bridge=lm.get('nose_bridge', []),
                    nose_tip=lm.get('nose_tip', []),
                    top_lip=lm.get('top_lip', []),
                    bottom_lip=lm.get('bottom_lip', []),
                )

            if embeddings_list and i < len(embeddings_list):
                face.embedding = embeddings_list[i]

            faces.append(face)

        return faces

    def _detect_dlib(
        self,
        img: np.ndarray,
        return_landmarks: bool = False,
        return_embeddings: bool = False,
    ) -> List[FaceInfo]:
        """Detect faces using dlib."""
        detector = self._get_dlib_detector()
        if detector is None:
            return self._detect_basic(img)

        # Detect faces
        dlib_rects = detector(img, 1)
        faces = []

        predictor = self._get_dlib_predictor() if return_landmarks else None

        for rect in dlib_rects:
            bbox = (rect.left(), rect.top(), rect.width(), rect.height())

            if bbox[2] < self.min_face_size or bbox[3] < self.min_face_size:
                continue

            face = FaceInfo(
                bbox=bbox,
                confidence=0.9,
            )

            if predictor and return_landmarks:
                shape = predictor(img, rect)
                points = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

                face.landmarks = FaceLandmarks(
                    left_eye=points[36] if len(points) > 36 else None,
                    right_eye=points[45] if len(points) > 45 else None,
                    nose=points[30] if len(points) > 30 else None,
                    left_mouth=points[48] if len(points) > 48 else None,
                    right_mouth=points[54] if len(points) > 54 else None,
                    jaw=points[0:17],
                    left_eyebrow=points[17:22],
                    right_eyebrow=points[22:27],
                    nose_bridge=points[27:31],
                    nose_tip=points[31:36],
                    left_eye_points=points[36:42],
                    right_eye_points=points[42:48],
                    top_lip=points[48:55] + [points[64], points[63], points[62], points[61], points[60]],
                    bottom_lip=points[54:60] + [points[48], points[60], points[67], points[66], points[65], points[64]],
                )

            faces.append(face)

        return faces

    def _detect_mediapipe(
        self,
        img: np.ndarray,
        return_landmarks: bool = False,
    ) -> List[FaceInfo]:
        """Detect faces using MediaPipe."""
        mp_detector = self._get_mediapipe()
        if mp_detector is None:
            return self._detect_basic(img)

        results = mp_detector.process(img)
        faces = []

        if results.detections:
            h, w = img.shape[:2]

            for detection in results.detections:
                bbox_data = detection.location_data.relative_bounding_box
                x = int(bbox_data.xmin * w)
                y = int(bbox_data.ymin * h)
                width = int(bbox_data.width * w)
                height = int(bbox_data.height * h)

                if width < self.min_face_size or height < self.min_face_size:
                    continue

                confidence = detection.score[0] if detection.score else 0.0

                if confidence < self.min_confidence:
                    continue

                face = FaceInfo(
                    bbox=(x, y, width, height),
                    confidence=confidence,
                )

                if return_landmarks:
                    keypoints = detection.location_data.relative_keypoints
                    if len(keypoints) >= 6:
                        face.landmarks = FaceLandmarks(
                            right_eye=(int(keypoints[0].x * w), int(keypoints[0].y * h)),
                            left_eye=(int(keypoints[1].x * w), int(keypoints[1].y * h)),
                            nose=(int(keypoints[2].x * w), int(keypoints[2].y * h)),
                            right_mouth=(int(keypoints[4].x * w), int(keypoints[4].y * h)),
                            left_mouth=(int(keypoints[5].x * w), int(keypoints[5].y * h)),
                        )

                faces.append(face)

        return faces

    def _detect_opencv(
        self,
        img: np.ndarray,
        return_landmarks: bool = False,
    ) -> List[FaceInfo]:
        """Detect faces using OpenCV."""
        cascade = self._get_opencv_cascade()
        if cascade is None:
            return self._detect_basic(img)

        import cv2

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Detect faces
        rects = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
        )

        faces = []
        for (x, y, w, h) in rects:
            face = FaceInfo(
                bbox=(int(x), int(y), int(w), int(h)),
                confidence=0.8,  # OpenCV cascade doesn't provide confidence
            )
            faces.append(face)

        return faces

    def _detect_basic(self, img: np.ndarray) -> List[FaceInfo]:
        """Basic face detection fallback."""
        # Without proper face detection, we just return empty
        return []

    def compare(
        self,
        image1: Union[str, Path, np.ndarray],
        image2: Union[str, Path, np.ndarray],
        tolerance: float = 0.6,
    ) -> Optional[FaceMatch]:
        """
        Compare faces in two images.

        Args:
            image1: First image
            image2: Second image
            tolerance: Distance tolerance for matching (lower = stricter)

        Returns:
            FaceMatch object if faces found, None otherwise
        """
        faces1 = self.detect(image1, return_embeddings=True)
        faces2 = self.detect(image2, return_embeddings=True)

        if not faces1 or not faces2:
            return None

        face1 = faces1[0]
        face2 = faces2[0]

        if face1.embedding is None or face2.embedding is None:
            return None

        # Calculate distance
        distance = np.linalg.norm(face1.embedding - face2.embedding)
        similarity = max(0, 1 - distance)
        is_match = distance <= tolerance

        return FaceMatch(
            face1=face1,
            face2=face2,
            similarity=similarity,
            is_match=is_match,
            distance=distance,
        )

    def find_matches(
        self,
        query_image: Union[str, Path, np.ndarray],
        database_images: List[Union[str, Path]],
        tolerance: float = 0.6,
        top_k: int = 5,
    ) -> List[Tuple[str, FaceMatch]]:
        """
        Find matching faces in a database.

        Args:
            query_image: Query image with face
            database_images: List of database image paths
            tolerance: Match tolerance
            top_k: Number of top matches to return

        Returns:
            List of (image_path, FaceMatch) tuples sorted by similarity
        """
        matches = []

        for db_image in database_images:
            try:
                match = self.compare(query_image, db_image, tolerance)
                if match:
                    matches.append((str(db_image), match))
            except Exception:
                continue

        # Sort by similarity (descending)
        matches.sort(key=lambda x: x[1].similarity, reverse=True)

        return matches[:top_k]


# Convenience functions
def detect_faces(
    image: Union[str, Path, np.ndarray],
    backend: str = 'auto',
    return_landmarks: bool = False,
) -> List[FaceInfo]:
    """Detect faces in an image."""
    detector = FaceDetector(backend=backend)
    return detector.detect(image, return_landmarks=return_landmarks)


def find_faces(
    image: Union[str, Path, np.ndarray],
    return_embeddings: bool = False,
) -> List[FaceInfo]:
    """Find all faces in an image with full details."""
    detector = FaceDetector()
    return detector.detect(image, return_landmarks=True, return_embeddings=return_embeddings)


def compare_faces(
    image1: Union[str, Path, np.ndarray],
    image2: Union[str, Path, np.ndarray],
    tolerance: float = 0.6,
) -> Optional[FaceMatch]:
    """Compare faces in two images."""
    detector = FaceDetector()
    return detector.compare(image1, image2, tolerance)

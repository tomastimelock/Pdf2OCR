"""Unified VisionAI Class.

The main entry point for all computer vision and AI analysis operations.
Provides a comprehensive, unified API for detection, analysis, document processing,
and media intelligence.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class VisionAIError(Exception):
    """Error in VisionAI operations."""
    pass


@dataclass
class VisionAnalysis:
    """Comprehensive vision analysis result."""
    path: str
    faces: List[Any] = field(default_factory=list)
    objects: List[Any] = field(default_factory=list)
    text_regions: List[Any] = field(default_factory=list)
    colors: Optional[Any] = None
    scene: Optional[Any] = None
    is_safe: bool = True
    confidence: float = 0.0
    barcodes: List[Any] = field(default_factory=list)
    emotions: List[Any] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    description: str = ""
    processing_time: float = 0.0

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def object_count(self) -> int:
        return len(self.objects)

    @property
    def has_faces(self) -> bool:
        return len(self.faces) > 0

    @property
    def has_text(self) -> bool:
        return len(self.text_regions) > 0

    @property
    def has_barcodes(self) -> bool:
        return len(self.barcodes) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path,
            'face_count': self.face_count,
            'object_count': self.object_count,
            'has_text': self.has_text,
            'is_safe': self.is_safe,
            'confidence': self.confidence,
            'tags': self.tags,
            'description': self.description,
            'scene': self.scene.to_dict() if self.scene and hasattr(self.scene, 'to_dict') else str(self.scene),
            'processing_time': self.processing_time,
        }


class VisionAI:
    """Unified Computer Vision and AI Analysis Engine.

    This class provides a single, unified API for all computer vision tasks:
    - Face detection and recognition
    - Object detection
    - Text detection and OCR
    - Color analysis
    - Scene classification
    - Content moderation (NSFW detection)
    - Barcode/QR code scanning
    - Emotion analysis
    - Document scanning and parsing
    - AI-powered image description and tagging
    - Similarity search

    All components are lazy-loaded for efficient memory usage.
    """

    def __init__(
        self,
        backends: str = 'auto',
        api_key: Optional[str] = None,
    ):
        """
        Initialize VisionAI.

        Args:
            backends: Backend preference ('auto', 'performance', 'accuracy')
            api_key: API key for cloud services (OpenAI, Anthropic)
        """
        self.backends = backends
        self.api_key = api_key

        # Lazy-loaded components
        self._face_detector = None
        self._object_detector = None
        self._logo_detector = None
        self._text_detector = None
        self._barcode_scanner = None
        self._emotion_analyzer = None
        self._scene_classifier = None
        self._color_analyzer = None
        self._nsfw_detector = None
        self._aesthetics_scorer = None
        self._ocr_engine = None
        self._document_scanner = None
        self._document_parser = None
        self._media_analyzer = None

    # =========================================================================
    # Component Getters (Lazy Loading)
    # =========================================================================

    def _get_face_detector(self):
        if self._face_detector is None:
            from .detection.faces import FaceDetector
            self._face_detector = FaceDetector(backend=self.backends)
        return self._face_detector

    def _get_object_detector(self):
        if self._object_detector is None:
            from .detection.objects import ObjectDetector
            self._object_detector = ObjectDetector(backend=self.backends)
        return self._object_detector

    def _get_logo_detector(self):
        if self._logo_detector is None:
            from .detection.logos import LogoDetector
            self._logo_detector = LogoDetector()
        return self._logo_detector

    def _get_text_detector(self):
        if self._text_detector is None:
            from .detection.text import TextDetector
            self._text_detector = TextDetector(backend=self.backends)
        return self._text_detector

    def _get_barcode_scanner(self):
        if self._barcode_scanner is None:
            from .detection.barcodes import BarcodeScanner
            self._barcode_scanner = BarcodeScanner(backend=self.backends)
        return self._barcode_scanner

    def _get_emotion_analyzer(self):
        if self._emotion_analyzer is None:
            from .analysis.emotions import EmotionAnalyzer
            self._emotion_analyzer = EmotionAnalyzer(backend=self.backends)
        return self._emotion_analyzer

    def _get_scene_classifier(self):
        if self._scene_classifier is None:
            from .analysis.scenes import SceneClassifier
            self._scene_classifier = SceneClassifier(backend=self.backends)
        return self._scene_classifier

    def _get_color_analyzer(self):
        if self._color_analyzer is None:
            from .analysis.colors import ColorAnalyzer
            self._color_analyzer = ColorAnalyzer()
        return self._color_analyzer

    def _get_nsfw_detector(self):
        if self._nsfw_detector is None:
            from .analysis.nsfw import NSFWDetector
            self._nsfw_detector = NSFWDetector(backend=self.backends)
        return self._nsfw_detector

    def _get_aesthetics_scorer(self):
        if self._aesthetics_scorer is None:
            from .analysis.aesthetics import AestheticsScorer
            self._aesthetics_scorer = AestheticsScorer(backend=self.backends)
        return self._aesthetics_scorer

    def _get_ocr_engine(self):
        if self._ocr_engine is None:
            from .documents.ocr import OCREngine
            self._ocr_engine = OCREngine(backend=self.backends)
        return self._ocr_engine

    def _get_document_scanner(self):
        if self._document_scanner is None:
            from .documents.scanner import DocumentScanner
            self._document_scanner = DocumentScanner()
        return self._document_scanner

    def _get_document_parser(self):
        if self._document_parser is None:
            from .documents.parser import DocumentParser
            self._document_parser = DocumentParser()
        return self._document_parser

    def _get_media_analyzer(self):
        if self._media_analyzer is None:
            from .intelligence.analyzer import MediaAnalyzer
            self._media_analyzer = MediaAnalyzer(backend=self.backends, api_key=self.api_key)
        return self._media_analyzer

    # =========================================================================
    # Image Loading
    # =========================================================================

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

    # =========================================================================
    # Full Analysis
    # =========================================================================

    def analyze(
        self,
        image: Union[str, Path, np.ndarray],
        detect_all: bool = True,
        detect_faces: bool = True,
        detect_objects: bool = True,
        detect_text: bool = True,
        analyze_colors: bool = True,
        classify_scene: bool = True,
        check_nsfw: bool = True,
        scan_barcodes: bool = False,
        analyze_emotions: bool = False,
    ) -> VisionAnalysis:
        """
        Perform comprehensive image analysis.

        Args:
            image: Image path or numpy array
            detect_all: Enable all detection/analysis (default)
            detect_faces: Detect faces
            detect_objects: Detect objects
            detect_text: Detect text regions
            analyze_colors: Analyze colors
            classify_scene: Classify scene
            check_nsfw: Check content safety
            scan_barcodes: Scan for barcodes
            analyze_emotions: Analyze emotions (requires faces)

        Returns:
            VisionAnalysis with comprehensive results
        """
        import time
        start_time = time.time()

        path = str(image) if not isinstance(image, np.ndarray) else "array"
        result = VisionAnalysis(path=path)

        # If detect_all is True, enable everything
        if detect_all:
            detect_faces = True
            detect_objects = True
            detect_text = True
            analyze_colors = True
            classify_scene = True
            check_nsfw = True

        img = self._load_image(image)

        # Face detection
        if detect_faces:
            try:
                result.faces = self._get_face_detector().detect(img, return_landmarks=True)
            except Exception:
                pass

        # Object detection
        if detect_objects:
            try:
                obj_result = self._get_object_detector().detect(img)
                result.objects = obj_result.objects
            except Exception:
                pass

        # Text detection
        if detect_text:
            try:
                text_result = self._get_text_detector().detect(img)
                result.text_regions = text_result.regions
            except Exception:
                pass

        # Color analysis
        if analyze_colors:
            try:
                result.colors = self._get_color_analyzer().get_palette(img)
            except Exception:
                pass

        # Scene classification
        if classify_scene:
            try:
                scene_result = self._get_scene_classifier().classify(img)
                result.scene = scene_result.top_scene
                result.tags.extend(scene_result.tags[:5])
            except Exception:
                pass

        # NSFW check
        if check_nsfw:
            try:
                safety = self._get_nsfw_detector().check(img)
                result.is_safe = safety.is_safe
                result.confidence = safety.confidence
            except Exception:
                pass

        # Barcode scanning
        if scan_barcodes:
            try:
                scan_result = self._get_barcode_scanner().scan(img)
                result.barcodes = scan_result.barcodes
            except Exception:
                pass

        # Emotion analysis
        if analyze_emotions and result.faces:
            try:
                result.emotions = self._get_emotion_analyzer().analyze(img)
            except Exception:
                pass

        result.processing_time = time.time() - start_time
        return result

    def analyze_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        **kwargs,
    ) -> List[VisionAnalysis]:
        """Analyze multiple images."""
        return [self.analyze(img, **kwargs) for img in images]

    # =========================================================================
    # Detection Operations
    # =========================================================================

    def detect_faces(
        self,
        image: Union[str, Path, np.ndarray],
        return_embeddings: bool = False,
        return_landmarks: bool = True,
    ) -> List[Any]:
        """Detect faces in an image."""
        detector = self._get_face_detector()
        return detector.detect(image, return_landmarks=return_landmarks, return_embeddings=return_embeddings)

    def detect_objects(
        self,
        image: Union[str, Path, np.ndarray],
        min_confidence: float = 0.5,
        classes: Optional[List[str]] = None,
    ) -> List[Any]:
        """Detect objects in an image."""
        detector = self._get_object_detector()
        detector.min_confidence = min_confidence
        result = detector.detect(image, classes=classes)
        return result.objects

    def detect_logos(
        self,
        image: Union[str, Path, np.ndarray],
        templates: Optional[List[str]] = None,
    ) -> List[Any]:
        """Detect logos in an image."""
        detector = self._get_logo_detector()
        result = detector.detect(image, templates=templates)
        return result.logos

    def detect_text(
        self,
        image: Union[str, Path, np.ndarray],
        with_content: bool = False,
    ) -> List[Any]:
        """Detect text regions in an image."""
        detector = self._get_text_detector()
        result = detector.detect(image, with_text=with_content)
        return result.regions

    def detect_barcodes(
        self,
        image: Union[str, Path, np.ndarray],
        barcode_types: Optional[List[str]] = None,
    ) -> List[Any]:
        """Detect and decode barcodes in an image."""
        scanner = self._get_barcode_scanner()
        result = scanner.scan(image, barcode_types=barcode_types)
        return result.barcodes

    # =========================================================================
    # Analysis Operations
    # =========================================================================

    def analyze_emotions(
        self,
        image: Union[str, Path, np.ndarray],
    ) -> List[Any]:
        """Analyze emotions in an image."""
        analyzer = self._get_emotion_analyzer()
        return analyzer.analyze(image)

    def analyze_scene(
        self,
        image: Union[str, Path, np.ndarray],
    ) -> Any:
        """Classify scene in an image."""
        classifier = self._get_scene_classifier()
        return classifier.classify(image)

    def analyze_colors(
        self,
        image: Union[str, Path, np.ndarray],
        num_colors: int = 5,
    ) -> Any:
        """Analyze colors in an image."""
        analyzer = self._get_color_analyzer()
        return analyzer.get_palette(image, count=num_colors)

    def check_nsfw(
        self,
        image: Union[str, Path, np.ndarray],
        threshold: float = 0.5,
    ) -> Any:
        """Check image for NSFW content."""
        detector = self._get_nsfw_detector()
        detector.threshold = threshold
        return detector.check(image)

    def score_aesthetics(
        self,
        image: Union[str, Path, np.ndarray],
        detailed: bool = True,
    ) -> Any:
        """Score image aesthetics."""
        scorer = self._get_aesthetics_scorer()
        return scorer.score(image, detailed=detailed)

    # =========================================================================
    # Document Operations
    # =========================================================================

    def ocr(
        self,
        image: Union[str, Path, np.ndarray],
        lang: str = 'eng',
    ) -> Any:
        """Extract text from image using OCR."""
        engine = self._get_ocr_engine()
        engine.language = lang
        return engine.extract(image)

    def scan_document(
        self,
        image: Union[str, Path, np.ndarray],
        enhance: bool = True,
        deskew: bool = True,
    ) -> Any:
        """Scan and enhance a document image."""
        scanner = self._get_document_scanner()
        from .documents.scanner import ScanConfig
        config = ScanConfig(enhance=enhance, deskew=deskew)
        return scanner.scan(image, config)

    def parse_receipt(
        self,
        image: Union[str, Path, np.ndarray],
    ) -> Any:
        """Parse receipt data from image."""
        parser = self._get_document_parser()
        return parser.parse_receipt(image)

    def parse_form(
        self,
        image: Union[str, Path, np.ndarray],
        template: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Parse form data from image."""
        parser = self._get_document_parser()
        return parser.parse_form(image, template)

    # =========================================================================
    # Intelligence Operations
    # =========================================================================

    def describe(
        self,
        image: Union[str, Path, np.ndarray],
        style: str = 'detailed',
    ) -> str:
        """Generate AI description of an image."""
        analyzer = self._get_media_analyzer()
        return analyzer.describe(image, style)

    def suggest_tags(
        self,
        image: Union[str, Path, np.ndarray],
        max_tags: int = 10,
    ) -> List[str]:
        """Suggest relevant tags for an image."""
        analyzer = self._get_media_analyzer()
        return analyzer.suggest_tags(image, max_tags)

    def find_similar(
        self,
        image: Union[str, Path, np.ndarray],
        database: List[Union[str, Path]],
        top_k: int = 10,
    ) -> List[Any]:
        """Find similar images in a database."""
        analyzer = self._get_media_analyzer()
        return analyzer.find_similar(image, database, top_k)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def compare_faces(
        self,
        image1: Union[str, Path, np.ndarray],
        image2: Union[str, Path, np.ndarray],
        tolerance: float = 0.6,
    ) -> Optional[Any]:
        """Compare faces in two images."""
        detector = self._get_face_detector()
        return detector.compare(image1, image2, tolerance)

    def read_barcodes(
        self,
        image: Union[str, Path, np.ndarray],
    ) -> List[str]:
        """Read barcode data from image."""
        barcodes = self.detect_barcodes(image)
        return [b.data for b in barcodes]

    def extract_text(
        self,
        image: Union[str, Path, np.ndarray],
        lang: str = 'eng',
    ) -> str:
        """Extract text from image (simple string output)."""
        result = self.ocr(image, lang)
        return result.text

    def is_safe_content(
        self,
        image: Union[str, Path, np.ndarray],
        threshold: float = 0.5,
    ) -> bool:
        """Quick check if content is safe."""
        result = self.check_nsfw(image, threshold)
        return result.is_safe

    def get_dominant_color(
        self,
        image: Union[str, Path, np.ndarray],
    ) -> Optional[Tuple[int, int, int]]:
        """Get the dominant color of an image."""
        palette = self.analyze_colors(image, num_colors=1)
        if palette and palette.colors:
            return palette.colors[0].rgb
        return None

    def count_faces(
        self,
        image: Union[str, Path, np.ndarray],
    ) -> int:
        """Count faces in an image."""
        faces = self.detect_faces(image, return_landmarks=False)
        return len(faces)

    def count_objects(
        self,
        image: Union[str, Path, np.ndarray],
        label: Optional[str] = None,
    ) -> int:
        """Count objects in an image."""
        objects = self.detect_objects(image)
        if label:
            return len([o for o in objects if o.label.lower() == label.lower()])
        return len(objects)


# Create a default instance
_default_vision = None


def get_vision_ai(
    backends: str = 'auto',
    api_key: Optional[str] = None,
) -> VisionAI:
    """Get or create the default VisionAI instance."""
    global _default_vision
    if _default_vision is None:
        _default_vision = VisionAI(backends=backends, api_key=api_key)
    return _default_vision

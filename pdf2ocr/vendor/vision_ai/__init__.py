"""VisionAI - Unified Computer Vision and AI Analysis Engine.

This super-module consolidates all AI/ML visual analysis capabilities into a single,
unified API. It combines face detection, object detection, OCR, scene classification,
content moderation, color analysis, and AI-powered image understanding.

Modules consolidated:
- Face detection and recognition
- Object detection
- Logo detection
- Text detection and OCR
- Barcode/QR code scanning
- Emotion analysis
- Scene classification
- Color analysis
- NSFW/content moderation
- Aesthetics scoring
- Document scanning and parsing
- AI-powered image description and tagging
- PDF toolkit (extraction, splitting, merging)
- Multi-engine OCR (Tesseract, OpenAI Vision, Mistral)
- Image metadata extraction

Basic Usage:
    from vision_ai import VisionAI

    # Create instance
    vision = VisionAI()

    # Full analysis
    result = vision.analyze("image.jpg")
    print(f"Faces: {result.face_count}")
    print(f"Objects: {result.object_count}")
    print(f"Safe: {result.is_safe}")

    # Individual operations
    faces = vision.detect_faces("image.jpg")
    objects = vision.detect_objects("image.jpg")
    text = vision.extract_text("document.jpg")
    palette = vision.analyze_colors("image.jpg")

    # AI description
    description = vision.describe("image.jpg")
    tags = vision.suggest_tags("image.jpg")

Example with document processing:
    # Scan and enhance document
    doc = vision.scan_document("photo.jpg")

    # OCR
    ocr_result = vision.ocr("document.jpg", lang="eng")
    print(ocr_result.text)

    # Parse receipt
    receipt = vision.parse_receipt("receipt.jpg")
    print(f"Total: ${receipt.total}")
"""

# Main VisionAI class
from .vision import VisionAI, VisionAnalysis, get_vision_ai

# Detection submodule
from .detection import (
    # Faces
    FaceDetector,
    FaceInfo,
    FaceMatch,
    FaceLandmarks,
    detect_faces,
    find_faces,
    compare_faces,
    # Objects
    ObjectDetector,
    DetectedObject,
    ObjectDetectionResult,
    detect_objects,
    find_objects,
    # Logos
    LogoDetector,
    LogoMatch,
    LogoDetectionResult,
    detect_logos,
    match_logo,
    # Text
    TextDetector,
    TextRegion,
    TextDetectionResult,
    detect_text,
    find_text_regions,
    # Barcodes
    BarcodeScanner,
    BarcodeInfo,
    BarcodeScanResult,
    scan_barcodes,
    decode_qr,
)

# Analysis submodule
from .analysis import (
    # Emotions
    EmotionAnalyzer,
    EmotionResult,
    EmotionScore,
    analyze_emotions,
    detect_mood,
    # Scenes
    SceneClassifier,
    SceneInfo,
    SceneClassification,
    classify_scene,
    get_scene_tags,
    # Colors
    ColorAnalyzer,
    ColorPalette,
    ColorInfo,
    ColorHistogram,
    analyze_colors,
    get_dominant_colors,
    get_palette,
    # NSFW
    NSFWDetector,
    SafetyResult,
    SafetyCategory,
    check_nsfw,
    is_safe,
    moderate_content,
    # Aesthetics
    AestheticsScorer,
    AestheticsScore,
    CompositionAnalysis,
    score_aesthetics,
    rate_image,
)

# Documents submodule
from .documents import (
    # OCR
    OCREngine,
    OCRResult,
    OCRLine,
    OCRWord,
    extract_text,
    read_text,
    # Scanner
    DocumentScanner,
    ScannedDocument,
    ScanConfig,
    scan_document,
    enhance_document,
    # Parser
    DocumentParser,
    ReceiptData,
    FormData,
    FieldValue,
    parse_receipt,
    parse_form,
    extract_fields,
)

# Intelligence submodule
from .intelligence import (
    MediaAnalyzer,
    AnalysisResult,
    ImageDescription,
    SimilarMatch,
    analyze_media,
    describe_image,
    suggest_tags,
)


# PDF Toolkit (from extraction) - optional
try:
    from .pdf_toolkit import PDFToolkitProvider
except ImportError:
    PDFToolkitProvider = None

# Multi-engine OCR (from extraction) - optional
try:
    from .ocr_multi_engine import MultiEngineOCR, OCRProviderFactory
except ImportError:
    MultiEngineOCR = None
    OCRProviderFactory = None

# Image Metadata Extractor (from extraction) - optional
try:
    from .image_metadata_extractor import ImageMetadataExtractor
except ImportError:
    ImageMetadataExtractor = None


__version__ = "1.0.0"
__author__ = "VisionAI"
__all__ = [
    # Main API
    "VisionAI",
    "VisionAnalysis",
    "get_vision_ai",

    # Detection - Faces
    "FaceDetector",
    "FaceInfo",
    "FaceMatch",
    "FaceLandmarks",
    "detect_faces",
    "find_faces",
    "compare_faces",

    # Detection - Objects
    "ObjectDetector",
    "DetectedObject",
    "ObjectDetectionResult",
    "detect_objects",
    "find_objects",

    # Detection - Logos
    "LogoDetector",
    "LogoMatch",
    "LogoDetectionResult",
    "detect_logos",
    "match_logo",

    # Detection - Text
    "TextDetector",
    "TextRegion",
    "TextDetectionResult",
    "detect_text",
    "find_text_regions",

    # Detection - Barcodes
    "BarcodeScanner",
    "BarcodeInfo",
    "BarcodeScanResult",
    "scan_barcodes",
    "decode_qr",

    # Analysis - Emotions
    "EmotionAnalyzer",
    "EmotionResult",
    "EmotionScore",
    "analyze_emotions",
    "detect_mood",

    # Analysis - Scenes
    "SceneClassifier",
    "SceneInfo",
    "SceneClassification",
    "classify_scene",
    "get_scene_tags",

    # Analysis - Colors
    "ColorAnalyzer",
    "ColorPalette",
    "ColorInfo",
    "ColorHistogram",
    "analyze_colors",
    "get_dominant_colors",
    "get_palette",

    # Analysis - NSFW
    "NSFWDetector",
    "SafetyResult",
    "SafetyCategory",
    "check_nsfw",
    "is_safe",
    "moderate_content",

    # Analysis - Aesthetics
    "AestheticsScorer",
    "AestheticsScore",
    "CompositionAnalysis",
    "score_aesthetics",
    "rate_image",

    # Documents - OCR
    "OCREngine",
    "OCRResult",
    "OCRLine",
    "OCRWord",
    "extract_text",
    "read_text",

    # Documents - Scanner
    "DocumentScanner",
    "ScannedDocument",
    "ScanConfig",
    "scan_document",
    "enhance_document",

    # Documents - Parser
    "DocumentParser",
    "ReceiptData",
    "FormData",
    "FieldValue",
    "parse_receipt",
    "parse_form",
    "extract_fields",

    # Intelligence
    "MediaAnalyzer",
    "AnalysisResult",
    "ImageDescription",
    "SimilarMatch",
    "analyze_media",
    "describe_image",
    "suggest_tags",

    # Extraction - PDF
    "PDFToolkitProvider",

    # Extraction - Multi-engine OCR
    "MultiEngineOCR",
    "OCRProviderFactory",

    # Extraction - Metadata
    "ImageMetadataExtractor",
]

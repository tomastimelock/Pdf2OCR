"""Detection submodule for VisionAI.

Provides detection capabilities for faces, objects, logos, text, and barcodes.
"""

from .faces import (
    FaceDetector,
    FaceInfo,
    FaceMatch,
    FaceLandmarks,
    detect_faces,
    find_faces,
    compare_faces,
)
from .objects import (
    ObjectDetector,
    DetectedObject,
    ObjectDetectionResult,
    detect_objects,
    find_objects,
)
from .logos import (
    LogoDetector,
    LogoMatch,
    LogoDetectionResult,
    detect_logos,
    match_logo,
)
from .text import (
    TextDetector,
    TextRegion,
    TextDetectionResult,
    detect_text,
    find_text_regions,
)
from .barcodes import (
    BarcodeScanner,
    BarcodeInfo,
    BarcodeScanResult,
    scan_barcodes,
    decode_qr,
)

__all__ = [
    # Faces
    "FaceDetector",
    "FaceInfo",
    "FaceMatch",
    "FaceLandmarks",
    "detect_faces",
    "find_faces",
    "compare_faces",
    # Objects
    "ObjectDetector",
    "DetectedObject",
    "ObjectDetectionResult",
    "detect_objects",
    "find_objects",
    # Logos
    "LogoDetector",
    "LogoMatch",
    "LogoDetectionResult",
    "detect_logos",
    "match_logo",
    # Text
    "TextDetector",
    "TextRegion",
    "TextDetectionResult",
    "detect_text",
    "find_text_regions",
    # Barcodes
    "BarcodeScanner",
    "BarcodeInfo",
    "BarcodeScanResult",
    "scan_barcodes",
    "decode_qr",
]

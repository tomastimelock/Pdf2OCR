"""Barcode and QR Code Scanning Module.

Provides barcode and QR code detection and decoding capabilities
with multiple backend support (pyzbar, zxing-cpp, opencv).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class BarcodeScanError(Exception):
    """Error during barcode scanning."""
    pass


# Barcode type categories
BARCODE_1D = ['EAN13', 'EAN8', 'UPCA', 'UPCE', 'CODE128', 'CODE39', 'CODE93', 'ITF', 'CODABAR']
BARCODE_2D = ['QRCODE', 'QR_CODE', 'DATAMATRIX', 'PDF417', 'AZTEC']


@dataclass
class BarcodeInfo:
    """A detected barcode or QR code."""
    data: str
    type: str
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    polygon: List[Tuple[int, int]] = field(default_factory=list)
    confidence: float = 1.0
    raw_data: Optional[bytes] = None

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
    def is_qr(self) -> bool:
        return self.type.upper() in ('QRCODE', 'QR_CODE', 'QR')

    @property
    def is_1d(self) -> bool:
        return self.type.upper() in BARCODE_1D

    @property
    def is_2d(self) -> bool:
        return self.type.upper() in BARCODE_2D

    def to_dict(self) -> Dict[str, Any]:
        return {
            'data': self.data,
            'type': self.type,
            'bbox': self.bbox,
            'polygon': self.polygon,
            'confidence': self.confidence,
            'is_qr': self.is_qr,
            'is_1d': self.is_1d,
        }


@dataclass
class BarcodeScanResult:
    """Barcode scan result for an image."""
    path: str
    barcodes: List[BarcodeInfo] = field(default_factory=list)
    processing_time: float = 0.0
    scanner: str = ""

    @property
    def count(self) -> int:
        return len(self.barcodes)

    @property
    def has_barcodes(self) -> bool:
        return self.count > 0

    @property
    def qr_codes(self) -> List[BarcodeInfo]:
        return [b for b in self.barcodes if b.is_qr]

    @property
    def barcodes_1d(self) -> List[BarcodeInfo]:
        return [b for b in self.barcodes if b.is_1d]

    @property
    def data_list(self) -> List[str]:
        return [b.data for b in self.barcodes]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path,
            'count': self.count,
            'barcodes': [b.to_dict() for b in self.barcodes],
            'data_list': self.data_list,
            'scanner': self.scanner,
        }


class BarcodeScanner:
    """Barcode and QR code scanner with multiple backend support."""

    BACKENDS = ['pyzbar', 'zxing', 'opencv', 'basic']

    def __init__(
        self,
        backend: str = 'auto',
        try_harder: bool = False,
        barcode_types: Optional[List[str]] = None,
    ):
        """
        Initialize barcode scanner.

        Args:
            backend: Scanning backend ('auto', 'pyzbar', 'zxing', 'opencv', 'basic')
            try_harder: Try harder to find barcodes (slower but more accurate)
            barcode_types: Filter to specific barcode types
        """
        self.try_harder = try_harder
        self.barcode_types = barcode_types
        self.backend = backend

        # Lazy-loaded decoders
        self._pyzbar = None
        self._zxing = None
        self._qr_detector = None

        if backend == 'auto':
            self.backend = self._find_best_backend()

    def _find_best_backend(self) -> str:
        """Find the best available backend."""
        # Try pyzbar
        try:
            from pyzbar import pyzbar
            return 'pyzbar'
        except ImportError:
            pass

        # Try zxing-cpp
        try:
            import zxingcpp
            return 'zxing'
        except ImportError:
            pass

        # Try OpenCV QR detector
        try:
            import cv2
            return 'opencv'
        except ImportError:
            pass

        return 'basic'

    def _get_pyzbar(self):
        """Get pyzbar decoder."""
        if self._pyzbar is None:
            try:
                from pyzbar import pyzbar
                self._pyzbar = pyzbar
            except ImportError:
                pass
        return self._pyzbar

    def _get_zxing(self):
        """Get zxing-cpp decoder."""
        if self._zxing is None:
            try:
                import zxingcpp
                self._zxing = zxingcpp
            except ImportError:
                pass
        return self._zxing

    def _get_qr_detector(self):
        """Get OpenCV QR detector."""
        if self._qr_detector is None:
            try:
                import cv2
                self._qr_detector = cv2.QRCodeDetector()
            except (ImportError, AttributeError):
                pass
        return self._qr_detector

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

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        """Convert to grayscale."""
        if len(img.shape) == 2:
            return img
        try:
            import cv2
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        except ImportError:
            return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    def scan(
        self,
        image: Union[str, Path, np.ndarray],
        barcode_types: Optional[List[str]] = None,
    ) -> BarcodeScanResult:
        """
        Scan image for barcodes and QR codes.

        Args:
            image: Image path or numpy array
            barcode_types: Filter to specific types (overrides init setting)

        Returns:
            BarcodeScanResult with detected barcodes
        """
        import time
        start_time = time.time()

        img = self._load_image(image)
        path = str(image) if not isinstance(image, np.ndarray) else "array"

        filter_types = barcode_types or self.barcode_types

        if self.backend == 'pyzbar':
            barcodes = self._scan_pyzbar(img)
            scanner_name = 'pyzbar'
        elif self.backend == 'zxing':
            barcodes = self._scan_zxing(img)
            scanner_name = 'zxing-cpp'
        elif self.backend == 'opencv':
            barcodes = self._scan_opencv(img)
            scanner_name = 'opencv'
        else:
            barcodes = []
            scanner_name = 'none'

        # Filter by type if requested
        if filter_types:
            filter_types_upper = [t.upper() for t in filter_types]
            barcodes = [b for b in barcodes if b.type.upper() in filter_types_upper]

        processing_time = time.time() - start_time

        return BarcodeScanResult(
            path=path,
            barcodes=barcodes,
            processing_time=processing_time,
            scanner=scanner_name,
        )

    def _scan_pyzbar(self, img: np.ndarray) -> List[BarcodeInfo]:
        """Scan using pyzbar."""
        pyzbar = self._get_pyzbar()
        if pyzbar is None:
            return []

        # pyzbar works with PIL Image
        try:
            from PIL import Image
            pil_img = Image.fromarray(img)
        except ImportError:
            pil_img = img

        decoded = pyzbar.decode(pil_img)
        barcodes = []

        for obj in decoded:
            # Get bounding box
            x, y, w, h = obj.rect

            # Get polygon points
            polygon = [(p.x, p.y) for p in obj.polygon]

            # Decode data
            try:
                data = obj.data.decode('utf-8')
            except UnicodeDecodeError:
                data = obj.data.decode('latin-1')

            barcodes.append(BarcodeInfo(
                data=data,
                type=obj.type,
                bbox=(x, y, w, h),
                polygon=polygon,
                confidence=1.0,
                raw_data=obj.data,
            ))

        return barcodes

    def _scan_zxing(self, img: np.ndarray) -> List[BarcodeInfo]:
        """Scan using zxing-cpp."""
        zxing = self._get_zxing()
        if zxing is None:
            return []

        results = zxing.read_barcodes(img)
        barcodes = []

        for result in results:
            # Get bounding box from position
            pos = result.position
            points = [
                (pos.top_left.x, pos.top_left.y),
                (pos.top_right.x, pos.top_right.y),
                (pos.bottom_right.x, pos.bottom_right.y),
                (pos.bottom_left.x, pos.bottom_left.y),
            ]

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            x = min(xs)
            y = min(ys)
            w = max(xs) - x
            h = max(ys) - y

            barcodes.append(BarcodeInfo(
                data=result.text,
                type=str(result.format).replace('BarcodeFormat.', ''),
                bbox=(x, y, w, h),
                polygon=points,
                confidence=1.0,
            ))

        return barcodes

    def _scan_opencv(self, img: np.ndarray) -> List[BarcodeInfo]:
        """Scan using OpenCV (QR codes only)."""
        detector = self._get_qr_detector()
        if detector is None:
            return []

        import cv2

        barcodes = []

        # Detect QR codes
        data, points, _ = detector.detectAndDecode(img)

        if data and points is not None:
            points = points[0].astype(int)

            xs = points[:, 0]
            ys = points[:, 1]

            x = int(min(xs))
            y = int(min(ys))
            w = int(max(xs) - x)
            h = int(max(ys) - y)

            polygon = [(int(p[0]), int(p[1])) for p in points]

            barcodes.append(BarcodeInfo(
                data=data,
                type='QRCODE',
                bbox=(x, y, w, h),
                polygon=polygon,
                confidence=1.0,
            ))

        # Try to detect multiple QR codes
        try:
            multi_detector = cv2.QRCodeDetectorAruco()
            retval, decoded_info, points_arr, _ = multi_detector.detectAndDecodeMulti(img)

            if retval and decoded_info:
                for i, data in enumerate(decoded_info):
                    if not data:
                        continue

                    pts = points_arr[i].astype(int)
                    xs = pts[:, 0]
                    ys = pts[:, 1]

                    x = int(min(xs))
                    y = int(min(ys))
                    w = int(max(xs) - x)
                    h = int(max(ys) - y)

                    polygon = [(int(p[0]), int(p[1])) for p in pts]

                    # Check if not already detected
                    if not any(b.data == data for b in barcodes):
                        barcodes.append(BarcodeInfo(
                            data=data,
                            type='QRCODE',
                            bbox=(x, y, w, h),
                            polygon=polygon,
                            confidence=1.0,
                        ))
        except (AttributeError, cv2.error):
            pass  # Multi QR detection not available

        return barcodes

    def scan_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        barcode_types: Optional[List[str]] = None,
    ) -> List[BarcodeScanResult]:
        """Scan multiple images for barcodes."""
        return [self.scan(img, barcode_types) for img in images]

    def scan_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = False,
        extensions: Optional[List[str]] = None,
    ) -> List[BarcodeScanResult]:
        """Scan all images in a directory."""
        directory = Path(directory)
        ext_set = set(extensions or ['.jpg', '.jpeg', '.png', '.webp', '.bmp'])

        results = []
        pattern = '**/*' if recursive else '*'

        for img_path in directory.glob(pattern):
            if img_path.suffix.lower() in ext_set:
                try:
                    result = self.scan(img_path)
                    results.append(result)
                except Exception:
                    pass

        return results

    def decode_first(self, image: Union[str, Path, np.ndarray]) -> Optional[str]:
        """Decode the first barcode found in image."""
        result = self.scan(image)
        return result.barcodes[0].data if result.barcodes else None


# Convenience functions
def scan_barcodes(
    image: Union[str, Path, np.ndarray],
    barcode_types: Optional[List[str]] = None,
) -> BarcodeScanResult:
    """Scan image for barcodes and QR codes."""
    scanner = BarcodeScanner()
    return scanner.scan(image, barcode_types)


def decode_qr(image: Union[str, Path, np.ndarray]) -> List[str]:
    """Decode QR codes in image, return data strings."""
    scanner = BarcodeScanner()
    result = scanner.scan(image, barcode_types=['QRCODE', 'QR_CODE'])
    return [b.data for b in result.barcodes]

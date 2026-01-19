"""OCR Engine Module.

Provides optical character recognition capabilities with multiple backend support
(Tesseract, EasyOCR, PaddleOCR, etc.) for text extraction from images.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class OCRError(Exception):
    """Error during OCR processing."""
    pass


@dataclass
class OCRWord:
    """A single recognized word."""
    text: str
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float = 0.0

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'bbox': self.bbox,
            'confidence': self.confidence,
        }


@dataclass
class OCRLine:
    """A line of recognized text."""
    text: str
    words: List[OCRWord] = field(default_factory=list)
    bbox: Optional[Tuple[int, int, int, int]] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'words': [w.to_dict() for w in self.words],
            'bbox': self.bbox,
            'confidence': self.confidence,
        }


@dataclass
class OCRResult:
    """OCR processing result."""
    path: str
    text: str
    lines: List[OCRLine] = field(default_factory=list)
    words: List[OCRWord] = field(default_factory=list)
    confidence: float = 0.0
    language: str = ""
    engine: str = ""
    processing_time: float = 0.0

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def line_count(self) -> int:
        return len(self.lines)

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())

    def get_text_in_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> str:
        """Get text within a specific region."""
        region_words = []
        for word in self.words:
            # Check if word center is in region
            cx = word.x + word.width // 2
            cy = word.y + word.height // 2
            if x <= cx <= x + width and y <= cy <= y + height:
                region_words.append(word)

        # Sort by position
        region_words.sort(key=lambda w: (w.y, w.x))
        return ' '.join(w.text for w in region_words)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path,
            'text': self.text,
            'lines': [l.to_dict() for l in self.lines],
            'word_count': self.word_count,
            'line_count': self.line_count,
            'confidence': self.confidence,
            'language': self.language,
            'engine': self.engine,
        }


class OCREngine:
    """OCR engine with multiple backend support."""

    BACKENDS = ['tesseract', 'easyocr', 'paddleocr', 'basic']
    LANGUAGES = {
        'eng': 'English',
        'fra': 'French',
        'deu': 'German',
        'spa': 'Spanish',
        'ita': 'Italian',
        'por': 'Portuguese',
        'chi_sim': 'Chinese (Simplified)',
        'jpn': 'Japanese',
        'kor': 'Korean',
        'ara': 'Arabic',
    }

    def __init__(
        self,
        backend: str = 'auto',
        language: str = 'eng',
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize OCR engine.

        Args:
            backend: OCR backend ('auto', 'tesseract', 'easyocr', 'paddleocr')
            language: Language code (e.g., 'eng', 'fra', 'deu')
            config: Additional configuration options
        """
        self.language = language
        self.config = config or {}
        self.backend = backend

        # Lazy-loaded engines
        self._tesseract = None
        self._easyocr = None
        self._paddleocr = None

        if backend == 'auto':
            self.backend = self._find_best_backend()

    def _find_best_backend(self) -> str:
        """Find the best available backend."""
        # Try Tesseract
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return 'tesseract'
        except (ImportError, Exception):
            pass

        # Try EasyOCR
        try:
            import easyocr
            return 'easyocr'
        except ImportError:
            pass

        # Try PaddleOCR
        try:
            from paddleocr import PaddleOCR
            return 'paddleocr'
        except ImportError:
            pass

        return 'basic'

    def _get_tesseract(self):
        """Get pytesseract module."""
        if self._tesseract is None:
            try:
                import pytesseract
                self._tesseract = pytesseract
            except ImportError:
                pass
        return self._tesseract

    def _get_easyocr(self):
        """Get EasyOCR reader."""
        if self._easyocr is None:
            try:
                import easyocr
                # Map language codes
                lang_map = {
                    'eng': 'en',
                    'fra': 'fr',
                    'deu': 'de',
                    'spa': 'es',
                    'ita': 'it',
                    'por': 'pt',
                    'chi_sim': 'ch_sim',
                    'jpn': 'ja',
                    'kor': 'ko',
                    'ara': 'ar',
                }
                lang = lang_map.get(self.language, 'en')
                self._easyocr = easyocr.Reader([lang], gpu=False)
            except ImportError:
                pass
        return self._easyocr

    def _get_paddleocr(self):
        """Get PaddleOCR instance."""
        if self._paddleocr is None:
            try:
                from paddleocr import PaddleOCR
                # Map language
                lang_map = {
                    'eng': 'en',
                    'chi_sim': 'ch',
                    'jpn': 'japan',
                    'kor': 'korean',
                    'fra': 'french',
                    'deu': 'german',
                }
                lang = lang_map.get(self.language, 'en')
                self._paddleocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
            except ImportError:
                pass
        return self._paddleocr

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

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
        try:
            import cv2

            # Convert to grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img

            # Apply thresholding if configured
            if self.config.get('threshold', False):
                gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            # Denoise if configured
            if self.config.get('denoise', False):
                gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

            return gray

        except ImportError:
            return img

    def extract(
        self,
        image: Union[str, Path, np.ndarray],
        preprocess: bool = True,
    ) -> OCRResult:
        """
        Extract text from an image.

        Args:
            image: Image path or numpy array
            preprocess: Apply preprocessing

        Returns:
            OCRResult with extracted text
        """
        import time
        start_time = time.time()

        img = self._load_image(image)
        path = str(image) if not isinstance(image, np.ndarray) else "array"

        if preprocess:
            processed = self._preprocess(img)
        else:
            processed = img

        if self.backend == 'tesseract':
            result = self._extract_tesseract(processed, img, path)
        elif self.backend == 'easyocr':
            result = self._extract_easyocr(img, path)
        elif self.backend == 'paddleocr':
            result = self._extract_paddleocr(img, path)
        else:
            result = self._extract_basic(img, path)

        result.processing_time = time.time() - start_time
        return result

    def _extract_tesseract(
        self,
        processed: np.ndarray,
        original: np.ndarray,
        path: str,
    ) -> OCRResult:
        """Extract text using Tesseract."""
        tesseract = self._get_tesseract()
        if tesseract is None:
            return self._extract_basic(original, path)

        try:
            from PIL import Image

            # Get detailed data
            pil_img = Image.fromarray(processed)
            data = tesseract.image_to_data(
                pil_img,
                lang=self.language,
                output_type=tesseract.Output.DICT,
            )

            # Build result
            words = []
            lines = []
            current_line = []
            current_line_num = -1

            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if not text:
                    continue

                conf = float(data['conf'][i])
                if conf < 0:
                    conf = 0

                word = OCRWord(
                    text=text,
                    bbox=(
                        data['left'][i],
                        data['top'][i],
                        data['width'][i],
                        data['height'][i],
                    ),
                    confidence=conf / 100,
                )
                words.append(word)

                # Group into lines
                line_num = data['line_num'][i]
                if line_num != current_line_num:
                    if current_line:
                        line_text = ' '.join(w.text for w in current_line)
                        line_conf = sum(w.confidence for w in current_line) / len(current_line)
                        lines.append(OCRLine(
                            text=line_text,
                            words=current_line,
                            confidence=line_conf,
                        ))
                    current_line = []
                    current_line_num = line_num

                current_line.append(word)

            # Don't forget last line
            if current_line:
                line_text = ' '.join(w.text for w in current_line)
                line_conf = sum(w.confidence for w in current_line) / len(current_line)
                lines.append(OCRLine(
                    text=line_text,
                    words=current_line,
                    confidence=line_conf,
                ))

            full_text = '\n'.join(line.text for line in lines)
            avg_conf = sum(w.confidence for w in words) / len(words) if words else 0

            return OCRResult(
                path=path,
                text=full_text,
                lines=lines,
                words=words,
                confidence=avg_conf,
                language=self.language,
                engine='tesseract',
            )

        except Exception as e:
            return OCRResult(
                path=path,
                text="",
                engine='tesseract',
                language=self.language,
            )

    def _extract_easyocr(
        self,
        img: np.ndarray,
        path: str,
    ) -> OCRResult:
        """Extract text using EasyOCR."""
        reader = self._get_easyocr()
        if reader is None:
            return self._extract_basic(img, path)

        try:
            results = reader.readtext(img)

            words = []
            lines = []

            for bbox, text, conf in results:
                # Convert polygon to bbox
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                x = int(min(xs))
                y = int(min(ys))
                w = int(max(xs) - x)
                h = int(max(ys) - y)

                word = OCRWord(
                    text=text,
                    bbox=(x, y, w, h),
                    confidence=conf,
                )
                words.append(word)

                # Each EasyOCR result is typically a line
                lines.append(OCRLine(
                    text=text,
                    words=[word],
                    bbox=(x, y, w, h),
                    confidence=conf,
                ))

            full_text = '\n'.join(line.text for line in lines)
            avg_conf = sum(w.confidence for w in words) / len(words) if words else 0

            return OCRResult(
                path=path,
                text=full_text,
                lines=lines,
                words=words,
                confidence=avg_conf,
                language=self.language,
                engine='easyocr',
            )

        except Exception:
            return self._extract_basic(img, path)

    def _extract_paddleocr(
        self,
        img: np.ndarray,
        path: str,
    ) -> OCRResult:
        """Extract text using PaddleOCR."""
        ocr = self._get_paddleocr()
        if ocr is None:
            return self._extract_basic(img, path)

        try:
            results = ocr.ocr(img, cls=True)

            words = []
            lines = []

            if results and results[0]:
                for result in results[0]:
                    bbox_points, (text, conf) = result

                    # Convert points to bbox
                    xs = [p[0] for p in bbox_points]
                    ys = [p[1] for p in bbox_points]
                    x = int(min(xs))
                    y = int(min(ys))
                    w = int(max(xs) - x)
                    h = int(max(ys) - y)

                    word = OCRWord(
                        text=text,
                        bbox=(x, y, w, h),
                        confidence=conf,
                    )
                    words.append(word)

                    lines.append(OCRLine(
                        text=text,
                        words=[word],
                        bbox=(x, y, w, h),
                        confidence=conf,
                    ))

            full_text = '\n'.join(line.text for line in lines)
            avg_conf = sum(w.confidence for w in words) / len(words) if words else 0

            return OCRResult(
                path=path,
                text=full_text,
                lines=lines,
                words=words,
                confidence=avg_conf,
                language=self.language,
                engine='paddleocr',
            )

        except Exception:
            return self._extract_basic(img, path)

    def _extract_basic(
        self,
        img: np.ndarray,
        path: str,
    ) -> OCRResult:
        """Basic OCR fallback - returns empty result."""
        return OCRResult(
            path=path,
            text="",
            lines=[],
            words=[],
            confidence=0,
            language=self.language,
            engine='none',
        )

    def extract_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        preprocess: bool = True,
    ) -> List[OCRResult]:
        """Extract text from multiple images."""
        return [self.extract(img, preprocess) for img in images]


# Convenience functions
def extract_text(
    image: Union[str, Path, np.ndarray],
    language: str = 'eng',
) -> OCRResult:
    """Extract text from an image."""
    engine = OCREngine(language=language)
    return engine.extract(image)


def read_text(
    image: Union[str, Path, np.ndarray],
    language: str = 'eng',
) -> str:
    """Read text from an image (simple string output)."""
    result = extract_text(image, language)
    return result.text

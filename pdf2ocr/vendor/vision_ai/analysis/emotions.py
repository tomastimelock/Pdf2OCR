"""Emotion Analysis Module.

Provides facial emotion recognition and mood analysis capabilities
using deep learning models and computer vision techniques.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import numpy as np


class EmotionAnalysisError(Exception):
    """Error during emotion analysis."""
    pass


class Emotion(str, Enum):
    """Standard emotion categories."""
    ANGRY = "angry"
    DISGUST = "disgust"
    FEAR = "fear"
    HAPPY = "happy"
    SAD = "sad"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"
    CONTEMPT = "contempt"


# Emotion to valence/arousal mapping
EMOTION_VALENCE = {
    Emotion.HAPPY: (0.8, 0.6),      # High valence, medium-high arousal
    Emotion.SURPRISE: (0.3, 0.8),    # Neutral valence, high arousal
    Emotion.NEUTRAL: (0.5, 0.2),     # Neutral valence, low arousal
    Emotion.SAD: (0.2, 0.3),         # Low valence, low-medium arousal
    Emotion.FEAR: (0.2, 0.8),        # Low valence, high arousal
    Emotion.ANGRY: (0.2, 0.7),       # Low valence, high arousal
    Emotion.DISGUST: (0.2, 0.5),     # Low valence, medium arousal
    Emotion.CONTEMPT: (0.3, 0.4),    # Low valence, medium arousal
}


@dataclass
class EmotionScore:
    """Score for a single emotion."""
    emotion: str
    score: float
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'emotion': self.emotion,
            'score': self.score,
            'confidence': self.confidence,
        }


@dataclass
class EmotionResult:
    """Emotion analysis result for a face."""
    dominant_emotion: str
    confidence: float
    emotions: Dict[str, float] = field(default_factory=dict)
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    valence: float = 0.5  # Positive/negative (-1 to 1)
    arousal: float = 0.5  # Calm/excited (-1 to 1)

    @property
    def is_positive(self) -> bool:
        return self.dominant_emotion in ('happy', 'surprise')

    @property
    def is_negative(self) -> bool:
        return self.dominant_emotion in ('angry', 'sad', 'fear', 'disgust', 'contempt')

    @property
    def emotion_scores(self) -> List[EmotionScore]:
        return [
            EmotionScore(emotion=k, score=v, confidence=v)
            for k, v in sorted(self.emotions.items(), key=lambda x: x[1], reverse=True)
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'dominant_emotion': self.dominant_emotion,
            'confidence': self.confidence,
            'emotions': self.emotions,
            'face_bbox': self.face_bbox,
            'valence': self.valence,
            'arousal': self.arousal,
            'is_positive': self.is_positive,
        }


class EmotionAnalyzer:
    """Facial emotion analysis with multiple backend support."""

    BACKENDS = ['fer', 'deepface', 'hsemotion', 'basic']
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def __init__(
        self,
        backend: str = 'auto',
        model_path: Optional[str] = None,
        min_face_confidence: float = 0.5,
    ):
        """
        Initialize emotion analyzer.

        Args:
            backend: Analysis backend ('auto', 'fer', 'deepface', 'hsemotion', 'basic')
            model_path: Path to custom model file
            min_face_confidence: Minimum face detection confidence
        """
        self.model_path = model_path
        self.min_face_confidence = min_face_confidence
        self.backend = backend

        # Lazy-loaded models
        self._fer = None
        self._deepface = None
        self._hsemotion = None
        self._face_detector = None

        if backend == 'auto':
            self.backend = self._find_best_backend()

    def _find_best_backend(self) -> str:
        """Find the best available backend."""
        # Try FER
        try:
            from fer import FER
            return 'fer'
        except ImportError:
            pass

        # Try DeepFace
        try:
            from deepface import DeepFace
            return 'deepface'
        except ImportError:
            pass

        # Try HSEmotion
        try:
            from hsemotion.facial_emotions import HSEmotionRecognizer
            return 'hsemotion'
        except ImportError:
            pass

        return 'basic'

    def _get_fer(self):
        """Get FER emotion detector."""
        if self._fer is None:
            try:
                from fer import FER
                self._fer = FER(mtcnn=True)
            except ImportError:
                pass
        return self._fer

    def _get_deepface(self):
        """Get DeepFace module."""
        if self._deepface is None:
            try:
                from deepface import DeepFace
                self._deepface = DeepFace
            except ImportError:
                pass
        return self._deepface

    def _get_hsemotion(self):
        """Get HSEmotion recognizer."""
        if self._hsemotion is None:
            try:
                from hsemotion.facial_emotions import HSEmotionRecognizer
                model_name = 'enet_b0_8_best_afew'
                self._hsemotion = HSEmotionRecognizer(model_name=model_name)
            except ImportError:
                pass
        return self._hsemotion

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

    def analyze(
        self,
        image: Union[str, Path, np.ndarray],
        detect_faces: bool = True,
    ) -> List[EmotionResult]:
        """
        Analyze emotions in an image.

        Args:
            image: Image path or numpy array
            detect_faces: Whether to detect faces first

        Returns:
            List of EmotionResult objects (one per face)
        """
        img = self._load_image(image)

        if self.backend == 'fer':
            return self._analyze_fer(img)
        elif self.backend == 'deepface':
            return self._analyze_deepface(img)
        elif self.backend == 'hsemotion':
            return self._analyze_hsemotion(img)
        else:
            return self._analyze_basic(img)

    def _analyze_fer(self, img: np.ndarray) -> List[EmotionResult]:
        """Analyze emotions using FER."""
        fer = self._get_fer()
        if fer is None:
            return self._analyze_basic(img)

        results = fer.detect_emotions(img)
        emotion_results = []

        for result in results:
            emotions = result['emotions']
            bbox = result['box']  # x, y, w, h

            # Get dominant emotion
            dominant = max(emotions, key=emotions.get)
            confidence = emotions[dominant]

            # Calculate valence and arousal
            emotion_enum = Emotion(dominant) if dominant in [e.value for e in Emotion] else Emotion.NEUTRAL
            valence, arousal = EMOTION_VALENCE.get(emotion_enum, (0.5, 0.5))

            emotion_results.append(EmotionResult(
                dominant_emotion=dominant,
                confidence=confidence,
                emotions=emotions,
                face_bbox=tuple(bbox) if bbox else None,
                valence=valence,
                arousal=arousal,
            ))

        return emotion_results

    def _analyze_deepface(self, img: np.ndarray) -> List[EmotionResult]:
        """Analyze emotions using DeepFace."""
        deepface = self._get_deepface()
        if deepface is None:
            return self._analyze_basic(img)

        try:
            results = deepface.analyze(
                img,
                actions=['emotion'],
                enforce_detection=False,
                silent=True,
            )

            if not isinstance(results, list):
                results = [results]

            emotion_results = []

            for result in results:
                emotions = result.get('emotion', {})
                region = result.get('region', {})

                if not emotions:
                    continue

                # Normalize scores
                total = sum(emotions.values())
                if total > 0:
                    emotions = {k: v / total for k, v in emotions.items()}

                # Get dominant emotion
                dominant = max(emotions, key=emotions.get)
                confidence = emotions[dominant]

                # Get bbox
                bbox = None
                if region:
                    bbox = (region.get('x', 0), region.get('y', 0),
                           region.get('w', 0), region.get('h', 0))

                # Calculate valence and arousal
                emotion_enum = Emotion(dominant.lower()) if dominant.lower() in [e.value for e in Emotion] else Emotion.NEUTRAL
                valence, arousal = EMOTION_VALENCE.get(emotion_enum, (0.5, 0.5))

                emotion_results.append(EmotionResult(
                    dominant_emotion=dominant.lower(),
                    confidence=confidence,
                    emotions={k.lower(): v for k, v in emotions.items()},
                    face_bbox=bbox,
                    valence=valence,
                    arousal=arousal,
                ))

            return emotion_results

        except Exception:
            return self._analyze_basic(img)

    def _analyze_hsemotion(self, img: np.ndarray) -> List[EmotionResult]:
        """Analyze emotions using HSEmotion."""
        hsemotion = self._get_hsemotion()
        if hsemotion is None:
            return self._analyze_basic(img)

        try:
            # Detect faces first
            from ..detection.faces import FaceDetector
            face_detector = FaceDetector()
            faces = face_detector.detect(img)

            emotion_results = []

            for face in faces:
                x, y, w, h = face.bbox

                # Extract face region
                face_img = img[y:y+h, x:x+w]
                if face_img.size == 0:
                    continue

                # Analyze emotion
                emotion, scores = hsemotion.predict_emotions(face_img, logits=True)

                # Map to standard emotions
                emotion_map = {
                    'Anger': 'angry',
                    'Contempt': 'contempt',
                    'Disgust': 'disgust',
                    'Fear': 'fear',
                    'Happiness': 'happy',
                    'Neutral': 'neutral',
                    'Sadness': 'sad',
                    'Surprise': 'surprise',
                }

                mapped_emotion = emotion_map.get(emotion, emotion.lower())
                emotions_dict = {emotion_map.get(k, k.lower()): v for k, v in scores.items()}

                emotion_enum = Emotion(mapped_emotion) if mapped_emotion in [e.value for e in Emotion] else Emotion.NEUTRAL
                valence, arousal = EMOTION_VALENCE.get(emotion_enum, (0.5, 0.5))

                emotion_results.append(EmotionResult(
                    dominant_emotion=mapped_emotion,
                    confidence=emotions_dict.get(mapped_emotion, 0.0),
                    emotions=emotions_dict,
                    face_bbox=(x, y, w, h),
                    valence=valence,
                    arousal=arousal,
                ))

            return emotion_results

        except Exception:
            return self._analyze_basic(img)

    def _analyze_basic(self, img: np.ndarray) -> List[EmotionResult]:
        """Basic emotion analysis fallback - returns neutral."""
        # Without a proper model, we can only return a default
        return [EmotionResult(
            dominant_emotion='neutral',
            confidence=0.5,
            emotions={'neutral': 0.5},
            valence=0.5,
            arousal=0.2,
        )]

    def get_overall_mood(self, image: Union[str, Path, np.ndarray]) -> str:
        """Get overall mood/atmosphere of an image."""
        results = self.analyze(image)

        if not results:
            return "neutral"

        # Aggregate emotions
        emotion_totals = {}
        for result in results:
            for emotion, score in result.emotions.items():
                emotion_totals[emotion] = emotion_totals.get(emotion, 0) + score

        # Get dominant overall emotion
        if emotion_totals:
            return max(emotion_totals, key=emotion_totals.get)

        return "neutral"


# Convenience functions
def analyze_emotions(
    image: Union[str, Path, np.ndarray],
) -> List[EmotionResult]:
    """Analyze emotions in an image."""
    analyzer = EmotionAnalyzer()
    return analyzer.analyze(image)


def detect_mood(image: Union[str, Path, np.ndarray]) -> str:
    """Detect overall mood of an image."""
    analyzer = EmotionAnalyzer()
    return analyzer.get_overall_mood(image)

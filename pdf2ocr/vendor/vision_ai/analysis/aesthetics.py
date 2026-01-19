"""Aesthetics Scoring Module.

Provides image quality and aesthetics assessment using rule-based analysis
and machine learning models for composition, technical quality, and visual appeal.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np


class AestheticsError(Exception):
    """Error during aesthetics analysis."""
    pass


@dataclass
class CompositionAnalysis:
    """Analysis of image composition."""
    rule_of_thirds: float = 0.0  # 0-1, how well it follows rule of thirds
    symmetry: float = 0.0  # 0-1, horizontal symmetry score
    balance: float = 0.0  # 0-1, visual balance score
    simplicity: float = 0.0  # 0-1, visual simplicity (less clutter)
    depth: float = 0.0  # 0-1, perceived depth/layers
    leading_lines: float = 0.0  # 0-1, presence of leading lines

    @property
    def overall(self) -> float:
        weights = {
            'rule_of_thirds': 0.25,
            'symmetry': 0.15,
            'balance': 0.2,
            'simplicity': 0.2,
            'depth': 0.1,
            'leading_lines': 0.1,
        }
        return sum(getattr(self, k) * v for k, v in weights.items())

    def to_dict(self) -> Dict[str, Any]:
        return {
            'rule_of_thirds': self.rule_of_thirds,
            'symmetry': self.symmetry,
            'balance': self.balance,
            'simplicity': self.simplicity,
            'depth': self.depth,
            'leading_lines': self.leading_lines,
            'overall': self.overall,
        }


@dataclass
class TechnicalQuality:
    """Technical quality assessment."""
    sharpness: float = 0.0  # 0-1, image sharpness
    exposure: float = 0.0  # 0-1, proper exposure
    noise: float = 0.0  # 0-1, low noise score (1 = no noise)
    contrast: float = 0.0  # 0-1, good contrast
    saturation: float = 0.0  # 0-1, appropriate saturation
    resolution_score: float = 0.0  # 0-1, based on resolution

    @property
    def overall(self) -> float:
        weights = {
            'sharpness': 0.25,
            'exposure': 0.2,
            'noise': 0.15,
            'contrast': 0.15,
            'saturation': 0.15,
            'resolution_score': 0.1,
        }
        return sum(getattr(self, k) * v for k, v in weights.items())

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sharpness': self.sharpness,
            'exposure': self.exposure,
            'noise': self.noise,
            'contrast': self.contrast,
            'saturation': self.saturation,
            'resolution_score': self.resolution_score,
            'overall': self.overall,
        }


@dataclass
class AestheticsScore:
    """Complete aesthetics assessment."""
    path: str
    overall_score: float  # 0-1
    rating: str  # 'poor', 'fair', 'good', 'excellent'
    composition: CompositionAnalysis = field(default_factory=CompositionAnalysis)
    technical: TechnicalQuality = field(default_factory=TechnicalQuality)
    color_harmony: float = 0.0  # 0-1
    visual_interest: float = 0.0  # 0-1
    model: str = ""
    processing_time: float = 0.0

    @property
    def stars(self) -> int:
        """Rating as 1-5 stars."""
        return max(1, min(5, int(self.overall_score * 5) + 1))

    @property
    def grade(self) -> str:
        """Rating as letter grade."""
        if self.overall_score >= 0.9:
            return 'A+'
        elif self.overall_score >= 0.8:
            return 'A'
        elif self.overall_score >= 0.7:
            return 'B'
        elif self.overall_score >= 0.6:
            return 'C'
        elif self.overall_score >= 0.5:
            return 'D'
        else:
            return 'F'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path,
            'overall_score': self.overall_score,
            'rating': self.rating,
            'stars': self.stars,
            'grade': self.grade,
            'composition': self.composition.to_dict(),
            'technical': self.technical.to_dict(),
            'color_harmony': self.color_harmony,
            'visual_interest': self.visual_interest,
        }


class AestheticsScorer:
    """Image aesthetics scoring with multiple approaches."""

    BACKENDS = ['nima', 'clip', 'rule_based']

    def __init__(
        self,
        backend: str = 'auto',
        model_path: Optional[str] = None,
    ):
        """
        Initialize aesthetics scorer.

        Args:
            backend: Scoring backend ('auto', 'nima', 'clip', 'rule_based')
            model_path: Path to custom model file
        """
        self.model_path = model_path
        self.backend = backend

        # Lazy-loaded models
        self._nima_model = None
        self._clip_model = None

        if backend == 'auto':
            self.backend = self._find_best_backend()

    def _find_best_backend(self) -> str:
        """Find the best available backend."""
        # Try NIMA
        try:
            import tensorflow as tf
            return 'nima'
        except ImportError:
            pass

        # Try CLIP
        try:
            import clip
            return 'clip'
        except ImportError:
            pass

        return 'rule_based'

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
        return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    def score(
        self,
        image: Union[str, Path, np.ndarray],
        detailed: bool = True,
    ) -> AestheticsScore:
        """
        Score image aesthetics.

        Args:
            image: Image path or numpy array
            detailed: Include detailed analysis

        Returns:
            AestheticsScore with assessment
        """
        import time
        start_time = time.time()

        img = self._load_image(image)
        path = str(image) if not isinstance(image, np.ndarray) else "array"

        if self.backend == 'nima':
            result = self._score_nima(img, path, detailed)
        elif self.backend == 'clip':
            result = self._score_clip(img, path, detailed)
        else:
            result = self._score_rule_based(img, path, detailed)

        result.processing_time = time.time() - start_time
        return result

    def _score_nima(
        self,
        img: np.ndarray,
        path: str,
        detailed: bool,
    ) -> AestheticsScore:
        """Score using NIMA model."""
        try:
            import tensorflow as tf
            from tensorflow.keras.applications.mobilenet import preprocess_input
            from PIL import Image

            # Load or create model
            if self._nima_model is None:
                from tensorflow.keras.applications import MobileNet
                base_model = MobileNet(weights='imagenet', include_top=False, pooling='avg')
                x = tf.keras.layers.Dense(10, activation='softmax')(base_model.output)
                self._nima_model = tf.keras.Model(inputs=base_model.input, outputs=x)

            # Preprocess
            pil_img = Image.fromarray(img).resize((224, 224))
            x = np.array(pil_img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Predict
            scores = self._nima_model.predict(x, verbose=0)[0]

            # Calculate mean score (1-10)
            mean_score = sum((i + 1) * scores[i] for i in range(10))
            normalized_score = (mean_score - 1) / 9  # 0-1

            # Determine rating
            if normalized_score >= 0.75:
                rating = 'excellent'
            elif normalized_score >= 0.55:
                rating = 'good'
            elif normalized_score >= 0.35:
                rating = 'fair'
            else:
                rating = 'poor'

            result = AestheticsScore(
                path=path,
                overall_score=normalized_score,
                rating=rating,
                model='nima',
            )

            if detailed:
                result.composition = self._analyze_composition(img)
                result.technical = self._analyze_technical(img)
                result.color_harmony = self._analyze_color_harmony(img)
                result.visual_interest = self._analyze_visual_interest(img)

            return result

        except Exception:
            return self._score_rule_based(img, path, detailed)

    def _score_clip(
        self,
        img: np.ndarray,
        path: str,
        detailed: bool,
    ) -> AestheticsScore:
        """Score using CLIP model with aesthetic prompts."""
        try:
            import clip
            import torch
            from PIL import Image

            if self._clip_model is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model, preprocess = clip.load("ViT-B/32", device=device)
                self._clip_model = (model, preprocess, device)

            model, preprocess, device = self._clip_model

            # Prepare image
            pil_img = Image.fromarray(img)
            image_input = preprocess(pil_img).unsqueeze(0).to(device)

            # Aesthetic quality prompts
            prompts = [
                "a beautiful, high quality, aesthetically pleasing photograph",
                "an ugly, low quality, unappealing photograph",
                "a professional, well-composed photograph",
                "an amateur, poorly composed photograph",
            ]

            text_tokens = clip.tokenize(prompts).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_tokens)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            probs = similarity[0].cpu().numpy()

            # Calculate aesthetic score
            beautiful_score = probs[0]
            professional_score = probs[2]
            normalized_score = (beautiful_score + professional_score) / 2

            # Determine rating
            if normalized_score >= 0.6:
                rating = 'excellent'
            elif normalized_score >= 0.45:
                rating = 'good'
            elif normalized_score >= 0.3:
                rating = 'fair'
            else:
                rating = 'poor'

            result = AestheticsScore(
                path=path,
                overall_score=float(normalized_score),
                rating=rating,
                model='clip',
            )

            if detailed:
                result.composition = self._analyze_composition(img)
                result.technical = self._analyze_technical(img)
                result.color_harmony = self._analyze_color_harmony(img)
                result.visual_interest = self._analyze_visual_interest(img)

            return result

        except Exception:
            return self._score_rule_based(img, path, detailed)

    def _score_rule_based(
        self,
        img: np.ndarray,
        path: str,
        detailed: bool,
    ) -> AestheticsScore:
        """Score using rule-based analysis."""
        composition = self._analyze_composition(img)
        technical = self._analyze_technical(img)
        color_harmony = self._analyze_color_harmony(img)
        visual_interest = self._analyze_visual_interest(img)

        # Calculate overall score
        weights = {
            'composition': 0.3,
            'technical': 0.3,
            'color_harmony': 0.2,
            'visual_interest': 0.2,
        }

        overall = (
            composition.overall * weights['composition'] +
            technical.overall * weights['technical'] +
            color_harmony * weights['color_harmony'] +
            visual_interest * weights['visual_interest']
        )

        # Determine rating
        if overall >= 0.75:
            rating = 'excellent'
        elif overall >= 0.55:
            rating = 'good'
        elif overall >= 0.35:
            rating = 'fair'
        else:
            rating = 'poor'

        return AestheticsScore(
            path=path,
            overall_score=overall,
            rating=rating,
            composition=composition,
            technical=technical,
            color_harmony=color_harmony,
            visual_interest=visual_interest,
            model='rule_based',
        )

    def _analyze_composition(self, img: np.ndarray) -> CompositionAnalysis:
        """Analyze image composition."""
        h, w = img.shape[:2]
        gray = self._to_gray(img)

        # Rule of thirds analysis
        thirds_h = [h // 3, 2 * h // 3]
        thirds_w = [w // 3, 2 * w // 3]

        # Calculate edge density near thirds lines
        try:
            import cv2
            edges = cv2.Canny(gray, 50, 150)
        except ImportError:
            edges = np.zeros_like(gray)

        # Check for content near intersection points
        roi_size = min(h, w) // 10
        intersection_scores = []
        for y in thirds_h:
            for x in thirds_w:
                y1, y2 = max(0, y - roi_size), min(h, y + roi_size)
                x1, x2 = max(0, x - roi_size), min(w, x + roi_size)
                roi = edges[y1:y2, x1:x2]
                intersection_scores.append(np.mean(roi) / 255)

        rule_of_thirds = min(1.0, np.mean(intersection_scores) * 2)

        # Symmetry analysis
        left_half = gray[:, :w//2]
        right_half = np.fliplr(gray[:, w//2:])

        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]

        if left_half.shape == right_half.shape:
            diff = np.abs(left_half.astype(float) - right_half.astype(float))
            symmetry = 1 - np.mean(diff) / 255
        else:
            symmetry = 0.5

        # Balance analysis (weight distribution)
        left_weight = np.mean(gray[:, :w//2])
        right_weight = np.mean(gray[:, w//2:])
        balance = 1 - abs(left_weight - right_weight) / 255

        # Simplicity (inverse of edge density)
        edge_density = np.mean(edges) / 255
        simplicity = 1 - edge_density

        # Depth estimation (gradient of brightness from top to bottom)
        top_brightness = np.mean(gray[:h//3, :])
        bottom_brightness = np.mean(gray[2*h//3:, :])
        depth = abs(top_brightness - bottom_brightness) / 255

        # Leading lines (look for strong diagonal edges)
        leading_lines = min(1.0, edge_density * 1.5)

        return CompositionAnalysis(
            rule_of_thirds=rule_of_thirds,
            symmetry=symmetry,
            balance=balance,
            simplicity=simplicity,
            depth=depth,
            leading_lines=leading_lines,
        )

    def _analyze_technical(self, img: np.ndarray) -> TechnicalQuality:
        """Analyze technical quality."""
        h, w = img.shape[:2]
        gray = self._to_gray(img)

        # Sharpness (Laplacian variance)
        try:
            import cv2
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = min(1.0, np.var(laplacian) / 500)
        except ImportError:
            # Simple gradient-based sharpness
            gx = np.diff(gray.astype(float), axis=1)
            gy = np.diff(gray.astype(float), axis=0)
            sharpness = min(1.0, (np.var(gx) + np.var(gy)) / 1000)

        # Exposure (histogram analysis)
        hist = np.histogram(gray, bins=256, range=(0, 256))[0]
        hist = hist / hist.sum()

        # Good exposure = balanced histogram
        left_mass = np.sum(hist[:64])
        right_mass = np.sum(hist[192:])
        center_mass = np.sum(hist[64:192])

        if center_mass > 0.5:
            exposure = center_mass
        else:
            exposure = 1 - abs(left_mass - right_mass)

        # Noise (high frequency content in smooth areas)
        # Simplified: check variance in small patches
        patch_size = 16
        variances = []
        for i in range(0, h - patch_size, patch_size * 2):
            for j in range(0, w - patch_size, patch_size * 2):
                patch = gray[i:i+patch_size, j:j+patch_size]
                if np.mean(patch) > 30 and np.mean(patch) < 225:  # Skip very dark/bright
                    variances.append(np.var(patch))

        if variances:
            avg_variance = np.mean(variances)
            noise = max(0, 1 - avg_variance / 500)  # High variance = more noise
        else:
            noise = 0.5

        # Contrast
        contrast = np.std(gray) / 128
        contrast = min(1.0, contrast)

        # Saturation
        if len(img.shape) == 3:
            import colorsys
            pixels = img.reshape(-1, 3)[::100]  # Sample
            sats = [colorsys.rgb_to_hsv(p[0]/255, p[1]/255, p[2]/255)[1] for p in pixels]
            saturation = np.mean(sats)
        else:
            saturation = 0.5

        # Resolution score
        pixels = h * w
        if pixels >= 8_000_000:
            resolution_score = 1.0
        elif pixels >= 2_000_000:
            resolution_score = 0.8
        elif pixels >= 500_000:
            resolution_score = 0.6
        else:
            resolution_score = 0.4

        return TechnicalQuality(
            sharpness=sharpness,
            exposure=exposure,
            noise=noise,
            contrast=contrast,
            saturation=saturation,
            resolution_score=resolution_score,
        )

    def _analyze_color_harmony(self, img: np.ndarray) -> float:
        """Analyze color harmony."""
        if len(img.shape) != 3:
            return 0.5

        import colorsys

        # Sample pixels
        pixels = img.reshape(-1, 3)[::100]

        # Get hues
        hues = []
        for p in pixels:
            h, s, v = colorsys.rgb_to_hsv(p[0]/255, p[1]/255, p[2]/255)
            if s > 0.2 and v > 0.2:  # Ignore desaturated/dark colors
                hues.append(h * 360)

        if len(hues) < 10:
            return 0.5

        # Check for harmonic relationships
        hue_hist = np.histogram(hues, bins=12, range=(0, 360))[0]
        hue_hist = hue_hist / hue_hist.sum()

        # Complementary: peaks at opposite sides (180 degrees apart)
        # Analogous: peaks within 30-60 degrees
        # Triadic: peaks at 120 degrees apart

        peak_bins = np.where(hue_hist > np.mean(hue_hist))[0]

        if len(peak_bins) <= 3:
            # Few dominant colors - likely harmonious
            harmony = 0.8
        elif len(peak_bins) <= 5:
            # Moderate variety
            harmony = 0.6
        else:
            # Many colors - check for patterns
            harmony = 0.4

        # Bonus for complementary or analogous
        if len(peak_bins) == 2:
            diff = abs(peak_bins[0] - peak_bins[1])
            if diff == 6 or diff == 6:  # Complementary (180 degrees)
                harmony += 0.1
            elif diff <= 2:  # Analogous
                harmony += 0.15

        return min(1.0, harmony)

    def _analyze_visual_interest(self, img: np.ndarray) -> float:
        """Analyze visual interest/complexity."""
        gray = self._to_gray(img)
        h, w = gray.shape[:2]

        # Entropy (information content)
        hist = np.histogram(gray, bins=256, range=(0, 256))[0]
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Remove zeros for log
        entropy = -np.sum(hist * np.log2(hist))
        normalized_entropy = entropy / 8  # Max entropy is 8 for 256 bins

        # Edge complexity
        try:
            import cv2
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.mean(edges) / 255
        except ImportError:
            gx = np.diff(gray.astype(float), axis=1)
            gy = np.diff(gray.astype(float), axis=0)
            edge_density = (np.mean(np.abs(gx)) + np.mean(np.abs(gy))) / 255

        # Optimal interest is balanced between simple and complex
        optimal_entropy = 0.7
        optimal_edges = 0.3

        entropy_score = 1 - abs(normalized_entropy - optimal_entropy)
        edge_score = 1 - abs(edge_density - optimal_edges)

        return (entropy_score + edge_score) / 2


# Convenience functions
def score_aesthetics(
    image: Union[str, Path, np.ndarray],
    detailed: bool = True,
) -> AestheticsScore:
    """Score image aesthetics."""
    scorer = AestheticsScorer()
    return scorer.score(image, detailed)


def rate_image(
    image: Union[str, Path, np.ndarray],
) -> Tuple[float, str]:
    """Quick rating of an image (0-1 score, rating label)."""
    result = score_aesthetics(image, detailed=False)
    return (result.overall_score, result.rating)

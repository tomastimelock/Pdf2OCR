"""Scene Classification Module.

Provides scene and place recognition capabilities using deep learning models
trained on scene datasets (Places365, SUN, etc.).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import numpy as np


class SceneClassificationError(Exception):
    """Error during scene classification."""
    pass


class SceneCategory(str, Enum):
    """High-level scene categories."""
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    NATURAL = "natural"
    URBAN = "urban"
    UNKNOWN = "unknown"


# Scene to category mapping
SCENE_CATEGORIES = {
    # Indoor scenes
    'bedroom': SceneCategory.INDOOR,
    'kitchen': SceneCategory.INDOOR,
    'bathroom': SceneCategory.INDOOR,
    'living_room': SceneCategory.INDOOR,
    'office': SceneCategory.INDOOR,
    'restaurant': SceneCategory.INDOOR,
    'classroom': SceneCategory.INDOOR,
    'library': SceneCategory.INDOOR,
    'gym': SceneCategory.INDOOR,
    'hospital': SceneCategory.INDOOR,
    # Outdoor natural
    'beach': SceneCategory.NATURAL,
    'mountain': SceneCategory.NATURAL,
    'forest': SceneCategory.NATURAL,
    'lake': SceneCategory.NATURAL,
    'ocean': SceneCategory.NATURAL,
    'desert': SceneCategory.NATURAL,
    'field': SceneCategory.NATURAL,
    'garden': SceneCategory.NATURAL,
    # Outdoor urban
    'street': SceneCategory.URBAN,
    'city': SceneCategory.URBAN,
    'highway': SceneCategory.URBAN,
    'parking_lot': SceneCategory.URBAN,
    'bridge': SceneCategory.URBAN,
    'building': SceneCategory.URBAN,
}

# Common scene labels for basic classification
COMMON_SCENES = [
    'beach', 'bedroom', 'bridge', 'building', 'city', 'classroom',
    'desert', 'field', 'forest', 'garden', 'gym', 'highway',
    'hospital', 'kitchen', 'lake', 'library', 'living_room',
    'mountain', 'ocean', 'office', 'parking_lot', 'restaurant',
    'street', 'studio', 'sunset', 'waterfall'
]


@dataclass
class SceneInfo:
    """Scene classification result."""
    scene: str
    confidence: float
    category: str = "unknown"
    attributes: List[str] = field(default_factory=list)
    is_indoor: bool = False
    is_outdoor: bool = False
    is_natural: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'scene': self.scene,
            'confidence': self.confidence,
            'category': self.category,
            'attributes': self.attributes,
            'is_indoor': self.is_indoor,
            'is_outdoor': self.is_outdoor,
            'is_natural': self.is_natural,
        }


@dataclass
class SceneClassification:
    """Full scene classification result."""
    path: str
    top_scene: SceneInfo
    all_scenes: List[SceneInfo] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    model: str = ""

    @property
    def scene(self) -> str:
        return self.top_scene.scene

    @property
    def confidence(self) -> float:
        return self.top_scene.confidence

    @property
    def category(self) -> str:
        return self.top_scene.category

    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path,
            'scene': self.scene,
            'confidence': self.confidence,
            'category': self.category,
            'all_scenes': [s.to_dict() for s in self.all_scenes],
            'tags': self.tags,
            'model': self.model,
        }


class SceneClassifier:
    """Scene classification with multiple backend support."""

    BACKENDS = ['places365', 'resnet', 'clip', 'basic']

    def __init__(
        self,
        backend: str = 'auto',
        model_path: Optional[str] = None,
        top_k: int = 5,
    ):
        """
        Initialize scene classifier.

        Args:
            backend: Classification backend ('auto', 'places365', 'resnet', 'clip', 'basic')
            model_path: Path to custom model file
            top_k: Number of top predictions to return
        """
        self.model_path = model_path
        self.top_k = top_k
        self.backend = backend

        # Lazy-loaded models
        self._places_model = None
        self._resnet_model = None
        self._clip_model = None
        self._labels = None

        if backend == 'auto':
            self.backend = self._find_best_backend()

    def _find_best_backend(self) -> str:
        """Find the best available backend."""
        # Try PyTorch with Places365
        try:
            import torch
            import torchvision
            return 'places365'
        except ImportError:
            pass

        # Try CLIP
        try:
            import clip
            return 'clip'
        except ImportError:
            pass

        # Try TensorFlow
        try:
            import tensorflow as tf
            return 'resnet'
        except ImportError:
            pass

        return 'basic'

    def _get_places_model(self):
        """Get Places365 model."""
        if self._places_model is None:
            try:
                import torch
                import torchvision.models as models

                # Use ResNet-50 trained on Places365
                model = models.resnet50(pretrained=False)
                model.fc = torch.nn.Linear(model.fc.in_features, 365)

                # Try to load pretrained weights
                weights_path = self.model_path or 'resnet50_places365.pth.tar'
                if Path(weights_path).exists():
                    checkpoint = torch.load(weights_path, map_location='cpu')
                    model.load_state_dict(checkpoint['state_dict'])

                model.eval()
                self._places_model = model

                # Load labels
                labels_path = 'categories_places365.txt'
                if Path(labels_path).exists():
                    with open(labels_path) as f:
                        self._labels = [line.strip().split(' ')[0][3:] for line in f]
                else:
                    self._labels = COMMON_SCENES

            except Exception:
                pass

        return self._places_model

    def _get_clip_model(self):
        """Get CLIP model for scene classification."""
        if self._clip_model is None:
            try:
                import clip
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model, preprocess = clip.load("ViT-B/32", device=device)
                self._clip_model = (model, preprocess, device)
                self._labels = COMMON_SCENES

            except ImportError:
                pass

        return self._clip_model

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

    def _get_category(self, scene: str) -> Tuple[str, bool, bool, bool]:
        """Get scene category and flags."""
        scene_lower = scene.lower().replace('_', ' ').replace('-', ' ')

        # Check known mappings
        for key, category in SCENE_CATEGORIES.items():
            if key in scene_lower:
                is_indoor = category == SceneCategory.INDOOR
                is_outdoor = category in (SceneCategory.OUTDOOR, SceneCategory.NATURAL, SceneCategory.URBAN)
                is_natural = category == SceneCategory.NATURAL
                return category.value, is_indoor, is_outdoor, is_natural

        return 'unknown', False, False, False

    def classify(
        self,
        image: Union[str, Path, np.ndarray],
        top_k: Optional[int] = None,
    ) -> SceneClassification:
        """
        Classify scene in an image.

        Args:
            image: Image path or numpy array
            top_k: Number of top predictions (overrides init setting)

        Returns:
            SceneClassification with results
        """
        import time
        start_time = time.time()

        img = self._load_image(image)
        path = str(image) if not isinstance(image, np.ndarray) else "array"
        k = top_k or self.top_k

        if self.backend == 'places365':
            scenes = self._classify_places(img, k)
            model_name = 'places365'
        elif self.backend == 'clip':
            scenes = self._classify_clip(img, k)
            model_name = 'clip'
        elif self.backend == 'resnet':
            scenes = self._classify_resnet(img, k)
            model_name = 'resnet'
        else:
            scenes = self._classify_basic(img, k)
            model_name = 'basic'

        # Get tags from top scenes
        tags = list(set(s.scene for s in scenes[:3]))
        for scene in scenes[:3]:
            tags.extend(scene.attributes)
        tags = list(set(tags))

        processing_time = time.time() - start_time

        return SceneClassification(
            path=path,
            top_scene=scenes[0] if scenes else SceneInfo(scene='unknown', confidence=0),
            all_scenes=scenes,
            tags=tags,
            processing_time=processing_time,
            model=model_name,
        )

    def _classify_places(self, img: np.ndarray, top_k: int) -> List[SceneInfo]:
        """Classify using Places365."""
        model = self._get_places_model()
        if model is None:
            return self._classify_basic(img, top_k)

        import torch
        from torchvision import transforms
        from PIL import Image

        # Preprocess
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        pil_img = Image.fromarray(img)
        input_tensor = preprocess(pil_img).unsqueeze(0)

        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)

        # Get top-k
        top_probs, top_indices = probs.topk(top_k)

        scenes = []
        for prob, idx in zip(top_probs.numpy(), top_indices.numpy()):
            scene_name = self._labels[idx] if idx < len(self._labels) else f"scene_{idx}"
            category, is_indoor, is_outdoor, is_natural = self._get_category(scene_name)

            scenes.append(SceneInfo(
                scene=scene_name,
                confidence=float(prob),
                category=category,
                is_indoor=is_indoor,
                is_outdoor=is_outdoor,
                is_natural=is_natural,
            ))

        return scenes

    def _classify_clip(self, img: np.ndarray, top_k: int) -> List[SceneInfo]:
        """Classify using CLIP."""
        clip_data = self._get_clip_model()
        if clip_data is None:
            return self._classify_basic(img, top_k)

        import torch
        import clip
        from PIL import Image

        model, preprocess, device = clip_data

        # Prepare image
        pil_img = Image.fromarray(img)
        image_input = preprocess(pil_img).unsqueeze(0).to(device)

        # Prepare text prompts
        text_prompts = [f"a photo of a {scene}" for scene in self._labels]
        text_tokens = clip.tokenize(text_prompts).to(device)

        # Get similarities
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_tokens)

            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # Get top-k
        probs, indices = similarity[0].topk(top_k)

        scenes = []
        for prob, idx in zip(probs.cpu().numpy(), indices.cpu().numpy()):
            scene_name = self._labels[idx]
            category, is_indoor, is_outdoor, is_natural = self._get_category(scene_name)

            scenes.append(SceneInfo(
                scene=scene_name,
                confidence=float(prob),
                category=category,
                is_indoor=is_indoor,
                is_outdoor=is_outdoor,
                is_natural=is_natural,
            ))

        return scenes

    def _classify_resnet(self, img: np.ndarray, top_k: int) -> List[SceneInfo]:
        """Classify using ResNet (ImageNet)."""
        try:
            import tensorflow as tf
            from tensorflow.keras.applications import ResNet50
            from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
            from tensorflow.keras.preprocessing.image import img_to_array
            from PIL import Image

            model = ResNet50(weights='imagenet')

            # Preprocess
            pil_img = Image.fromarray(img).resize((224, 224))
            x = img_to_array(pil_img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Predict
            preds = model.predict(x, verbose=0)
            decoded = decode_predictions(preds, top=top_k)[0]

            scenes = []
            for _, label, prob in decoded:
                category, is_indoor, is_outdoor, is_natural = self._get_category(label)

                scenes.append(SceneInfo(
                    scene=label.replace('_', ' '),
                    confidence=float(prob),
                    category=category,
                    is_indoor=is_indoor,
                    is_outdoor=is_outdoor,
                    is_natural=is_natural,
                ))

            return scenes

        except Exception:
            return self._classify_basic(img, top_k)

    def _classify_basic(self, img: np.ndarray, top_k: int) -> List[SceneInfo]:
        """Basic scene classification using color/texture heuristics."""
        # Analyze image characteristics
        h, w = img.shape[:2]
        pixels = img.reshape(-1, 3)

        # Calculate averages
        avg_color = np.mean(pixels, axis=0)
        brightness = np.mean(img)

        # Simple heuristics
        is_outdoor = brightness > 120  # Brighter images tend to be outdoor
        is_natural = avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]  # Green dominant

        # Check for blue (sky/water)
        blue_ratio = avg_color[2] / (np.mean(avg_color) + 1)
        has_sky = blue_ratio > 1.1 and brightness > 100

        # Determine scene
        if is_natural:
            if has_sky:
                scene = 'landscape'
            else:
                scene = 'forest'
            category = 'natural'
        elif is_outdoor:
            if has_sky:
                scene = 'city'
            else:
                scene = 'street'
            category = 'urban'
        else:
            scene = 'indoor'
            category = 'indoor'

        return [SceneInfo(
            scene=scene,
            confidence=0.5,
            category=category,
            is_indoor=not is_outdoor,
            is_outdoor=is_outdoor,
            is_natural=is_natural,
        )]


# Convenience functions
def classify_scene(image: Union[str, Path, np.ndarray]) -> SceneClassification:
    """Classify scene in an image."""
    classifier = SceneClassifier()
    return classifier.classify(image)


def get_scene_tags(image: Union[str, Path, np.ndarray], max_tags: int = 5) -> List[str]:
    """Get scene-related tags for an image."""
    result = classify_scene(image)
    return result.tags[:max_tags]

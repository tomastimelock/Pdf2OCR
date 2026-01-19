"""Media Intelligence Analyzer Module.

Provides high-level AI-powered image analysis including description generation,
tag suggestion, similarity search, and comprehensive media understanding.
"""

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class AnalyzerError(Exception):
    """Error during media analysis."""
    pass


@dataclass
class ImageDescription:
    """AI-generated image description."""
    short_description: str = ""
    detailed_description: str = ""
    mood: str = ""
    style: str = ""
    setting: str = ""
    subjects: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'short': self.short_description,
            'detailed': self.detailed_description,
            'mood': self.mood,
            'style': self.style,
            'setting': self.setting,
            'subjects': self.subjects,
            'actions': self.actions,
        }


@dataclass
class SimilarMatch:
    """Similar image match result."""
    path: str
    similarity: float
    match_type: str = "visual"  # visual, semantic, color
    shared_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path,
            'similarity': self.similarity,
            'match_type': self.match_type,
            'shared_tags': self.shared_tags,
        }


@dataclass
class AnalysisResult:
    """Comprehensive media analysis result."""
    path: str
    description: ImageDescription = field(default_factory=ImageDescription)
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    colors: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path,
            'description': self.description.to_dict(),
            'tags': self.tags,
            'categories': self.categories,
            'objects': self.objects,
            'colors': self.colors,
            'metadata': self.metadata,
        }


class MediaAnalyzer:
    """High-level media intelligence analyzer."""

    BACKENDS = ['openai', 'anthropic', 'clip', 'local']

    def __init__(
        self,
        backend: str = 'auto',
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize media analyzer.

        Args:
            backend: Analysis backend ('auto', 'openai', 'anthropic', 'clip', 'local')
            api_key: API key for cloud services
            model: Specific model to use
        """
        self.api_key = api_key
        self.model = model
        self.backend = backend

        # Lazy-loaded components
        self._openai_client = None
        self._anthropic_client = None
        self._clip_model = None
        self._local_model = None

        # Feature cache for similarity search
        self._embedding_cache: Dict[str, np.ndarray] = {}

        if backend == 'auto':
            self.backend = self._find_best_backend()

    def _find_best_backend(self) -> str:
        """Find the best available backend."""
        # Try OpenAI
        try:
            from openai import OpenAI
            return 'openai'
        except ImportError:
            pass

        # Try Anthropic
        try:
            from anthropic import Anthropic
            return 'anthropic'
        except ImportError:
            pass

        # Try CLIP
        try:
            import clip
            return 'clip'
        except ImportError:
            pass

        return 'local'

    def _get_openai_client(self):
        """Get OpenAI client."""
        if self._openai_client is None:
            try:
                from openai import OpenAI
                if self.api_key:
                    self._openai_client = OpenAI(api_key=self.api_key)
                else:
                    self._openai_client = OpenAI()
            except ImportError:
                pass
        return self._openai_client

    def _get_anthropic_client(self):
        """Get Anthropic client."""
        if self._anthropic_client is None:
            try:
                from anthropic import Anthropic
                if self.api_key:
                    self._anthropic_client = Anthropic(api_key=self.api_key)
                else:
                    self._anthropic_client = Anthropic()
            except ImportError:
                pass
        return self._anthropic_client

    def _get_clip_model(self):
        """Get CLIP model."""
        if self._clip_model is None:
            try:
                import clip
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model, preprocess = clip.load("ViT-B/32", device=device)
                self._clip_model = (model, preprocess, device)
            except ImportError:
                pass
        return self._clip_model

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

    def _encode_image_base64(self, image: Union[str, Path, np.ndarray]) -> str:
        """Encode image to base64."""
        if isinstance(image, np.ndarray):
            from PIL import Image
            import io
            pil_img = Image.fromarray(image)
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG')
            return base64.standard_b64encode(buffer.getvalue()).decode('utf-8')
        else:
            with open(image, 'rb') as f:
                return base64.standard_b64encode(f.read()).decode('utf-8')

    def analyze(
        self,
        image: Union[str, Path, np.ndarray],
        detailed: bool = True,
    ) -> AnalysisResult:
        """
        Perform comprehensive image analysis.

        Args:
            image: Image path or numpy array
            detailed: Include detailed analysis

        Returns:
            AnalysisResult with comprehensive analysis
        """
        import time
        start_time = time.time()

        path = str(image) if not isinstance(image, np.ndarray) else "array"

        if self.backend == 'openai':
            result = self._analyze_openai(image, path, detailed)
        elif self.backend == 'anthropic':
            result = self._analyze_anthropic(image, path, detailed)
        elif self.backend == 'clip':
            result = self._analyze_clip(image, path, detailed)
        else:
            result = self._analyze_local(image, path, detailed)

        result.processing_time = time.time() - start_time
        return result

    def _analyze_openai(
        self,
        image: Union[str, Path, np.ndarray],
        path: str,
        detailed: bool,
    ) -> AnalysisResult:
        """Analyze using OpenAI Vision API."""
        client = self._get_openai_client()
        if client is None:
            return self._analyze_local(image, path, detailed)

        try:
            base64_image = self._encode_image_base64(image)
            model = self.model or "gpt-4o"

            prompt = """Analyze this image and provide:
1. A short description (1 sentence)
2. A detailed description (2-3 sentences)
3. The mood/atmosphere
4. The visual style
5. The setting/location
6. Main subjects (list)
7. Actions/activities (list)
8. 10 relevant tags
9. Main categories
10. Visible objects
11. Dominant colors

Respond in JSON format with keys: short_description, detailed_description,
mood, style, setting, subjects, actions, tags, categories, objects, colors"""

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
            )

            response_text = response.choices[0].message.content

            # Parse JSON response
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0:
                    data = json.loads(response_text[json_start:json_end])
                else:
                    data = {}
            except json.JSONDecodeError:
                data = {}

            description = ImageDescription(
                short_description=data.get('short_description', ''),
                detailed_description=data.get('detailed_description', ''),
                mood=data.get('mood', ''),
                style=data.get('style', ''),
                setting=data.get('setting', ''),
                subjects=data.get('subjects', []),
                actions=data.get('actions', []),
                confidence=0.9,
            )

            return AnalysisResult(
                path=path,
                description=description,
                tags=data.get('tags', []),
                categories=data.get('categories', []),
                objects=data.get('objects', []),
                colors=data.get('colors', []),
            )

        except Exception:
            return self._analyze_local(image, path, detailed)

    def _analyze_anthropic(
        self,
        image: Union[str, Path, np.ndarray],
        path: str,
        detailed: bool,
    ) -> AnalysisResult:
        """Analyze using Anthropic Claude Vision API."""
        client = self._get_anthropic_client()
        if client is None:
            return self._analyze_local(image, path, detailed)

        try:
            base64_image = self._encode_image_base64(image)
            model = self.model or "claude-sonnet-4-20250514"

            prompt = """Analyze this image and provide a JSON response with:
- short_description: 1 sentence summary
- detailed_description: 2-3 sentence description
- mood: the mood/atmosphere
- style: visual style
- setting: location/setting
- subjects: list of main subjects
- actions: list of activities
- tags: list of 10 relevant tags
- categories: list of categories
- objects: list of visible objects
- colors: list of dominant colors"""

            # Determine media type
            media_type = "image/jpeg"
            if isinstance(image, (str, Path)):
                ext = Path(image).suffix.lower()
                if ext == '.png':
                    media_type = "image/png"
                elif ext == '.webp':
                    media_type = "image/webp"
                elif ext == '.gif':
                    media_type = "image/gif"

            response = client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_image,
                                }
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            )

            response_text = response.content[0].text

            # Parse JSON
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0:
                    data = json.loads(response_text[json_start:json_end])
                else:
                    data = {}
            except json.JSONDecodeError:
                data = {}

            description = ImageDescription(
                short_description=data.get('short_description', ''),
                detailed_description=data.get('detailed_description', ''),
                mood=data.get('mood', ''),
                style=data.get('style', ''),
                setting=data.get('setting', ''),
                subjects=data.get('subjects', []),
                actions=data.get('actions', []),
                confidence=0.9,
            )

            return AnalysisResult(
                path=path,
                description=description,
                tags=data.get('tags', []),
                categories=data.get('categories', []),
                objects=data.get('objects', []),
                colors=data.get('colors', []),
            )

        except Exception:
            return self._analyze_local(image, path, detailed)

    def _analyze_clip(
        self,
        image: Union[str, Path, np.ndarray],
        path: str,
        detailed: bool,
    ) -> AnalysisResult:
        """Analyze using CLIP model."""
        clip_data = self._get_clip_model()
        if clip_data is None:
            return self._analyze_local(image, path, detailed)

        try:
            import clip
            import torch
            from PIL import Image

            model, preprocess, device = clip_data
            img = self._load_image(image)
            pil_img = Image.fromarray(img)
            image_input = preprocess(pil_img).unsqueeze(0).to(device)

            # Define categories to check
            categories = [
                "a photo of a landscape", "a photo of a portrait",
                "a photo of architecture", "a photo of food",
                "a photo of animals", "a photo of nature",
                "a photo of people", "a photo of a cityscape",
                "a photo of art", "a photo of sports",
            ]

            tags = [
                "beautiful", "colorful", "dark", "bright", "vintage",
                "modern", "artistic", "professional", "casual", "dramatic",
                "peaceful", "energetic", "minimalist", "detailed", "abstract",
            ]

            with torch.no_grad():
                # Get image embedding
                image_features = model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Match categories
                cat_tokens = clip.tokenize(categories).to(device)
                cat_features = model.encode_text(cat_tokens)
                cat_features = cat_features / cat_features.norm(dim=-1, keepdim=True)
                cat_similarity = (100.0 * image_features @ cat_features.T).softmax(dim=-1)
                top_cats = cat_similarity[0].topk(3)

                # Match tags
                tag_tokens = clip.tokenize([f"a {t} photo" for t in tags]).to(device)
                tag_features = model.encode_text(tag_tokens)
                tag_features = tag_features / tag_features.norm(dim=-1, keepdim=True)
                tag_similarity = (100.0 * image_features @ tag_features.T).softmax(dim=-1)
                top_tags = tag_similarity[0].topk(5)

            # Extract results
            result_cats = [categories[i].replace("a photo of ", "") for i in top_cats.indices.cpu().numpy()]
            result_tags = [tags[i] for i in top_tags.indices.cpu().numpy()]

            # Store embedding
            embedding = image_features.cpu().numpy()[0]

            return AnalysisResult(
                path=path,
                description=ImageDescription(
                    short_description=f"A {result_tags[0]} {result_cats[0]}",
                    confidence=float(top_cats.values[0]),
                ),
                tags=result_tags,
                categories=result_cats,
                embedding=embedding,
            )

        except Exception:
            return self._analyze_local(image, path, detailed)

    def _analyze_local(
        self,
        image: Union[str, Path, np.ndarray],
        path: str,
        detailed: bool,
    ) -> AnalysisResult:
        """Local analysis using available modules."""
        img = self._load_image(image)

        # Use available analyzers
        tags = []
        categories = []
        objects_list = []
        colors_list = []

        # Try color analysis
        try:
            from ..analysis.colors import ColorAnalyzer
            analyzer = ColorAnalyzer()
            colors = analyzer.get_dominant_colors(img, 5)
            colors_list = [c.name for c in colors]
        except Exception:
            pass

        # Try scene classification
        try:
            from ..analysis.scenes import SceneClassifier
            classifier = SceneClassifier()
            result = classifier.classify(img, top_k=3)
            categories = [s.scene for s in result.all_scenes]
            tags.extend(result.tags)
        except Exception:
            pass

        # Try object detection
        try:
            from ..detection.objects import ObjectDetector
            detector = ObjectDetector()
            result = detector.detect(img)
            objects_list = list(set(obj.label for obj in result.objects))
            tags.extend(objects_list[:5])
        except Exception:
            pass

        return AnalysisResult(
            path=path,
            description=ImageDescription(
                short_description=f"An image containing {', '.join(objects_list[:3]) if objects_list else 'visual content'}",
            ),
            tags=list(set(tags))[:10],
            categories=categories,
            objects=objects_list,
            colors=colors_list,
        )

    def describe(
        self,
        image: Union[str, Path, np.ndarray],
        style: str = 'detailed',
    ) -> str:
        """
        Generate a text description of an image.

        Args:
            image: Image path or numpy array
            style: Description style ('short', 'detailed', 'poetic', 'technical')

        Returns:
            Text description
        """
        result = self.analyze(image)

        if style == 'short':
            return result.description.short_description
        elif style == 'detailed':
            return result.description.detailed_description or result.description.short_description
        elif style == 'poetic':
            mood = result.description.mood or 'evocative'
            return f"A {mood} scene: {result.description.short_description}"
        else:  # technical
            parts = []
            if result.objects:
                parts.append(f"Contains: {', '.join(result.objects[:5])}")
            if result.colors:
                parts.append(f"Colors: {', '.join(result.colors[:3])}")
            if result.categories:
                parts.append(f"Category: {result.categories[0]}")
            return ". ".join(parts) if parts else "Technical analysis unavailable."

    def suggest_tags(
        self,
        image: Union[str, Path, np.ndarray],
        max_tags: int = 10,
    ) -> List[str]:
        """
        Suggest relevant tags for an image.

        Args:
            image: Image path or numpy array
            max_tags: Maximum number of tags

        Returns:
            List of suggested tags
        """
        result = self.analyze(image)
        return result.tags[:max_tags]

    def find_similar(
        self,
        query_image: Union[str, Path, np.ndarray],
        database: List[Union[str, Path]],
        top_k: int = 10,
    ) -> List[SimilarMatch]:
        """
        Find similar images in a database.

        Args:
            query_image: Query image
            database: List of image paths to search
            top_k: Number of results to return

        Returns:
            List of SimilarMatch objects
        """
        # Get query embedding
        query_result = self.analyze(query_image)
        query_embedding = query_result.embedding

        if query_embedding is None:
            # Fall back to feature-based matching
            return self._find_similar_features(query_image, database, top_k)

        # Get embeddings for database
        similarities = []
        for img_path in database:
            path_str = str(img_path)

            if path_str in self._embedding_cache:
                db_embedding = self._embedding_cache[path_str]
            else:
                try:
                    db_result = self.analyze(img_path)
                    if db_result.embedding is not None:
                        db_embedding = db_result.embedding
                        self._embedding_cache[path_str] = db_embedding
                    else:
                        continue
                except Exception:
                    continue

            # Calculate cosine similarity
            similarity = np.dot(query_embedding, db_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(db_embedding)
            )

            similarities.append((path_str, float(similarity)))

        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)

        return [
            SimilarMatch(path=path, similarity=sim, match_type='embedding')
            for path, sim in similarities[:top_k]
        ]

    def _find_similar_features(
        self,
        query_image: Union[str, Path, np.ndarray],
        database: List[Union[str, Path]],
        top_k: int,
    ) -> List[SimilarMatch]:
        """Find similar images using color histograms."""
        query_img = self._load_image(query_image)

        # Calculate query histogram
        query_hist = self._calculate_histogram(query_img)

        similarities = []
        for img_path in database:
            try:
                db_img = self._load_image(img_path)
                db_hist = self._calculate_histogram(db_img)

                # Calculate histogram intersection
                similarity = np.minimum(query_hist, db_hist).sum()
                similarities.append((str(img_path), similarity))
            except Exception:
                continue

        similarities.sort(key=lambda x: x[1], reverse=True)

        return [
            SimilarMatch(path=path, similarity=sim, match_type='histogram')
            for path, sim in similarities[:top_k]
        ]

    def _calculate_histogram(self, img: np.ndarray) -> np.ndarray:
        """Calculate normalized color histogram."""
        # Resize for speed
        try:
            from PIL import Image
            pil_img = Image.fromarray(img).resize((64, 64))
            img = np.array(pil_img)
        except ImportError:
            pass

        # Calculate histogram for each channel
        hist = []
        for i in range(3):
            h, _ = np.histogram(img[:, :, i], bins=16, range=(0, 256))
            hist.extend(h)

        # Normalize
        hist = np.array(hist, dtype=np.float32)
        return hist / hist.sum() if hist.sum() > 0 else hist


# Convenience functions
def analyze_media(
    image: Union[str, Path, np.ndarray],
    detailed: bool = True,
) -> AnalysisResult:
    """Analyze an image comprehensively."""
    analyzer = MediaAnalyzer()
    return analyzer.analyze(image, detailed)


def describe_image(
    image: Union[str, Path, np.ndarray],
    style: str = 'detailed',
) -> str:
    """Generate a description for an image."""
    analyzer = MediaAnalyzer()
    return analyzer.describe(image, style)


def suggest_tags(
    image: Union[str, Path, np.ndarray],
    max_tags: int = 10,
) -> List[str]:
    """Suggest tags for an image."""
    analyzer = MediaAnalyzer()
    return analyzer.suggest_tags(image, max_tags)

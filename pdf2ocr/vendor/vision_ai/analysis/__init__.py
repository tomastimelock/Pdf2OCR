"""Analysis submodule for VisionAI.

Provides visual analysis capabilities for emotions, scenes, colors, content moderation, and aesthetics.
"""

from .emotions import (
    EmotionAnalyzer,
    EmotionResult,
    EmotionScore,
    analyze_emotions,
    detect_mood,
)
from .scenes import (
    SceneClassifier,
    SceneInfo,
    SceneClassification,
    classify_scene,
    get_scene_tags,
)
from .colors import (
    ColorAnalyzer,
    ColorPalette,
    ColorInfo,
    ColorHistogram,
    analyze_colors,
    get_dominant_colors,
    get_palette,
)
from .nsfw import (
    NSFWDetector,
    SafetyResult,
    SafetyCategory,
    check_nsfw,
    is_safe,
    moderate_content,
)
from .aesthetics import (
    AestheticsScorer,
    AestheticsScore,
    CompositionAnalysis,
    score_aesthetics,
    rate_image,
)

__all__ = [
    # Emotions
    "EmotionAnalyzer",
    "EmotionResult",
    "EmotionScore",
    "analyze_emotions",
    "detect_mood",
    # Scenes
    "SceneClassifier",
    "SceneInfo",
    "SceneClassification",
    "classify_scene",
    "get_scene_tags",
    # Colors
    "ColorAnalyzer",
    "ColorPalette",
    "ColorInfo",
    "ColorHistogram",
    "analyze_colors",
    "get_dominant_colors",
    "get_palette",
    # NSFW
    "NSFWDetector",
    "SafetyResult",
    "SafetyCategory",
    "check_nsfw",
    "is_safe",
    "moderate_content",
    # Aesthetics
    "AestheticsScorer",
    "AestheticsScore",
    "CompositionAnalysis",
    "score_aesthetics",
    "rate_image",
]

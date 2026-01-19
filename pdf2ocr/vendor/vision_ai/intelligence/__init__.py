"""Intelligence submodule for VisionAI.

Provides high-level media intelligence capabilities including
AI-powered analysis, description generation, and similarity search.
"""

from .analyzer import (
    MediaAnalyzer,
    AnalysisResult,
    ImageDescription,
    SimilarMatch,
    analyze_media,
    describe_image,
    suggest_tags,
)

__all__ = [
    "MediaAnalyzer",
    "AnalysisResult",
    "ImageDescription",
    "SimilarMatch",
    "analyze_media",
    "describe_image",
    "suggest_tags",
]

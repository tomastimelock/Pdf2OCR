# Filepath: code_migration/ai_providers/openai_vision/provider.py
# Description: OpenAI Vision Provider implementation
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/openai/vision/provider.py

"""
OpenAI Vision Provider
Implements image analysis and vision tasks using OpenAI's GPT-4 Vision API.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from openai import OpenAI

from .model_config import VISION_MODELS, get_vision_model_info
from .utils import encode_image_to_base64, is_url, validate_image_path


class OpenAIVisionProvider:
    """
    Provider for OpenAI Vision/Image Analysis.

    Supports:
    - Single and multi-image analysis
    - Image descriptions
    - Text extraction (OCR)
    - Question answering about images
    - Local files (auto base64 encoding) and URLs

    Example:
        >>> provider = OpenAIVisionProvider(api_key="sk-...")
        >>> result = provider.analyze_image("photo.jpg", "What's in this image?")
        >>> print(result)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        default_detail: str = "auto"
    ):
        """
        Initialize the OpenAI Vision Provider.

        Args:
            api_key: OpenAI API key. If not provided, loads from OPENAI_API_KEY env var.
            model: Model to use. Defaults to "gpt-4o".
            default_detail: Default detail level ("low", "high", "auto"). Defaults to "auto".

        Raises:
            ValueError: If API key is not provided and not in environment.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided or set in OPENAI_API_KEY environment variable"
            )

        self.model = model or os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
        self.default_detail = default_detail
        self.client = OpenAI(api_key=self.api_key)

    @staticmethod
    def list_models() -> Dict[str, Any]:
        """
        List all available vision models with their configurations.

        Returns:
            Dictionary mapping model names to their configuration details.

        Example:
            >>> models = OpenAIVisionProvider.list_models()
            >>> print(models["gpt-4o"]["description"])
            'GPT-4o - fast multimodal model with vision'
        """
        return VISION_MODELS

    @staticmethod
    def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.

        Args:
            model_name: Name of the model (e.g., "gpt-4o")

        Returns:
            Model configuration dictionary or None if not found.
        """
        return get_vision_model_info(model_name)

    def _prepare_image_content(
        self,
        image_path: str,
        detail: str = "auto"
    ) -> Dict[str, Any]:
        """
        Prepare image content for the API.

        Args:
            image_path: Path to local file or URL
            detail: Detail level (low, high, auto)

        Returns:
            Image content dictionary for the API

        Raises:
            FileNotFoundError: If local file doesn't exist
            ValueError: If image format is not supported
        """
        # Check if it's a URL
        if is_url(image_path):
            return {
                "type": "image_url",
                "image_url": {
                    "url": image_path,
                    "detail": detail
                }
            }

        # It's a local file - validate and encode
        validate_image_path(image_path)
        base64_image = encode_image_to_base64(image_path)

        return {
            "type": "image_url",
            "image_url": {
                "url": base64_image,
                "detail": detail
            }
        }

    def analyze_image(
        self,
        image_path_or_url: str,
        prompt: str,
        model: Optional[str] = None,
        detail: str = "auto",
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        Analyze an image with a custom prompt.

        Args:
            image_path_or_url: Path to image file or URL
            prompt: Question or instruction about the image
            model: Model to use (defaults to instance model)
            detail: Detail level (low, high, auto). Defaults to "auto".
            max_tokens: Maximum tokens in response. Defaults to 1024.
            **kwargs: Additional parameters for the API

        Returns:
            The analysis text

        Example:
            >>> provider = OpenAIVisionProvider()
            >>> result = provider.analyze_image(
            ...     "photo.jpg",
            ...     "Describe what you see in detail"
            ... )
        """
        image_content = self._prepare_image_content(image_path_or_url, detail)

        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_content
                ]
            }],
            max_tokens=max_tokens,
            **kwargs
        )

        return response.choices[0].message.content

    def analyze_images(
        self,
        images: List[str],
        prompt: str,
        model: Optional[str] = None,
        detail: str = "auto",
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        Analyze multiple images together.

        Args:
            images: List of image paths or URLs
            prompt: Question or instruction about the images
            model: Model to use (defaults to instance model)
            detail: Detail level (low, high, auto). Defaults to "auto".
            max_tokens: Maximum tokens in response. Defaults to 1024.
            **kwargs: Additional parameters for the API

        Returns:
            The analysis text

        Example:
            >>> provider = OpenAIVisionProvider()
            >>> result = provider.analyze_images(
            ...     ["before.jpg", "after.jpg"],
            ...     "What changed between these images?"
            ... )
        """
        content = [{"type": "text", "text": prompt}]

        for img in images:
            content.append(self._prepare_image_content(img, detail))

        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=[{
                "role": "user",
                "content": content
            }],
            max_tokens=max_tokens,
            **kwargs
        )

        return response.choices[0].message.content

    def describe_image(
        self,
        image: str,
        detail_level: str = "auto",
        model: Optional[str] = None,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        Generate a detailed description of an image.

        Args:
            image: Path to image file or URL
            detail_level: Detail level (low, high, auto). Defaults to "auto".
            model: Model to use (defaults to instance model)
            max_tokens: Maximum tokens in response. Defaults to 1024.
            **kwargs: Additional parameters for the API

        Returns:
            The image description

        Example:
            >>> provider = OpenAIVisionProvider()
            >>> desc = provider.describe_image("landscape.jpg")
            >>> print(desc)
        """
        return self.analyze_image(
            image_path_or_url=image,
            prompt=(
                "Provide a detailed description of this image. "
                "Include all notable elements, colors, composition, mood, "
                "and any text visible. Be thorough and objective."
            ),
            model=model,
            detail=detail_level,
            max_tokens=max_tokens,
            **kwargs
        )

    def extract_text(
        self,
        image: str,
        model: Optional[str] = None,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """
        Extract all visible text from an image (OCR-like functionality).

        Args:
            image: Path to image file or URL
            model: Model to use (defaults to instance model)
            max_tokens: Maximum tokens in response. Defaults to 2048.
            **kwargs: Additional parameters for the API

        Returns:
            Extracted text from the image

        Example:
            >>> provider = OpenAIVisionProvider()
            >>> text = provider.extract_text("document.png")
            >>> print(text)
        """
        return self.analyze_image(
            image_path_or_url=image,
            prompt=(
                "Extract and return all text visible in this image. "
                "Preserve the layout and formatting as much as possible. "
                "Include all text you can see, even if partially obscured. "
                "If there is no text, respond with 'No text found in image.'"
            ),
            model=model,
            detail="high",  # High detail for better OCR accuracy
            max_tokens=max_tokens,
            **kwargs
        )

    def answer_about_image(
        self,
        image: str,
        question: str,
        model: Optional[str] = None,
        detail: str = "auto",
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        Answer a specific question about an image.

        Args:
            image: Path to image file or URL
            question: Question to answer about the image
            model: Model to use (defaults to instance model)
            detail: Detail level (low, high, auto). Defaults to "auto".
            max_tokens: Maximum tokens in response. Defaults to 1024.
            **kwargs: Additional parameters for the API

        Returns:
            Answer to the question

        Example:
            >>> provider = OpenAIVisionProvider()
            >>> answer = provider.answer_about_image(
            ...     "crowd.jpg",
            ...     "How many people are visible?"
            ... )
        """
        return self.analyze_image(
            image_path_or_url=image,
            prompt=question,
            model=model,
            detail=detail,
            max_tokens=max_tokens,
            **kwargs
        )

    def compare_images(
        self,
        image1: str,
        image2: str,
        aspect: Optional[str] = None,
        model: Optional[str] = None,
        detail: str = "auto",
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        Compare two images and describe differences.

        Args:
            image1: Path to first image or URL
            image2: Path to second image or URL
            aspect: Optional specific aspect to compare (e.g., "colors", "layout")
            model: Model to use (defaults to instance model)
            detail: Detail level (low, high, auto). Defaults to "auto".
            max_tokens: Maximum tokens in response. Defaults to 1024.
            **kwargs: Additional parameters for the API

        Returns:
            Comparison analysis

        Example:
            >>> provider = OpenAIVisionProvider()
            >>> comparison = provider.compare_images(
            ...     "version1.png",
            ...     "version2.png",
            ...     aspect="UI layout"
            ... )
        """
        if aspect:
            prompt = f"Compare these two images focusing on {aspect}. What are the key differences?"
        else:
            prompt = "Compare these two images. What are the key differences and similarities?"

        return self.analyze_images(
            images=[image1, image2],
            prompt=prompt,
            model=model,
            detail=detail,
            max_tokens=max_tokens,
            **kwargs
        )


def main():
    """Example usage of the OpenAI Vision Provider."""
    try:
        # Initialize the provider
        provider = OpenAIVisionProvider()

        print("=== OpenAI Vision Provider Examples ===\n")

        # Example 1: List available models
        print("Available Vision Models:")
        print("-" * 50)
        models = provider.list_models()
        for name, info in models.items():
            print(f"  {name}: {info['description']}")
            if info.get('notes'):
                print(f"    Note: {info['notes']}")
        print()

        # Example 2: Show usage patterns
        print("Usage Examples:")
        print("-" * 50)
        print("""
# Analyze a local image
result = provider.analyze_image(
    "photo.jpg",
    "What is in this image?"
)

# Describe an image
description = provider.describe_image("landscape.jpg")

# Extract text (OCR)
text = provider.extract_text("document.png")

# Answer a question
answer = provider.answer_about_image(
    "diagram.png",
    "What does this flowchart represent?"
)

# Compare images
comparison = provider.compare_images(
    "before.jpg",
    "after.jpg",
    aspect="visual changes"
)

# Analyze multiple images
analysis = provider.analyze_images(
    ["img1.jpg", "img2.jpg", "img3.jpg"],
    "What's the common theme across these images?"
)
        """)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

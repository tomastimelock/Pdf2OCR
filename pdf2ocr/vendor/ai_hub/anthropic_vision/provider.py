# Filepath: code_migration/ai_providers/anthropic_vision/provider.py
# Description: Anthropic Vision Provider for image analysis with Claude
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/Anthropic/vision/provider.py

"""
Anthropic Vision Provider

Provides image analysis capabilities using Claude's vision models.
"""

import os
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Iterator
from anthropic import Anthropic

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

from .model_config import (
    get_vision_model_config,
    get_default_vision_model,
    calculate_image_tokens,
    get_media_type,
    validate_image_format,
    is_optimal_size,
    VISION_CONFIG
)


class VisionError(Exception):
    """Exception raised for vision-related errors."""
    pass


class AnthropicVisionProvider:
    """
    Provider for Anthropic vision/image analysis using the Messages API.

    Supports:
        - Single and multiple image analysis
        - OCR-like text extraction
        - Image comparison
        - Object detection
        - Document and chart analysis
        - Streaming responses
        - Token estimation
    """

    SUPPORTED_FORMATS = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 2
    ):
        """
        Initialize the Anthropic Vision Provider.

        Args:
            api_key: Anthropic API key. If not provided, loads from ANTHROPIC_API_KEY env var.
            model: Model to use. If not provided, uses default (claude-sonnet-4-5-20250929).
            max_retries: Maximum number of retries for failed requests

        Raises:
            VisionError: If API key is not provided or found in environment
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise VisionError(
                "Anthropic API key must be provided or set in ANTHROPIC_API_KEY environment variable"
            )

        self.model = model or os.getenv("ANTHROPIC_MODEL", get_default_vision_model())
        self.client = Anthropic(api_key=self.api_key, max_retries=max_retries)

    def _get_media_type(self, file_path: str) -> str:
        """
        Get the MIME type for an image file.

        Args:
            file_path: Path to image file

        Returns:
            MIME type string

        Raises:
            VisionError: If image format is not supported
        """
        ext = Path(file_path).suffix.lower()
        media_type = get_media_type(ext)

        if not media_type:
            raise VisionError(
                f"Unsupported image format: {ext}. "
                f"Supported: {list(self.SUPPORTED_FORMATS.keys())}"
            )

        return media_type

    def _encode_image(self, image_path: str) -> tuple[str, str]:
        """
        Read and encode an image file to base64.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (base64_data, media_type)

        Raises:
            VisionError: If file cannot be read or format is unsupported
        """
        try:
            media_type = self._get_media_type(image_path)
            with open(image_path, "rb") as f:
                image_data = base64.standard_b64encode(f.read()).decode("utf-8")
            return image_data, media_type
        except FileNotFoundError:
            raise VisionError(f"Image file not found: {image_path}")
        except Exception as e:
            raise VisionError(f"Failed to encode image {image_path}: {str(e)}")

    def _create_image_content(self, image_source: str) -> Dict[str, Any]:
        """
        Create an image content block from a path or URL.

        Args:
            image_source: File path or URL to the image

        Returns:
            Image content block for the API
        """
        if image_source.startswith(('http://', 'https://')):
            # URL-based image
            return {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": image_source
                }
            }
        else:
            # File-based image (base64)
            image_data, media_type = self._encode_image(image_source)
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data
                }
            }

    def _extract_text(self, response) -> str:
        """
        Extract text content from a response object.

        Args:
            response: Anthropic API response

        Returns:
            Extracted text string
        """
        text_parts = []
        for block in response.content:
            if hasattr(block, 'text'):
                text_parts.append(block.text)
        return ''.join(text_parts)

    def analyze_image(
        self,
        image: str,
        prompt: str = "What's in this image?",
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        **kwargs
    ) -> str:
        """
        Analyze a single image with Claude.

        Args:
            image: Path to image file or URL
            prompt: Question or instruction about the image
            model: Model to use (defaults to instance model)
            system: System prompt for context
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            **kwargs: Additional parameters for the API

        Returns:
            Claude's analysis of the image

        Raises:
            VisionError: If API call fails

        Examples:
            >>> provider = AnthropicVisionProvider()
            >>> result = provider.analyze_image("photo.jpg", "What's in this image?")
            >>> print(result)
        """
        used_model = model or self.model

        # Build content array - images should come before text
        content = [
            self._create_image_content(image),
            {"type": "text", "text": prompt}
        ]

        params = {
            "model": used_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": content}]
        }

        if system:
            params["system"] = system

        params.update(kwargs)

        try:
            response = self.client.messages.create(**params)
            return self._extract_text(response)
        except Exception as e:
            raise VisionError(f"Failed to analyze image: {str(e)}")

    def analyze_images(
        self,
        images: List[str],
        prompt: str = "Describe these images",
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        **kwargs
    ) -> str:
        """
        Analyze multiple images with Claude.

        Args:
            images: List of image paths or URLs
            prompt: Question or instruction about the images
            model: Model to use
            system: System prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature (0.0-1.0)
            **kwargs: Additional parameters

        Returns:
            Claude's analysis of the images

        Raises:
            VisionError: If too many images or API call fails

        Examples:
            >>> provider = AnthropicVisionProvider()
            >>> result = provider.analyze_images(
            ...     images=["img1.jpg", "img2.jpg", "img3.jpg"],
            ...     prompt="Describe the sequence"
            ... )
        """
        max_images = VISION_CONFIG["max_images_api"]
        if len(images) > max_images:
            raise VisionError(f"Maximum {max_images} images allowed per request")

        used_model = model or self.model

        # Build content array - all images first, then text
        content = [self._create_image_content(img) for img in images]
        content.append({"type": "text", "text": prompt})

        params = {
            "model": used_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": content}]
        }

        if system:
            params["system"] = system

        params.update(kwargs)

        try:
            response = self.client.messages.create(**params)
            return self._extract_text(response)
        except Exception as e:
            raise VisionError(f"Failed to analyze images: {str(e)}")

    def compare_images(
        self,
        image1: str,
        image2: str,
        prompt: str = "Compare these images and describe their differences",
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Compare two images.

        Args:
            image1: Path to first image or URL
            image2: Path to second image or URL
            prompt: Comparison instruction
            model: Model to use
            **kwargs: Additional parameters

        Returns:
            Comparison analysis

        Examples:
            >>> provider = AnthropicVisionProvider()
            >>> comparison = provider.compare_images(
            ...     image1="before.jpg",
            ...     image2="after.jpg"
            ... )
        """
        return self.analyze_images(
            images=[image1, image2],
            prompt=prompt,
            model=model,
            **kwargs
        )

    def describe_image(
        self,
        image: str,
        detail_level: str = "detailed",
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Get a description of an image.

        Args:
            image: Path to image or URL
            detail_level: "brief", "detailed", or "comprehensive"
            model: Model to use
            **kwargs: Additional parameters

        Returns:
            Description of the image

        Examples:
            >>> provider = AnthropicVisionProvider()
            >>> desc = provider.describe_image(
            ...     image="scene.jpg",
            ...     detail_level="comprehensive"
            ... )
        """
        prompts = {
            "brief": "Provide a brief, one-sentence description of this image.",
            "detailed": "Describe this image in detail, including the main subjects, setting, colors, and mood.",
            "comprehensive": (
                "Provide a comprehensive analysis of this image including: "
                "main subjects, background elements, colors, lighting, composition, "
                "mood/atmosphere, and any text or symbols visible."
            )
        }

        prompt = prompts.get(detail_level, prompts["detailed"])
        return self.analyze_image(image=image, prompt=prompt, model=model, **kwargs)

    def extract_text(
        self,
        image: str,
        model: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Extract text from an image (OCR).

        Args:
            image: Path to image or URL
            model: Model to use
            language: Optional language hint (e.g., "swedish", "english")
            **kwargs: Additional parameters

        Returns:
            Extracted text from the image

        Examples:
            >>> provider = AnthropicVisionProvider()
            >>> text = provider.extract_text("document.png")
            >>> # For Swedish documents
            >>> text = provider.extract_text("protokoll.pdf", language="swedish")
        """
        prompt = "Extract and transcribe all text visible in this image. Preserve the original formatting as much as possible."

        if language:
            prompt += f" The text is in {language}."

        return self.analyze_image(
            image=image,
            prompt=prompt,
            model=model,
            temperature=0.1,  # Lower temperature for accuracy
            **kwargs
        )

    def detect_objects(
        self,
        image: str,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Detect and list objects in an image.

        Args:
            image: Path to image or URL
            model: Model to use
            **kwargs: Additional parameters

        Returns:
            List of detected objects with descriptions

        Examples:
            >>> provider = AnthropicVisionProvider()
            >>> objects = provider.detect_objects("scene.jpg")
        """
        prompt = (
            "List and describe all distinct objects visible in this image. "
            "For each object, provide:\n"
            "1. Object name/type\n"
            "2. Brief description\n"
            "3. Location in image (if relevant)\n"
            "4. Notable characteristics"
        )

        return self.analyze_image(image=image, prompt=prompt, model=model, **kwargs)

    def analyze_chart(
        self,
        image: str,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Analyze a chart or graph.

        Args:
            image: Path to chart image or URL
            model: Model to use
            **kwargs: Additional parameters

        Returns:
            Analysis of the chart

        Examples:
            >>> provider = AnthropicVisionProvider()
            >>> analysis = provider.analyze_chart("sales_chart.png")
        """
        prompt = """Analyze this chart/graph and provide:
1. Chart type (bar, line, pie, scatter, etc.)
2. Title and axis labels
3. Data trends and patterns
4. Key insights and takeaways
5. Any notable data points or outliers"""

        return self.analyze_image(image=image, prompt=prompt, model=model, **kwargs)

    def analyze_document(
        self,
        image: str,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Analyze a document image.

        Args:
            image: Path to document image or URL
            model: Model to use
            **kwargs: Additional parameters

        Returns:
            Document analysis

        Examples:
            >>> provider = AnthropicVisionProvider()
            >>> analysis = provider.analyze_document("contract.jpg")
        """
        prompt = """Analyze this document and provide:
1. Document type
2. Main content summary
3. Key information extracted
4. Any tables or structured data
5. Notable elements (headers, signatures, stamps, logos, etc.)"""

        return self.analyze_image(image=image, prompt=prompt, model=model, **kwargs)

    def analyze_image_stream(
        self,
        image: str,
        prompt: str = "What's in this image?",
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        **kwargs
    ) -> Iterator[str]:
        """
        Analyze image with streaming response.

        Args:
            image: Path to image file or URL
            prompt: Question or instruction about the image
            model: Model to use
            system: System prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Yields:
            Text chunks as they arrive

        Examples:
            >>> provider = AnthropicVisionProvider()
            >>> for chunk in provider.analyze_image_stream("photo.jpg", "Describe in detail"):
            ...     print(chunk, end="", flush=True)
        """
        used_model = model or self.model

        content = [
            self._create_image_content(image),
            {"type": "text", "text": prompt}
        ]

        params = {
            "model": used_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": content}],
            "stream": True
        }

        if system:
            params["system"] = system

        params.update(kwargs)

        try:
            with self.client.messages.stream(**params) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            raise VisionError(f"Failed to stream image analysis: {str(e)}")

    def estimate_tokens(self, image_path: str) -> Dict[str, Any]:
        """
        Estimate the token cost for an image.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with image dimensions and estimated tokens

        Examples:
            >>> provider = AnthropicVisionProvider()
            >>> info = provider.estimate_tokens("large_photo.jpg")
            >>> print(f"Tokens: {info['estimated_tokens']}")
            >>> print(f"Optimal: {info['optimal']}")
        """
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                width, height = img.size
                tokens = calculate_image_tokens(width, height)
                optimal = is_optimal_size(width, height)

                return {
                    "width": width,
                    "height": height,
                    "megapixels": (width * height) / 1_000_000,
                    "estimated_tokens": tokens,
                    "optimal": optimal,
                    "recommendation": (
                        "Image size is optimal" if optimal
                        else f"Consider resizing to {VISION_CONFIG['optimal_max_dimension']}px max dimension"
                    )
                }
        except ImportError:
            raise VisionError(
                "PIL/Pillow not installed. Install with: pip install Pillow"
            )
        except Exception as e:
            raise VisionError(f"Failed to estimate tokens: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary with model information
        """
        config = get_vision_model_config(self.model)

        if not config:
            return {"error": f"Unknown model: {self.model}"}

        return {
            "name": config.name,
            "description": config.description,
            "context_window": config.context_window,
            "max_output": config.max_output,
            "vision_capable": config.vision_capable,
            "supported_formats": config.supported_formats,
            "max_images": config.max_images,
            "notes": config.notes
        }


def main():
    """Example usage of the Anthropic Vision Provider."""
    import sys

    try:
        # Initialize provider
        provider = AnthropicVisionProvider()

        print("=" * 60)
        print("Anthropic Vision Provider - Initialized")
        print("=" * 60)

        # Show model info
        info = provider.get_model_info()
        print(f"\nCurrent Model: {info['name']}")
        print(f"Description: {info['description']}")
        print(f"Context Window: {info['context_window']:,} tokens")
        print(f"Max Output: {info['max_output']:,} tokens")
        print(f"Supported Formats: {', '.join([ext for ext in provider.SUPPORTED_FORMATS.keys()])}")

        print("\n" + "=" * 60)
        print("Usage Examples")
        print("=" * 60)

        examples = """
# Analyze single image
result = provider.analyze_image(
    image="photo.jpg",
    prompt="What's in this image?"
)

# Extract text (OCR)
text = provider.extract_text("document.png")

# Compare images
comparison = provider.compare_images(
    image1="before.jpg",
    image2="after.jpg"
)

# Analyze with streaming
for chunk in provider.analyze_image_stream("photo.jpg", "Describe in detail"):
    print(chunk, end="", flush=True)

# Estimate token cost
info = provider.estimate_tokens("large_image.jpg")
print(f"Tokens: {info['estimated_tokens']}")

# Swedish language support
text = provider.extract_text(
    image="protokoll.pdf",
    language="swedish"
)
"""
        print(examples)

        print("\n" + "=" * 60)
        print("Ready to analyze images!")
        print("=" * 60)

    except VisionError as e:
        print(f"Vision Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

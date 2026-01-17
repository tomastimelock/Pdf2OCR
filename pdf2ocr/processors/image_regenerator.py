"""
Image Regenerator - Regenerate images (photos, illustrations) using OpenAI.

Uses OpenAI GPT-4o for image description and DALL-E 3 for regeneration.
"""

import os
import base64
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
import logging
import requests

logger = logging.getLogger(__name__)


@dataclass
class ImageDescription:
    """Description of an image for regeneration."""
    image_path: str
    page_number: int
    image_index: int
    image_type: str  # photo, illustration, diagram, icon, etc.
    description: str
    style: str  # realistic, artistic, sketch, etc.
    dominant_colors: List[str] = field(default_factory=list)
    subjects: List[str] = field(default_factory=list)
    context: str = ""


@dataclass
class RegeneratedImage:
    """A regenerated image."""
    original_path: str
    regenerated_path: str
    page_number: int
    image_index: int
    description: str
    prompt_used: str
    width: int
    height: int
    success: bool = True
    error: Optional[str] = None


@dataclass
class ImageRegenerationResult:
    """Result of image regeneration for a document."""
    images: List[RegeneratedImage] = field(default_factory=list)
    total_detected: int = 0
    total_regenerated: int = 0
    total_skipped: int = 0
    total_failed: int = 0


class ImageRegenerator:
    """
    Regenerate images (photos, illustrations) using OpenAI.

    Process:
    1. Analyze image using GPT-4o Vision to get detailed description
    2. Classify image type (photo, illustration, etc.)
    3. Generate new image using DALL-E 3 based on description
    4. Save regenerated image

    Note: Only regenerates photos and illustrations, not charts/tables/diagrams
    which should be handled by ChartRegenerator for SVG output.
    """

    # Image types to regenerate (photos, illustrations)
    REGENERATE_TYPES = {"photo", "photograph", "illustration", "artwork", "painting", "drawing"}

    # Image types to skip (handled by other processors)
    SKIP_TYPES = {"chart", "graph", "table", "diagram", "flowchart", "screenshot", "icon", "logo"}

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the image regenerator.

        Args:
            api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None

    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "OpenAI API key not provided. "
                    "Set OPENAI_API_KEY environment variable or pass api_key parameter."
                )
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        return bool(self.api_key)

    def _encode_image(self, image_path: Path) -> str:
        """Read and base64 encode an image."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_mime_type(self, image_path: Path) -> str:
        """Get MIME type from file extension."""
        ext = image_path.suffix.lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(ext, "image/jpeg")

    def analyze_image(
        self,
        image_path: str | Path,
        page_number: int = 1,
        image_index: int = 1
    ) -> ImageDescription:
        """
        Analyze an image to get detailed description for regeneration.

        Args:
            image_path: Path to the image file
            page_number: Page number where image was found
            image_index: Index of image on the page

        Returns:
            ImageDescription with analysis results
        """
        image_path = Path(image_path)

        if not image_path.exists():
            return ImageDescription(
                image_path=str(image_path),
                page_number=page_number,
                image_index=image_index,
                image_type="unknown",
                description="Image not found",
                style="unknown"
            )

        try:
            base64_image = self._encode_image(image_path)
            mime_type = self._get_mime_type(image_path)

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}",
                                    "detail": "high"
                                }
                            },
                            {
                                "type": "text",
                                "text": """Analyze this image for recreation purposes. Provide:

1. IMAGE_TYPE: One of: photo, photograph, illustration, artwork, painting, drawing, chart, graph, table, diagram, flowchart, screenshot, icon, logo, other

2. STYLE: The artistic/visual style (e.g., realistic, photorealistic, artistic, watercolor, sketch, digital art, cartoon, minimalist, etc.)

3. DESCRIPTION: A detailed description of the image content that could be used to recreate it. Include:
   - Main subjects and their positions
   - Background and setting
   - Lighting and mood
   - Important details and textures
   - Any text visible in the image

4. DOMINANT_COLORS: List the 3-5 main colors

5. SUBJECTS: List the main subjects/objects

6. CONTEXT: What context or purpose does this image serve in a document?

Format your response as:
IMAGE_TYPE: [type]
STYLE: [style]
DESCRIPTION: [detailed description]
DOMINANT_COLORS: [color1, color2, ...]
SUBJECTS: [subject1, subject2, ...]
CONTEXT: [context]"""
                            }
                        ]
                    }
                ],
                max_tokens=1500,
                temperature=0.3
            )

            content = response.choices[0].message.content
            return self._parse_analysis(content, str(image_path), page_number, image_index)

        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return ImageDescription(
                image_path=str(image_path),
                page_number=page_number,
                image_index=image_index,
                image_type="unknown",
                description=f"Analysis failed: {e}",
                style="unknown"
            )

    def _parse_analysis(
        self,
        content: str,
        image_path: str,
        page_number: int,
        image_index: int
    ) -> ImageDescription:
        """Parse the analysis response into ImageDescription."""
        # Extract fields using regex
        image_type = "unknown"
        style = "unknown"
        description = ""
        colors = []
        subjects = []
        context = ""

        type_match = re.search(r'IMAGE_TYPE:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
        if type_match:
            image_type = type_match.group(1).strip().lower()

        style_match = re.search(r'STYLE:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
        if style_match:
            style = style_match.group(1).strip()

        desc_match = re.search(r'DESCRIPTION:\s*(.+?)(?=DOMINANT_COLORS:|SUBJECTS:|CONTEXT:|$)', content, re.IGNORECASE | re.DOTALL)
        if desc_match:
            description = desc_match.group(1).strip()

        colors_match = re.search(r'DOMINANT_COLORS:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
        if colors_match:
            colors = [c.strip() for c in colors_match.group(1).split(',')]

        subjects_match = re.search(r'SUBJECTS:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
        if subjects_match:
            subjects = [s.strip() for s in subjects_match.group(1).split(',')]

        context_match = re.search(r'CONTEXT:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
        if context_match:
            context = context_match.group(1).strip()

        return ImageDescription(
            image_path=image_path,
            page_number=page_number,
            image_index=image_index,
            image_type=image_type,
            description=description,
            style=style,
            dominant_colors=colors,
            subjects=subjects,
            context=context
        )

    def should_regenerate(self, description: ImageDescription) -> bool:
        """
        Determine if an image should be regenerated.

        Only regenerates photos and illustrations, not charts/tables/diagrams.
        """
        image_type = description.image_type.lower()

        # Check if it's a type we should regenerate
        for regen_type in self.REGENERATE_TYPES:
            if regen_type in image_type:
                return True

        # Check if it's a type we should skip
        for skip_type in self.SKIP_TYPES:
            if skip_type in image_type:
                return False

        # Default: regenerate if we're unsure
        return True

    def regenerate_image(
        self,
        description: ImageDescription,
        output_path: str | Path,
        size: str = "1024x1024"
    ) -> RegeneratedImage:
        """
        Regenerate an image using DALL-E 3.

        Args:
            description: ImageDescription from analysis
            output_path: Path to save the regenerated image
            size: Image size (1024x1024, 1792x1024, or 1024x1792)

        Returns:
            RegeneratedImage with results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build prompt for DALL-E
        prompt = self._build_generation_prompt(description)

        try:
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality="standard",
                n=1
            )

            # Get image URL and download
            image_url = response.data[0].url

            # Download and save image
            img_response = requests.get(image_url, timeout=60)
            img_response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(img_response.content)

            # Parse dimensions
            width, height = map(int, size.split("x"))

            logger.info(f"Regenerated image saved: {output_path}")

            return RegeneratedImage(
                original_path=description.image_path,
                regenerated_path=str(output_path),
                page_number=description.page_number,
                image_index=description.image_index,
                description=description.description,
                prompt_used=prompt,
                width=width,
                height=height,
                success=True
            )

        except Exception as e:
            logger.error(f"Image regeneration failed: {e}")
            return RegeneratedImage(
                original_path=description.image_path,
                regenerated_path="",
                page_number=description.page_number,
                image_index=description.image_index,
                description=description.description,
                prompt_used=prompt if 'prompt' in locals() else "",
                width=0,
                height=0,
                success=False,
                error=str(e)
            )

    def _build_generation_prompt(self, description: ImageDescription) -> str:
        """Build a prompt for DALL-E image generation."""
        parts = []

        # Add style prefix
        if description.style and description.style != "unknown":
            parts.append(f"Create a {description.style} image:")
        else:
            parts.append("Create an image:")

        # Add main description
        if description.description:
            parts.append(description.description)

        # Add color guidance
        if description.dominant_colors:
            colors = ", ".join(description.dominant_colors[:3])
            parts.append(f"Use these dominant colors: {colors}.")

        # Combine and truncate to DALL-E's limit
        prompt = " ".join(parts)

        # DALL-E 3 has a 4000 character limit
        if len(prompt) > 3900:
            prompt = prompt[:3900] + "..."

        return prompt

    def process_extracted_images(
        self,
        images_dir: str | Path,
        output_dir: str | Path,
        progress_callback: Optional[Callable[[int, int, RegeneratedImage], None]] = None
    ) -> ImageRegenerationResult:
        """
        Process all extracted images in a directory.

        Args:
            images_dir: Directory containing extracted images
            output_dir: Directory to save regenerated images
            progress_callback: Callback for progress (current, total, result)

        Returns:
            ImageRegenerationResult with all results
        """
        images_dir = Path(images_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = ImageRegenerationResult()

        # Find all images
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        image_files = sorted([
            f for f in images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ])

        result.total_detected = len(image_files)

        for idx, image_path in enumerate(image_files, start=1):
            # Extract page number and image index from filename
            match = re.search(r'page_(\d+)_img_(\d+)', image_path.stem)
            if match:
                page_num = int(match.group(1))
                img_idx = int(match.group(2))
            else:
                page_num = 1
                img_idx = idx

            # Analyze image
            description = self.analyze_image(image_path, page_num, img_idx)

            # Check if we should regenerate
            if not self.should_regenerate(description):
                logger.info(f"Skipping {image_path.name}: type={description.image_type}")
                result.total_skipped += 1
                continue

            # Regenerate
            output_filename = f"regen_page_{page_num:03d}_img_{img_idx:03d}.png"
            output_path = output_dir / output_filename

            regen_result = self.regenerate_image(description, output_path)
            result.images.append(regen_result)

            if regen_result.success:
                result.total_regenerated += 1
            else:
                result.total_failed += 1

            if progress_callback:
                progress_callback(idx, len(image_files), regen_result)

        logger.info(
            f"Image regeneration complete: {result.total_regenerated} regenerated, "
            f"{result.total_skipped} skipped, {result.total_failed} failed"
        )

        return result

    def process_page_for_images(
        self,
        page_path: str | Path,
        page_number: int,
        output_dir: str | Path
    ) -> List[RegeneratedImage]:
        """
        Detect and regenerate images from a page image.

        Note: This analyzes a full page image and identifies distinct images within it.
        For pre-extracted images, use process_extracted_images() instead.

        Args:
            page_path: Path to the page image
            page_number: Page number
            output_dir: Directory to save regenerated images

        Returns:
            List of RegeneratedImage objects
        """
        # For full page analysis, we treat the entire page as potentially
        # containing one or more images. This is a simplified approach.
        # For better results, use PDFSplitter.extract_images() first.

        description = self.analyze_image(page_path, page_number, 1)

        if not self.should_regenerate(description):
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"regen_page_{page_number:03d}_img_001.png"
        result = self.regenerate_image(description, output_path)

        return [result] if result.success else []

"""
Image border and frame operations.

Provides various border styles including solid, gradient,
patterned, and artistic frames.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any, List
import logging

from PIL import Image, ImageDraw, ImageOps

logger = logging.getLogger(__name__)


class BorderStyle(Enum):
    """Border style options."""
    SOLID = "solid"
    GRADIENT = "gradient"
    DOUBLE = "double"
    ROUNDED = "rounded"
    SHADOW = "shadow"
    POLAROID = "polaroid"
    FILM_STRIP = "film_strip"
    TORN = "torn"


@dataclass
class BorderResult:
    """Result of a border operation."""
    success: bool
    original_size: Tuple[int, int]
    new_size: Tuple[int, int]
    border_style: str
    border_width: int
    output_path: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'success': self.success,
            'original_size': self.original_size,
            'new_size': self.new_size,
            'border_style': self.border_style,
            'border_width': self.border_width,
            'output_path': self.output_path,
            'error': self.error,
            'metadata': self.metadata,
        }


class BorderMaker:
    """
    Image border and frame engine.

    Provides various border styles and frame effects.
    """

    def __init__(self):
        """Initialize the border maker."""
        pass

    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """Load image from path or return if already an Image."""
        if isinstance(image, Image.Image):
            return image
        return Image.open(str(image))

    def _parse_color(self, color: Union[str, Tuple]) -> Tuple:
        """Parse color specification to RGB/RGBA tuple."""
        if isinstance(color, tuple):
            return color
        if isinstance(color, str):
            if color.startswith('#'):
                hex_color = color.lstrip('#')
                if len(hex_color) == 6:
                    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                elif len(hex_color) == 8:
                    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4, 6))
            # Named colors
            color_map = {
                'white': (255, 255, 255),
                'black': (0, 0, 0),
                'red': (255, 0, 0),
                'green': (0, 255, 0),
                'blue': (0, 0, 255),
                'gray': (128, 128, 128),
                'grey': (128, 128, 128),
            }
            return color_map.get(color.lower(), (255, 255, 255))
        return (255, 255, 255)

    def _add_solid_border(
        self,
        image: Image.Image,
        width: int,
        color: Tuple
    ) -> Image.Image:
        """Add a solid color border."""
        return ImageOps.expand(image, border=width, fill=color)

    def _add_gradient_border(
        self,
        image: Image.Image,
        width: int,
        color1: Tuple,
        color2: Tuple
    ) -> Image.Image:
        """Add a gradient border."""
        try:
            import numpy as np

            # Create new image with border size
            new_w = image.width + width * 2
            new_h = image.height + width * 2

            # Create gradient
            gradient = np.zeros((new_h, new_w, len(color1)), dtype=np.uint8)

            for i in range(new_h):
                ratio = i / new_h
                for c in range(len(color1)):
                    gradient[i, :, c] = int(color1[c] * (1 - ratio) + color2[c] * ratio)

            mode = 'RGBA' if len(color1) == 4 else 'RGB'
            result = Image.fromarray(gradient, mode=mode)

            # Paste original image
            if image.mode != mode:
                image = image.convert(mode)
            result.paste(image, (width, width))

            return result

        except ImportError:
            # Fallback to solid border
            return self._add_solid_border(image, width, color1)

    def _add_double_border(
        self,
        image: Image.Image,
        width: int,
        outer_color: Tuple,
        inner_color: Tuple
    ) -> Image.Image:
        """Add a double border."""
        inner_width = width // 3
        outer_width = width - inner_width

        # Add inner border
        result = ImageOps.expand(image, border=inner_width, fill=inner_color)
        # Add outer border
        result = ImageOps.expand(result, border=outer_width, fill=outer_color)

        return result

    def _add_rounded_border(
        self,
        image: Image.Image,
        width: int,
        color: Tuple,
        radius: int
    ) -> Image.Image:
        """Add a border with rounded corners."""
        # Create new image with border
        new_w = image.width + width * 2
        new_h = image.height + width * 2

        mode = 'RGBA' if len(color) == 4 or image.mode == 'RGBA' else 'RGB'
        result = Image.new(mode, (new_w, new_h), color)

        # Create rounded mask
        mask = Image.new('L', (new_w, new_h), 0)
        draw = ImageDraw.Draw(mask)

        # Draw rounded rectangle for the inner area
        inner_radius = max(0, radius - width)
        draw.rounded_rectangle(
            [width, width, new_w - width, new_h - width],
            radius=inner_radius,
            fill=255
        )

        # Convert image to same mode
        if image.mode != mode:
            image = image.convert(mode)

        # Paste image with mask
        temp = Image.new(mode, (new_w, new_h), color)
        temp.paste(image, (width, width))

        result = Image.composite(temp, result, mask)

        return result

    def _add_shadow_border(
        self,
        image: Image.Image,
        width: int,
        shadow_color: Tuple = (0, 0, 0, 128),
        background_color: Tuple = (255, 255, 255, 255)
    ) -> Image.Image:
        """Add a shadow effect border."""
        shadow_offset = width // 3

        # Create new image with space for shadow
        new_w = image.width + width * 2
        new_h = image.height + width * 2

        result = Image.new('RGBA', (new_w, new_h), background_color)

        # Create shadow
        shadow = Image.new('RGBA', image.size, shadow_color)

        # Apply blur to shadow
        from PIL import ImageFilter
        shadow_expanded = ImageOps.expand(shadow, border=shadow_offset, fill=(0, 0, 0, 0))
        shadow_blurred = shadow_expanded.filter(ImageFilter.GaussianBlur(radius=shadow_offset))

        # Paste shadow
        result.paste(shadow_blurred, (width + shadow_offset, width + shadow_offset), shadow_blurred)

        # Paste original image
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        result.paste(image, (width, width), image)

        return result

    def _add_polaroid_border(
        self,
        image: Image.Image,
        width: int
    ) -> Image.Image:
        """Add a Polaroid-style border."""
        # Polaroid has thick bottom border
        top_width = width
        side_width = width
        bottom_width = width * 3

        new_w = image.width + side_width * 2
        new_h = image.height + top_width + bottom_width

        result = Image.new('RGB', (new_w, new_h), (255, 255, 255))

        # Paste image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        result.paste(image, (side_width, top_width))

        return result

    def _add_film_strip_border(
        self,
        image: Image.Image,
        width: int
    ) -> Image.Image:
        """Add a film strip style border."""
        # Create border with film perforations
        new_w = image.width + width * 2
        new_h = image.height + width * 2

        result = Image.new('RGB', (new_w, new_h), (20, 20, 20))
        draw = ImageDraw.Draw(result)

        # Draw perforations
        perf_width = width // 2
        perf_height = width // 3
        perf_spacing = width

        # Top and bottom perforations
        for x in range(perf_spacing // 2, new_w, perf_spacing):
            # Top
            draw.rectangle(
                [x - perf_width // 2, 2, x + perf_width // 2, 2 + perf_height],
                fill=(255, 255, 255)
            )
            # Bottom
            draw.rectangle(
                [x - perf_width // 2, new_h - 2 - perf_height, x + perf_width // 2, new_h - 2],
                fill=(255, 255, 255)
            )

        # Paste image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        result.paste(image, (width, width))

        return result

    def add_border(
        self,
        image: Union[str, Path, Image.Image],
        width: int,
        color: Union[str, Tuple] = 'white',
        style: BorderStyle = BorderStyle.SOLID,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95,
        **kwargs
    ) -> Union[BorderResult, Tuple[Image.Image, BorderResult]]:
        """
        Add a border to an image.

        Args:
            image: Input image (path or PIL Image)
            width: Border width in pixels
            color: Border color
            style: Border style
            output: Output path (if None, returns PIL Image)
            quality: JPEG quality (1-100)
            **kwargs: Style-specific options

        Returns:
            BorderResult if output path provided, otherwise (Image, BorderResult)
        """
        try:
            img = self._load_image(image)
            original_size = img.size
            parsed_color = self._parse_color(color)

            if style == BorderStyle.SOLID:
                result_img = self._add_solid_border(img, width, parsed_color)

            elif style == BorderStyle.GRADIENT:
                color2 = self._parse_color(kwargs.get('color2', 'black'))
                result_img = self._add_gradient_border(img, width, parsed_color, color2)

            elif style == BorderStyle.DOUBLE:
                inner_color = self._parse_color(kwargs.get('inner_color', 'black'))
                result_img = self._add_double_border(img, width, parsed_color, inner_color)

            elif style == BorderStyle.ROUNDED:
                radius = kwargs.get('radius', width * 2)
                result_img = self._add_rounded_border(img, width, parsed_color, radius)

            elif style == BorderStyle.SHADOW:
                shadow_color = self._parse_color(kwargs.get('shadow_color', (0, 0, 0, 128)))
                result_img = self._add_shadow_border(img, width, shadow_color, parsed_color)

            elif style == BorderStyle.POLAROID:
                result_img = self._add_polaroid_border(img, width)

            elif style == BorderStyle.FILM_STRIP:
                result_img = self._add_film_strip_border(img, width)

            else:
                result_img = self._add_solid_border(img, width, parsed_color)

            result = BorderResult(
                success=True,
                original_size=original_size,
                new_size=result_img.size,
                border_style=style.value,
                border_width=width,
                metadata={'color': str(color)}
            )

            if output:
                output_path = Path(output)

                save_kwargs = {}
                ext = output_path.suffix.lower()

                if ext in ['.jpg', '.jpeg']:
                    if result_img.mode in ('RGBA', 'P'):
                        result_img = result_img.convert('RGB')
                    save_kwargs['quality'] = quality
                    save_kwargs['optimize'] = True
                elif ext == '.png':
                    save_kwargs['optimize'] = True
                elif ext == '.webp':
                    save_kwargs['quality'] = quality

                result_img.save(str(output_path), **save_kwargs)
                result.output_path = str(output_path)
                return result
            else:
                return (result_img, result)

        except Exception as e:
            logger.error(f"Border error: {e}")
            return BorderResult(
                success=False,
                original_size=(0, 0),
                new_size=(0, 0),
                border_style=style.value if style else 'unknown',
                border_width=width,
                error=str(e)
            )

    def add_simple_border(
        self,
        image: Union[str, Path, Image.Image],
        width: int,
        color: Union[str, Tuple] = 'white',
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[BorderResult, Tuple[Image.Image, BorderResult]]:
        """Convenience method for simple solid border."""
        return self.add_border(
            image, width, color,
            style=BorderStyle.SOLID,
            output=output, quality=quality
        )

    def add_polaroid_frame(
        self,
        image: Union[str, Path, Image.Image],
        border_width: int = 20,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[BorderResult, Tuple[Image.Image, BorderResult]]:
        """Add Polaroid-style frame."""
        return self.add_border(
            image, border_width, 'white',
            style=BorderStyle.POLAROID,
            output=output, quality=quality
        )

    def add_drop_shadow(
        self,
        image: Union[str, Path, Image.Image],
        shadow_size: int = 10,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[BorderResult, Tuple[Image.Image, BorderResult]]:
        """Add drop shadow effect."""
        return self.add_border(
            image, shadow_size, 'white',
            style=BorderStyle.SHADOW,
            output=output, quality=quality
        )

    def list_styles(self) -> List[str]:
        """Get list of available border styles."""
        return [s.value for s in BorderStyle]

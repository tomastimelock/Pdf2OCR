"""
Image overlay and watermark operations.

Provides overlay, watermark, and text stamping functionality.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any
import logging

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


class WatermarkPosition(Enum):
    """Watermark position options."""
    TOP_LEFT = "top-left"
    TOP_CENTER = "top-center"
    TOP_RIGHT = "top-right"
    CENTER_LEFT = "center-left"
    CENTER = "center"
    CENTER_RIGHT = "center-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_CENTER = "bottom-center"
    BOTTOM_RIGHT = "bottom-right"
    TILE = "tile"


@dataclass
class OverlayResult:
    """Result of an overlay operation."""
    success: bool
    original_size: Tuple[int, int]
    new_size: Tuple[int, int]
    overlay_type: str
    position: str
    opacity: float
    output_path: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'success': self.success,
            'original_size': self.original_size,
            'new_size': self.new_size,
            'overlay_type': self.overlay_type,
            'position': self.position,
            'opacity': self.opacity,
            'output_path': self.output_path,
            'error': self.error,
            'metadata': self.metadata,
        }


class OverlayTool:
    """
    Image overlay and watermark engine.

    Provides text and image overlay capabilities with
    various positioning and opacity options.
    """

    def __init__(self, default_font: Optional[str] = None):
        """
        Initialize the overlay tool.

        Args:
            default_font: Path to default font file
        """
        self.default_font = default_font

    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """Load image from path or return if already an Image."""
        if isinstance(image, Image.Image):
            return image
        return Image.open(str(image))

    def _get_font(self, font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
        """Get a font object."""
        try:
            if font_path:
                return ImageFont.truetype(font_path, size)
            elif self.default_font:
                return ImageFont.truetype(self.default_font, size)
            else:
                # Try common system fonts
                common_fonts = [
                    'arial.ttf',
                    'Arial.ttf',
                    'DejaVuSans.ttf',
                    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                    'C:/Windows/Fonts/arial.ttf',
                ]
                for font in common_fonts:
                    try:
                        return ImageFont.truetype(font, size)
                    except (OSError, IOError):
                        continue
                # Fallback to default
                return ImageFont.load_default()
        except Exception:
            return ImageFont.load_default()

    def _calculate_position(
        self,
        base_size: Tuple[int, int],
        overlay_size: Tuple[int, int],
        position: WatermarkPosition,
        margin: int = 10
    ) -> Tuple[int, int]:
        """Calculate position for overlay."""
        base_w, base_h = base_size
        overlay_w, overlay_h = overlay_size

        positions = {
            WatermarkPosition.TOP_LEFT: (margin, margin),
            WatermarkPosition.TOP_CENTER: ((base_w - overlay_w) // 2, margin),
            WatermarkPosition.TOP_RIGHT: (base_w - overlay_w - margin, margin),
            WatermarkPosition.CENTER_LEFT: (margin, (base_h - overlay_h) // 2),
            WatermarkPosition.CENTER: ((base_w - overlay_w) // 2, (base_h - overlay_h) // 2),
            WatermarkPosition.CENTER_RIGHT: (base_w - overlay_w - margin, (base_h - overlay_h) // 2),
            WatermarkPosition.BOTTOM_LEFT: (margin, base_h - overlay_h - margin),
            WatermarkPosition.BOTTOM_CENTER: ((base_w - overlay_w) // 2, base_h - overlay_h - margin),
            WatermarkPosition.BOTTOM_RIGHT: (base_w - overlay_w - margin, base_h - overlay_h - margin),
        }

        return positions.get(position, positions[WatermarkPosition.BOTTOM_RIGHT])

    def _create_text_image(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        color: Tuple[int, int, int, int]
    ) -> Image.Image:
        """Create an image from text."""
        # Get text bounding box
        dummy_img = Image.new('RGBA', (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        bbox = dummy_draw.textbbox((0, 0), text, font=font)

        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Add padding
        padding = 10
        img_w = text_w + padding * 2
        img_h = text_h + padding * 2

        # Create text image
        text_img = Image.new('RGBA', (img_w, img_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_img)
        draw.text((padding - bbox[0], padding - bbox[1]), text, font=font, fill=color)

        return text_img

    def add_text_watermark(
        self,
        image: Union[str, Path, Image.Image],
        text: str,
        position: Union[str, WatermarkPosition] = WatermarkPosition.BOTTOM_RIGHT,
        opacity: float = 0.5,
        font_size: int = 24,
        font_path: Optional[str] = None,
        color: Tuple[int, int, int] = (255, 255, 255),
        output: Optional[Union[str, Path]] = None,
        quality: int = 95,
        margin: int = 10
    ) -> Union[OverlayResult, Tuple[Image.Image, OverlayResult]]:
        """
        Add a text watermark to an image.

        Args:
            image: Input image (path or PIL Image)
            text: Watermark text
            position: Position of watermark
            opacity: Watermark opacity (0-1)
            font_size: Font size
            font_path: Path to font file
            color: Text color (RGB)
            output: Output path (if None, returns PIL Image)
            quality: JPEG quality (1-100)
            margin: Margin from edges

        Returns:
            OverlayResult if output path provided, otherwise (Image, OverlayResult)
        """
        try:
            img = self._load_image(image)
            original_size = img.size

            # Parse position
            if isinstance(position, str):
                position = WatermarkPosition(position)

            # Ensure RGBA mode
            if img.mode != 'RGBA':
                img = img.convert('RGBA')

            # Get font
            font = self._get_font(font_path, font_size)

            # Create text image
            alpha = int(255 * opacity)
            text_color = (*color, alpha)
            text_img = self._create_text_image(text, font, text_color)

            # Handle tile mode
            if position == WatermarkPosition.TILE:
                # Tile the watermark across the image
                tile_spacing_x = text_img.width + 50
                tile_spacing_y = text_img.height + 50

                for y in range(0, img.height, tile_spacing_y):
                    for x in range(0, img.width, tile_spacing_x):
                        img.paste(text_img, (x, y), text_img)
            else:
                # Single watermark
                pos = self._calculate_position(img.size, text_img.size, position, margin)
                img.paste(text_img, pos, text_img)

            result = OverlayResult(
                success=True,
                original_size=original_size,
                new_size=img.size,
                overlay_type='text',
                position=position.value,
                opacity=opacity,
                metadata={'text': text, 'font_size': font_size}
            )

            if output:
                output_path = Path(output)

                save_kwargs = {}
                ext = output_path.suffix.lower()

                if ext in ['.jpg', '.jpeg']:
                    img = img.convert('RGB')
                    save_kwargs['quality'] = quality
                    save_kwargs['optimize'] = True
                elif ext == '.png':
                    save_kwargs['optimize'] = True
                elif ext == '.webp':
                    save_kwargs['quality'] = quality

                img.save(str(output_path), **save_kwargs)
                result.output_path = str(output_path)
                return result
            else:
                return (img, result)

        except Exception as e:
            logger.error(f"Text watermark error: {e}")
            return OverlayResult(
                success=False,
                original_size=(0, 0),
                new_size=(0, 0),
                overlay_type='text',
                position=str(position),
                opacity=opacity,
                error=str(e)
            )

    def add_image_watermark(
        self,
        image: Union[str, Path, Image.Image],
        watermark: Union[str, Path, Image.Image],
        position: Union[str, WatermarkPosition] = WatermarkPosition.BOTTOM_RIGHT,
        opacity: float = 0.5,
        scale: float = 0.2,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95,
        margin: int = 10
    ) -> Union[OverlayResult, Tuple[Image.Image, OverlayResult]]:
        """
        Add an image watermark to an image.

        Args:
            image: Input image (path or PIL Image)
            watermark: Watermark image (path or PIL Image)
            position: Position of watermark
            opacity: Watermark opacity (0-1)
            scale: Scale of watermark relative to image
            output: Output path (if None, returns PIL Image)
            quality: JPEG quality (1-100)
            margin: Margin from edges

        Returns:
            OverlayResult if output path provided, otherwise (Image, OverlayResult)
        """
        try:
            img = self._load_image(image)
            wm = self._load_image(watermark)
            original_size = img.size

            # Parse position
            if isinstance(position, str):
                position = WatermarkPosition(position)

            # Ensure RGBA mode
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            if wm.mode != 'RGBA':
                wm = wm.convert('RGBA')

            # Scale watermark
            wm_w = int(img.width * scale)
            wm_h = int(wm.height * (wm_w / wm.width))
            wm = wm.resize((wm_w, wm_h), Image.Resampling.LANCZOS)

            # Apply opacity
            if opacity < 1.0:
                # Modify alpha channel
                r, g, b, a = wm.split()
                a = a.point(lambda x: int(x * opacity))
                wm = Image.merge('RGBA', (r, g, b, a))

            # Handle tile mode
            if position == WatermarkPosition.TILE:
                tile_spacing_x = wm.width + 50
                tile_spacing_y = wm.height + 50

                for y in range(0, img.height, tile_spacing_y):
                    for x in range(0, img.width, tile_spacing_x):
                        img.paste(wm, (x, y), wm)
            else:
                # Single watermark
                pos = self._calculate_position(img.size, wm.size, position, margin)
                img.paste(wm, pos, wm)

            result = OverlayResult(
                success=True,
                original_size=original_size,
                new_size=img.size,
                overlay_type='image',
                position=position.value,
                opacity=opacity,
                metadata={'scale': scale}
            )

            if output:
                output_path = Path(output)

                save_kwargs = {}
                ext = output_path.suffix.lower()

                if ext in ['.jpg', '.jpeg']:
                    img = img.convert('RGB')
                    save_kwargs['quality'] = quality
                    save_kwargs['optimize'] = True
                elif ext == '.png':
                    save_kwargs['optimize'] = True
                elif ext == '.webp':
                    save_kwargs['quality'] = quality

                img.save(str(output_path), **save_kwargs)
                result.output_path = str(output_path)
                return result
            else:
                return (img, result)

        except Exception as e:
            logger.error(f"Image watermark error: {e}")
            return OverlayResult(
                success=False,
                original_size=(0, 0),
                new_size=(0, 0),
                overlay_type='image',
                position=str(position),
                opacity=opacity,
                error=str(e)
            )

    def add_overlay(
        self,
        base_image: Union[str, Path, Image.Image],
        overlay_image: Union[str, Path, Image.Image],
        blend_mode: str = 'normal',
        opacity: float = 0.5,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95
    ) -> Union[OverlayResult, Tuple[Image.Image, OverlayResult]]:
        """
        Add a full-size overlay to an image.

        Args:
            base_image: Base image
            overlay_image: Overlay image
            blend_mode: Blend mode (normal, multiply, screen, overlay)
            opacity: Overlay opacity (0-1)
            output: Output path
            quality: JPEG quality

        Returns:
            OverlayResult or (Image, OverlayResult)
        """
        try:
            base = self._load_image(base_image)
            overlay = self._load_image(overlay_image)
            original_size = base.size

            # Resize overlay to match base
            if overlay.size != base.size:
                overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)

            # Ensure RGBA
            if base.mode != 'RGBA':
                base = base.convert('RGBA')
            if overlay.mode != 'RGBA':
                overlay = overlay.convert('RGBA')

            # Apply blend mode
            try:
                import numpy as np

                base_arr = np.array(base, dtype=np.float64) / 255.0
                overlay_arr = np.array(overlay, dtype=np.float64) / 255.0

                if blend_mode == 'multiply':
                    blended = base_arr * overlay_arr
                elif blend_mode == 'screen':
                    blended = 1 - (1 - base_arr) * (1 - overlay_arr)
                elif blend_mode == 'overlay':
                    mask = base_arr < 0.5
                    blended = np.where(
                        mask,
                        2 * base_arr * overlay_arr,
                        1 - 2 * (1 - base_arr) * (1 - overlay_arr)
                    )
                else:  # normal
                    blended = overlay_arr

                # Apply opacity
                result_arr = base_arr * (1 - opacity) + blended * opacity
                result_arr = np.clip(result_arr * 255, 0, 255).astype(np.uint8)

                result_img = Image.fromarray(result_arr, mode='RGBA')

            except ImportError:
                # Simple blend fallback
                result_img = Image.blend(base, overlay, opacity)

            result = OverlayResult(
                success=True,
                original_size=original_size,
                new_size=result_img.size,
                overlay_type='overlay',
                position='full',
                opacity=opacity,
                metadata={'blend_mode': blend_mode}
            )

            if output:
                output_path = Path(output)

                save_kwargs = {}
                ext = output_path.suffix.lower()

                if ext in ['.jpg', '.jpeg']:
                    result_img = result_img.convert('RGB')
                    save_kwargs['quality'] = quality
                elif ext == '.webp':
                    save_kwargs['quality'] = quality

                result_img.save(str(output_path), **save_kwargs)
                result.output_path = str(output_path)
                return result
            else:
                return (result_img, result)

        except Exception as e:
            logger.error(f"Overlay error: {e}")
            return OverlayResult(
                success=False,
                original_size=(0, 0),
                new_size=(0, 0),
                overlay_type='overlay',
                position='full',
                opacity=opacity,
                error=str(e)
            )

    def add_watermark(
        self,
        image: Union[str, Path, Image.Image],
        text_or_image: Union[str, Path, Image.Image],
        position: Union[str, WatermarkPosition] = WatermarkPosition.BOTTOM_RIGHT,
        opacity: float = 0.5,
        output: Optional[Union[str, Path]] = None,
        quality: int = 95,
        **kwargs
    ) -> Union[OverlayResult, Tuple[Image.Image, OverlayResult]]:
        """
        Add a watermark (auto-detect text or image).

        Args:
            image: Input image
            text_or_image: Text string or image path/object
            position: Watermark position
            opacity: Opacity
            output: Output path
            quality: JPEG quality
            **kwargs: Additional options

        Returns:
            OverlayResult or (Image, OverlayResult)
        """
        # Determine if it's text or image
        if isinstance(text_or_image, str):
            # Check if it's a file path
            if Path(text_or_image).exists():
                return self.add_image_watermark(
                    image, text_or_image, position, opacity,
                    output=output, quality=quality, **kwargs
                )
            else:
                # Treat as text
                return self.add_text_watermark(
                    image, text_or_image, position, opacity,
                    output=output, quality=quality, **kwargs
                )
        elif isinstance(text_or_image, Image.Image):
            return self.add_image_watermark(
                image, text_or_image, position, opacity,
                output=output, quality=quality, **kwargs
            )
        else:
            return self.add_text_watermark(
                image, str(text_or_image), position, opacity,
                output=output, quality=quality, **kwargs
            )

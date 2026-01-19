"""
Contact sheet generation operations.

Provides contact sheet and image grid creation functionality.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any, List
import logging

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


@dataclass
class ContactSheetResult:
    """Result of a contact sheet creation."""
    success: bool
    image_count: int
    grid_size: Tuple[int, int]
    output_size: Tuple[int, int]
    output_path: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'success': self.success,
            'image_count': self.image_count,
            'grid_size': self.grid_size,
            'output_size': self.output_size,
            'output_path': self.output_path,
            'error': self.error,
            'metadata': self.metadata,
        }


class ContactSheetGenerator:
    """
    Contact sheet generation engine.

    Creates grid layouts of multiple images with optional
    labels and customizable spacing.
    """

    def __init__(
        self,
        thumb_size: Tuple[int, int] = (150, 150),
        padding: int = 10,
        background_color: Tuple[int, int, int] = (255, 255, 255)
    ):
        """
        Initialize the generator.

        Args:
            thumb_size: Default thumbnail size
            padding: Default padding between images
            background_color: Default background color
        """
        self.thumb_size = thumb_size
        self.padding = padding
        self.background_color = background_color

    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """Load image from path or return if already an Image."""
        if isinstance(image, Image.Image):
            return image
        return Image.open(str(image))

    def _create_thumbnail(
        self,
        image: Image.Image,
        size: Tuple[int, int],
        mode: str = 'fit'
    ) -> Image.Image:
        """Create a thumbnail of the image."""
        img = image.copy()

        if mode == 'fit':
            # Fit within bounds, maintain aspect ratio
            img.thumbnail(size, Image.Resampling.LANCZOS)

            # Center on background
            thumb = Image.new('RGB', size, self.background_color)
            offset = ((size[0] - img.width) // 2, (size[1] - img.height) // 2)

            if img.mode == 'RGBA':
                thumb.paste(img, offset, img.split()[3])
            else:
                thumb.paste(img, offset)
            return thumb

        elif mode == 'fill':
            # Fill target, center crop
            orig_ratio = image.width / image.height
            target_ratio = size[0] / size[1]

            if orig_ratio > target_ratio:
                new_height = size[1]
                new_width = int(new_height * orig_ratio)
            else:
                new_width = size[0]
                new_height = int(new_width / orig_ratio)

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            left = (new_width - size[0]) // 2
            top = (new_height - size[1]) // 2
            return img.crop((left, top, left + size[0], top + size[1]))

        else:
            img.thumbnail(size, Image.Resampling.LANCZOS)
            return img

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Get a font for labels."""
        try:
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
            return ImageFont.load_default()
        except Exception:
            return ImageFont.load_default()

    def create(
        self,
        images: List[Union[str, Path, Image.Image]],
        output: Union[str, Path],
        columns: int = 4,
        thumb_size: Optional[Tuple[int, int]] = None,
        padding: Optional[int] = None,
        background_color: Optional[Tuple[int, int, int]] = None,
        show_labels: bool = False,
        title: Optional[str] = None,
        quality: int = 95
    ) -> ContactSheetResult:
        """
        Create a contact sheet from multiple images.

        Args:
            images: List of input images (paths or PIL Images)
            output: Output path
            columns: Number of columns
            thumb_size: Thumbnail size
            padding: Padding between images
            background_color: Background color
            show_labels: Whether to show image labels
            title: Optional title for the contact sheet
            quality: JPEG quality (1-100)

        Returns:
            ContactSheetResult with sheet details
        """
        try:
            if not images:
                return ContactSheetResult(
                    success=False,
                    image_count=0,
                    grid_size=(0, 0),
                    output_size=(0, 0),
                    error="No images provided"
                )

            # Use defaults if not specified
            thumb_size = thumb_size or self.thumb_size
            padding = padding if padding is not None else self.padding
            background_color = background_color or self.background_color

            # Calculate grid dimensions
            image_count = len(images)
            rows = (image_count + columns - 1) // columns

            # Calculate label height
            label_height = 20 if show_labels else 0
            title_height = 40 if title else 0

            # Calculate sheet size
            cell_width = thumb_size[0] + padding
            cell_height = thumb_size[1] + padding + label_height

            sheet_width = columns * cell_width + padding
            sheet_height = rows * cell_height + padding + title_height

            # Create contact sheet
            sheet = Image.new('RGB', (sheet_width, sheet_height), background_color)
            draw = ImageDraw.Draw(sheet)

            # Add title
            if title:
                title_font = self._get_font(24)
                title_bbox = draw.textbbox((0, 0), title, font=title_font)
                title_x = (sheet_width - (title_bbox[2] - title_bbox[0])) // 2
                draw.text((title_x, 10), title, fill=(0, 0, 0), font=title_font)

            # Add thumbnails
            label_font = self._get_font(12) if show_labels else None

            for i, img_source in enumerate(images):
                row = i // columns
                col = i % columns

                # Calculate position
                x = col * cell_width + padding
                y = row * cell_height + padding + title_height

                # Load and create thumbnail
                try:
                    img = self._load_image(img_source)
                    thumb = self._create_thumbnail(img, thumb_size)

                    # Paste thumbnail
                    sheet.paste(thumb, (x, y))

                    # Add label
                    if show_labels and label_font:
                        if isinstance(img_source, (str, Path)):
                            label = Path(img_source).name
                            # Truncate if too long
                            max_chars = thumb_size[0] // 7
                            if len(label) > max_chars:
                                label = label[:max_chars-3] + '...'
                        else:
                            label = f"Image {i + 1}"

                        label_y = y + thumb_size[1] + 2
                        draw.text((x, label_y), label, fill=(0, 0, 0), font=label_font)

                except Exception as e:
                    logger.warning(f"Failed to add image {i}: {e}")
                    # Draw placeholder
                    draw.rectangle(
                        [x, y, x + thumb_size[0], y + thumb_size[1]],
                        fill=(200, 200, 200),
                        outline=(150, 150, 150)
                    )
                    draw.text(
                        (x + 10, y + thumb_size[1] // 2),
                        "Error",
                        fill=(100, 100, 100)
                    )

            # Save contact sheet
            output_path = Path(output)

            save_kwargs = {}
            ext = output_path.suffix.lower()

            if ext in ['.jpg', '.jpeg']:
                save_kwargs['quality'] = quality
                save_kwargs['optimize'] = True
            elif ext == '.png':
                save_kwargs['optimize'] = True
            elif ext == '.webp':
                save_kwargs['quality'] = quality

            sheet.save(str(output_path), **save_kwargs)

            return ContactSheetResult(
                success=True,
                image_count=image_count,
                grid_size=(columns, rows),
                output_size=sheet.size,
                output_path=str(output_path),
                metadata={
                    'thumb_size': thumb_size,
                    'padding': padding,
                    'title': title,
                    'show_labels': show_labels,
                }
            )

        except Exception as e:
            logger.error(f"Contact sheet error: {e}")
            return ContactSheetResult(
                success=False,
                image_count=len(images) if images else 0,
                grid_size=(0, 0),
                output_size=(0, 0),
                error=str(e)
            )

    def create_from_directory(
        self,
        directory: Union[str, Path],
        output: Union[str, Path],
        columns: int = 4,
        extensions: List[str] = None,
        **kwargs
    ) -> ContactSheetResult:
        """
        Create a contact sheet from all images in a directory.

        Args:
            directory: Directory containing images
            output: Output path
            columns: Number of columns
            extensions: File extensions to include
            **kwargs: Additional options for create()

        Returns:
            ContactSheetResult with sheet details
        """
        directory = Path(directory)

        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']

        # Find all images
        images = []
        for ext in extensions:
            images.extend(directory.glob(f'*{ext}'))
            images.extend(directory.glob(f'*{ext.upper()}'))

        # Sort by name
        images = sorted(set(images))

        if not images:
            return ContactSheetResult(
                success=False,
                image_count=0,
                grid_size=(0, 0),
                output_size=(0, 0),
                error=f"No images found in {directory}"
            )

        return self.create(images, output, columns, **kwargs)

    def create_comparison(
        self,
        image_pairs: List[Tuple[Union[str, Path, Image.Image], Union[str, Path, Image.Image]]],
        output: Union[str, Path],
        labels: Optional[Tuple[str, str]] = None,
        quality: int = 95
    ) -> ContactSheetResult:
        """
        Create a before/after comparison sheet.

        Args:
            image_pairs: List of (before, after) image pairs
            output: Output path
            labels: Labels for columns (e.g., ("Before", "After"))
            quality: JPEG quality

        Returns:
            ContactSheetResult with sheet details
        """
        try:
            if not image_pairs:
                return ContactSheetResult(
                    success=False,
                    image_count=0,
                    grid_size=(0, 0),
                    output_size=(0, 0),
                    error="No image pairs provided"
                )

            thumb_size = self.thumb_size
            padding = self.padding
            label_height = 30 if labels else 0

            # Calculate dimensions
            pair_count = len(image_pairs)
            sheet_width = 2 * thumb_size[0] + 3 * padding
            sheet_height = pair_count * (thumb_size[1] + padding) + padding + label_height

            sheet = Image.new('RGB', (sheet_width, sheet_height), self.background_color)
            draw = ImageDraw.Draw(sheet)

            # Add column labels
            if labels:
                font = self._get_font(16)
                x1 = padding + thumb_size[0] // 2
                x2 = padding * 2 + thumb_size[0] + thumb_size[0] // 2

                draw.text((x1 - len(labels[0]) * 4, 5), labels[0], fill=(0, 0, 0), font=font)
                draw.text((x2 - len(labels[1]) * 4, 5), labels[1], fill=(0, 0, 0), font=font)

            # Add image pairs
            for i, (before, after) in enumerate(image_pairs):
                y = i * (thumb_size[1] + padding) + padding + label_height

                # Before image
                try:
                    before_img = self._load_image(before)
                    before_thumb = self._create_thumbnail(before_img, thumb_size)
                    sheet.paste(before_thumb, (padding, y))
                except Exception as e:
                    logger.warning(f"Failed to add before image {i}: {e}")

                # After image
                try:
                    after_img = self._load_image(after)
                    after_thumb = self._create_thumbnail(after_img, thumb_size)
                    sheet.paste(after_thumb, (padding * 2 + thumb_size[0], y))
                except Exception as e:
                    logger.warning(f"Failed to add after image {i}: {e}")

            # Save
            output_path = Path(output)

            save_kwargs = {}
            ext = output_path.suffix.lower()

            if ext in ['.jpg', '.jpeg']:
                save_kwargs['quality'] = quality
            elif ext == '.webp':
                save_kwargs['quality'] = quality

            sheet.save(str(output_path), **save_kwargs)

            return ContactSheetResult(
                success=True,
                image_count=len(image_pairs) * 2,
                grid_size=(2, len(image_pairs)),
                output_size=sheet.size,
                output_path=str(output_path),
                metadata={'type': 'comparison', 'labels': labels}
            )

        except Exception as e:
            logger.error(f"Comparison sheet error: {e}")
            return ContactSheetResult(
                success=False,
                image_count=0,
                grid_size=(0, 0),
                output_size=(0, 0),
                error=str(e)
            )

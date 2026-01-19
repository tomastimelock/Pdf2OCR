"""Color Analysis Module.

Provides comprehensive color extraction, palette generation, histogram analysis,
and color harmony evaluation for images.
"""

import colorsys
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class ColorAnalysisError(Exception):
    """Error during color analysis."""
    pass


# Color conversion utilities
def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB to hex color string."""
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join(c * 2 for c in hex_color)
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hsl(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert RGB (0-255) to HSL (H: 0-360, S: 0-1, L: 0-1)."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return (h * 360, s, l)


def hsl_to_rgb(h: float, s: float, l: float) -> Tuple[int, int, int]:
    """Convert HSL (H: 0-360, S: 0-1, L: 0-1) to RGB (0-255)."""
    r, g, b = colorsys.hls_to_rgb(h / 360, l, s)
    return (int(r * 255), int(g * 255), int(b * 255))


def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert RGB (0-255) to HSV (H: 0-360, S: 0-1, V: 0-1)."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return (h * 360, s, v)


def color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    """Calculate Euclidean distance between two colors."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


# Basic color names with RGB values
COLOR_NAMES = {
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'orange': (255, 165, 0),
    'pink': (255, 192, 203),
    'purple': (128, 0, 128),
    'brown': (139, 69, 19),
    'gray': (128, 128, 128),
    'navy': (0, 0, 128),
    'teal': (0, 128, 128),
    'olive': (128, 128, 0),
    'maroon': (128, 0, 0),
    'lime': (0, 255, 0),
    'aqua': (0, 255, 255),
    'silver': (192, 192, 192),
    'gold': (255, 215, 0),
    'coral': (255, 127, 80),
    'salmon': (250, 128, 114),
    'khaki': (240, 230, 140),
    'plum': (221, 160, 221),
    'violet': (238, 130, 238),
    'indigo': (75, 0, 130),
    'turquoise': (64, 224, 208),
    'tan': (210, 180, 140),
    'beige': (245, 245, 220),
    'ivory': (255, 255, 240),
    'lavender': (230, 230, 250),
    'crimson': (220, 20, 60),
    'chocolate': (210, 105, 30),
}


def get_color_name(r: int, g: int, b: int) -> str:
    """Get the closest named color for an RGB value."""
    min_distance = float('inf')
    closest_name = 'unknown'

    for name, rgb in COLOR_NAMES.items():
        dist = color_distance((r, g, b), rgb)
        if dist < min_distance:
            min_distance = dist
            closest_name = name

    return closest_name


@dataclass
class ColorInfo:
    """Information about a color."""
    rgb: Tuple[int, int, int]
    count: int = 0
    percentage: float = 0.0

    @property
    def hex(self) -> str:
        return rgb_to_hex(*self.rgb)

    @property
    def hsl(self) -> Tuple[float, float, float]:
        return rgb_to_hsl(*self.rgb)

    @property
    def hsv(self) -> Tuple[float, float, float]:
        return rgb_to_hsv(*self.rgb)

    @property
    def name(self) -> str:
        return get_color_name(*self.rgb)

    @property
    def brightness(self) -> float:
        """Perceived brightness (0-1)."""
        r, g, b = self.rgb
        return (0.299 * r + 0.587 * g + 0.114 * b) / 255

    @property
    def is_light(self) -> bool:
        return self.brightness > 0.5

    @property
    def is_dark(self) -> bool:
        return self.brightness <= 0.5

    @property
    def is_saturated(self) -> bool:
        _, s, _ = self.hsv
        return s > 0.5

    def complementary(self) -> 'ColorInfo':
        """Get complementary color."""
        h, s, l = self.hsl
        new_h = (h + 180) % 360
        new_rgb = hsl_to_rgb(new_h, s, l)
        return ColorInfo(rgb=new_rgb)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'rgb': self.rgb,
            'hex': self.hex,
            'hsl': self.hsl,
            'name': self.name,
            'count': self.count,
            'percentage': self.percentage,
            'brightness': self.brightness,
        }


@dataclass
class ColorPalette:
    """A color palette extracted from an image."""
    colors: List[ColorInfo] = field(default_factory=list)
    source_image: str = ""

    @property
    def dominant(self) -> Optional[ColorInfo]:
        return self.colors[0] if self.colors else None

    @property
    def hex_list(self) -> List[str]:
        return [c.hex for c in self.colors]

    @property
    def rgb_list(self) -> List[Tuple[int, int, int]]:
        return [c.rgb for c in self.colors]

    @property
    def is_mostly_light(self) -> bool:
        if not self.colors:
            return False
        avg_brightness = sum(c.brightness * c.percentage for c in self.colors)
        return avg_brightness > 0.5

    @property
    def is_mostly_dark(self) -> bool:
        return not self.is_mostly_light

    @property
    def is_monochromatic(self) -> bool:
        """Check if palette is mostly one hue."""
        if len(self.colors) < 2:
            return True
        hues = [c.hsl[0] for c in self.colors[:3]]
        hue_range = max(hues) - min(hues)
        return hue_range < 30 or hue_range > 330

    def to_dict(self) -> Dict[str, Any]:
        return {
            'colors': [c.to_dict() for c in self.colors],
            'hex_list': self.hex_list,
            'is_mostly_light': self.is_mostly_light,
            'is_monochromatic': self.is_monochromatic,
            'source_image': self.source_image,
        }


@dataclass
class ColorHistogram:
    """Color histogram data."""
    red: List[int] = field(default_factory=list)
    green: List[int] = field(default_factory=list)
    blue: List[int] = field(default_factory=list)
    luminance: List[int] = field(default_factory=list)

    @property
    def red_mean(self) -> float:
        return self._channel_mean(self.red)

    @property
    def green_mean(self) -> float:
        return self._channel_mean(self.green)

    @property
    def blue_mean(self) -> float:
        return self._channel_mean(self.blue)

    @property
    def luminance_mean(self) -> float:
        return self._channel_mean(self.luminance)

    def _channel_mean(self, channel: List[int]) -> float:
        if not channel:
            return 0
        total = sum(channel)
        if total == 0:
            return 0
        return sum(i * v for i, v in enumerate(channel)) / total

    def to_dict(self) -> Dict[str, Any]:
        return {
            'red': self.red,
            'green': self.green,
            'blue': self.blue,
            'luminance': self.luminance,
            'red_mean': self.red_mean,
            'green_mean': self.green_mean,
            'blue_mean': self.blue_mean,
        }


class ColorAnalyzer:
    """Comprehensive color analysis for images."""

    def __init__(
        self,
        sample_size: int = 10000,
        quantize_bits: int = 4,
    ):
        """
        Initialize color analyzer.

        Args:
            sample_size: Maximum number of pixels to sample
            quantize_bits: Bits per channel for color quantization (1-8)
        """
        self.sample_size = sample_size
        self.quantize_bits = min(8, max(1, quantize_bits))

    def _load_image(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load image as numpy array (RGB)."""
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                # Grayscale - convert to RGB
                return np.stack([image, image, image], axis=-1)
            return image

        try:
            from PIL import Image
            img = Image.open(image).convert('RGB')
            return np.array(img)
        except ImportError:
            import cv2
            img = cv2.imread(str(image))
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _quantize_color(self, color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Reduce color precision for grouping similar colors."""
        shift = 8 - self.quantize_bits
        mask = 0xFF << shift
        return tuple((c & mask) + (1 << (shift - 1)) if shift > 0 else c for c in color)

    def get_dominant_colors(
        self,
        image: Union[str, Path, np.ndarray],
        count: int = 5,
        exclude_white: bool = False,
        exclude_black: bool = False,
    ) -> List[ColorInfo]:
        """
        Extract dominant colors from an image.

        Args:
            image: Image path or numpy array
            count: Number of colors to extract
            exclude_white: Exclude near-white colors
            exclude_black: Exclude near-black colors

        Returns:
            List of ColorInfo objects sorted by frequency
        """
        img = self._load_image(image)

        # Resize for faster processing
        h, w = img.shape[:2]
        total_pixels = h * w
        if total_pixels > self.sample_size:
            ratio = math.sqrt(self.sample_size / total_pixels)
            new_h = max(1, int(h * ratio))
            new_w = max(1, int(w * ratio))
            try:
                from PIL import Image
                pil_img = Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS)
                img = np.array(pil_img)
            except ImportError:
                import cv2
                img = cv2.resize(img, (new_w, new_h))

        # Get pixels and quantize
        pixels = img.reshape(-1, 3)
        quantized = [self._quantize_color(tuple(p)) for p in pixels]

        # Filter colors if requested
        if exclude_white or exclude_black:
            filtered = []
            for q in quantized:
                brightness = sum(q) / 3
                if exclude_white and brightness > 240:
                    continue
                if exclude_black and brightness < 15:
                    continue
                filtered.append(q)
            quantized = filtered if filtered else quantized

        # Count colors
        color_counts = Counter(quantized)
        total = len(quantized)

        # Get top colors
        top_colors = color_counts.most_common(count)

        return [
            ColorInfo(
                rgb=color,
                count=cnt,
                percentage=cnt / total if total > 0 else 0,
            )
            for color, cnt in top_colors
        ]

    def get_palette(
        self,
        image: Union[str, Path, np.ndarray],
        count: int = 8,
    ) -> ColorPalette:
        """
        Generate a color palette from an image.

        Args:
            image: Image path or numpy array
            count: Number of colors in palette

        Returns:
            ColorPalette object
        """
        colors = self.get_dominant_colors(image, count=count)
        path = str(image) if not isinstance(image, np.ndarray) else ""
        return ColorPalette(colors=colors, source_image=path)

    def get_histogram(self, image: Union[str, Path, np.ndarray]) -> ColorHistogram:
        """
        Get color histogram for an image.

        Args:
            image: Image path or numpy array

        Returns:
            ColorHistogram object with channel data
        """
        img = self._load_image(image)

        # Calculate histogram for each channel
        red = np.histogram(img[:, :, 0], bins=256, range=(0, 256))[0].tolist()
        green = np.histogram(img[:, :, 1], bins=256, range=(0, 256))[0].tolist()
        blue = np.histogram(img[:, :, 2], bins=256, range=(0, 256))[0].tolist()

        # Calculate luminance
        luminance_values = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)
        luminance = np.histogram(luminance_values, bins=256, range=(0, 256))[0].tolist()

        return ColorHistogram(red=red, green=green, blue=blue, luminance=luminance)

    def get_average_color(self, image: Union[str, Path, np.ndarray]) -> ColorInfo:
        """Get the average color of an image."""
        img = self._load_image(image)

        # Resize for speed
        try:
            from PIL import Image
            pil_img = Image.fromarray(img).resize((100, 100), Image.LANCZOS)
            img = np.array(pil_img)
        except ImportError:
            import cv2
            img = cv2.resize(img, (100, 100))

        avg = np.mean(img.reshape(-1, 3), axis=0).astype(int)

        return ColorInfo(
            rgb=tuple(avg),
            count=img.shape[0] * img.shape[1],
            percentage=1.0,
        )

    def get_brightness(self, image: Union[str, Path, np.ndarray]) -> float:
        """Get overall brightness of an image (0-1)."""
        avg = self.get_average_color(image)
        return avg.brightness

    def get_contrast(self, image: Union[str, Path, np.ndarray]) -> float:
        """Get contrast level of an image (0-1)."""
        hist = self.get_histogram(image)

        # Calculate standard deviation of luminance
        total = sum(hist.luminance)
        if total == 0:
            return 0

        mean = sum(i * v for i, v in enumerate(hist.luminance)) / total
        variance = sum(((i - mean) ** 2) * v for i, v in enumerate(hist.luminance)) / total
        std_dev = math.sqrt(variance)

        # Normalize to 0-1 (max std dev for uniform distribution is ~73.9)
        return min(1.0, std_dev / 74)

    def get_saturation(self, image: Union[str, Path, np.ndarray]) -> float:
        """Get average saturation of an image (0-1)."""
        img = self._load_image(image)

        # Convert to HSV
        pixels = img.reshape(-1, 3)
        saturations = []

        for p in pixels[::max(1, len(pixels) // 1000)]:  # Sample
            _, s, _ = colorsys.rgb_to_hsv(p[0] / 255, p[1] / 255, p[2] / 255)
            saturations.append(s)

        return sum(saturations) / len(saturations) if saturations else 0

    def analyze(
        self,
        image: Union[str, Path, np.ndarray],
        palette_size: int = 8,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive color analysis.

        Args:
            image: Image path or numpy array
            palette_size: Number of colors in palette

        Returns:
            Dictionary with all analysis results
        """
        return {
            'dominant_colors': [c.to_dict() for c in self.get_dominant_colors(image, 5)],
            'palette': self.get_palette(image, palette_size).to_dict(),
            'histogram': self.get_histogram(image).to_dict(),
            'average_color': self.get_average_color(image).to_dict(),
            'brightness': self.get_brightness(image),
            'contrast': self.get_contrast(image),
            'saturation': self.get_saturation(image),
        }


# Convenience functions
def analyze_colors(
    image: Union[str, Path, np.ndarray],
    palette_size: int = 8,
) -> Dict[str, Any]:
    """Perform comprehensive color analysis."""
    analyzer = ColorAnalyzer()
    return analyzer.analyze(image, palette_size)


def get_dominant_colors(
    image: Union[str, Path, np.ndarray],
    count: int = 5,
) -> List[ColorInfo]:
    """Get dominant colors from an image."""
    analyzer = ColorAnalyzer()
    return analyzer.get_dominant_colors(image, count)


def get_palette(
    image: Union[str, Path, np.ndarray],
    count: int = 8,
) -> ColorPalette:
    """Get color palette from an image."""
    analyzer = ColorAnalyzer()
    return analyzer.get_palette(image, count)

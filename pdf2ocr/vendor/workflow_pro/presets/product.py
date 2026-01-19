"""Product Photography Workflow Preset.

Pre-configured workflow for e-commerce product photography.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum


class Platform(Enum):
    """E-commerce platform presets."""
    AMAZON = "amazon"
    EBAY = "ebay"
    SHOPIFY = "shopify"
    ETSY = "etsy"
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    PINTEREST = "pinterest"
    WEBSITE = "website"


@dataclass
class BackgroundConfig:
    """Background removal configuration."""
    remove: bool = True
    color: Tuple[int, int, int] = (255, 255, 255)
    transparent: bool = True
    edge_refinement: bool = True


@dataclass
class SizeConfig:
    """Image sizing configuration."""
    main_size: Tuple[int, int] = (2000, 2000)
    thumbnail_size: Tuple[int, int] = (500, 500)
    maintain_aspect: bool = True
    pad_to_square: bool = True


@dataclass
class MockupConfig:
    """Mockup generation configuration."""
    enabled: bool = True
    templates: List[str] = field(default_factory=lambda: ['lifestyle', 'white_bg', 'shadow'])
    count: int = 3


@dataclass
class WatermarkConfig:
    """Watermark configuration."""
    enabled: bool = False
    text: str = ""
    opacity: float = 0.3
    position: str = "bottom-right"


@dataclass
class ProductPresetConfig:
    """Complete product workflow configuration."""
    platforms: List[Platform] = field(default_factory=lambda: [Platform.AMAZON, Platform.SHOPIFY])
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    sizing: SizeConfig = field(default_factory=SizeConfig)
    mockups: MockupConfig = field(default_factory=MockupConfig)
    watermark: WatermarkConfig = field(default_factory=WatermarkConfig)
    extract_colors: bool = True
    quality: int = 90

    def to_dict(self) -> Dict:
        return {
            'platforms': [p.value for p in self.platforms],
            'background': {
                'remove': self.background.remove,
                'transparent': self.background.transparent,
            },
            'sizing': {
                'main_size': self.sizing.main_size,
                'thumbnail_size': self.sizing.thumbnail_size,
            },
            'mockups': {
                'enabled': self.mockups.enabled,
                'count': self.mockups.count,
            },
            'extract_colors': self.extract_colors,
        }


class ProductPreset:
    """Product photography workflow preset.

    Provides pre-configured workflows for product photography including:
    - Amazon listing
    - Multi-platform e-commerce
    - Social media
    - Catalog/wholesale
    """

    AMAZON = ProductPresetConfig(
        platforms=[Platform.AMAZON],
        background=BackgroundConfig(remove=True, color=(255, 255, 255)),
        sizing=SizeConfig(main_size=(2000, 2000), thumbnail_size=(500, 500)),
        mockups=MockupConfig(enabled=False),
        extract_colors=True,
    )

    MULTI_PLATFORM = ProductPresetConfig(
        platforms=[Platform.AMAZON, Platform.EBAY, Platform.SHOPIFY, Platform.ETSY],
        background=BackgroundConfig(remove=True, transparent=True),
        sizing=SizeConfig(main_size=(2048, 2048)),
        mockups=MockupConfig(enabled=True, count=3),
        extract_colors=True,
    )

    SOCIAL_MEDIA = ProductPresetConfig(
        platforms=[Platform.INSTAGRAM, Platform.FACEBOOK, Platform.PINTEREST],
        background=BackgroundConfig(remove=True),
        sizing=SizeConfig(main_size=(1080, 1080)),
        mockups=MockupConfig(enabled=True, templates=['lifestyle']),
        extract_colors=True,
    )

    CATALOG = ProductPresetConfig(
        platforms=[Platform.WEBSITE],
        background=BackgroundConfig(remove=True),
        sizing=SizeConfig(main_size=(1920, 1920), thumbnail_size=(400, 400)),
        mockups=MockupConfig(enabled=False),
        watermark=WatermarkConfig(enabled=True, text="SAMPLE"),
        extract_colors=True,
    )

    def __init__(self, config: Optional[ProductPresetConfig] = None):
        """
        Initialize product preset.

        Args:
            config: Custom configuration or use one of the preset constants
        """
        self.config = config or ProductPresetConfig()
        self._studio = None

    def _get_studio(self):
        """Lazy load ProductStudio."""
        if self._studio is None:
            try:
                from product_studio import ProductStudio
                self._studio = ProductStudio()
            except ImportError:
                pass
        return self._studio

    def process(
        self,
        image: Union[str, Path],
        output_dir: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Process product image using preset configuration.

        Args:
            image: Product image path
            output_dir: Output directory

        Returns:
            Processing result dictionary
        """
        img_path = Path(image)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        result = {
            'source': str(img_path),
            'output': str(output),
            'success': False,
        }

        studio = self._get_studio()

        if studio:
            try:
                # Convert Platform enum to studio Platform
                from product_studio import Platform as StudioPlatform

                platform_list = []
                for p in self.config.platforms:
                    try:
                        platform_list.append(StudioPlatform(p.value))
                    except ValueError:
                        pass

                package = studio.process_product(
                    image=img_path,
                    output_dir=output,
                    platforms=platform_list or None,
                    remove_background=self.config.background.remove,
                    generate_mockups=self.config.mockups.enabled,
                    extract_colors=self.config.extract_colors,
                    add_watermark=self.config.watermark.enabled,
                )

                result.update({
                    'total_assets': package.total_assets,
                    'assets': [a.to_dict() for a in package.assets],
                    'colors': package.colors.to_dict() if package.colors else None,
                    'transparent_png': package.transparent_png,
                    'success': package.success,
                })

            except Exception as e:
                result['error'] = str(e)
        else:
            result['error'] = "ProductStudio module not available"

        return result

    def batch_process(
        self,
        directory: Union[str, Path],
        output_dir: Union[str, Path],
    ) -> List[Dict[str, Any]]:
        """
        Process multiple product images.

        Args:
            directory: Directory containing product images
            output_dir: Output base directory

        Returns:
            List of processing results
        """
        source = Path(directory)
        output = Path(output_dir)
        results = []

        extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
        images = []
        for ext in extensions:
            images.extend(source.glob(ext))

        for img in images:
            result = self.process(img, output / img.stem)
            results.append(result)

        return results

    def create_amazon_preset(
        self,
        seller_name: Optional[str] = None,
    ) -> ProductPresetConfig:
        """Create Amazon-optimized preset."""
        return ProductPresetConfig(
            platforms=[Platform.AMAZON],
            background=BackgroundConfig(
                remove=True,
                color=(255, 255, 255),
                transparent=False,
            ),
            sizing=SizeConfig(
                main_size=(2000, 2000),
                thumbnail_size=(500, 500),
                pad_to_square=True,
            ),
            mockups=MockupConfig(enabled=False),
            watermark=WatermarkConfig(enabled=False),
            quality=95,
        )

    def create_social_preset(
        self,
        brand_name: str,
        watermark: bool = True,
    ) -> ProductPresetConfig:
        """Create social media optimized preset."""
        return ProductPresetConfig(
            platforms=[Platform.INSTAGRAM, Platform.FACEBOOK, Platform.PINTEREST],
            background=BackgroundConfig(remove=True),
            sizing=SizeConfig(main_size=(1080, 1080)),
            mockups=MockupConfig(enabled=True, templates=['lifestyle'], count=2),
            watermark=WatermarkConfig(
                enabled=watermark,
                text=f"@{brand_name}",
                opacity=0.25,
            ),
        )

    @staticmethod
    def list_presets() -> List[str]:
        """List available preset names."""
        return ['AMAZON', 'MULTI_PLATFORM', 'SOCIAL_MEDIA', 'CATALOG']

    @classmethod
    def get_preset(cls, name: str) -> ProductPresetConfig:
        """Get a preset by name."""
        presets = {
            'AMAZON': cls.AMAZON,
            'MULTI_PLATFORM': cls.MULTI_PLATFORM,
            'SOCIAL_MEDIA': cls.SOCIAL_MEDIA,
            'CATALOG': cls.CATALOG,
        }
        return presets.get(name.upper(), ProductPresetConfig())

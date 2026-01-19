"""Print Preparation Workflow Preset.

Pre-configured workflow for professional print output.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum


class PrintProfile(Enum):
    """Print output profiles."""
    PHOTO_GLOSSY = "photo_glossy"
    PHOTO_MATTE = "photo_matte"
    FINE_ART = "fine_art"
    CANVAS = "canvas"
    METAL = "metal"
    MAGAZINE = "magazine"
    NEWSPAPER = "newspaper"


class PaperSize(Enum):
    """Standard paper sizes."""
    A4 = "a4"
    A3 = "a3"
    LETTER = "letter"
    PHOTO_4X6 = "4x6"
    PHOTO_5X7 = "5x7"
    PHOTO_8X10 = "8x10"
    PHOTO_11X14 = "11x14"
    CUSTOM = "custom"


class ColorSpace(Enum):
    """Color space options."""
    SRGB = "srgb"
    ADOBE_RGB = "adobe_rgb"
    PROPHOTO_RGB = "prophoto_rgb"
    CMYK = "cmyk"


@dataclass
class RawConfig:
    """RAW development configuration."""
    auto_exposure: bool = True
    exposure_compensation: float = 0.0
    white_balance: str = "auto"
    highlight_recovery: bool = True
    shadow_recovery: bool = True


@dataclass
class CorrectionConfig:
    """Image correction configuration."""
    lens_correction: bool = True
    chromatic_aberration: bool = True
    vignette_correction: bool = True
    distortion_correction: bool = True


@dataclass
class EnhancementConfig:
    """Enhancement configuration."""
    noise_reduction: float = 0.5
    sharpening: float = 0.5
    output_sharpening: bool = True
    color_enhancement: float = 0.0


@dataclass
class OutputConfig:
    """Output configuration."""
    profile: PrintProfile = PrintProfile.PHOTO_GLOSSY
    paper_size: PaperSize = PaperSize.PHOTO_8X10
    color_space: ColorSpace = ColorSpace.SRGB
    dpi: int = 300
    format: str = "tiff"
    bit_depth: int = 16


@dataclass
class PrintPresetConfig:
    """Complete print workflow configuration."""
    raw: RawConfig = field(default_factory=RawConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    enhancement: EnhancementConfig = field(default_factory=EnhancementConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    create_proof: bool = True
    proof_watermark: str = "PROOF"

    def to_dict(self) -> Dict:
        return {
            'raw': {
                'auto_exposure': self.raw.auto_exposure,
                'white_balance': self.raw.white_balance,
            },
            'correction': {
                'lens_correction': self.correction.lens_correction,
                'chromatic_aberration': self.correction.chromatic_aberration,
            },
            'enhancement': {
                'noise_reduction': self.enhancement.noise_reduction,
                'sharpening': self.enhancement.sharpening,
            },
            'output': {
                'profile': self.output.profile.value,
                'dpi': self.output.dpi,
                'format': self.output.format,
            },
            'create_proof': self.create_proof,
        }


class PrintPreset:
    """Print preparation workflow preset.

    Provides pre-configured workflows for print output including:
    - Photo prints (glossy/matte)
    - Fine art prints
    - Canvas/metal prints
    - Magazine/publication
    """

    PHOTO_PRINT = PrintPresetConfig(
        raw=RawConfig(auto_exposure=True),
        correction=CorrectionConfig(lens_correction=True),
        enhancement=EnhancementConfig(sharpening=0.6, output_sharpening=True),
        output=OutputConfig(
            profile=PrintProfile.PHOTO_GLOSSY,
            dpi=300,
            format='jpeg',
        ),
    )

    FINE_ART = PrintPresetConfig(
        raw=RawConfig(auto_exposure=True, highlight_recovery=True),
        correction=CorrectionConfig(lens_correction=True),
        enhancement=EnhancementConfig(
            noise_reduction=0.3,
            sharpening=0.4,
            output_sharpening=True,
        ),
        output=OutputConfig(
            profile=PrintProfile.FINE_ART,
            color_space=ColorSpace.ADOBE_RGB,
            dpi=360,
            format='tiff',
            bit_depth=16,
        ),
    )

    CANVAS = PrintPresetConfig(
        raw=RawConfig(auto_exposure=True),
        correction=CorrectionConfig(lens_correction=True),
        enhancement=EnhancementConfig(
            noise_reduction=0.4,
            sharpening=0.3,  # Less sharpening for canvas
        ),
        output=OutputConfig(
            profile=PrintProfile.CANVAS,
            dpi=150,  # Lower DPI for canvas
            format='tiff',
        ),
    )

    MAGAZINE = PrintPresetConfig(
        raw=RawConfig(auto_exposure=True),
        correction=CorrectionConfig(lens_correction=True),
        enhancement=EnhancementConfig(
            sharpening=0.7,
            output_sharpening=True,
        ),
        output=OutputConfig(
            profile=PrintProfile.MAGAZINE,
            color_space=ColorSpace.CMYK,
            dpi=300,
            format='tiff',
        ),
    )

    def __init__(self, config: Optional[PrintPresetConfig] = None):
        """
        Initialize print preset.

        Args:
            config: Custom configuration or use one of the preset constants
        """
        self.config = config or PrintPresetConfig()
        self._workflow = None

    def _get_workflow(self):
        """Lazy load ProPrintWorkflow."""
        if self._workflow is None:
            try:
                from pro_print_workflow import ProPrintWorkflow, ProcessingSettings

                settings = ProcessingSettings(
                    auto_exposure=self.config.raw.auto_exposure,
                    exposure_compensation=self.config.raw.exposure_compensation,
                    lens_correction=self.config.correction.lens_correction,
                    chromatic_aberration=self.config.correction.chromatic_aberration,
                    vignette_correction=self.config.correction.vignette_correction,
                    distortion_correction=self.config.correction.distortion_correction,
                    noise_reduction=self.config.enhancement.noise_reduction,
                    sharpening=self.config.enhancement.sharpening,
                    output_sharpening=self.config.enhancement.output_sharpening,
                    output_dpi=self.config.output.dpi,
                    output_format=self.config.output.format,
                )

                self._workflow = ProPrintWorkflow(settings=settings)
            except ImportError:
                pass
        return self._workflow

    def process(
        self,
        image: Union[str, Path],
        output: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Process image for print using preset configuration.

        Args:
            image: Input image path
            output: Output path

        Returns:
            Processing result dictionary
        """
        img_path = Path(image)
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        result = {
            'source': str(img_path),
            'output': str(out_path),
            'success': False,
        }

        workflow = self._get_workflow()

        if workflow:
            try:
                wf_result = workflow.process(img_path, out_path)

                result.update({
                    'final_output': wf_result.final_output,
                    'print_ready': wf_result.print_ready,
                    'duration': wf_result.total_duration,
                    'color_info': wf_result.color_info,
                    'steps': [s.to_dict() for s in wf_result.steps],
                    'success': wf_result.success,
                })

                # Create proof if configured
                if self.config.create_proof and wf_result.success:
                    proof_result = workflow.create_proof(
                        img_path,
                        out_path.parent,
                        watermark_text=self.config.proof_watermark,
                    )
                    result['proof'] = proof_result.to_dict()

            except Exception as e:
                result['error'] = str(e)
        else:
            result['error'] = "ProPrintWorkflow module not available"

        return result

    def batch_process(
        self,
        directory: Union[str, Path],
        output_dir: Union[str, Path],
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images for print.

        Args:
            directory: Input directory
            output_dir: Output directory

        Returns:
            List of processing results
        """
        source = Path(directory)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        results = []

        extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.cr2', '*.nef', '*.arw', '*.dng']
        images = []
        for ext in extensions:
            images.extend(source.glob(ext))
            images.extend(source.glob(ext.upper()))

        for img in images:
            out_ext = f".{self.config.output.format}"
            out_path = output / f"{img.stem}_print{out_ext}"
            result = self.process(img, out_path)
            results.append(result)

        return results

    def create_photo_preset(
        self,
        size: str = "8x10",
        finish: str = "glossy",
    ) -> PrintPresetConfig:
        """Create photo print preset."""
        paper_map = {
            '4x6': PaperSize.PHOTO_4X6,
            '5x7': PaperSize.PHOTO_5X7,
            '8x10': PaperSize.PHOTO_8X10,
            '11x14': PaperSize.PHOTO_11X14,
        }

        profile_map = {
            'glossy': PrintProfile.PHOTO_GLOSSY,
            'matte': PrintProfile.PHOTO_MATTE,
        }

        return PrintPresetConfig(
            output=OutputConfig(
                profile=profile_map.get(finish, PrintProfile.PHOTO_GLOSSY),
                paper_size=paper_map.get(size, PaperSize.PHOTO_8X10),
                dpi=300,
                format='jpeg',
            ),
        )

    def create_fine_art_preset(
        self,
        color_space: str = "adobe_rgb",
    ) -> PrintPresetConfig:
        """Create fine art print preset."""
        cs_map = {
            'srgb': ColorSpace.SRGB,
            'adobe_rgb': ColorSpace.ADOBE_RGB,
            'prophoto_rgb': ColorSpace.PROPHOTO_RGB,
        }

        return PrintPresetConfig(
            enhancement=EnhancementConfig(
                noise_reduction=0.3,
                sharpening=0.4,
            ),
            output=OutputConfig(
                profile=PrintProfile.FINE_ART,
                color_space=cs_map.get(color_space, ColorSpace.ADOBE_RGB),
                dpi=360,
                format='tiff',
                bit_depth=16,
            ),
        )

    @staticmethod
    def list_presets() -> List[str]:
        """List available preset names."""
        return ['PHOTO_PRINT', 'FINE_ART', 'CANVAS', 'MAGAZINE']

    @classmethod
    def get_preset(cls, name: str) -> PrintPresetConfig:
        """Get a preset by name."""
        presets = {
            'PHOTO_PRINT': cls.PHOTO_PRINT,
            'FINE_ART': cls.FINE_ART,
            'CANVAS': cls.CANVAS,
            'MAGAZINE': cls.MAGAZINE,
        }
        return presets.get(name.upper(), PrintPresetConfig())

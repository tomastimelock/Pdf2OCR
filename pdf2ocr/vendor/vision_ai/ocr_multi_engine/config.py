# Filepath: code_migration/extraction/ocr_multi_engine/config.py
# Description: OCR Provider Configuration with Pydantic-style dataclass
# Layer: Extractor
# References: reference_codebase/OCR_extractor/provider/config.py

"""
OCR Provider Configuration

Provides configuration management for OCR providers.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import os


@dataclass
class OCRConfig:
    """
    Configuration for OCR providers.

    Attributes:
        mistral_api_key: Mistral API key
        openai_api_key: OpenAI API key
        openai_model: OpenAI model to use (default: gpt-4o)
        openai_prompt: Custom prompt for OpenAI Vision
        tesseract_lang: Tesseract language code (default: eng)
        default_engine: Default OCR engine to use
        engine_order: Priority order for multi-engine OCR
        min_quality: Minimum quality threshold for multi-engine OCR
        fallback_enabled: Whether to enable automatic fallback
    """

    # API Keys
    mistral_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # OpenAI settings
    openai_model: str = "gpt-4o"
    openai_prompt: Optional[str] = None

    # Tesseract settings
    tesseract_lang: str = "eng"

    # Engine settings
    default_engine: str = "openai"
    engine_order: List[str] = field(default_factory=lambda: ["mistral", "openai", "tesseract"])
    min_quality: float = 0.7
    fallback_enabled: bool = True

    @classmethod
    def from_env(cls) -> "OCRConfig":
        """
        Create configuration from environment variables.

        Environment variables:
            MISTRAL_API_KEY: Mistral API key
            OPENAI_API_KEY: OpenAI API key
            OCR_OPENAI_MODEL: OpenAI model (default: gpt-4o)
            OCR_TESSERACT_LANG: Tesseract language (default: eng)
            OCR_DEFAULT_ENGINE: Default engine (default: openai)
            OCR_ENGINE_ORDER: Comma-separated engine order
            OCR_MIN_QUALITY: Minimum quality threshold (default: 0.7)
            OCR_FALLBACK_ENABLED: Enable fallback (default: true)

        Returns:
            OCRConfig instance
        """
        engine_order_str = os.getenv("OCR_ENGINE_ORDER", "mistral,openai,tesseract")
        engine_order = [e.strip() for e in engine_order_str.split(",")]

        fallback_str = os.getenv("OCR_FALLBACK_ENABLED", "true").lower()
        fallback_enabled = fallback_str in ("true", "1", "yes")

        min_quality_str = os.getenv("OCR_MIN_QUALITY", "0.7")
        try:
            min_quality = float(min_quality_str)
        except ValueError:
            min_quality = 0.7

        return cls(
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OCR_OPENAI_MODEL", "gpt-4o"),
            tesseract_lang=os.getenv("OCR_TESSERACT_LANG", "eng"),
            default_engine=os.getenv("OCR_DEFAULT_ENGINE", "openai"),
            engine_order=engine_order,
            min_quality=min_quality,
            fallback_enabled=fallback_enabled,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for factory usage.

        Returns:
            Configuration dictionary
        """
        return {
            "mistral_api_key": self.mistral_api_key,
            "openai_api_key": self.openai_api_key,
            "openai_model": self.openai_model,
            "openai_prompt": self.openai_prompt,
            "tesseract_lang": self.tesseract_lang,
            "min_quality": self.min_quality,
        }

    def validate(self) -> List[str]:
        """
        Validate the configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check if at least one engine is configured
        has_engine = False

        if self.mistral_api_key:
            has_engine = True

        if self.openai_api_key:
            has_engine = True

        # Tesseract is always available if dependencies are installed
        has_engine = True

        if not has_engine and self.fallback_enabled:
            errors.append("No OCR engine is configured. Set at least one API key or install Tesseract.")

        if self.min_quality < 0 or self.min_quality > 1:
            errors.append(f"min_quality must be between 0 and 1, got {self.min_quality}")

        valid_engines = {"mistral", "openai", "tesseract"}
        for engine in self.engine_order:
            if engine not in valid_engines:
                errors.append(f"Unknown engine in engine_order: {engine}")

        if self.default_engine not in valid_engines:
            errors.append(f"Unknown default_engine: {self.default_engine}")

        return errors


# Global configuration singleton
_config: Optional[OCRConfig] = None


def get_config() -> OCRConfig:
    """
    Get the global OCR configuration.

    Returns:
        OCRConfig instance
    """
    global _config
    if _config is None:
        _config = OCRConfig.from_env()
    return _config


def set_config(config: OCRConfig) -> None:
    """
    Set the global OCR configuration.

    Args:
        config: OCRConfig instance
    """
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration to default (from environment)."""
    global _config
    _config = None

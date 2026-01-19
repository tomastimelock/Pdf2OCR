# Filepath: code_migration/extraction/ocr_multi_engine/factory.py
# Description: OCR Provider Factory with multi-engine fallback support
# Layer: Extractor
# References: reference_codebase/OCR_extractor/provider/factory.py

"""
OCR Provider Factory

Creates OCR provider instances dynamically based on configuration.
"""

from typing import Dict, Any, Optional, List
import logging
import os

from .base import BaseOCRProvider, OCRProviderUnavailableError, OCRError


class MultiEngineOCR(BaseOCRProvider):
    """
    Multi-engine OCR with automatic fallback.

    Tries engines in order until one succeeds or all fail.
    """

    def __init__(
        self,
        engines: List[BaseOCRProvider],
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize multi-engine OCR.

        Args:
            engines: List of OCR providers in priority order
            logger: Logger instance
            config: Optional configuration dictionary
        """
        super().__init__(logger, config)
        self.engines = engines
        self.min_quality = config.get("min_quality", 0.7) if config else 0.7

    def get_provider_name(self) -> str:
        """Get provider name."""
        engine_names = [engine.get_provider_name() for engine in self.engines]
        return f"MultiEngine({', '.join(engine_names)})"

    def extract_text(self, image_path) -> str:
        """
        Extract text using multiple engines with fallback.

        Args:
            image_path: Path to image file

        Returns:
            Extracted text content

        Raises:
            OCRError: If all engines fail
        """
        last_error = None
        best_result = ""
        best_quality = 0.0

        for i, engine in enumerate(self.engines, 1):
            try:
                self.logger.info(
                    f"Attempting OCR with {engine.get_provider_name()} "
                    f"({i}/{len(self.engines)})..."
                )

                text = engine.extract_text(image_path)
                quality = self.get_quality_score(text)

                self.logger.info(
                    f"{engine.get_provider_name()} extracted {len(text)} characters "
                    f"(quality: {quality:.2f})"
                )

                # Keep track of best result
                if quality > best_quality:
                    best_result = text
                    best_quality = quality

                # If quality is good enough, use this result
                if quality >= self.min_quality:
                    self.logger.info(
                        f"Using result from {engine.get_provider_name()} "
                        f"(quality: {quality:.2f} >= {self.min_quality})"
                    )
                    self.ocr_stats["successful"] += 1
                    self.ocr_stats["total_characters_extracted"] += len(text)
                    return text

            except Exception as e:
                self.logger.warning(
                    f"{engine.get_provider_name()} failed: {str(e)}"
                )
                last_error = e
                continue

        # If we got here, no engine met the quality threshold
        if best_result:
            self.logger.warning(
                f"Best quality was {best_quality:.2f} (below threshold {self.min_quality}). "
                f"Using best result anyway."
            )
            self.ocr_stats["successful"] += 1
            self.ocr_stats["total_characters_extracted"] += len(best_result)
            return best_result

        # All engines failed
        self.ocr_stats["failed"] += 1
        error_msg = f"All OCR engines failed. Last error: {str(last_error)}"
        raise OCRError(error_msg)

    def is_available(self) -> bool:
        """Check if at least one engine is available."""
        return len(self.engines) > 0


class OCRProviderFactory:
    """
    Factory for creating OCR providers.

    Supports creating individual providers or multi-engine OCR with fallback.
    """

    @staticmethod
    def create(
        provider_name: str,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict] = None
    ) -> BaseOCRProvider:
        """
        Create an OCR provider by name.

        Args:
            provider_name: Name of the provider ("mistral", "openai", "tesseract")
            logger: Logger instance (creates default if not provided)
            config: Optional configuration dictionary with provider-specific settings

        Returns:
            BaseOCRProvider instance

        Raises:
            ValueError: If provider name is unknown
            OCRProviderUnavailableError: If provider is not available
        """
        from .engines.mistral import MistralOCR
        from .engines.openai_vision import OpenAIVision
        from .engines.tesseract import TesseractOCR

        if logger is None:
            logger = logging.getLogger("OCRProvider")

        config = config or {}
        provider_name_lower = provider_name.lower()

        # Create provider based on type
        if provider_name_lower == "mistral":
            api_key = config.get("mistral_api_key") or os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise OCRProviderUnavailableError(
                    "Mistral API key not found in config or environment (MISTRAL_API_KEY)"
                )
            provider = MistralOCR(api_key=api_key, logger=logger)

        elif provider_name_lower == "openai":
            api_key = config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise OCRProviderUnavailableError(
                    "OpenAI API key not found in config or environment (OPENAI_API_KEY)"
                )
            model = config.get("openai_model", "gpt-4o")
            prompt = config.get("openai_prompt")
            provider = OpenAIVision(api_key=api_key, model=model, prompt=prompt, logger=logger)

        elif provider_name_lower == "tesseract":
            lang = config.get("tesseract_lang", "eng")
            provider = TesseractOCR(lang=lang, logger=logger)

        else:
            available = "mistral, openai, tesseract"
            raise ValueError(
                f"Unknown OCR provider: {provider_name}. "
                f"Available providers: {available}"
            )

        if not provider.is_available():
            raise OCRProviderUnavailableError(
                f"OCR provider '{provider_name}' is not available. "
                f"Check configuration and dependencies."
            )

        return provider

    @staticmethod
    def create_multi_engine(
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict] = None,
        engine_order: Optional[List[str]] = None
    ) -> MultiEngineOCR:
        """
        Create multi-engine OCR with automatic fallback.

        Args:
            logger: Logger instance (creates default if not provided)
            config: Optional configuration dictionary
            engine_order: Optional list of engine names in priority order

        Returns:
            MultiEngineOCR instance
        """
        if logger is None:
            logger = logging.getLogger("OCRProvider")

        if engine_order is None:
            engine_order = ["mistral", "openai", "tesseract"]

        engines = []
        for engine_name in engine_order:
            try:
                engine = OCRProviderFactory.create(engine_name, logger, config)
                engines.append(engine)
                logger.info(f"Loaded OCR engine: {engine_name}")
            except (ValueError, OCRProviderUnavailableError) as e:
                logger.warning(f"Could not load OCR engine '{engine_name}': {e}")

        if not engines:
            raise OCRProviderUnavailableError(
                "No OCR engines available. At least one engine must be configured."
            )

        return MultiEngineOCR(engines, logger, config)

    @staticmethod
    def list_available_providers() -> List[str]:
        """
        List all available provider types.

        Returns:
            List of provider names
        """
        return ["mistral", "openai", "tesseract"]

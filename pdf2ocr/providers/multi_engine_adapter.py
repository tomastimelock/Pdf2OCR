"""Multi-Engine OCR Adapter - Unified interface for multiple OCR engines."""

import os
import logging
from pathlib import Path
from typing import Optional, List

from pdf2ocr.providers.base import BaseOCRProvider, OCRResult


class MultiEngineOCRProvider(BaseOCRProvider):
    """
    Multi-engine OCR provider with automatic fallback.

    Wraps multiple OCR providers and tries them in order until one succeeds
    or meets the quality threshold. Provides a unified interface for PDF2OCR.

    Supported engines:
    - mistral: Mistral AI OCR (cloud, high quality)
    - openai: OpenAI GPT-4 Vision (cloud, high quality)
    - tesseract: Tesseract OCR (local, free)
    """

    name: str = "multi_engine"
    quality_score: float = 0.85

    # Default engine priority order
    DEFAULT_ENGINE_ORDER = ["mistral", "openai", "tesseract"]

    def __init__(
        self,
        engine_order: Optional[List[str]] = None,
        quality_threshold: float = 0.7,
        mistral_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        tesseract_lang: str = "eng",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize multi-engine OCR provider.

        Args:
            engine_order: List of engine names in priority order
            quality_threshold: Minimum quality score to accept (0.0-1.0)
            mistral_api_key: Mistral API key (uses env var if not provided)
            openai_api_key: OpenAI API key (uses env var if not provided)
            tesseract_lang: Tesseract language code
            logger: Logger instance
        """
        self.engine_order = engine_order or self.DEFAULT_ENGINE_ORDER
        self.quality_threshold = quality_threshold
        self.mistral_api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.tesseract_lang = tesseract_lang
        self.logger = logger or logging.getLogger(__name__)

        self._engines: List[BaseOCRProvider] = []
        self._engines_initialized = False

    def _initialize_engines(self):
        """Lazy initialize OCR engines based on availability."""
        if self._engines_initialized:
            return

        for engine_name in self.engine_order:
            try:
                engine = self._create_engine(engine_name)
                if engine and engine.is_available():
                    self._engines.append(engine)
                    self.logger.info(f"Loaded OCR engine: {engine_name}")
            except Exception as e:
                self.logger.warning(f"Could not load OCR engine '{engine_name}': {e}")

        self._engines_initialized = True

        if not self._engines:
            self.logger.warning("No OCR engines available!")

    def _create_engine(self, engine_name: str) -> Optional[BaseOCRProvider]:
        """Create a single OCR engine by name."""
        engine_name = engine_name.lower()

        if engine_name == "mistral":
            if not self.mistral_api_key:
                return None
            from pdf2ocr.providers.mistral_provider import MistralOCRProvider
            return MistralOCRProvider(api_key=self.mistral_api_key)

        elif engine_name == "openai":
            if not self.openai_api_key:
                return None
            from pdf2ocr.providers.openai_ocr_provider import OpenAIOCRProvider
            return OpenAIOCRProvider(api_key=self.openai_api_key)

        elif engine_name == "tesseract":
            from pdf2ocr.providers.tesseract_provider import TesseractOCRProvider
            provider = TesseractOCRProvider(language=self.tesseract_lang)
            if provider.is_available():
                return provider
            return None

        else:
            self.logger.warning(f"Unknown OCR engine: {engine_name}")
            return None

    @property
    def engines(self) -> List[BaseOCRProvider]:
        """Get list of available engines."""
        self._initialize_engines()
        return self._engines

    def is_available(self) -> bool:
        """Check if at least one engine is available."""
        return len(self.engines) > 0

    def get_available_engines(self) -> List[str]:
        """Get list of available engine names."""
        return [e.name for e in self.engines]

    def extract_text(self, image_path: str | Path, page_number: int = 1) -> OCRResult:
        """
        Extract text from an image using multiple engines with fallback.

        Tries engines in priority order until one meets the quality threshold
        or all engines have been tried (returns best result).

        Args:
            image_path: Path to the image file
            page_number: Page number for tracking

        Returns:
            OCRResult with extracted text from best-performing engine
        """
        image_path = Path(image_path)

        if not self.engines:
            return OCRResult(
                text="",
                page_number=page_number,
                provider=self.name,
                quality_score=0.0,
                confidence=0.0,
                metadata={"error": "No OCR engines available"}
            )

        best_result: Optional[OCRResult] = None

        for i, engine in enumerate(self.engines, 1):
            try:
                self.logger.debug(
                    f"Attempting OCR with {engine.name} ({i}/{len(self.engines)})..."
                )

                result = engine.extract_text(image_path, page_number)

                self.logger.debug(
                    f"{engine.name} extracted {len(result.text)} characters "
                    f"(quality: {result.quality_score:.2f})"
                )

                # Track best result
                if best_result is None or result.quality_score > best_result.quality_score:
                    best_result = result

                # If quality meets threshold, use this result
                if result.quality_score >= self.quality_threshold:
                    self.logger.info(
                        f"Using result from {engine.name} "
                        f"(quality: {result.quality_score:.2f})"
                    )
                    return result

            except Exception as e:
                self.logger.warning(f"{engine.name} failed: {e}")
                continue

        # Return best result even if below threshold
        if best_result:
            self.logger.info(
                f"Best quality was {best_result.quality_score:.2f} "
                f"(below threshold {self.quality_threshold})"
            )
            return best_result

        # All engines failed completely
        return OCRResult(
            text="",
            page_number=page_number,
            provider=self.name,
            quality_score=0.0,
            confidence=0.0,
            metadata={"error": "All OCR engines failed"}
        )

    def extract_text_from_pdf(self, pdf_path: str | Path, page_number: int = 1) -> OCRResult:
        """
        Extract text from a PDF page using multiple engines with fallback.

        Args:
            pdf_path: Path to the PDF file
            page_number: Page number to extract (1-indexed)

        Returns:
            OCRResult with extracted text from best-performing engine
        """
        pdf_path = Path(pdf_path)

        if not self.engines:
            return OCRResult(
                text="",
                page_number=page_number,
                provider=self.name,
                quality_score=0.0,
                confidence=0.0,
                metadata={"error": "No OCR engines available"}
            )

        best_result: Optional[OCRResult] = None

        for i, engine in enumerate(self.engines, 1):
            try:
                self.logger.debug(
                    f"Attempting PDF OCR with {engine.name} ({i}/{len(self.engines)})..."
                )

                result = engine.extract_text_from_pdf(pdf_path, page_number)

                # Track best result
                if best_result is None or result.quality_score > best_result.quality_score:
                    best_result = result

                # If quality meets threshold, use this result
                if result.quality_score >= self.quality_threshold:
                    return result

            except Exception as e:
                self.logger.warning(f"{engine.name} failed on PDF: {e}")
                continue

        # Return best result
        if best_result:
            return best_result

        return OCRResult(
            text="",
            page_number=page_number,
            provider=self.name,
            quality_score=0.0,
            confidence=0.0,
            metadata={"error": "All OCR engines failed on PDF"}
        )


def create_multi_engine_provider(
    engine_order: Optional[List[str]] = None,
    quality_threshold: float = 0.7,
    **kwargs
) -> MultiEngineOCRProvider:
    """
    Factory function to create a multi-engine OCR provider.

    Args:
        engine_order: Priority order of engines (default: mistral, openai, tesseract)
        quality_threshold: Minimum quality to accept
        **kwargs: Additional arguments passed to MultiEngineOCRProvider

    Returns:
        Configured MultiEngineOCRProvider instance
    """
    return MultiEngineOCRProvider(
        engine_order=engine_order,
        quality_threshold=quality_threshold,
        **kwargs
    )

"""OCR and chart providers for PDF2OCR."""

from pdf2ocr.providers.mistral_provider import MistralOCRProvider
from pdf2ocr.providers.base import BaseOCRProvider, OCRResult
from pdf2ocr.providers.openai_provider import OpenAIChartProvider
from pdf2ocr.providers.anthropic_provider import AnthropicSVGProvider
from pdf2ocr.providers.tesseract_provider import TesseractOCRProvider
from pdf2ocr.providers.openai_ocr_provider import OpenAIOCRProvider
from pdf2ocr.providers.multi_engine_adapter import (
    MultiEngineOCRProvider,
    create_multi_engine_provider
)
from pdf2ocr.providers.llm_adapter import LLMAdapter, LLMResponse

__all__ = [
    # Base
    "BaseOCRProvider",
    "OCRResult",
    # OCR Providers
    "MistralOCRProvider",
    "TesseractOCRProvider",
    "OpenAIOCRProvider",
    "MultiEngineOCRProvider",
    "create_multi_engine_provider",
    # Chart/SVG Providers
    "OpenAIChartProvider",
    "AnthropicSVGProvider",
    # LLM Adapter
    "LLMAdapter",
    "LLMResponse",
]

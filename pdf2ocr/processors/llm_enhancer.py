"""LLM Enhancer - Use LLMs to improve OCR results and extract structured data."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json
import re

from pdf2ocr.providers.llm_adapter import LLMAdapter, ProviderType


@dataclass
class EnhancedText:
    """Result of LLM text enhancement."""
    original_text: str
    enhanced_text: str
    corrections_made: int
    provider: str
    model: str
    confidence: float


@dataclass
class ExtractedStructure:
    """Structured data extracted from text."""
    title: Optional[str] = None
    sections: List[Dict[str, Any]] = field(default_factory=list)
    key_points: List[str] = field(default_factory=list)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMEnhancer:
    """
    Use LLMs to enhance OCR output and extract structured information.

    Capabilities:
    - Fix common OCR errors
    - Improve formatting and structure
    - Extract key information
    - Generate summaries
    - Parse specific document types
    """

    # Prompts for different enhancement tasks
    OCR_CORRECTION_PROMPT = """You are an OCR correction specialist. The following text was extracted from a document using OCR and may contain errors.

Please correct any obvious OCR errors such as:
- Character substitutions (0/O, 1/l/I, rn/m, etc.)
- Missing spaces or extra spaces
- Broken words across lines
- Garbled characters

Maintain the original structure and formatting. Only fix clear errors, don't rephrase or rewrite.

OCR TEXT:
{text}

CORRECTED TEXT:"""

    STRUCTURE_EXTRACTION_PROMPT = """Analyze this document text and extract structured information.

Return a JSON object with:
- "title": document title if present
- "sections": list of {"heading": "...", "content": "..."} objects
- "key_points": list of main points or findings
- "entities": {"people": [], "organizations": [], "dates": [], "locations": []}
- "summary": 2-3 sentence summary

DOCUMENT TEXT:
{text}

JSON OUTPUT:"""

    DOCUMENT_TYPE_PROMPT = """Identify the type of this document and extract relevant fields.

Common types: invoice, receipt, contract, report, letter, form, resume, article

Return JSON with:
- "document_type": the identified type
- "confidence": 0.0-1.0 confidence score
- "fields": type-specific extracted fields

DOCUMENT TEXT:
{text}

JSON OUTPUT:"""

    def __init__(
        self,
        adapter: Optional[LLMAdapter] = None,
        default_provider: ProviderType = "anthropic",
        default_model: Optional[str] = None
    ):
        """
        Initialize the LLM enhancer.

        Args:
            adapter: LLM adapter instance (creates one if not provided)
            default_provider: Default LLM provider
            default_model: Default model to use
        """
        self.adapter = adapter or LLMAdapter(default_provider=default_provider)
        self.default_provider = default_provider
        self.default_model = default_model

    def correct_ocr_errors(
        self,
        text: str,
        provider: Optional[ProviderType] = None,
        model: Optional[str] = None
    ) -> EnhancedText:
        """
        Use LLM to correct common OCR errors in text.

        Args:
            text: OCR-extracted text with potential errors
            provider: LLM provider to use
            model: Model to use

        Returns:
            EnhancedText with corrected text
        """
        provider = provider or self.default_provider
        model = model or self.default_model

        prompt = self.OCR_CORRECTION_PROMPT.format(text=text)

        response = self.adapter.generate_text(
            prompt=prompt,
            provider=provider,
            model=model,
            temperature=0.1  # Low temperature for consistency
        )

        enhanced_text = response.text.strip()

        # Count approximate corrections
        corrections = self._count_differences(text, enhanced_text)

        return EnhancedText(
            original_text=text,
            enhanced_text=enhanced_text,
            corrections_made=corrections,
            provider=response.provider,
            model=response.model,
            confidence=0.9 if corrections < 50 else 0.7
        )

    def extract_structure(
        self,
        text: str,
        provider: Optional[ProviderType] = None,
        model: Optional[str] = None
    ) -> ExtractedStructure:
        """
        Extract structured information from document text.

        Args:
            text: Document text
            provider: LLM provider to use
            model: Model to use

        Returns:
            ExtractedStructure with parsed data
        """
        provider = provider or self.default_provider
        model = model or self.default_model

        prompt = self.STRUCTURE_EXTRACTION_PROMPT.format(text=text[:8000])  # Limit text length

        response = self.adapter.generate_text(
            prompt=prompt,
            provider=provider,
            model=model,
            temperature=0.2
        )

        # Parse JSON from response
        try:
            data = self._extract_json(response.text)
            return ExtractedStructure(
                title=data.get("title"),
                sections=data.get("sections", []),
                key_points=data.get("key_points", []),
                entities=data.get("entities", {}),
                summary=data.get("summary"),
                metadata={"provider": response.provider, "model": response.model}
            )
        except Exception:
            return ExtractedStructure(
                metadata={"error": "Failed to parse structure", "raw": response.text}
            )

    def identify_document_type(
        self,
        text: str,
        provider: Optional[ProviderType] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Identify the document type and extract type-specific fields.

        Args:
            text: Document text
            provider: LLM provider to use
            model: Model to use

        Returns:
            Dict with document_type, confidence, and extracted fields
        """
        provider = provider or self.default_provider
        model = model or self.default_model

        prompt = self.DOCUMENT_TYPE_PROMPT.format(text=text[:4000])

        response = self.adapter.generate_text(
            prompt=prompt,
            provider=provider,
            model=model,
            temperature=0.2
        )

        try:
            return self._extract_json(response.text)
        except Exception:
            return {
                "document_type": "unknown",
                "confidence": 0.0,
                "fields": {},
                "error": "Failed to parse response"
            }

    def summarize(
        self,
        text: str,
        max_sentences: int = 3,
        provider: Optional[ProviderType] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Generate a summary of the document.

        Args:
            text: Document text
            max_sentences: Maximum sentences in summary
            provider: LLM provider to use
            model: Model to use

        Returns:
            Summary text
        """
        provider = provider or self.default_provider
        model = model or self.default_model

        prompt = f"""Summarize this document in {max_sentences} sentences or less.
Focus on the main points and key information.

DOCUMENT:
{text[:8000]}

SUMMARY:"""

        response = self.adapter.generate_text(
            prompt=prompt,
            provider=provider,
            model=model,
            temperature=0.3
        )

        return response.text.strip()

    def enhance_formatting(
        self,
        text: str,
        provider: Optional[ProviderType] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Improve the formatting of OCR text.

        Args:
            text: Raw OCR text
            provider: LLM provider to use
            model: Model to use

        Returns:
            Better-formatted text
        """
        provider = provider or self.default_provider
        model = model or self.default_model

        prompt = f"""Improve the formatting of this OCR-extracted text.
- Fix paragraph breaks
- Restore bullet points and lists
- Correct obvious spacing issues
- Maintain the original content without changing meaning

TEXT:
{text}

FORMATTED TEXT:"""

        response = self.adapter.generate_text(
            prompt=prompt,
            provider=provider,
            model=model,
            temperature=0.1
        )

        return response.text.strip()

    def _count_differences(self, original: str, corrected: str) -> int:
        """Count approximate number of character differences."""
        # Simple character-level diff count
        diff_count = 0
        min_len = min(len(original), len(corrected))

        for i in range(min_len):
            if original[i] != corrected[i]:
                diff_count += 1

        diff_count += abs(len(original) - len(corrected))
        return diff_count

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response text."""
        # Try to find JSON in the response
        text = text.strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try markdown code block
        code_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass

        raise ValueError("Could not extract JSON from response")

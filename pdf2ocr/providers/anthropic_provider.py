"""Anthropic Provider for SVG generation using Claude Vision."""

import os
import base64
from pathlib import Path
from typing import Optional
import logging

from pdf2ocr.utils.svg_validator import (
    validate_and_repair_svg,
    validate_and_repair_svg_enhanced,
    extract_svg_dimensions,
    extract_svg_from_response,
    sanitize_svg,
    ensure_svg_complete,
    get_svg_error_feedback,
    SVGValidationResult,
)

logger = logging.getLogger(__name__)


class AnthropicSVGProvider:
    """
    Anthropic provider for SVG generation using Claude Vision.

    Uses Claude to regenerate charts as production-ready SVG code.
    Includes retry logic for SVG validation failures.
    """

    DEFAULT_MAX_RETRIES = 2

    def __init__(self, api_key: Optional[str] = None, max_retries: int = None):
        """
        Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key. If not provided, reads from ANTHROPIC_API_KEY env var.
            max_retries: Max retries for SVG validation failures (default: 2)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None
        self._model = "claude-sonnet-4-5-20250929"
        self.max_retries = max_retries if max_retries is not None else self.DEFAULT_MAX_RETRIES

    @property
    def client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "Anthropic API key not provided. "
                    "Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
                )
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def is_available(self) -> bool:
        """Check if Anthropic API is available."""
        return bool(self.api_key)

    def _get_mime_type(self, file_path: Path) -> str:
        """Determine MIME type from file extension."""
        ext = file_path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".pdf": "application/pdf",
        }
        return mime_types.get(ext, "image/jpeg")

    def _encode_file(self, file_path: Path) -> str:
        """Read and base64 encode a file."""
        with open(file_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def generate_svg(
        self,
        image_path: str | Path,
        description: str
    ) -> dict:
        """
        Generate SVG from an image of a chart/diagram.

        Args:
            image_path: Path to the image file
            description: Description of the chart to recreate

        Returns:
            Dict with keys:
            - svg_code: The generated SVG code
            - width: SVG width
            - height: SVG height
            - success: Whether generation succeeded
            - error: Error message if failed
        """
        image_path = Path(image_path)

        if not image_path.exists():
            return {
                "svg_code": "",
                "width": 0,
                "height": 0,
                "success": False,
                "error": f"File not found: {image_path}"
            }

        try:
            base64_data = self._encode_file(image_path)
            media_type = self._get_mime_type(image_path)

            return self._generate_svg_from_base64(base64_data, media_type, description)

        except Exception as e:
            logger.error(f"SVG generation failed: {e}")
            return {
                "svg_code": "",
                "width": 0,
                "height": 0,
                "success": False,
                "error": str(e)
            }

    def _generate_svg_from_base64(
        self,
        base64_data: str,
        media_type: str,
        description: str
    ) -> dict:
        """
        Generate SVG from base64-encoded image data with retry logic.

        If the generated SVG has validation errors, will retry with
        error feedback to help Claude fix the issues.

        Args:
            base64_data: Base64-encoded image
            media_type: MIME type
            description: Chart description

        Returns:
            SVG result dict
        """
        system_prompt = """You are an expert SVG artist. Create clean, optimized, VALID SVG code.

Guidelines:
- Use appropriate viewBox dimensions (typically 400x300 for charts)
- Make the design professional and visually appealing
- Use clear, readable fonts (Arial, Helvetica, sans-serif)
- Include proper labels and legends
- Ensure the SVG is self-contained with no external dependencies
- Use semantic colors that convey meaning
- Return ONLY the SVG code, no explanation or markdown formatting
- Start with <svg and end with </svg>

CRITICAL XML RULES:
- Every opening tag MUST have a matching closing tag
- Use self-closing syntax for empty elements: <path d="..." />, <rect ... />, <circle ... />
- Escape special characters in text: use &amp; for &, &lt; for <, &gt; for >
- Ensure all attribute values are properly quoted"""

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_data
                        }
                    },
                    {
                        "type": "text",
                        "text": f"""Recreate this chart as SVG code.

Description: {description}

Requirements:
1. Match the visual style and data as closely as possible
2. Use the same chart type and layout
3. Include all labels, titles, and legends
4. Return only valid SVG code starting with <svg and ending with </svg>
5. Ensure the XML is well-formed with properly matched tags"""
                    }
                ]
            }
        ]

        last_error = None
        last_svg = ""

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self._model,
                    max_tokens=8192,
                    system=system_prompt,
                    messages=messages
                )

                svg_content = response.content[0].text

                # Extract SVG from response if wrapped in other content
                extracted_svg = extract_svg_from_response(svg_content)
                if extracted_svg:
                    svg_content = extracted_svg

                # Ensure SVG is complete (has closing tag)
                svg_content = ensure_svg_complete(svg_content)

                # Sanitize SVG (remove scripts, event handlers)
                svg_content = sanitize_svg(svg_content)

                # Use enhanced validation with multiple repair strategies
                validation_result = validate_and_repair_svg_enhanced(svg_content)

                if validation_result.is_valid:
                    if validation_result.was_repaired:
                        logger.debug(f"SVG was repaired: {validation_result.repair_actions}")

                    # Extract dimensions from viewBox or width/height
                    width, height = extract_svg_dimensions(validation_result.svg_code)

                    return {
                        "svg_code": validation_result.svg_code,
                        "width": width,
                        "height": height,
                        "success": True,
                        "provider": "anthropic",
                        "model": self._model,
                        "attempts": attempt + 1,
                        "was_repaired": validation_result.was_repaired
                    }

                # Validation failed - prepare for retry
                last_error = validation_result.error_summary
                last_svg = validation_result.svg_code

                if attempt < self.max_retries:
                    # Add the failed SVG and error feedback for retry
                    error_feedback = get_svg_error_feedback(validation_result)
                    logger.info(f"SVG validation failed (attempt {attempt + 1}), retrying with feedback")

                    messages.append({
                        "role": "assistant",
                        "content": svg_content
                    })
                    messages.append({
                        "role": "user",
                        "content": f"""{error_feedback}

Please regenerate the SVG with these issues fixed. Return ONLY the corrected SVG code."""
                    })
                else:
                    logger.warning(f"SVG validation failed after {self.max_retries + 1} attempts: {last_error}")

            except Exception as e:
                logger.error(f"SVG generation failed (attempt {attempt + 1}): {e}")
                last_error = str(e)

                if attempt >= self.max_retries:
                    break

        # Return the last attempt even if validation failed
        # Browser may still render it partially
        if last_svg:
            width, height = extract_svg_dimensions(last_svg)
            return {
                "svg_code": last_svg,
                "width": width,
                "height": height,
                "success": False,
                "error": f"SVG validation failed: {last_error}",
                "provider": "anthropic",
                "model": self._model,
                "attempts": self.max_retries + 1
            }

        return {
            "svg_code": "",
            "width": 0,
            "height": 0,
            "success": False,
            "error": last_error or "Unknown error",
            "provider": "anthropic",
            "model": self._model
        }

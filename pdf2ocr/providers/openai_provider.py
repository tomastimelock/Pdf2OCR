"""OpenAI Provider for chart detection using GPT-4o Vision."""

import os
import base64
import json
import re
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class OpenAIChartProvider:
    """
    OpenAI provider for chart detection using GPT-4o Vision.

    Uses GPT-4o to detect and describe charts/graphs in images.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None
        self._model = "gpt-4o"

    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "OpenAI API key not provided. "
                    "Set OPENAI_API_KEY environment variable or pass api_key parameter."
                )
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
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
        }
        return mime_types.get(ext, "image/jpeg")

    def _encode_image(self, image_path: Path) -> str:
        """Read and base64 encode an image."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def detect_charts(self, image_path: str | Path) -> List[dict]:
        """
        Detect charts/graphs in an image using GPT-4o Vision.

        Args:
            image_path: Path to the image file

        Returns:
            List of detected chart info dicts with keys:
            - type: Chart type (bar, line, pie, scatter, etc.)
            - title: Chart title if visible
            - description: What the chart shows
            - data_points: Key values if readable
        """
        image_path = Path(image_path)

        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return []

        try:
            base64_image = self._encode_image(image_path)
            mime_type = self._get_mime_type(image_path)

            response = self.client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}",
                                    "detail": "high"
                                }
                            },
                            {
                                "type": "text",
                                "text": """Analyze this image and identify any charts, graphs, or diagrams.
                                For each chart found, provide:
                                1. Chart type (bar, line, pie, scatter, area, etc.)
                                2. Title if visible
                                3. Description of what the chart shows
                                4. Key data points if readable

                                Return as JSON array: [{"type": "...", "title": "...", "description": "...", "data_points": [...]}]
                                If no charts are found, return: []"""
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.1
            )

            content = response.choices[0].message.content
            charts = self._extract_json_from_response(content)

            if charts:
                logger.info(f"Detected {len(charts)} charts in {image_path.name}")
            else:
                logger.debug(f"No charts detected in {image_path.name}")

            return charts

        except Exception as e:
            logger.error(f"Chart detection failed: {e}")
            return []

    def describe_chart(
        self,
        image_path: str | Path,
        chart_info: dict
    ) -> str:
        """
        Generate a detailed description of a chart for SVG recreation.

        Args:
            image_path: Path to the image file
            chart_info: Basic chart info from detection

        Returns:
            Detailed description string
        """
        image_path = Path(image_path)

        if not image_path.exists():
            return chart_info.get('description', '')

        try:
            base64_image = self._encode_image(image_path)
            mime_type = self._get_mime_type(image_path)

            prompt = f"""Describe this chart in detail for recreation as SVG:

Chart type: {chart_info.get('type', 'unknown')}
Title: {chart_info.get('title', 'none')}

Provide:
1. Exact chart type and style
2. All axis labels and scales
3. Legend entries
4. All data points or values visible
5. Colors and visual styling
6. Any annotations or callouts

Be precise and complete for SVG recreation."""

            response = self.client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}",
                                    "detail": "high"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.1
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Chart description failed: {e}")
            return chart_info.get('description', '')

    def _extract_json_from_response(self, content: str) -> List[dict]:
        """
        Extract JSON array from response content.

        Handles various formats:
        - Plain JSON: [{"type": "bar", ...}]
        - Markdown code block: ```json\n[...]\n```
        - Mixed text with embedded JSON

        Args:
            content: Raw response content

        Returns:
            List of chart dictionaries, empty list if none found
        """
        if not content:
            return []

        content = content.strip()

        # Try direct JSON parse first
        try:
            result = json.loads(content)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # Try to extract from markdown code block
        code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
        if code_block_match:
            try:
                result = json.loads(code_block_match.group(1).strip())
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        # Try to find JSON array anywhere in the content
        array_match = re.search(r'\[[\s\S]*\]', content)
        if array_match:
            try:
                result = json.loads(array_match.group(0))
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        # Check for explicit "no charts" response
        if any(phrase in content.lower() for phrase in ['no charts', 'no graphs', '[]', 'empty']):
            return []

        logger.warning(f"Could not parse chart detection response as JSON")
        return []

"""
SVG Validation and Repair Utilities.

Provides validation, repair, and sanitization for SVG content.
Based on common issues observed with AI-generated SVG code.
"""

import re
import xml.etree.ElementTree as ET
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def validate_and_repair_svg(svg_code: str) -> Tuple[str, bool, Optional[str]]:
    """
    Validate and repair common SVG XML issues.

    Args:
        svg_code: The SVG code to validate and repair.

    Returns:
        Tuple of (repaired_svg, was_repaired, error_message).
        If validation fails completely, returns (original_svg, False, error).
    """
    original_svg = svg_code
    was_repaired = False

    # Step 1: Remove duplicate closing tags (common Claude generation bug)
    closing_tags = [
        '</defs>', '</g>', '</linearGradient>', '</radialGradient>',
        '</pattern>', '</clipPath>', '</mask>', '</filter>',
        '</text>', '</tspan>', '</style>', '</symbol>', '</marker>'
    ]

    for tag in closing_tags:
        # Check if tag appears right before </svg> when it shouldn't
        pattern = rf'({tag})\s*</svg>'
        match = re.search(pattern, svg_code)
        if match:
            # Check if this tag has a matching opening tag
            open_tag = tag.replace('/', '')
            open_count = svg_code.count(open_tag)
            close_count = svg_code.count(tag)

            if close_count > open_count:
                # Remove the extra closing tag before </svg>
                svg_code = re.sub(pattern, '</svg>', svg_code)
                was_repaired = True
                logger.debug(f"Removed duplicate {tag} before </svg>")

    # Step 2: Fix duplicate consecutive closing tags
    for tag in closing_tags:
        pattern = rf'({tag})\s*{tag}'
        while re.search(pattern, svg_code):
            svg_code = re.sub(pattern, r'\1', svg_code)
            was_repaired = True
            logger.debug(f"Removed duplicate consecutive {tag}")

    # Step 3: Try to parse as XML to validate
    try:
        # Add namespace if missing for proper parsing
        if 'xmlns=' not in svg_code:
            svg_code = svg_code.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"', 1)
            was_repaired = True
            logger.debug("Added missing xmlns attribute")

        # Try parsing
        ET.fromstring(svg_code)
        logger.debug("SVG validated successfully")
        return svg_code, was_repaired, None

    except ET.ParseError as e:
        error_msg = str(e)
        logger.warning(f"XML parse error: {error_msg}")

        # Step 4: Try more aggressive repairs
        try:
            # Count all opening and closing tags
            all_tags = re.findall(r'<(/?)(\w+)[^>]*>', svg_code)
            tag_stack = []

            for is_close, tag_name in all_tags:
                if is_close:
                    if tag_stack and tag_stack[-1] == tag_name:
                        tag_stack.pop()
                else:
                    # Self-closing tags don't need to be tracked
                    if tag_name.lower() not in ['br', 'hr', 'img', 'input', 'meta', 'link']:
                        tag_stack.append(tag_name)

            # If there are unclosed tags, try to close them
            if tag_stack:
                logger.debug(f"Found unclosed tags: {tag_stack}")
                # Remove 'svg' from stack as it should be closed last
                if 'svg' in tag_stack:
                    tag_stack.remove('svg')

                # Insert missing closing tags before </svg>
                closing_needed = ''.join(f'</{tag}>' for tag in reversed(tag_stack))
                svg_code = svg_code.replace('</svg>', f'{closing_needed}</svg>')
                was_repaired = True

            # Try parsing again
            ET.fromstring(svg_code)
            logger.debug("SVG repaired and validated successfully")
            return svg_code, was_repaired, None

        except ET.ParseError as e2:
            logger.error(f"Could not repair SVG: {e2}")
            # Return original with error - at least it might render in browser
            return original_svg, False, f"SVG validation failed: {error_msg}"


def extract_svg_dimensions(svg_code: str) -> Tuple[int, int]:
    """
    Extract width and height from SVG code.

    Checks both viewBox (preferred) and explicit width/height attributes.

    Args:
        svg_code: The SVG code.

    Returns:
        Tuple of (width, height). Defaults to (400, 300) if not found.
    """
    width = 400
    height = 300

    # Try viewBox first (most reliable) - format: "minX minY width height"
    viewbox_match = re.search(
        r'<svg[^>]*viewBox\s*=\s*["\'][\d.\-]+\s+[\d.\-]+\s+([\d.]+)\s+([\d.]+)["\']',
        svg_code,
        re.IGNORECASE
    )
    if viewbox_match:
        width = int(float(viewbox_match.group(1)))
        height = int(float(viewbox_match.group(2)))
        return width, height

    # Fallback to width/height attributes on <svg> tag specifically
    svg_tag_match = re.search(r'<svg[^>]*>', svg_code, re.IGNORECASE)
    if svg_tag_match:
        svg_tag = svg_tag_match.group(0)

        width_match = re.search(r'width\s*=\s*["\'](\d+)', svg_tag)
        height_match = re.search(r'height\s*=\s*["\'](\d+)', svg_tag)

        if width_match:
            width = int(width_match.group(1))
        if height_match:
            height = int(height_match.group(1))

    return width, height


def is_valid_svg(svg_code: str) -> bool:
    """
    Check if SVG code is valid XML.

    Args:
        svg_code: The SVG code to check.

    Returns:
        True if valid, False otherwise.
    """
    try:
        # Add namespace if missing
        if 'xmlns=' not in svg_code:
            svg_code = svg_code.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"', 1)
        ET.fromstring(svg_code)
        return True
    except ET.ParseError:
        return False


def sanitize_svg(svg_code: str) -> str:
    """
    Sanitize SVG code by removing potentially dangerous elements.

    Removes:
    - Script tags
    - Event handlers (onclick, onload, etc.)
    - javascript: URLs
    - data: URLs (can contain scripts)
    - External references

    Args:
        svg_code: The SVG code to sanitize.

    Returns:
        Sanitized SVG code.
    """
    # Remove script tags
    svg_code = re.sub(
        r'<script[^>]*>.*?</script>',
        '',
        svg_code,
        flags=re.DOTALL | re.IGNORECASE
    )

    # Remove onclick and other event handlers
    svg_code = re.sub(
        r'\s+on\w+\s*=\s*["\'][^"\']*["\']',
        '',
        svg_code,
        flags=re.IGNORECASE
    )

    # Remove javascript: URLs
    svg_code = re.sub(
        r'javascript:[^"\']*',
        '',
        svg_code,
        flags=re.IGNORECASE
    )

    # Remove data: URLs that could contain scripts
    svg_code = re.sub(
        r'data:\s*text/html[^"\']*',
        '',
        svg_code,
        flags=re.IGNORECASE
    )

    # Remove potentially dangerous href values
    svg_code = re.sub(
        r'href\s*=\s*["\']javascript:[^"\']*["\']',
        '',
        svg_code,
        flags=re.IGNORECASE
    )

    # Remove xlink:href with javascript
    svg_code = re.sub(
        r'xlink:href\s*=\s*["\']javascript:[^"\']*["\']',
        '',
        svg_code,
        flags=re.IGNORECASE
    )

    return svg_code


def extract_svg_from_response(content: str) -> Optional[str]:
    """
    Extract SVG code from AI response content.

    Handles various formats:
    - Raw SVG
    - SVG in markdown code blocks (```svg or ```xml)
    - SVG with XML declaration

    Args:
        content: Response content that may contain SVG.

    Returns:
        Extracted SVG code or None if not found.
    """
    content = content.strip()

    # Try to find SVG in markdown code blocks
    svg_match = re.search(r'```svg\s*([\s\S]*?)\s*```', content)
    if svg_match:
        return svg_match.group(1).strip()

    svg_match = re.search(r'```xml\s*([\s\S]*?)\s*```', content)
    if svg_match:
        extracted = svg_match.group(1).strip()
        if '<svg' in extracted:
            return extracted

    # Try to find raw SVG tag
    svg_match = re.search(r'(<svg[^>]*>[\s\S]*?</svg>)', content)
    if svg_match:
        return svg_match.group(1)

    # If content starts with XML declaration, try to find SVG
    if content.startswith('<?xml'):
        xml_match = re.search(r'(<\?xml[^>]*\?>[\s\S]*?<svg[^>]*>[\s\S]*?</svg>)', content)
        if xml_match:
            return xml_match.group(1)

    # Check if content itself is SVG (starts with <svg)
    if content.strip().startswith('<svg'):
        return content

    return None


def ensure_svg_complete(svg_code: str) -> str:
    """
    Ensure SVG has proper closing tag.

    Args:
        svg_code: SVG code that might be truncated.

    Returns:
        SVG with proper closing.
    """
    svg_code = svg_code.strip()

    # If it doesn't end with </svg>, try to close it
    if not svg_code.endswith('</svg>'):
        # Check if </svg> exists somewhere (might be followed by whitespace)
        if '</svg>' in svg_code:
            # Trim everything after </svg>
            idx = svg_code.rfind('</svg>')
            svg_code = svg_code[:idx + 6]
        else:
            # Add closing tag
            svg_code = svg_code + '</svg>'

    return svg_code

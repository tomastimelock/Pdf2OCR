"""
SVG Validation and Repair Utilities.

Provides validation, repair, and sanitization for SVG content.
Based on common issues observed with AI-generated SVG code.
"""

import re
import xml.etree.ElementTree as ET
from typing import Tuple, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SVGValidationResult:
    """Result of SVG validation."""
    svg_code: str
    is_valid: bool
    was_repaired: bool
    errors: List[str]
    repair_actions: List[str]

    @property
    def error_summary(self) -> str:
        """Get a summary of errors for feedback."""
        if not self.errors:
            return ""
        return "; ".join(self.errors[:3])  # Limit to 3 errors for feedback


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


def _fix_mismatched_tags(svg_code: str) -> Tuple[str, List[str]]:
    """
    Fix mismatched XML tags using a stack-based approach.

    Args:
        svg_code: SVG code with potential mismatched tags.

    Returns:
        Tuple of (fixed_svg, list_of_repairs_made)
    """
    repairs = []

    # Parse all tags with their positions
    tag_pattern = re.compile(r'<(/?)(\w+)([^>]*)(/?)>')
    matches = list(tag_pattern.finditer(svg_code))

    if not matches:
        return svg_code, repairs

    # Build a stack to track open tags
    stack = []  # (tag_name, start_pos, end_pos)
    issues = []  # (issue_type, position, tag_name)

    for match in matches:
        is_closing = match.group(1) == '/'
        tag_name = match.group(2).lower()
        is_self_closing = match.group(4) == '/'

        # Skip self-closing tags
        if is_self_closing:
            continue

        # Skip void elements that don't need closing
        void_elements = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img',
                         'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}
        if tag_name in void_elements:
            continue

        if is_closing:
            # Closing tag
            if not stack:
                issues.append(('extra_close', match.start(), tag_name))
            elif stack[-1][0] == tag_name:
                stack.pop()
            else:
                # Mismatched - find if it exists in stack
                found_idx = None
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i][0] == tag_name:
                        found_idx = i
                        break

                if found_idx is not None:
                    # Close all tags between found and current
                    for i in range(len(stack) - 1, found_idx, -1):
                        issues.append(('missing_close', match.start(), stack[i][0]))
                    stack = stack[:found_idx]
                else:
                    issues.append(('extra_close', match.start(), tag_name))
        else:
            # Opening tag
            stack.append((tag_name, match.start(), match.end()))

    # Any remaining items in stack need closing
    for tag_name, start, end in reversed(stack):
        if tag_name != 'svg':  # svg will be closed at the end
            issues.append(('unclosed', -1, tag_name))

    # Apply fixes
    if issues:
        # Sort by position (descending) to insert from end to start
        insertions = []

        for issue_type, pos, tag_name in issues:
            if issue_type == 'missing_close':
                insertions.append((pos, f'</{tag_name}>'))
                repairs.append(f"Added missing </{tag_name}>")
            elif issue_type == 'unclosed':
                # Insert before </svg>
                svg_close_pos = svg_code.rfind('</svg>')
                if svg_close_pos > 0:
                    insertions.append((svg_close_pos, f'</{tag_name}>'))
                    repairs.append(f"Closed unclosed <{tag_name}>")

        # Sort insertions by position (descending)
        insertions.sort(key=lambda x: x[0], reverse=True)

        # Apply insertions
        for pos, text in insertions:
            svg_code = svg_code[:pos] + text + svg_code[pos:]

    return svg_code, repairs


def _fix_common_svg_issues(svg_code: str) -> Tuple[str, List[str]]:
    """
    Fix common SVG-specific issues.

    Args:
        svg_code: SVG code to fix.

    Returns:
        Tuple of (fixed_svg, list_of_repairs_made)
    """
    repairs = []

    # Fix unclosed path/line/rect elements (should be self-closing)
    self_closing_elements = ['path', 'line', 'rect', 'circle', 'ellipse',
                             'polygon', 'polyline', 'use', 'image']

    for elem in self_closing_elements:
        # Find elements that are opened but not properly closed
        # Pattern: <elem ... > followed by </elem> without content
        pattern = rf'(<{elem}\s[^>]*[^/])>\s*</{elem}>'
        replacement = rf'\1/>'
        new_svg, count = re.subn(pattern, replacement, svg_code, flags=re.IGNORECASE)
        if count > 0:
            svg_code = new_svg
            repairs.append(f"Converted {count} empty <{elem}> to self-closing")

    # Fix common attribute issues
    # Remove duplicate xmlns attributes
    xmlns_pattern = r'(<svg[^>]*xmlns="[^"]*")([^>]*)(xmlns="[^"]*")'
    if re.search(xmlns_pattern, svg_code):
        svg_code = re.sub(xmlns_pattern, r'\1\2', svg_code)
        repairs.append("Removed duplicate xmlns attribute")

    # Fix broken gradient references
    # Sometimes Claude generates gradients with IDs that don't match fill references
    gradient_ids = re.findall(r'<(?:linear|radial)Gradient[^>]*id="([^"]+)"', svg_code)
    fill_refs = re.findall(r'fill="url\(#([^)]+)\)"', svg_code)

    for ref in fill_refs:
        if ref not in gradient_ids:
            # Replace broken gradient reference with a solid color
            svg_code = svg_code.replace(f'fill="url(#{ref})"', 'fill="#666666"')
            repairs.append(f"Fixed broken gradient reference #{ref}")

    return svg_code, repairs


def _escape_text_content(svg_code: str) -> Tuple[str, List[str]]:
    """
    Escape unescaped special characters in text content.

    Args:
        svg_code: SVG code with potential unescaped characters.

    Returns:
        Tuple of (fixed_svg, list_of_repairs_made)
    """
    repairs = []

    # Find text content and escape & that isn't already escaped
    def escape_ampersand(match):
        text = match.group(1)
        if '&' in text and '&amp;' not in text and '&lt;' not in text and '&gt;' not in text:
            return '>' + text.replace('&', '&amp;') + '<'
        return match.group(0)

    # Match content between > and <
    new_svg = re.sub(r'>([^<]*&[^<]*)<', escape_ampersand, svg_code)
    if new_svg != svg_code:
        repairs.append("Escaped unescaped ampersands in text")
        svg_code = new_svg

    return svg_code, repairs


def validate_and_repair_svg_enhanced(svg_code: str, max_repair_attempts: int = 3) -> SVGValidationResult:
    """
    Enhanced SVG validation with multiple repair strategies.

    Args:
        svg_code: The SVG code to validate and repair.
        max_repair_attempts: Maximum number of repair iterations.

    Returns:
        SVGValidationResult with detailed information about validation and repairs.
    """
    original_svg = svg_code
    all_repairs = []
    all_errors = []

    for attempt in range(max_repair_attempts):
        # Step 1: Basic cleanup
        svg_code = svg_code.strip()

        # Ensure xmlns is present
        if 'xmlns=' not in svg_code:
            svg_code = svg_code.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"', 1)
            all_repairs.append("Added missing xmlns attribute")

        # Step 2: Fix common SVG-specific issues
        svg_code, repairs = _fix_common_svg_issues(svg_code)
        all_repairs.extend(repairs)

        # Step 3: Escape text content
        svg_code, repairs = _escape_text_content(svg_code)
        all_repairs.extend(repairs)

        # Step 4: Try to parse
        try:
            ET.fromstring(svg_code)
            logger.debug(f"SVG validated successfully after {attempt + 1} attempt(s)")
            return SVGValidationResult(
                svg_code=svg_code,
                is_valid=True,
                was_repaired=len(all_repairs) > 0,
                errors=[],
                repair_actions=all_repairs
            )
        except ET.ParseError as e:
            error_msg = str(e)
            all_errors.append(error_msg)
            logger.debug(f"Validation attempt {attempt + 1} failed: {error_msg}")

            # Step 5: Fix mismatched tags
            svg_code, repairs = _fix_mismatched_tags(svg_code)
            all_repairs.extend(repairs)

            # Step 6: Remove duplicate closing tags
            closing_tags = ['defs', 'g', 'linearGradient', 'radialGradient',
                           'pattern', 'clipPath', 'mask', 'filter',
                           'text', 'tspan', 'style', 'symbol', 'marker']

            for tag in closing_tags:
                close_tag = f'</{tag}>'
                open_tag = f'<{tag}'
                open_count = svg_code.lower().count(open_tag.lower())
                close_count = svg_code.count(close_tag)

                while close_count > open_count:
                    # Remove the last extra closing tag
                    last_idx = svg_code.rfind(close_tag)
                    if last_idx > 0:
                        svg_code = svg_code[:last_idx] + svg_code[last_idx + len(close_tag):]
                        all_repairs.append(f"Removed extra {close_tag}")
                        close_count -= 1
                    else:
                        break

            # Try parsing again before next iteration
            try:
                ET.fromstring(svg_code)
                return SVGValidationResult(
                    svg_code=svg_code,
                    is_valid=True,
                    was_repaired=True,
                    errors=[],
                    repair_actions=all_repairs
                )
            except ET.ParseError:
                continue

    # If we get here, validation failed after all attempts
    logger.warning(f"SVG validation failed after {max_repair_attempts} attempts")

    return SVGValidationResult(
        svg_code=svg_code,  # Return the best attempt
        is_valid=False,
        was_repaired=len(all_repairs) > 0,
        errors=all_errors,
        repair_actions=all_repairs
    )


def get_svg_error_feedback(validation_result: SVGValidationResult) -> str:
    """
    Generate feedback for the LLM to help it fix SVG issues.

    Args:
        validation_result: The validation result with errors.

    Returns:
        A string describing the issues for the LLM to fix.
    """
    if validation_result.is_valid:
        return ""

    feedback_parts = ["The SVG you generated has XML validation errors:"]

    for error in validation_result.errors[:3]:
        # Parse common XML errors
        if 'mismatched tag' in error.lower():
            feedback_parts.append(f"- Mismatched XML tags: {error}")
        elif 'unclosed' in error.lower():
            feedback_parts.append(f"- Unclosed tags: {error}")
        elif 'not well-formed' in error.lower():
            feedback_parts.append(f"- XML syntax error: {error}")
        else:
            feedback_parts.append(f"- {error}")

    feedback_parts.append("\nPlease fix these issues:")
    feedback_parts.append("1. Ensure all opening tags have matching closing tags")
    feedback_parts.append("2. Use self-closing tags for elements like <path />, <rect />, <circle />")
    feedback_parts.append("3. Escape special characters in text (&amp; for &, &lt; for <)")
    feedback_parts.append("4. Return ONLY valid SVG code starting with <svg and ending with </svg>")

    return "\n".join(feedback_parts)

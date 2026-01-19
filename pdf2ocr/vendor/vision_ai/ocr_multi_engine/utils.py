# Filepath: code_migration/extraction/ocr_multi_engine/utils.py
# Description: Image preprocessing and enhancement utilities for better OCR
# Layer: Extractor
# References: reference_codebase/OCR_extractor/provider/utils.py

"""
OCR Provider Utilities

Image preprocessing and enhancement utilities for better OCR results.
"""

from typing import Union, Optional, Tuple
from pathlib import Path
import logging


def enhance_image_for_ocr(image, contrast: float = 1.5, sharpness: float = 1.5, brightness: float = 1.1):
    """
    Enhance an image for better OCR results.

    Args:
        image: PIL Image object
        contrast: Contrast enhancement factor (default 1.5)
        sharpness: Sharpness enhancement factor (default 1.5)
        brightness: Brightness enhancement factor (default 1.1)

    Returns:
        Enhanced PIL Image object
    """
    try:
        from PIL import ImageEnhance

        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)

        # Increase sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness)

        # Increase brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)

        return image
    except Exception:
        return image


def convert_to_rgb(image):
    """
    Convert image to RGB mode if needed.

    Args:
        image: PIL Image object

    Returns:
        RGB PIL Image object
    """
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image


def get_image_mime_type(file_path: Union[str, Path]) -> str:
    """
    Get MIME type for an image file based on extension.

    Args:
        file_path: Path to the image file

    Returns:
        MIME type string
    """
    suffix = Path(file_path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
        '.tiff': 'image/tiff',
        '.tif': 'image/tiff',
    }
    return mime_types.get(suffix, 'image/jpeg')


def is_image_file(file_path: Union[str, Path]) -> bool:
    """
    Check if a file is a supported image format.

    Args:
        file_path: Path to check

    Returns:
        True if file is a supported image
    """
    supported = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
    return Path(file_path).suffix.lower() in supported


def is_pdf_file(file_path: Union[str, Path]) -> bool:
    """
    Check if a file is a PDF.

    Args:
        file_path: Path to check

    Returns:
        True if file is a PDF
    """
    return Path(file_path).suffix.lower() == '.pdf'


def get_file_type(file_path: Union[str, Path]) -> str:
    """
    Get the type of a file based on extension.

    Args:
        file_path: Path to the file

    Returns:
        File type string ('image', 'pdf', 'office', 'text', 'unknown')
    """
    suffix = Path(file_path).suffix.lower()

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
    pdf_extensions = {'.pdf'}
    office_extensions = {'.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt'}
    text_extensions = {'.txt', '.md', '.markdown', '.rst', '.log', '.csv', '.json', '.xml', '.html', '.htm'}

    if suffix in image_extensions:
        return 'image'
    elif suffix in pdf_extensions:
        return 'pdf'
    elif suffix in office_extensions:
        return 'office'
    elif suffix in text_extensions:
        return 'text'
    else:
        return 'unknown'


def calculate_text_quality(text: str) -> Tuple[float, dict]:
    """
    Calculate comprehensive quality metrics for extracted text.

    Args:
        text: Extracted text to analyze

    Returns:
        Tuple of (quality_score, metrics_dict)
    """
    if not text or not text.strip():
        return 0.0, {"error": "empty_text"}

    metrics = {}

    # Basic metrics
    char_count = len(text)
    word_count = len(text.split())
    line_count = len(text.split('\n'))

    metrics['char_count'] = char_count
    metrics['word_count'] = word_count
    metrics['line_count'] = line_count

    if word_count > 0:
        metrics['avg_word_length'] = char_count / word_count
    else:
        metrics['avg_word_length'] = 0

    # Character distribution
    alpha_count = sum(1 for c in text if c.isalpha())
    digit_count = sum(1 for c in text if c.isdigit())
    space_count = sum(1 for c in text if c.isspace())
    special_count = char_count - alpha_count - digit_count - space_count

    metrics['alpha_ratio'] = alpha_count / char_count if char_count > 0 else 0
    metrics['digit_ratio'] = digit_count / char_count if char_count > 0 else 0
    metrics['special_ratio'] = special_count / char_count if char_count > 0 else 0

    # OCR artifact detection
    artifacts = ['|||', '###', '---', '~~~', '...', '???', '[?]', '[unreadable]']
    artifact_count = sum(1 for a in artifacts if a in text)
    metrics['artifact_count'] = artifact_count

    # Calculate quality score
    score = 1.0

    # Penalize short text
    if char_count < 50:
        score -= 0.3
    elif char_count < 100:
        score -= 0.1

    # Penalize unrealistic word length
    if metrics['avg_word_length'] < 2 or metrics['avg_word_length'] > 15:
        score -= 0.2

    # Penalize too many special characters
    if metrics['special_ratio'] > 0.3:
        score -= 0.2

    # Penalize artifacts
    score -= artifact_count * 0.05

    # Penalize low alpha ratio
    if metrics['alpha_ratio'] < 0.3:
        score -= 0.1

    # Bonus for structured content
    if '\n\n' in text:  # Paragraphs
        score += 0.05
    if ':' in text:  # Labels
        score += 0.02
    if '-' in text or '•' in text:  # Lists
        score += 0.02

    metrics['quality_score'] = max(0.0, min(1.0, score))

    return metrics['quality_score'], metrics

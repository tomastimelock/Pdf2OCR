# Filepath: code_migration/ai_providers/openai_vision/utils.py
# Description: Utility functions for OpenAI Vision module
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/openai/vision/provider.py

"""
Utility Functions for OpenAI Vision
Helper functions for image encoding, validation, and URL handling.
"""

import base64
import os
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

from PIL import Image

from .model_config import SUPPORTED_FORMATS, IMAGE_LIMITS


def is_url(path: str) -> bool:
    """
    Check if a string is a valid HTTP(S) URL.

    Args:
        path: String to check

    Returns:
        True if the string is a valid URL, False otherwise

    Example:
        >>> is_url("https://example.com/image.jpg")
        True
        >>> is_url("local_file.jpg")
        False
    """
    if not isinstance(path, str):
        return False

    try:
        result = urlparse(path)
        return all([result.scheme in ('http', 'https'), result.netloc])
    except Exception:
        return False


def validate_image_path(image_path: str) -> None:
    """
    Validate that an image file exists and is a supported format.

    Args:
        image_path: Path to the image file

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported

    Example:
        >>> validate_image_path("photo.jpg")  # OK if file exists
        >>> validate_image_path("missing.jpg")  # Raises FileNotFoundError
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {image_path}")

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        supported = ", ".join(SUPPORTED_FORMATS.keys())
        raise ValueError(
            f"Unsupported image format: {path.suffix}. "
            f"Supported formats: {supported}"
        )


def get_image_mime_type(image_path: str) -> str:
    """
    Get the MIME type for an image file.

    Args:
        image_path: Path to the image file

    Returns:
        MIME type string (e.g., "image/png")

    Raises:
        ValueError: If the file format is not supported

    Example:
        >>> get_image_mime_type("photo.jpg")
        'image/jpeg'
        >>> get_image_mime_type("diagram.png")
        'image/png'
    """
    path = Path(image_path)
    suffix = path.suffix.lower()

    mime_type = SUPPORTED_FORMATS.get(suffix)
    if not mime_type:
        supported = ", ".join(SUPPORTED_FORMATS.keys())
        raise ValueError(
            f"Unsupported image format: {suffix}. "
            f"Supported formats: {supported}"
        )

    return mime_type


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 data URL.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded data URL (e.g., "data:image/png;base64,...")

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported

    Example:
        >>> encoded = encode_image_to_base64("photo.jpg")
        >>> print(encoded[:50])
        'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgA...'
    """
    validate_image_path(image_path)
    mime_type = get_image_mime_type(image_path)

    with open(image_path, 'rb') as f:
        image_data = base64.standard_b64encode(f.read()).decode('utf-8')

    return f"data:{mime_type};base64,{image_data}"


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """
    Get the dimensions of an image file.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (width, height) in pixels

    Raises:
        FileNotFoundError: If the file doesn't exist

    Example:
        >>> width, height = get_image_dimensions("photo.jpg")
        >>> print(f"Image is {width}x{height}")
        'Image is 1920x1080'
    """
    validate_image_path(image_path)

    with Image.open(image_path) as img:
        return img.size


def get_image_file_size(image_path: str) -> int:
    """
    Get the file size of an image in bytes.

    Args:
        image_path: Path to the image file

    Returns:
        File size in bytes

    Raises:
        FileNotFoundError: If the file doesn't exist

    Example:
        >>> size = get_image_file_size("photo.jpg")
        >>> print(f"File size: {size / 1024 / 1024:.2f} MB")
        'File size: 2.45 MB'
    """
    validate_image_path(image_path)
    return os.path.getsize(image_path)


def validate_image_size(image_path: str) -> None:
    """
    Validate that an image meets size requirements.

    Args:
        image_path: Path to the image file

    Raises:
        ValueError: If the image exceeds size limits
        FileNotFoundError: If the file doesn't exist

    Example:
        >>> validate_image_size("photo.jpg")  # OK if within limits
        >>> validate_image_size("huge.jpg")  # Raises ValueError if too large
    """
    # Check file size
    file_size_mb = get_image_file_size(image_path) / 1024 / 1024
    max_size_mb = IMAGE_LIMITS["max_file_size_mb"]

    if file_size_mb > max_size_mb:
        raise ValueError(
            f"Image file size ({file_size_mb:.2f} MB) exceeds "
            f"maximum allowed ({max_size_mb} MB)"
        )

    # Check dimensions
    width, height = get_image_dimensions(image_path)
    max_dimension = IMAGE_LIMITS["max_dimension_pixels"]

    if width > max_dimension or height > max_dimension:
        raise ValueError(
            f"Image dimensions ({width}x{height}) exceed "
            f"maximum allowed ({max_dimension}x{max_dimension})"
        )


def resize_image_if_needed(
    image_path: str,
    output_path: Optional[str] = None,
    max_dimension: Optional[int] = None
) -> str:
    """
    Resize an image if it exceeds maximum dimensions.

    Args:
        image_path: Path to the input image file
        output_path: Path to save resized image (optional, defaults to overwrite)
        max_dimension: Maximum dimension in pixels (optional, uses recommended limit)

    Returns:
        Path to the output image (same as input if no resize needed)

    Example:
        >>> resized = resize_image_if_needed("large.jpg", max_dimension=2048)
        >>> print(resized)
        'large.jpg'  # or path to resized version
    """
    validate_image_path(image_path)

    if max_dimension is None:
        max_dimension = IMAGE_LIMITS["recommended_max_pixels"]

    width, height = get_image_dimensions(image_path)

    # Check if resize is needed
    if width <= max_dimension and height <= max_dimension:
        return image_path

    # Calculate new dimensions maintaining aspect ratio
    if width > height:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))
    else:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))

    # Resize image
    with Image.open(image_path) as img:
        resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        if output_path is None:
            output_path = image_path

        resized.save(output_path)

    return output_path


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: File size in bytes

    Returns:
        Formatted string (e.g., "2.45 MB")

    Example:
        >>> format_file_size(2560000)
        '2.44 MB'
        >>> format_file_size(1024)
        '1.00 KB'
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def get_image_info(image_path: str) -> dict:
    """
    Get comprehensive information about an image.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary with image information

    Example:
        >>> info = get_image_info("photo.jpg")
        >>> print(info)
        {
            'path': 'photo.jpg',
            'format': '.jpg',
            'mime_type': 'image/jpeg',
            'dimensions': (1920, 1080),
            'file_size': 2560000,
            'file_size_formatted': '2.44 MB',
            'within_limits': True
        }
    """
    validate_image_path(image_path)

    path = Path(image_path)
    dimensions = get_image_dimensions(image_path)
    file_size = get_image_file_size(image_path)

    # Check if within limits
    within_limits = True
    try:
        validate_image_size(image_path)
    except ValueError:
        within_limits = False

    return {
        'path': str(path),
        'name': path.name,
        'format': path.suffix.lower(),
        'mime_type': get_image_mime_type(image_path),
        'dimensions': dimensions,
        'width': dimensions[0],
        'height': dimensions[1],
        'file_size': file_size,
        'file_size_formatted': format_file_size(file_size),
        'within_limits': within_limits
    }

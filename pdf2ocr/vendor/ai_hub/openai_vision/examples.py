# Filepath: code_migration/ai_providers/openai_vision/examples.py
# Description: Comprehensive usage examples for OpenAI Vision Provider
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/openai/vision/

"""
OpenAI Vision Provider Examples
Comprehensive examples demonstrating all features of the module.
"""

import os
from pathlib import Path
from openai_vision import OpenAIVisionProvider


def example_basic_analysis():
    """Example 1: Basic image analysis"""
    print("\n" + "=" * 60)
    print("Example 1: Basic Image Analysis")
    print("=" * 60)

    provider = OpenAIVisionProvider()

    # Analyze a local image
    print("\nAnalyzing local image...")
    result = provider.analyze_image(
        image_path_or_url="path/to/photo.jpg",
        prompt="What is in this image? Describe in detail."
    )
    print(f"Result: {result[:200]}...")

    # Analyze from URL
    print("\nAnalyzing image from URL...")
    result = provider.analyze_image(
        image_path_or_url="https://example.com/image.jpg",
        prompt="Describe this image"
    )
    print(f"Result: {result[:200]}...")


def example_image_description():
    """Example 2: Generate detailed image descriptions"""
    print("\n" + "=" * 60)
    print("Example 2: Image Description")
    print("=" * 60)

    provider = OpenAIVisionProvider()

    # Get a detailed description
    description = provider.describe_image(
        image="landscape.jpg",
        detail_level="auto"
    )
    print(f"Description: {description}")

    # High-detail description
    detailed = provider.describe_image(
        image="complex_scene.jpg",
        detail_level="high",
        max_tokens=2048
    )
    print(f"\nDetailed description: {detailed}")


def example_text_extraction():
    """Example 3: OCR-like text extraction"""
    print("\n" + "=" * 60)
    print("Example 3: Text Extraction (OCR)")
    print("=" * 60)

    provider = OpenAIVisionProvider()

    # Extract text from document
    text = provider.extract_text("document.png")
    print(f"Extracted text:\n{text}")

    # Extract text from screenshot
    text = provider.extract_text(
        image="screenshot.png",
        model="gpt-4o",
        max_tokens=4096
    )
    print(f"\nScreenshot text:\n{text}")


def example_qa_about_images():
    """Example 4: Question-answering about images"""
    print("\n" + "=" * 60)
    print("Example 4: Question-Answering")
    print("=" * 60)

    provider = OpenAIVisionProvider()

    # Ask specific questions
    questions = [
        "How many people are in this photo?",
        "What is the dominant color?",
        "What time of day does this appear to be?",
        "Are there any text or signs visible?"
    ]

    for question in questions:
        answer = provider.answer_about_image(
            image="photo.jpg",
            question=question
        )
        print(f"\nQ: {question}")
        print(f"A: {answer}")


def example_compare_images():
    """Example 5: Image comparison"""
    print("\n" + "=" * 60)
    print("Example 5: Image Comparison")
    print("=" * 60)

    provider = OpenAIVisionProvider()

    # Compare two versions
    comparison = provider.compare_images(
        image1="version1.png",
        image2="version2.png",
        aspect="UI layout and design"
    )
    print(f"Comparison:\n{comparison}")

    # General comparison
    comparison = provider.compare_images(
        image1="before.jpg",
        image2="after.jpg"
    )
    print(f"\nGeneral comparison:\n{comparison}")


def example_multi_image_analysis():
    """Example 6: Analyze multiple images together"""
    print("\n" + "=" * 60)
    print("Example 6: Multi-Image Analysis")
    print("=" * 60)

    provider = OpenAIVisionProvider()

    # Analyze a sequence
    analysis = provider.analyze_images(
        images=["step1.png", "step2.png", "step3.png"],
        prompt="Describe the progression shown in these steps"
    )
    print(f"Sequence analysis:\n{analysis}")

    # Find common theme
    analysis = provider.analyze_images(
        images=["img1.jpg", "img2.jpg", "img3.jpg"],
        prompt="What is the common theme or pattern across these images?"
    )
    print(f"\nCommon theme:\n{analysis}")


def example_different_models():
    """Example 7: Using different models"""
    print("\n" + "=" * 60)
    print("Example 7: Different Models")
    print("=" * 60)

    # High-quality analysis
    print("\nUsing GPT-4o (default):")
    provider_4o = OpenAIVisionProvider(model="gpt-4o")
    result = provider_4o.analyze_image(
        "complex_image.jpg",
        "Analyze this image in detail"
    )
    print(result[:200])

    # Cost-effective processing
    print("\nUsing GPT-4o-mini (cost-effective):")
    provider_mini = OpenAIVisionProvider(model="gpt-4o-mini")
    result = provider_mini.analyze_image(
        "simple_image.jpg",
        "What is this?"
    )
    print(result[:200])

    # High-quality reasoning
    print("\nUsing GPT-4-turbo (high quality):")
    provider_turbo = OpenAIVisionProvider(model="gpt-4-turbo")
    result = provider_turbo.analyze_image(
        "technical_diagram.png",
        "Explain this technical diagram"
    )
    print(result[:200])


def example_detail_levels():
    """Example 8: Different detail levels"""
    print("\n" + "=" * 60)
    print("Example 8: Detail Levels")
    print("=" * 60)

    provider = OpenAIVisionProvider()

    # Low detail (fast, cheap)
    print("\nLow detail (512x512):")
    result = provider.analyze_image(
        "icon.png",
        "What color is this icon?",
        detail="low"
    )
    print(result)

    # High detail (accurate, slower)
    print("\nHigh detail (tiled):")
    result = provider.analyze_image(
        "detailed_diagram.png",
        "Explain all the components in this diagram",
        detail="high"
    )
    print(result[:200])

    # Auto detail (balanced)
    print("\nAuto detail (model decides):")
    result = provider.analyze_image(
        "photo.jpg",
        "Describe this photo",
        detail="auto"
    )
    print(result[:200])


def example_batch_processing():
    """Example 9: Batch processing multiple images"""
    print("\n" + "=" * 60)
    print("Example 9: Batch Processing")
    print("=" * 60)

    provider = OpenAIVisionProvider(model="gpt-4o-mini")  # Cost-effective

    image_folder = "images"
    results = []

    # Process all images in a folder
    if os.path.exists(image_folder):
        for filename in os.listdir(image_folder):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                print(f"\nProcessing {filename}...")

                result = provider.describe_image(
                    os.path.join(image_folder, filename),
                    detail_level="low"  # Fast processing
                )

                results.append({
                    'file': filename,
                    'description': result
                })

        # Print results
        for item in results:
            print(f"\n{item['file']}:")
            print(f"  {item['description'][:100]}...")
    else:
        print(f"Folder '{image_folder}' not found. Skipping batch processing demo.")


def example_structured_extraction():
    """Example 10: Structured data extraction"""
    print("\n" + "=" * 60)
    print("Example 10: Structured Data Extraction")
    print("=" * 60)

    provider = OpenAIVisionProvider()

    # Extract structured data from invoice
    print("\nExtracting invoice data:")
    result = provider.analyze_image(
        "invoice.png",
        prompt="""Extract the following information from this invoice:
        - Invoice Number: [number]
        - Date: [date]
        - Total Amount: [amount]
        - Vendor Name: [name]
        - Line Items: [list]
        """,
        detail="high"
    )
    print(result)

    # Extract chart data
    print("\nExtracting chart data:")
    result = provider.analyze_image(
        "chart.png",
        prompt="""Analyze this chart and provide:
        1. Chart Type: [type]
        2. Title: [title]
        3. X-axis Label: [label]
        4. Y-axis Label: [label]
        5. Key Data Points: [list]
        6. Main Trend: [description]
        """,
        detail="high"
    )
    print(result)


def example_model_info():
    """Example 11: Get model information"""
    print("\n" + "=" * 60)
    print("Example 11: Model Information")
    print("=" * 60)

    # List all models
    print("\nAvailable models:")
    models = OpenAIVisionProvider.list_models()
    for name, info in models.items():
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        print(f"  Detail levels: {', '.join(info['detail_levels'])}")
        if info.get('notes'):
            print(f"  Notes: {info['notes']}")

    # Get specific model info
    print("\n" + "-" * 60)
    print("GPT-4o details:")
    info = OpenAIVisionProvider.get_model_info("gpt-4o")
    if info:
        for key, value in info.items():
            print(f"  {key}: {value}")


def example_utility_functions():
    """Example 12: Using utility functions"""
    print("\n" + "=" * 60)
    print("Example 12: Utility Functions")
    print("=" * 60)

    from openai_vision.utils import (
        is_url,
        get_image_info,
        get_image_dimensions,
        get_image_mime_type
    )

    # Check if string is URL
    print("\nChecking URLs:")
    print(f"is_url('https://example.com/img.jpg'): {is_url('https://example.com/img.jpg')}")
    print(f"is_url('local_file.jpg'): {is_url('local_file.jpg')}")

    # Get image info
    image_path = "photo.jpg"
    if os.path.exists(image_path):
        print(f"\nImage info for '{image_path}':")
        info = get_image_info(image_path)
        for key, value in info.items():
            print(f"  {key}: {value}")

        # Get dimensions
        width, height = get_image_dimensions(image_path)
        print(f"\nDimensions: {width}x{height}")

        # Get MIME type
        mime = get_image_mime_type(image_path)
        print(f"MIME type: {mime}")
    else:
        print(f"\nImage '{image_path}' not found. Skipping utility demo.")


def example_error_handling():
    """Example 13: Error handling"""
    print("\n" + "=" * 60)
    print("Example 13: Error Handling")
    print("=" * 60)

    provider = OpenAIVisionProvider()

    # Handle file not found
    print("\nHandling missing file:")
    try:
        result = provider.analyze_image(
            "nonexistent.jpg",
            "What is this?"
        )
    except FileNotFoundError as e:
        print(f"Caught error: {e}")

    # Handle invalid format
    print("\nHandling invalid format:")
    try:
        result = provider.analyze_image(
            "document.pdf",  # Not a supported image format
            "What is this?"
        )
    except ValueError as e:
        print(f"Caught error: {e}")

    # Handle API errors
    print("\nHandling API errors:")
    try:
        invalid_provider = OpenAIVisionProvider(api_key="invalid-key")
        result = invalid_provider.analyze_image(
            "https://example.com/image.jpg",
            "What is this?"
        )
    except Exception as e:
        print(f"Caught API error: {type(e).__name__}")


def example_docflow_integration():
    """Example 14: DocFlow integration patterns"""
    print("\n" + "=" * 60)
    print("Example 14: DocFlow Integration")
    print("=" * 60)

    provider = OpenAIVisionProvider()

    # Document classification
    print("\n1. Document Classification:")
    doc_type = provider.analyze_image(
        "document.png",
        "Classify this document as one of: invoice, receipt, contract, letter, form, report"
    )
    print(f"Document type: {doc_type}")

    # OCR for scanned documents
    print("\n2. OCR for Scanned Documents:")
    text = provider.extract_text("scanned_page.png")
    print(f"Extracted text: {text[:200]}...")

    # Extract structured data
    print("\n3. Structured Data Extraction:")
    data = provider.analyze_image(
        "form.png",
        prompt="""Extract all form fields and their values.
        Format as: Field Name: Value
        """,
        detail="high"
    )
    print(f"Form data:\n{data}")

    # Quality check
    print("\n4. Document Quality Check:")
    quality = provider.analyze_image(
        "document.png",
        "Assess the quality of this document scan. Is it clear enough to read? Any issues?"
    )
    print(f"Quality assessment: {quality}")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print(" OpenAI Vision Provider - Comprehensive Examples")
    print("=" * 70)

    examples = [
        ("Basic Analysis", example_basic_analysis),
        ("Image Description", example_image_description),
        ("Text Extraction (OCR)", example_text_extraction),
        ("Question-Answering", example_qa_about_images),
        ("Image Comparison", example_compare_images),
        ("Multi-Image Analysis", example_multi_image_analysis),
        ("Different Models", example_different_models),
        ("Detail Levels", example_detail_levels),
        ("Batch Processing", example_batch_processing),
        ("Structured Extraction", example_structured_extraction),
        ("Model Information", example_model_info),
        ("Utility Functions", example_utility_functions),
        ("Error Handling", example_error_handling),
        ("DocFlow Integration", example_docflow_integration),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nNote: Most examples require actual image files to run.")
    print("Modify the image paths in each example to match your files.")

    # Uncomment to run specific examples:
    # example_model_info()
    # example_utility_functions()
    # example_error_handling()


if __name__ == "__main__":
    main()

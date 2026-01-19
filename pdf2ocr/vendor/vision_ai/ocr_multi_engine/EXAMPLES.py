# Filepath: code_migration/extraction/ocr_multi_engine/EXAMPLES.py
# Description: Usage examples for OCR Multi-Engine Module
# Layer: Extractor

"""
OCR Multi-Engine Module - Usage Examples

This file demonstrates various usage patterns for the OCR multi-engine module.
"""

import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)


def example_1_basic_usage():
    """Example 1: Basic single-engine OCR"""
    print("\n=== Example 1: Basic Single-Engine OCR ===")

    from ocr_multi_engine import OCRProviderFactory

    # Create Tesseract OCR (local, no API key needed)
    ocr = OCRProviderFactory.create("tesseract", config={
        "tesseract_lang": "eng"
    })

    # Extract text from image
    text = ocr.extract_text("sample.jpg")
    print(f"Extracted {len(text)} characters")
    print(f"Quality: {ocr.get_quality_score(text):.2f}")


def example_2_multi_engine():
    """Example 2: Multi-engine with automatic fallback"""
    print("\n=== Example 2: Multi-Engine with Fallback ===")

    from ocr_multi_engine import OCRProviderFactory

    # Create multi-engine OCR (tries engines in order)
    ocr = OCRProviderFactory.create_multi_engine(
        engine_order=["openai", "mistral", "tesseract"],
        config={
            "openai_api_key": "sk-...",
            "mistral_api_key": "your-key",
            "tesseract_lang": "eng",
            "min_quality": 0.7  # Minimum acceptable quality
        }
    )

    print(f"Using: {ocr.get_provider_name()}")

    # Will try OpenAI first, fallback to Mistral if quality < 0.7, etc.
    text = ocr.extract_text("complex_document.jpg")
    print(f"Extracted {len(text)} characters")

    # Get statistics
    stats = ocr.get_stats()
    print(f"Stats: {stats}")


def example_3_swedish_documents():
    """Example 3: Processing Swedish documents"""
    print("\n=== Example 3: Swedish Document Processing ===")

    from ocr_multi_engine import OCRConfig, OCRProviderFactory

    # Configure for Swedish language
    config = OCRConfig(
        tesseract_lang="swe+eng",  # Swedish + English
        engine_order=["openai", "tesseract"],
        min_quality=0.7,
        openai_api_key="sk-..."
    )

    # Create OCR
    ocr = OCRProviderFactory.create_multi_engine(
        config=config.to_dict(),
        engine_order=config.engine_order
    )

    # Process Swedish municipal report
    text = ocr.extract_text("årsredovisning.jpg")

    # Swedish characters (å, ä, ö) should be preserved
    print("Checking Swedish characters...")
    for char in ['å', 'ä', 'ö', 'Å', 'Ä', 'Ö']:
        count = text.count(char)
        if count > 0:
            print(f"  Found {count} instances of '{char}'")


def example_4_document_processor():
    """Example 4: Using DocumentProcessor for images and PDFs"""
    print("\n=== Example 4: Document Processor ===")

    from ocr_multi_engine import create_document_processor

    # Create processor (handles both images and PDFs)
    processor = create_document_processor()

    # Process different file types
    files = [
        "document.pdf",
        "scan.jpg",
        "photo.png"
    ]

    for file_path in files:
        if Path(file_path).exists():
            print(f"\nProcessing {file_path}...")
            text = processor.process(file_path)
            print(f"Extracted {len(text)} characters")

    # Get overall statistics
    stats = processor.get_stats()
    print(f"\nTotal processed: {stats['total_processed']}")
    print(f"Success rate: {stats['success_rate']}%")
    print(f"By processor: {stats['by_processor']}")


def example_5_pdf_with_ocr_fallback():
    """Example 5: PDF processing with OCR fallback"""
    print("\n=== Example 5: PDF with OCR Fallback ===")

    from ocr_multi_engine import PDFProcessor, OCRProviderFactory

    # Create OCR for fallback
    ocr = OCRProviderFactory.create_multi_engine()

    # Create PDF processor with OCR fallback
    processor = PDFProcessor(ocr, config={
        "min_text_length": 100,  # If direct extraction < 100 chars, use OCR
        "use_ocr_fallback": True,
        "ocr_dpi": 200  # DPI for OCR conversion
    })

    # Process PDF (tries direct extraction first, then OCR)
    text = processor.process("document.pdf")
    print(f"Extracted {len(text)} characters")


def example_6_custom_openai_prompt():
    """Example 6: Custom OpenAI Vision prompt"""
    print("\n=== Example 6: Custom OpenAI Prompt ===")

    from ocr_multi_engine import OpenAIVision

    custom_prompt = """
    Extract all text from this Swedish municipal report.
    Pay special attention to:
    - Legal references (e.g., "1 kap. 1 § Regeringsformen")
    - Dates in YYYY-MM-DD format
    - Budget figures in Swedish format (1 234,56 kr)
    - Administrative terminology
    - Section headers and structure

    Preserve the original layout and formatting.
    """

    ocr = OpenAIVision(
        api_key="sk-...",
        model="gpt-4o",
        prompt=custom_prompt
    )

    text = ocr.extract_text("kommunal_rapport.jpg")
    print(f"Extracted {len(text)} characters")


def example_7_quality_assessment():
    """Example 7: Text quality assessment"""
    print("\n=== Example 7: Quality Assessment ===")

    from ocr_multi_engine import calculate_text_quality

    # Sample extracted texts
    texts = {
        "good": "This is a well-formatted document with clear text and proper structure.",
        "poor": "|||###???...|||",
        "medium": "Some text with a few artifacts|||but mostly readable"
    }

    for label, text in texts.items():
        quality, metrics = calculate_text_quality(text)
        print(f"\n{label.upper()} TEXT:")
        print(f"  Quality: {quality:.2f}")
        print(f"  Metrics: {metrics}")


def example_8_batch_processing():
    """Example 8: Batch processing with statistics"""
    print("\n=== Example 8: Batch Processing ===")

    from ocr_multi_engine import create_document_processor
    from pathlib import Path

    processor = create_document_processor()

    # Process all images in a directory
    image_dir = Path("documents/")
    if image_dir.exists():
        for image_file in image_dir.glob("*.jpg"):
            print(f"Processing {image_file.name}...")
            try:
                text = processor.process(image_file)
                print(f"  Extracted {len(text)} characters")
            except Exception as e:
                print(f"  Error: {e}")

        # Final statistics
        stats = processor.get_stats()
        print(f"\nBatch Summary:")
        print(f"  Total: {stats['total_processed']}")
        print(f"  Success: {stats['successful']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Success rate: {stats['success_rate']}%")


def example_9_engine_comparison():
    """Example 9: Compare different OCR engines"""
    print("\n=== Example 9: Engine Comparison ===")

    from ocr_multi_engine import OCRProviderFactory
    import time

    engines = ["tesseract", "openai", "mistral"]
    test_image = "sample.jpg"

    if Path(test_image).exists():
        for engine_name in engines:
            try:
                print(f"\nTesting {engine_name}...")

                # Create engine
                ocr = OCRProviderFactory.create(engine_name, config={
                    "openai_api_key": "sk-...",
                    "mistral_api_key": "your-key",
                    "tesseract_lang": "eng"
                })

                # Check availability
                if not ocr.is_available():
                    print(f"  {engine_name} not available (missing API key or dependencies)")
                    continue

                # Time the extraction
                start = time.time()
                text = ocr.extract_text(test_image)
                duration = time.time() - start

                # Assess quality
                quality = ocr.get_quality_score(text)

                print(f"  Provider: {ocr.get_provider_name()}")
                print(f"  Characters: {len(text)}")
                print(f"  Quality: {quality:.2f}")
                print(f"  Time: {duration:.2f}s")

            except Exception as e:
                print(f"  Error: {e}")


def example_10_environment_config():
    """Example 10: Using environment configuration"""
    print("\n=== Example 10: Environment Configuration ===")

    import os
    from ocr_multi_engine import OCRConfig, OCRProviderFactory

    # Set environment variables (usually done in .env file)
    os.environ["OPENAI_API_KEY"] = "sk-..."
    os.environ["MISTRAL_API_KEY"] = "your-key"
    os.environ["OCR_ENGINE_ORDER"] = "openai,mistral,tesseract"
    os.environ["OCR_MIN_QUALITY"] = "0.75"
    os.environ["OCR_TESSERACT_LANG"] = "swe+eng"

    # Load config from environment
    config = OCRConfig.from_env()

    print(f"Engine order: {config.engine_order}")
    print(f"Min quality: {config.min_quality}")
    print(f"Tesseract lang: {config.tesseract_lang}")

    # Validate
    errors = config.validate()
    if errors:
        print(f"Validation errors: {errors}")
    else:
        print("Configuration valid!")

    # Create OCR with environment config
    ocr = OCRProviderFactory.create_multi_engine(
        config=config.to_dict(),
        engine_order=config.engine_order
    )


def example_11_error_handling():
    """Example 11: Proper error handling"""
    print("\n=== Example 11: Error Handling ===")

    from ocr_multi_engine import (
        OCRProviderFactory,
        OCRError,
        OCRProviderUnavailableError,
        ProcessingError,
        UnsupportedFileTypeError
    )

    try:
        # Try to create provider without API key
        ocr = OCRProviderFactory.create("openai")
    except OCRProviderUnavailableError as e:
        print(f"Provider unavailable: {e}")

    try:
        # Try to process unsupported file type
        processor = create_document_processor()
        processor.process("document.docx")
    except UnsupportedFileTypeError as e:
        print(f"Unsupported file: {e}")

    try:
        # Try to extract from non-existent file
        ocr = OCRProviderFactory.create("tesseract")
        ocr.extract_text("nonexistent.jpg")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except OCRError as e:
        print(f"OCR error: {e}")


def example_12_image_preprocessing():
    """Example 12: Image preprocessing for better OCR"""
    print("\n=== Example 12: Image Preprocessing ===")

    from ocr_multi_engine import enhance_image_for_ocr, convert_to_rgb
    from PIL import Image
    import tempfile

    # Load image
    image = Image.open("sample.jpg")
    print(f"Original size: {image.size}")
    print(f"Original mode: {image.mode}")

    # Convert to RGB
    image = convert_to_rgb(image)
    print(f"After RGB conversion: {image.mode}")

    # Enhance for OCR
    enhanced = enhance_image_for_ocr(
        image,
        contrast=1.5,    # Increase contrast
        sharpness=1.5,   # Increase sharpness
        brightness=1.1   # Slight brightness boost
    )

    # Save enhanced version
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        enhanced.save(tmp.name)
        print(f"Enhanced image saved to: {tmp.name}")

        # OCR the enhanced image
        from ocr_multi_engine import create_default_ocr
        ocr = create_default_ocr()
        text = ocr.extract_text(tmp.name)
        print(f"Extracted {len(text)} characters from enhanced image")


if __name__ == "__main__":
    print("OCR Multi-Engine Module - Usage Examples")
    print("=" * 50)

    # Run examples (comment out as needed)
    # example_1_basic_usage()
    # example_2_multi_engine()
    # example_3_swedish_documents()
    # example_4_document_processor()
    # example_5_pdf_with_ocr_fallback()
    # example_6_custom_openai_prompt()
    # example_7_quality_assessment()
    # example_8_batch_processing()
    # example_9_engine_comparison()
    # example_10_environment_config()
    # example_11_error_handling()
    # example_12_image_preprocessing()

    print("\nAll examples defined. Uncomment specific examples to run them.")

"""
Runner script for full PDF2OCR pipeline with image regeneration and export.

Processes the specified PDF with:
- PDF splitting to images
- OCR using Mistral AI
- Table extraction to JSON/SVG
- Chart detection and SVG regeneration
- Image regeneration using OpenAI DALL-E
- Structured JSON output
- Export to Word and PDF formats
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
PDF_PATH = Path(r"C:\Users\tomas\PycharmProjects\Pdf2OCR\data\10-strategier-for-klimatresistens---gis-vagledning-for-energieffektiviserande-mikroklimatsmodifiering.pdf")
OUTPUT_DIR = Path(r"C:\Users\tomas\PycharmProjects\Pdf2OCR\output")

# Settings
DPI = 200
IMAGE_FORMAT = "jpg"
VERBOSE = True


def main():
    """Run the full PDF2OCR pipeline."""

    # Check API keys
    mistral_key = os.getenv("MISTRAL_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    print("=" * 60)
    print("PDF2OCR Full Pipeline Runner")
    print("=" * 60)

    print(f"\nInput PDF: {PDF_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")

    print("\nAPI Keys Status:")
    print(f"  MISTRAL_API_KEY: {'[OK]' if mistral_key else '[MISSING]'}")
    print(f"  OPENAI_API_KEY: {'[OK]' if openai_key else '[MISSING]'}")
    print(f"  ANTHROPIC_API_KEY: {'[OK]' if anthropic_key else '[MISSING]'}")

    if not PDF_PATH.exists():
        print(f"\nError: PDF file not found: {PDF_PATH}")
        sys.exit(1)

    if not mistral_key:
        print("\nError: MISTRAL_API_KEY is required for OCR.")
        print("Set it in your .env file or environment variables.")
        sys.exit(1)

    # Import processors
    from pdf2ocr.processors.pdf_splitter import PDFSplitter
    from pdf2ocr.processors.ocr_processor import OCRProcessor
    from pdf2ocr.providers.mistral_provider import MistralOCRProvider
    from pdf2ocr.extractors.information_extractor import InformationExtractor
    from pdf2ocr.extractors.table_extractor import TableExtractor
    from pdf2ocr.extractors.document_structurer import DocumentStructurer

    # Setup output directories
    doc_name = PDF_PATH.stem
    doc_output = OUTPUT_DIR / doc_name
    pages_dir = doc_output / "pages"
    txt_dir = doc_output / "txt"
    images_dir = doc_output / "images"
    json_dir = doc_output / "json"
    svg_dir = doc_output / "svg"
    regen_dir = doc_output / "regenerated"

    # Initialize processors
    splitter = PDFSplitter(dpi=DPI, image_format=IMAGE_FORMAT)
    provider = MistralOCRProvider(api_key=mistral_key)
    ocr_processor = OCRProcessor(provider=provider)
    extractor = InformationExtractor()
    table_extractor = TableExtractor()
    doc_structurer = DocumentStructurer()

    # Get PDF metadata
    print("\n" + "-" * 60)
    metadata = splitter.get_metadata(PDF_PATH)
    print(f"Document: {metadata.title or doc_name}")
    print(f"Pages: {metadata.page_count}")
    if metadata.author:
        print(f"Author: {metadata.author}")

    # Step 1: Split PDF to images
    print("\n" + "-" * 60)
    print("Step 1: Splitting PDF to images...")
    pages = splitter.split_to_images(PDF_PATH, pages_dir)
    print(f"  Created {len(pages)} page images")

    # Step 2: Extract embedded images
    print("\nStep 2: Extracting embedded images...")
    embedded = splitter.extract_images(PDF_PATH, images_dir)
    print(f"  Found {len(embedded)} embedded images")

    # Step 3: OCR Processing
    print("\n" + "-" * 60)
    print("Step 3: Running OCR with Mistral AI...")

    def ocr_progress(current, total, result):
        if VERBOSE:
            print(f"  Page {current}/{total}: quality={result.quality_score:.2f}")

    ocr_result = ocr_processor.process_pdf(
        PDF_PATH,
        txt_dir,
        progress_callback=ocr_progress
    )

    print(f"  OCR complete: {ocr_result.successful_pages}/{ocr_result.total_pages} pages")
    print(f"  Average quality: {ocr_result.average_quality:.2f}")

    # Save combined text
    combined_path = doc_output / "combined.txt"
    ocr_processor.save_combined_text(ocr_result, combined_path)
    print(f"  Saved combined text: {combined_path}")

    # Step 4: Extract information
    print("\nStep 4: Extracting structured information...")
    info = extractor.extract_from_result(ocr_result)

    info_json_path = doc_output / "extracted_info.json"
    extractor.save_to_json(info, info_json_path)

    summary_path = doc_output / "summary.txt"
    extractor.save_summary(info, summary_path)

    print(f"  Found: {len(info.key_values)} key-values, "
          f"{len(info.emails)} emails, {len(info.dates)} dates")

    # Step 5: Table extraction
    print("\n" + "-" * 60)
    print("Step 5: Extracting tables...")
    table_result = table_extractor.extract_and_save(
        PDF_PATH,
        doc_output,
        save_svg=True
    )

    print(f"  Tables: {table_result.total_tables} extracted "
          f"from {table_result.pages_with_tables} pages")

    # Step 6: Chart regeneration (if API keys available)
    if openai_key and anthropic_key:
        print("\n" + "-" * 60)
        print("Step 6: Detecting and regenerating charts...")

        from pdf2ocr.processors.chart_regenerator import ChartRegenerator
        chart_regenerator = ChartRegenerator(
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key
        )

        def chart_progress(page_num, total, charts):
            if VERBOSE and charts:
                print(f"  Page {page_num}/{total}: {len(charts)} charts")

        chart_result = chart_regenerator.process_document(
            pages_dir,
            svg_dir,
            progress_callback=chart_progress
        )

        print(f"  Charts: {chart_result.total_regenerated}/{chart_result.total_detected} "
              f"regenerated from {chart_result.pages_with_charts} pages")
    else:
        print("\n" + "-" * 60)
        print("Step 6: Skipping chart regeneration (missing API keys)")

    # Step 7: Image regeneration (if OpenAI key available)
    if openai_key and images_dir.exists() and any(images_dir.iterdir()):
        print("\n" + "-" * 60)
        print("Step 7: Regenerating images (photos, illustrations)...")

        from pdf2ocr.processors.image_regenerator import ImageRegenerator
        image_regenerator = ImageRegenerator(api_key=openai_key)

        def img_progress(current, total, result):
            if VERBOSE:
                status = "OK" if result.success else "FAILED"
                print(f"  Image {current}/{total}: {status}")

        img_result = image_regenerator.process_extracted_images(
            images_dir,
            regen_dir,
            progress_callback=img_progress
        )

        print(f"  Images: {img_result.total_regenerated}/{img_result.total_detected} "
              f"regenerated, {img_result.total_skipped} skipped")
    else:
        print("\n" + "-" * 60)
        print("Step 7: Skipping image regeneration (no images or missing API key)")

    # Step 8: Create structured output
    print("\n" + "-" * 60)
    print("Step 8: Creating structured document output...")

    structured_output_path = doc_output / "document.json"
    structured_doc = doc_structurer.structure_document(
        doc_output,
        PDF_PATH,
        structured_output_path
    )

    summary = structured_doc.processing_summary
    print(f"  Structured output: {structured_output_path}")
    print(f"    - Pages: {summary.get('total_pages', 0)}")
    print(f"    - Words: {summary.get('total_words', 0)}")
    print(f"    - Tables: {summary.get('total_tables', 0)}")
    print(f"    - Charts: {summary.get('total_charts', 0)}")
    print(f"    - Sections: {summary.get('sections_detected', 0)}")

    # Step 9: Export to Word
    print("\n" + "-" * 60)
    print("Step 9: Exporting to Word document...")

    try:
        from pdf2ocr.exporters import WordExporter
        word_exporter = WordExporter()

        word_output = doc_output / f"{doc_name}.docx"
        word_path = word_exporter.export(
            doc_output,
            word_output,
            include_regenerated_images=True,
            include_original_images=False
        )
        print(f"  Word document saved: {word_path}")
    except ImportError as e:
        print(f"  Skipping Word export (missing dependency: {e})")
        print("  Install with: pip install pdf2ocr[export]")
    except Exception as e:
        print(f"  Error exporting to Word: {e}")

    # Step 10: Export to PDF
    print("\n" + "-" * 60)
    print("Step 10: Exporting to PDF document...")

    try:
        from pdf2ocr.exporters import PDFExporter
        pdf_exporter = PDFExporter()

        pdf_output = doc_output / f"{doc_name}_processed.pdf"
        pdf_path_out = pdf_exporter.export(
            doc_output,
            pdf_output,
            include_regenerated_images=True,
            include_original_images=False
        )
        print(f"  PDF document saved: {pdf_path_out}")
    except ImportError as e:
        print(f"  Skipping PDF export (missing dependency: {e})")
        print("  Install with: pip install pdf2ocr[export]")
    except Exception as e:
        print(f"  Error exporting to PDF: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {doc_output}")
    print("\nGenerated files:")
    print(f"  - Page images: {pages_dir}")
    print(f"  - OCR text: {txt_dir}")
    print(f"  - Combined text: {combined_path}")
    print(f"  - Tables (JSON): {json_dir}")
    print(f"  - Charts/Tables (SVG): {svg_dir}")
    if regen_dir.exists():
        print(f"  - Regenerated images: {regen_dir}")
    print(f"  - Structured data: {structured_output_path}")
    print(f"  - Word export: {doc_output / f'{doc_name}.docx'}")
    print(f"  - PDF export: {doc_output / f'{doc_name}_processed.pdf'}")


if __name__ == "__main__":
    main()

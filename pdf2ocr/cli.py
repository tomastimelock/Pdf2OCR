"""Command-line interface for PDF2OCR."""

import os
import sys
import argparse
from pathlib import Path

from dotenv import load_dotenv


def main():
    """Main entry point for the CLI."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="pdf2ocr",
        description="PDF to OCR processing with Mistral AI, chart regeneration, and table extraction"
    )

    parser.add_argument(
        "input",
        type=str,
        help="Input PDF file or directory containing PDFs"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory (default: ./output)"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for image conversion (default: 200)"
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["jpg", "png"],
        default="jpg",
        help="Image format for page conversion (default: jpg)"
    )

    parser.add_argument(
        "--split-only",
        action="store_true",
        help="Only split PDF into pages/images, skip OCR"
    )

    parser.add_argument(
        "--ocr-only",
        action="store_true",
        help="Only perform OCR on existing images"
    )

    parser.add_argument(
        "--extract-images",
        action="store_true",
        help="Extract embedded images from PDF"
    )

    parser.add_argument(
        "--extract-info",
        action="store_true",
        help="Extract structured information from OCR text"
    )

    parser.add_argument(
        "--extract-tables",
        action="store_true",
        help="Extract tables from PDF and save as JSON/SVG"
    )

    parser.add_argument(
        "--extract-charts",
        action="store_true",
        help="Detect and regenerate charts as SVG (requires OpenAI and Anthropic API keys)"
    )

    parser.add_argument(
        "--structure-output",
        action="store_true",
        help="Create structured JSON output combining all extracted data"
    )

    parser.add_argument(
        "--regenerate-images",
        action="store_true",
        help="Regenerate images (photos, illustrations) using OpenAI DALL-E (requires OpenAI API key)"
    )

    parser.add_argument(
        "--export-word",
        type=str,
        default=None,
        metavar="PATH",
        help="Export processed document to Word (.docx) format"
    )

    parser.add_argument(
        "--export-pdf",
        type=str,
        default=None,
        metavar="PATH",
        help="Export processed document to PDF format"
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full pipeline: OCR, tables, charts (if keys available), and structured output"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Mistral API key (or set MISTRAL_API_KEY env var)"
    )

    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key for chart detection (or set OPENAI_API_KEY env var)"
    )

    parser.add_argument(
        "--anthropic-api-key",
        type=str,
        default=None,
        help="Anthropic API key for SVG generation (or set ANTHROPIC_API_KEY env var)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Handle --full flag
    if args.full:
        args.extract_images = True
        args.extract_info = True
        args.extract_tables = True
        args.extract_charts = True
        args.structure_output = True

    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else Path("./output")

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    api_key = args.api_key or os.getenv("MISTRAL_API_KEY")
    openai_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    anthropic_key = args.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

    if not args.split_only and not api_key:
        print("Warning: No Mistral API key provided. OCR will fail.")
        print("Set MISTRAL_API_KEY environment variable or use --api-key")

    if args.extract_charts and (not openai_key or not anthropic_key):
        print("Warning: Chart extraction requires both OpenAI and Anthropic API keys.")
        if not openai_key:
            print("  Missing: OPENAI_API_KEY")
        if not anthropic_key:
            print("  Missing: ANTHROPIC_API_KEY")
        args.extract_charts = False

    if args.regenerate_images and not openai_key:
        print("Warning: Image regeneration requires OpenAI API key.")
        print("  Missing: OPENAI_API_KEY")
        args.regenerate_images = False

    from pdf2ocr.processors.pdf_splitter import PDFSplitter
    from pdf2ocr.processors.ocr_processor import OCRProcessor
    from pdf2ocr.extractors.information_extractor import InformationExtractor
    from pdf2ocr.providers.mistral_provider import MistralOCRProvider

    splitter = PDFSplitter(dpi=args.dpi, image_format=args.format)

    provider = MistralOCRProvider(api_key=api_key) if api_key else None
    ocr_processor = OCRProcessor(provider=provider) if provider else None

    extractor = InformationExtractor()

    # Initialize table extractor
    table_extractor = None
    if args.extract_tables:
        from pdf2ocr.extractors.table_extractor import TableExtractor
        table_extractor = TableExtractor()

    # Initialize chart regenerator if needed
    chart_regenerator = None
    if args.extract_charts and openai_key and anthropic_key:
        from pdf2ocr.processors.chart_regenerator import ChartRegenerator
        chart_regenerator = ChartRegenerator(
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key
        )

    # Initialize image regenerator if needed
    image_regenerator = None
    if args.regenerate_images and openai_key:
        from pdf2ocr.processors.image_regenerator import ImageRegenerator
        image_regenerator = ImageRegenerator(api_key=openai_key)

    # Initialize document structurer
    doc_structurer = None
    if args.structure_output:
        from pdf2ocr.extractors.document_structurer import DocumentStructurer
        doc_structurer = DocumentStructurer()

    # Initialize exporters if needed
    word_exporter = None
    pdf_exporter = None
    if args.export_word:
        from pdf2ocr.exporters import WordExporter
        word_exporter = WordExporter()
    if args.export_pdf:
        from pdf2ocr.exporters import PDFExporter
        pdf_exporter = PDFExporter()

    if input_path.is_file():
        pdfs = [input_path]
    else:
        pdfs = list(input_path.glob("*.pdf"))
        if not pdfs:
            print(f"No PDF files found in {input_path}")
            sys.exit(1)

    for pdf_path in pdfs:
        print(f"\nProcessing: {pdf_path.name}")
        print("-" * 50)

        doc_output = output_dir / pdf_path.stem
        pages_dir = doc_output / "pages"
        txt_dir = doc_output / "txt"
        images_dir = doc_output / "images"
        json_dir = doc_output / "json"
        svg_dir = doc_output / "svg"

        metadata = splitter.get_metadata(pdf_path)
        print(f"  Pages: {metadata.page_count}")
        if metadata.title:
            print(f"  Title: {metadata.title}")

        if not args.ocr_only:
            print(f"  Splitting to {args.format.upper()} images...")
            pages = splitter.split_to_images(pdf_path, pages_dir)
            print(f"  Created {len(pages)} page images")

            if args.extract_images:
                print("  Extracting embedded images...")
                embedded = splitter.extract_images(pdf_path, images_dir)
                print(f"  Found {len(embedded)} embedded images")

        if args.split_only:
            continue

        # OCR Processing
        ocr_result = None
        if not ocr_processor:
            print("  Skipping OCR (no API key)")
        else:
            if args.ocr_only:
                if not pages_dir.exists():
                    print(f"  Error: Pages directory not found: {pages_dir}")
                    continue
                print("  Running OCR on existing images...")
                ocr_result = ocr_processor.process_images_directory(
                    pages_dir,
                    txt_dir,
                    progress_callback=lambda c, t, r: print(
                        f"    Page {c}/{t}: quality={r.quality_score:.2f}"
                    ) if args.verbose else None
                )
            else:
                print("  Running OCR...")
                ocr_result = ocr_processor.process_pdf(
                    pdf_path,
                    txt_dir,
                    progress_callback=lambda c, t, r: print(
                        f"    Page {c}/{t}: quality={r.quality_score:.2f}"
                    ) if args.verbose else None
                )

            print(f"  OCR complete: {ocr_result.successful_pages}/{ocr_result.total_pages} pages")
            print(f"  Average quality: {ocr_result.average_quality:.2f}")

            combined_path = doc_output / "combined.txt"
            ocr_processor.save_combined_text(ocr_result, combined_path)
            print(f"  Saved combined text: {combined_path}")

            if args.extract_info:
                print("  Extracting information...")
                info = extractor.extract_from_result(ocr_result)

                info_json_path = doc_output / "extracted_info.json"
                extractor.save_to_json(info, info_json_path)
                print(f"  Saved: {info_json_path}")

                summary_path = doc_output / "summary.txt"
                extractor.save_summary(info, summary_path)
                print(f"  Saved: {summary_path}")

                print(f"  Found: {len(info.key_values)} key-values, "
                      f"{len(info.emails)} emails, {len(info.dates)} dates")

        # Table extraction
        if table_extractor:
            print("  Extracting tables...")
            table_result = table_extractor.extract_and_save(
                pdf_path,
                doc_output,
                save_svg=True
            )

            print(f"  Tables: {table_result.total_tables} extracted "
                  f"from {table_result.pages_with_tables} pages")

            if table_result.total_tables > 0:
                print(f"  JSON files saved to: {json_dir}")
                print(f"  SVG files saved to: {svg_dir}")

        # Chart extraction
        if chart_regenerator and pages_dir.exists():
            print("  Detecting and regenerating charts...")

            def chart_progress(page_num, total, charts):
                if args.verbose and charts:
                    print(f"    Page {page_num}/{total}: {len(charts)} charts")

            chart_result = chart_regenerator.process_document(
                pages_dir,
                svg_dir,
                progress_callback=chart_progress
            )

            print(f"  Charts: {chart_result.total_regenerated}/{chart_result.total_detected} "
                  f"regenerated from {chart_result.pages_with_charts} pages")

            if chart_result.total_regenerated > 0:
                print(f"  Chart SVGs saved to: {svg_dir}")

        # Image regeneration
        if image_regenerator and images_dir.exists():
            print("  Regenerating images (photos, illustrations)...")

            regen_dir = doc_output / "regenerated"

            def img_progress(current, total, result):
                if args.verbose:
                    status = "OK" if result.success else "FAILED"
                    print(f"    Image {current}/{total}: {status}")

            img_result = image_regenerator.process_extracted_images(
                images_dir,
                regen_dir,
                progress_callback=img_progress
            )

            print(f"  Images: {img_result.total_regenerated}/{img_result.total_detected} "
                  f"regenerated, {img_result.total_skipped} skipped")

            if img_result.total_regenerated > 0:
                print(f"  Regenerated images saved to: {regen_dir}")

        # Create structured output
        if doc_structurer:
            print("  Creating structured document output...")

            structured_output_path = doc_output / "document.json"
            structured_doc = doc_structurer.structure_document(
                doc_output,
                pdf_path,
                structured_output_path
            )

            summary = structured_doc.processing_summary
            print(f"  Structured output: {structured_output_path}")
            print(f"    - Pages: {summary.get('total_pages', 0)}")
            print(f"    - Words: {summary.get('total_words', 0)}")
            print(f"    - Tables: {summary.get('total_tables', 0)}")
            print(f"    - Charts: {summary.get('total_charts', 0)}")
            print(f"    - Sections: {summary.get('sections_detected', 0)}")

        # Export to Word
        if word_exporter:
            print("  Exporting to Word document...")
            if len(pdfs) == 1 and args.export_word:
                word_output = Path(args.export_word)
            else:
                word_output = doc_output / f"{pdf_path.stem}.docx"

            try:
                word_path = word_exporter.export(
                    doc_output,
                    word_output,
                    include_regenerated_images=True,
                    include_original_images=False
                )
                print(f"  Word document saved: {word_path}")
            except Exception as e:
                print(f"  Error exporting to Word: {e}")

        # Export to PDF
        if pdf_exporter:
            print("  Exporting to PDF document...")
            if len(pdfs) == 1 and args.export_pdf:
                pdf_output = Path(args.export_pdf)
            else:
                pdf_output = doc_output / f"{pdf_path.stem}_processed.pdf"

            try:
                pdf_path_out = pdf_exporter.export(
                    doc_output,
                    pdf_output,
                    include_regenerated_images=True,
                    include_original_images=False
                )
                print(f"  PDF document saved: {pdf_path_out}")
            except Exception as e:
                print(f"  Error exporting to PDF: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()

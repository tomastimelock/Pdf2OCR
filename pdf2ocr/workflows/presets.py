"""Preset workflows for common PDF2OCR tasks."""

from pdf2ocr.workflows.builder import WorkflowBuilder, Workflow


def basic_workflow() -> Workflow:
    """
    Basic workflow: Split → OCR → Export

    Minimal processing for quick text extraction.
    """
    return (WorkflowBuilder("basic")
        .describe("Basic PDF to text extraction")
        .split_pdf(dpi=150)
        .ocr(engine="auto")
        .structure_output()
        .export_word()
        .build()
    )


def standard_workflow() -> Workflow:
    """
    Standard workflow: Split → Enhance → OCR → Tables → Export

    Balanced processing with image enhancement and table extraction.
    """
    return (WorkflowBuilder("standard")
        .describe("Standard PDF processing with tables")
        .split_pdf(dpi=200)
        .extract_images()
        .enhance_images(preset="document", auto_enhance=True)
        .ocr(engine="auto")
        .extract_tables(save_svg=True)
        .structure_output()
        .export_word()
        .export_pdf()
        .build()
    )


def full_workflow() -> Workflow:
    """
    Full workflow: All 10 original pipeline steps

    Complete processing with charts and image regeneration.
    """
    return (WorkflowBuilder("full")
        .describe("Full PDF processing pipeline")
        .split_pdf(dpi=200)
        .extract_images()
        .enhance_images(preset="document")
        .ocr(engine="auto")
        .extract_tables(save_svg=True)
        .detect_charts()
        .regenerate_charts()
        .regenerate_images()
        .structure_output()
        .export_word()
        .export_pdf()
        .build()
    )


def tables_only_workflow() -> Workflow:
    """
    Tables-only workflow: Focus on table extraction.

    Optimized for documents with important tabular data.
    """
    return (WorkflowBuilder("tables_only")
        .describe("Extract tables from PDF")
        .split_pdf(dpi=200)
        .ocr(engine="auto")
        .extract_tables(save_svg=True)
        .structure_output()
        .export_word()
        .build()
    )


def text_only_workflow() -> Workflow:
    """
    Text-only workflow: Pure text extraction.

    Fast extraction without image processing or exports.
    """
    return (WorkflowBuilder("text_only")
        .describe("Extract text only")
        .split_pdf(dpi=150)
        .ocr(engine="auto")
        .structure_output()
        .build()
    )


def high_quality_workflow() -> Workflow:
    """
    High-quality workflow: Maximum quality processing.

    Uses LLM enhancement and high DPI for best results.
    """
    return (WorkflowBuilder("high_quality")
        .describe("High-quality PDF processing with LLM enhancement")
        .split_pdf(dpi=300)
        .extract_images()
        .enhance_images(preset="scan")
        .ocr(engine="mistral")
        .llm_enhance(provider="anthropic", correct_errors=True)
        .extract_tables(save_svg=True)
        .detect_charts()
        .regenerate_charts()
        .structure_output()
        .export_word()
        .export_pdf()
        .build()
    )


def scanned_document_workflow() -> Workflow:
    """
    Scanned document workflow: Optimized for scanned PDFs.

    Aggressive enhancement for low-quality scans.
    """
    return (WorkflowBuilder("scanned")
        .describe("Process scanned documents")
        .split_pdf(dpi=300)
        .enhance_images(preset="low_quality")
        .ocr(engine="auto", quality_threshold=0.5)
        .llm_enhance(correct_errors=True)
        .structure_output()
        .export_word()
        .build()
    )


# Registry of all preset workflows
PRESET_WORKFLOWS = {
    "basic": basic_workflow,
    "standard": standard_workflow,
    "full": full_workflow,
    "tables_only": tables_only_workflow,
    "text_only": text_only_workflow,
    "high_quality": high_quality_workflow,
    "scanned": scanned_document_workflow,
}


def get_preset(name: str) -> Workflow:
    """
    Get a preset workflow by name.

    Args:
        name: Preset name

    Returns:
        Workflow instance

    Raises:
        KeyError: If preset not found
    """
    if name not in PRESET_WORKFLOWS:
        available = ", ".join(PRESET_WORKFLOWS.keys())
        raise KeyError(f"Unknown preset: {name}. Available: {available}")
    return PRESET_WORKFLOWS[name]()


def list_presets() -> list[str]:
    """List all available preset names."""
    return list(PRESET_WORKFLOWS.keys())

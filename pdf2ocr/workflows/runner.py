"""Workflow Runner - Execute PDF2OCR workflows."""

import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, List, Union

from pdf2ocr.workflows.steps import (
    WorkflowStep, StepType, StepResult, ErrorAction
)
from pdf2ocr.workflows.builder import Workflow


@dataclass
class WorkflowResult:
    """Result of executing a workflow."""
    workflow_name: str
    source_path: Path
    output_dir: Path
    success: bool
    steps_completed: int
    steps_total: int
    step_results: List[StepResult] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: Optional[str] = None
    outputs: Dict[str, Any] = field(default_factory=dict)


class WorkflowRunner:
    """
    Execute PDF2OCR workflows.

    Runs workflow steps in sequence, handling errors and
    passing context between steps.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the workflow runner.

        Args:
            logger: Logger instance (creates default if not provided)
        """
        self.logger = logger or logging.getLogger(__name__)
        self._step_handlers: Dict[StepType, Callable] = {}
        self._register_handlers()

    def _register_handlers(self):
        """Register handlers for each step type."""
        self._step_handlers = {
            StepType.SPLIT_PDF: self._run_split_pdf,
            StepType.EXTRACT_IMAGES: self._run_extract_images,
            StepType.ENHANCE_IMAGES: self._run_enhance_images,
            StepType.OCR: self._run_ocr,
            StepType.LLM_ENHANCE: self._run_llm_enhance,
            StepType.EXTRACT_TABLES: self._run_extract_tables,
            StepType.DETECT_CHARTS: self._run_detect_charts,
            StepType.REGENERATE_CHARTS: self._run_regenerate_charts,
            StepType.REGENERATE_IMAGES: self._run_regenerate_images,
            StepType.STRUCTURE_OUTPUT: self._run_structure_output,
            StepType.EXPORT_WORD: self._run_export_word,
            StepType.EXPORT_PDF: self._run_export_pdf,
        }

    def run(
        self,
        workflow: Workflow,
        pdf_path: Union[str, Path],
        output_dir: Union[str, Path],
        progress_callback: Optional[Callable[[int, int, StepResult], None]] = None
    ) -> WorkflowResult:
        """
        Execute a workflow on a PDF document.

        Args:
            workflow: Workflow to execute
            pdf_path: Path to the input PDF
            output_dir: Directory for outputs
            progress_callback: Callback for progress updates

        Returns:
            WorkflowResult with execution details
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        step_results = []
        context = {
            "pdf_path": pdf_path,
            "output_dir": output_dir,
            "workflow": workflow,
            "pages": [],
            "ocr_result": None,
            "tables": [],
            "charts": [],
        }

        self.logger.info(f"Starting workflow '{workflow.name}' on {pdf_path.name}")

        steps_to_run = [s for s in workflow.steps if s.should_run(context)]
        total_steps = len(steps_to_run)

        for idx, step in enumerate(steps_to_run, 1):
            self.logger.info(f"Step {idx}/{total_steps}: {step.name}")

            result = self._execute_step(step, context)
            step_results.append(result)

            if progress_callback:
                progress_callback(idx, total_steps, result)

            if not result.success:
                if step.on_error == ErrorAction.STOP:
                    self.logger.error(f"Step failed, stopping workflow: {result.error}")
                    return WorkflowResult(
                        workflow_name=workflow.name,
                        source_path=pdf_path,
                        output_dir=output_dir,
                        success=False,
                        steps_completed=idx - 1,
                        steps_total=total_steps,
                        step_results=step_results,
                        duration_seconds=time.time() - start_time,
                        error=result.error,
                        outputs=context.get("outputs", {})
                    )
                elif step.on_error == ErrorAction.SKIP:
                    self.logger.warning(f"Step failed, skipping: {result.error}")
                    continue

        duration = time.time() - start_time
        self.logger.info(f"Workflow completed in {duration:.2f}s")

        return WorkflowResult(
            workflow_name=workflow.name,
            source_path=pdf_path,
            output_dir=output_dir,
            success=True,
            steps_completed=len(step_results),
            steps_total=total_steps,
            step_results=step_results,
            duration_seconds=duration,
            outputs=context.get("outputs", {})
        )

    def _execute_step(
        self,
        step: WorkflowStep,
        context: Dict[str, Any]
    ) -> StepResult:
        """Execute a single workflow step."""
        start_time = time.time()
        retries = 0

        while retries <= step.max_retries:
            try:
                handler = self._step_handlers.get(step.step_type)
                if handler is None:
                    raise ValueError(f"No handler for step type: {step.step_type}")

                output = handler(step, context)

                return StepResult(
                    step_id=step.step_id,
                    step_type=step.step_type,
                    success=True,
                    output=output,
                    duration_seconds=time.time() - start_time,
                    retries=retries
                )

            except Exception as e:
                retries += 1
                self.logger.warning(f"Step {step.step_id} failed (attempt {retries}): {e}")

                if retries > step.max_retries:
                    return StepResult(
                        step_id=step.step_id,
                        step_type=step.step_type,
                        success=False,
                        error=str(e),
                        duration_seconds=time.time() - start_time,
                        retries=retries
                    )

        return StepResult(
            step_id=step.step_id,
            step_type=step.step_type,
            success=False,
            error="Max retries exceeded",
            duration_seconds=time.time() - start_time,
            retries=retries
        )

    # Step handlers
    def _run_split_pdf(self, step: WorkflowStep, context: Dict) -> Any:
        """Execute PDF splitting step."""
        from pdf2ocr.processors import PDFSplitter

        config = step.config
        splitter = PDFSplitter(
            dpi=config.get("dpi", 200),
            image_format=config.get("image_format", "jpg")
        )

        output_subdir = config.get("output_subdir", "pages")
        pages_dir = context["output_dir"] / output_subdir

        pages = splitter.split_to_images(context["pdf_path"], pages_dir)
        context["pages"] = pages
        context["pages_dir"] = pages_dir

        return {"pages_count": len(pages), "pages_dir": str(pages_dir)}

    def _run_extract_images(self, step: WorkflowStep, context: Dict) -> Any:
        """Execute embedded image extraction step."""
        from pdf2ocr.processors import PDFSplitter

        config = step.config
        splitter = PDFSplitter()

        output_subdir = config.get("output_subdir", "images")
        images_dir = context["output_dir"] / output_subdir

        images = splitter.extract_images(
            context["pdf_path"],
            images_dir,
            min_size=config.get("min_size", 50)
        )
        context["extracted_images"] = images

        return {"images_count": len(images)}

    def _run_enhance_images(self, step: WorkflowStep, context: Dict) -> Any:
        """Execute image enhancement step."""
        from pdf2ocr.processors import ImageEnhancer

        config = step.config
        enhancer = ImageEnhancer(
            preset=config.get("preset", "document"),
            auto_enhance=config.get("auto_enhance", True)
        )

        pages_dir = context.get("pages_dir")
        if not pages_dir:
            return {"enhanced": 0, "skipped": "No pages to enhance"}

        enhanced_count = 0
        for page_info in context.get("pages", []):
            if page_info.image_path:
                result = enhancer.enhance_for_ocr(page_info.image_path)
                if result.success and result.operations_applied:
                    enhanced_count += 1

        return {"enhanced_count": enhanced_count}

    def _run_ocr(self, step: WorkflowStep, context: Dict) -> Any:
        """Execute OCR processing step."""
        from pdf2ocr.processors import OCRProcessor

        config = step.config
        processor = OCRProcessor(
            engine=config.get("engine", "auto"),
            quality_threshold=config.get("quality_threshold", 0.7)
        )

        output_subdir = config.get("output_subdir", "txt")
        txt_dir = context["output_dir"] / output_subdir

        result = processor.process_pdf(context["pdf_path"], txt_dir)
        context["ocr_result"] = result

        # Save combined text
        combined_path = context["output_dir"] / "combined.txt"
        processor.save_combined_text(result, combined_path)
        context["combined_text_path"] = combined_path

        return {
            "pages_processed": result.total_pages,
            "average_quality": result.average_quality
        }

    def _run_llm_enhance(self, step: WorkflowStep, context: Dict) -> Any:
        """Execute LLM text enhancement step."""
        from pdf2ocr.processors import LLMEnhancer

        config = step.config
        enhancer = LLMEnhancer(default_provider=config.get("provider", "anthropic"))

        ocr_result = context.get("ocr_result")
        if not ocr_result:
            return {"enhanced": False, "reason": "No OCR result"}

        enhanced_count = 0
        if config.get("correct_errors", True):
            for page_result in ocr_result.pages:
                if page_result.text:
                    enhanced = enhancer.correct_ocr_errors(page_result.text)
                    page_result.text = enhanced.enhanced_text
                    enhanced_count += 1

        return {"pages_enhanced": enhanced_count}

    def _run_extract_tables(self, step: WorkflowStep, context: Dict) -> Any:
        """Execute table extraction step."""
        from pdf2ocr.extractors import TableExtractor

        config = step.config
        extractor = TableExtractor()

        result = extractor.extract_and_save(
            context["pdf_path"],
            context["output_dir"],
            save_svg=config.get("save_svg", True)
        )
        context["tables"] = result

        return {"tables_count": result.total_tables}

    def _run_detect_charts(self, step: WorkflowStep, context: Dict) -> Any:
        """Execute chart detection step."""
        # Chart detection is handled by regenerate_charts
        return {"status": "Charts will be detected during regeneration"}

    def _run_regenerate_charts(self, step: WorkflowStep, context: Dict) -> Any:
        """Execute chart regeneration step."""
        from pdf2ocr.processors import ChartRegenerator
        import os

        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        if not openai_key or not anthropic_key:
            return {"regenerated": 0, "reason": "Missing API keys"}

        config = step.config
        regenerator = ChartRegenerator(
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key
        )

        pages_dir = context.get("pages_dir")
        if not pages_dir:
            return {"regenerated": 0, "reason": "No pages directory"}

        output_subdir = config.get("output_subdir", "svg")
        svg_dir = context["output_dir"] / output_subdir

        result = regenerator.process_document(pages_dir, svg_dir)
        context["charts"] = result

        return {"regenerated": result.total_regenerated}

    def _run_regenerate_images(self, step: WorkflowStep, context: Dict) -> Any:
        """Execute image regeneration step."""
        from pdf2ocr.processors import ImageRegenerator
        import os

        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            return {"regenerated": 0, "reason": "Missing OpenAI API key"}

        config = step.config
        regenerator = ImageRegenerator(api_key=openai_key)

        images_dir = context["output_dir"] / "images"
        if not images_dir.exists():
            return {"regenerated": 0, "reason": "No images to regenerate"}

        output_subdir = config.get("output_subdir", "regenerated")
        regen_dir = context["output_dir"] / output_subdir

        result = regenerator.process_extracted_images(images_dir, regen_dir)

        return {"regenerated": result.total_regenerated}

    def _run_structure_output(self, step: WorkflowStep, context: Dict) -> Any:
        """Execute structured output generation step."""
        from pdf2ocr.extractors import DocumentStructurer

        config = step.config
        structurer = DocumentStructurer()

        output_file = config.get("output_file", "document.json")
        output_path = context["output_dir"] / output_file

        doc = structurer.structure_document(
            context["output_dir"],
            context["pdf_path"],
            output_path
        )
        context["structured_doc"] = doc

        return {"output_path": str(output_path)}

    def _run_export_word(self, step: WorkflowStep, context: Dict) -> Any:
        """Execute Word export step."""
        from pdf2ocr.exporters import WordExporter

        config = step.config
        exporter = WordExporter()

        doc_name = context["pdf_path"].stem
        output_path = context["output_dir"] / f"{doc_name}.docx"

        path = exporter.export(
            context["output_dir"],
            output_path,
            include_regenerated_images=config.get("include_regenerated", True),
            include_original_images=config.get("include_original", False)
        )

        context.setdefault("outputs", {})["word"] = str(path)
        return {"output_path": str(path)}

    def _run_export_pdf(self, step: WorkflowStep, context: Dict) -> Any:
        """Execute PDF export step."""
        from pdf2ocr.exporters import PDFExporter

        config = step.config
        exporter = PDFExporter()

        doc_name = context["pdf_path"].stem
        output_path = context["output_dir"] / f"{doc_name}_processed.pdf"

        path = exporter.export(
            context["output_dir"],
            output_path,
            include_regenerated_images=config.get("include_regenerated", True),
            include_original_images=config.get("include_original", False)
        )

        context.setdefault("outputs", {})["pdf"] = str(path)
        return {"output_path": str(path)}

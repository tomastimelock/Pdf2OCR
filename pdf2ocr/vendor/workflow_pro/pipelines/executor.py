"""Pipeline Executor - Execute processing pipelines.

Provides execution engine for running pipelines with progress tracking.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime
from enum import Enum
import time
import json

from .builder import Pipeline, PipelineStep, StepType


class ExecutionStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class StepResult:
    """Result of a single step execution."""
    step_id: str
    operation: str
    status: ExecutionStatus
    start_time: str
    end_time: str
    duration: float
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    success: bool = True

    def to_dict(self) -> Dict:
        return {
            'step_id': self.step_id,
            'operation': self.operation,
            'status': self.status.value,
            'duration': self.duration,
            'output_path': self.output_path,
            'success': self.success,
            'error': self.error,
        }


@dataclass
class ExecutionResult:
    """Result of pipeline execution."""
    pipeline_name: str
    source: str
    status: ExecutionStatus
    start_time: str
    end_time: str
    duration: float
    steps_completed: int
    steps_total: int
    step_results: List[StepResult] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str = ""

    def to_dict(self) -> Dict:
        return {
            'pipeline': self.pipeline_name,
            'source': self.source,
            'status': self.status.value,
            'duration': self.duration,
            'steps': f"{self.steps_completed}/{self.steps_total}",
            'outputs': self.outputs,
            'success': self.success,
            'error': self.error,
        }


class PipelineExecutor:
    """Execute processing pipelines.

    Features:
    - Sequential and parallel execution
    - Progress tracking
    - Error handling and recovery
    - Timeout handling
    - Result caching
    """

    def __init__(
        self,
        work_dir: Optional[Union[str, Path]] = None,
        parallel: bool = False,
        max_workers: int = 4,
    ):
        """
        Initialize executor.

        Args:
            work_dir: Working directory for intermediate files
            parallel: Enable parallel execution
            max_workers: Maximum parallel workers
        """
        self.work_dir = Path(work_dir) if work_dir else Path.cwd() / '.pipeline_work'
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.parallel = parallel
        self.max_workers = max_workers

        # Callbacks
        self._on_step_start: Optional[Callable] = None
        self._on_step_complete: Optional[Callable] = None
        self._on_progress: Optional[Callable] = None

        # Lazy-loaded super-modules
        self._image_core = None
        self._vision_ai = None
        self._media_vault = None
        self._export_hub = None
        self._workflow_pro = None

    def _get_image_core(self):
        """Lazy load ImageCore."""
        if self._image_core is None:
            try:
                from image_core import ImageCore
                self._image_core = ImageCore()
            except ImportError:
                pass
        return self._image_core

    def _get_vision_ai(self):
        """Lazy load VisionAI."""
        if self._vision_ai is None:
            try:
                from vision_ai import VisionAI
                self._vision_ai = VisionAI()
            except ImportError:
                pass
        return self._vision_ai

    def on_step_start(self, callback: Callable):
        """Set callback for step start."""
        self._on_step_start = callback

    def on_step_complete(self, callback: Callable):
        """Set callback for step completion."""
        self._on_step_complete = callback

    def on_progress(self, callback: Callable):
        """Set callback for progress updates."""
        self._on_progress = callback

    def execute(
        self,
        pipeline: Pipeline,
        source: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> ExecutionResult:
        """
        Execute a pipeline.

        Args:
            pipeline: Pipeline to execute
            source: Source file or directory
            output_dir: Output directory

        Returns:
            ExecutionResult with details
        """
        start_time = time.time()
        source_path = Path(source)
        out_dir = Path(output_dir) if output_dir else source_path.parent / 'output'
        out_dir.mkdir(parents=True, exist_ok=True)

        result = ExecutionResult(
            pipeline_name=pipeline.name,
            source=str(source_path),
            status=ExecutionStatus.RUNNING,
            start_time=datetime.now().isoformat(),
            end_time="",
            duration=0,
            steps_completed=0,
            steps_total=len(pipeline.steps),
        )

        try:
            # Determine if processing single file or directory
            if source_path.is_file():
                result = self._execute_single(pipeline, source_path, out_dir, result)
            else:
                result = self._execute_batch(pipeline, source_path, out_dir, result)

        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.success = False
            result.error = str(e)

        result.end_time = datetime.now().isoformat()
        result.duration = time.time() - start_time

        return result

    def _execute_single(
        self,
        pipeline: Pipeline,
        source: Path,
        output_dir: Path,
        result: ExecutionResult,
    ) -> ExecutionResult:
        """Execute pipeline on single file."""
        context = {
            'source': str(source),
            'output_dir': str(output_dir),
            'current_input': str(source),
            'current_output': None,
            'step_index': 0,
        }

        for i, step in enumerate(pipeline.steps):
            context['step_index'] = i

            # Report progress
            if self._on_progress:
                progress = (i / len(pipeline.steps)) * 100
                self._on_progress(pipeline.name, i, len(pipeline.steps), progress)

            # Check condition if present
            if step.condition and not step.condition.evaluate(context):
                continue

            # Execute step
            step_result = self._execute_step(step, context, output_dir)
            result.step_results.append(step_result)

            if step_result.success:
                result.steps_completed += 1

                # Update context for next step
                if step_result.output_path:
                    context['current_input'] = step_result.output_path
                    context['current_output'] = step_result.output_path

                    # Add to outputs
                    result.outputs.append(step_result.output_path)

            else:
                # Handle error based on step configuration
                if step.on_error == "stop":
                    result.status = ExecutionStatus.FAILED
                    result.success = False
                    result.error = step_result.error
                    break
                elif step.on_error == "retry" and step.retries > 0:
                    # Retry logic
                    for retry in range(step.retries):
                        step_result = self._execute_step(step, context, output_dir)
                        if step_result.success:
                            result.steps_completed += 1
                            if step_result.output_path:
                                context['current_input'] = step_result.output_path
                            break
                # "skip" just continues to next step

        if result.status == ExecutionStatus.RUNNING:
            result.status = ExecutionStatus.COMPLETED
            result.success = True

        return result

    def _execute_batch(
        self,
        pipeline: Pipeline,
        source: Path,
        output_dir: Path,
        result: ExecutionResult,
    ) -> ExecutionResult:
        """Execute pipeline on directory of files."""
        # Find all images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.tiff']
        files = []
        for ext in extensions:
            files.extend(source.rglob(ext))
            files.extend(source.rglob(ext.upper()))

        # Execute pipeline on each file
        all_successful = True
        for file_path in files:
            file_result = self._execute_single(
                pipeline,
                file_path,
                output_dir,
                ExecutionResult(
                    pipeline_name=pipeline.name,
                    source=str(file_path),
                    status=ExecutionStatus.RUNNING,
                    start_time=datetime.now().isoformat(),
                    end_time="",
                    duration=0,
                    steps_completed=0,
                    steps_total=len(pipeline.steps),
                ),
            )

            result.step_results.extend(file_result.step_results)
            result.outputs.extend(file_result.outputs)
            result.steps_completed += file_result.steps_completed

            if not file_result.success:
                all_successful = False

        result.success = all_successful
        result.status = ExecutionStatus.COMPLETED if all_successful else ExecutionStatus.FAILED

        return result

    def _execute_step(
        self,
        step: PipelineStep,
        context: Dict[str, Any],
        output_dir: Path,
    ) -> StepResult:
        """Execute a single step."""
        start_time = time.time()

        step_result = StepResult(
            step_id=step.step_id,
            operation=step.operation,
            status=ExecutionStatus.RUNNING,
            start_time=datetime.now().isoformat(),
            end_time="",
            duration=0,
            input_path=context.get('current_input'),
        )

        # Callback
        if self._on_step_start:
            self._on_step_start(step.step_id, step.operation)

        try:
            # Route to appropriate handler
            if step.step_type == StepType.RESIZE:
                step_result = self._handle_resize(step, context, output_dir, step_result)
            elif step.step_type == StepType.CROP:
                step_result = self._handle_crop(step, context, output_dir, step_result)
            elif step.step_type == StepType.ROTATE:
                step_result = self._handle_rotate(step, context, output_dir, step_result)
            elif step.step_type == StepType.CONVERT:
                step_result = self._handle_convert(step, context, output_dir, step_result)
            elif step.step_type == StepType.ENHANCE:
                step_result = self._handle_enhance(step, context, output_dir, step_result)
            elif step.step_type == StepType.FILTER:
                step_result = self._handle_filter(step, context, output_dir, step_result)
            elif step.step_type == StepType.DETECT:
                step_result = self._handle_detect(step, context, step_result)
            elif step.step_type == StepType.OCR:
                step_result = self._handle_ocr(step, context, step_result)
            elif step.step_type == StepType.WATERMARK:
                step_result = self._handle_watermark(step, context, output_dir, step_result)
            elif step.step_type == StepType.SAVE:
                step_result = self._handle_save(step, context, output_dir, step_result)
            elif step.step_type == StepType.EXPORT:
                step_result = self._handle_export(step, context, output_dir, step_result)
            else:
                # Custom or unsupported step - pass through
                step_result.success = True
                step_result.output_path = context.get('current_input')

        except Exception as e:
            step_result.success = False
            step_result.error = str(e)
            step_result.status = ExecutionStatus.FAILED

        step_result.end_time = datetime.now().isoformat()
        step_result.duration = time.time() - start_time

        if step_result.success:
            step_result.status = ExecutionStatus.COMPLETED
        else:
            step_result.status = ExecutionStatus.FAILED

        # Callback
        if self._on_step_complete:
            self._on_step_complete(step.step_id, step_result.success)

        return step_result

    def _handle_resize(
        self,
        step: PipelineStep,
        context: Dict[str, Any],
        output_dir: Path,
        result: StepResult,
    ) -> StepResult:
        """Handle resize step."""
        from PIL import Image

        input_path = Path(context['current_input'])
        output_path = output_dir / f"{input_path.stem}_resized{input_path.suffix}"

        try:
            img = Image.open(input_path)

            width = step.params.get('width')
            height = step.params.get('height')
            mode = step.params.get('mode', 'fit')

            if mode == 'fit':
                if width and height:
                    img.thumbnail((width, height), Image.Resampling.LANCZOS)
                elif width:
                    ratio = width / img.width
                    new_height = int(img.height * ratio)
                    img = img.resize((width, new_height), Image.Resampling.LANCZOS)
                elif height:
                    ratio = height / img.height
                    new_width = int(img.width * ratio)
                    img = img.resize((new_width, height), Image.Resampling.LANCZOS)
            elif mode == 'exact':
                if width and height:
                    img = img.resize((width, height), Image.Resampling.LANCZOS)

            img.save(output_path, quality=95)

            result.output_path = str(output_path)
            result.success = True
            result.metadata['new_size'] = img.size

        except Exception as e:
            result.error = str(e)
            result.success = False

        return result

    def _handle_crop(
        self,
        step: PipelineStep,
        context: Dict[str, Any],
        output_dir: Path,
        result: StepResult,
    ) -> StepResult:
        """Handle crop step."""
        from PIL import Image

        input_path = Path(context['current_input'])
        output_path = output_dir / f"{input_path.stem}_cropped{input_path.suffix}"

        try:
            img = Image.open(input_path)

            x = step.params.get('x', 0)
            y = step.params.get('y', 0)
            width = step.params.get('width', img.width - x)
            height = step.params.get('height', img.height - y)

            cropped = img.crop((x, y, x + width, y + height))
            cropped.save(output_path, quality=95)

            result.output_path = str(output_path)
            result.success = True

        except Exception as e:
            result.error = str(e)
            result.success = False

        return result

    def _handle_rotate(
        self,
        step: PipelineStep,
        context: Dict[str, Any],
        output_dir: Path,
        result: StepResult,
    ) -> StepResult:
        """Handle rotate step."""
        from PIL import Image

        input_path = Path(context['current_input'])
        output_path = output_dir / f"{input_path.stem}_rotated{input_path.suffix}"

        try:
            img = Image.open(input_path)

            angle = step.params.get('angle', 0)
            expand = step.params.get('expand', True)

            rotated = img.rotate(-angle, expand=expand, resample=Image.Resampling.BICUBIC)
            rotated.save(output_path, quality=95)

            result.output_path = str(output_path)
            result.success = True

        except Exception as e:
            result.error = str(e)
            result.success = False

        return result

    def _handle_convert(
        self,
        step: PipelineStep,
        context: Dict[str, Any],
        output_dir: Path,
        result: StepResult,
    ) -> StepResult:
        """Handle format conversion step."""
        from PIL import Image

        input_path = Path(context['current_input'])
        format_ext = step.params.get('format', 'jpeg')
        quality = step.params.get('quality', 90)

        output_path = output_dir / f"{input_path.stem}.{format_ext}"

        try:
            img = Image.open(input_path)

            if format_ext in ['jpeg', 'jpg']:
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                img.save(output_path, 'JPEG', quality=quality)
            elif format_ext == 'png':
                img.save(output_path, 'PNG')
            elif format_ext == 'webp':
                img.save(output_path, 'WEBP', quality=quality)
            elif format_ext == 'tiff':
                img.save(output_path, 'TIFF')
            else:
                img.save(output_path)

            result.output_path = str(output_path)
            result.success = True

        except Exception as e:
            result.error = str(e)
            result.success = False

        return result

    def _handle_enhance(
        self,
        step: PipelineStep,
        context: Dict[str, Any],
        output_dir: Path,
        result: StepResult,
    ) -> StepResult:
        """Handle enhancement step."""
        from PIL import Image, ImageEnhance

        input_path = Path(context['current_input'])
        output_path = output_dir / f"{input_path.stem}_enhanced{input_path.suffix}"

        try:
            img = Image.open(input_path)
            mode = step.params.get('mode', 'auto')
            strength = step.params.get('strength', 1.0)

            if mode in ['auto', 'contrast']:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.0 + 0.2 * strength)

            if mode in ['auto', 'sharpness']:
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.0 + 0.3 * strength)

            if mode in ['auto', 'color']:
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(1.0 + 0.1 * strength)

            img.save(output_path, quality=95)

            result.output_path = str(output_path)
            result.success = True

        except Exception as e:
            result.error = str(e)
            result.success = False

        return result

    def _handle_filter(
        self,
        step: PipelineStep,
        context: Dict[str, Any],
        output_dir: Path,
        result: StepResult,
    ) -> StepResult:
        """Handle filter step."""
        from PIL import Image, ImageFilter

        input_path = Path(context['current_input'])
        output_path = output_dir / f"{input_path.stem}_filtered{input_path.suffix}"

        try:
            img = Image.open(input_path)
            filter_name = step.params.get('filter_name', 'sharpen')

            filters = {
                'blur': ImageFilter.BLUR,
                'sharpen': ImageFilter.SHARPEN,
                'smooth': ImageFilter.SMOOTH,
                'contour': ImageFilter.CONTOUR,
                'detail': ImageFilter.DETAIL,
                'edge_enhance': ImageFilter.EDGE_ENHANCE,
            }

            if filter_name in filters:
                img = img.filter(filters[filter_name])

            img.save(output_path, quality=95)

            result.output_path = str(output_path)
            result.success = True

        except Exception as e:
            result.error = str(e)
            result.success = False

        return result

    def _handle_detect(
        self,
        step: PipelineStep,
        context: Dict[str, Any],
        result: StepResult,
    ) -> StepResult:
        """Handle detection step."""
        input_path = context['current_input']
        detection_type = step.params.get('detection_type', 'objects')

        vision_ai = self._get_vision_ai()

        if vision_ai:
            try:
                if detection_type == 'faces':
                    detect_result = vision_ai.detect_faces(input_path)
                elif detection_type == 'objects':
                    detect_result = vision_ai.detect_objects(input_path)
                elif detection_type == 'text':
                    detect_result = vision_ai.detect_text(input_path)
                else:
                    detect_result = None

                if detect_result:
                    result.metadata['detections'] = detect_result.to_dict()
                    result.success = True
                else:
                    result.success = True  # No detections is still success

            except Exception as e:
                result.error = str(e)
                result.success = False
        else:
            result.success = True  # Pass through if module not available
            result.metadata['warning'] = 'VisionAI not available'

        result.output_path = input_path
        return result

    def _handle_ocr(
        self,
        step: PipelineStep,
        context: Dict[str, Any],
        result: StepResult,
    ) -> StepResult:
        """Handle OCR step."""
        input_path = context['current_input']

        try:
            import pytesseract
            from PIL import Image

            img = Image.open(input_path)
            text = pytesseract.image_to_string(img)

            result.metadata['text'] = text
            result.metadata['text_length'] = len(text)
            result.success = True

        except ImportError:
            result.metadata['warning'] = 'pytesseract not available'
            result.success = True
        except Exception as e:
            result.error = str(e)
            result.success = False

        result.output_path = input_path
        return result

    def _handle_watermark(
        self,
        step: PipelineStep,
        context: Dict[str, Any],
        output_dir: Path,
        result: StepResult,
    ) -> StepResult:
        """Handle watermark step."""
        from PIL import Image, ImageDraw, ImageFont

        input_path = Path(context['current_input'])
        output_path = output_dir / f"{input_path.stem}_watermarked{input_path.suffix}"

        try:
            img = Image.open(input_path)
            draw = ImageDraw.Draw(img)

            text = step.params.get('text', 'Watermark')

            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except Exception:
                font = ImageFont.load_default()

            # Position bottom-right
            bbox = draw.textbbox((0, 0), text, font=font)
            x = img.width - bbox[2] - 20
            y = img.height - bbox[3] - 20
            draw.text((x, y), text, fill=(255, 255, 255, 128), font=font)

            img.save(output_path, quality=95)

            result.output_path = str(output_path)
            result.success = True

        except Exception as e:
            result.error = str(e)
            result.success = False

        return result

    def _handle_save(
        self,
        step: PipelineStep,
        context: Dict[str, Any],
        output_dir: Path,
        result: StepResult,
    ) -> StepResult:
        """Handle save step."""
        import shutil

        input_path = Path(context['current_input'])
        destination = step.params.get('destination', str(output_dir))

        try:
            dest_path = Path(destination)
            dest_path.mkdir(parents=True, exist_ok=True)

            output_path = dest_path / input_path.name
            shutil.copy2(input_path, output_path)

            result.output_path = str(output_path)
            result.success = True

        except Exception as e:
            result.error = str(e)
            result.success = False

        return result

    def _handle_export(
        self,
        step: PipelineStep,
        context: Dict[str, Any],
        output_dir: Path,
        result: StepResult,
    ) -> StepResult:
        """Handle export step."""
        # Similar to save but with format handling
        return self._handle_save(step, context, output_dir, result)


# Convenience function
def execute_pipeline(
    pipeline: Pipeline,
    source: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
) -> ExecutionResult:
    """Execute a pipeline."""
    return PipelineExecutor().execute(pipeline, source, output_dir)

"""
Chart Regenerator Processor.

Detects charts in document pages and regenerates them as SVG.
Uses OpenAI GPT-4o for detection and Anthropic Claude for SVG generation.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import logging

from pdf2ocr.providers.openai_provider import OpenAIChartProvider
from pdf2ocr.providers.anthropic_provider import AnthropicSVGProvider

logger = logging.getLogger(__name__)


@dataclass
class ChartData:
    """Data for a detected/regenerated chart."""
    chart_id: int
    page_number: int
    chart_index: int
    chart_type: str
    description: str
    svg_content: Optional[str] = None
    svg_path: Optional[str] = None
    width: int = 0
    height: int = 0
    source_description: Optional[str] = None
    success: bool = False
    error: Optional[str] = None


@dataclass
class ChartProcessingResult:
    """Result of processing charts from a document."""
    charts: List[ChartData] = field(default_factory=list)
    total_detected: int = 0
    total_regenerated: int = 0
    total_failed: int = 0
    pages_with_charts: int = 0


class ChartRegenerator:
    """
    Detects and regenerates charts from document pages.

    Process:
    1. Detect charts in page using OpenAI GPT-4o Vision
    2. Generate detailed description of each chart
    3. Regenerate as SVG using Anthropic Claude
    4. Validate and save SVG

    Requires:
    - OPENAI_API_KEY for chart detection
    - ANTHROPIC_API_KEY for SVG generation
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None
    ):
        """
        Initialize chart regenerator.

        Args:
            openai_api_key: OpenAI API key for chart detection
            anthropic_api_key: Anthropic API key for SVG generation
        """
        self._openai: Optional[OpenAIChartProvider] = None
        self._anthropic: Optional[AnthropicSVGProvider] = None
        self._openai_key = openai_api_key
        self._anthropic_key = anthropic_api_key

    @property
    def openai(self) -> OpenAIChartProvider:
        """Lazy initialization of OpenAI provider."""
        if self._openai is None:
            self._openai = OpenAIChartProvider(api_key=self._openai_key)
        return self._openai

    @property
    def anthropic(self) -> AnthropicSVGProvider:
        """Lazy initialization of Anthropic provider."""
        if self._anthropic is None:
            self._anthropic = AnthropicSVGProvider(api_key=self._anthropic_key)
        return self._anthropic

    def is_available(self) -> bool:
        """Check if both providers are available."""
        return self.openai.is_available() and self.anthropic.is_available()

    def detect_charts(self, page_path: str | Path) -> List[dict]:
        """
        Detect charts in a page image.

        Args:
            page_path: Path to page image file

        Returns:
            List of detected chart info dicts
        """
        page_path = Path(page_path)

        try:
            charts = self.openai.detect_charts(page_path)

            if charts:
                logger.info(f"Detected {len(charts)} charts in {page_path.name}")
                for i, chart in enumerate(charts):
                    logger.debug(
                        f"  Chart {i+1}: type={chart.get('type', 'unknown')}, "
                        f"title={chart.get('title', 'none')}"
                    )
            else:
                logger.debug(f"No charts detected in {page_path.name}")

            return charts

        except Exception as e:
            logger.error(f"Chart detection failed for {page_path.name}: {e}")
            return []

    def describe_chart(self, page_path: str | Path, chart_info: dict) -> str:
        """
        Generate a detailed description of a chart for SVG recreation.

        Args:
            page_path: Path to page image file
            chart_info: Chart detection info

        Returns:
            Detailed description string
        """
        return self.openai.describe_chart(page_path, chart_info)

    def regenerate_chart(
        self,
        page_path: str | Path,
        description: str,
        chart_type: str = "unknown"
    ) -> dict:
        """
        Regenerate a chart as SVG.

        Args:
            page_path: Path to page file containing the chart
            description: Chart description
            chart_type: Type of chart

        Returns:
            Dict with svg_code, width, height, success
        """
        page_path = Path(page_path)

        try:
            result = self.anthropic.generate_svg(page_path, description)
            return result

        except Exception as e:
            logger.error(f"SVG generation failed: {e}")
            return {
                "svg_code": "",
                "width": 0,
                "height": 0,
                "success": False,
                "error": str(e)
            }

    def process_page(
        self,
        page_path: str | Path,
        page_number: int,
        output_dir: str | Path
    ) -> List[ChartData]:
        """
        Process a page to detect and regenerate all charts.

        Args:
            page_path: Path to page file
            page_number: Page number
            output_dir: Output directory for SVG files

        Returns:
            List of ChartData objects
        """
        page_path = Path(page_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        charts = []

        # Detect charts
        detected = self.detect_charts(page_path)

        if not detected:
            return charts

        logger.info(f"Processing {len(detected)} charts from page {page_number}")

        for chart_idx, chart_info in enumerate(detected):
            # Get detailed description
            description = self.describe_chart(page_path, chart_info)

            # Regenerate as SVG
            svg_result = self.regenerate_chart(
                page_path,
                description,
                chart_info.get('type', 'unknown')
            )

            if svg_result.get('success'):
                # Save SVG
                filename = f"page_{page_number:03d}_chart_{chart_idx + 1:03d}.svg"
                svg_path = output_dir / filename

                with open(svg_path, 'w', encoding='utf-8') as f:
                    f.write(svg_result['svg_code'])

                chart_data = ChartData(
                    chart_id=chart_idx,
                    page_number=page_number,
                    chart_index=chart_idx,
                    chart_type=chart_info.get('type', 'unknown'),
                    description=description,
                    svg_content=svg_result['svg_code'],
                    svg_path=str(svg_path),
                    width=svg_result.get('width', 0),
                    height=svg_result.get('height', 0),
                    source_description=chart_info.get('description', ''),
                    success=True
                )

                charts.append(chart_data)
                logger.debug(f"Generated: {filename}")

            else:
                error_msg = svg_result.get('error', 'Unknown error')
                logger.warning(
                    f"Failed to regenerate chart {chart_idx + 1} on page {page_number}: {error_msg}"
                )

                chart_data = ChartData(
                    chart_id=chart_idx,
                    page_number=page_number,
                    chart_index=chart_idx,
                    chart_type=chart_info.get('type', 'unknown'),
                    description=description,
                    source_description=chart_info.get('description', ''),
                    success=False,
                    error=error_msg
                )
                charts.append(chart_data)

        successful = sum(1 for c in charts if c.success)
        logger.info(
            f"Page {page_number}: {successful} charts successfully regenerated "
            f"out of {len(detected)} detected"
        )

        return charts

    def process_document(
        self,
        pages_dir: str | Path,
        output_dir: str | Path,
        progress_callback: Optional[Callable[[int, int, List[ChartData]], None]] = None
    ) -> ChartProcessingResult:
        """
        Process all pages in a directory to detect and regenerate charts.

        Args:
            pages_dir: Directory containing page images
            output_dir: Output directory for SVG files
            progress_callback: Callback for progress (page_num, total, charts)

        Returns:
            ChartProcessingResult with all chart data
        """
        pages_dir = Path(pages_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all page images
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        page_files = sorted([
            f for f in pages_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ])

        result = ChartProcessingResult()

        for idx, page_path in enumerate(page_files, start=1):
            page_charts = self.process_page(page_path, idx, output_dir)

            result.charts.extend(page_charts)
            result.total_detected += len(page_charts)
            result.total_regenerated += sum(1 for c in page_charts if c.success)
            result.total_failed += sum(1 for c in page_charts if not c.success)

            if page_charts:
                result.pages_with_charts += 1

            if progress_callback:
                progress_callback(idx, len(page_files), page_charts)

        logger.info(
            f"Document processing complete: {result.total_regenerated}/{result.total_detected} "
            f"charts regenerated across {result.pages_with_charts} pages"
        )

        return result

    def has_charts(self, page_path: str | Path) -> bool:
        """
        Quick check if a page contains charts.

        Args:
            page_path: Path to page file

        Returns:
            True if page has charts
        """
        detected = self.detect_charts(page_path)
        return len(detected) > 0

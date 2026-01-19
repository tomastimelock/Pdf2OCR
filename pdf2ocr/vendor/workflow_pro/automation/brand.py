"""Brand Management Automation.

Provides automated brand compliance and asset management.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime
from enum import Enum


class ComplianceStatus(Enum):
    """Brand compliance status."""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    UNKNOWN = "unknown"


class AssetType(Enum):
    """Brand asset type."""
    LOGO = "logo"
    ICON = "icon"
    PHOTO = "photo"
    ILLUSTRATION = "illustration"
    SOCIAL_POST = "social_post"
    ADVERTISEMENT = "advertisement"


@dataclass
class BrandGuidelines:
    """Brand guidelines specification."""
    name: str

    # Colors
    primary_colors: List[Tuple[int, int, int]] = field(default_factory=list)
    secondary_colors: List[Tuple[int, int, int]] = field(default_factory=list)
    forbidden_colors: List[Tuple[int, int, int]] = field(default_factory=list)
    color_tolerance: int = 30

    # Logo
    logo_files: List[str] = field(default_factory=list)
    min_logo_size: Tuple[int, int] = (50, 50)

    # Requirements
    min_resolution: Tuple[int, int] = (800, 600)
    max_file_size: int = 10_000_000
    allowed_formats: List[str] = field(default_factory=lambda: ['jpg', 'png', 'webp'])

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'primary_colors': self.primary_colors,
            'secondary_colors': self.secondary_colors,
            'min_resolution': self.min_resolution,
            'allowed_formats': self.allowed_formats,
        }


@dataclass
class ComplianceResult:
    """Brand compliance check result."""
    path: str
    status: ComplianceStatus
    score: float
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    logo_detected: bool = False
    colors_compliant: bool = True
    resolution_ok: bool = True

    def to_dict(self) -> Dict:
        return {
            'path': self.path,
            'status': self.status.value,
            'score': self.score,
            'issues': self.issues,
            'warnings': self.warnings,
            'logo_detected': self.logo_detected,
        }


@dataclass
class BatchComplianceResult:
    """Batch compliance check result."""
    total: int
    compliant: int
    warnings: int
    violations: int
    top_issues: List[str] = field(default_factory=list)
    success: bool = True

    def to_dict(self) -> Dict:
        return {
            'total': self.total,
            'compliant': self.compliant,
            'warnings': self.warnings,
            'violations': self.violations,
            'top_issues': self.top_issues,
        }


@dataclass
class BrandAsset:
    """A brand asset in the library."""
    asset_id: str
    path: str
    asset_type: AssetType
    name: str
    tags: List[str]
    approved: bool
    created_at: str

    def to_dict(self) -> Dict:
        return {
            'asset_id': self.asset_id,
            'path': self.path,
            'type': self.asset_type.value,
            'name': self.name,
            'tags': self.tags,
            'approved': self.approved,
        }


class BrandManagement:
    """Brand asset management and compliance.

    Features:
    - Brand guidelines enforcement
    - Logo detection and validation
    - Color palette compliance
    - Asset library management
    - Usage monitoring
    - Compliance reporting
    """

    def __init__(
        self,
        guidelines: Optional[BrandGuidelines] = None,
        database_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize brand management.

        Args:
            guidelines: Brand guidelines to enforce
            database_path: Path to asset database
        """
        self.guidelines = guidelines or BrandGuidelines(name="Default")
        self.database_path = Path(database_path) if database_path else None

        # Asset library
        self.assets: Dict[str, BrandAsset] = {}

        # Authorized domains
        self.authorized_domains: List[str] = []

        # Callbacks
        self._on_violation: Optional[Callable] = None

        # Lazy-loaded modules
        self._brand_guardian = None

    def _get_brand_guardian(self):
        """Lazy load BrandGuardian."""
        if self._brand_guardian is None:
            try:
                from brand_guardian import BrandGuardian, BrandGuidelines as BG

                bg = BG(
                    name=self.guidelines.name,
                    primary_colors=self.guidelines.primary_colors,
                    secondary_colors=self.guidelines.secondary_colors,
                    forbidden_colors=self.guidelines.forbidden_colors,
                    color_tolerance=self.guidelines.color_tolerance,
                    logo_files=self.guidelines.logo_files,
                    min_resolution=self.guidelines.min_resolution,
                    max_file_size=self.guidelines.max_file_size,
                    allowed_formats=self.guidelines.allowed_formats,
                )

                self._brand_guardian = BrandGuardian(
                    database_path=self.database_path,
                    guidelines=bg,
                )

            except ImportError:
                pass
        return self._brand_guardian

    def on_violation(self, callback: Callable):
        """Set callback for violations."""
        self._on_violation = callback

    def set_guidelines(self, guidelines: BrandGuidelines):
        """Update brand guidelines."""
        self.guidelines = guidelines
        self._brand_guardian = None  # Reset

    def add_authorized_domain(self, domain: str):
        """Add authorized domain for brand usage."""
        self.authorized_domains.append(domain.lower())

    def check_compliance(
        self,
        image: Union[str, Path],
        strict: bool = False,
    ) -> ComplianceResult:
        """
        Check image for brand compliance.

        Args:
            image: Image path
            strict: Enable strict mode

        Returns:
            ComplianceResult with details
        """
        path = Path(image)

        result = ComplianceResult(
            path=str(path),
            status=ComplianceStatus.UNKNOWN,
            score=1.0,
        )

        guardian = self._get_brand_guardian()

        if guardian:
            try:
                comp_result = guardian.check_compliance(path, strict)

                try:
                    result.status = ComplianceStatus(comp_result.status.value)
                except ValueError:
                    result.status = ComplianceStatus.UNKNOWN

                result.score = comp_result.score
                result.issues = comp_result.issues
                result.warnings = comp_result.warnings
                result.logo_detected = comp_result.logo_detected
                result.colors_compliant = comp_result.colors_compliant
                result.resolution_ok = comp_result.resolution_ok

                # Callback for violations
                if result.status == ComplianceStatus.VIOLATION and self._on_violation:
                    self._on_violation(result)

            except Exception as e:
                result.issues.append(str(e))
        else:
            # Fallback: basic checks
            result = self._basic_compliance_check(path)

        return result

    def _basic_compliance_check(self, path: Path) -> ComplianceResult:
        """Basic compliance check without full module."""
        from PIL import Image

        result = ComplianceResult(
            path=str(path),
            status=ComplianceStatus.COMPLIANT,
            score=1.0,
        )

        try:
            img = Image.open(path)

            # Check format
            ext = path.suffix.lower().strip('.')
            if ext not in self.guidelines.allowed_formats:
                result.issues.append(f"Format '{ext}' not allowed")
                result.score -= 0.2

            # Check resolution
            if (img.width < self.guidelines.min_resolution[0] or
                img.height < self.guidelines.min_resolution[1]):
                result.issues.append(f"Resolution below minimum")
                result.resolution_ok = False
                result.score -= 0.2

            # Check file size
            file_size = path.stat().st_size
            if file_size > self.guidelines.max_file_size:
                result.warnings.append(f"File size exceeds limit")
                result.score -= 0.1

            # Determine status
            if result.score >= 0.9:
                result.status = ComplianceStatus.COMPLIANT
            elif result.score >= 0.6:
                result.status = ComplianceStatus.WARNING
            else:
                result.status = ComplianceStatus.VIOLATION

        except Exception as e:
            result.status = ComplianceStatus.UNKNOWN
            result.issues.append(str(e))

        return result

    def check_batch(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
    ) -> BatchComplianceResult:
        """
        Check compliance for all images in directory.

        Args:
            directory: Directory to check
            recursive: Search subdirectories

        Returns:
            BatchComplianceResult with statistics
        """
        source = Path(directory)

        result = BatchComplianceResult(
            total=0,
            compliant=0,
            warnings=0,
            violations=0,
        )

        # Track issues
        issue_counts: Dict[str, int] = {}

        # Find images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
        images = []
        for ext in extensions:
            if recursive:
                images.extend(source.rglob(ext))
            else:
                images.extend(source.glob(ext))

        result.total = len(images)

        for img_path in images:
            comp_result = self.check_compliance(img_path)

            if comp_result.status == ComplianceStatus.COMPLIANT:
                result.compliant += 1
            elif comp_result.status == ComplianceStatus.WARNING:
                result.warnings += 1
            else:
                result.violations += 1

            # Count issues
            for issue in comp_result.issues:
                if issue not in issue_counts:
                    issue_counts[issue] = 0
                issue_counts[issue] += 1

        # Get top issues
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        result.top_issues = [issue for issue, count in sorted_issues[:5]]
        result.success = result.violations == 0

        return result

    def add_asset(
        self,
        image: Union[str, Path],
        asset_type: AssetType,
        name: str,
        tags: Optional[List[str]] = None,
        approved: bool = False,
    ) -> BrandAsset:
        """
        Add asset to brand library.

        Args:
            image: Image path
            asset_type: Type of asset
            name: Asset name
            tags: Tags for the asset
            approved: Approval status

        Returns:
            Created BrandAsset
        """
        import uuid

        path = Path(image)
        asset_id = str(uuid.uuid4())[:8]

        asset = BrandAsset(
            asset_id=asset_id,
            path=str(path),
            asset_type=asset_type,
            name=name,
            tags=tags or [],
            approved=approved,
            created_at=datetime.now().isoformat(),
        )

        # Store in memory
        self.assets[asset_id] = asset

        # Store in database if available
        guardian = self._get_brand_guardian()
        if guardian:
            try:
                from brand_guardian import AssetType as AT
                guardian.add_asset(
                    path,
                    AT(asset_type.value),
                    name,
                    tags=tags,
                    approved=approved,
                )
            except Exception:
                pass

        return asset

    def approve_asset(self, asset_id: str):
        """Approve an asset for use."""
        if asset_id in self.assets:
            self.assets[asset_id].approved = True

        guardian = self._get_brand_guardian()
        if guardian:
            try:
                guardian.approve_asset(asset_id)
            except Exception:
                pass

    def search_assets(
        self,
        asset_type: Optional[AssetType] = None,
        tags: Optional[List[str]] = None,
        approved_only: bool = False,
    ) -> List[BrandAsset]:
        """
        Search brand assets.

        Args:
            asset_type: Filter by type
            tags: Filter by tags
            approved_only: Only approved assets

        Returns:
            List of matching assets
        """
        results = []

        for asset in self.assets.values():
            if asset_type and asset.asset_type != asset_type:
                continue
            if approved_only and not asset.approved:
                continue
            if tags and not any(t in asset.tags for t in tags):
                continue
            results.append(asset)

        return results

    def create_branded_version(
        self,
        image: Union[str, Path],
        output: Union[str, Path],
        add_watermark: bool = True,
        watermark_text: Optional[str] = None,
    ) -> str:
        """
        Create branded version with watermark.

        Args:
            image: Input image
            output: Output path
            add_watermark: Add watermark
            watermark_text: Custom watermark text

        Returns:
            Output path
        """
        guardian = self._get_brand_guardian()

        if guardian:
            try:
                return guardian.create_branded_version(
                    image, output, add_watermark, watermark_text
                )
            except Exception:
                pass

        # Fallback: basic watermark
        return self._add_watermark(Path(image), Path(output), watermark_text)

    def _add_watermark(
        self,
        input_path: Path,
        output_path: Path,
        text: Optional[str] = None,
    ) -> str:
        """Add basic watermark to image."""
        from PIL import Image, ImageDraw, ImageFont

        img = Image.open(input_path)
        draw = ImageDraw.Draw(img)

        watermark = text or f"(C) {self.guidelines.name}"

        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except Exception:
            font = ImageFont.load_default()

        # Position bottom-right
        bbox = draw.textbbox((0, 0), watermark, font=font)
        x = img.width - bbox[2] - 20
        y = img.height - bbox[3] - 20
        draw.text((x, y), watermark, fill=(255, 255, 255, 128), font=font)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, quality=95)

        return str(output_path)

    def generate_report(self) -> Dict[str, Any]:
        """Generate brand compliance report."""
        guardian = self._get_brand_guardian()

        if guardian:
            try:
                report = guardian.generate_report()
                return report.to_dict()
            except Exception:
                pass

        # Fallback report
        return {
            'total_assets': len(self.assets),
            'approved': sum(1 for a in self.assets.values() if a.approved),
            'guidelines': self.guidelines.to_dict(),
            'generated_at': datetime.now().isoformat(),
        }

    def export_report(
        self,
        output: Union[str, Path],
        format: str = 'json',
    ) -> str:
        """Export brand report."""
        import json

        report = self.generate_report()
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        return str(out_path)


# Convenience functions
def create_guidelines(
    name: str,
    primary_colors: List[Tuple[int, int, int]],
    logo_files: Optional[List[str]] = None,
) -> BrandGuidelines:
    """Create brand guidelines."""
    return BrandGuidelines(
        name=name,
        primary_colors=primary_colors,
        logo_files=logo_files or [],
    )


def check_brand_compliance(
    image: Union[str, Path],
    guidelines: BrandGuidelines,
) -> ComplianceResult:
    """Quick brand compliance check."""
    return BrandManagement(guidelines=guidelines).check_compliance(image)

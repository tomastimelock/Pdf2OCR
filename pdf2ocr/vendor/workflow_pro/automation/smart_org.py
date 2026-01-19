"""Smart Organization Automation.

Provides intelligent photo library organization with AI capabilities.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime


@dataclass
class OrganizationRule:
    """Rule for automatic organization."""
    name: str
    criteria: Dict[str, Any]
    action: str  # move, copy, tag, collection
    destination: Optional[str] = None
    priority: int = 0

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'criteria': self.criteria,
            'action': self.action,
            'destination': self.destination,
        }


@dataclass
class OrganizationResult:
    """Result of organization operation."""
    total_files: int
    organized: int
    duplicates: int
    collections_created: int
    rules_applied: int
    by_category: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    success: bool = True

    def to_dict(self) -> Dict:
        return {
            'total_files': self.total_files,
            'organized': self.organized,
            'duplicates': self.duplicates,
            'collections': self.collections_created,
            'success': self.success,
        }


class SmartOrganization:
    """Smart photo organization with AI.

    Features:
    - Scene-based organization
    - Face detection and grouping
    - Date/time organization
    - Location-based grouping
    - Duplicate detection
    - Quality-based filtering
    - Custom organization rules
    """

    def __init__(
        self,
        library_path: Optional[Union[str, Path]] = None,
        database_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize smart organization.

        Args:
            library_path: Photo library root path
            database_path: Path for organization database
        """
        self.library_path = Path(library_path) if library_path else None
        self.database_path = Path(database_path) if database_path else None

        # Organization rules
        self.rules: List[OrganizationRule] = []

        # Callbacks
        self._on_organize: Optional[Callable] = None
        self._on_duplicate: Optional[Callable] = None

        # Lazy-loaded modules
        self._smart_organizer = None
        self._vision_ai = None

    def _get_smart_organizer(self):
        """Lazy load SmartOrganizer."""
        if self._smart_organizer is None:
            try:
                from smart_organizer import SmartOrganizer
                self._smart_organizer = SmartOrganizer(
                    library_path=self.library_path,
                    database_path=self.database_path,
                )
            except ImportError:
                pass
        return self._smart_organizer

    def _get_vision_ai(self):
        """Lazy load VisionAI super-module."""
        if self._vision_ai is None:
            try:
                from vision_ai import VisionAI
                self._vision_ai = VisionAI()
            except ImportError:
                pass
        return self._vision_ai

    def on_organize(self, callback: Callable):
        """Set callback for organization events."""
        self._on_organize = callback

    def on_duplicate(self, callback: Callable):
        """Set callback for duplicate detection."""
        self._on_duplicate = callback

    def add_rule(self, rule: OrganizationRule):
        """Add an organization rule."""
        self.rules.append(rule)
        # Sort by priority
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def clear_rules(self):
        """Clear all organization rules."""
        self.rules.clear()

    def organize(
        self,
        source: Union[str, Path],
        output: Optional[Union[str, Path]] = None,
        use_ai: bool = True,
        detect_duplicates: bool = True,
        create_collections: bool = True,
        min_quality: float = 0.0,
    ) -> OrganizationResult:
        """
        Organize photos intelligently.

        Args:
            source: Source directory
            output: Output directory (default: source/organized)
            use_ai: Enable AI-powered features
            detect_duplicates: Find and handle duplicates
            create_collections: Auto-create collections
            min_quality: Minimum quality threshold

        Returns:
            OrganizationResult with statistics
        """
        source_path = Path(source)
        output_path = Path(output) if output else source_path / 'organized'
        output_path.mkdir(parents=True, exist_ok=True)

        result = OrganizationResult(
            total_files=0,
            organized=0,
            duplicates=0,
            collections_created=0,
            rules_applied=0,
        )

        organizer = self._get_smart_organizer()

        if organizer and use_ai:
            try:
                org_result = organizer.organize_library(
                    source_dir=source_path,
                    recursive=True,
                    detect_duplicates=detect_duplicates,
                    create_collections=create_collections,
                    min_quality=min_quality,
                )

                result.total_files = org_result.total_photos
                result.organized = org_result.photos_organized
                result.duplicates = org_result.duplicates_found
                result.collections_created = org_result.collections_created
                result.success = org_result.success
                result.errors = org_result.errors

            except Exception as e:
                result.errors.append(str(e))
                result.success = False
        else:
            # Fallback: basic organization
            result = self._basic_organize(source_path, output_path)

        # Apply custom rules
        if self.rules:
            rules_applied = self._apply_rules(source_path, output_path)
            result.rules_applied = rules_applied

        return result

    def _basic_organize(
        self,
        source: Path,
        output: Path,
    ) -> OrganizationResult:
        """Basic organization by date."""
        from PIL import Image
        from PIL.ExifTags import TAGS
        import shutil
        import hashlib

        result = OrganizationResult(
            total_files=0,
            organized=0,
            duplicates=0,
            collections_created=0,
            rules_applied=0,
        )

        # Find images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff']
        images = []
        for ext in extensions:
            images.extend(source.rglob(ext))
            images.extend(source.rglob(ext.upper()))

        result.total_files = len(images)

        # Track duplicates by hash
        seen_hashes: Dict[str, str] = {}

        for img_path in images:
            try:
                # Calculate hash for duplicate detection
                with open(img_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()

                if file_hash in seen_hashes:
                    result.duplicates += 1
                    if self._on_duplicate:
                        self._on_duplicate(img_path, seen_hashes[file_hash])
                    continue

                seen_hashes[file_hash] = str(img_path)

                # Get date from EXIF
                date_folder = self._get_date_folder(img_path)

                # Create destination
                dest_dir = output / date_folder
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / img_path.name

                shutil.copy2(img_path, dest)
                result.organized += 1

                if date_folder not in result.by_category:
                    result.by_category[date_folder] = 0
                result.by_category[date_folder] += 1

                if self._on_organize:
                    self._on_organize(img_path, dest)

            except Exception as e:
                result.errors.append(f"{img_path}: {str(e)}")

        result.collections_created = len(result.by_category)
        result.success = len(result.errors) == 0

        return result

    def _get_date_folder(self, img_path: Path) -> str:
        """Get date folder name from image."""
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS

            img = Image.open(img_path)
            exif = img._getexif()

            if exif:
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == 'DateTimeOriginal':
                        return value[:7].replace(':', '-')

        except Exception:
            pass

        # Fallback: use file modification date
        mtime = datetime.fromtimestamp(img_path.stat().st_mtime)
        return mtime.strftime('%Y-%m')

    def _apply_rules(self, source: Path, output: Path) -> int:
        """Apply custom organization rules."""
        rules_applied = 0

        for rule in self.rules:
            try:
                # Apply rule based on criteria
                if rule.action == 'move':
                    pass  # Implement move logic
                elif rule.action == 'copy':
                    pass  # Implement copy logic
                elif rule.action == 'tag':
                    pass  # Implement tagging logic
                elif rule.action == 'collection':
                    pass  # Implement collection logic

                rules_applied += 1
            except Exception:
                pass

        return rules_applied

    def analyze_photo(
        self,
        image: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Analyze a single photo.

        Args:
            image: Image path

        Returns:
            Analysis results
        """
        img_path = Path(image)

        organizer = self._get_smart_organizer()

        if organizer:
            try:
                analysis = organizer.analyze_photo(img_path)
                return analysis.to_dict()
            except Exception:
                pass

        # Fallback: basic analysis
        return self._basic_analyze(img_path)

    def _basic_analyze(self, img_path: Path) -> Dict[str, Any]:
        """Basic image analysis."""
        from PIL import Image
        from PIL.ExifTags import TAGS
        import hashlib

        result = {
            'path': str(img_path),
            'date': None,
            'size': None,
            'hash': None,
        }

        try:
            with open(img_path, 'rb') as f:
                result['hash'] = hashlib.md5(f.read()).hexdigest()

            img = Image.open(img_path)
            result['size'] = img.size

            exif = img._getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == 'DateTimeOriginal':
                        result['date'] = str(value)
                        break

        except Exception:
            pass

        return result

    def find_duplicates(
        self,
        directory: Union[str, Path],
    ) -> Dict[str, List[str]]:
        """
        Find duplicate photos.

        Args:
            directory: Directory to scan

        Returns:
            Dictionary of hash -> list of duplicate paths
        """
        import hashlib

        dir_path = Path(directory)
        hash_to_files: Dict[str, List[str]] = {}

        extensions = ['*.jpg', '*.jpeg', '*.png']
        images = []
        for ext in extensions:
            images.extend(dir_path.rglob(ext))
            images.extend(dir_path.rglob(ext.upper()))

        for img_path in images:
            try:
                with open(img_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()

                if file_hash not in hash_to_files:
                    hash_to_files[file_hash] = []
                hash_to_files[file_hash].append(str(img_path))

            except Exception:
                pass

        # Return only groups with duplicates
        return {k: v for k, v in hash_to_files.items() if len(v) > 1}

    def search(
        self,
        query: str,
        directory: Optional[Union[str, Path]] = None,
        scene: Optional[str] = None,
        has_faces: Optional[bool] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search photos with criteria.

        Args:
            query: Search query
            directory: Directory to search (default: library_path)
            scene: Filter by scene type
            has_faces: Filter by face presence
            date_from: Start date
            date_to: End date

        Returns:
            List of matching photos
        """
        organizer = self._get_smart_organizer()

        if organizer:
            try:
                result = organizer.search(
                    query=query,
                    scene=scene,
                    has_faces=has_faces,
                    date_from=date_from,
                    date_to=date_to,
                )
                return [p.to_dict() for p in result.photos]
            except Exception:
                pass

        return []

    def get_best_photos(
        self,
        directory: Optional[Union[str, Path]] = None,
        count: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get the best photos by quality.

        Args:
            directory: Directory to scan
            count: Number of photos to return

        Returns:
            List of best photos
        """
        organizer = self._get_smart_organizer()

        if organizer:
            try:
                photos = organizer.get_best_photos(count=count)
                return [p.to_dict() for p in photos]
            except Exception:
                pass

        return []

    def export_report(
        self,
        output: Union[str, Path],
        format: str = 'json',
    ) -> str:
        """
        Export organization report.

        Args:
            output: Output path
            format: Report format (json, text)

        Returns:
            Output path
        """
        organizer = self._get_smart_organizer()

        if organizer:
            try:
                return organizer.export_report(output, format)
            except Exception:
                pass

        return str(output)


# Pre-built organization rules
def create_scene_rule(scene: str, destination: str) -> OrganizationRule:
    """Create a scene-based organization rule."""
    return OrganizationRule(
        name=f"Scene: {scene}",
        criteria={'scene': scene},
        action='move',
        destination=destination,
    )


def create_date_rule(year: int, month: Optional[int] = None, destination: str = "") -> OrganizationRule:
    """Create a date-based organization rule."""
    criteria = {'year': year}
    if month:
        criteria['month'] = month

    return OrganizationRule(
        name=f"Date: {year}" + (f"-{month:02d}" if month else ""),
        criteria=criteria,
        action='move',
        destination=destination or f"{year}/{month:02d}" if month else str(year),
    )


def create_quality_rule(min_score: float, destination: str) -> OrganizationRule:
    """Create a quality-based organization rule."""
    return OrganizationRule(
        name=f"Quality >= {min_score}",
        criteria={'min_quality': min_score},
        action='copy',
        destination=destination,
        priority=10,
    )

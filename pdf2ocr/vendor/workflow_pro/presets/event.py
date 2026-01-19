"""Event Photography Workflow Preset.

Pre-configured workflow for event photography automation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime


@dataclass
class CullingConfig:
    """Photo culling configuration."""
    min_quality: float = 0.5
    reject_blurry: bool = True
    reject_duplicates: bool = True
    blur_threshold: float = 100.0
    duplicate_threshold: float = 0.95


@dataclass
class PeopleConfig:
    """People detection configuration."""
    enabled: bool = True
    group_by_person: bool = True
    create_person_albums: bool = True
    min_photos_per_person: int = 3


@dataclass
class AlbumConfig:
    """Album creation configuration."""
    create_best_of: bool = True
    best_of_count: int = 50
    create_group_shots: bool = True
    create_people_album: bool = True
    create_timeline: bool = True


@dataclass
class DeliveryConfig:
    """Delivery configuration."""
    create_gallery: bool = True
    create_slideshow: bool = True
    slideshow_duration: int = 3
    slideshow_transition: str = "fade"
    gallery_template: str = "default"
    enable_download: bool = True


@dataclass
class EventPresetConfig:
    """Complete event workflow configuration."""
    event_name: str = ""
    event_date: Optional[str] = None
    culling: CullingConfig = field(default_factory=CullingConfig)
    people: PeopleConfig = field(default_factory=PeopleConfig)
    albums: AlbumConfig = field(default_factory=AlbumConfig)
    delivery: DeliveryConfig = field(default_factory=DeliveryConfig)

    def to_dict(self) -> Dict:
        return {
            'event_name': self.event_name,
            'event_date': self.event_date,
            'culling': {
                'min_quality': self.culling.min_quality,
                'reject_blurry': self.culling.reject_blurry,
                'reject_duplicates': self.culling.reject_duplicates,
            },
            'people': {
                'enabled': self.people.enabled,
                'group_by_person': self.people.group_by_person,
            },
            'albums': {
                'create_best_of': self.albums.create_best_of,
                'best_of_count': self.albums.best_of_count,
            },
            'delivery': {
                'create_gallery': self.delivery.create_gallery,
                'create_slideshow': self.delivery.create_slideshow,
            },
        }


class EventPreset:
    """Event photography workflow preset.

    Provides pre-configured workflows for event photography including:
    - Wedding photography
    - Corporate events
    - Party/celebration
    - Conference/seminar
    - Sports event
    """

    WEDDING = EventPresetConfig(
        culling=CullingConfig(min_quality=0.6, reject_blurry=True),
        people=PeopleConfig(enabled=True, create_person_albums=True),
        albums=AlbumConfig(create_best_of=True, best_of_count=100),
        delivery=DeliveryConfig(create_slideshow=True, slideshow_duration=4),
    )

    CORPORATE = EventPresetConfig(
        culling=CullingConfig(min_quality=0.5, reject_blurry=True),
        people=PeopleConfig(enabled=True, create_person_albums=False),
        albums=AlbumConfig(create_best_of=True, best_of_count=30),
        delivery=DeliveryConfig(create_slideshow=False, create_gallery=True),
    )

    PARTY = EventPresetConfig(
        culling=CullingConfig(min_quality=0.4, reject_blurry=False),
        people=PeopleConfig(enabled=True, create_person_albums=True),
        albums=AlbumConfig(create_best_of=True, best_of_count=50),
        delivery=DeliveryConfig(create_slideshow=True, slideshow_duration=2),
    )

    CONFERENCE = EventPresetConfig(
        culling=CullingConfig(min_quality=0.5, reject_blurry=True),
        people=PeopleConfig(enabled=False),
        albums=AlbumConfig(create_best_of=True, create_timeline=True),
        delivery=DeliveryConfig(create_gallery=True, create_slideshow=False),
    )

    SPORTS = EventPresetConfig(
        culling=CullingConfig(min_quality=0.6, reject_blurry=True, blur_threshold=150),
        people=PeopleConfig(enabled=False),
        albums=AlbumConfig(create_best_of=True, best_of_count=100),
        delivery=DeliveryConfig(create_slideshow=True, slideshow_duration=2),
    )

    def __init__(self, config: Optional[EventPresetConfig] = None):
        """
        Initialize event preset.

        Args:
            config: Custom configuration or use one of the preset constants
        """
        self.config = config or EventPresetConfig()
        self._photographer = None

    def _get_photographer(self):
        """Lazy load EventPhotographer."""
        if self._photographer is None:
            try:
                from event_photographer import EventPhotographer
                self._photographer = EventPhotographer(
                    quality_threshold=self.config.culling.min_quality
                )
            except ImportError:
                pass
        return self._photographer

    def process(
        self,
        source_dir: Union[str, Path],
        output_dir: Union[str, Path],
        event_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process event photos using preset configuration.

        Args:
            source_dir: Directory containing event photos
            output_dir: Output directory
            event_name: Event name (overrides config)

        Returns:
            Processing result dictionary
        """
        source = Path(source_dir)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        name = event_name or self.config.event_name or source.name

        result = {
            'event_name': name,
            'source': str(source),
            'output': str(output),
            'success': False,
        }

        photographer = self._get_photographer()

        if photographer:
            try:
                event_result = photographer.process_event(
                    source_dir=source,
                    event_name=name,
                    output_dir=output,
                    cull=True,
                    detect_people=self.config.people.enabled,
                    create_albums=True,
                    create_slideshow=self.config.delivery.create_slideshow,
                    create_gallery=self.config.delivery.create_gallery,
                )

                result.update({
                    'total_photos': event_result.total_photos,
                    'selected_photos': event_result.selected_photos,
                    'albums_created': event_result.albums_created,
                    'people_detected': event_result.people_detected,
                    'slideshow': event_result.slideshow_path,
                    'gallery': event_result.gallery_path,
                    'success': event_result.success,
                })

            except Exception as e:
                result['error'] = str(e)
        else:
            result['error'] = "EventPhotographer module not available"

        return result

    def create_wedding_preset(
        self,
        couple_names: str,
        event_date: str,
    ) -> EventPresetConfig:
        """Create a wedding-specific preset."""
        config = EventPresetConfig(
            event_name=f"{couple_names} Wedding",
            event_date=event_date,
            culling=CullingConfig(min_quality=0.65, reject_blurry=True),
            people=PeopleConfig(
                enabled=True,
                create_person_albums=True,
                min_photos_per_person=5,
            ),
            albums=AlbumConfig(
                create_best_of=True,
                best_of_count=150,
                create_group_shots=True,
            ),
            delivery=DeliveryConfig(
                create_gallery=True,
                create_slideshow=True,
                slideshow_duration=4,
                enable_download=True,
            ),
        )
        return config

    def create_corporate_preset(
        self,
        company_name: str,
        event_type: str,
    ) -> EventPresetConfig:
        """Create a corporate event preset."""
        config = EventPresetConfig(
            event_name=f"{company_name} - {event_type}",
            culling=CullingConfig(min_quality=0.55),
            people=PeopleConfig(enabled=True, create_person_albums=False),
            albums=AlbumConfig(
                create_best_of=True,
                best_of_count=50,
                create_timeline=True,
            ),
            delivery=DeliveryConfig(
                create_gallery=True,
                create_slideshow=False,
            ),
        )
        return config

    @staticmethod
    def list_presets() -> List[str]:
        """List available preset names."""
        return ['WEDDING', 'CORPORATE', 'PARTY', 'CONFERENCE', 'SPORTS']

    @classmethod
    def get_preset(cls, name: str) -> EventPresetConfig:
        """Get a preset by name."""
        presets = {
            'WEDDING': cls.WEDDING,
            'CORPORATE': cls.CORPORATE,
            'PARTY': cls.PARTY,
            'CONFERENCE': cls.CONFERENCE,
            'SPORTS': cls.SPORTS,
        }
        return presets.get(name.upper(), EventPresetConfig())

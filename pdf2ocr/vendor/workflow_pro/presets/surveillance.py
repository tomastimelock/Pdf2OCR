"""Surveillance Workflow Preset.

Pre-configured workflow for video surveillance and monitoring.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from datetime import datetime
from enum import Enum


class AlertLevel(Enum):
    """Alert severity level."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventType(Enum):
    """Detection event type."""
    MOTION = "motion"
    PERSON = "person"
    VEHICLE = "vehicle"
    FACE_KNOWN = "face_known"
    FACE_UNKNOWN = "face_unknown"
    OBJECT = "object"


@dataclass
class MotionConfig:
    """Motion detection configuration."""
    enabled: bool = True
    sensitivity: float = 0.5
    min_area: int = 500
    alert_on_motion: bool = True


@dataclass
class ObjectConfig:
    """Object detection configuration."""
    enabled: bool = True
    detect_people: bool = True
    detect_vehicles: bool = True
    detect_objects: bool = False
    confidence_threshold: float = 0.5


@dataclass
class FaceConfig:
    """Face detection configuration."""
    enabled: bool = True
    recognize_known: bool = True
    alert_unknown: bool = False
    known_faces_dir: Optional[str] = None


@dataclass
class AlertConfig:
    """Alert configuration."""
    enabled: bool = True
    webhook_url: Optional[str] = None
    email_alerts: bool = False
    email_recipients: List[str] = field(default_factory=list)
    alert_cooldown: int = 60  # seconds between alerts


@dataclass
class StorageConfig:
    """Storage configuration."""
    save_frames: bool = True
    frames_dir: Optional[str] = None
    save_events: bool = True
    retention_days: int = 30
    database_path: Optional[str] = None


@dataclass
class SurveillancePresetConfig:
    """Complete surveillance workflow configuration."""
    motion: MotionConfig = field(default_factory=MotionConfig)
    objects: ObjectConfig = field(default_factory=ObjectConfig)
    faces: FaceConfig = field(default_factory=FaceConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    frame_interval: float = 1.0  # seconds

    def to_dict(self) -> Dict:
        return {
            'motion': {
                'enabled': self.motion.enabled,
                'sensitivity': self.motion.sensitivity,
            },
            'objects': {
                'enabled': self.objects.enabled,
                'detect_people': self.objects.detect_people,
                'detect_vehicles': self.objects.detect_vehicles,
            },
            'faces': {
                'enabled': self.faces.enabled,
                'recognize_known': self.faces.recognize_known,
            },
            'alerts': {
                'enabled': self.alerts.enabled,
                'webhook_url': self.alerts.webhook_url,
            },
            'storage': {
                'save_frames': self.storage.save_frames,
                'retention_days': self.storage.retention_days,
            },
            'frame_interval': self.frame_interval,
        }


class SurveillancePreset:
    """Surveillance workflow preset.

    Provides pre-configured workflows for surveillance including:
    - Basic motion detection
    - Person detection
    - Full security monitoring
    - Access control
    """

    MOTION_ONLY = SurveillancePresetConfig(
        motion=MotionConfig(enabled=True, sensitivity=0.5),
        objects=ObjectConfig(enabled=False),
        faces=FaceConfig(enabled=False),
        frame_interval=0.5,
    )

    PERSON_DETECTION = SurveillancePresetConfig(
        motion=MotionConfig(enabled=True, sensitivity=0.4),
        objects=ObjectConfig(
            enabled=True,
            detect_people=True,
            detect_vehicles=False,
        ),
        faces=FaceConfig(enabled=False),
        frame_interval=1.0,
    )

    FULL_SECURITY = SurveillancePresetConfig(
        motion=MotionConfig(enabled=True, sensitivity=0.3),
        objects=ObjectConfig(
            enabled=True,
            detect_people=True,
            detect_vehicles=True,
            detect_objects=True,
        ),
        faces=FaceConfig(
            enabled=True,
            recognize_known=True,
            alert_unknown=True,
        ),
        alerts=AlertConfig(enabled=True),
        storage=StorageConfig(
            save_frames=True,
            retention_days=30,
        ),
        frame_interval=0.5,
    )

    ACCESS_CONTROL = SurveillancePresetConfig(
        motion=MotionConfig(enabled=False),
        objects=ObjectConfig(
            enabled=True,
            detect_people=True,
        ),
        faces=FaceConfig(
            enabled=True,
            recognize_known=True,
            alert_unknown=True,
        ),
        alerts=AlertConfig(
            enabled=True,
            alert_cooldown=30,
        ),
        frame_interval=0.25,
    )

    def __init__(self, config: Optional[SurveillancePresetConfig] = None):
        """
        Initialize surveillance preset.

        Args:
            config: Custom configuration or use one of the preset constants
        """
        self.config = config or SurveillancePresetConfig()
        self._surveillance = None
        self._on_event: Optional[Callable] = None
        self._on_alert: Optional[Callable] = None

    def _get_surveillance(self):
        """Lazy load SurveillanceAI."""
        if self._surveillance is None:
            try:
                from surveillance_ai import SurveillanceAI

                self._surveillance = SurveillanceAI(
                    database_path=self.config.storage.database_path,
                    frames_dir=self.config.storage.frames_dir,
                )

                # Register known faces if configured
                if self.config.faces.known_faces_dir:
                    self._load_known_faces()

                # Set up callbacks
                if self._on_event:
                    self._surveillance.on_event(self._on_event)
                if self._on_alert:
                    self._surveillance.on_alert(self._on_alert)

            except ImportError:
                pass
        return self._surveillance

    def _load_known_faces(self):
        """Load known faces from directory."""
        if not self.config.faces.known_faces_dir:
            return

        faces_dir = Path(self.config.faces.known_faces_dir)
        if not faces_dir.exists():
            return

        # Load face images and register them
        for face_file in faces_dir.glob("*.jpg"):
            name = face_file.stem
            self._surveillance.register_known_face(name, name)

    def on_event(self, callback: Callable):
        """Set event callback."""
        self._on_event = callback
        if self._surveillance:
            self._surveillance.on_event(callback)

    def on_alert(self, callback: Callable):
        """Set alert callback."""
        self._on_alert = callback
        if self._surveillance:
            self._surveillance.on_alert(callback)

    def process_video(
        self,
        video_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Process video for surveillance events.

        Args:
            video_path: Path to video file

        Returns:
            Processing result dictionary
        """
        path = Path(video_path)

        result = {
            'source': str(path),
            'success': False,
        }

        surveillance = self._get_surveillance()

        if surveillance:
            try:
                monitor_result = surveillance.process_video(
                    video_path=path,
                    extract_interval=self.config.frame_interval,
                    detect_motion=self.config.motion.enabled,
                    detect_objects=self.config.objects.enabled,
                    detect_faces=self.config.faces.enabled,
                )

                result.update({
                    'duration': monitor_result.duration,
                    'events_detected': monitor_result.events_detected,
                    'alerts_triggered': monitor_result.alerts_triggered,
                    'timeline': [t.to_dict() for t in monitor_result.timeline],
                    'events': [e.to_dict() for e in monitor_result.events[:50]],  # Limit
                    'success': monitor_result.success,
                })

                if not monitor_result.success:
                    result['error'] = monitor_result.error

            except Exception as e:
                result['error'] = str(e)
        else:
            result['error'] = "SurveillanceAI module not available"

        return result

    def search_events(
        self,
        event_type: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        min_confidence: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search surveillance events.

        Args:
            event_type: Filter by event type
            date_from: Start date
            date_to: End date
            min_confidence: Minimum confidence threshold

        Returns:
            List of matching events
        """
        surveillance = self._get_surveillance()

        if surveillance:
            try:
                et = EventType(event_type) if event_type else None
                results = surveillance.search_events(
                    event_type=et,
                    date_from=date_from,
                    date_to=date_to,
                    min_confidence=min_confidence,
                )
                return [e.to_dict() for e in results.events]

            except Exception:
                pass

        return []

    def get_activity_summary(
        self,
        date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get activity summary.

        Args:
            date: Date to summarize (default: today)

        Returns:
            Activity summary
        """
        surveillance = self._get_surveillance()

        if surveillance:
            try:
                return surveillance.get_activity_summary(date)
            except Exception:
                pass

        return {}

    def cleanup(self, days: Optional[int] = None):
        """
        Cleanup old frames.

        Args:
            days: Days to retain (default: from config)
        """
        surveillance = self._get_surveillance()

        if surveillance:
            retention = days or self.config.storage.retention_days
            surveillance.cleanup_old_frames(retention)

    def add_alert_rule(
        self,
        name: str,
        event_types: List[str],
        level: str = "medium",
        time_window: Optional[Tuple[int, int]] = None,
        webhook_url: Optional[str] = None,
    ):
        """
        Add an alert rule.

        Args:
            name: Rule name
            event_types: Event types to trigger on
            level: Alert level
            time_window: Optional time window (start_hour, end_hour)
            webhook_url: Optional webhook URL
        """
        surveillance = self._get_surveillance()

        if surveillance:
            try:
                from surveillance_ai import AlertRule, AlertLevel as AL, EventType as ET

                types = [ET(et) for et in event_types]
                al = AL(level)

                rule = AlertRule(
                    name=name,
                    event_types=types,
                    alert_level=al,
                    time_window=time_window,
                    notify=True,
                    webhook_url=webhook_url or self.config.alerts.webhook_url,
                )
                surveillance.add_alert_rule(rule)

            except Exception:
                pass

    @staticmethod
    def list_presets() -> List[str]:
        """List available preset names."""
        return ['MOTION_ONLY', 'PERSON_DETECTION', 'FULL_SECURITY', 'ACCESS_CONTROL']

    @classmethod
    def get_preset(cls, name: str) -> SurveillancePresetConfig:
        """Get a preset by name."""
        presets = {
            'MOTION_ONLY': cls.MOTION_ONLY,
            'PERSON_DETECTION': cls.PERSON_DETECTION,
            'FULL_SECURITY': cls.FULL_SECURITY,
            'ACCESS_CONTROL': cls.ACCESS_CONTROL,
        }
        return presets.get(name.upper(), SurveillancePresetConfig())

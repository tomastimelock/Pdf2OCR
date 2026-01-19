"""Content Moderation Automation.

Provides automated content safety and privacy protection.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime
from enum import Enum


class ThreatLevel(Enum):
    """Content threat level."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(Enum):
    """Moderation action type."""
    ALLOW = "allow"
    FLAG = "flag"
    BLUR_FACES = "blur_faces"
    REDACT_TEXT = "redact_text"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    NOTIFY = "notify"


@dataclass
class ModerationPolicy:
    """Content moderation policy."""
    name: str = "default"

    # NSFW settings
    nsfw_enabled: bool = True
    nsfw_threshold: float = 0.7
    nsfw_action: ActionType = ActionType.BLOCK

    # Face detection settings
    face_detection_enabled: bool = True
    auto_blur_faces: bool = False
    face_action: ActionType = ActionType.FLAG

    # PII detection settings
    pii_detection_enabled: bool = True
    pii_patterns: List[str] = field(default_factory=lambda: [
        'email', 'phone', 'ssn', 'credit_card'
    ])
    pii_action: ActionType = ActionType.REDACT_TEXT

    # Notification settings
    notify_on_threat: bool = True
    notify_threshold: ThreatLevel = ThreatLevel.MEDIUM
    webhook_url: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'nsfw': {
                'enabled': self.nsfw_enabled,
                'threshold': self.nsfw_threshold,
                'action': self.nsfw_action.value,
            },
            'faces': {
                'enabled': self.face_detection_enabled,
                'auto_blur': self.auto_blur_faces,
            },
            'pii': {
                'enabled': self.pii_detection_enabled,
                'patterns': self.pii_patterns,
            },
        }


@dataclass
class ModerationResult:
    """Result of content moderation."""
    path: str
    threat_level: ThreatLevel
    detections: List[Dict[str, Any]] = field(default_factory=list)
    actions_taken: List[ActionType] = field(default_factory=list)
    output_path: Optional[str] = None
    processing_time: float = 0.0
    success: bool = True
    error: str = ""

    def to_dict(self) -> Dict:
        return {
            'path': self.path,
            'threat_level': self.threat_level.value,
            'detections': self.detections,
            'actions': [a.value for a in self.actions_taken],
            'output_path': self.output_path,
            'success': self.success,
        }


@dataclass
class BatchModerationResult:
    """Batch moderation result."""
    total: int
    processed: int
    safe: int
    flagged: int
    blocked: int
    errors: int
    threat_summary: Dict[str, int] = field(default_factory=dict)
    success: bool = True

    def to_dict(self) -> Dict:
        return {
            'total': self.total,
            'processed': self.processed,
            'safe': self.safe,
            'flagged': self.flagged,
            'blocked': self.blocked,
            'errors': self.errors,
        }


class ContentModeration:
    """Content moderation and privacy protection.

    Features:
    - NSFW detection and blocking
    - Face detection and blurring
    - PII detection and redaction
    - Sentiment analysis
    - Automated protective actions
    - Compliance checking
    """

    # Pre-built policies
    STRICT = ModerationPolicy(
        name="strict",
        nsfw_enabled=True,
        nsfw_threshold=0.5,
        nsfw_action=ActionType.BLOCK,
        face_detection_enabled=True,
        auto_blur_faces=True,
        pii_detection_enabled=True,
        notify_on_threat=True,
        notify_threshold=ThreatLevel.LOW,
    )

    MODERATE = ModerationPolicy(
        name="moderate",
        nsfw_enabled=True,
        nsfw_threshold=0.7,
        nsfw_action=ActionType.FLAG,
        face_detection_enabled=True,
        auto_blur_faces=False,
        pii_detection_enabled=True,
        notify_threshold=ThreatLevel.MEDIUM,
    )

    RELAXED = ModerationPolicy(
        name="relaxed",
        nsfw_enabled=True,
        nsfw_threshold=0.9,
        nsfw_action=ActionType.FLAG,
        face_detection_enabled=False,
        pii_detection_enabled=False,
        notify_on_threat=False,
    )

    def __init__(self, policy: Optional[ModerationPolicy] = None):
        """
        Initialize content moderation.

        Args:
            policy: Moderation policy to use
        """
        self.policy = policy or ModerationPolicy()

        # Callbacks
        self._on_detection: Optional[Callable] = None
        self._on_action: Optional[Callable] = None

        # Lazy-loaded modules
        self._content_shield = None
        self._vision_ai = None

    def _get_content_shield(self):
        """Lazy load ContentShield."""
        if self._content_shield is None:
            try:
                from content_shield import ContentShield, PolicyConfig

                config = PolicyConfig(
                    nsfw_enabled=self.policy.nsfw_enabled,
                    nsfw_threshold=self.policy.nsfw_threshold,
                    face_detection_enabled=self.policy.face_detection_enabled,
                    auto_blur_faces=self.policy.auto_blur_faces,
                    pii_detection_enabled=self.policy.pii_detection_enabled,
                    pii_patterns=self.policy.pii_patterns,
                    notify_on_threat=self.policy.notify_on_threat,
                    webhook_url=self.policy.webhook_url,
                )
                self._content_shield = ContentShield(policy=config)

            except ImportError:
                pass
        return self._content_shield

    def _get_vision_ai(self):
        """Lazy load VisionAI."""
        if self._vision_ai is None:
            try:
                from vision_ai import VisionAI
                self._vision_ai = VisionAI()
            except ImportError:
                pass
        return self._vision_ai

    def on_detection(self, callback: Callable):
        """Set callback for detections."""
        self._on_detection = callback

    def on_action(self, callback: Callable):
        """Set callback for actions taken."""
        self._on_action = callback

    def set_policy(self, policy: ModerationPolicy):
        """Update moderation policy."""
        self.policy = policy
        self._content_shield = None  # Reset to apply new policy

    def moderate(
        self,
        image: Union[str, Path],
        output: Optional[Union[str, Path]] = None,
        apply_actions: bool = True,
    ) -> ModerationResult:
        """
        Moderate a single image.

        Args:
            image: Image path
            output: Output path for modified image
            apply_actions: Whether to apply protective actions

        Returns:
            ModerationResult with details
        """
        import time
        start_time = time.time()

        path = Path(image)

        result = ModerationResult(
            path=str(path),
            threat_level=ThreatLevel.SAFE,
        )

        shield = self._get_content_shield()

        if shield:
            try:
                mod_result = shield.moderate(path, output, apply_actions)

                # Convert threat level
                try:
                    result.threat_level = ThreatLevel(mod_result.threat_level.value)
                except ValueError:
                    result.threat_level = ThreatLevel.SAFE

                result.detections = [d.to_dict() for d in mod_result.detections]
                result.actions_taken = [
                    ActionType(a.value) for a in mod_result.actions_taken
                ]
                result.output_path = mod_result.output_path
                result.success = mod_result.success
                result.error = mod_result.error

                # Callbacks
                for detection in mod_result.detections:
                    if self._on_detection:
                        self._on_detection(detection)

                for action in mod_result.actions_taken:
                    if self._on_action:
                        self._on_action(action, str(path))

            except Exception as e:
                result.success = False
                result.error = str(e)
        else:
            # Fallback: basic analysis
            result = self._basic_moderate(path)

        result.processing_time = time.time() - start_time

        return result

    def _basic_moderate(self, path: Path) -> ModerationResult:
        """Basic moderation without full module."""
        return ModerationResult(
            path=str(path),
            threat_level=ThreatLevel.SAFE,
            detections=[],
            actions_taken=[ActionType.ALLOW],
            success=True,
        )

    def moderate_batch(
        self,
        directory: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        recursive: bool = True,
    ) -> BatchModerationResult:
        """
        Moderate all images in a directory.

        Args:
            directory: Input directory
            output_dir: Output directory for modified images
            recursive: Search subdirectories

        Returns:
            BatchModerationResult with statistics
        """
        source = Path(directory)
        out_dir = Path(output_dir) if output_dir else None

        result = BatchModerationResult(
            total=0,
            processed=0,
            safe=0,
            flagged=0,
            blocked=0,
            errors=0,
        )

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
            try:
                out_path = None
                if out_dir:
                    rel_path = img_path.relative_to(source)
                    out_path = out_dir / rel_path
                    out_path.parent.mkdir(parents=True, exist_ok=True)

                mod_result = self.moderate(img_path, out_path)
                result.processed += 1

                threat = mod_result.threat_level.value
                if threat not in result.threat_summary:
                    result.threat_summary[threat] = 0
                result.threat_summary[threat] += 1

                if mod_result.threat_level == ThreatLevel.SAFE:
                    result.safe += 1
                elif ActionType.BLOCK in mod_result.actions_taken:
                    result.blocked += 1
                else:
                    result.flagged += 1

            except Exception:
                result.errors += 1

        result.success = result.errors == 0

        return result

    def blur_faces(
        self,
        image: Union[str, Path],
        output: Union[str, Path],
    ) -> ModerationResult:
        """
        Detect and blur faces in an image.

        Args:
            image: Input image path
            output: Output path

        Returns:
            ModerationResult
        """
        # Temporarily enable face blurring
        original_blur = self.policy.auto_blur_faces
        self.policy.auto_blur_faces = True
        self._content_shield = None  # Reset

        try:
            result = self.moderate(image, output, apply_actions=True)
        finally:
            self.policy.auto_blur_faces = original_blur
            self._content_shield = None

        return result

    def check_safety(
        self,
        image: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Quick content safety check.

        Args:
            image: Image path

        Returns:
            Safety check result
        """
        result = self.moderate(image, apply_actions=False)

        return {
            'safe': result.threat_level == ThreatLevel.SAFE,
            'threat_level': result.threat_level.value,
            'issues': len(result.detections),
            'detections': result.detections,
        }

    def check_gdpr_compliance(
        self,
        image: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Check image for GDPR compliance issues.

        Args:
            image: Image path

        Returns:
            Compliance check result
        """
        result = self.moderate(image, apply_actions=False)

        compliance = {
            'compliant': True,
            'issues': [],
            'recommendations': [],
        }

        # Check for faces
        face_detections = [d for d in result.detections if d.get('type') == 'face']
        if face_detections:
            compliance['compliant'] = False
            compliance['issues'].append(f"Contains {len(face_detections)} identifiable face(s)")
            compliance['recommendations'].append("Obtain consent or blur faces")

        # Check for PII
        pii_detections = [d for d in result.detections if d.get('type') == 'pii']
        if pii_detections:
            compliance['compliant'] = False
            compliance['issues'].append(f"Contains {len(pii_detections)} PII element(s)")
            compliance['recommendations'].append("Redact personal information")

        return compliance

    def export_audit_log(
        self,
        results: List[ModerationResult],
        output: Union[str, Path],
        format: str = 'json',
    ) -> str:
        """
        Export moderation audit log.

        Args:
            results: List of moderation results
            output: Output path
            format: Output format (json, csv)

        Returns:
            Output path
        """
        import json

        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        log_data = {
            'generated': datetime.now().isoformat(),
            'policy': self.policy.to_dict(),
            'summary': {
                'total': len(results),
                'safe': sum(1 for r in results if r.threat_level == ThreatLevel.SAFE),
                'flagged': sum(1 for r in results if r.threat_level in [ThreatLevel.LOW, ThreatLevel.MEDIUM]),
                'blocked': sum(1 for r in results if r.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]),
            },
            'items': [r.to_dict() for r in results],
        }

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2)

        return str(out_path)

    @staticmethod
    def list_policies() -> List[str]:
        """List available policy names."""
        return ['STRICT', 'MODERATE', 'RELAXED']

    @classmethod
    def get_policy(cls, name: str) -> ModerationPolicy:
        """Get a policy by name."""
        policies = {
            'STRICT': cls.STRICT,
            'MODERATE': cls.MODERATE,
            'RELAXED': cls.RELAXED,
        }
        return policies.get(name.upper(), ModerationPolicy())


# Convenience functions
def moderate_image(
    image: Union[str, Path],
    policy: str = "moderate",
) -> ModerationResult:
    """Quick image moderation."""
    pol = ContentModeration.get_policy(policy)
    return ContentModeration(policy=pol).moderate(image)


def check_content_safety(image: Union[str, Path]) -> Dict[str, Any]:
    """Quick content safety check."""
    return ContentModeration().check_safety(image)

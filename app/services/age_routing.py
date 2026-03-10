"""Age-based routing for reference subject classification."""
import logging
from enum import Enum
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)


class SubjectAgeClass(str, Enum):
    """Subject age classification for routing."""

    ADULT = "adult"
    TEEN_OR_MINOR = "teen_or_minor"
    UNKNOWN = "unknown"


class AgeRouter:
    """Routes reference images based on subject age classification."""

    def __init__(self, classifier_type: str = "local"):
        """
        Initialize age router.

        Args:
            classifier_type: Type of classifier ("local" or "external")
        """
        self.classifier_type = classifier_type
        self._classifier = None

    def classify_reference_subject(self, reference_image: Image.Image) -> SubjectAgeClass:
        """
        Classify reference subject age for routing.

        Args:
            reference_image: Reference face/photo image

        Returns:
            SubjectAgeClass: ADULT, TEEN_OR_MINOR, or UNKNOWN
        """
        if self.classifier_type == "local":
            return self._classify_local(reference_image)
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type}")

    def _classify_local(self, image: Image.Image) -> SubjectAgeClass:
        """
        Local classification using heuristics.

        This is a simple heuristic-based classifier. For production,
        consider using a proper age estimation model.

        Args:
            image: Reference image

        Returns:
            SubjectAgeClass
        """
        try:
            from app.services.detection.dog_detector import DogDetector

            # Detect human face/head
            detector = DogDetector()
            detection = detector.detect(image, entity_type="human")

            # Use head size ratio as a heuristic
            # Teens/minors typically have larger head-to-body ratio
            # This is a simplified heuristic - in production use proper age estimation
            image_area = image.width * image.height
            head_area = detection.head_bbox.area if detection.head_bbox else 0
            head_ratio = head_area / image_area if image_area > 0 else 0

            # Heuristic: larger head ratio might indicate younger subject
            # But this is very rough - should use proper age estimation model
            # For now, we'll be conservative and return UNKNOWN
            logger.info(
                f"Age classification heuristic: head_ratio={head_ratio:.3f}, "
                f"confidence={detection.confidence:.2f}"
            )

            # Conservative approach: return UNKNOWN for now
            # In production, integrate proper age estimation model here
            logger.warning(
                "Using conservative age classification - returning UNKNOWN. "
                "Consider integrating proper age estimation model for production."
            )
            return SubjectAgeClass.UNKNOWN

        except Exception as e:
            logger.warning(f"Age classification failed: {e}, returning UNKNOWN")
            return SubjectAgeClass.UNKNOWN

"""Crop planning service."""
import logging

from PIL import Image

from app.core.exceptions import CompositingError
from app.models.schemas import BBox, DetectionResult

logger = logging.getLogger(__name__)


class CropPlanner:
    """Plans crop for head-and-shoulders extraction."""

    def plan_crop(
        self,
        image: Image.Image,
        detection: DetectionResult,
        crop_config: dict,
    ) -> BBox:
        """
        Plan crop for head-and-shoulders extraction.

        Args:
            image: Source image
            detection: Detection result
            crop_config: Crop configuration from template

        Returns:
            Crop bounding box
        """
        try:
            # Use head bbox as base, or dog bbox if head not available
            base_bbox = detection.head_bbox or detection.dog_bbox

            # Get expansion factors
            expansion = crop_config.get("crop_expansion", {})
            expand_left = expansion.get("left", 0.18)
            expand_right = expansion.get("right", 0.18)
            expand_top = expansion.get("top", 0.28)
            expand_bottom = expansion.get("bottom", 0.20)

            # Calculate expanded bbox
            width = base_bbox.width
            height = base_bbox.height

            x1 = base_bbox.x1 - width * expand_left
            x2 = base_bbox.x2 + width * expand_right
            y1 = base_bbox.y1 - height * expand_top
            y2 = base_bbox.y2 + height * expand_bottom

            # Clamp to image bounds
            img_width, img_height = image.size
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(x1, min(x2, img_width))
            y2 = max(y1, min(y2, img_height))

            crop_bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)

            logger.debug(f"Crop planned: {crop_bbox.to_dict()}")
            return crop_bbox

        except Exception as e:
            logger.error(f"Crop planning error: {e}", exc_info=True)
            raise CompositingError(f"Ошибка планирования кропа: {e}") from e

    def extract_crop(self, image: Image.Image, crop_bbox: BBox) -> Image.Image:
        """
        Extract crop from image.

        Args:
            image: Source image
            crop_bbox: Crop bounding box

        Returns:
            Cropped image
        """
        x1, y1, x2, y2 = int(crop_bbox.x1), int(crop_bbox.y1), int(crop_bbox.x2), int(crop_bbox.y2)
        return image.crop((x1, y1, x2, y2))

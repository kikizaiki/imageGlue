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
        entity_type: str = "dog",
    ) -> BBox:
        """
        Plan crop for head-and-shoulders extraction.

        Args:
            image: Source image
            detection: Detection result
            crop_config: Crop configuration from template
            entity_type: Type of entity ("dog" or "human")

        Returns:
            Crop bounding box
        """
        try:
            # Для человека используем entity_bbox (весь человек) как базу,
            # так как нужно захватить голову + плечи + верх груди
            # Для собаки используем head_bbox как базу (голова собаки обычно больше относительно тела)
            if entity_type == "human":
                # Для человека: используем entity_bbox и расширяем его, чтобы захватить голову + плечи
                base_bbox = detection.dog_bbox  # dog_bbox содержит entity_bbox
                logger.debug(f"Using entity_bbox for human crop: {base_bbox.to_dict()}")
            else:
                # Для собаки: используем head_bbox как базу
                base_bbox = detection.head_bbox or detection.dog_bbox
                logger.debug(f"Using head_bbox for dog crop: {base_bbox.to_dict()}")

            # Get expansion factors
            expansion = crop_config.get("crop_expansion", {})
            expand_left = expansion.get("left", 0.18)
            expand_right = expansion.get("right", 0.18)
            expand_top = expansion.get("top", 0.28)
            expand_bottom = expansion.get("bottom", 0.20)

            # Для человека увеличиваем расширение вниз, чтобы захватить плечи и верх груди
            if entity_type == "human":
                # Если bottom expansion маленький, увеличиваем его
                if expand_bottom < 0.3:
                    expand_bottom = 0.4  # Захватываем больше вниз для плеч
                # Уменьшаем top expansion, так как entity_bbox уже включает голову
                if expand_top > 0.2:
                    expand_top = 0.15  # Меньше расширяем вверх

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

            logger.info(
                f"Crop planned for {entity_type}: base={base_bbox.to_dict()}, "
                f"expansion=(L:{expand_left:.2f}, R:{expand_right:.2f}, T:{expand_top:.2f}, B:{expand_bottom:.2f}), "
                f"crop={crop_bbox.to_dict()}"
            )
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

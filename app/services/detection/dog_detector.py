"""Dog detection service."""
import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.core.config import settings
from app.core.exceptions import DetectionError
from app.models.schemas import BBox, DetectionResult

logger = logging.getLogger(__name__)


class DogDetector:
    """Detects dogs, humans and their heads in images."""

    def __init__(self):
        """Initialize detector."""
        self._model = None

    def _load_model(self):
        """Lazy load YOLO model."""
        if self._model is None:
            try:
                from ultralytics import YOLO

                # Use YOLOv8n for speed, can be upgraded to YOLOv8s/m for better accuracy
                self._model = YOLO("yolov8n.pt")
                logger.info("YOLO model loaded")
            except ImportError:
                raise DetectionError(
                    "ultralytics not installed. Install with: pip install ultralytics"
                )
            except Exception as e:
                raise DetectionError(f"Failed to load YOLO model: {e}")

        return self._model

    def detect(self, image: Image.Image, entity_type: str = "dog") -> DetectionResult:
        """
        Detect entity (dog or human) and head in image.

        Args:
            image: PIL Image
            entity_type: Type of entity to detect ("dog" or "human")

        Returns:
            DetectionResult

        Raises:
            DetectionError: If detection fails
        """
        try:
            model = self._load_model()

            # Convert to numpy
            img_array = np.array(image.convert("RGB"))

            # COCO classes: 0 = person, 16 = dog
            entity_class = 16 if entity_type == "dog" else 0
            entity_name_ru = "собака" if entity_type == "dog" else "человек"
            entity_name_ru_genitive = "собаки" if entity_type == "dog" else "человека"
            
            logger.info(
                f"Detecting {entity_type} (COCO class {entity_class}) "
                f"with confidence threshold {settings.DETECTION_CONFIDENCE_THRESHOLD}"
            )
            
            # Run detection
            results = model.predict(
                img_array,
                conf=settings.DETECTION_CONFIDENCE_THRESHOLD,
                classes=[entity_class],
                verbose=False,
            )
            
            logger.debug(f"YOLO detection results: {len(results)} result(s)")

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    logger.debug(f"No boxes found in YOLO result for {entity_type}")
                    continue

                logger.debug(f"Found {len(boxes)} detection(s) for {entity_type} (class {entity_class})")
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    detected_class = int(box.cls[0].cpu().numpy()) if hasattr(box, 'cls') and box.cls is not None else entity_class
                    
                    logger.debug(
                        f"Detection: class={detected_class} (expected {entity_class}), "
                        f"confidence={confidence:.2f}, bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})"
                    )

                    entity_bbox = BBox(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                    )

                    # Estimate head bbox (upper portion of entity bbox)
                    head_bbox = self._estimate_head_bbox(entity_bbox, img_array.shape, entity_type)

                    detections.append(
                        {
                            "entity_bbox": entity_bbox,
                            "head_bbox": head_bbox,
                            "confidence": confidence,
                            "detected_class": detected_class,
                        }
                    )

            if not detections:
                raise DetectionError(f"{entity_name_ru.capitalize()} не обнаружен на изображении")

            # Select best detection (highest confidence, then largest area)
            best = max(
                detections,
                key=lambda d: (d["confidence"], d["entity_bbox"].area),
            )

            # Check if entity is large enough
            image_area = image.width * image.height
            entity_area_ratio = best["entity_bbox"].area / image_area

            if entity_area_ratio < settings.MIN_DOG_AREA_RATIO:
                raise DetectionError(
                    f"{entity_name_ru.capitalize()} слишком маленький на фото (занимает {entity_area_ratio*100:.1f}% кадра, "
                    f"минимум {settings.MIN_DOG_AREA_RATIO*100:.1f}%). "
                    f"Загрузите фото, где {entity_name_ru_genitive} снят крупнее."
                )

            # Estimate orientation
            orientation = self._estimate_orientation(best["entity_bbox"], img_array.shape)

            result = DetectionResult(
                dog_bbox=best["entity_bbox"],  # Используем entity_bbox как dog_bbox для обратной совместимости
                head_bbox=best["head_bbox"],
                confidence=best["confidence"],
                orientation=orientation,
            )

            logger.info(
                f"Detection ({entity_type}): confidence={result.confidence:.2f}, "
                f"bbox={result.dog_bbox.to_dict()}, "
                f"head_bbox={result.head_bbox.to_dict() if result.head_bbox else None}"
            )

            return result

        except DetectionError:
            raise
        except Exception as e:
            logger.error(f"Detection error: {e}", exc_info=True)
            raise DetectionError(f"Ошибка детекции: {e}") from e

    def _estimate_head_bbox(self, entity_bbox: BBox, image_shape: tuple, entity_type: str = "dog") -> BBox:
        """
        Estimate head bounding box from entity bbox.

        Args:
            entity_bbox: Entity bounding box (dog or human)
            image_shape: Image shape (height, width, channels)
            entity_type: Type of entity ("dog" or "human")

        Returns:
            Estimated head bounding box
        """
        # Head parameters differ for dogs and humans
        if entity_type == "human":
            # For humans, head is typically in upper 20-25% of body bbox
            head_height_ratio = 0.25
            head_width_ratio = 0.5
        else:
            # For dogs, head is typically in upper 40% of dog bbox
            head_height_ratio = 0.4
            head_width_ratio = 0.6

        entity_width = entity_bbox.width
        entity_height = entity_bbox.height

        head_width = entity_width * head_width_ratio
        head_height = entity_height * head_height_ratio

        # Center horizontally, top-aligned
        head_x1 = entity_bbox.center_x - head_width / 2
        head_y1 = entity_bbox.y1
        head_x2 = head_x1 + head_width
        head_y2 = head_y1 + head_height

        # Clamp to image bounds
        img_height, img_width = image_shape[:2]
        head_x1 = max(0, min(head_x1, img_width))
        head_y1 = max(0, min(head_y1, img_height))
        head_x2 = max(head_x1, min(head_x2, img_width))
        head_y2 = max(head_y1, min(head_y2, img_height))

        return BBox(x1=head_x1, y1=head_y1, x2=head_x2, y2=head_y2)

    def _estimate_orientation(self, dog_bbox: BBox, image_shape: tuple) -> str:
        """
        Estimate dog orientation.

        Args:
            dog_bbox: Dog bounding box
            image_shape: Image shape

        Returns:
            Orientation string
        """
        # Simple heuristic: compare width to height
        aspect_ratio = dog_bbox.width / dog_bbox.height

        if aspect_ratio > 1.3:
            return "side"
        elif aspect_ratio < 0.8:
            return "front"
        else:
            return "front"  # Default to front

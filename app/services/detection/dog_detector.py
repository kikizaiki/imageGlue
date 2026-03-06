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
    """Detects dogs and dog heads in images."""

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

    def detect(self, image: Image.Image) -> DetectionResult:
        """
        Detect dog and head in image.

        Args:
            image: PIL Image

        Returns:
            DetectionResult

        Raises:
            DetectionError: If detection fails
        """
        try:
            model = self._load_model()

            # Convert to numpy
            img_array = np.array(image.convert("RGB"))

            # Run detection (class 16 = dog in COCO)
            results = model.predict(
                img_array,
                conf=settings.DETECTION_CONFIDENCE_THRESHOLD,
                classes=[16],  # dog class
                verbose=False,
            )

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())

                    dog_bbox = BBox(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                    )

                    # Estimate head bbox (upper portion of dog bbox)
                    head_bbox = self._estimate_head_bbox(dog_bbox, img_array.shape)

                    detections.append(
                        {
                            "dog_bbox": dog_bbox,
                            "head_bbox": head_bbox,
                            "confidence": confidence,
                        }
                    )

            if not detections:
                raise DetectionError("Собака не обнаружена на изображении")

            # Select best detection (highest confidence, then largest area)
            best = max(
                detections,
                key=lambda d: (d["confidence"], d["dog_bbox"].area),
            )

            # Check if dog is large enough
            image_area = image.width * image.height
            dog_area_ratio = best["dog_bbox"].area / image_area

            if dog_area_ratio < settings.MIN_DOG_AREA_RATIO:
                raise DetectionError(
                    f"Собака слишком маленькая на фото (занимает {dog_area_ratio*100:.1f}% кадра, "
                    f"минимум {settings.MIN_DOG_AREA_RATIO*100:.1f}%). "
                    "Загрузите фото, где собака снята крупнее."
                )

            # Estimate orientation
            orientation = self._estimate_orientation(best["dog_bbox"], img_array.shape)

            result = DetectionResult(
                dog_bbox=best["dog_bbox"],
                head_bbox=best["head_bbox"],
                confidence=best["confidence"],
                orientation=orientation,
            )

            logger.info(
                f"Detection: confidence={result.confidence:.2f}, "
                f"dog_bbox={result.dog_bbox.to_dict()}, "
                f"head_bbox={result.head_bbox.to_dict() if result.head_bbox else None}"
            )

            return result

        except DetectionError:
            raise
        except Exception as e:
            logger.error(f"Detection error: {e}", exc_info=True)
            raise DetectionError(f"Ошибка детекции: {e}") from e

    def _estimate_head_bbox(self, dog_bbox: BBox, image_shape: tuple) -> BBox:
        """
        Estimate head bounding box from dog bbox.

        Args:
            dog_bbox: Dog bounding box
            image_shape: Image shape (height, width, channels)

        Returns:
            Estimated head bounding box
        """
        # Head is typically in upper 40% of dog bbox, centered horizontally
        head_height_ratio = 0.4
        head_width_ratio = 0.6

        dog_width = dog_bbox.width
        dog_height = dog_bbox.height

        head_width = dog_width * head_width_ratio
        head_height = dog_height * head_height_ratio

        # Center horizontally, top-aligned
        head_x1 = dog_bbox.center_x - head_width / 2
        head_y1 = dog_bbox.y1
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

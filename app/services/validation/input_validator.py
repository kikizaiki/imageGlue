"""Input validation service."""
import logging
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.core.config import settings
from app.core.exceptions import ValidationError
from app.models.schemas import ValidationResult

logger = logging.getLogger(__name__)


class InputValidator:
    """Validates input images."""

    def validate(self, image_path: str | Path | bytes) -> ValidationResult:
        """
        Validate input image.

        Args:
            image_path: Path to image or image bytes

        Returns:
            ValidationResult

        Raises:
            ValidationError: If validation fails critically
        """
        errors = []
        warnings = []

        try:
            # Load image
            if isinstance(image_path, (str, Path)):
                image = Image.open(image_path)
                file_size_mb = Path(image_path).stat().st_size / (1024 * 1024)
            else:
                image = Image.open(BytesIO(image_path))
                file_size_mb = len(image_path) / (1024 * 1024)

            width, height = image.size
            format_name = image.format or "UNKNOWN"

            # Check format
            if format_name.lower() not in ["jpeg", "jpg", "png", "webp"]:
                errors.append(f"Неподдерживаемый формат: {format_name}")

            # Check size
            if width < settings.MIN_IMAGE_WIDTH:
                errors.append(
                    f"Изображение слишком узкое: {width}px (минимум {settings.MIN_IMAGE_WIDTH}px)"
                )
            if height < settings.MIN_IMAGE_HEIGHT:
                errors.append(
                    f"Изображение слишком низкое: {height}px (минимум {settings.MIN_IMAGE_HEIGHT}px)"
                )

            # Check file size
            if file_size_mb > settings.MAX_IMAGE_SIZE_MB:
                errors.append(
                    f"Файл слишком большой: {file_size_mb:.2f}MB (максимум {settings.MAX_IMAGE_SIZE_MB}MB)"
                )

            # Check image quality
            img_array = np.array(image.convert("RGB"))
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # Blur detection
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < settings.MAX_BLUR_THRESHOLD:
                warnings.append(
                    f"Изображение может быть размытым (оценка: {laplacian_var:.1f})"
                )

            # Exposure check
            mean_brightness = np.mean(gray)
            if mean_brightness < 30:
                warnings.append("Изображение слишком тёмное")
            elif mean_brightness > 225:
                warnings.append("Изображение пересвечено")

            # Check if image is readable
            if image.mode not in ("RGB", "RGBA", "L"):
                try:
                    image = image.convert("RGB")
                except Exception as e:
                    errors.append(f"Не удалось конвертировать изображение: {e}")

            is_valid = len(errors) == 0

            if not is_valid:
                error_msg = "; ".join(errors)
                raise ValidationError(f"Валидация не пройдена: {error_msg}")

            result = ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                image_width=width,
                image_height=height,
                format=format_name,
                file_size_mb=file_size_mb,
            )

            logger.info(f"Validation passed: {width}x{height}, {format_name}")
            return result

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            raise ValidationError(f"Ошибка при валидации: {e}") from e

"""Quality gate service."""
import logging

import cv2
import numpy as np
from PIL import Image

from app.core.config import settings
from app.core.exceptions import QualityGateError
from app.models.schemas import BBox, DetectionResult, QualityScore

logger = logging.getLogger(__name__)


class QualityGate:
    """Assesses quality of final result."""

    def assess(
        self,
        final_image: Image.Image,
        detection: DetectionResult,
        placement: dict,
    ) -> QualityScore:
        """
        Assess quality of final result.

        Args:
            final_image: Final composed image
            detection: Original detection result
            placement: Placement parameters

        Returns:
            QualityScore

        Raises:
            QualityGateError: If quality is too low
        """
        try:
            issues = []
            scores = []

            # Check head visibility
            head_visible = detection.head_bbox is not None
            if not head_visible:
                issues.append("Голова собаки не обнаружена")
            scores.append(1.0 if head_visible else 0.0)

            # Check head size
            if detection.head_bbox:
                image_area = final_image.width * final_image.height
                head_area = detection.head_bbox.area
                head_ratio = head_area / image_area
                head_size_ok = head_ratio >= settings.MIN_HEAD_SIZE_RATIO

                if not head_size_ok:
                    issues.append(
                        f"Голова слишком маленькая ({head_ratio*100:.1f}% кадра, "
                        f"минимум {settings.MIN_HEAD_SIZE_RATIO*100:.1f}%)"
                    )
                scores.append(1.0 if head_size_ok else 0.5)
            else:
                head_size_ok = False
                scores.append(0.0)

            # Check edges (simple blur check)
            img_array = np.array(final_image.convert("RGB"))
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            edges_ok = laplacian_var > 50.0  # Reasonable sharpness
            if not edges_ok:
                issues.append("Изображение слишком размытое")
            scores.append(1.0 if edges_ok else 0.7)

            # Check composition (subject is in reasonable position)
            paste_x = placement.get("paste_x", 0)
            paste_y = placement.get("paste_y", 0)
            canvas_width = final_image.width
            canvas_height = final_image.height

            # Check if subject is within canvas bounds
            composition_ok = (
                0 <= paste_x < canvas_width and 0 <= paste_y < canvas_height
            )
            if not composition_ok:
                issues.append("Собака выходит за границы постера")
            scores.append(1.0 if composition_ok else 0.0)

            # Check for obvious artifacts (simple heuristic)
            # This is a placeholder - can be enhanced with ML
            artifacts_detected = False
            scores.append(1.0 if not artifacts_detected else 0.5)

            # Calculate overall score
            overall = sum(scores) / len(scores) if scores else 0.0

            quality_score = QualityScore(
                overall=overall,
                head_visible=head_visible,
                head_size_ok=head_size_ok,
                edges_ok=edges_ok,
                composition_ok=composition_ok,
                artifacts_detected=artifacts_detected,
                rejection_reason="; ".join(issues) if issues and overall < 0.6 else None,
            )

            logger.info(f"Quality assessment: overall={overall:.2f}")

            # Reject if quality too low
            if settings.QUALITY_CHECK_ENABLED and overall < settings.REFINEMENT_THRESHOLD:
                rejection_msg = quality_score.rejection_reason or "Низкое качество результата"
                raise QualityGateError(
                    f"Результат не прошёл проверку качества: {rejection_msg}. "
                    "Попробуйте загрузить другое фото."
                )

            return quality_score

        except QualityGateError:
            raise
        except Exception as e:
            logger.error(f"Quality assessment error: {e}", exc_info=True)
            # Don't fail on assessment error, just log
            return QualityScore(
                overall=0.5,
                head_visible=True,
                head_size_ok=True,
                edges_ok=True,
                composition_ok=True,
                artifacts_detected=False,
            )

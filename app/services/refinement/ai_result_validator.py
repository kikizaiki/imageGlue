"""Validator for AI-generated results to detect no-op changes."""
import logging
from typing import Any

import numpy as np
from PIL import Image

from app.core.exceptions import CompositingError

logger = logging.getLogger(__name__)

# Thresholds for validation
MIN_FULL_IMAGE_DIFF_SCORE = 0.05  # 5% difference required for full image
MIN_TARGET_REGION_DIFF_SCORE = 0.10  # 10% difference required for target region
MIN_CHANGED_PIXELS_RATIO = 0.05  # 5% of pixels must change
MIN_MEAN_ABS_DIFF = 10.0  # Mean absolute difference in pixel values


class AIResultValidator:
    """Validates AI-generated results to detect no-op changes."""

    def __init__(
        self,
        min_full_diff: float = MIN_FULL_IMAGE_DIFF_SCORE,
        min_target_diff: float = MIN_TARGET_REGION_DIFF_SCORE,
        min_changed_pixels: float = MIN_CHANGED_PIXELS_RATIO,
        min_mean_abs_diff: float = MIN_MEAN_ABS_DIFF,
    ):
        """
        Initialize validator.

        Args:
            min_full_diff: Minimum full image diff score (0.0-1.0)
            min_target_diff: Minimum target region diff score (0.0-1.0)
            min_changed_pixels: Minimum ratio of changed pixels (0.0-1.0)
            min_mean_abs_diff: Minimum mean absolute difference in pixel values
        """
        self.min_full_diff = min_full_diff
        self.min_target_diff = min_target_diff
        self.min_changed_pixels = min_changed_pixels
        self.min_mean_abs_diff = min_mean_abs_diff

    def validate_ai_result(
        self,
        original_image: Image.Image,
        ai_result: Image.Image,
        target_region: dict[str, int] | None = None,
        debug: bool = False,
        storage: Any = None,
    ) -> dict[str, Any]:
        """
        Validate AI result against original image.

        Args:
            original_image: Original poster/image
            ai_result: AI-generated result
            target_region: Optional target region dict with x, y, width, height
            debug: Whether to save debug artifacts
            storage: Storage instance for debug artifacts

        Returns:
            Validation result dict with:
            - accepted: bool
            - rejection_reason: str | None
            - metrics: dict with diff scores
        """
        logger.info("🔍 Validating AI result against original image...")

        # Ensure same size
        if original_image.size != ai_result.size:
            logger.warning(
                f"⚠️  Size mismatch: original={original_image.size}, ai_result={ai_result.size}. "
                f"Resizing AI result to match original."
            )
            ai_result = ai_result.resize(original_image.size, Image.Resampling.LANCZOS)

        # Convert to RGB if needed
        original_rgb = original_image.convert("RGB")
        ai_result_rgb = ai_result.convert("RGB")

        # Calculate full image diff
        full_metrics = self._calculate_diff_metrics(original_rgb, ai_result_rgb)
        logger.info(
            f"📊 Full image diff metrics:\n"
            f"   - Mean absolute diff: {full_metrics['mean_abs_diff']:.2f}\n"
            f"   - Changed pixels ratio: {full_metrics['changed_pixels_ratio']:.4f}\n"
            f"   - Diff score: {full_metrics['diff_score']:.4f}"
        )

        # Calculate target region diff if provided
        target_metrics = None
        if target_region:
            # Log target region coordinates before calculation
            logger.info(
                f"🎯 Target region coordinates:\n"
                f"   - Original image size: {original_image.size}\n"
                f"   - AI result size: {ai_result.size}\n"
                f"   - Target box: x={target_region.get('x', 0)}, y={target_region.get('y', 0)}, "
                f"w={target_region.get('width', 0)}, h={target_region.get('height', 0)}\n"
                f"   - Config field used: face_region (from template ai_integration.face_region)"
            )
            
            try:
                target_metrics = self._calculate_region_diff_metrics(
                    original_rgb, ai_result_rgb, target_region
                )
                if target_metrics:
                    logger.info(
                        f"📊 Target region diff metrics:\n"
                        f"   - Mean absolute diff: {target_metrics['mean_abs_diff']:.2f}\n"
                        f"   - Changed pixels ratio: {target_metrics['changed_pixels_ratio']:.4f}\n"
                        f"   - Diff score: {target_metrics['diff_score']:.4f}"
                    )
                else:
                    logger.warning("⚠️  Target region metrics calculation returned None")
            except Exception as e:
                logger.warning(f"⚠️  Failed to calculate target region metrics: {e}")
                target_metrics = None

        # Save debug artifacts
        if debug and storage:
            self._save_debug_artifacts(
                original_rgb, ai_result_rgb, full_metrics, target_metrics, target_region, storage
            )

        # Validate against thresholds
        validation_result = self._validate_against_thresholds(
            full_metrics, target_metrics, target_region is not None
        )

        # Safe formatting for target region diff score
        target_diff_score_str = "N/A"
        if target_metrics and isinstance(target_metrics, dict):
            target_diff_score = target_metrics.get('diff_score', 0.0)
            target_diff_score_str = f"{target_diff_score:.4f}"
        elif target_region:
            target_diff_score_str = "N/A (calculation failed)"
        
        min_target_diff_str = f"{self.min_target_diff}" if target_region else "N/A"
        
        logger.info(
            f"✅ Validation result:\n"
            f"   - Accepted: {validation_result['accepted']}\n"
            f"   - Rejection reason: {validation_result.get('rejection_reason', 'None')}\n"
            f"   - Full image diff score: {full_metrics['diff_score']:.4f} (min: {self.min_full_diff})\n"
            f"   - Target region diff score: {target_diff_score_str} (min: {min_target_diff_str})"
        )

        return validation_result

    def _calculate_diff_metrics(
        self, original: Image.Image, result: Image.Image
    ) -> dict[str, float]:
        """Calculate diff metrics between two images."""
        orig_array = np.array(original, dtype=np.float32)
        result_array = np.array(result, dtype=np.float32)

        # Absolute difference
        abs_diff = np.abs(orig_array - result_array)
        mean_abs_diff = np.mean(abs_diff)

        # Changed pixels (difference > threshold)
        pixel_diff_threshold = 5  # Consider pixel changed if diff > 5 in any channel
        changed_pixels = np.any(abs_diff > pixel_diff_threshold, axis=2)
        changed_pixels_count = np.sum(changed_pixels)
        total_pixels = changed_pixels.size
        changed_pixels_ratio = changed_pixels_count / total_pixels if total_pixels > 0 else 0.0

        # Diff score (normalized 0.0-1.0)
        # Based on mean absolute difference normalized to 0-255 range
        diff_score = min(1.0, mean_abs_diff / 255.0)

        return {
            "mean_abs_diff": float(mean_abs_diff),
            "changed_pixels_ratio": float(changed_pixels_ratio),
            "diff_score": float(diff_score),
            "changed_pixels_count": int(changed_pixels_count),
            "total_pixels": int(total_pixels),
        }

    def _calculate_region_diff_metrics(
        self, original: Image.Image, result: Image.Image, region: dict[str, int]
    ) -> dict[str, float] | None:
        """Calculate diff metrics for a specific region."""
        x = region.get("x", 0)
        y = region.get("y", 0)
        width = region.get("width", original.width)
        height = region.get("height", original.height)
        
        original_x = x
        original_y = y
        original_width = width
        original_height = height

        # Clip region to image bounds
        x = max(0, min(x, original.width - 1))
        y = max(0, min(y, original.height - 1))
        width = min(width, original.width - x)
        height = min(height, original.height - y)
        
        # Log if clipping occurred
        if (x != original_x or y != original_y or 
            width != original_width or height != original_height):
            logger.warning(
                f"⚠️  Target region clipped to image bounds:\n"
                f"   - Original: x={original_x}, y={original_y}, w={original_width}, h={original_height}\n"
                f"   - Clipped: x={x}, y={y}, w={width}, h={height}\n"
                f"   - Image size: {original.size}"
            )

        if width <= 0 or height <= 0:
            logger.warning(
                f"⚠️  Invalid or empty region after clipping: x={x}, y={y}, w={width}, h={height}. "
                f"Original region: x={original_x}, y={original_y}, w={original_width}, h={original_height}, "
                f"Image size: {original.size}"
            )
            return {
                "mean_abs_diff": 0.0,
                "changed_pixels_ratio": 0.0,
                "diff_score": 0.0,
                "changed_pixels_count": 0,
                "total_pixels": 0,
            }

        try:
            # Crop regions
            orig_region = original.crop((x, y, x + width, y + height))
            result_region = result.crop((x, y, x + width, y + height))
            
            if orig_region.size != result_region.size:
                logger.warning(
                    f"⚠️  Region crop size mismatch: orig={orig_region.size}, result={result_region.size}"
                )
                return None

            return self._calculate_diff_metrics(orig_region, result_region)
        except Exception as e:
            logger.warning(f"⚠️  Failed to crop or calculate region metrics: {e}")
            return None

    def _validate_against_thresholds(
        self,
        full_metrics: dict[str, float],
        target_metrics: dict[str, float] | None,
        has_target_region: bool,
    ) -> dict[str, Any]:
        """Validate metrics against thresholds."""
        rejection_reasons = []

        # Check full image diff score
        if full_metrics["diff_score"] < self.min_full_diff:
            rejection_reasons.append(
                f"Full image diff score {full_metrics['diff_score']:.4f} < {self.min_full_diff}"
            )

        # Check mean absolute difference
        if full_metrics["mean_abs_diff"] < self.min_mean_abs_diff:
            rejection_reasons.append(
                f"Mean absolute diff {full_metrics['mean_abs_diff']:.2f} < {self.min_mean_abs_diff}"
            )

        # Check changed pixels ratio
        if full_metrics["changed_pixels_ratio"] < self.min_changed_pixels:
            rejection_reasons.append(
                f"Changed pixels ratio {full_metrics['changed_pixels_ratio']:.4f} < {self.min_changed_pixels}"
            )

        # Check target region if provided
        if has_target_region and target_metrics:
            if target_metrics["diff_score"] < self.min_target_diff:
                rejection_reasons.append(
                    f"Target region diff score {target_metrics['diff_score']:.4f} < {self.min_target_diff}"
                )

        accepted = len(rejection_reasons) == 0
        rejection_reason = "; ".join(rejection_reasons) if rejection_reasons else None

        return {
            "accepted": accepted,
            "rejection_reason": rejection_reason,
            "metrics": {
                "full_image": full_metrics,
                "target_region": target_metrics,
            },
        }

    def _save_debug_artifacts(
        self,
        original: Image.Image,
        result: Image.Image,
        full_metrics: dict[str, float],
        target_metrics: dict[str, float] | None,
        target_region: dict[str, int] | None,
        storage: Any,
    ) -> None:
        """Save debug artifacts for validation."""
        try:
            # Save original and result
            storage.save_debug(original, "validation_00_original_poster.png")
            storage.save_debug(result, "validation_01_ai_returned_result.png")

            # Calculate and save absolute diff
            orig_array = np.array(original, dtype=np.float32)
            result_array = np.array(result, dtype=np.float32)
            abs_diff = np.abs(orig_array - result_array)
            # Normalize to 0-255 range for visualization
            abs_diff_normalized = np.clip(abs_diff * 10, 0, 255).astype(np.uint8)
            diff_image = Image.fromarray(abs_diff_normalized)
            storage.save_debug(diff_image, "validation_02_absolute_diff_full.png")

            # Save target region diff if provided
            if target_region and target_metrics:
                x = target_region.get("x", 0)
                y = target_region.get("y", 0)
                width = target_region.get("width", original.width)
                height = target_region.get("height", original.height)

                # Ensure region is within bounds
                x = max(0, min(x, original.width))
                y = max(0, min(y, original.height))
                width = min(width, original.width - x)
                height = min(height, original.height - y)

                if width > 0 and height > 0:
                    orig_region = original.crop((x, y, x + width, y + height))
                    result_region = result.crop((x, y, x + width, y + height))

                    orig_array = np.array(orig_region, dtype=np.float32)
                    result_array = np.array(result_region, dtype=np.float32)
                    abs_diff = np.abs(orig_array - result_array)
                    abs_diff_normalized = np.clip(abs_diff * 10, 0, 255).astype(np.uint8)
                    diff_region_image = Image.fromarray(abs_diff_normalized)
                    storage.save_debug(
                        diff_region_image, "validation_03_absolute_diff_target_region.png"
                    )

            logger.debug("Saved validation debug artifacts")
        except Exception as e:
            logger.warning(f"Failed to save validation debug artifacts: {e}")

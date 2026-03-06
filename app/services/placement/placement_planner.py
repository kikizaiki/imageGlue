"""Placement planning service."""
import logging
from typing import Any

from app.models.schemas import BBox

logger = logging.getLogger(__name__)


class PlacementPlanner:
    """Plans placement of subject in template."""

    def plan_placement(
        self,
        subject_width: int,
        subject_height: int,
        insert_zone: dict[str, Any],
        placement_config: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Plan placement of subject in insert zone.

        Args:
            subject_width: Subject image width
            subject_height: Subject image height
            insert_zone: Insert zone definition
            placement_config: Placement configuration

        Returns:
            Placement parameters dict
        """
        try:
            zone_x = insert_zone["x"]
            zone_y = insert_zone["y"]
            zone_width = insert_zone["width"]
            zone_height = insert_zone["height"]

            # Apply padding
            padding = placement_config.get("padding", {})
            pad_left = int(zone_width * padding.get("left", 0.06))
            pad_right = int(zone_width * padding.get("right", 0.06))
            pad_top = int(zone_height * padding.get("top", 0.08))
            pad_bottom = int(zone_height * padding.get("bottom", 0.04))

            target_width = zone_width - pad_left - pad_right
            target_height = zone_height - pad_top - pad_bottom

            # Calculate scale (contain mode)
            scale_w = target_width / subject_width
            scale_h = target_height / subject_height
            scale = min(scale_w, scale_h)

            # Apply scale multiplier
            scale_multiplier = placement_config.get("scale_multiplier", 1.0)
            scale *= scale_multiplier

            # Clamp scale
            min_scale = placement_config.get("min_scale", 0.7)
            max_scale = placement_config.get("max_scale", 2.2)
            scale = max(min_scale, min(max_scale, scale))

            # Calculate scaled dimensions
            scaled_width = int(subject_width * scale)
            scaled_height = int(subject_height * scale)

            # Calculate position (centered with bias)
            horizontal_bias = placement_config.get("horizontal_bias", 0.0)
            vertical_bias = placement_config.get("vertical_bias", -0.02)

            paste_x = zone_x + pad_left + (target_width - scaled_width) // 2
            paste_y = zone_y + pad_top + (target_height - scaled_height) // 2

            # Apply biases
            max_h_bias = target_width // 2
            max_v_bias = target_height // 2
            paste_x += int(horizontal_bias * max_h_bias)
            paste_y += int(vertical_bias * max_v_bias)

            result = {
                "scale": scale,
                "paste_x": int(paste_x),
                "paste_y": int(paste_y),
                "scaled_width": scaled_width,
                "scaled_height": scaled_height,
                "target_width": target_width,
                "target_height": target_height,
            }

            logger.debug(f"Placement planned: {result}")
            return result

        except Exception as e:
            logger.error(f"Placement planning error: {e}", exc_info=True)
            raise

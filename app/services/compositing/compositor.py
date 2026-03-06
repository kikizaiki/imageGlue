"""Compositing service."""
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from app.core.exceptions import CompositingError

logger = logging.getLogger(__name__)


class Compositor:
    """Composes final image from layers."""

    def __init__(self, template_config: dict):
        """
        Initialize compositor.

        Args:
            template_config: Template configuration
        """
        self.config = template_config
        self.template_dir = Path(template_config.get("_template_dir", ""))

    def compose(
        self,
        subject: Image.Image,
        placement: dict,
    ) -> Image.Image:
        """
        Compose final image.

        Layers order:
        1. base_clean
        2. subject (placed)
        3. occlusion_mask
        4. glass_fx
        5. postprocess

        Args:
            subject: Subject image (RGBA)
            placement: Placement parameters

        Returns:
            Composed image
        """
        try:
            # Load base_clean
            base_path = self.template_dir / self.config["assets"]["base_clean"]
            base = Image.open(base_path).convert("RGB")
            canvas_width, canvas_height = base.size

            # Resize subject
            subject_resized = subject.resize(
                (placement["scaled_width"], placement["scaled_height"]),
                Image.LANCZOS,
            )

            # Apply edge feathering
            postprocess = self.config.get("postprocess", {})
            if postprocess.get("edge_feather", 0) > 0:
                subject_resized = self._feather_edges(
                    subject_resized, postprocess["edge_feather"]
                )

            # Create canvas
            canvas = base.copy().convert("RGBA")

            # Paste subject
            temp_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            temp_layer.paste(
                subject_resized,
                (placement["paste_x"], placement["paste_y"]),
                subject_resized,
            )

            # Apply light wrap if configured
            if postprocess.get("light_wrap", 0) > 0:
                temp_layer = self._apply_light_wrap(
                    temp_layer, base, postprocess["light_wrap"]
                )

            # Composite subject onto base
            canvas = Image.alpha_composite(canvas, temp_layer)

            # Apply occlusion mask
            if "occlusion_mask" in self.config["assets"]:
                occlusion_path = (
                    self.template_dir / self.config["assets"]["occlusion_mask"]
                )
                if occlusion_path.exists():
                    occlusion = Image.open(occlusion_path).convert("RGBA")
                    occlusion = occlusion.resize((canvas_width, canvas_height), Image.LANCZOS)
                    canvas = Image.alpha_composite(canvas, occlusion)

            # Apply glass fx
            if "glass_fx" in self.config["assets"]:
                glass_path = self.template_dir / self.config["assets"]["glass_fx"]
                if glass_path.exists():
                    glass = Image.open(glass_path).convert("RGBA")
                    glass = glass.resize((canvas_width, canvas_height), Image.LANCZOS)
                    canvas = Image.alpha_composite(canvas, glass)

            # Apply color matching
            if postprocess.get("color_match", False):
                canvas = self._match_colors(canvas, base)

            # Convert to RGB
            result = Image.new("RGB", canvas.size, (255, 255, 255))
            result.paste(canvas, mask=canvas.split()[3])

            logger.info("Composition completed")
            return result

        except Exception as e:
            logger.error(f"Compositing error: {e}", exc_info=True)
            raise CompositingError(f"Ошибка композитинга: {e}") from e

    def _feather_edges(self, image: Image.Image, pixels: int) -> Image.Image:
        """Feather alpha edges."""
        if image.mode != "RGBA":
            return image

        alpha = image.split()[3]
        feathered_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=pixels))
        result = image.copy()
        result.putalpha(feathered_alpha)
        return result

    def _apply_light_wrap(
        self, subject_layer: Image.Image, background: Image.Image, strength: float
    ) -> Image.Image:
        """Apply light wrap effect."""
        # Simple implementation: blend edges with background
        # More sophisticated version would analyze edge lighting
        return subject_layer  # Placeholder

    def _match_colors(self, image: Image.Image, reference: Image.Image) -> Image.Image:
        """Match colors to reference."""
        # Simple color matching - can be enhanced
        return image  # Placeholder

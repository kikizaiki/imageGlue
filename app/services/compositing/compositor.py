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

            # Apply light wrap if configured (after compositing)
            # Will be applied to final composite

            # Composite subject onto base
            canvas = Image.alpha_composite(canvas, temp_layer)

            # Apply light wrap if configured
            if postprocess.get("light_wrap", 0) > 0:
                canvas = self._apply_light_wrap_to_composite(
                    canvas, base, postprocess["light_wrap"]
                )

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

    def _apply_light_wrap_to_composite(
        self, composite: Image.Image, background: Image.Image, strength: float
    ) -> Image.Image:
        """Apply light wrap effect to composite image."""
        if strength <= 0:
            return composite

        try:
            import cv2

            # Convert to numpy
            comp_array = np.array(composite.convert("RGBA"))
            bg_array = np.array(background.convert("RGB"))

            # Extract alpha to find edges
            alpha = comp_array[:, :, 3] / 255.0

            # Get edge region (dilate alpha slightly)
            kernel = np.ones((5, 5), np.uint8)
            alpha_dilated = cv2.dilate(alpha, kernel, iterations=1)
            edge_mask = (alpha_dilated - alpha) > 0.1
            edge_mask = edge_mask[:, :, np.newaxis]

            # Blend edge with background (light wrap effect)
            comp_rgb = comp_array[:, :, :3]
            # Get background at edge positions
            bg_at_edge = bg_array * edge_mask
            # Blend: subject gets some background color at edges
            blended = comp_rgb * (1 - edge_mask * strength * 0.5) + bg_at_edge * (
                edge_mask * strength
            )
            blended = np.clip(blended, 0, 255).astype(np.uint8)

            # Combine back
            result_array = np.concatenate(
                [blended, comp_array[:, :, 3:4]], axis=2
            ).astype(np.uint8)
            return Image.fromarray(result_array, mode="RGBA")

        except Exception as e:
            logger.warning(f"Light wrap failed: {e}, returning original")
            return composite

    def _match_colors(self, image: Image.Image, reference: Image.Image) -> Image.Image:
        """Match colors to reference using histogram matching."""
        try:
            import cv2

            # Convert to RGB
            img_rgb = np.array(image.convert("RGB"))
            ref_rgb = np.array(reference.convert("RGB"))

            # Match histograms for each channel
            matched = img_rgb.copy()
            for i in range(3):
                matched[:, :, i] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
                    matched[:, :, i]
                )

            # Blend with original (subtle matching)
            result = (img_rgb * 0.7 + matched * 0.3).astype(np.uint8)

            # Restore alpha if present
            if image.mode == "RGBA":
                alpha = np.array(image.split()[3])
                result_rgba = np.dstack([result, alpha])
                return Image.fromarray(result_rgba, mode="RGBA")

            return Image.fromarray(result, mode="RGB")

        except Exception as e:
            logger.warning(f"Color matching failed: {e}, returning original")
            return image

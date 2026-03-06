"""Sandwich compositing: background + subject + foreground."""
import logging
from pathlib import Path

from PIL import Image

from app.core.exceptions import CompositingError
from app.services.composer.color_integration import integrate_colors
from app.services.composer.edge_cleanup import cleanup_alpha_edges
from app.services.composer.image_ops import load_image, resize_image

logger = logging.getLogger(__name__)


class SandwichCompositor:
    """Composes images using sandwich layering: background + subject + foreground."""

    def __init__(self, template_config: dict):
        """
        Initialize compositor with template configuration.

        Args:
            template_config: Template configuration dictionary
        """
        self.config = template_config
        self.canvas_width = template_config["canvas"]["width"]
        self.canvas_height = template_config["canvas"]["height"]
        self.background_path = template_config["layers"]["background"]["_resolved_path"]
        self.foreground_path = (
            template_config["layers"]["foreground"]["_resolved_path"]
            if "foreground" in template_config["layers"]
            and "_resolved_path" in template_config["layers"]["foreground"]
            else None
        )

        # Load visor mask if present
        self.visor_mask_path = None
        if "visor_mask" in template_config and isinstance(template_config["visor_mask"], dict):
            template_dir = Path(template_config.get("_template_dir", ""))
            visor_mask_rel = template_config["visor_mask"].get("path", "")
            if visor_mask_rel and template_dir:
                self.visor_mask_path = template_dir / visor_mask_rel
                if not self.visor_mask_path.exists():
                    logger.warning(f"Visor mask not found: {self.visor_mask_path}")
                    self.visor_mask_path = None
                else:
                    logger.debug(f"Visor mask loaded: {self.visor_mask_path}")

    def compose(
        self, subject_image: Image.Image, placement: dict, save_debug: bool = False
    ) -> tuple[Image.Image, dict]:
        """
        Compose final image using sandwich method with mask support.

        Args:
            subject_image: Subject image with alpha channel (RGBA)
            placement: Placement result dictionary with scale, paste_x, paste_y, etc.
            save_debug: Whether to return debug images

        Returns:
            Tuple of (composed image, debug_images dict)

        Raises:
            CompositingError: If compositing fails
        """
        debug_images = {}

        try:
            # Load background
            background = load_image(self.background_path)
            background = resize_image(
                background, self.canvas_width, self.canvas_height
            )

            # Ensure background is RGB or RGBA
            if background.mode == "RGBA":
                # Composite on white background if needed
                bg_rgb = Image.new("RGB", background.size, (255, 255, 255))
                bg_rgb.paste(background, mask=background.split()[3])
                background = bg_rgb
            elif background.mode != "RGB":
                background = background.convert("RGB")

            # Create canvas
            canvas = background.copy()

            # Resize subject
            subject_width = placement["subject_width"]
            subject_height = placement["subject_height"]
            subject_resized = resize_image(
                subject_image, subject_width, subject_height
            )

            # Ensure subject has alpha
            if subject_resized.mode != "RGBA":
                subject_resized = subject_resized.convert("RGBA")

            # Apply edge cleanup if configured
            compositing_config = self.config.get("compositing", {})
            if compositing_config.get("feather_alpha", False) or compositing_config.get("edge_cleanup", False):
                feather_px = compositing_config.get("alpha_feather_px", 0)
                erode_px = compositing_config.get("alpha_erode_px", 0)
                remove_halo = compositing_config.get("remove_halo", False)
                halo_threshold = compositing_config.get("halo_threshold", 240)

                subject_resized = cleanup_alpha_edges(
                    subject_resized,
                    feather_px=feather_px,
                    erode_px=erode_px,
                    remove_halo=remove_halo,
                    halo_threshold=halo_threshold,
                )

            # Apply color integration if configured
            color_config = self.config.get("color_match", {})
            if color_config.get("enabled", False):
                contrast = color_config.get("contrast", 1.0)
                brightness = color_config.get("brightness", 1.0)
                tint = color_config.get("tint_rgb")
                if tint:
                    subject_resized = integrate_colors(
                        subject_resized,
                        contrast=contrast,
                        brightness=brightness,
                        tint_rgb=tuple(tint),
                    )

            # Paste subject with alpha compositing
            paste_x = placement["paste_x"]
            paste_y = placement["paste_y"]

            # Create temporary layer with subject
            canvas_rgba = canvas.convert("RGBA")
            temp_canvas = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
            temp_canvas.paste(subject_resized, (paste_x, paste_y), subject_resized)

            if save_debug:
                debug_images["subject_placed"] = temp_canvas.copy()

            # Apply visor mask if present
            if self.visor_mask_path and self.visor_mask_path.exists():
                try:
                    visor_mask = load_image(self.visor_mask_path)
                    visor_mask = resize_image(visor_mask, self.canvas_width, self.canvas_height)

                    # Ensure mask is grayscale alpha
                    if visor_mask.mode == "RGBA":
                        visor_alpha = visor_mask.split()[3]
                    elif visor_mask.mode == "L":
                        visor_alpha = visor_mask
                    else:
                        visor_mask = visor_mask.convert("L")
                        visor_alpha = visor_mask

                    # Get subject alpha from temp_canvas
                    subject_alpha = temp_canvas.split()[3]

                    # Resize visor alpha to match canvas if needed
                    if visor_alpha.size != subject_alpha.size:
                        visor_alpha = visor_alpha.resize(subject_alpha.size, Image.LANCZOS)

                    # Multiply alpha channels
                    import numpy as np
                    subject_alpha_array = np.array(subject_alpha, dtype=np.float32) / 255.0
                    visor_alpha_array = np.array(visor_alpha, dtype=np.float32) / 255.0
                    combined_alpha = (subject_alpha_array * visor_alpha_array * 255.0).astype(np.uint8)
                    masked_alpha = Image.fromarray(combined_alpha)

                    # Apply masked alpha to subject
                    temp_canvas.putalpha(masked_alpha)

                    if save_debug:
                        debug_images["subject_masked_by_visor"] = temp_canvas.copy()

                    logger.debug("Applied visor mask to subject")
                except Exception as e:
                    logger.warning(f"Failed to apply visor mask: {e}")

            # Composite subject onto background
            canvas_rgba = Image.alpha_composite(canvas_rgba, temp_canvas)

            # Apply foreground if present
            if self.foreground_path:
                foreground = load_image(self.foreground_path)
                foreground = resize_image(
                    foreground, self.canvas_width, self.canvas_height
                )

                if foreground.mode == "RGBA":
                    canvas_rgba = Image.alpha_composite(canvas_rgba, foreground)
                else:
                    foreground_rgba = foreground.convert("RGBA")
                    canvas_rgba = Image.alpha_composite(canvas_rgba, foreground_rgba)

            # Convert back to RGB if no transparency needed
            if self.foreground_path is None or not any(
                p in self.config.get("output", {})
                for p in ["transparent", "alpha"]
            ):
                final = Image.new("RGB", canvas_rgba.size, (255, 255, 255))
                final.paste(canvas_rgba, mask=canvas_rgba.split()[3])
                return final, debug_images

            return canvas_rgba, debug_images

        except Exception as e:
            logger.error(f"Compositing failed: {e}", exc_info=True)
            raise CompositingError(f"Compositing failed: {e}")

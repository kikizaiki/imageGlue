"""Color integration utilities for matching subject to template."""
import logging
from typing import Optional

from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)


def adjust_contrast(image: Image.Image, factor: float = 1.0) -> Image.Image:
    """
    Adjust image contrast.

    Args:
        image: PIL Image
        factor: Contrast factor (1.0 = no change, >1.0 = more contrast)

    Returns:
        Image with adjusted contrast
    """
    if factor == 1.0:
        return image

    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def adjust_brightness(image: Image.Image, factor: float = 1.0) -> Image.Image:
    """
    Adjust image brightness.

    Args:
        image: PIL Image
        factor: Brightness factor (1.0 = no change, >1.0 = brighter)

    Returns:
        Image with adjusted brightness
    """
    if factor == 1.0:
        return image

    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def apply_tint(image: Image.Image, tint_rgb: tuple[float, float, float]) -> Image.Image:
    """
    Apply a color tint to an image.

    Args:
        image: PIL Image (RGB or RGBA)
        tint_rgb: Tint color as (r, g, b) multipliers (1.0 = no change)

    Returns:
        Tinted image
    """
    if image.mode == "RGBA":
        rgb_image = image.convert("RGB")
        alpha = image.split()[3]
    else:
        rgb_image = image.convert("RGB")
        alpha = None

    # Apply tint by multiplying each channel
    import numpy as np

    img_array = np.array(rgb_image, dtype=np.float32)
    img_array[:, :, 0] *= tint_rgb[0]  # R
    img_array[:, :, 1] *= tint_rgb[1]  # G
    img_array[:, :, 2] *= tint_rgb[2]  # B

    # Clamp to valid range
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    result = Image.fromarray(img_array)

    # Restore alpha if present
    if alpha:
        result = result.convert("RGBA")
        result.putalpha(alpha)

    return result


def integrate_colors(
    image: Image.Image,
    contrast: float = 1.0,
    brightness: float = 1.0,
    tint_rgb: Optional[tuple[float, float, float]] = None,
) -> Image.Image:
    """
    Apply color integration adjustments.

    Args:
        image: PIL Image
        contrast: Contrast adjustment factor
        brightness: Brightness adjustment factor
        tint_rgb: Optional tint color (r, g, b) multipliers

    Returns:
        Image with color integration applied
    """
    result = image.copy()

    # Apply adjustments in order
    if contrast != 1.0:
        result = adjust_contrast(result, contrast)

    if brightness != 1.0:
        result = adjust_brightness(result, brightness)

    if tint_rgb:
        result = apply_tint(result, tint_rgb)

    return result

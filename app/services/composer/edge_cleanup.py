"""Edge cleanup utilities for alpha channel refinement."""
import logging
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


def feather_alpha(image: Image.Image, pixels: int = 2) -> Image.Image:
    """
    Apply feathering to alpha channel edges.

    Args:
        image: RGBA PIL Image
        pixels: Number of pixels to feather

    Returns:
        Image with feathered alpha
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    if pixels <= 0:
        return image

    # Extract alpha channel
    alpha = image.split()[3]

    # Apply blur to alpha
    feathered_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=pixels))

    # Combine back
    result = image.copy()
    result.putalpha(feathered_alpha)

    return result


def erode_alpha(image: Image.Image, pixels: int = 1) -> Image.Image:
    """
    Erode alpha channel (shrink transparent areas).

    Args:
        image: RGBA PIL Image
        pixels: Number of pixels to erode

    Returns:
        Image with eroded alpha
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    if pixels <= 0:
        return image

    try:
        from scipy import ndimage
    except ImportError:
        logger.warning("scipy not available, skipping erode operation")
        return image

    # Convert to numpy
    alpha_array = np.array(image.split()[3])

    # Create kernel for erosion
    kernel_size = pixels * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    # Erode (minimum filter)
    eroded = ndimage.minimum_filter(alpha_array, footprint=kernel)

    # Combine back
    result = image.copy()
    result.putalpha(Image.fromarray(eroded))

    return result


def dilate_alpha(image: Image.Image, pixels: int = 1) -> Image.Image:
    """
    Dilate alpha channel (expand opaque areas).

    Args:
        image: RGBA PIL Image
        pixels: Number of pixels to dilate

    Returns:
        Image with dilated alpha
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    if pixels <= 0:
        return image

    try:
        from scipy import ndimage
    except ImportError:
        logger.warning("scipy not available, skipping dilate operation")
        return image

    # Convert to numpy
    alpha_array = np.array(image.split()[3])

    # Create kernel for dilation
    kernel_size = pixels * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    # Dilate (maximum filter)
    dilated = ndimage.maximum_filter(alpha_array, footprint=kernel)

    # Combine back
    result = image.copy()
    result.putalpha(Image.fromarray(dilated))

    return result


def remove_white_halo(image: Image.Image, threshold: int = 240) -> Image.Image:
    """
    Remove white halo around edges by making near-white pixels transparent.

    Args:
        image: RGBA PIL Image
        threshold: RGB threshold for white detection (0-255)

    Returns:
        Image with white halo removed
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Convert to numpy
    img_array = np.array(image)
    r, g, b, a = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2], img_array[:, :, 3]

    # Find near-white pixels
    white_mask = (r > threshold) & (g > threshold) & (b > threshold)

    # Make white pixels transparent
    a[white_mask] = 0

    # Combine back
    result = Image.fromarray(img_array)

    return result


def cleanup_alpha_edges(
    image: Image.Image,
    feather_px: int = 0,
    erode_px: int = 0,
    dilate_px: int = 0,
    remove_halo: bool = False,
    halo_threshold: int = 240,
) -> Image.Image:
    """
    Apply multiple alpha edge cleanup operations.

    Args:
        image: RGBA PIL Image
        feather_px: Pixels to feather
        erode_px: Pixels to erode
        dilate_px: Pixels to dilate
        remove_halo: Whether to remove white halo
        halo_threshold: Threshold for white detection

    Returns:
        Image with cleaned alpha edges
    """
    result = image.copy()

    if result.mode != "RGBA":
        result = result.convert("RGBA")

    # Apply operations in order
    if erode_px > 0:
        result = erode_alpha(result, erode_px)

    if dilate_px > 0:
        result = dilate_alpha(result, dilate_px)

    if remove_halo:
        result = remove_white_halo(result, halo_threshold)

    if feather_px > 0:
        result = feather_alpha(result, feather_px)

    return result

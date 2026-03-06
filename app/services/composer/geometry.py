"""Geometry utilities for image placement and scaling."""
from app.domain.enums import FitMode
from app.domain.models import BBox, PlacementResult


def expand_bbox(
    bbox: BBox, expansion_factor: float, image_width: int, image_height: int
) -> BBox:
    """
    Expand a bounding box by a factor while clamping to image boundaries.

    Args:
        bbox: Original bounding box
        expansion_factor: Multiplier for expansion (1.0 = no change, 1.5 = 50% larger)
        image_width: Image width
        image_height: Image height

    Returns:
        Expanded bounding box clamped to image boundaries
    """
    center_x = bbox.cx
    center_y = bbox.cy

    new_width = bbox.w * expansion_factor
    new_height = bbox.h * expansion_factor

    x1 = max(0, center_x - new_width / 2)
    y1 = max(0, center_y - new_height / 2)
    x2 = min(image_width, center_x + new_width / 2)
    y2 = min(image_height, center_y + new_height / 2)

    # Ensure minimum size
    if x2 - x1 < 1:
        x1 = max(0, center_x - 0.5)
        x2 = min(image_width, center_x + 0.5)
    if y2 - y1 < 1:
        y1 = max(0, center_y - 0.5)
        y2 = min(image_height, center_y + 0.5)

    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


def expand_bbox_directional(
    bbox: BBox,
    expand_left: float = 0.0,
    expand_right: float = 0.0,
    expand_top: float = 0.0,
    expand_bottom: float = 0.0,
    image_width: int = 0,
    image_height: int = 0,
    vertical_shift: float = 0.0,
) -> BBox:
    """
    Expand a bounding box directionally with optional vertical shift.

    Args:
        bbox: Original bounding box
        expand_left: Expansion factor for left side (multiplier of bbox width)
        expand_right: Expansion factor for right side (multiplier of bbox width)
        expand_top: Expansion factor for top side (multiplier of bbox height)
        expand_bottom: Expansion factor for bottom side (multiplier of bbox height)
        image_width: Image width for clamping
        image_height: Image height for clamping
        vertical_shift: Vertical shift downward (positive = down, in pixels)

    Returns:
        Expanded bounding box clamped to image boundaries
    """
    # Calculate expansions in pixels
    left_expand_px = bbox.w * expand_left
    right_expand_px = bbox.w * expand_right
    top_expand_px = bbox.h * expand_top
    bottom_expand_px = bbox.h * expand_bottom

    # Apply expansions
    x1 = bbox.x1 - left_expand_px
    x2 = bbox.x2 + right_expand_px
    y1 = bbox.y1 - top_expand_px
    y2 = bbox.y2 + bottom_expand_px

    # Apply vertical shift (downward = positive)
    y1 += vertical_shift
    y2 += vertical_shift

    # Clamp to image boundaries
    if image_width > 0:
        x1 = max(0, min(x1, image_width))
        x2 = max(x1, min(x2, image_width))
    if image_height > 0:
        y1 = max(0, min(y1, image_height))
        y2 = max(y1, min(y2, image_height))

    # Ensure minimum size
    if x2 - x1 < 1:
        center_x = (bbox.x1 + bbox.x2) / 2
        x1 = max(0, center_x - 0.5)
        x2 = min(image_width if image_width > 0 else center_x + 0.5, center_x + 0.5)
    if y2 - y1 < 1:
        center_y = (bbox.y1 + bbox.y2) / 2
        y1 = max(0, center_y - 0.5)
        y2 = min(image_height if image_height > 0 else center_y + 0.5, center_y + 0.5)

    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


def compute_padded_target_area(
    target_x: int,
    target_y: int,
    target_width: int,
    target_height: int,
    padding: int,
) -> tuple[int, int, int, int]:
    """
    Compute target area with padding applied.

    Args:
        target_x: Target area X coordinate
        target_y: Target area Y coordinate
        target_width: Target area width
        target_height: Target area height
        padding: Padding in pixels

    Returns:
        Tuple of (padded_x, padded_y, padded_width, padded_height)
    """
    padded_x = target_x + padding
    padded_y = target_y + padding
    padded_width = max(1, target_width - 2 * padding)
    padded_height = max(1, target_height - 2 * padding)

    return (padded_x, padded_y, padded_width, padded_height)


def compute_scale(
    subject_width: int,
    subject_height: int,
    target_width: int,
    target_height: int,
    fit_mode: FitMode,
    min_scale: float = 0.1,
    max_scale: float = 10.0,
) -> float:
    """
    Compute scaling factor for subject to fit target area.

    Args:
        subject_width: Subject image width
        subject_height: Subject image height
        target_width: Target area width
        target_height: Target area height
        fit_mode: How to fit (contain or cover)
        min_scale: Minimum allowed scale
        max_scale: Maximum allowed scale

    Returns:
        Scale factor
    """
    if subject_width <= 0 or subject_height <= 0:
        return 1.0
    if target_width <= 0 or target_height <= 0:
        return 1.0

    scale_w = target_width / subject_width
    scale_h = target_height / subject_height

    if fit_mode == FitMode.CONTAIN:
        # Use smaller scale to fit entirely
        scale = min(scale_w, scale_h)
    elif fit_mode == FitMode.COVER:
        # Use larger scale to cover entire area
        scale = max(scale_w, scale_h)
    else:
        scale = min(scale_w, scale_h)

    # Clamp to min/max
    scale = max(min_scale, min(max_scale, scale))

    return scale


def compute_placement(
    subject_width: int,
    subject_height: int,
    target_x: int,
    target_y: int,
    target_width: int,
    target_height: int,
    fit_mode: FitMode,
    anchor_mode: str = "center",
    horizontal_bias: float = 0.0,
    vertical_bias: float = 0.0,
    min_scale: float = 0.1,
    max_scale: float = 10.0,
    padding: int = 0,
    scale_multiplier: float = 1.0,
) -> PlacementResult:
    """
    Compute placement parameters for subject in target area.

    Args:
        subject_width: Subject image width
        subject_height: Subject image height
        target_x: Target area X coordinate
        target_y: Target area Y coordinate
        target_width: Target area width
        target_height: Target area height
        fit_mode: How to fit subject
        anchor_mode: Anchor point (currently only "center" supported)
        horizontal_bias: Horizontal offset bias (-1.0 to 1.0)
        vertical_bias: Vertical offset bias (-1.0 to 1.0)
        min_scale: Minimum scale factor
        max_scale: Maximum scale factor
        padding: Padding around target area

    Returns:
        PlacementResult with computed parameters
    """
    # Apply padding
    padded_x, padded_y, padded_width, padded_height = compute_padded_target_area(
        target_x, target_y, target_width, target_height, padding
    )

    # Compute scale
    scale = compute_scale(
        subject_width,
        subject_height,
        padded_width,
        padded_height,
        fit_mode,
        min_scale,
        max_scale,
    )

    # Apply scale multiplier
    scale *= scale_multiplier

    # Compute scaled dimensions
    scaled_width = int(subject_width * scale)
    scaled_height = int(subject_height * scale)

    # Compute paste position (centered by default)
    if anchor_mode == "center":
        # Center in padded area
        paste_x = padded_x + (padded_width - scaled_width) // 2
        paste_y = padded_y + (padded_height - scaled_height) // 2

        # Apply biases
        max_h_bias = padded_width // 2
        max_v_bias = padded_height // 2
        paste_x += int(horizontal_bias * max_h_bias)
        paste_y += int(vertical_bias * max_v_bias)
    else:
        # Default to top-left if unknown anchor mode
        paste_x = padded_x
        paste_y = padded_y

    return PlacementResult(
        scale=scale,
        paste_x=paste_x,
        paste_y=paste_y,
        target_width=target_width,
        target_height=target_height,
        subject_width=scaled_width,
        subject_height=scaled_height,
    )

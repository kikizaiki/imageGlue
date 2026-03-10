"""Prompt builders for adult and teen refinement strategies."""
import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_adult_realistic_prompt(
    base_prompt: str | None = None,
    template_config: dict[str, Any] | None = None,
) -> str:
    """
    Build prompt for adult realistic face integration.

    Args:
        base_prompt: Base prompt from template config
        template_config: Template configuration

    Returns:
        Formatted prompt for adult flow
    """
    if base_prompt:
        return base_prompt

    # Default adult prompt
    return (
        "The first image is the base poster. The second image is a reference face for identity. "
        "Replace the character on the poster with a character that visually matches the second image. "
        "Preserve the overall composition, pose, lighting, and poster style. "
        "Create a seamless natural integration without artifacts or visible seams. "
        "CRITICAL: Do not add additional elements, duplicates, or new images. "
        "Replace only the existing character on the poster, keeping the rest of the composition unchanged."
    )


def build_teen_stylized_prompt(
    base_prompt: str | None = None,
    template_config: dict[str, Any] | None = None,
) -> str:
    """
    Build prompt for teen stylized face integration.

    Args:
        base_prompt: Base prompt from template config (may be ignored for teen flow)
        template_config: Template configuration

    Returns:
        Formatted prompt for teen flow
    """
    # Teen flow always uses stylized prompt, not exact replacement
    return (
        "First image is the head-and-shoulders crop of the poster character. "
        "Second image is a facial reference only. "
        "Create a stylized cinematic poster version of the character so the face is visually inspired by the second image. "
        "Preserve the original pose, costume, scale, lighting, and poster composition. "
        "Edit only the head/face area. "
        "Do not paste the second image as a foreground object. "
        "Do not create an exact photo copy. "
        "Keep the result natural, clean, and poster-like. "
        "The face should be inspired by the reference, not an exact copy."
    )

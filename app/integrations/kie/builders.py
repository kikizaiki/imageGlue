"""Payload builders for different KIE.ai models."""
import logging
from typing import Any

from app.integrations.kie.models import KIEModel, KIEValidationError

logger = logging.getLogger(__name__)


def build_kie_payload_for_gpt_image_i2i(
    prompt: str,
    poster_url: str | None = None,
    reference_url: str | None = None,
    aspect_ratio: str | None = None,
    quality: str = "high",
) -> dict[str, Any]:
    """
    Build payload for GPT Image 1.5 image-to-image model.

    Based on KIE.ai documentation:
    - https://docs.kie.ai/market/gpt-image/1-5-image-to-image

    Args:
        prompt: Editing prompt
        poster_url: URL of poster/background image (optional for single image editing)
        reference_url: URL of reference image (optional)
        aspect_ratio: Target aspect ratio (e.g., "3:2", "16:9")
        quality: Output quality ("low", "medium", "high")

    Returns:
        Payload dictionary for createTask endpoint

    Raises:
        KIEValidationError: If validation fails
    """
    # Validate prompt
    if not prompt or not prompt.strip():
        raise KIEValidationError(
            "Prompt is required for GPT Image 1.5",
            model=KIEModel.GPT_IMAGE_15_I2I.value,
        )

    # Build input_urls list
    input_urls = []
    if poster_url:
        input_urls.append(poster_url)
    if reference_url:
        input_urls.append(reference_url)

    # For GPT Image 1.5, at least one image URL is typically required
    # But we allow empty list for text-to-image scenarios if needed
    # (though current use case always requires images)

    # Build input object
    input_obj: dict[str, Any] = {
        "prompt": prompt.strip(),
    }

    if input_urls:
        input_obj["input_urls"] = input_urls

    if aspect_ratio:
        input_obj["aspect_ratio"] = aspect_ratio

    if quality:
        input_obj["quality"] = quality

    payload = {
        "model": KIEModel.GPT_IMAGE_15_I2I.value,
        "input": input_obj,
    }

    logger.debug(
        f"Built GPT Image 1.5 payload: model={payload['model']}, "
        f"input_keys={list(input_obj.keys())}, "
        f"input_urls_count={len(input_urls)}"
    )

    return payload


def build_kie_payload_for_nano_banana_edit(
    prompt: str,
    poster_url: str | None = None,
    reference_url: str | None = None,
    output_format: str = "png",
) -> dict[str, Any]:
    """
    Build payload for Google Nano Banana Edit model.

    Args:
        prompt: Editing prompt
        poster_url: URL of poster/background image (REQUIRED)
        reference_url: URL of reference image (REQUIRED)
        output_format: Output format ("png", "jpg", etc.)

    Returns:
        Payload dictionary for createTask endpoint

    Raises:
        KIEValidationError: If validation fails
    """
    # Validate prompt
    if not prompt or not prompt.strip():
        raise KIEValidationError(
            "Prompt is required for Nano Banana Edit",
            model=KIEModel.NANO_BANANA_EDIT.value,
        )

    # Validate required URLs
    if not poster_url:
        raise KIEValidationError(
            "poster_url is required for Nano Banana Edit",
            model=KIEModel.NANO_BANANA_EDIT.value,
            missing_field="poster_url",
        )

    if not reference_url:
        raise KIEValidationError(
            "reference_url is required for Nano Banana Edit",
            model=KIEModel.NANO_BANANA_EDIT.value,
            missing_field="reference_url",
        )

    # Validate URLs are actually URLs (not local paths)
    for url_name, url_value in [("poster_url", poster_url), ("reference_url", reference_url)]:
        if url_value and not (url_value.startswith("http://") or url_value.startswith("https://")):
            raise KIEValidationError(
                f"{url_name} must be a public HTTP(S) URL, not a local file path. "
                f"Got: {url_value[:50]}...",
                model=KIEModel.NANO_BANANA_EDIT.value,
                invalid_field=url_name,
                provided_value=url_value,
            )

    # Build image_urls list (REQUIRED for Nano Banana)
    image_urls = []
    if poster_url:
        image_urls.append(poster_url)
    if reference_url:
        image_urls.append(reference_url)

    # Build input object
    input_obj: dict[str, Any] = {
        "prompt": prompt.strip(),
        "image_urls": image_urls,  # REQUIRED field for Nano Banana
        "output_format": output_format,
    }

    payload = {
        "model": KIEModel.NANO_BANANA_EDIT.value,
        "input": input_obj,
    }

    logger.debug(
        f"Built Nano Banana Edit payload: model={payload['model']}, "
        f"input_keys={list(input_obj.keys())}, "
        f"image_urls_count={len(image_urls)}"
    )

    return payload


def build_kie_payload_for_flux_kontext(
    prompt: str,
    poster_url: str | None = None,
    reference_url: str | None = None,
    output_format: str = "png",
) -> dict[str, Any]:
    """
    Build payload for Flux Kontext Pro model.

    Note: Flux Kontext may use a different endpoint structure.
    This is a placeholder - adjust based on actual KIE.ai documentation.

    Args:
        prompt: Editing prompt
        poster_url: URL of poster/background image (REQUIRED)
        reference_url: URL of reference image (REQUIRED)
        output_format: Output format ("png", "jpg", etc.)

    Returns:
        Payload dictionary for createTask endpoint

    Raises:
        KIEValidationError: If validation fails
    """
    # Validate prompt
    if not prompt or not prompt.strip():
        raise KIEValidationError(
            "Prompt is required for Flux Kontext Pro",
            model=KIEModel.FLUX_KONTEXT_PRO.value,
        )

    # Validate required URLs
    if not poster_url:
        raise KIEValidationError(
            "poster_url is required for Flux Kontext Pro",
            model=KIEModel.FLUX_KONTEXT_PRO.value,
            missing_field="poster_url",
        )

    if not reference_url:
        raise KIEValidationError(
            "reference_url is required for Flux Kontext Pro",
            model=KIEModel.FLUX_KONTEXT_PRO.value,
            missing_field="reference_url",
        )

    # Build input object (similar to nano-banana, but may differ)
    input_obj: dict[str, Any] = {
        "prompt": prompt.strip(),
        "image_urls": [poster_url, reference_url],
        "output_format": output_format,
    }

    payload = {
        "model": KIEModel.FLUX_KONTEXT_PRO.value,
        "input": input_obj,
    }

    logger.debug(
        f"Built Flux Kontext Pro payload: model={payload['model']}, "
        f"input_keys={list(input_obj.keys())}, "
        f"image_urls_count={len(input_obj['image_urls'])}"
    )

    return payload


def build_kie_payload_for_qwen_image_edit(
    prompt: str,
    poster_url: str | None = None,
    reference_url: str | None = None,
    output_format: str = "png",
) -> dict[str, Any]:
    """
    Build payload for Qwen Image Edit model.

    Args:
        prompt: Editing prompt
        poster_url: URL of poster/background image (REQUIRED)
        reference_url: URL of reference image (REQUIRED)
        output_format: Output format ("png", "jpg", etc.)

    Returns:
        Payload dictionary for createTask endpoint

    Raises:
        KIEValidationError: If validation fails
    """
    # Validate prompt
    if not prompt or not prompt.strip():
        raise KIEValidationError(
            "Prompt is required for Qwen Image Edit",
            model=KIEModel.QWEN_IMAGE_EDIT.value,
        )

    # Validate required URLs
    if not poster_url:
        raise KIEValidationError(
            "poster_url is required for Qwen Image Edit",
            model=KIEModel.QWEN_IMAGE_EDIT.value,
            missing_field="poster_url",
        )

    if not reference_url:
        raise KIEValidationError(
            "reference_url is required for Qwen Image Edit",
            model=KIEModel.QWEN_IMAGE_EDIT.value,
            missing_field="reference_url",
        )

    # Build input object
    input_obj: dict[str, Any] = {
        "prompt": prompt.strip(),
        "image_urls": [poster_url, reference_url],
        "output_format": output_format,
    }

    payload = {
        "model": KIEModel.QWEN_IMAGE_EDIT.value,
        "input": input_obj,
    }

    logger.debug(
        f"Built Qwen Image Edit payload: model={payload['model']}, "
        f"input_keys={list(input_obj.keys())}, "
        f"image_urls_count={len(input_obj['image_urls'])}"
    )

    return payload


def build_kie_payload_for_seedream_edit(
    prompt: str,
    poster_url: str | None = None,
    reference_url: str | None = None,
    output_format: str = "png",
) -> dict[str, Any]:
    """
    Build payload for Seedream 4-5 Edit model.

    Args:
        prompt: Editing prompt
        poster_url: URL of poster/background image (REQUIRED)
        reference_url: URL of reference image (REQUIRED)
        output_format: Output format ("png", "jpg", etc.)

    Returns:
        Payload dictionary for createTask endpoint

    Raises:
        KIEValidationError: If validation fails
    """
    # Validate prompt
    if not prompt or not prompt.strip():
        raise KIEValidationError(
            "Prompt is required for Seedream 4-5 Edit",
            model=KIEModel.SEEDREAM_4_5_EDIT.value,
        )

    # Validate required URLs
    if not poster_url:
        raise KIEValidationError(
            "poster_url is required for Seedream 4-5 Edit",
            model=KIEModel.SEEDREAM_4_5_EDIT.value,
            missing_field="poster_url",
        )

    if not reference_url:
        raise KIEValidationError(
            "reference_url is required for Seedream 4-5 Edit",
            model=KIEModel.SEEDREAM_4_5_EDIT.value,
            missing_field="reference_url",
        )

    # Build input object
    input_obj: dict[str, Any] = {
        "prompt": prompt.strip(),
        "image_urls": [poster_url, reference_url],
        "output_format": output_format,
    }

    payload = {
        "model": KIEModel.SEEDREAM_4_5_EDIT.value,
        "input": input_obj,
    }

    logger.debug(
        f"Built Seedream 4-5 Edit payload: model={payload['model']}, "
        f"input_keys={list(input_obj.keys())}, "
        f"image_urls_count={len(input_obj['image_urls'])}"
    )

    return payload


def build_kie_payload(
    model: str,
    prompt: str,
    poster_url: str | None = None,
    reference_url: str | None = None,
    aspect_ratio: str | None = None,
    quality: str = "high",
    output_format: str = "png",
) -> dict[str, Any]:
    """
    Build payload for specified KIE model.

    This is the main entry point that routes to model-specific builders.

    Args:
        model: Model name (KIEModel value)
        prompt: Editing prompt
        poster_url: URL of poster/background image
        reference_url: URL of reference image
        aspect_ratio: Target aspect ratio (for GPT Image 1.5)
        quality: Output quality (for GPT Image 1.5)
        output_format: Output format (for Nano Banana Edit)

    Returns:
        Payload dictionary for createTask endpoint

    Raises:
        UnsupportedModelError: If model is not supported
        KIEValidationError: If validation fails
    """
    from app.integrations.kie.models import KIEModel

    logger.info(
        f"🔧 build_kie_payload called:\n"
        f"   - Input model: {model}\n"
        f"   - Prompt length: {len(prompt)} chars\n"
        f"   - Poster URL: {'present' if poster_url else 'missing'}\n"
        f"   - Reference URL: {'present' if reference_url else 'missing'}"
    )

    try:
        kie_model = KIEModel.from_string(model)
        logger.info(f"🔧 Validated model enum: {kie_model} ({kie_model.value})")
    except ValueError as e:
        logger.error(f"❌ Model validation failed: {e}")
        raise

    if kie_model == KIEModel.GPT_IMAGE_15_I2I:
        logger.info(f"🔧 Routing to GPT Image 1.5 builder")
        payload = build_kie_payload_for_gpt_image_i2i(
            prompt=prompt,
            poster_url=poster_url,
            reference_url=reference_url,
            aspect_ratio=aspect_ratio,
            quality=quality,
        )
        logger.info(
            f"🔧 GPT Image 1.5 payload built:\n"
            f"   - Payload model: {payload.get('model')}\n"
            f"   - Input keys: {list(payload.get('input', {}).keys())}\n"
            f"   - Has input_urls: {'input_urls' in payload.get('input', {})}\n"
            f"   - Has image_urls: {'image_urls' in payload.get('input', {})}"
        )
        return payload
    elif kie_model == KIEModel.NANO_BANANA_EDIT:
        logger.info(f"🔧 Routing to Nano Banana Edit builder")
        payload = build_kie_payload_for_nano_banana_edit(
            prompt=prompt,
            poster_url=poster_url,
            reference_url=reference_url,
            output_format=output_format,
        )
        logger.info(
            f"🔧 Nano Banana Edit payload built:\n"
            f"   - Payload model: {payload.get('model')}\n"
            f"   - Input keys: {list(payload.get('input', {}).keys())}\n"
            f"   - Has input_urls: {'input_urls' in payload.get('input', {})}\n"
            f"   - Has image_urls: {'image_urls' in payload.get('input', {})}"
        )
        return payload
    elif kie_model == KIEModel.FLUX_KONTEXT_PRO:
        logger.info(f"🔧 Routing to Flux Kontext Pro builder")
        payload = build_kie_payload_for_flux_kontext(
            prompt=prompt,
            poster_url=poster_url,
            reference_url=reference_url,
            output_format=output_format,
        )
        logger.info(
            f"🔧 Flux Kontext Pro payload built:\n"
            f"   - Payload model: {payload.get('model')}\n"
            f"   - Input keys: {list(payload.get('input', {}).keys())}"
        )
        return payload
    elif kie_model == KIEModel.QWEN_IMAGE_EDIT:
        logger.info(f"🔧 Routing to Qwen Image Edit builder")
        payload = build_kie_payload_for_qwen_image_edit(
            prompt=prompt,
            poster_url=poster_url,
            reference_url=reference_url,
            output_format=output_format,
        )
        logger.info(
            f"🔧 Qwen Image Edit payload built:\n"
            f"   - Payload model: {payload.get('model')}\n"
            f"   - Input keys: {list(payload.get('input', {}).keys())}"
        )
        return payload
    elif kie_model == KIEModel.SEEDREAM_4_5_EDIT:
        logger.info(f"🔧 Routing to Seedream 4-5 Edit builder")
        payload = build_kie_payload_for_seedream_edit(
            prompt=prompt,
            poster_url=poster_url,
            reference_url=reference_url,
            output_format=output_format,
        )
        logger.info(
            f"🔧 Seedream 4-5 Edit payload built:\n"
            f"   - Payload model: {payload.get('model')}\n"
            f"   - Input keys: {list(payload.get('input', {}).keys())}"
        )
        return payload
    else:
        # This should never happen due to enum validation, but for type safety:
        logger.error(f"❌ Unhandled model in routing: {model}")
        raise ValueError(f"Unhandled model: {model}")

# KIE.ai Integration

Unified integration layer for KIE.ai image editing with multi-model support and fallback logic.

## Architecture

### Why separate payload builders?

Different KIE.ai models have different payload requirements:

1. **GPT Image 1.5** uses `input_urls` (array of URLs)
2. **Nano Banana Edit** uses `image_urls` (array of URLs, REQUIRED field)
3. Each model may have different optional parameters (aspect_ratio, quality, output_format)

Using a single payload format for all models would:
- Break API contracts for models that require specific field names
- Make it impossible to use model-specific features
- Cause validation errors when required fields are missing

### Why Nano Banana requires image_urls?

According to KIE.ai documentation, `google/nano-banana-edit` requires:
- `image_urls` field (not `input_urls`)
- Both `poster_url` and `reference_url` must be provided
- URLs must be public HTTP(S) URLs, not local file paths

This is different from GPT Image 1.5 which:
- Uses `input_urls` (optional, can be empty for text-to-image)
- Accepts single or multiple images
- More flexible with image requirements

### Why validate model early?

Early validation prevents:
- Sending invalid requests to API (saves API calls and time)
- Confusing error messages from API (we provide clear validation errors)
- Wasted resources on invalid configurations

## Usage

### Basic Usage

```python
from app.integrations.kie.client import KIEClient

client = KIEClient(
    api_key="your-api-key",
    primary_model="gpt-image/1.5-image-to-image",
    fallback_model="google/nano-banana-edit",
)

# Create task with automatic fallback
task_id = client.create_image_edit_task(
    prompt="Replace the person in the poster with the person from reference image",
    poster_url="https://example.com/poster.png",
    reference_url="https://example.com/reference.png",
)
```

### Configuration

Add to `.env`:

```bash
# Primary model (required)
KIE_PRIMARY_MODEL=gpt-image/1.5-image-to-image

# Fallback model (optional, empty = no fallback)
KIE_FALLBACK_MODEL=google/nano-banana-edit

# API credentials
KIE_API_KEY=your-api-key
KIE_API_URL=https://api.kie.ai
```

### Model Selection

The client automatically:
1. Tries primary model first
2. If primary fails, tries fallback model (if configured)
3. Raises error only if both fail

### Error Handling

All errors are typed:
- `KIEValidationError`: Validation failed (missing fields, invalid URLs, etc.)
- `KIETaskError`: Task creation or execution failed
- `UnsupportedModelError`: Model not supported

## Supported Models

### GPT Image 1.5 Image-to-Image

- Model: `gpt-image/1.5-image-to-image`
- Payload field: `input_urls` (optional)
- Optional params: `aspect_ratio`, `quality`
- Documentation: https://docs.kie.ai/market/gpt-image/1-5-image-to-image

### Google Nano Banana Edit

- Model: `google/nano-banana-edit`
- Payload field: `image_urls` (REQUIRED)
- Required: `poster_url`, `reference_url` (both must be public URLs)
- Optional params: `output_format`
- Documentation: https://docs.kie.ai/market/google/nano-banana-edit

## Examples

### Example 1: GPT Image 1.5

```python
payload = build_kie_payload_for_gpt_image_i2i(
    prompt="Improve the image quality",
    poster_url="https://example.com/poster.png",
    reference_url="https://example.com/reference.png",
    aspect_ratio="3:2",
    quality="high",
)
```

### Example 2: Nano Banana Edit

```python
payload = build_kie_payload_for_nano_banana_edit(
    prompt="Replace person in poster",
    poster_url="https://example.com/poster.png",  # REQUIRED
    reference_url="https://example.com/reference.png",  # REQUIRED
    output_format="png",
)
```

### Example 3: Fallback Scenario

```python
try:
    task_id = client.create_image_edit_task(
        model="gpt-image/1.5-image-to-image",  # Primary
        prompt="...",
        poster_url="...",
        reference_url="...",
        use_fallback=True,  # Enable fallback
    )
except KIETaskError as e:
    # Both primary and fallback failed
    print(f"Error: {e}")
    print(f"Primary error: {e.details.get('primary_error')}")
    print(f"Fallback error: {e.details.get('fallback_error')}")
```

## Validation Rules

### Common Rules
- `prompt` is always required (non-empty string)
- `model` must be a supported model name

### GPT Image 1.5
- `input_urls` is optional (can be empty for text-to-image)
- `aspect_ratio` is optional (e.g., "3:2", "16:9")
- `quality` is optional ("low", "medium", "high")

### Nano Banana Edit
- `poster_url` is REQUIRED (must be HTTP(S) URL)
- `reference_url` is REQUIRED (must be HTTP(S) URL)
- `image_urls` is automatically built from poster_url + reference_url
- Local file paths are rejected (must upload first to get public URL)

## Debugging

Enable debug logging to see:
- Selected model
- Endpoint used
- Payload keys
- Image URL counts
- Task IDs
- Status polling results

```python
import logging
logging.getLogger("app.integrations.kie").setLevel(logging.DEBUG)
```

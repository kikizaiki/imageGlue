"""KIE.ai refinement service."""
import base64
import logging
from io import BytesIO

import httpx
from PIL import Image

from app.core.config import settings
from app.core.exceptions import CompositingError

logger = logging.getLogger(__name__)


class KIERefiner:
    """Refines images using KIE.ai API."""

    def __init__(self):
        """Initialize KIE refiner."""
        self.api_key = settings.KIE_API_KEY
        self.api_url = settings.KIE_API_URL

        if not self.api_key:
            logger.warning("KIE_API_KEY not set, refinement will be disabled")

    def refine(
        self,
        image: Image.Image,
        prompt: str = "Improve the edges and lighting to make the dog blend naturally into the poster",
    ) -> Image.Image:
        """
        Refine image using KIE.ai API.

        Args:
            image: Image to refine
            prompt: Refinement prompt

        Returns:
            Refined image

        Raises:
            CompositingError: If refinement fails
        """
        if not self.api_key:
            logger.warning("KIE API key not configured, skipping refinement")
            return image

        try:
            # Convert image to base64
            img_bytes = BytesIO()
            image.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")

            # Prepare request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "image": f"data:image/png;base64,{img_base64}",
                "prompt": prompt,
                "model": "image-edit",  # Adjust based on KIE.ai API
            }

            # Make request
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{self.api_url}/images/edit",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()

                result_data = response.json()

                # Extract image from response (adjust based on actual API format)
                if "image" in result_data:
                    result_base64 = result_data["image"].split(",")[1]
                    result_bytes = base64.b64decode(result_base64)
                    result_image = Image.open(BytesIO(result_bytes))
                    logger.info("Image refined using KIE.ai")
                    return result_image
                else:
                    logger.warning("KIE.ai response format unexpected, returning original")
                    return image

        except httpx.HTTPError as e:
            logger.error(f"KIE.ai API error: {e}")
            raise CompositingError(f"Ошибка KIE.ai API: {e}") from e
        except Exception as e:
            logger.error(f"Refinement error: {e}", exc_info=True)
            raise CompositingError(f"Ошибка refinement: {e}") from e

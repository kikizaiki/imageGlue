"""Rembg-based background removal provider."""
import logging
from io import BytesIO

from PIL import Image

from app.core.exceptions import BackgroundRemovalError
from app.services.bgremove.base import BaseBackgroundRemovalProvider

logger = logging.getLogger(__name__)


class RembgBackgroundRemovalProvider(BaseBackgroundRemovalProvider):
    """Provider using rembg library for background removal."""

    def __init__(self, model_name: str = "u2net"):
        """
        Initialize rembg provider.

        Args:
            model_name: Model to use (u2net, u2net_human_seg, silueta, etc.)
        """
        self.model_name = model_name
        self._rembg_session = None

    def _get_session(self):
        """Lazy load rembg session."""
        if self._rembg_session is None:
            try:
                from rembg import new_session

                self._rembg_session = new_session(self.model_name)
                logger.info(f"Rembg session initialized with model: {self.model_name}")
            except ImportError:
                raise BackgroundRemovalError(
                    "rembg not installed. Install with: pip install rembg[new]"
                )
            except Exception as e:
                raise BackgroundRemovalError(f"Failed to initialize rembg: {e}")

        return self._rembg_session

    def remove_background(self, image: Image.Image) -> Image.Image:
        """
        Remove background using rembg.

        Args:
            image: Input PIL Image

        Returns:
            PIL Image with RGBA mode (alpha channel)

        Raises:
            BackgroundRemovalError: If removal fails
        """
        try:
            from rembg import remove

            # Convert PIL to bytes
            img_bytes = BytesIO()
            if image.mode == "RGBA":
                # Save as PNG to preserve alpha
                image.save(img_bytes, format="PNG")
            else:
                image.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            # Remove background
            output_bytes = remove(img_bytes.getvalue(), session=self._get_session())

            # Convert back to PIL Image
            result_image = Image.open(BytesIO(output_bytes))
            result_image = result_image.convert("RGBA")

            logger.debug(f"Background removed: {image.size} -> {result_image.size}")
            return result_image

        except BackgroundRemovalError:
            raise
        except Exception as e:
            logger.error(f"Rembg background removal failed: {e}", exc_info=True)
            raise BackgroundRemovalError(f"Background removal failed: {e}")

    def get_name(self) -> str:
        """Get provider name."""
        return f"rembg_{self.model_name}"

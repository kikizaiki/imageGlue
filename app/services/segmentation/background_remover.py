"""Background removal service."""
import logging
from io import BytesIO

import numpy as np
from PIL import Image

from app.core.config import settings
from app.core.exceptions import SegmentationError

logger = logging.getLogger(__name__)


class BackgroundRemover:
    """Removes background from images."""

    def __init__(self):
        """Initialize background remover."""
        self._rembg_session = None

    def _get_session(self):
        """Lazy load rembg session."""
        if self._rembg_session is None:
            try:
                from rembg import new_session

                self._rembg_session = new_session(settings.REMBG_MODEL)
                logger.info(f"Rembg session initialized with model: {settings.REMBG_MODEL}")
            except ImportError as e:
                raise SegmentationError(
                    f"rembg not installed. Install with: pip install rembg. Error: {e}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize rembg with model {settings.REMBG_MODEL}: {e}")
                # Try with default model
                try:
                    self._rembg_session = new_session()
                    logger.info("Rembg session initialized with default model")
                except Exception as e2:
                    raise SegmentationError(f"Failed to initialize rembg: {e2}")

        return self._rembg_session

    def remove_background(self, image: Image.Image) -> Image.Image:
        """
        Remove background from image.

        Args:
            image: PIL Image

        Returns:
            PIL Image with RGBA mode (alpha channel)

        Raises:
            SegmentationError: If removal fails
        """
        # Try rembg first
        try:
            from rembg import remove

            # Convert to bytes
            img_bytes = BytesIO()
            image.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            # Remove background
            output_bytes = remove(img_bytes.getvalue(), session=self._get_session())

            # Convert back to PIL
            result = Image.open(BytesIO(output_bytes))
            result = result.convert("RGBA")

            logger.debug(f"Background removed with rembg: {image.size} -> {result.size}")
            return result

        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"rembg not available ({e}), using fallback method")
            return self._remove_background_fallback(image)
        except SegmentationError:
            raise
        except Exception as e:
            logger.warning(f"rembg failed ({e}), using fallback method")
            return self._remove_background_fallback(image)

    def _remove_background_fallback(self, image: Image.Image) -> Image.Image:
        """
        Fallback background removal using OpenCV grabcut.

        Args:
            image: PIL Image

        Returns:
            PIL Image with RGBA mode
        """
        try:
            import cv2

            # Convert PIL to OpenCV
            img_array = np.array(image.convert("RGB"))
            height, width = img_array.shape[:2]

            # Simple method: use grabcut with automatic rectangle
            # Create rectangle in center 80% of image
            rect = (
                int(width * 0.1),
                int(height * 0.1),
                int(width * 0.8),
                int(height * 0.8),
            )

            mask = np.zeros((height, width), np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            cv2.grabCut(img_array, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

            # Create alpha mask (0 and 2 = background, 1 and 3 = foreground)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")

            # Apply morphological operations to clean up mask
            kernel = np.ones((3, 3), np.uint8)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)

            # Create RGBA image
            result = Image.fromarray(img_array).convert("RGBA")
            alpha = Image.fromarray(mask2, mode="L")
            result.putalpha(alpha)

            logger.info("Background removed using OpenCV fallback method")
            return result

        except Exception as e:
            logger.error(f"Fallback background removal failed: {e}")
            # Last resort: return image with full opacity
            if image.mode != "RGBA":
                result = image.convert("RGBA")
            else:
                result = image.copy()
            logger.warning("Using full-opacity fallback (no background removal)")
            return result

    def get_mask(self, image: Image.Image) -> Image.Image:
        """
        Get alpha mask from RGBA image.

        Args:
            image: PIL Image with RGBA mode

        Returns:
            Grayscale mask image
        """
        if image.mode != "RGBA":
            raise SegmentationError("Image must be RGBA to extract mask")

        mask = image.split()[3]  # Alpha channel
        return mask

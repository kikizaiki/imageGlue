"""Teen flow refiner for stylized face integration."""
import base64
import logging
from io import BytesIO
from typing import Any

import httpx
import numpy as np
from PIL import Image

from app.core.config import settings
from app.core.exceptions import CompositingError
from app.integrations.kie.client import KIEClient
from app.integrations.kie.models import KIETaskError, KIEValidationError, UnsupportedModelError
from app.services.refinement.kie_upload import KIEUploader
from app.services.refinement.prompt_builders import build_teen_stylized_prompt

logger = logging.getLogger(__name__)


class TeenRefiner:
    """Refines images using teen flow: face_region mode with stylized prompts."""

    def __init__(self):
        """Initialize teen refiner."""
        self.api_key = settings.KIE_API_KEY
        self.api_url = settings.KIE_API_URL.rstrip("/")
        self.upload_base_url = settings.KIE_UPLOAD_BASE_URL.rstrip("/")
        
        # Get teen flow models from config
        teen_models_str = settings.TEEN_FLOW_MODELS
        self.teen_models = [m.strip() for m in teen_models_str.split(",") if m.strip()]
        
        if not self.teen_models:
            logger.warning("No teen flow models configured, using default")
            self.teen_models = ["google/nano-banana-edit"]
        
        logger.info(f"Teen flow models: {self.teen_models}")
        
        # Initialize KIE client (will use models from teen_models list)
        # Note: upload_base_url is stored separately for upload methods, not passed to KIEClient
        self.client = KIEClient(
            api_key=self.api_key,
            api_url=self.api_url,
            primary_model=self.teen_models[0] if self.teen_models else "google/nano-banana-edit",
            fallback_model=self.teen_models[1] if len(self.teen_models) > 1 else None,
        )
        
        # Initialize reliable uploader
        self.uploader = KIEUploader(
            api_key=self.api_key,
            upload_base_url=self.upload_base_url,
        )

    def refine_face_region(
        self,
        reference_image: Image.Image,
        poster_background: Image.Image,
        template_config: dict[str, Any],
        job_id: str | None = None,
        debug: bool = False,
    ) -> Image.Image:
        """
        Refine face region using teen flow with stylized prompts.

        Args:
            reference_image: Reference face image
            poster_background: Full poster background
            template_config: Template configuration with face_region coordinates
            job_id: Job ID for debug artifacts
            debug: Whether to save debug artifacts

        Returns:
            Final image with stylized face replacement
        """
        from app.core.storage import Storage

        logger.info("🎯 Starting teen flow: face_region mode with stylized prompts")
        
        # Get face_region coordinates from template config
        ai_integration_config = template_config.get("ai_integration", {})
        
        # Try face_region, head_region, or edit_region
        face_region = (
            ai_integration_config.get("face_region") or
            ai_integration_config.get("head_region") or
            ai_integration_config.get("edit_region") or
            {}
        )
        
        if not face_region:
            raise CompositingError(
                "face_region/head_region/edit_region coordinates not found in template config"
            )
        
        x = int(face_region.get("x", 0))
        y = int(face_region.get("y", 0))
        width = int(face_region.get("width", 0))
        height = int(face_region.get("height", 0))
        
        if width <= 0 or height <= 0:
            raise CompositingError(f"Invalid face_region dimensions: {width}x{height}")
        
        logger.info(f"Face region: x={x}, y={y}, w={width}, h={height}")
        
        # Initialize storage for debug artifacts
        storage = None
        if debug and job_id:
            storage = Storage(job_id)
            storage.save_debug(poster_background, "teen_00_original_poster.png")
            storage.save_debug(reference_image, "teen_01_reference_face.png")
        
        # Step 1: Crop head region from poster
        logger.info("Step 1: Cropping head region from poster...")
        poster_width, poster_height = poster_background.size
        
        # Ensure coordinates are within bounds
        x = max(0, min(x, poster_width))
        y = max(0, min(y, poster_height))
        width = min(width, poster_width - x)
        height = min(height, poster_height - y)
        
        head_region_crop = poster_background.crop((x, y, x + width, y + height))
        logger.info(f"Cropped head region: {head_region_crop.size}")
        
        if debug and storage:
            storage.save_debug(head_region_crop, "teen_02_head_region_crop.png")
        
        # Step 2: Normalize reference face
        logger.info("Step 2: Normalizing reference face...")
        normalized_reference = self._normalize_reference_face(reference_image, debug, storage)
        logger.info(f"Normalized reference: {normalized_reference.size}")
        
        # Step 3: Build teen stylized prompt
        logger.info("Step 3: Building teen stylized prompt...")
        teen_prompt = build_teen_stylized_prompt(template_config=template_config)
        logger.info(f"Teen prompt length: {len(teen_prompt)} chars")
        
        # Step 4: Try models in order with fallback
        logger.info(f"Step 4: Trying teen models in order: {self.teen_models}")
        
        edited_head_region = None
        successful_model = None
        
        for model_idx, model_name in enumerate(self.teen_models):
            logger.info(f"Trying model {model_idx + 1}/{len(self.teen_models)}: {model_name}")
            
            try:
                # Upload images
                head_region_url = self._upload_file(head_region_crop)
                reference_url = self._upload_file(normalized_reference)
                
                logger.info(
                    f"🔧 Creating task with teen model:\n"
                    f"   - Model: {model_name}\n"
                    f"   - Prompt: {teen_prompt[:100]}...\n"
                    f"   - Head region URL: {head_region_url[:50]}...\n"
                    f"   - Reference URL: {reference_url[:50]}..."
                )
                
                # Create task with specific model
                logger.info(
                    f"🔧 Creating KIE task:\n"
                    f"   - Requested model: {model_name}\n"
                    f"   - Client primary_model: {self.client.primary_model}\n"
                    f"   - Client fallback_model: {self.client.fallback_model or '(not configured)'}\n"
                    f"   - use_fallback: False (manual fallback handling)"
                )
                
                task_id = self.client.create_image_edit_task(
                    model=model_name,
                    prompt=teen_prompt,
                    poster_url=head_region_url,
                    reference_url=reference_url,
                    use_fallback=False,  # We handle fallback manually
                )
                
                logger.info(f"✅ Task created with model {model_name}, ID: {task_id}")
                
                # Wait for completion
                task_result = self.client.wait_for_task_completion(task_id, max_wait=300)
                
                # Download result
                import json
                result_json_raw = task_result.get("resultJson")
                if not result_json_raw:
                    raise CompositingError(f"No resultJson in KIE.ai response: {task_result}")
                
                result_json = json.loads(result_json_raw)
                result_urls = result_json.get("resultUrls") or []
                if not result_urls:
                    raise CompositingError(f"No resultUrls in KIE.ai resultJson: {result_json}")
                
                result_url = result_urls[0]
                edited_head_region = self._download_image_from_url(result_url)
                successful_model = model_name
                
                # Log model from response if available
                response_model = task_result.get("model") or task_result.get("data", {}).get("model")
                logger.info(
                    f"✅ Successfully processed with model {model_name}.\n"
                    f"   - Requested model: {model_name}\n"
                    f"   - Response model: {response_model or '(not in response)'}\n"
                    f"   - Result size: {edited_head_region.size}\n"
                    f"   - Expected crop size: {width}x{height}"
                )
                break
                
            except (KIEValidationError, UnsupportedModelError, KIETaskError) as e:
                logger.warning(f"Model {model_name} failed: {e}")
                if model_idx < len(self.teen_models) - 1:
                    logger.info(f"Trying next model in chain...")
                    continue
                else:
                    raise CompositingError(
                        f"All teen flow models failed. Last error: {e}"
                    ) from e
            except Exception as e:
                logger.error(f"Unexpected error with model {model_name}: {e}", exc_info=True)
                if model_idx < len(self.teen_models) - 1:
                    logger.info(f"Trying next model in chain...")
                    continue
                else:
                    raise CompositingError(
                        f"All teen flow models failed. Last error: {e}"
                    ) from e
        
        if not edited_head_region:
            raise CompositingError("Failed to get edited head region from any teen model")
        
        if debug and storage:
            from app.core.storage import sanitize_for_filename
            sanitized_model = sanitize_for_filename(successful_model)
            storage.save_debug(
                edited_head_region, f"teen_03_edited_head_region_{sanitized_model}.png"
            )
        
        # Step 5: Recompose edited head region back into poster
        logger.info("Step 5: Recomposing edited head region into poster...")
        logger.info(
            f"📊 Recomposition details:\n"
            f"   - Flow: teen_kie_stylized\n"
            f"   - Refinement mode: face_region\n"
            f"   - Provider: KIE.ai\n"
            f"   - Model used: {successful_model}\n"
            f"   - Original poster size: {poster_background.size}\n"
            f"   - Crop box: x={x}, y={y}, w={width}, h={height}\n"
            f"   - Cropped region sent to AI: {head_region_crop.size}\n"
            f"   - AI returned image size: {edited_head_region.size}\n"
            f"   - Target crop size: {width}x{height}"
        )
        
        final_image = self._recompose_face_region(
            poster_background,
            edited_head_region,
            x, y, width, height,
            debug=debug,
            storage=storage,
        )
        
        logger.info(
            f"✅ Teen flow completed successfully.\n"
            f"   - Model used: {successful_model}\n"
            f"   - Final image size: {final_image.size}"
        )
        
        if debug and storage:
            storage.save_debug(final_image, "teen_04_final_recomposed.png")
        
        return final_image

    def _normalize_reference_face(
        self,
        reference_image: Image.Image,
        debug: bool = False,
        storage: Any = None,
    ) -> Image.Image:
        """Normalize reference face: detect face, crop with padding, avoid extreme close-up."""
        from app.services.detection.dog_detector import DogDetector
        
        logger.info("Normalizing reference face for teen flow...")
        
        # Detect face/head in reference image
        detector = DogDetector()
        try:
            detection = detector.detect(reference_image, entity_type="human")
            head_bbox = detection.head_bbox
            
            if not head_bbox:
                logger.warning("No head detected, using entity bbox")
                head_bbox = detection.dog_bbox
        except Exception as e:
            logger.warning(f"Face detection failed: {e}, using center crop")
            # Fallback: use center crop
            img_width, img_height = reference_image.size
            from app.models.schemas import BBox
            head_bbox = BBox(
                x1=img_width * 0.2,
                y1=img_height * 0.1,
                x2=img_width * 0.8,
                y2=img_height * 0.6,
            )
        
        # Add padding (30% on each side)
        padding_ratio = 0.3
        bbox_width = head_bbox.x2 - head_bbox.x1
        bbox_height = head_bbox.y2 - head_bbox.y1
        
        padding_x = bbox_width * padding_ratio
        padding_y = bbox_height * padding_ratio
        
        crop_x1 = max(0, head_bbox.x1 - padding_x)
        crop_y1 = max(0, head_bbox.y1 - padding_y)
        crop_x2 = min(reference_image.width, head_bbox.x2 + padding_x)
        crop_y2 = min(reference_image.height, head_bbox.y2 + padding_y)
        
        # Crop with padding
        normalized = reference_image.crop((int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)))
        
        # Ensure minimum size (avoid extreme close-up)
        min_size = 256
        if normalized.width < min_size or normalized.height < min_size:
            aspect = normalized.width / normalized.height
            if normalized.width < min_size:
                new_width = min_size
                new_height = int(new_width / aspect)
            else:
                new_height = min_size
                new_width = int(new_height * aspect)
            normalized = normalized.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        logger.info(f"Normalized reference face: {normalized.size} (from {reference_image.size})")
        
        if debug and storage:
            storage.save_debug(normalized, "teen_01b_normalized_reference.png")
        
        return normalized

    def _recompose_face_region(
        self,
        poster: Image.Image,
        edited_head_region: Image.Image,
        x: int,
        y: int,
        width: int,
        height: int,
        debug: bool = False,
        storage: Any = None,
    ) -> Image.Image:
        """
        Recompose edited head region back into poster with alpha feathering.

        Args:
            poster: Original poster
            edited_head_region: Edited head region from KIE
            x, y, width, height: Original crop coordinates
            debug: Whether to save debug artifacts
            storage: Storage instance for debug artifacts

        Returns:
            Recomposed poster with replaced face
        """
        logger.info(
            f"🔧 Starting recomposition:\n"
            f"   - Poster size: {poster.size}\n"
            f"   - Edited region size: {edited_head_region.size}\n"
            f"   - Target crop: {width}x{height} at ({x}, {y})"
        )

        # Save debug artifacts
        if debug and storage:
            storage.save_debug(poster, "teen_recompose_00_original_poster.png")
            # Create target crop visualization
            target_crop = poster.crop((x, y, x + width, y + height))
            storage.save_debug(target_crop, "teen_recompose_01_target_crop.png")
            storage.save_debug(edited_head_region, "teen_recompose_02_ai_returned_crop.png")

        # Validate aspect ratio
        target_aspect = width / height if height > 0 else 1.0
        returned_aspect = edited_head_region.width / edited_head_region.height if edited_head_region.height > 0 else 1.0
        aspect_diff = abs(target_aspect - returned_aspect) / target_aspect

        logger.info(
            f"📐 Aspect ratio validation:\n"
            f"   - Target aspect: {target_aspect:.3f} ({width}/{height})\n"
            f"   - Returned aspect: {returned_aspect:.3f} ({edited_head_region.width}/{edited_head_region.height})\n"
            f"   - Difference: {aspect_diff * 100:.1f}%"
        )

        if aspect_diff > 0.2:  # More than 20% difference
            logger.warning(
                f"⚠️  Aspect ratio mismatch > 20% ({aspect_diff * 100:.1f}%). "
                f"Normalizing returned image to target aspect."
            )

        # Resize edited head region to match original crop size
        if edited_head_region.size != (width, height):
            logger.info(
                f"🔧 Resizing edited head region:\n"
                f"   - From: {edited_head_region.size}\n"
                f"   - To: ({width}, {height})\n"
                f"   - Method: LANCZOS"
            )
            edited_head_region = edited_head_region.resize((width, height), Image.Resampling.LANCZOS)
            
            if debug and storage:
                storage.save_debug(edited_head_region, "teen_recompose_03_resized_ai_crop.png")

        # Create alpha feather mask for smooth blending
        # Feather size: ~5% of crop dimensions
        feather_size = max(8, min(width, height) // 20)
        logger.info(f"🎭 Creating alpha feather mask (feather_size={feather_size}px)")

        # Create mask with feathering
        mask = Image.new("L", (width, height), 255)
        mask_array = np.array(mask)

        # Create gradient from edges
        for i in range(feather_size):
            alpha = int(255 * (i + 1) / feather_size)
            # Top edge
            mask_array[i, :] = np.minimum(mask_array[i, :], alpha)
            # Bottom edge
            mask_array[height - 1 - i, :] = np.minimum(mask_array[height - 1 - i, :], alpha)
            # Left edge
            mask_array[:, i] = np.minimum(mask_array[:, i], alpha)
            # Right edge
            mask_array[:, width - 1 - i] = np.minimum(mask_array[:, width - 1 - i], alpha)

        # Apply corner feathering (circular gradient in corners)
        center_x, center_y = width // 2, height // 2
        max_dist = np.sqrt(center_x**2 + center_y**2)
        for py in range(height):
            for px in range(width):
                # Distance from center
                dist_from_center = np.sqrt((px - center_x)**2 + (py - center_y)**2)
                # Distance from nearest edge
                dist_from_edge = min(px, width - 1 - px, py, height - 1 - py)
                # Use edge distance for feathering
                if dist_from_edge < feather_size:
                    edge_alpha = int(255 * dist_from_edge / feather_size)
                    mask_array[py, px] = min(mask_array[py, px], edge_alpha)

        feather_mask = Image.fromarray(mask_array, mode="L")

        if debug and storage:
            storage.save_debug(feather_mask, "teen_recompose_04_recomposition_mask.png")

        # Convert edited region to RGBA if needed
        if edited_head_region.mode != "RGBA":
            edited_rgba = edited_head_region.convert("RGBA")
        else:
            edited_rgba = edited_head_region.copy()

        # Apply feather mask to alpha channel
        alpha_channel = edited_rgba.split()[3]
        alpha_channel = Image.blend(
            Image.new("L", alpha_channel.size, 0),
            alpha_channel,
            1.0
        )
        # Apply feather mask
        alpha_channel = Image.composite(
            alpha_channel,
            Image.new("L", alpha_channel.size, 0),
            feather_mask
        )
        edited_rgba.putalpha(alpha_channel)

        # Create result with RGBA support
        if poster.mode != "RGBA":
            result = poster.convert("RGBA")
        else:
            result = poster.copy()

        # Paste with alpha blending
        logger.info(
            f"📌 Pasting edited region:\n"
            f"   - Coordinates: ({x}, {y})\n"
            f"   - Size: {width}x{height}\n"
            f"   - Using alpha blending with feather mask"
        )
        result.paste(edited_rgba, (x, y), edited_rgba)

        # Convert back to RGB if original was RGB
        if poster.mode == "RGB":
            result = result.convert("RGB")

        logger.info(f"✅ Recomposition completed. Final size: {result.size}")

        if debug and storage:
            storage.save_debug(result, "teen_recompose_05_final_recomposed.png")

        return result

    def _upload_file(self, image: Image.Image) -> str:
        """Upload image to KIE.ai using reliable upload utility."""
        return self.uploader.upload_image(image)

    def _download_image_from_url(self, file_url: str) -> Image.Image:
        """Download image from URL."""
        import httpx
        from io import BytesIO
        
        try:
            # Check if it's a tempfile URL (needs download-url endpoint)
            if "tempfile" in file_url or "kie.ai" in file_url:
                # Get temporary download URL first
                download_url = self._get_download_url(file_url)
                logger.debug(f"Got download URL: {download_url[:50]}...")
                url_to_download = download_url
            else:
                # Direct URL
                url_to_download = file_url

            # Download image
            with httpx.Client(timeout=120.0) as client:
                response = client.get(url_to_download)
                response.raise_for_status()
                result_image = Image.open(BytesIO(response.content))
                logger.info("Image downloaded from KIE.ai")
                return result_image

        except Exception as e:
            logger.error(f"Failed to download image from URL: {e}")
            raise CompositingError(f"Failed to download image: {e}") from e

    def _get_download_url(self, file_url: str) -> str:
        """Get temporary download link for generated file from KIE.ai."""
        import httpx
        
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {"url": file_url}
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self.api_url}/api/v1/common/download-url",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                result_data = response.json()
                
                if "data" in result_data:
                    return result_data["data"]
                elif "url" in result_data:
                    return result_data["url"]
                else:
                    raise CompositingError(f"Unexpected download URL response format: {result_data}")
        except Exception as e:
            logger.error(f"Failed to get download URL: {e}")
            raise CompositingError(f"Failed to get download URL: {e}") from e

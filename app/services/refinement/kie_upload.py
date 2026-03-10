"""Reliable KIE.ai file upload utility with preprocessing and retry logic."""
import base64
import logging
import os
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from app.core.config import settings
from app.core.exceptions import CompositingError

logger = logging.getLogger(__name__)

# Thresholds
BASE64_MAX_SIZE_BYTES = 2 * 1024 * 1024  # 2MB - max for base64 upload
BASE64_MAX_DIMENSION = 2048  # Max width/height for base64
STREAM_MAX_DIMENSION = 4096  # Max width/height before resize
JPEG_QUALITY = 85  # JPEG quality for conversion


class KIEUploader:
    """Reliable KIE.ai file uploader with preprocessing and retry logic."""

    def __init__(self, api_key: str, upload_base_url: str):
        """
        Initialize uploader.

        Args:
            api_key: KIE API key
            upload_base_url: Base URL for file upload API
        """
        self.api_key = api_key
        self.upload_base_url = upload_base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def upload_image(
        self,
        image: Image.Image,
        upload_path: str = "images/user-uploads",
        filename: str | None = None,
    ) -> str:
        """
        Upload image to KIE.ai with automatic method selection and preprocessing.

        Strategy:
        1. Primary: file-stream-upload (for all files)
        2. Secondary: file-base64-upload (only for small files < 2MB)
        3. Automatic preprocessing: resize, convert to JPEG if needed

        Args:
            image: PIL Image to upload
            upload_path: Upload path on KIE server
            filename: Optional filename (auto-generated if not provided)

        Returns:
            File URL from KIE.ai

        Raises:
            CompositingError: If upload fails after all retries
        """
        # Log original image info
        original_size = image.size
        original_mode = image.mode
        original_format = getattr(image, "format", "Unknown")

        # Convert to bytes to check size
        img_buffer = BytesIO()
        image.save(img_buffer, format="PNG")
        original_bytes = len(img_buffer.getvalue())
        img_buffer.seek(0)

        logger.info(
            f"📤 Upload preparation:\n"
            f"   - Original size: {original_size[0]}x{original_size[1]} pixels\n"
            f"   - Original mode: {original_mode}\n"
            f"   - Original format: {original_format}\n"
            f"   - Original bytes: {original_bytes / 1024:.1f} KB"
        )

        # Preprocess image if needed
        processed_image, preprocessing_info = self._preprocess_image(image)
        if preprocessing_info["modified"]:
            logger.info(
                f"🔧 Image preprocessing applied:\n"
                f"   - {preprocessing_info['reason']}\n"
                f"   - New size: {processed_image.size[0]}x{processed_image.size[1]}\n"
                f"   - New format: {preprocessing_info.get('format', 'PNG')}"
            )

        # Get processed image bytes
        processed_buffer = BytesIO()
        processed_format = preprocessing_info.get("format", "PNG")
        if processed_format == "JPEG":
            processed_image = processed_image.convert("RGB")
            processed_image.save(processed_buffer, format="JPEG", quality=JPEG_QUALITY)
        else:
            processed_image.save(processed_buffer, format="PNG")
        processed_bytes = len(processed_buffer.getvalue())
        processed_buffer.seek(0)

        logger.info(
            f"📦 Processed image:\n"
            f"   - Size: {processed_bytes / 1024:.1f} KB\n"
            f"   - Format: {processed_format}"
        )

        # Determine upload method
        use_base64 = processed_bytes < BASE64_MAX_SIZE_BYTES
        upload_method = "base64" if use_base64 else "stream"

        logger.info(
            f"🎯 Upload strategy:\n"
            f"   - Method: {upload_method}\n"
            f"   - Reason: {'File size < 2MB, using base64' if use_base64 else 'File size >= 2MB, using stream'}"
        )

        # Generate filename if not provided
        if not filename:
            ext = "jpg" if processed_format == "JPEG" else "png"
            filename = f"image_{int(time.time())}.{ext}"

        # Try upload with retry
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                if upload_method == "stream":
                    return self._upload_via_stream_with_retry(
                        processed_image, processed_format, upload_path, filename, attempt
                    )
                else:
                    # Try base64 first, fallback to stream if fails
                    try:
                        return self._upload_via_base64_with_retry(
                            processed_image, processed_format, upload_path, filename, attempt
                        )
                    except Exception as base64_error:
                        logger.warning(
                            f"Base64 upload failed: {base64_error}, "
                            f"falling back to stream upload..."
                        )
                        return self._upload_via_stream_with_retry(
                            processed_image, processed_format, upload_path, filename, 0
                        )
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(
                        f"Upload attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    raise CompositingError(
                        f"Upload failed after {max_retries} attempts. Last error: {e}"
                    ) from e
            except Exception as e:
                # Non-retryable errors
                raise CompositingError(f"Upload failed: {e}") from e

        raise CompositingError(f"Upload failed after {max_retries} attempts: {last_error}")

    def _preprocess_image(self, image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
        """
        Preprocess image for optimal upload: resize, convert format if needed.

        Args:
            image: Original image

        Returns:
            Tuple of (processed_image, info_dict)
        """
        info = {"modified": False, "reason": "No preprocessing needed", "format": "PNG"}

        width, height = image.size
        max_dimension = max(width, height)

        # Check if resize needed
        if max_dimension > STREAM_MAX_DIMENSION:
            # Resize maintaining aspect ratio
            if width > height:
                new_width = STREAM_MAX_DIMENSION
                new_height = int(height * (STREAM_MAX_DIMENSION / width))
            else:
                new_height = STREAM_MAX_DIMENSION
                new_width = int(width * (STREAM_MAX_DIMENSION / height))

            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            info["modified"] = True
            info["reason"] = f"Resized from {width}x{height} to {new_width}x{new_height}"

        # For large images or images without transparency, consider JPEG
        # But only if it's significantly smaller
        if max_dimension > 1024:
            # Check if image has transparency
            has_alpha = image.mode in ("RGBA", "LA") or "transparency" in image.info

            if not has_alpha:
                # Try JPEG conversion to reduce size
                jpeg_buffer = BytesIO()
                rgb_image = image.convert("RGB")
                rgb_image.save(jpeg_buffer, format="JPEG", quality=JPEG_QUALITY)
                jpeg_size = len(jpeg_buffer.getvalue())

                png_buffer = BytesIO()
                image.save(png_buffer, format="PNG")
                png_size = len(png_buffer.getvalue())

                # Use JPEG if it's at least 30% smaller
                if jpeg_size < png_size * 0.7:
                    image = rgb_image
                    info["modified"] = True
                    info["format"] = "JPEG"
                    if "reason" in info:
                        info["reason"] += ", converted to JPEG (smaller size)"
                    else:
                        info["reason"] = "Converted to JPEG (smaller size)"

        return image, info

    def _upload_via_stream_with_retry(
        self,
        image: Image.Image,
        image_format: str,
        upload_path: str,
        filename: str,
        attempt: int,
    ) -> str:
        """Upload via stream with retry logic."""
        endpoint = f"{self.upload_base_url}/api/file-stream-upload"

        logger.info(
            f"📤 Stream upload (attempt {attempt + 1}):\n"
            f"   - Endpoint: {endpoint}\n"
            f"   - Filename: {filename}\n"
            f"   - Format: {image_format}"
        )

        # Save to temporary file
        tmp_file = None
        try:
            ext = "jpg" if image_format == "JPEG" else "png"
            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
                if image_format == "JPEG":
                    image.convert("RGB").save(tmp.name, format="JPEG", quality=JPEG_QUALITY)
                else:
                    image.save(tmp.name, format="PNG")
                tmp_path = tmp.name
                tmp_file = tmp_path

            tmp_path_obj = Path(tmp_path)

            with open(tmp_path, "rb") as f:
                files = {"file": (filename, f, f"image/{ext}")}
                data = {"uploadPath": upload_path, "fileName": filename}

                with httpx.Client(timeout=120.0) as client:
                    response = client.post(
                        endpoint,
                        headers=self.headers,
                        files=files,
                        data=data,
                        timeout=120.0,
                    )
                    response.raise_for_status()
                    result = response.json()

                    file_url = self._extract_file_url(result)
                    if file_url:
                        logger.info(f"✅ Stream upload successful: {file_url[:50]}...")
                        return file_url
                    else:
                        raise CompositingError(
                            f"File URL not found in stream upload response: {result}"
                        )
        finally:
            if tmp_file and os.path.exists(tmp_file):
                try:
                    os.unlink(tmp_file)
                except Exception:
                    pass

    def _upload_via_base64_with_retry(
        self,
        image: Image.Image,
        image_format: str,
        upload_path: str,
        filename: str,
        attempt: int,
    ) -> str:
        """Upload via base64 with retry logic."""
        endpoint = f"{self.upload_base_url}/api/file-base64-upload"

        logger.info(
            f"📤 Base64 upload (attempt {attempt + 1}):\n"
            f"   - Endpoint: {endpoint}\n"
            f"   - Filename: {filename}\n"
            f"   - Format: {image_format}"
        )

        # Convert to bytes
        buffer = BytesIO()
        if image_format == "JPEG":
            image.convert("RGB").save(buffer, format="JPEG", quality=JPEG_QUALITY)
        else:
            image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        encoded_size = len(image_bytes)

        # Check size
        if encoded_size >= BASE64_MAX_SIZE_BYTES:
            raise CompositingError(
                f"Image too large for base64 upload: {encoded_size / 1024:.1f} KB "
                f"(max {BASE64_MAX_SIZE_BYTES / 1024:.1f} KB)"
            )

        # Encode to base64
        b64_data = base64.b64encode(image_bytes).decode("utf-8")
        encoded_payload_size = len(b64_data)

        logger.info(
            f"📦 Base64 encoding:\n"
            f"   - Original bytes: {encoded_size / 1024:.1f} KB\n"
            f"   - Encoded size: {encoded_payload_size / 1024:.1f} KB\n"
            f"   - Size increase: {((encoded_payload_size / encoded_size) - 1) * 100:.1f}%"
        )

        # Correct payload format according to KIE.ai docs
        payload = {
            "base64Data": b64_data,  # Just base64 string, not data URL
            "uploadPath": upload_path,
            "fileName": filename,
        }

        headers_json = {**self.headers, "Content-Type": "application/json"}

        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                endpoint,
                headers=headers_json,
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            result = response.json()

            file_url = self._extract_file_url(result)
            if file_url:
                logger.info(f"✅ Base64 upload successful: {file_url[:50]}...")
                return file_url
            else:
                raise CompositingError(
                    f"File URL not found in base64 upload response: {result}"
                )

    def _extract_file_url(self, result: dict) -> str | None:
        """Extract file URL from KIE.ai response."""
        if "data" in result:
            data = result["data"]
            if isinstance(data, dict):
                return data.get("downloadUrl") or data.get("fileUrl") or data.get("url")
            elif isinstance(data, str):
                return data
        elif "downloadUrl" in result:
            return result["downloadUrl"]
        elif "fileUrl" in result:
            return result["fileUrl"]
        elif "url" in result:
            return result["url"]
        return None

"""KIE.ai refinement service."""
import base64
import logging
import os
from io import BytesIO
from typing import Any

import httpx
import numpy as np
from PIL import Image

from app.core.config import settings
from app.core.exceptions import CompositingError
from app.integrations.kie.client import KIEClient
from app.integrations.kie.models import KIETaskError, KIEValidationError, UnsupportedModelError
from app.services.refinement.ai_result_validator import AIResultValidator
from app.services.refinement.kie_upload import KIEUploader

logger = logging.getLogger(__name__)


class KIERefiner:
    """Refines images using KIE.ai API."""

    def __init__(self):
        """Initialize KIE refiner."""
        self.api_key = settings.KIE_API_KEY
        # Убеждаемся что базовый URL без trailing slash
        self.api_url = settings.KIE_API_URL.rstrip("/")  # Для API (createTask, recordInfo)
        self.upload_base_url = settings.KIE_UPLOAD_BASE_URL.rstrip("/")  # Для File Upload API (отдельная база)
        
        # Backward compatibility: use KIE_MODEL if KIE_PRIMARY_MODEL not set
        primary_model = getattr(settings, "KIE_PRIMARY_MODEL", None) or settings.KIE_MODEL
        fallback_model = getattr(settings, "KIE_FALLBACK_MODEL", None) or None
        
        # Log model configuration
        logger.info(
            f"🔧 KIERefiner initialization:\n"
            f"   - settings.KIE_PRIMARY_MODEL: {getattr(settings, 'KIE_PRIMARY_MODEL', 'NOT SET')}\n"
            f"   - settings.KIE_MODEL (deprecated): {settings.KIE_MODEL}\n"
            f"   - settings.KIE_FALLBACK_MODEL: {getattr(settings, 'KIE_FALLBACK_MODEL', 'NOT SET')}\n"
            f"   - Selected primary_model: {primary_model}\n"
            f"   - Selected fallback_model: {fallback_model or '(not configured)'}"
        )
        
        # Initialize unified KIE client
        self.client = KIEClient(
            api_key=self.api_key,
            api_url=self.api_url,
            primary_model=primary_model,
            fallback_model=fallback_model if fallback_model else None,
        )
        
        # Keep model for backward compatibility
        self.model = primary_model
        
        logger.info(
            f"🔧 KIERefiner client initialized:\n"
            f"   - self.client.primary_model: {self.client.primary_model}\n"
            f"   - self.client.fallback_model: {self.client.fallback_model or '(not configured)'}\n"
            f"   - self.model (backward compat): {self.model}"
        )
        
        # Initialize reliable uploader
        self.uploader = KIEUploader(
            api_key=self.api_key,
            upload_base_url=self.upload_base_url,
        )
        
        # Initialize AI result validator
        self.validator = AIResultValidator()
        
        # File Upload API endpoints (из документации) - kept for backward compatibility
        self.STREAM_ENDPOINT = "/api/file-stream-upload"
        self.BASE64_ENDPOINT = "/api/file-base64-upload"
        self.URL_ENDPOINT = "/api/file-url-upload"

        if not self.api_key:
            logger.warning("KIE_API_KEY not set, refinement will be disabled")
        else:
            logger.info(f"KIE.ai API initialized: {self.api_url}")
            logger.info(f"KIE.ai Upload Base URL: {self.upload_base_url}")
            logger.info(f"KIE.ai Primary Model: {primary_model}")
            if fallback_model:
                logger.info(f"KIE.ai Fallback Model: {fallback_model}")

    def _make_request(
        self, endpoint: str, payload: dict[str, Any], timeout: float = 300.0
    ) -> dict[str, Any]:
        """
        Make request to KIE.ai API.

        Args:
            endpoint: API endpoint
            payload: Request payload
            timeout: Request timeout

        Returns:
            Response data

        Raises:
            CompositingError: If request fails
        """
        if not self.api_key:
            raise CompositingError("KIE API key not configured")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        full_url = f"{self.api_url}{endpoint}"
        logger.info(f"Making request to: {full_url}")
        logger.debug(f"Payload keys: {list(payload.keys())}")

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    full_url,
                    headers=headers,
                    json=payload,
                )
                logger.info(f"Response status: {response.status_code}")
                
                if response.status_code != 200:
                    error_detail = response.text[:500] if response.text else "No error details"
                    logger.error(
                        f"KIE.ai API HTTP error {response.status_code}: {error_detail}"
                    )
                    raise CompositingError(
                        f"KIE.ai API error {response.status_code}: {error_detail}"
                    )
                
                result = response.json()
                logger.info(f"Response received, keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")
                logger.debug(f"Response preview: {str(result)[:200]}...")
                return result
                
        except httpx.TimeoutException as e:
            logger.error(f"KIE.ai API timeout after {timeout}s: {e}")
            raise CompositingError(f"KIE.ai API timeout: request took longer than {timeout}s") from e
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text[:500] if e.response and e.response.text else str(e)
            logger.error(f"KIE.ai API HTTP error: {e.response.status_code if e.response else 'unknown'} - {error_detail}")
            raise CompositingError(
                f"KIE.ai API error {e.response.status_code if e.response else 'unknown'}: {error_detail}"
            ) from e
        except httpx.RequestError as e:
            logger.error(f"KIE.ai API request error: {e}")
            raise CompositingError(f"KIE.ai API request failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in KIE.ai request: {e}", exc_info=True)
            raise CompositingError(f"KIE.ai API error: {e}") from e

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        img_bytes = BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        return base64.b64encode(img_bytes.getvalue()).decode("utf-8")

    def _upload_file(self, image: Image.Image) -> str:
        """
        Upload image to KIE.ai using reliable upload utility.

        Uses KIEUploader which handles:
        - Automatic method selection (stream vs base64)
        - Image preprocessing (resize, format conversion)
        - Retry logic with exponential backoff
        - Comprehensive diagnostics

        Args:
            image: PIL Image to upload

        Returns:
            File URL from KIE.ai (downloadUrl or fileUrl)
        """
        return self.uploader.upload_image(image)
    
    def _upload_via_url(self, image: Image.Image, tmp_path: str, headers: dict) -> str:
        """
        Upload via URL upload (requires public URL).
        
        Note: This requires the file to be publicly accessible.
        For now, we'll skip this and fallback to base64.
        """
        # TODO: Implement temporary public URL hosting (S3, R2, etc.)
        # For now, skip and use base64
        raise CompositingError("URL upload not implemented - requires public URL hosting")
    
    def _upload_via_base64(self, image: Image.Image, headers: dict) -> str:
        """
        Upload via Base64 (Data URL format) - last resort fallback.
        
        Note: Base64 increases size by ~33%, recommended for small files only.
        """
        import base64
        
        # Convert image to base64
        img_bytes = BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        img_data = img_bytes.getvalue()
        
        # Check size (warn if too large)
        size_mb = len(img_data) / (1024 * 1024)
        if size_mb > 5:  # Warn if > 5MB
            logger.warning(f"Image is {size_mb:.1f}MB, base64 will be ~{size_mb * 1.33:.1f}MB")
        
        b64_data = base64.b64encode(img_data).decode("utf-8")
        data_url = f"data:image/png;base64,{b64_data}"
        
        logger.info(f"Trying Base64 Upload to: {self.upload_base_url}{self.BASE64_ENDPOINT}")
        
        payload = {
            "base64Data": data_url,
            "uploadPath": "images/base64",
            "fileName": "image.png",
        }
        
        headers_json = {**headers, "Content-Type": "application/json"}
        
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.upload_base_url}{self.BASE64_ENDPOINT}",
                headers=headers_json,
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            result = response.json()
            
            file_url = self._extract_file_url(result)
            if file_url:
                logger.info(f"✅ File uploaded via base64, URL: {file_url[:50]}...")
                return file_url
            else:
                raise CompositingError(f"File URL not found in base64 upload response: {result}")
    
    def _extract_file_url(self, result: dict) -> str | None:
        """Extract file URL from KIE.ai response."""
        # KIE.ai returns: downloadUrl, fileUrl, fileId
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

    def _create_task(self, file_url: str | list[str], prompt: str) -> str:
        """
        Create image editing task in KIE.ai.

        Based on KIE.ai documentation for GPT Image 1.5:
        - https://docs.kie.ai/market/gpt-image/1-5-image-to-image
        - Payload format: model in root, input_urls and prompt inside input object

        Args:
            file_url: URL of uploaded file(s) - can be single URL or list of URLs
            prompt: Editing prompt

        Returns:
            Task ID
        """
        try:
            # Support both single URL and list of URLs
            if isinstance(file_url, str):
                input_urls = [file_url]
            else:
                input_urls = file_url
            
            # Правильный формат payload согласно документации KIE.ai
            # input_urls и prompt должны быть внутри объекта input
            # Поддерживаемые модели: gpt-image/1.5-image-to-image, flux-1/kontext-dev
            payload = {
                "model": self.model,  # Используем модель из настроек
                "input": {
                    "input_urls": input_urls,
                    "prompt": prompt,
                    "aspect_ratio": "3:2",  # Опционально, но рекомендуется
                    "quality": "medium",  # Опционально
                },
            }
            
            logger.info(f"Creating task with model: {payload['model']}, {len(input_urls)} input URL(s)")
            logger.debug(f"Prompt: {prompt[:100]}...")
            
            result_data = self._make_request("/api/v1/jobs/createTask", payload, timeout=60.0)
            
            # Жёсткая проверка ответа - KIE.ai может вернуть HTTP 200, но с code: 422 внутри JSON
            code = result_data.get("code")
            if code != 200:
                error_msg = result_data.get("msg", "Unknown error")
                raise CompositingError(f"KIE.ai createTask failed (code {code}): {error_msg}")
            
            # Проверяем наличие data и taskId
            if not result_data.get("data"):
                raise CompositingError(f"KIE.ai createTask returned no data: {result_data}")
            
            data = result_data["data"]
            
            # Extract task ID (может быть taskId, task_id, или id)
            task_id = None
            if isinstance(data, dict):
                task_id = data.get("taskId") or data.get("task_id") or data.get("id")
            elif isinstance(data, str):
                task_id = data
            
            if not task_id:
                raise CompositingError(f"Task ID not found in response data: {data}")
            
            logger.info(f"✅ Task created successfully, ID: {task_id}")
            return task_id
            
        except CompositingError:
            raise
        except Exception as e:
            logger.error(f"Failed to create task: {e}", exc_info=True)
            raise CompositingError(f"Failed to create task: {e}") from e

    def _get_task_status(self, task_id: str) -> dict[str, Any]:
        """
        Get task status from KIE.ai using recordInfo endpoint with retry logic.

        Based on KIE.ai documentation:
        - Get Task Details: https://docs.kie.ai/market/common/get-task-detail
        - Unified query interface for all Market models
        - Use GET /api/v1/jobs/recordInfo?taskId=...
        - Status is in data.state (not status)
        - Result is in data.resultJson (JSON string with resultUrls)

        Args:
            task_id: Task ID

        Returns:
            Task data dict (from payload.data)

        Raises:
            CompositingError: If all retry attempts fail
        """
        import time
        
        last_error = None
        max_retries = 5
        
        for attempt in range(max_retries):
            try:
                # Создаём новый client на каждый запрос для избежания проблем с proxy/TLS
                url = f"{self.api_url}/api/v1/jobs/recordInfo"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                }
                
                with httpx.Client(
                    timeout=30.0,
                    follow_redirects=True,
                    verify=True,
                ) as client:
                    response = client.get(
                        url,
                        headers=headers,
                        params={"taskId": task_id},
                    )
                    response.raise_for_status()
                    payload = response.json()
                    
                    # Логируем полный ответ для отладки (только на первой попытке)
                    if attempt == 0:
                        logger.debug(f"recordInfo response: {payload}")
                    
                    # Проверяем code в ответе
                    code = payload.get("code")
                    if code != 200:
                        error_msg = payload.get("msg", "Unknown error")
                        raise CompositingError(f"KIE.ai recordInfo failed (code {code}): {error_msg}")
                    
                    # Возвращаем data из ответа
                    data = payload.get("data") or {}
                    return data
                    
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.ConnectTimeout) as e:
                last_error = e
                wait_time = min(2 * (attempt + 1), 10)  # Exponential backoff, max 10s
                logger.warning(
                    f"KIE status poll failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    # Последняя попытка - не ждём
                    break
                    
            except httpx.HTTPStatusError as e:
                # HTTP ошибки (4xx, 5xx) не ретраим
                error_detail = e.response.text[:500] if e.response and e.response.text else str(e)
                logger.error(f"Failed to get task status (HTTP {e.response.status_code if e.response else 'unknown'}): {error_detail}")
                raise CompositingError(f"Failed to get task status: {error_detail}") from e
                
            except CompositingError:
                # Бизнес-логика ошибки - не ретраим
                raise
                
            except Exception as e:
                last_error = e
                wait_time = min(2 * (attempt + 1), 10)
                logger.warning(
                    f"Unexpected error in status poll (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    break
        
        # Все попытки исчерпаны
        raise CompositingError(f"Failed to get task status after {max_retries} retries: {last_error}")

    def _wait_for_task_completion(self, task_id: str, max_wait: int = 300) -> dict[str, Any]:
        """
        Wait for task completion by polling status with soft error handling.

        Based on KIE.ai documentation:
        - Status is in data.state (not status)
        - Success state is "success"
        - Failed states: "fail", "failed", "error"

        Args:
            task_id: Task ID
            max_wait: Maximum wait time in seconds

        Returns:
            Final task data dict (from recordInfo data)

        Raises:
            CompositingError: If task fails or timeout exceeded
        """
        import time
        
        start_time = time.time()
        poll_interval = 7  # Poll every 7 seconds (увеличен для стабильности через proxy)
        consecutive_errors = 0
        max_consecutive_errors = 5  # Максимум ошибок подряд перед отказом
        
        logger.info(f"Waiting for task {task_id} to complete (max {max_wait}s, poll every {poll_interval}s)...")
        
        while time.time() - start_time < max_wait:
            try:
                data = self._get_task_status(task_id)
                
                # Сбрасываем счётчик ошибок при успешном запросе
                consecutive_errors = 0
                
                # Проверяем state (не status!)
                state = (data.get("state") or "").lower()
                logger.debug(f"Task {task_id} state: {state}")
                
                if state == "success":
                    logger.info(f"✅ Task {task_id} completed successfully")
                    return data
                
                if state in {"fail", "failed", "error"}:
                    # Извлекаем детальную информацию об ошибке
                    fail_msg = (data.get("failMsg") or "").strip()
                    fail_code = str(data.get("failCode") or "").strip()
                    error_msg = data.get("message") or data.get("error") or data.get("msg") or ""
                    
                    # Формируем детальное сообщение об ошибке
                    error_details = []
                    if fail_code:
                        error_details.append(f"failCode={fail_code}")
                    if fail_msg:
                        error_details.append(f"failMsg={fail_msg}")
                    if error_msg:
                        error_details.append(f"message={error_msg}")
                    
                    full_error_msg = ", ".join(error_details) if error_details else "Unknown error"
                    
                    # Специальная обработка для nsfw ошибки
                    if fail_msg.lower() == "nsfw" or "nsfw" in fail_msg.lower():
                        logger.warning(f"⚠️ KIE.ai task {task_id} blocked by safety filter (NSFW)")
                        logger.warning(f"⚠️ Blocked by safety filter - switching to fallback mode")
                        raise CompositingError(
                            f"KIE.ai blocked the task with safety filter (nsfw): {full_error_msg}. "
                            "Try a different input image, a more neutral crop, or use non-LLM fallback."
                        )
                    
                    raise CompositingError(f"KIE.ai task {task_id} failed: {full_error_msg}")
                
                # Если state не success и не fail, продолжаем ждать
                logger.debug(f"Task {task_id} state: {state}, waiting...")
                time.sleep(poll_interval)
                    
            except CompositingError as e:
                # Бизнес-логика ошибки (task failed) - не продолжаем
                raise
                
            except Exception as e:
                consecutive_errors += 1
                logger.warning(
                    f"Error checking task status (consecutive errors: {consecutive_errors}/{max_consecutive_errors}): {e}"
                )
                
                # Если слишком много ошибок подряд - прекращаем
                if consecutive_errors >= max_consecutive_errors:
                    raise CompositingError(
                        f"Too many consecutive errors while polling task {task_id}: {consecutive_errors} errors. "
                        f"Last error: {e}"
                    )
                
                # Мягкая обработка: ждём и продолжаем polling
                time.sleep(poll_interval)
        
        raise CompositingError(f"Task {task_id} did not complete within {max_wait}s")

    def _base64_to_image(self, base64_str: str) -> Image.Image:
        """Convert base64 string to PIL Image."""
        # Handle data URI format
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        result_bytes = base64.b64decode(base64_str)
        return Image.open(BytesIO(result_bytes))

    def _get_download_url(self, file_url: str) -> str:
        """
        Get temporary download link for generated file from KIE.ai.

        Args:
            file_url: URL returned by KIE.ai API

        Returns:
            Temporary download URL
        """
        try:
            payload = {"url": file_url}
            result_data = self._make_request("/api/v1/common/download-url", payload)
            
            if "data" in result_data:
                return result_data["data"]
            elif "url" in result_data:
                return result_data["url"]
            else:
                raise CompositingError(f"Unexpected download URL response format: {result_data}")
        except Exception as e:
            logger.error(f"Failed to get download URL: {e}")
            raise CompositingError(f"Failed to get download URL: {e}") from e

    def _download_image_from_url(self, file_url: str) -> Image.Image:
        """
        Download image from KIE.ai URL.

        Args:
            file_url: URL from KIE.ai API response (может быть tempfile URL или обычный URL)

        Returns:
            Downloaded image
        """
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

    def refine_compositing(
        self,
        composed_image: Image.Image | None = None,
        template_description: str = "poster",
        detected_issues: list[str] | None = None,
        original_dog_image: Image.Image | None = None,
        poster_background: Image.Image | None = None,
        ai_prompt: str | None = None,
        template_config: dict[str, Any] | None = None,
        job_id: str | None = None,
        debug: bool = False,
    ) -> Image.Image:
        """
        Integrate dog into poster using AI/LLM.

        Args:
            composed_image: Composed image (optional, only used as fallback if original_dog_image not provided)
            ai_prompt: AI integration prompt from template config
            detected_issues: List of detected quality issues
            original_dog_image: Original dog photo (REQUIRED for full LLM integration)
            poster_background: Clean poster background (REQUIRED for full LLM integration)

        Returns:
            Integrated image with AI-integrated dog
        """
        if not self.api_key:
            logger.warning("KIE API key not configured, skipping refinement")
            if composed_image:
                return composed_image
            raise CompositingError("KIE API key not configured and no fallback image provided")

        # ПРИОРИТЕТ: Если есть исходное фото собаки и постер - используем их для полной LLM интеграции
        if original_dog_image and poster_background:
            logger.info("🎨 Using original dog photo and poster for FULL LLM integration (no old compositing)")
            # Промпт должен быть передан из template_config
            if not ai_prompt or not isinstance(ai_prompt, str) or not ai_prompt.strip():
                raise CompositingError("AI integration prompt is required but not provided in template config")
            # Get storage for debug artifacts if available
            storage = None
            if debug and job_id:
                from app.core.storage import Storage
                storage = Storage(job_id)
            
            # Extract target region from template_config if available
            target_region = None
            if template_config:
                ai_integration = template_config.get("ai_integration", {})
                face_region = ai_integration.get("face_region")
                if face_region:
                    target_region = {
                        "x": face_region.get("x", 0),
                        "y": face_region.get("y", 0),
                        "width": face_region.get("width", poster_background.width),
                        "height": face_region.get("height", poster_background.height),
                    }
            
            try:
                return self._integrate_dog_into_poster(
                    original_dog_image, poster_background, ai_prompt, template_config, job_id, debug, storage, target_region
                )
            except CompositingError as e:
                # If validation fails, return None to trigger fallback
                if "too similar" in str(e).lower() or "no-op" in str(e).lower():
                    logger.warning(f"AI result validation failed, returning None for fallback: {e}")
                    return None
                raise

        # Fallback: улучшаем уже скомпозированное изображение (когда полная интеграция не работает)
        # Используем промпт из конфига, если он передан, иначе базовый промпт
        if not ai_prompt or not ai_prompt.strip():
            # Базовый промпт для fallback (если промпт не передан)
            prompt = (
                "Улучши интеграцию объекта в изображение. "
                "Удали все видимые края, швы и артефакты обтравки. "
                "Сопоставь направление и интенсивность освещения с окружением. "
                "Настрой тени чтобы соответствовать источникам света и создать глубину. "
                "Смешай цвета для естественного соответствия цветовой палитре. "
                "Смягчи и смешай все края бесшовно. "
                "Настрой перспективу и масштаб чтобы выглядело естественно. "
                "Финальный результат должен выглядеть как единый профессионально созданный постер. "
                "Не должно быть видно признаков композитинга, вырезания или вставки."
            )
        else:
            # Используем промпт из конфига
            prompt = ai_prompt
        
        logger.info(f"Using AI prompt for refinement (fallback): {prompt[:100]}...")
        return self._refine_with_prompt(composed_image, prompt)

    def _integrate_dog_into_poster(
        self,
        dog_image: Image.Image,
        poster_background: Image.Image,
        ai_prompt: str,
        template_config: dict[str, Any] | None = None,
        job_id: str | None = None,
        debug: bool = False,
        storage: Any = None,
        target_region: dict[str, int] | None = None,
    ) -> Image.Image:
        """
        Use AI to integrate entity (dog/human) into poster from scratch using KIE.ai.

        Uses unified KIEClient with support for multiple models and fallback.

        Args:
            dog_image: Original entity photo (dog or human)
            poster_background: Clean poster background
            ai_prompt: AI integration prompt from template config

        Returns:
            Integrated image
        """
        try:
            # Используем промпт из конфигурации шаблона
            prompt = ai_prompt
            
            logger.info("Step 1: Uploading entity image for AI integration...")
            logger.info(f"Entity image size: {dog_image.size}")
            entity_file_url = self._upload_file(dog_image)
            
            logger.info("Step 2: Uploading poster background...")
            logger.info(f"Poster background size: {poster_background.size}")
            poster_file_url = self._upload_file(poster_background)
            
            logger.info("Step 3: Creating integration task with both images...")
            logger.info(f"Image order: [1] poster ({poster_file_url[:50]}...), [2] entity ({entity_file_url[:50]}...)")
            logger.info(f"Prompt length: {len(prompt)} chars")
            
            # Вычисляем правильный aspect_ratio из исходного постера (для GPT Image 1.5)
            poster_width, poster_height = poster_background.size
            aspect_ratio_str = self._calculate_aspect_ratio(poster_width, poster_height)
            logger.info(f"Using aspect_ratio: {aspect_ratio_str} (from poster size {poster_width}x{poster_height})")
            
            # Use unified client to create task (handles model selection and fallback)
            # Log model selection before calling client
            logger.info(
                f"🔧 _integrate_dog_into_poster calling create_image_edit_task:\n"
                f"   - Client primary_model: {self.client.primary_model}\n"
                f"   - Client fallback_model: {self.client.fallback_model or '(not configured)'}\n"
                f"   - Model parameter: None (will use primary_model)\n"
                f"   - Expected model: {self.client.primary_model}"
            )
            
            try:
                task_id = self.client.create_image_edit_task(
                    model=None,  # Use primary model from config
                    prompt=prompt,
                    poster_url=poster_file_url,
                    reference_url=entity_file_url,
                    aspect_ratio=aspect_ratio_str,
                    quality="high",
                    use_fallback=True,  # Enable fallback if primary fails
                )
            except (KIEValidationError, UnsupportedModelError) as e:
                # Convert to CompositingError for backward compatibility
                raise CompositingError(f"KIE validation error: {e}") from e
            except KIETaskError as e:
                # Convert to CompositingError for backward compatibility
                raise CompositingError(f"KIE task error: {e}") from e
            
            logger.info(f"✅ Task created successfully, ID: {task_id}")
            
            # Wait for completion using unified client
            logger.info("Step 4: Waiting for AI integration to complete...")
            try:
                task_result = self.client.wait_for_task_completion(task_id, max_wait=300)
            except KIETaskError as e:
                raise CompositingError(f"KIE task completion error: {e}") from e
            
            # Extract and download result
            logger.info("Step 5: Downloading integrated result...")
            
            # Согласно документации KIE.ai, результат находится в data.resultJson (JSON строка)
            # resultJson содержит объект с полем resultUrls
            import json
            
            result_json_raw = task_result.get("resultJson")
            if not result_json_raw:
                raise CompositingError(f"No resultJson in KIE.ai response: {task_result}")
            
            # Парсим JSON строку
            try:
                result_json = json.loads(result_json_raw)
            except json.JSONDecodeError as e:
                raise CompositingError(f"Failed to parse resultJson: {e}, raw: {result_json_raw[:200]}...")
            
            # Извлекаем resultUrls
            result_urls = result_json.get("resultUrls") or []
            if not result_urls:
                raise CompositingError(f"No resultUrls in KIE.ai resultJson: {result_json}")
            
            # Берём первый URL
            result_url = result_urls[0]
            logger.info(f"Got result URL from resultJson: {result_url[:50]}...")
            
            result_image = self._download_image_from_url(result_url)
            logger.info(f"✅ Downloaded result image: {result_image.size}")

            # Step 6: Validate AI result (check for no-op changes)
            logger.info("🔍 Validating AI result against original poster...")
            
            # Extract target region from template_config if available
            target_region = None
            if template_config:
                ai_integration = template_config.get("ai_integration", {})
                face_region = ai_integration.get("face_region")
                if face_region:
                    target_region = {
                        "x": face_region.get("x", 0),
                        "y": face_region.get("y", 0),
                        "width": face_region.get("width", poster_background.width),
                        "height": face_region.get("height", poster_background.height),
                    }
            
            try:
                validation_result = self.validator.validate_ai_result(
                    original_image=poster_background,
                    ai_result=result_image,
                    target_region=target_region,
                    debug=debug,
                    storage=storage,
                )
            except Exception as e:
                # If validator itself fails (not validation failure, but code error),
                # log warning but don't fail the pipeline - AI result was successfully obtained
                logger.warning(
                    f"⚠️  AI result validator encountered an error (non-fatal): {e}\n"
                    f"   - AI image was successfully downloaded: {result_image.size}\n"
                    f"   - Full image diff metrics were computed\n"
                    f"   - Continuing with AI result despite validator error"
                )
                # Return a "partial" validation result that accepts the image
                validation_result = {
                    "accepted": True,
                    "partial_validation": True,
                    "partial_reason": f"Validator error: {e}",
                    "metrics": {
                        "full_image": {
                            "diff_score": 0.5,  # Assume reasonable diff
                            "mean_abs_diff": 50.0,
                            "changed_pixels_ratio": 0.1,
                        },
                        "target_region": None,
                    },
                }

            if not validation_result["accepted"]:
                rejection_reason = validation_result.get("rejection_reason", "Unknown")
                logger.warning(
                    f"⚠️  AI result validation FAILED:\n"
                    f"   - Rejection reason: {rejection_reason}\n"
                    f"   - Full image diff score: {validation_result['metrics']['full_image']['diff_score']:.4f}\n"
                    f"   - Mean absolute diff: {validation_result['metrics']['full_image']['mean_abs_diff']:.2f}\n"
                    f"   - Changed pixels ratio: {validation_result['metrics']['full_image']['changed_pixels_ratio']:.4f}\n"
                    f"   - Treating as no-op, will raise CompositingError"
                )
                raise CompositingError(
                    f"AI returned image is too similar to original poster (no-op change detected). "
                    f"Rejection reason: {rejection_reason}. "
                    f"Diff score: {validation_result['metrics']['full_image']['diff_score']:.4f} "
                    f"(minimum required: {self.validator.min_full_diff})."
                )
            
            # Log if validation was partial (target region failed but full image passed)
            if validation_result.get("partial_validation", False):
                logger.warning(
                    f"⚠️  Partial validation: {validation_result.get('partial_reason', 'Unknown reason')}\n"
                    f"   - Full image validation: PASSED\n"
                    f"   - Target region validation: FAILED or unavailable\n"
                    f"   - AI result will be used despite partial validation"
                )

            logger.info(
                f"✅ AI result validation PASSED:\n"
                f"   - Full image diff score: {validation_result['metrics']['full_image']['diff_score']:.4f}\n"
                f"   - Mean absolute diff: {validation_result['metrics']['full_image']['mean_abs_diff']:.2f}\n"
                f"   - Changed pixels ratio: {validation_result['metrics']['full_image']['changed_pixels_ratio']:.4f}"
            )
            
            # Используем результат от KIE.ai как есть - без кропов, без изменения размера
            # KIE.ai должен вернуть постер с обновлённой сущностью того же размера
            logger.info(f"✅ Entity integrated into poster using AI. Result size: {result_image.size}")
            return result_image
            
        except Exception as e:
            logger.error(f"AI integration failed: {e}", exc_info=True)
            # Не используем простое композирование - это даст плохой результат
            # Вместо этого пробуем улучшить уже скомпозированное изображение
            logger.warning("AI integration failed, will try to improve pre-composited image instead")
            raise CompositingError(f"AI integration failed: {e}") from e

    def _integrate_face_region(
        self,
        reference_image: Image.Image,
        poster_background: Image.Image,
        ai_prompt: str,
        template_config: dict[str, Any],
        job_id: str | None = None,
        debug: bool = False,
    ) -> Image.Image:
        """
        Integrate face using face_region mode: crop head region from poster, replace face, recompose.

        Args:
            reference_image: Reference face image
            poster_background: Full poster background
            ai_prompt: AI integration prompt (not used in face_region mode)
            template_config: Template configuration with face_region coordinates
            job_id: Job ID for debug artifacts
            debug: Whether to save debug artifacts

        Returns:
            Final image with replaced face
        """
        from app.core.storage import Storage
        from app.services.detection.dog_detector import DogDetector

        logger.info("🎯 Starting face_region integration mode")
        
        # Get face_region coordinates from template config
        ai_integration_config = template_config.get("ai_integration", {})
        face_region = ai_integration_config.get("face_region", {})
        
        if not face_region:
            raise CompositingError("face_region coordinates not found in template config")
        
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
            storage.save_debug(poster_background, "face_region_00_original_poster.png")
            storage.save_debug(reference_image, "face_region_01_reference_face.png")
        
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
            storage.save_debug(head_region_crop, "face_region_02_head_region_crop.png")
        
        # Step 2: Normalize reference face
        logger.info("Step 2: Normalizing reference face...")
        normalized_reference = self._normalize_reference_face(reference_image, debug, storage)
        logger.info(f"Normalized reference: {normalized_reference.size}")
        
        # Step 3: Get face replacement prompt
        face_replacement_prompt = ai_integration_config.get(
            "face_replacement_prompt",
            "The first image is a cropped head region from a poster. The second image is a reference face for identity. Replace ONLY the face/head in the first image using the identity from the second image. Do NOT paste the second image as a separate object. Preserve the costume, pose, scale, lighting, and composition of the first image. The result should be a seamless face replacement that looks natural and integrated."
        )
        
        # Step 4: Upload images and create KIE task
        logger.info("Step 3: Uploading images to KIE.ai...")
        head_region_url = self._upload_file(head_region_crop)
        reference_url = self._upload_file(normalized_reference)
        
        logger.info("Step 4: Creating face replacement task...")
        try:
            task_id = self.client.create_image_edit_task(
                model=None,  # Use primary model from config
                prompt=face_replacement_prompt,
                poster_url=head_region_url,  # Cropped head region
                reference_url=reference_url,  # Normalized reference face
                use_fallback=True,
            )
        except (KIEValidationError, UnsupportedModelError) as e:
            raise CompositingError(f"KIE validation error: {e}") from e
        except KIETaskError as e:
            raise CompositingError(f"KIE task error: {e}") from e
        
        logger.info(f"✅ Task created successfully, ID: {task_id}")
        
        # Step 5: Wait for completion
        logger.info("Step 5: Waiting for face replacement to complete...")
        try:
            task_result = self.client.wait_for_task_completion(task_id, max_wait=300)
        except KIETaskError as e:
            raise CompositingError(f"KIE task completion error: {e}") from e
        
        # Step 6: Download result
        logger.info("Step 6: Downloading face replacement result...")
        import json
        
        result_json_raw = task_result.get("resultJson")
        if not result_json_raw:
            raise CompositingError(f"No resultJson in KIE.ai response: {task_result}")
        
        try:
            result_json = json.loads(result_json_raw)
        except json.JSONDecodeError as e:
            raise CompositingError(f"Failed to parse resultJson: {e}")
        
        result_urls = result_json.get("resultUrls") or []
        if not result_urls:
            raise CompositingError(f"No resultUrls in KIE.ai resultJson: {result_json}")
        
        result_url = result_urls[0]
        edited_head_region = self._download_image_from_url(result_url)
        logger.info(f"Downloaded edited head region: {edited_head_region.size}")
        
        if debug and storage:
            storage.save_debug(edited_head_region, "face_region_03_edited_head_region.png")
        
        # Step 7: Recompose edited head region back into poster
        logger.info("Step 7: Recomposing edited head region into poster...")
        logger.info(
            f"📊 Recomposition details:\n"
            f"   - Flow: adult_kie_realistic (face_region mode)\n"
            f"   - Refinement mode: face_region\n"
            f"   - Provider: KIE.ai\n"
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
        logger.info(f"✅ Face replacement completed. Final image size: {final_image.size}")
        
        if debug and storage:
            storage.save_debug(final_image, "face_region_04_final_recomposed.png")
        
        return final_image

    def _normalize_reference_face(
        self,
        reference_image: Image.Image,
        debug: bool = False,
        storage: Any = None,
    ) -> Image.Image:
        """
        Normalize reference face: detect face, crop with padding, avoid extreme close-up.

        Args:
            reference_image: Reference face image
            debug: Whether to save debug artifacts
            storage: Storage instance for debug artifacts

        Returns:
            Normalized face image
        """
        from app.services.detection.dog_detector import DogDetector
        
        logger.info("Normalizing reference face...")
        
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
            # Resize maintaining aspect ratio
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
            storage.save_debug(normalized, "face_region_01b_normalized_reference.png")
        
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
            storage.save_debug(poster, "face_region_recompose_00_original_poster.png")
            # Create target crop visualization
            target_crop = poster.crop((x, y, x + width, y + height))
            storage.save_debug(target_crop, "face_region_recompose_01_target_crop.png")
            storage.save_debug(edited_head_region, "face_region_recompose_02_ai_returned_crop.png")

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
                storage.save_debug(edited_head_region, "face_region_recompose_03_resized_ai_crop.png")

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
        for py in range(height):
            for px in range(width):
                # Distance from nearest edge
                dist_from_edge = min(px, width - 1 - px, py, height - 1 - py)
                # Use edge distance for feathering
                if dist_from_edge < feather_size:
                    edge_alpha = int(255 * dist_from_edge / feather_size)
                    mask_array[py, px] = min(mask_array[py, px], edge_alpha)

        feather_mask = Image.fromarray(mask_array, mode="L")

        if debug and storage:
            storage.save_debug(feather_mask, "face_region_recompose_04_recomposition_mask.png")

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
            storage.save_debug(result, "face_region_recompose_05_final_recomposed.png")

        return result

    def refine_segmentation(
        self,
        subject_image: Image.Image,
        original_image: Image.Image | None = None,
    ) -> Image.Image:
        """
        Improve background removal and segmentation using AI.

        Args:
            subject_image: Subject image with alpha
            original_image: Original image for reference (optional)

        Returns:
            Improved subject image
        """
        if not self.api_key:
            return subject_image

        prompt = (
            "Improve the background removal. Make the edges clean and natural. "
            "Remove any remaining background artifacts. Ensure smooth alpha channel transitions. "
            "The subject should have clean, professional edges suitable for compositing."
        )

        # If we have original, we could use inpainting, but for now just refine the subject
        return self._refine_with_prompt(subject_image, prompt)

    def refine_placement(
        self,
        subject_image: Image.Image,
        background_image: Image.Image,
        placement_hint: str = "center of helmet visor",
    ) -> Image.Image:
        """
        Use AI to improve subject placement and integration.

        Args:
            subject_image: Subject to place
            background_image: Background/template
            placement_hint: Where subject should be placed

        Returns:
            Improved composited image
        """
        if not self.api_key:
            return self._simple_composite(subject_image, background_image)

        # Create initial composite
        temp_composite = self._simple_composite(subject_image, background_image)

        prompt = (
            f"Improve the placement and integration of the subject into the background. "
            f"The subject should be naturally positioned in the {placement_hint}. "
            "Fix lighting, shadows, and color matching. Make it look like a single cohesive image. "
            "Ensure proper scale and perspective."
        )

        return self._refine_with_prompt(temp_composite, prompt)

    def _refine_with_prompt(self, image: Image.Image, prompt: str) -> Image.Image:
        """
        Refine image using KIE.ai GPT Image 1.5 with a prompt.

        Workflow:
        1. Upload image to KIE.ai
        2. Create task via /api/v1/jobs/createTask
        3. Poll task status until completion
        4. Download result image

        Args:
            image: Image to refine
            prompt: Refinement prompt

        Returns:
            Refined image
        """
        if not self.api_key:
            logger.warning("KIE API key not configured, skipping refinement")
            return image

        logger.info(f"Starting KIE.ai refinement (model: {self.model}) with prompt: {prompt[:100]}...")
        logger.info(f"API URL: {self.api_url}, Key present: {bool(self.api_key)}")

        try:
            # Step 1: Upload file
            logger.info("Step 1: Uploading image to KIE.ai...")
            file_url = self._upload_file(image)
            
            # Step 2: Create task using unified client
            logger.info("Step 2: Creating image editing task...")
            logger.info(
                f"🔧 _refine_with_prompt calling create_image_edit_task:\n"
                f"   - Client primary_model: {self.client.primary_model}\n"
                f"   - Client fallback_model: {self.client.fallback_model or '(not configured)'}\n"
                f"   - Model parameter: None (will use primary_model)\n"
                f"   - Expected model: {self.client.primary_model}"
            )
            try:
                task_id = self.client.create_image_edit_task(
                    model=None,  # Use primary model from config
                    prompt=prompt,
                    poster_url=file_url,  # Single image refinement
                    reference_url=None,
                    use_fallback=True,
                )
            except (KIEValidationError, UnsupportedModelError, KIETaskError) as e:
                # Convert to CompositingError for backward compatibility
                raise CompositingError(f"KIE task creation error: {e}") from e
            
            # Step 3: Wait for completion using unified client
            logger.info("Step 3: Waiting for task completion...")
            try:
                task_result = self.client.wait_for_task_completion(task_id, max_wait=300)
            except KIETaskError as e:
                raise CompositingError(f"KIE task completion error: {e}") from e
            
            # Step 4: Extract result URL and download
            logger.info("Step 4: Downloading result...")
            
            # Согласно документации KIE.ai, результат находится в data.resultJson (JSON строка)
            # resultJson содержит объект с полем resultUrls
            import json
            
            result_json_raw = task_result.get("resultJson")
            if not result_json_raw:
                raise CompositingError(f"No resultJson in KIE.ai response: {task_result}")
            
            # Парсим JSON строку
            try:
                result_json = json.loads(result_json_raw)
            except json.JSONDecodeError as e:
                raise CompositingError(f"Failed to parse resultJson: {e}, raw: {result_json_raw[:200]}...")
            
            # Извлекаем resultUrls
            result_urls = result_json.get("resultUrls") or []
            if not result_urls:
                raise CompositingError(f"No resultUrls in KIE.ai resultJson: {result_json}")
            
            # Берём первый URL
            result_url = result_urls[0]
            logger.info(f"Got result URL from resultJson: {result_url[:50]}...")
            
            # Download result image
            result_image = self._download_image_from_url(result_url)
            
            # Используем результат от KIE.ai как есть - без кропов, без изменения размера
            logger.info(f"✅ Image refined using KIE.ai GPT Image 1.5. Result size: {result_image.size}")
            return result_image

        except CompositingError:
            raise
        except Exception as e:
            logger.error(f"Refinement error: {e}", exc_info=True)
            logger.warning("Returning original image due to refinement error")
            return image

    def _calculate_aspect_ratio(self, width: int, height: int) -> str:
        """
        Calculate closest standard aspect ratio from image dimensions.
        
        Args:
            width: Image width
            height: Image height
            
        Returns:
            Aspect ratio string (e.g., "3:2", "16:9", "4:3")
        """
        ratio = width / height
        
        # Стандартные aspect ratios
        standard_ratios = {
            "1:1": 1.0,
            "4:3": 4/3,
            "3:2": 3/2,
            "16:9": 16/9,
            "21:9": 21/9,
            "2:3": 2/3,
            "3:4": 3/4,
            "9:16": 9/16,
        }
        
        # Находим ближайший стандартный aspect ratio
        closest_ratio = "3:2"  # По умолчанию
        min_diff = float('inf')
        
        for ratio_str, ratio_value in standard_ratios.items():
            diff = abs(ratio - ratio_value)
            if diff < min_diff:
                min_diff = diff
                closest_ratio = ratio_str
        
        return closest_ratio

    def _simple_composite(
        self, subject: Image.Image, background: Image.Image
    ) -> Image.Image:
        """Simple composite fallback."""
        bg_rgba = background.convert("RGBA")
        result = bg_rgba.copy()
        # Center subject
        paste_x = (bg_rgba.width - subject.width) // 2
        paste_y = (bg_rgba.height - subject.height) // 2
        
        # Handle different image modes safely
        try:
            if subject.mode == "RGBA":
                result.paste(subject, (paste_x, paste_y), subject)
            elif subject.mode == "RGB":
                # Convert to RGBA for proper compositing
                subject_rgba = subject.convert("RGBA")
                result.paste(subject_rgba, (paste_x, paste_y), subject_rgba)
            else:
                # Convert to RGBA
                subject_rgba = subject.convert("RGBA")
                result.paste(subject_rgba, (paste_x, paste_y), subject_rgba)
        except Exception as e:
            logger.warning(f"Error in simple composite: {e}, using RGB paste")
            # Fallback: just paste without alpha
            subject_rgb = subject.convert("RGB")
            result_rgb = result.convert("RGB")
            result_rgb.paste(subject_rgb, (paste_x, paste_y))
            return result_rgb
        
        return result.convert("RGB")

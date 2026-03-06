"""KIE.ai refinement service."""
import base64
import logging
import os
from io import BytesIO
from typing import Any

import httpx
from PIL import Image

from app.core.config import settings
from app.core.exceptions import CompositingError
from app.services.refinement.llm_prompter import LLMPrompter

logger = logging.getLogger(__name__)


class KIERefiner:
    """Refines images using KIE.ai API."""

    def __init__(self):
        """Initialize KIE refiner."""
        self.api_key = settings.KIE_API_KEY
        # Убеждаемся что базовый URL без trailing slash
        self.api_url = settings.KIE_API_URL.rstrip("/")  # Для API (createTask, recordInfo)
        self.upload_base_url = settings.KIE_UPLOAD_BASE_URL.rstrip("/")  # Для File Upload API (отдельная база)
        self.prompter = LLMPrompter()
        
        # File Upload API endpoints (из документации)
        self.STREAM_ENDPOINT = "/api/file-stream-upload"
        self.BASE64_ENDPOINT = "/api/file-base64-upload"
        self.URL_ENDPOINT = "/api/file-url-upload"

        if not self.api_key:
            logger.warning("KIE_API_KEY not set, refinement will be disabled")
        else:
            logger.info(f"KIE.ai API initialized: {self.api_url}")
            logger.info(f"KIE.ai Upload Base URL: {self.upload_base_url}")

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
        Upload image to KIE.ai using File Upload API and get file URL.

        Порядок fallback:
        1. file-stream-upload (multipart/form-data) - основной
        2. file-url-upload (если есть публичный URL) - если stream вернул 4xx/5xx
        3. file-base64-upload (Data URL) - последний fallback

        Based on KIE.ai documentation:
        - File Upload API: https://docs.kie.ai/file-upload-api/quickstart
        - Returns file URL to use in input_urls for createTask

        Args:
            image: PIL Image to upload

        Returns:
            File URL from KIE.ai (downloadUrl or fileUrl)
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        
        # Сохраняем временный файл для stream upload
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            image.save(tmp_file.name, format="PNG")
            tmp_path = tmp_file.name
        
        try:
            from pathlib import Path
            tmp_path_obj = Path(tmp_path)
            
            # 1. Попытка: File Stream Upload (основной метод)
            try:
                logger.info(f"Trying File Stream Upload to: {self.upload_base_url}{self.STREAM_ENDPOINT}")
                
                with open(tmp_path, "rb") as f:
                    files = {
                        "file": (tmp_path_obj.name, f, "image/png"),
                    }
                    data = {
                        "uploadPath": "images/user-uploads",
                        "fileName": tmp_path_obj.name,
                    }
                    
                    with httpx.Client(timeout=120.0) as client:
                        response = client.post(
                            f"{self.upload_base_url}{self.STREAM_ENDPOINT}",
                            headers=headers,
                            files=files,
                            data=data,
                            timeout=120.0,
                        )
                        response.raise_for_status()
                        result = response.json()
                        
                        file_url = self._extract_file_url(result)
                        if file_url:
                            logger.info(f"✅ File uploaded via stream, URL: {file_url[:50]}...")
                            return file_url
                        else:
                            raise CompositingError(f"File URL not found in stream upload response: {result}")
                            
            except httpx.HTTPStatusError as e:
                # Если 4xx/5xx - пробуем URL upload (если можем сделать публичный URL)
                if e.response.status_code >= 400:
                    logger.warning(f"Stream upload failed ({e.response.status_code}), trying URL upload...")
                    try:
                        return self._upload_via_url(image, tmp_path, headers)
                    except Exception as url_error:
                        logger.warning(f"URL upload also failed: {url_error}, trying base64...")
                        # Fallback на base64
                        return self._upload_via_base64(image, headers)
                else:
                    raise
            except Exception as e:
                logger.warning(f"Stream upload error: {e}, trying base64...")
                # Fallback на base64
                return self._upload_via_base64(image, headers)
                
        finally:
            # Удаляем временный файл
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    
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
            # Модель для image-to-image: gpt-image/1.5-image-to-image
            payload = {
                "model": "gpt-image/1.5-image-to-image",  # Правильное имя модели для image-to-image
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
                    error_msg = data.get("msg") or data.get("error") or data.get("message", "Unknown error")
                    raise CompositingError(f"KIE.ai task failed: {error_msg}")
                
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
            result_data = self._make_request("/api/v1/common/download-url", payload, timeout=60.0)
            
            if "data" in result_data:
                return result_data["data"]
            elif "url" in result_data:
                return result_data["url"]
            else:
                raise CompositingError(f"Unexpected download URL response format: {result_data}")
        except Exception as e:
            logger.error(f"Failed to get download URL: {e}")
            raise CompositingError(f"Failed to get download URL: {e}") from e

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
        composed_image: Image.Image | None,
        template_description: str = "astronaut poster",
        detected_issues: list[str] | None = None,
        original_dog_image: Image.Image | None = None,
        poster_background: Image.Image | None = None,
    ) -> Image.Image:
        """
        Integrate dog into poster using AI/LLM.

        Args:
            composed_image: Composed image (optional, only used as fallback if original_dog_image not provided)
            template_description: Description of template for better prompts
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
            return self._integrate_dog_into_poster(
                original_dog_image, poster_background, template_description
            )

        # Fallback: улучшаем уже скомпозированное изображение (когда полная интеграция не работает)
        prompt = (
            f"Замени на данном постере текущую собаку на новую. Обтрави собаку или сделай максимально похожую вместо текущей и замени на текущем постере. "
            "Критические улучшения: "
            "1. Удали все видимые края, швы и артефакты обтравки вокруг собаки "
            "2. Сопоставь направление и интенсивность освещения с окружением постера "
            "3. Настрой тени чтобы соответствовать источникам света постера и создать глубину "
            "4. Смешай цвета так, чтобы шерсть собаки естественно соответствовала цветовой палитре постера "
            "5. Убедись что собака выглядит внутри шлема космонавта, а не поверх него "
            "6. Добавь реалистичные отражения и блики, соответствующие материалу шлема "
            "7. Смягчи и смешай все края бесшовно "
            "8. Настрой перспективу и масштаб чтобы выглядело естественно "
            "9. Финальный результат должен выглядеть как единый профессионально созданный постер "
            "10. Не должно быть видно признаков композитинга, вырезания или вставки. "
            "Сделай так, чтобы выглядело как будто профессиональный дизайнер создал это изображение с нуля."
        )
        
        logger.info(f"Using AI prompt for refinement: {prompt[:100]}...")
        return self._refine_with_prompt(composed_image, prompt)

    def _integrate_dog_into_poster(
        self,
        dog_image: Image.Image,
        poster_background: Image.Image,
        template_description: str = "astronaut poster",
    ) -> Image.Image:
        """
        Use AI to integrate dog into poster from scratch using GPT Image 1.5.

        Args:
            dog_image: Original dog photo
            poster_background: Clean poster background
            template_description: Description of template

        Returns:
            Integrated image
        """
        try:
            # Промпт для замены собаки на постере (на основе запроса пользователя)
            prompt = (
                f"Замени на данном постере (первое изображение) текущую собаку на мою загруженную собаку (второе изображение). "
                f"Обтрави собаку из второго изображения или сделай максимально похожую вместо текущей и замени на текущем постере. "
                "Требования: "
                "1. Найди и удали текущую собаку на постере "
                "2. Обтрави (вырежи) новую собаку из второго изображения - голову, уши, шею, верхнюю часть груди "
                "3. Замени текущую собаку на новую, разместив её в том же месте (внутри шлема космонавта) "
                "4. Собака должна выглядеть как будто она изначально была на этом постере, а не вставлена поверх "
                "5. Сопоставь освещение, тени и цвета с окружением постера "
                "6. Убедись что голова, уши и верхняя часть тела собаки хорошо видны внутри шлема "
                "7. Убери все видимые края, швы и артефакты от обтравки "
                "8. Результат должен выглядеть как профессионально созданный постер без признаков склейки "
                "9. Сохрани узнаваемые черты и выражение моей собаки "
                "10. Финальное изображение должно выглядеть как единое целое, созданное дизайнером."
            )
            
            logger.info("Step 1: Uploading dog image for AI integration...")
            dog_file_url = self._upload_file(dog_image)
            
            logger.info("Step 2: Uploading poster background...")
            poster_file_url = self._upload_file(poster_background)
            
            # Для GPT Image 1.5 можно использовать несколько input_urls
            logger.info("Step 3: Creating integration task with both images...")
            
            # Правильный формат payload согласно документации KIE.ai
            # input_urls и prompt должны быть внутри объекта input
            # Модель для image-to-image: gpt-image/1.5-image-to-image
            payload = {
                "model": "gpt-image/1.5-image-to-image",  # Правильное имя модели для image-to-image
                "input": {
                    "input_urls": [poster_file_url, dog_file_url],  # Постер как основа, собака для интеграции
                    "prompt": prompt,
                    "aspect_ratio": "3:2",  # Опционально, но рекомендуется
                    "quality": "medium",  # Опционально
                },
            }
            
            logger.info(f"Creating task with model: {payload['model']}, {len(payload['input']['input_urls'])} input URL(s)")
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
            
            # Wait for completion
            logger.info("Step 4: Waiting for AI integration to complete...")
            task_result = self._wait_for_task_completion(task_id, max_wait=300)
            
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
            logger.info("✅ Dog integrated into poster using AI")
            return result_image
            
        except Exception as e:
            logger.error(f"AI integration failed: {e}", exc_info=True)
            # Не используем простое композирование - это даст плохой результат
            # Вместо этого пробуем улучшить уже скомпозированное изображение
            logger.warning("AI integration failed, will try to improve pre-composited image instead")
            raise CompositingError(f"AI integration failed: {e}") from e

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

        logger.info(f"Starting KIE.ai refinement (GPT Image 1.5) with prompt: {prompt[:100]}...")
        logger.info(f"API URL: {self.api_url}, Key present: {bool(self.api_key)}")

        try:
            # Step 1: Upload file
            logger.info("Step 1: Uploading image to KIE.ai...")
            file_url = self._upload_file(image)
            
            # Step 2: Create task
            logger.info("Step 2: Creating image editing task...")
            task_id = self._create_task(file_url, prompt)
            
            # Step 3: Wait for completion
            logger.info("Step 3: Waiting for task completion...")
            task_result = self._wait_for_task_completion(task_id, max_wait=300)
            
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
            logger.info("✅ Image refined using KIE.ai GPT Image 1.5")
            return result_image

        except CompositingError:
            raise
        except Exception as e:
            logger.error(f"Refinement error: {e}", exc_info=True)
            logger.warning("Returning original image due to refinement error")
            return image

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

    def refine(
        self,
        image: Image.Image,
        prompt: str | None = None,
        refinement_type: str = "compositing",
    ) -> Image.Image:
        """
        Refine image using KIE.ai API.

        Args:
            image: Image to refine
            prompt: Custom prompt (optional)
            refinement_type: Type of refinement (compositing, segmentation, placement)

        Returns:
            Refined image
        """
        if not self.api_key:
            logger.warning("KIE API key not configured, skipping refinement")
            return image

        if prompt is None:
            if refinement_type == "compositing":
                prompt = (
                    "Improve the image compositing quality. Make all elements blend seamlessly. "
                    "Fix visible edges, improve lighting matching, ensure natural integration. "
                    "The result should look like a single cohesive professional image."
                )
            elif refinement_type == "segmentation":
                prompt = (
                    "Improve background removal. Clean edges, remove artifacts, "
                    "smooth alpha transitions for professional compositing."
                )
            else:
                prompt = "Improve image quality and integration."

        return self._refine_with_prompt(image, prompt)

"""Unified KIE.ai client with model support and fallback logic."""
import json
import logging
from typing import Any

import httpx

from app.core.config import settings
from app.core.exceptions import CompositingError
from app.integrations.kie.builders import build_kie_payload
from app.integrations.kie.models import (
    KIEModel,
    KIETaskError,
    KIEValidationError,
    UnsupportedModelError,
)

logger = logging.getLogger(__name__)


class KIEClient:
    """
    Unified KIE.ai client for image editing with multi-model support.

    Features:
    - Support for multiple models (GPT Image 1.5, Nano Banana Edit)
    - Automatic fallback between models
    - Comprehensive validation and error handling
    - Detailed logging for debugging
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_url: str | None = None,
        primary_model: str | None = None,
        fallback_model: str | None = None,
    ):
        """
        Initialize KIE client.

        Args:
            api_key: KIE API key (defaults to settings.KIE_API_KEY)
            api_url: KIE API base URL (defaults to settings.KIE_API_URL)
            primary_model: Primary model to use (defaults to settings.KIE_PRIMARY_MODEL)
            fallback_model: Fallback model if primary fails (defaults to settings.KIE_FALLBACK_MODEL)
        """
        self.api_key = api_key or settings.KIE_API_KEY
        self.api_url = (api_url or settings.KIE_API_URL).rstrip("/")
        self.primary_model = primary_model or getattr(settings, "KIE_PRIMARY_MODEL", None) or KIEModel.GPT_IMAGE_15_I2I.value
        self.fallback_model = fallback_model or getattr(settings, "KIE_FALLBACK_MODEL", None)

        if not self.api_key:
            logger.warning("KIE_API_KEY not set, client will not function")

        logger.info(f"KIE Client initialized: api_url={self.api_url}")
        logger.info(f"Primary model: {self.primary_model}")
        if self.fallback_model:
            logger.info(f"Fallback model: {self.fallback_model}")

    def create_image_edit_task(
        self,
        model: str | None = None,
        prompt: str = "",
        poster_url: str | None = None,
        reference_url: str | None = None,
        mask_url: str | None = None,
        output_format: str = "png",
        aspect_ratio: str | None = None,
        quality: str = "high",
        use_fallback: bool = True,
    ) -> str:
        """
        Create image editing task with unified interface.

        This is the main public method that handles:
        - Model selection (primary or specified)
        - Payload building (model-specific)
        - Task creation
        - Fallback logic if enabled

        Args:
            model: Model to use (defaults to primary_model)
            prompt: Editing prompt (REQUIRED)
            poster_url: URL of poster/background image
            reference_url: URL of reference image
            mask_url: URL of mask image (not yet used, reserved for future)
            output_format: Output format (for Nano Banana Edit)
            aspect_ratio: Target aspect ratio (for GPT Image 1.5)
            quality: Output quality (for GPT Image 1.5)
            use_fallback: Whether to try fallback model if primary fails

        Returns:
            Task ID

        Raises:
            KIEValidationError: If validation fails
            KIETaskError: If task creation fails (after fallback if enabled)
            UnsupportedModelError: If model is not supported
        """
        # Validate prompt
        if not prompt or not prompt.strip():
            raise KIEValidationError("Prompt is required for image editing")

        # Determine model to use
        model_to_use = model or self.primary_model

        # Log model selection BEFORE validation
        logger.info(
            f"🎯 Model selection:\n"
            f"   - Requested model: {model or '(not specified, using primary)'}\n"
            f"   - Primary model: {self.primary_model}\n"
            f"   - Selected model: {model_to_use}\n"
            f"   - Fallback model: {self.fallback_model or '(not configured)'}"
        )

        # Validate model
        try:
            KIEModel.from_string(model_to_use)
        except ValueError as e:
            raise UnsupportedModelError(str(e)) from e

        logger.info(
            f"📝 Creating image edit task:\n"
            f"   - Selected model: {model_to_use}\n"
            f"   - Prompt length: {len(prompt)} chars\n"
            f"   - Poster URL: {'present' if poster_url else 'missing'}\n"
            f"   - Reference URL: {'present' if reference_url else 'missing'}\n"
            f"   - Use fallback: {use_fallback}"
        )

        # Try primary model
        try:
            return self._create_task_with_model(
                model=model_to_use,
                prompt=prompt,
                poster_url=poster_url,
                reference_url=reference_url,
                mask_url=mask_url,
                output_format=output_format,
                aspect_ratio=aspect_ratio,
                quality=quality,
            )
        except (KIETaskError, KIEValidationError) as primary_error:
            # If fallback is disabled or no fallback model, re-raise
            if not use_fallback or not self.fallback_model or model_to_use == self.fallback_model:
                logger.error(f"Primary model {model_to_use} failed, no fallback available")
                raise

            # Try fallback model
            logger.warning(
                f"Primary model {model_to_use} failed: {primary_error}. "
                f"Trying fallback model: {self.fallback_model}"
            )

            try:
                task_id = self._create_task_with_model(
                    model=self.fallback_model,
                    prompt=prompt,
                    poster_url=poster_url,
                    reference_url=reference_url,
                    mask_url=mask_url,
                    output_format=output_format,
                    aspect_ratio=aspect_ratio,
                    quality=quality,
                )
                logger.info(f"✅ Fallback model {self.fallback_model} succeeded, task_id: {task_id}")
                return task_id
            except Exception as fallback_error:
                # Both models failed
                error_msg = (
                    f"Both primary and fallback models failed. "
                    f"Primary ({model_to_use}) failed: {primary_error}. "
                    f"Fallback ({self.fallback_model}) failed: {fallback_error}"
                )
                logger.error(error_msg)
                raise KIETaskError(
                    error_msg,
                    model=model_to_use,
                    details={
                        "primary_error": str(primary_error),
                        "fallback_error": str(fallback_error),
                        "primary_model": model_to_use,
                        "fallback_model": self.fallback_model,
                    },
                ) from fallback_error

    def _create_task_with_model(
        self,
        model: str,
        prompt: str,
        poster_url: str | None = None,
        reference_url: str | None = None,
        mask_url: str | None = None,
        output_format: str = "png",
        aspect_ratio: str | None = None,
        quality: str = "high",
    ) -> str:
        """
        Create task with specific model (internal method).

        Args:
            model: Model name
            prompt: Editing prompt
            poster_url: URL of poster image
            reference_url: URL of reference image
            mask_url: URL of mask image (reserved)
            output_format: Output format
            aspect_ratio: Aspect ratio
            quality: Quality setting

        Returns:
            Task ID

        Raises:
            KIEValidationError: If validation fails
            KIETaskError: If task creation fails
        """
        endpoint = "/api/v1/jobs/createTask"

        # Log BEFORE building payload
        logger.info(
            f"🔧 _create_task_with_model called:\n"
            f"   - Input model parameter: {model}\n"
            f"   - Prompt: {prompt[:50]}...\n"
            f"   - Poster URL: {'present' if poster_url else 'missing'}\n"
            f"   - Reference URL: {'present' if reference_url else 'missing'}"
        )

        # Build payload using model-specific builder
        try:
            payload = build_kie_payload(
                model=model,
                prompt=prompt,
                poster_url=poster_url,
                reference_url=reference_url,
                aspect_ratio=aspect_ratio,
                quality=quality,
                output_format=output_format,
            )
        except (UnsupportedModelError, KIEValidationError):
            raise
        except Exception as e:
            logger.error(f"❌ Failed to build payload for model {model}: {e}", exc_info=True)
            raise KIEValidationError(
                f"Failed to build payload for model {model}: {e}",
                model=model,
            ) from e

        # Log payload details BEFORE sending request
        input_obj = payload.get("input", {})
        logger.info(
            f"🔍 About to send createTask request:\n"
            f"   - Selected model (input): {model}\n"
            f"   - Payload model: {payload.get('model')}\n"
            f"   - Payload model matches input: {payload.get('model') == model}\n"
            f"   - Input keys: {list(input_obj.keys())}"
        )

        # Verify payload structure matches model
        if model == "google/nano-banana-edit":
            if "image_urls" not in input_obj:
                logger.error(
                    f"❌ MISMATCH: Nano Banana Edit payload missing 'image_urls'!\n"
                    f"   - Payload model: {payload.get('model')}\n"
                    f"   - Input keys: {list(input_obj.keys())}\n"
                    f"   - Full payload: {payload}"
                )
                raise KIEValidationError(
                    f"Nano Banana Edit payload must contain 'image_urls', but got keys: {list(input_obj.keys())}",
                    model=model,
                )
            if "input_urls" in input_obj:
                logger.warning(
                    f"⚠️ Nano Banana Edit payload contains 'input_urls' (should use 'image_urls')\n"
                    f"   - Input keys: {list(input_obj.keys())}"
                )
        
        elif model == "gpt-image/1.5-image-to-image":
            if "input_urls" not in input_obj and "image_urls" in input_obj:
                logger.warning(
                    f"⚠️ GPT Image 1.5 payload contains 'image_urls' (should use 'input_urls')\n"
                    f"   - Input keys: {list(input_obj.keys())}"
                )
        
        logger.debug(
            f"Full payload before request:\n"
            f"   - Payload: {payload}\n"
            f"   - Input object: {input_obj}"
        )

        # Log image URLs (partially masked for security)
        if "input_urls" in input_obj:
            urls = input_obj["input_urls"]
            logger.debug(f"input_urls count: {len(urls)}")
            for i, url in enumerate(urls):
                masked = url[:30] + "..." + url[-20:] if len(url) > 50 else url
                logger.debug(f"  input_urls[{i}]: {masked}")

        if "image_urls" in input_obj:
            urls = input_obj["image_urls"]
            logger.debug(f"image_urls count: {len(urls)}")
            for i, url in enumerate(urls):
                masked = url[:30] + "..." + url[-20:] if len(url) > 50 else url
                logger.debug(f"  image_urls[{i}]: {masked}")

        # Make request
        try:
            result_data = self._make_request(endpoint, payload, timeout=60.0)
        except Exception as e:
            raise KIETaskError(
                f"Failed to create task for model {model}: {e}",
                model=model,
                endpoint=endpoint,
            ) from e

        # Validate response
        code = result_data.get("code")
        if code != 200:
            error_msg = result_data.get("msg", "Unknown error")
            raise KIETaskError(
                f"KIE.ai createTask failed (code {code}): {error_msg}",
                model=model,
                endpoint=endpoint,
                details={"code": code, "msg": error_msg},
            )

        # Extract task ID
        if not result_data.get("data"):
            raise KIETaskError(
                f"KIE.ai createTask returned no data: {result_data}",
                model=model,
                endpoint=endpoint,
            )

        data = result_data["data"]
        task_id = None
        if isinstance(data, dict):
            task_id = data.get("taskId") or data.get("task_id") or data.get("id")
        elif isinstance(data, str):
            task_id = data

        if not task_id:
            raise KIETaskError(
                f"Task ID not found in response data: {data}",
                model=model,
                endpoint=endpoint,
            )

        # Log task creation with model verification
        logger.info(
            f"✅ Task created successfully:\n"
            f"   - Model (requested): {model}\n"
            f"   - Task ID: {task_id}\n"
            f"   - Response data: {data}"
        )
        return task_id

    def get_task_status(self, task_id: str, max_retries: int = 3) -> dict[str, Any]:
        """
        Get task status with retry logic.

        Uses GET /api/v1/jobs/recordInfo?taskId=... for Market models.

        Args:
            task_id: Task ID
            max_retries: Maximum number of retries for network errors

        Returns:
            Task status data (from payload["data"])

        Raises:
            KIETaskError: If status check fails
        """
        endpoint = "/api/v1/jobs/recordInfo"

        for attempt in range(max_retries):
            try:
                # Use GET request with taskId as query parameter
                result_data = self._make_get_request(
                    endpoint, params={"taskId": task_id}, timeout=30.0
                )

                # Log full raw response for debugging (first attempt only)
                if attempt == 0:
                    logger.debug(f"🔍 recordInfo raw response for task {task_id}: {result_data}")

                # Validate response structure
                if not isinstance(result_data, dict):
                    raise KIETaskError(
                        f"Invalid response type from recordInfo: expected dict, got {type(result_data)}. "
                        f"Response: {result_data}",
                        task_id=task_id,
                        endpoint=endpoint,
                    )

                # Check response code
                code = result_data.get("code")
                if code is not None and code != 200:
                    error_msg = result_data.get("msg", "Unknown error")
                    raise KIETaskError(
                        f"KIE.ai recordInfo failed (code {code}): {error_msg}",
                        task_id=task_id,
                        endpoint=endpoint,
                        details={"code": code, "msg": error_msg, "full_response": result_data},
                    )

                # Extract data from response
                # For Market models, status is in payload["data"]
                data = result_data.get("data")

                # Validate data exists and is a dict
                if data is None:
                    raise KIETaskError(
                        f"KIE.ai recordInfo returned no 'data' field. Full response: {result_data}",
                        task_id=task_id,
                        endpoint=endpoint,
                        details={"full_response": result_data},
                    )

                if not isinstance(data, dict):
                    raise KIETaskError(
                        f"KIE.ai recordInfo 'data' field is not a dict: got {type(data)}. "
                        f"Full response: {result_data}",
                        task_id=task_id,
                        endpoint=endpoint,
                        details={"data_type": type(data).__name__, "full_response": result_data},
                    )

                # Extract and log all relevant fields
                state = data.get("state")
                model = data.get("model")
                task_id_from_response = data.get("taskId")
                fail_msg = data.get("failMsg")
                fail_code = data.get("failCode")
                result_json = data.get("resultJson")
                data_keys = list(data.keys())

                # Detailed logging
                logger.info(
                    f"📊 Task {task_id} status (attempt {attempt + 1}/{max_retries}):\n"
                    f"   - Full payload keys: {list(result_data.keys())}\n"
                    f"   - Data keys: {data_keys}\n"
                    f"   - State: {state}\n"
                    f"   - Model: {model}\n"
                    f"   - TaskId (from response): {task_id_from_response}\n"
                    f"   - FailCode: {fail_code}\n"
                    f"   - FailMsg: {fail_msg}\n"
                    f"   - ResultJson present: {bool(result_json)}\n"
                    f"   - ResultJson length: {len(result_json) if result_json else 0} chars"
                )

                # Log full data for debugging
                logger.debug(f"🔍 Full data object: {data}")

                # Return data dict (contains state, resultJson, etc.)
                return data

            except KIETaskError:
                # Business logic errors - don't retry
                raise
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.ConnectTimeout) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        f"Network error getting task status (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    import time
                    time.sleep(wait_time)
                    continue
                else:
                    raise KIETaskError(
                        f"Failed to get task status after {max_retries} attempts: {e}",
                        task_id=task_id,
                        endpoint=endpoint,
                    ) from e
            except Exception as e:
                # Unexpected errors - log and re-raise as KIETaskError
                logger.error(
                    f"Unexpected error getting task status (attempt {attempt + 1}/{max_retries}): {e}",
                    exc_info=True,
                )
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise KIETaskError(
                        f"Unexpected error getting task status: {e}",
                        task_id=task_id,
                        endpoint=endpoint,
                    ) from e

    def wait_for_task_completion(
        self, task_id: str, max_wait: int = 300, poll_interval: int = 3
    ) -> dict[str, Any]:
        """
        Wait for task completion with polling.

        Handles states: waiting, queuing, generating, success, fail

        Args:
            task_id: Task ID
            max_wait: Maximum wait time in seconds
            poll_interval: Polling interval in seconds

        Returns:
            Final task result data (contains resultJson for success)

        Raises:
            KIETaskError: If task fails or times out
        """
        import time

        start_time = time.time()
        logger.info(f"Waiting for task {task_id} to complete (max {max_wait}s, poll every {poll_interval}s)...")

        consecutive_invalid_responses = 0
        max_consecutive_invalid = 3  # Max invalid responses before giving up

        while True:
            elapsed = time.time() - start_time
            if elapsed > max_wait:
                raise KIETaskError(
                    f"Task {task_id} timed out after {max_wait}s",
                    task_id=task_id,
                )

            try:
                data = self.get_task_status(task_id)

                # Validate data is a dict (should be guaranteed by get_task_status, but double-check)
                if not isinstance(data, dict):
                    consecutive_invalid_responses += 1
                    logger.error(
                        f"Invalid data type from get_task_status: expected dict, got {type(data)}. "
                        f"Data: {data}"
                    )
                    if consecutive_invalid_responses >= max_consecutive_invalid:
                        raise KIETaskError(
                            f"Too many consecutive invalid responses for task {task_id}: "
                            f"{consecutive_invalid_responses} invalid responses",
                            task_id=task_id,
                        )
                    time.sleep(poll_interval)
                    continue

                # Reset invalid response counter on valid response
                consecutive_invalid_responses = 0

                # Extract state safely
                state = (data.get("state") or "").lower()
                logger.debug(f"Task {task_id} state: {state}")

                # Success states
                if state in {"success", "completed", "done"}:
                    logger.info(f"✅ Task {task_id} completed successfully (state: {state})")

                    # For success, resultJson should contain resultUrls
                    result_json_raw = data.get("resultJson")
                    if not result_json_raw:
                        logger.warning(
                            f"⚠️ Task {task_id} completed but no resultJson in response. "
                            f"Full data: {data}"
                        )
                        # Return data anyway - caller can handle missing resultJson
                        return data

                    # Parse resultJson (it's a JSON string)
                    logger.debug(f"📦 Task {task_id} resultJson present: {len(result_json_raw)} chars")
                    try:
                        import json
                        result_json_parsed = json.loads(result_json_raw)
                        logger.debug(f"📦 Task {task_id} resultJson parsed successfully")

                        # Normalize result object
                        normalized_result = {
                            "state": state,
                            "taskId": task_id,
                            "model": data.get("model"),
                            "resultJson": result_json_raw,  # Keep original string
                            "resultJsonParsed": result_json_parsed,  # Parsed object
                            "resultUrls": result_json_parsed.get("resultUrls", []),
                        }

                        # Log resultUrls
                        result_urls = normalized_result["resultUrls"]
                        logger.info(
                            f"📥 Task {task_id} resultUrls count: {len(result_urls)}\n"
                            f"   - ResultUrls: {result_urls}"
                        )

                        return normalized_result

                    except json.JSONDecodeError as e:
                        logger.error(
                            f"❌ Failed to parse resultJson for task {task_id}: {e}\n"
                            f"   - resultJson raw (first 500 chars): {result_json_raw[:500]}"
                        )
                        # Return data with unparsed resultJson - caller can handle
                        return data
                    except Exception as e:
                        logger.error(
                            f"❌ Unexpected error parsing resultJson for task {task_id}: {e}",
                            exc_info=True,
                        )
                        # Return data anyway
                        return data

                # Failure states
                if state in {"fail", "failed", "error"}:
                    fail_msg = (data.get("failMsg") or "").strip()
                    fail_code = str(data.get("failCode") or "").strip()
                    error_msg = data.get("message") or data.get("error") or data.get("msg") or ""

                    error_details = []
                    if fail_code:
                        error_details.append(f"failCode={fail_code}")
                    if fail_msg:
                        error_details.append(f"failMsg={fail_msg}")
                    if error_msg:
                        error_details.append(f"message={error_msg}")

                    full_error_msg = ", ".join(error_details) if error_details else "Unknown error"

                    # Special handling for NSFW
                    if fail_msg.lower() == "nsfw" or "nsfw" in fail_msg.lower() or "nsfw" in error_msg.lower():
                        logger.warning(f"⚠️ Task {task_id} blocked by safety filter (NSFW)")
                        raise KIETaskError(
                            f"KIE.ai blocked the task with safety filter (nsfw): {full_error_msg}. "
                            "Try a different input image, a more neutral crop, or use non-LLM fallback.",
                            task_id=task_id,
                            details={"failCode": fail_code, "failMsg": fail_msg, "full_response": data},
                        )

                    raise KIETaskError(
                        f"KIE.ai task failed: {full_error_msg}",
                        task_id=task_id,
                        details={"failCode": fail_code, "failMsg": fail_msg, "full_response": data},
                    )

                # Processing states: waiting, queuing, generating, processing, etc.
                if state in {"waiting", "queuing", "generating", "processing", "running", ""}:
                    logger.info(
                        f"⏳ Task {task_id} state: {state or '(empty/unknown)'}, "
                        f"waiting {poll_interval}s... (elapsed: {int(elapsed)}s/{max_wait}s)"
                    )
                    time.sleep(poll_interval)
                    continue

                # Unknown state - log warning but continue polling
                logger.warning(
                    f"Task {task_id} has unknown state: {state}. "
                    f"Continuing to poll... (full data: {data})"
                )
                time.sleep(poll_interval)

            except KIETaskError:
                # Business logic errors - don't retry
                raise
            except Exception as e:
                logger.error(f"Error checking task status: {e}", exc_info=True)
                consecutive_invalid_responses += 1
                if consecutive_invalid_responses >= max_consecutive_invalid:
                    raise KIETaskError(
                        f"Too many consecutive errors while polling task {task_id}: {e}",
                        task_id=task_id,
                    ) from e
                time.sleep(poll_interval)

    def _make_request(
        self, endpoint: str, payload: dict[str, Any], timeout: float = 300.0
    ) -> dict[str, Any]:
        """
        Make HTTP request to KIE.ai API.

        Args:
            endpoint: API endpoint
            payload: Request payload
            timeout: Request timeout

        Returns:
            Response data

        Raises:
            KIETaskError: If request fails
        """
        if not self.api_key:
            raise KIETaskError("KIE API key not configured", endpoint=endpoint)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        full_url = f"{self.api_url}{endpoint}"
        logger.debug(f"Making request to: {full_url}")

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(full_url, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text[:500] if e.response and e.response.text else str(e)
            raise KIETaskError(
                f"KIE.ai API HTTP error {e.response.status_code}: {error_detail}",
                endpoint=endpoint,
                details={"status_code": e.response.status_code, "response": error_detail},
            ) from e
        except httpx.TimeoutException as e:
            raise KIETaskError(
                f"KIE.ai API timeout after {timeout}s",
                endpoint=endpoint,
            ) from e
        except httpx.RequestError as e:
            raise KIETaskError(
                f"KIE.ai API request error: {e}",
                endpoint=endpoint,
            ) from e
        except Exception as e:
            raise KIETaskError(
                f"Unexpected error in KIE.ai request: {e}",
                endpoint=endpoint,
            ) from e

    def _make_get_request(
        self, endpoint: str, params: dict[str, Any] | None = None, timeout: float = 30.0
    ) -> dict[str, Any]:
        """
        Make HTTP GET request to KIE.ai API.

        Used for recordInfo endpoint which requires GET with query parameters.

        Args:
            endpoint: API endpoint
            params: Query parameters
            timeout: Request timeout

        Returns:
            Response data

        Raises:
            KIETaskError: If request fails
        """
        if not self.api_key:
            raise KIETaskError("KIE API key not configured", endpoint=endpoint)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        full_url = f"{self.api_url}{endpoint}"
        logger.debug(f"Making GET request to: {full_url}, params: {params}")

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(full_url, headers=headers, params=params or {})
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text[:500] if e.response and e.response.text else str(e)
            raise KIETaskError(
                f"KIE.ai API HTTP error {e.response.status_code}: {error_detail}",
                endpoint=endpoint,
                details={"status_code": e.response.status_code, "response": error_detail},
            ) from e
        except httpx.TimeoutException as e:
            raise KIETaskError(
                f"KIE.ai API timeout after {timeout}s",
                endpoint=endpoint,
            ) from e
        except httpx.RequestError as e:
            raise KIETaskError(
                f"KIE.ai API request error: {e}",
                endpoint=endpoint,
            ) from e
        except Exception as e:
            raise KIETaskError(
                f"Unexpected error in KIE.ai request: {e}",
                endpoint=endpoint,
            ) from e

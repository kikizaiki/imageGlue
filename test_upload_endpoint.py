"""Test script to find correct KIE.ai file upload endpoint."""
import base64
import json
import logging
import os
import sys
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_upload_endpoint(endpoint: str, image_path: Path):
    """Test a specific upload endpoint."""
    api_key = settings.KIE_API_KEY
    api_url = settings.KIE_API_URL.rstrip("/")
    
    if not api_key:
        logger.error("KIE_API_KEY not set in .env")
        return False
    
    # Load image
    image = Image.open(image_path)
    img_bytes = BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    
    files = {
        "file": ("test_image.png", img_bytes, "image/png")
    }
    
    full_url = f"{api_url}{endpoint}"
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {full_url}")
    logger.info(f"{'='*60}")
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(full_url, headers=headers, files=files)
            
            logger.info(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    logger.info(f"✅ SUCCESS! Response: {json.dumps(result, indent=2)}")
                    return True
                except Exception as e:
                    logger.info(f"✅ SUCCESS! (Non-JSON response): {response.text[:200]}")
                    return True
            else:
                logger.warning(f"❌ Failed: {response.status_code}")
                try:
                    error_data = response.json()
                    logger.warning(f"Error: {json.dumps(error_data, indent=2)}")
                except:
                    logger.warning(f"Error text: {response.text[:500]}")
                return False
                
    except httpx.RequestError as e:
        logger.error(f"❌ Request error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False


def main():
    """Test multiple upload endpoints."""
    if len(sys.argv) < 2:
        print("Usage: python test_upload_endpoint.py <path_to_image.jpg>")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)
    
    # List of endpoints to test (based on KIE.ai File Upload API documentation)
    # Актуальная документация: https://docs.kie.ai/file-upload-api/quickstart
    # Задокументированные endpoints:
    # - /api/file-stream-upload (для локальных файлов - multipart/form-data)
    # - /api/file-base64-upload (для base64)
    # - /api/file-url-upload (для публичных URL)
    endpoints_to_test = [
        "/api/file-stream-upload",  # File Stream Upload (для локальных файлов)
        "/api/file-base64-upload",  # Base64 Upload (альтернатива)
        # /api/file-url-upload не тестируем, т.к. нужен публичный URL
    ]
    
    # Check if custom endpoint provided via env
    custom_endpoint = os.getenv("KIE_UPLOAD_ENDPOINT")
    if custom_endpoint:
        endpoints_to_test.insert(0, custom_endpoint)
        logger.info(f"Using custom endpoint from KIE_UPLOAD_ENDPOINT: {custom_endpoint}")
    
    logger.info(f"Testing {len(endpoints_to_test)} endpoints...")
    logger.info(f"Image: {image_path}")
    logger.info(f"API URL: {settings.KIE_API_URL}")
    logger.info(f"API Key: {'*' * 20 if settings.KIE_API_KEY else 'NOT SET'}\n")
    
    success_count = 0
    for endpoint in endpoints_to_test:
        if test_upload_endpoint(endpoint, image_path):
            success_count += 1
            logger.info(f"\n🎉 Found working endpoint: {endpoint}")
            logger.info(f"Use this in your code or set KIE_UPLOAD_ENDPOINT={endpoint}")
            break
    
    if success_count == 0:
        logger.error("\n❌ No working endpoint found!")
        logger.error("Please check KIE.ai documentation for the correct file upload endpoint.")
        logger.error("You can also set KIE_UPLOAD_ENDPOINT environment variable to test a specific endpoint.")


if __name__ == "__main__":
    main()

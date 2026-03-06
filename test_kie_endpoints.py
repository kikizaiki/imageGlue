#!/usr/bin/env python3
"""Тест различных endpoints KIE.ai для редактирования изображений."""
import base64
import json
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image

from app.core.config import settings

def test_endpoints(image_path: str):
    """Тест различных endpoints."""
    if not settings.KIE_API_KEY:
        print("❌ KIE_API_KEY не установлен")
        return
    
    # Load image
    if not Path(image_path).exists():
        print(f"❌ Изображение не найдено: {image_path}")
        return
    
    image = Image.open(image_path)
    img_bytes = BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
    
    headers = {
        "Authorization": f"Bearer {settings.KIE_API_KEY}",
        "Content-Type": "application/json",
    }
    
    base_url = settings.KIE_API_URL.rstrip("/")
    prompt = "Improve image quality"
    
    # Разные варианты payload
    payloads = [
        {"image": f"data:image/png;base64,{img_base64}", "prompt": prompt},
        {"input_image": f"data:image/png;base64,{img_base64}", "instruction": prompt},
        {"image_url": f"data:image/png;base64,{img_base64}", "prompt": prompt},
        {"messages": [{"role": "user", "content": prompt, "image": f"data:image/png;base64,{img_base64}"}]},
    ]
    
    # Endpoints для проверки
    endpoints = [
        "/api/v1/chat/images",
        "/api/v1/chat/image-edit",
        "/api/v1/image/edit",
        "/api/v1/images/edit",
        "/api/v1/image/inpaint",
        "/api/v1/image/refine",
        "/api/v1/image/upscale",
    ]
    
    print("🧪 Тестирование endpoints KIE.ai\n")
    print(f"Base URL: {base_url}")
    print(f"API Key: {settings.KIE_API_KEY[:10]}...\n")
    
    for endpoint in endpoints:
        print(f"\n📡 Endpoint: {endpoint}")
        for i, payload in enumerate(payloads, 1):
            print(f"   Вариант {i}: {list(payload.keys())}")
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.post(
                        f"{base_url}{endpoint}",
                        headers=headers,
                        json=payload,
                    )
                    print(f"   Статус: {response.status_code}")
                    
                    if response.status_code == 200:
                        result = response.json()
                        print(f"   ✅ УСПЕХ! Ответ: {list(result.keys())}")
                        print(f"   Данные: {str(result)[:300]}...")
                        print(f"\n🎉 Рабочий endpoint: {endpoint}")
                        print(f"   Рабочий payload: вариант {i}")
                        return endpoint, payload
                    elif response.status_code == 404:
                        print(f"   ❌ 404 - Endpoint не найден")
                    elif response.status_code == 400:
                        error = response.text[:200] if response.text else "No error"
                        print(f"   ⚠️  400 - Неправильный запрос: {error}")
                    else:
                        error = response.text[:200] if response.text else "No error"
                        print(f"   ⚠️  {response.status_code}: {error}")
            except httpx.TimeoutException:
                print(f"   ⏱️  Timeout")
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
    
    print("\n❌ Ни один endpoint не сработал")
    print("\n💡 Проверьте документацию KIE.ai для правильного endpoint")
    print("   Или укажите правильный endpoint вручную в коде")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Использование: python test_kie_endpoints.py <путь_к_изображению>")
        sys.exit(1)
    test_endpoints(sys.argv[1])

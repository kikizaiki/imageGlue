#!/usr/bin/env python3
"""Тест KIE.ai API напрямую."""
import base64
import json
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image

from app.core.config import settings

def test_kie_api(image_path: str):
    """Тест KIE.ai API с реальным изображением."""
    print("🧪 Тест KIE.ai API")
    print("=" * 50)
    
    if not settings.KIE_API_KEY:
        print("❌ KIE_API_KEY не установлен в .env")
        return
    
    print(f"✅ API Key: {settings.KIE_API_KEY[:10]}...")
    print(f"✅ API URL: {settings.KIE_API_URL}")
    
    # Load image
    if not Path(image_path).exists():
        print(f"❌ Изображение не найдено: {image_path}")
        return
    
    image = Image.open(image_path)
    print(f"✅ Изображение загружено: {image.size}")
    
    # Convert to base64
    img_bytes = BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
    print(f"✅ Изображение конвертировано в base64: {len(img_base64)} символов")
    
    # Prepare request
    headers = {
        "Authorization": f"Bearer {settings.KIE_API_KEY}",
        "Content-Type": "application/json",
    }
    
    prompt = "Improve the image quality, make it look professional"
    
    # Try different endpoints and formats
    endpoints = [
        "/v1/images/edit",
        "/api/v1/images/edit",
        "/images/edit",
        "/v1/image/edit",
        "/v1/image/refine",
    ]
    
    payload_variants = [
        {
            "image": f"data:image/png;base64,{img_base64}",
            "prompt": prompt,
        },
        {
            "image": f"data:image/png;base64,{img_base64}",
            "prompt": prompt,
            "model": "image-edit",
        },
        {
            "image": img_base64,  # Without data URI
            "prompt": prompt,
        },
        {
            "input_image": f"data:image/png;base64,{img_base64}",
            "instruction": prompt,
        },
    ]
    
    for endpoint in endpoints:
        print(f"\n🔍 Пробую endpoint: {endpoint}")
        full_url = f"{settings.KIE_API_URL.rstrip('/')}{endpoint}"
        
        for i, payload in enumerate(payload_variants):
            print(f"   Вариант {i+1}: {list(payload.keys())}")
            try:
                with httpx.Client(timeout=60.0) as client:
                    response = client.post(
                        full_url,
                        headers=headers,
                        json=payload,
                    )
                    
                    print(f"   Статус: {response.status_code}")
                    
                    if response.status_code == 200:
                        result = response.json()
                        print(f"   ✅ Успех! Ответ содержит ключи: {list(result.keys())}")
                        
                        # Try to extract image
                        if "image" in result:
                            print(f"   📸 Найден ключ 'image'")
                            return result
                        elif "data" in result:
                            print(f"   📸 Найден ключ 'data'")
                            return result
                        elif "url" in result:
                            print(f"   📸 Найден ключ 'url': {result['url']}")
                            return result
                        else:
                            print(f"   ⚠️  Неожиданный формат ответа")
                            print(f"   Ответ: {json.dumps(result, indent=2)[:500]}")
                    else:
                        error_text = response.text[:200] if response.text else "No error"
                        print(f"   ❌ Ошибка: {error_text}")
                        
            except Exception as e:
                print(f"   ❌ Исключение: {e}")
                continue
    
    print("\n❌ Все варианты не сработали")
    print("\n💡 Рекомендации:")
    print("   1. Проверьте документацию KIE.ai API")
    print("   2. Проверьте формат запроса в логах")
    print("   3. Адаптируйте код в app/services/refinement/kie_refiner.py")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Использование: python test_kie_api.py <путь_к_изображению>")
        sys.exit(1)
    
    test_kie_api(sys.argv[1])

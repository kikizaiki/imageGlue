#!/usr/bin/env python3
"""Интерактивный скрипт для тестирования imageGlue API."""
import json
import sys
import time
from pathlib import Path

import httpx


API_URL = "http://127.0.0.1:8000"
BASE_URL = f"{API_URL}/v1"


def print_section(title: str):
    """Печатает заголовок секции."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")


def test_health():
    """Тест health check."""
    print_section("1. Health Check")
    response = httpx.get(f"{API_URL}/health")
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    return True


def test_list_templates():
    """Тест списка шаблонов."""
    print_section("2. Список шаблонов")
    response = httpx.get(f"{BASE_URL}/templates")
    response.raise_for_status()
    templates = response.json()
    print(json.dumps(templates, indent=2, ensure_ascii=False))
    return templates


def test_get_template(template_id: str):
    """Тест получения информации о шаблоне."""
    print_section(f"3. Информация о шаблоне: {template_id}")
    response = httpx.get(f"{BASE_URL}/templates/{template_id}")
    response.raise_for_status()
    template = response.json()
    print(json.dumps(template, indent=2, ensure_ascii=False))
    return template


def test_create_job(image_path: str, template_id: str = "astronaut_dog", entity_type: str = "dog"):
    """Тест создания задачи."""
    print_section("4. Создание задачи")
    
    if not Path(image_path).exists():
        print(f"❌ Файл не найден: {image_path}")
        return None
    
    print(f"📤 Загрузка: {image_path}")
    print(f"   Шаблон: {template_id}")
    print(f"   Тип: {entity_type}")
    
    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        data = {
            "template_id": template_id,
            "entity_type": entity_type,
        }
        
        response = httpx.post(f"{BASE_URL}/jobs", files=files, data=data, timeout=300.0)
        response.raise_for_status()
        job = response.json()
        print(json.dumps(job, indent=2, ensure_ascii=False))
        
        return job


def test_get_job_status(job_id: str):
    """Тест получения статуса задачи."""
    print_section(f"5. Статус задачи: {job_id}")
    
    max_attempts = 30
    attempt = 0
    
    while attempt < max_attempts:
        response = httpx.get(f"{BASE_URL}/jobs/{job_id}")
        response.raise_for_status()
        job = response.json()
        
        status = job["status"]
        current_stage = job.get("current_stage", "N/A")
        
        print(f"   Попытка {attempt + 1}/{max_attempts}")
        print(f"   Статус: {status}")
        print(f"   Этап: {current_stage}")
        
        if status == "completed":
            print("\n✅ Задача завершена!")
            print(json.dumps(job, indent=2, ensure_ascii=False))
            
            final_url = job.get("final_image_url")
            if final_url:
                print(f"\n📸 Финальное изображение:")
                print(f"   {API_URL}{final_url}")
            
            return job
        elif status == "failed":
            print("\n❌ Задача завершилась с ошибкой")
            print(json.dumps(job, indent=2, ensure_ascii=False))
            return job
        
        attempt += 1
        if attempt < max_attempts:
            time.sleep(2)
    
    print(f"\n⏱️  Превышено время ожидания ({max_attempts * 2} секунд)")
    return job


def main():
    """Главная функция."""
    print("\n🧪 Тестирование imageGlue API")
    print("=" * 50)
    
    try:
        # 1. Health check
        test_health()
        
        # 2. List templates
        templates = test_list_templates()
        
        # 3. Get template info
        if templates:
            template_id = templates[0]["template_id"]
            test_get_template(template_id)
        
        # 4. Create job (if image provided)
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
            job = test_create_job(image_path)
            
            if job:
                job_id = job["job_id"]
                
                # 5. Get job status
                test_get_job_status(job_id)
        else:
            print_section("4. Создание задачи (пропущено)")
            print("   Использование: python scripts/test_api.py <путь_к_изображению.jpg>")
            print("\n   Пример:")
            print("   python scripts/test_api.py ~/Pictures/dog.jpg")
        
        print("\n" + "=" * 50)
        print("✅ Тестирование завершено")
        print("=" * 50)
        
    except httpx.HTTPStatusError as e:
        print(f"\n❌ HTTP ошибка: {e.response.status_code}")
        print(e.response.text)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

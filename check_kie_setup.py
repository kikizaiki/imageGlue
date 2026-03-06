#!/usr/bin/env python3
"""Проверка настройки KIE.ai API."""
import os
from pathlib import Path
from app.core.config import settings

print("🔍 Проверка настройки KIE.ai API")
print("=" * 50)

# Check .env file
env_path = Path(".env")
if env_path.exists():
    print(f"✅ Файл .env найден: {env_path.absolute()}")
    with open(env_path) as f:
        has_key = any("KIE_API_KEY" in line and "=" in line for line in f)
        if has_key:
            print("✅ KIE_API_KEY найден в .env")
        else:
            print("❌ KIE_API_KEY НЕ найден в .env")
else:
    print(f"❌ Файл .env НЕ найден в {Path.cwd()}")

# Check settings
print(f"\n📋 Загруженные настройки:")
print(f"   KIE_API_KEY установлен: {bool(settings.KIE_API_KEY)}")
if settings.KIE_API_KEY:
    key = settings.KIE_API_KEY
    print(f"   Длина ключ_URL: {settings.KIE_API_URL}")
print(f"   REFINEMENT_ENABLED: {settings.REFINEMENT_ENABLED}")

# Check environment variables
env_key = os.getenv("KIE_API_KEY")
if env_key:
    print(f"\n🌍 KIE_API_KEY найден в переменных окружения")
    print(f"   Длина: {len(env_key)}")
else:
    print(f"\n🌍 KIE_API_KEY НЕ найден в переменных окружения")

print("\n" + "=" * 50)
if settings.KIE_API_KEY:
    print("✅ Всё настроено правильно!")
else:
    print("❌ Ключ не загружен. Проверьте .env файл.")

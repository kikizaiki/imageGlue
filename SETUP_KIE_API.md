# Настройка KIE.ai API

## Проблема: API ключ не загружается

Если результат не улучшается после добавления API ключа, проверьте следующее:

## Шаг 1: Создайте файл .env

```bash
# В корне проекта
cd /Users/sp1rit/Documents/Cursor/imageGlue
cp .env.example .env
```

## Шаг 2: Добавьте API ключ

Откройте `.env` и добавьте ваш ключ:

```bash
# KIE.ai API
KIE_API_KEY=ваш_реальный_ключ_здесь
KIE_API_URL=https://api.kie.ai/v1

# Убедитесь что refinement включен
REFINEMENT_ENABLED=true
```

**Важно:**
- Не используйте кавычки вокруг ключа
- Не добавляйте пробелы вокруг `=`
- Ключ должен быть на одной строке

## Шаг 3: Проверьте загрузку

```bash
source .venv/bin/activate
python -c "from app.core.config import settings; print(f'KIE_API_KEY loaded: {bool(settings.KIE_API_KEY)}')"
```

Должно вывести: `KIE_API_KEY loaded: True`

## Шаг 4: Запустите с debug

```bash
python main.py --input dog.jpg --template dog_cosmonaut_v1 --output result.png --debug
```

## Что искать в логах

### ✅ Если API работает:

```
INFO - Starting KIE.ai refinement with prompt: ...
INFO - Making request to: https://api.kie.ai/v1/...
INFO - ✅ Image refined using KIE.ai via /v1/images/edit
```

### ❌ Если API не работает:

```
WARNING - KIE API key not configured, skipping refinement
```

или

```
WARNING - All KIE.ai endpoints failed. Last error: ...
```

## Проверка формата API

Если API не работает, возможно нужно адаптировать формат запроса под ваш API.

Проверьте логи - там будет видно:
- Какой endpoint пробуется
- Какой статус код возвращается
- Какой формат ответа ожидается

## Альтернатива: Установка через переменные окружения

Если `.env` не работает, можно установить напрямую:

```bash
export KIE_API_KEY=ваш_ключ
export KIE_API_URL=https://api.kie.ai/v1
python main.py --input dog.jpg --template dog_cosmonaut_v1 --output result.png --debug
```

## Отладка

Если всё равно не работает:

1. **Проверьте формат API:**
   - Откройте `app/services/refinement/kie_refiner.py`
   - Посмотрите какие endpoints пробуются
   - Адаптируйте под ваш API формат

2. **Проверьте логи:**
   - С флагом `--debug` все запросы логируются
   - Смотрите файлы в `runs/debug/<job_id>/`

3. **Проверьте ответ API:**
   - В логах будет видно формат ответа
   - Адаптируйте парсинг ответа в `_refine_with_prompt()`

## Пример правильного .env

```bash
# Application
APP_ENV=development
DEBUG=true

# KIE.ai API
KIE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
KIE_API_URL=https://api.kie.ai/v1

# Refinement
REFINEMENT_ENABLED=true
REFINEMENT_TIMEOUT=120
```

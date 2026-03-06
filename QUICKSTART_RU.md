# Быстрый старт imageGlue

## Шаг 1: Активация виртуального окружения

```bash
source .venv/bin/activate
```

## Шаг 2: Запуск API сервера

```bash
python main.py
```

Или используйте скрипт:
```bash
./scripts/run_api.sh
```

## Шаг 3: Откройте в браузере

- **API документация**: http://127.0.0.1:8000/docs
- **Альтернативная документация**: http://127.0.0.1:8000/redoc
- **Health check**: http://127.0.0.1:8000/health

## Тестирование API

### Создать задачу (job)

```bash
curl -X POST "http://127.0.0.1:8000/v1/jobs" \
  -F "template_id=astronaut_dog" \
  -F "entity_type=dog" \
  -F "file=@путь/к/изображению.jpg"
```

### Получить статус задачи

```bash
curl "http://127.0.0.1:8000/v1/jobs/{job_id}"
```

### Список шаблонов

```bash
curl "http://127.0.0.1:8000/v1/templates"
```

## Что уже настроено

✅ Виртуальное окружение (.venv)  
✅ Зависимости установлены  
✅ Файл конфигурации (.env) создан  
✅ Шаблон astronaut_dog с изображениями  
✅ Директории для хранения созданы  

## Режимы работы

По умолчанию используется **sync режим** (синхронная обработка).

Для фоновой обработки:
1. Установите Redis: `brew install redis`
2. Запустите Redis: `brew services start redis`
3. Измените в `.env`: `JOB_MODE=queue`
4. Запустите worker: `./scripts/run_worker.sh`

## Детекторы

По умолчанию используется **mock детектор** (для тестирования).

Для использования YOLO:
1. Установите зависимости: `pip install torch ultralytics`
2. Измените в `.env`: `DETECTOR_BACKEND=yolo`

## Устранение проблем

### Порт занят
Измените `APP_PORT` в файле `.env`

### Ошибки импорта
Убедитесь, что виртуальное окружение активировано:
```bash
source .venv/bin/activate
```

### Шаблон не найден
Проверьте наличие файлов в `templates/astronaut_dog/`:
- background.png
- foreground.png
- config.json

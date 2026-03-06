# imageGlue

Сервис автоматической замены собаки в постере.

## Описание

imageGlue принимает пользовательское фото собаки и автоматически подставляет её в заранее подготовленный постер, сохраняя узнаваемость питомца и добиваясь органичного визуального результата.

## Установка

```bash
# Создать виртуальное окружение
python3.11 -m venv venv
source venv/bin/activate

# Установить зависимости
pip install -r requirements.txt

# Настроить .env
cp .env.example .env
# Отредактировать .env и добавить KIE_API_KEY
```

## Использование

### CLI

```bash
python main.py --input path/to/dog.jpg --template dog_cosmonaut_v1 --output result.png --debug
```

### API

```bash
# Запустить сервер
uvicorn app.api.main:app --reload

# Отправить запрос
curl -X POST "http://localhost:8000/render?template_id=dog_cosmonaut_v1&debug=true" \
  -F "image=@path/to/dog.jpg" \
  -o result.png
```

## Структура проекта

```
imageGlue/
  app/
    api/              # FastAPI endpoints
    core/              # Конфигурация, логирование, хранилище
    models/            # Модели данных
    pipelines/         # Основной pipeline
    services/          # Сервисы обработки
      validation/      # Валидация входных данных
      detection/       # Детекция собаки
      segmentation/    # Удаление фона
      crop_planning/   # Планирование кропа
      placement/       # Планирование размещения
      compositing/     # Композитинг
      refinement/      # AI refinement (kie.ai)
      quality/         # Quality gate
    utils/             # Утилиты
  templates/           # Шаблоны постеров
    dog_cosmonaut_v1/
      base_clean.png
      occlusion_mask.png
      glass_fx.png
      template_config.json
  runs/
    debug/             # Debug артефакты
    output/            # Финальные результаты
```

## Pipeline

1. **Input Validation** - Проверка входного изображения
2. **Subject Analysis** - Детекция собаки и головы
3. **Segmentation** - Удаление фона
4. **Crop Planning** - Планирование кропа (head-and-shoulders)
5. **Placement Planning** - Планирование размещения в шаблоне
6. **Compositing** - Композитинг слоёв
7. **Postprocess** - Постобработка (feather, color match)
8. **AI Refinement** - Опциональное улучшение через kie.ai
9. **Quality Gate** - Проверка качества результата

## Подготовка шаблона

Для каждого шаблона нужно подготовить:

1. `base_clean.png` - Постер без исходной собаки
2. `occlusion_mask.png` - Элементы, перекрывающие собаку
3. `glass_fx.png` - Эффекты (блики, стекло)
4. `template_config.json` - Конфигурация размещения

## Конфигурация

Основные параметры в `.env`:

- `KIE_API_KEY` - API ключ для kie.ai
- `REFINEMENT_ENABLED` - Включить AI refinement
- `QUALITY_CHECK_ENABLED` - Включить проверку качества

## Debug режим

С флагом `--debug` сохраняются все промежуточные этапы:

- `00_original.png` - Исходное изображение
- `01_detection_overlay.png` - Результат детекции
- `02_crop.png` - Кроп после планирования
- `03_subject_rgba.png` - Субъект после удаления фона
- `04_composited.png` - Результат композитинга
- `05_refined.png` - Результат после refinement (если применён)
- `metadata.json` - Полная метаданная обработки

## Требования

- Python 3.11+
- macOS (для локальной разработки)
- KIE.ai API ключ (для refinement)

## Лицензия

[Указать лицензию]

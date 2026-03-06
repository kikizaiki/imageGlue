# Интеграция с KIE.ai для улучшения качества

## Обзор

Система использует KIE.ai API для улучшения качества композитинга на нескольких этапах pipeline.

## Этапы использования AI

### 1. Улучшение сегментации (Stage C)

После удаления фона можно использовать AI для улучшения краёв:

```python
# Включено через настройку в template_config.json
"refinement": {
    "improve_segmentation": true
}
```

**Что делает:**
- Улучшает края удаления фона
- Убирает артефакты
- Сглаживает альфа-канал

### 2. Улучшение размещения (Stage E) - опционально

AI может улучшить размещение субъекта:

```python
"refinement": {
    "improve_placement": true
}
```

### 3. Финальное улучшение композитинга (Stage H)

Основной этап AI refinement:

- **Первый проход**: Улучшение общего композитинга
- **Второй проход** (опционально): Тонкая настройка

## Настройка

### 1. Добавить API ключ

В `.env`:
```
KIE_API_KEY=your_api_key_here
KIE_API_URL=https://api.kie.ai/v1
```

### 2. Настроить шаблон

В `template_config.json`:
```json
{
  "refinement": {
    "enabled": true,
    "improve_segmentation": true,
    "improve_placement": false,
    "second_pass": true,
    "prompt_template": "Ваш кастомный промпт (опционально)"
  }
}
```

## Использование LLM для промптов

Система автоматически генерирует оптимизированные промпты через LLM:

1. Анализирует шаблон и обнаруженные проблемы
2. Генерирует специфичный промпт через KIE.ai LLM API
3. Использует промпт для улучшения изображения

Если LLM недоступен, используется шаблонный промпт.

## API Endpoints

Система пробует следующие endpoints KIE.ai:

- `/v1/images/edit`
- `/api/v1/images/edit`
- `/images/edit`
- `/v1/image/edit`

И различные форматы запросов для совместимости.

## Формат запроса

```json
{
  "image": "data:image/png;base64,...",
  "prompt": "Сгенерированный или кастомный промпт",
  "model": "image-edit"  // или другой доступный
}
```

## Формат ответа

Система поддерживает несколько форматов:

1. `{"image": "data:image/png;base64,..."}`
2. `{"data": {"image": "..."}}`
3. `{"url": "https://..."}` - скачивание по URL

## Примеры промптов

### Автоматически сгенерированные:

```
"Improve the image compositing quality for a astronaut poster. 
Make all elements blend seamlessly together. 
Fix visible edges, improve lighting matching, ensure natural shadows and highlights. 
The composition should look like a single cohesive professional image, 
not like elements were pasted together."
```

### С обнаруженными проблемами:

```
"... Specifically address: subject may be too small, visible edges."
```

## Отладка

С флагом `--debug` сохраняются:

- `03_subject_rgba_ai_improved.png` - улучшенная сегментация
- `03_ai_placement.png` - улучшенное размещение (если включено)
- `05_refined_pass1.png` - первый проход refinement
- `05_refined_pass2.png` - второй проход (если включен)
- `05_refined.png` - финальный результат

## Troubleshooting

### API не отвечает

1. Проверьте `KIE_API_KEY` в `.env`
2. Проверьте `KIE_API_URL` (должен быть без trailing slash)
3. Проверьте логи на ошибки HTTP

### Неправильный формат ответа

Система автоматически пробует разные форматы. Если не работает, проверьте документацию KIE.ai API и обновите код в `kie_refiner.py`.

### Таймауты

Увеличьте `REFINEMENT_TIMEOUT` в `.env` если запросы долгие.

## Рекомендации

1. **Всегда используйте AI refinement** для финального результата
2. **Включите второй проход** для лучшего качества
3. **Настройте промпты** под ваш шаблон в `template_config.json`
4. **Проверяйте debug артефакты** для понимания что улучшает AI

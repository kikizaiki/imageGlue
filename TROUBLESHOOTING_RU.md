# Решение проблем imageGlue

## Проблема: /docs не открывается в браузере

### Решение 1: Проверьте, что сервер запущен

```bash
# Проверьте процессы
ps aux | grep "python main.py"

# Или проверьте порт
lsof -i :8000
```

### Решение 2: Перезапустите сервер

Если сервер уже запущен, но не отвечает:

```bash
# Остановите все процессы
pkill -f "python main.py"

# Подождите 2 секунды
sleep 2

# Запустите заново
source .venv/bin/activate
python main.py
```

### Решение 3: Проверьте в терминале

```bash
# Проверьте, что сервер отвечает
curl http://127.0.0.1:8000/health

# Проверьте docs
curl http://127.0.0.1:8000/docs
```

Если curl возвращает HTML, значит сервер работает и проблема в браузере.

### Решение 4: Очистите кэш браузера

- Chrome/Edge: Ctrl+Shift+Delete (Cmd+Shift+Delete на Mac)
- Firefox: Ctrl+Shift+Delete
- Safari: Cmd+Option+E

### Решение 5: Попробуйте другой адрес

- http://localhost:8000/docs
- http://127.0.0.1:8000/docs
- http://0.0.0.0:8000/docs

### Решение 6: Проверьте файрвол

Убедитесь, что порт 8000 не заблокирован:

```bash
# macOS
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate
```

### Решение 7: Измените порт

Если порт 8000 занят другим приложением:

1. Откройте `.env`
2. Измените `APP_PORT=8000` на `APP_PORT=8001`
3. Перезапустите сервер
4. Откройте http://127.0.0.1:8001/docs

## Проблема: "Address already in use"

Порт занят другим процессом:

```bash
# Найдите процесс
lsof -ti:8000

# Остановите его
kill -9 $(lsof -ti:8000)

# Или используйте другой порт (см. Решение 7 выше)
```

## Проблема: Ошибки импорта

```bash
# Убедитесь, что виртуальное окружение активировано
source .venv/bin/activate

# Переустановите зависимости
pip install -r requirements.txt
```

## Проблема: Модуль не найден

```bash
# Установите недостающие зависимости
pip install fastapi uvicorn python-multipart pydantic pydantic-settings Pillow
```

## Проверка работоспособности

Выполните эти команды по порядку:

```bash
# 1. Активируйте окружение
source .venv/bin/activate

# 2. Проверьте импорты
python -c "from main import app; print('OK')"

# 3. Проверьте конфигурацию
python -c "from app.core.config import settings; print(f'Port: {settings.APP_PORT}')"

# 4. Запустите сервер
python main.py
```

В терминале должно появиться:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

## Логи для диагностики

Если проблема сохраняется, проверьте логи в терминале, где запущен сервер. Там будут указаны конкретные ошибки.

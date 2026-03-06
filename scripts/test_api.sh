#!/bin/bash
# Скрипт для тестирования imageGlue API

set -e

API_URL="http://127.0.0.1:8000"
BASE_URL="${API_URL}/v1"

echo "🧪 Тестирование imageGlue API"
echo "================================"
echo ""

# Цвета для вывода
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Health check
echo -e "${BLUE}1. Health Check${NC}"
echo "GET ${API_URL}/health"
response=$(curl -s "${API_URL}/health")
echo "$response" | python -m json.tool
echo ""

# 2. List templates
echo -e "${BLUE}2. Список шаблонов${NC}"
echo "GET ${BASE_URL}/templates"
response=$(curl -s "${BASE_URL}/templates")
echo "$response" | python -m json.tool
echo ""

# 3. Get template info
echo -e "${BLUE}3. Информация о шаблоне${NC}"
echo "GET ${BASE_URL}/templates/astronaut_dog"
response=$(curl -s "${BASE_URL}/templates/astronaut_dog")
echo "$response" | python -m json.tool
echo ""

# 4. Test job creation (if image provided)
if [ -n "$1" ] && [ -f "$1" ]; then
    echo -e "${BLUE}4. Создание задачи${NC}"
    echo "POST ${BASE_URL}/jobs"
    echo "Файл: $1"
    
    job_response=$(curl -s -X POST "${BASE_URL}/jobs" \
        -F "template_id=astronaut_dog" \
        -F "entity_type=dog" \
        -F "file=@$1")
    
    echo "$job_response" | python -m json.tool
    
    # Extract job_id
    job_id=$(echo "$job_response" | python -c "import sys, json; print(json.load(sys.stdin)['job_id'])" 2>/dev/null || echo "")
    
    if [ -n "$job_id" ]; then
        echo ""
        echo -e "${GREEN}✅ Задача создана: ${job_id}${NC}"
        echo ""
        
        # 5. Get job status
        echo -e "${BLUE}5. Статус задачи${NC}"
        echo "GET ${BASE_URL}/jobs/${job_id}"
        sleep 2  # Даем время на обработку
        status_response=$(curl -s "${BASE_URL}/jobs/${job_id}")
        echo "$status_response" | python -m json.tool
        
        # Check if completed
        status=$(echo "$status_response" | python -c "import sys, json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "")
        
        if [ "$status" = "completed" ]; then
            echo ""
            echo -e "${GREEN}✅ Задача завершена!${NC}"
            final_url=$(echo "$status_response" | python -c "import sys, json; print(json.load(sys.stdin).get('final_image_url', ''))" 2>/dev/null || echo "")
            if [ -n "$final_url" ]; then
                echo -e "${YELLOW}📸 Финальное изображение: ${API_URL}${final_url}${NC}"
            fi
        fi
    fi
else
    echo -e "${YELLOW}4. Создание задачи (пропущено - не указан файл)${NC}"
    echo "   Использование: $0 <путь_к_изображению.jpg>"
    echo ""
fi

echo ""
echo -e "${GREEN}✅ Тестирование завершено${NC}"

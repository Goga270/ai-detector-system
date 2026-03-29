FROM python:3.10-slim

WORKDIR /app

COPY api/requirements.txt ./api/requirements.txt

RUN pip install --no-cache-dir -r api/requirements.txt

COPY api/ ./api/
COPY models/ ./models/

WORKDIR /app/api

# 7. Запускаем сервер
CMD["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

ARG SERVICE_NAME
RUN mkdir -p backend/${SERVICE_NAME}
COPY backend/${SERVICE_NAME} ./backend/${SERVICE_NAME}

#ENV TOKENIZERS_PARALLELISM=false
#ENV OMP_NUM_THREADS=1
#ENV MKL_NUM_THREADS=1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "asyncio"]

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

from handler import predict_text_probability, ai_detection_api

# 1. Инициализация приложения
app = FastAPI(
    title="AI Detection ML Service",
    description="Микросервис для детекции сгенерированных текстов",
    version="1.0.0"
)


class DetectRequest(BaseModel):
    text: str


@app.post("/api/v1/detect")
async def detect_ai_text(request: DetectRequest):
    """
    Принимает текст, прогоняет через Baseline (LogReg + TF-IDF)
    Возвращает JSON с вероятностями.
    """
    if not request.text or len(request.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Текст слишком короткий или пустой")

    response_data = ai_detection_api(request.text)

    if not response_data.get('success'):
        raise HTTPException(status_code=500, detail=response_data.get('error', 'Unknown Error'))

    return response_data

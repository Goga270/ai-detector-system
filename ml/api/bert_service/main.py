from fastapi import FastAPI
from pydantic import BaseModel
import asyncio

app = FastAPI(title="BERT Service")


class TextRequest(BaseModel):
    text: str


@app.post("/score")
async def get_score(req: TextRequest):
    # Здесь логика вашего handler.py (предобработка + инференс) ТЕМА ЧЕРКАНИ СЮДА ТО ЧТО НАДО ИЛИ СКАЖИ ЧТО НАДО))
    await asyncio.sleep(1.2)

    # Mock
    logit_score = 0.82
    return {"model": "bert", "score": logit_score}
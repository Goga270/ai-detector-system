from fastapi import FastAPI
from pydantic import BaseModel
import asyncio

app = FastAPI(title="DetectGPT Service")


class TextRequest(BaseModel):
    text: str


@app.post("/score")
async def get_score(req: TextRequest):
    # ТЕМААА СЮДА НАДО ПО DetectGPT то что надо
    await asyncio.sleep(2)

    # Mock
    perturbation_score = 0.89
    return {"model": "detectgpt", "score": perturbation_score}
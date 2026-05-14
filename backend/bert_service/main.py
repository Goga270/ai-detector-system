import os
from pathlib import Path

import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, Response
from pydantic import BaseModel

from detector import BERTService

load_dotenv()

app = FastAPI(title="BERT Detector Service")
router = APIRouter(prefix="/bert", tags=["bert"])

service: BERTService | None = None


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _resolve_bert_model_path() -> str:
    raw = (os.getenv("BERT_MODEL_PATH") or "backend/bert_service/bert_model").strip()
    if os.path.isdir(raw):
        return str(Path(raw).resolve())
    here = Path(__file__).resolve().parent
    repo = here.parent.parent
    for base in (here, repo):
        cand = (base / raw).resolve()
        if cand.is_dir():
            return str(cand)
    if os.path.isabs(raw):
        return raw
    return raw


@app.on_event("startup")
def load_model():
    global service
    model_path = _resolve_bert_model_path()
    device_env = os.getenv("BERT_DEVICE", "").strip()
    if device_env:
        device = device_env
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    window_size = int(os.getenv("BERT_WINDOW_SIZE", "128"))
    overlap_ratio = float(os.getenv("BERT_OVERLAP_RATIO", "0.8"))
    service = BERTService(
        model_path=model_path,
        device=device,
        window_size=window_size,
        overlap_ratio=overlap_ratio,
    )


class TextRequest(BaseModel):
    text: str
    min_confidence: float = 0.5
    min_span_tokens: int = 3


class SpanResponse(BaseModel):
    start_char: int
    end_char: int
    text: str
    avg_confidence: float


class PredictionResponse(BaseModel):
    spans: list[SpanResponse]
    num_spans: int


@router.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    spans = service.predict_spans(
        text=request.text,
        min_confidence=request.min_confidence,
        min_span_tokens=request.min_span_tokens,
    )
    formatted_spans = [
        SpanResponse(
            start_char=s["start_char"],
            end_char=s["end_char"],
            text=s["text"],
            avg_confidence=s["avg_confidence"],
        )
        for s in spans
    ]
    return PredictionResponse(spans=formatted_spans, num_spans=len(formatted_spans))


@router.get("/health")
def health(response: Response):
    if service is None:
        response.status_code = 503
        return {"status": "loading", "service": "bert"}
    return {"status": "ok", "service": "bert"}


app.include_router(router)


if __name__ == "__main__":
    host = os.getenv("BERT_HOST", "0.0.0.0")
    port = int(os.getenv("BERT_PORT", "8000"))
    reload = _bool("UVICORN_RELOAD", False)
    uvicorn.run("main:app", host=host, port=port, reload=reload)

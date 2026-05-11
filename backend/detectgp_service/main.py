import os

import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel

from detector import DetectGPTService

load_dotenv()

app = FastAPI(title="DetectGPT Service")
router = APIRouter(prefix="/detectgpt", tags=["detectgpt"])

service: DetectGPTService | None = None


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@app.on_event("startup")
def load_model():
    global service
    device_env = os.getenv("DETECTGPT_DEVICE", "").strip()
    if device_env:
        device = device_env
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    service = DetectGPTService(
        target_model_name=os.getenv("DETECTGPT_TARGET_MODEL", "gpt2"),
        perturbation_model_name=os.getenv("DETECTGPT_PERTURBATION_MODEL", "t5-base"),
        num_perturbations=int(os.getenv("DETECTGPT_NUM_PERTURBATIONS", "5")),
        batch_size=int(os.getenv("DETECTGPT_BATCH_SIZE", "4")),
        device=device,
        mask_rate=float(os.getenv("DETECTGPT_MASK_RATE", "0.15")),
        span_length=int(os.getenv("DETECTGPT_SPAN_LENGTH", "4")),
        max_length=int(os.getenv("DETECTGPT_MAX_LENGTH", "512")),
        log_prob_type=os.getenv("DETECTGPT_LOG_PROB_TYPE", "mean"),
    )


class TextRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    curvature: float
    normalized_curvature: float
    original_log_prob: float | None = None
    mean_perturbed_log_prob: float | None = None
    std_perturbed_log_prob: float | None = None


@router.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = service.predict(request.text)
        return PredictionResponse(
            curvature=result["curvature"],
            normalized_curvature=result["normalized_curvature"],
            original_log_prob=result.get("original_log_prob"),
            mean_perturbed_log_prob=result.get("mean_perturbed_log_prob"),
            std_perturbed_log_prob=result.get("std_perturbed_log_prob"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def health():
    return {"status": "ok", "service": "detectgpt"}


app.include_router(router)


if __name__ == "__main__":
    host = os.getenv("DETECTGPT_HOST", "0.0.0.0")
    port = int(os.getenv("DETECTGPT_PORT", "8001"))
    reload = _bool("UVICORN_RELOAD", False)
    uvicorn.run("main:app", host=host, port=port, reload=reload)

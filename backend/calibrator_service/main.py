import os

import uvicorn
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from calibrator import CalibratorService, FinalResult

load_dotenv()

bert_url = os.getenv("CALIBRATOR_BERT_URL", "http://127.0.0.1:8000/bert/predict")
dgpt_url = os.getenv("CALIBRATOR_DETECTGPT_URL", "http://127.0.0.1:8001/detectgpt/predict")
reasoner_url = os.getenv("CALIBRATOR_REASONER_URL", "http://127.0.0.1:8002/arbiter/reasoner")
audit_url = os.getenv("CALIBRATOR_AUDIT_URL", "http://127.0.0.1:8002/arbiter/audit")
defense_url = os.getenv("CALIBRATOR_DEFENSE_URL", "http://127.0.0.1:8002/arbiter/defend")
timeout = int(os.getenv("CALIBRATOR_HTTP_TIMEOUT", "300"))

app = FastAPI(title="Calibrator Service")
router = APIRouter(prefix="/calibrator", tags=["calibrator"])

service = CalibratorService(
    bert_url=bert_url,
    dgpt_url=dgpt_url,
    reasoner_url=reasoner_url,
    audit_url=audit_url,
    defense_url=defense_url,
    timeout=timeout,
)


class TextRequest(BaseModel):
    text: str


class SpanResult(BaseModel):
    start_char: int
    end_char: int
    text: str
    avg_confidence: float


class FinalResultResponse(BaseModel):
    verdict: str
    confidence: float
    ai_percentage: float
    risk_level: str
    spans: List[SpanResult] = []
    explanation: str
    technical_consensus: str
    judge_agreement: float
    needs_human_review: bool
    review_reason: str


@router.post("/calibrate", response_model=FinalResultResponse)
def calibrate(request: TextRequest):
    try:
        result: FinalResult = service.calibrate(request.text)
        formatted_spans = [
            SpanResult(
                start_char=s.get("start_char", 0),
                end_char=s.get("end_char", 0),
                text=s.get("text", ""),
                avg_confidence=s.get("avg_confidence", 0.0),
            )
            for s in result.spans
        ]
        return FinalResultResponse(
            verdict=result.verdict,
            confidence=result.confidence,
            ai_percentage=result.ai_percentage,
            risk_level=result.risk_level,
            spans=formatted_spans,
            explanation=result.explanation,
            technical_consensus=result.technical_consensus,
            judge_agreement=result.judge_agreement,
            needs_human_review=result.needs_human_review,
            review_reason=result.review_reason,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def health():
    return {"status": "ok", "service": "calibrator"}


app.include_router(router)


if __name__ == "__main__":
    host = os.getenv("CALIBRATOR_HOST", "0.0.0.0")
    port = int(os.getenv("CALIBRATOR_PORT", "8003"))
    reload = os.getenv("UVICORN_RELOAD", "0").strip().lower() in ("1", "true", "yes", "on")
    uvicorn.run("main:app", host=host, port=port, reload=reload)

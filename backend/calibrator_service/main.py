import logging
import os
import sys

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


@app.on_event("startup")
def _setup_calibrator_logging() -> None:
    """
    Только setLevel недостаточно: у root по умолчанию lastResort с уровнем WARNING,
    поэтому INFO из дочернего логгера не попадает в stderr/файл логов.
    """
    name = os.getenv("CALIBRATOR_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, name, logging.INFO)
    log = logging.getLogger("calibrator")
    log.setLevel(level)
    if not log.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setLevel(level)
        h.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
        )
        log.addHandler(h)
    log.propagate = False


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


def _build_review_reason(result: FinalResult) -> str:
    raw = result.raw_signals or {}
    parts: List[str] = []
    if raw.get("has_conflict"):
        parts.append("Конфликт сигналов LLM и DetectGPT")
    if not raw.get("audit_passed", True):
        parts.append("Аудит не пройден")
    if result.needs_human_review and float(result.confidence) < service.conf_threshold:
        parts.append("Низкая уверенность после калибровки")
    if result.needs_human_review and not parts:
        parts.append("Рекомендуется ручная проверка")
    return "; ".join(parts) if parts else ""


@router.post("/calibrate", response_model=FinalResultResponse)
def calibrate(request: TextRequest):
    try:
        result: FinalResult = service.calibrate(request.text)
        raw = result.raw_signals or {}
        verdict = str(raw.get("final_verdict_before_calibration", "MIXED"))
        ai_percentage = round(float(result.doc_score) * 100.0, 1)
        technical_consensus = " | ".join(result.top_signals) if result.top_signals else ""

        formatted_spans = [
            SpanResult(
                start_char=s.start,
                end_char=s.end,
                text=s.text,
                avg_confidence=float(s.score),
            )
            for s in result.spans
        ]
        return FinalResultResponse(
            verdict=verdict,
            confidence=float(result.confidence),
            ai_percentage=ai_percentage,
            risk_level=result.risk_level,
            spans=formatted_spans,
            explanation=result.explanation,
            technical_consensus=technical_consensus,
            judge_agreement=float(result.judge_agreement),
            needs_human_review=result.needs_human_review,
            review_reason=_build_review_reason(result),
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

import os

import uvicorn
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from llm_client import YandexGPTClient
from llm_reasoner import LLMReasoner
from judge_d1 import JudgeD1
from judge_d2 import JudgeD2

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def get_api_key() -> str:
    key = os.getenv("YANDEX_API_KEY")
    if not key:
        raise RuntimeError("YANDEX_API_KEY")
    return key


def get_folder_id() -> str:
    return os.getenv("YANDEX_FOLDER_ID", "")


app = FastAPI(title="LLM Arbiter Service")
router = APIRouter(prefix="/arbiter", tags=["arbiter"])

api_key = get_api_key()
folder_id = get_folder_id()
model_type = os.getenv("YANDEX_MODEL_TYPE", "yandexgpt")

llm_client = YandexGPTClient(api_key=api_key, folder_id=folder_id, model_type=model_type)
reasoner = LLMReasoner(llm_client)
judge_d1 = JudgeD1(llm_client)
judge_d2 = JudgeD2(llm_client)


class SpanItem(BaseModel):
    start_char: int
    end_char: int
    confidence: float


class ReasonerRequest(BaseModel):
    text: str
    bert_mean: float
    dgpt_score: float
    spans: List[SpanItem] = []


class ReasonerResponse(BaseModel):
    verdict: str
    confidence: float
    ai_percentage: float
    reasoning: str
    bert_span_analysis: Optional[List[dict]] = None
    detected_spans: Optional[List[dict]] = None
    technical_consensus: Optional[str] = None
    needs_human_review: bool
    review_reason: Optional[str] = None


class AuditRequest(BaseModel):
    text: str
    reasoner_result: ReasonerResponse
    bert_mean: float
    dgpt_score: float
    spans: List[SpanItem] = []


class AuditResponse(BaseModel):
    audit_passed: bool
    critical_errors: List[str]
    bert_spans_ignored: bool
    detectgpt_underestimated: bool
    adjusted_verdict: str
    adjusted_confidence: float
    needs_human_review: bool


class DefenseRequest(BaseModel):
    text: str
    reasoner_result: ReasonerResponse
    bert_mean: float
    dgpt_score: float
    spans: List[SpanItem] = []


class DefenseResponse(BaseModel):
    defense_arguments: List[dict]
    would_overturn: bool
    defense_confidence: float
    proposed_verdict: Optional[str] = None
    explanation: str


def prepare_spans(spans: List[SpanItem]) -> list:
    return [(s.start_char, s.end_char, s.confidence) for s in spans]


@router.post("/reasoner", response_model=ReasonerResponse)
async def run_reasoner(request: ReasonerRequest):
    try:
        result = await reasoner.analyze(
            text=request.text,
            bert_mean=request.bert_mean,
            dgpt_score=request.dgpt_score,
            spans=prepare_spans(request.spans),
        )
        return ReasonerResponse(
            verdict=result.get("verdict", "UNKNOWN"),
            confidence=result.get("confidence", 0.0),
            ai_percentage=result.get("ai_percentage", 0.0),
            reasoning=result.get("reasoning", ""),
            bert_span_analysis=result.get("bert_span_analysis"),
            detected_spans=result.get("detected_spans"),
            technical_consensus=result.get("technical_consensus"),
            needs_human_review=result.get("needs_human_review", False),
            review_reason=result.get("review_reason", ""),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audit", response_model=AuditResponse)
async def run_audit(request: AuditRequest):
    try:
        reasoner_dict = request.reasoner_result.model_dump()
        result = await judge_d1.audit(
            text=request.text,
            reasoner_result=reasoner_dict,
            bert_mean=request.bert_mean,
            dgpt_score=request.dgpt_score,
            spans=prepare_spans(request.spans),
        )
        return AuditResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/defend", response_model=DefenseResponse)
async def run_defense(request: DefenseRequest):
    try:
        reasoner_dict = request.reasoner_result.model_dump()
        result = await judge_d2.defend(
            text=request.text,
            reasoner_result=reasoner_dict,
            bert_mean=request.bert_mean,
            dgpt_score=request.dgpt_score,
            spans=prepare_spans(request.spans),
        )
        return DefenseResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def health():
    return {"status": "ok", "service": "arbiter"}


app.include_router(router)


if __name__ == "__main__":
    host = os.getenv("LLM_ARBITER_HOST", "0.0.0.0")
    port = int(os.getenv("LLM_ARBITER_PORT", "8002"))
    reload = _bool("UVICORN_RELOAD", False)
    uvicorn.run("main:app", host=host, port=port, reload=reload)

import os
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict

load_dotenv()

CALIBRATOR_URL = os.getenv(
    "EXTERNAL_CALIBRATOR_URL",
    "http://127.0.0.1:8003/calibrator/calibrate",
).rstrip("/")

CALIBRATOR_HEALTH_URL = os.getenv(
    "EXTERNAL_CALIBRATOR_HEALTH_URL",
    "http://127.0.0.1:8003/calibrator/health",
)
TIMEOUT = float(os.getenv("EXTERNAL_HTTP_TIMEOUT", "300"))

app = FastAPI(title="external_service")
router = APIRouter(prefix="/gateway", tags=["gateway"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем все источники (для продакшена ограничить)
    allow_credentials=True,
    allow_methods=["*"],  # Разрешаем все методы (GET, POST, OPTIONS и т.д.)
    allow_headers=["*"],  # Разрешаем все заголовки (Content-Type и т.д.)
)

_client: Optional[httpx.Client] = None


@app.on_event("startup")
def startup():
    global _client
    _client = httpx.Client(timeout=TIMEOUT)


@app.on_event("shutdown")
def shutdown():
    global _client
    if _client is not None:
        _client.close()
        _client = None


class TextRequest(BaseModel):
    text: str


class SpanResult(BaseModel):
    start_char: int
    end_char: int
    text: str
    avg_confidence: float


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

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


def _call_calibrator(text: str) -> Dict[str, Any]:
    if _client is None:
        raise HTTPException(status_code=503, detail="HTTP client not ready")
    try:
        r = _client.post(CALIBRATOR_URL, json={"text": text})
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text or r.reason_phrase)
    return r.json()


@router.post("/detect/text", response_model=AnalysisResponse)
def detect_text(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")
    data = _call_calibrator(req.text.strip())
    return AnalysisResponse(**data)


@router.post("/detect/pdf", response_model=AnalysisResponse)
async def detect_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Upload a PDF file")
    contents = await file.read()
    try:
        from pdf_utils import extract_text_from_pdf
    except ImportError as e:
        raise HTTPException(
            status_code=501,
            detail="pdf_utils / pypdf",
        ) from e
    text = extract_text_from_pdf(contents)
    if not text.strip():
        raise HTTPException(status_code=400, detail="PDF contains no extractable text")
    data = _call_calibrator(text.strip())
    return AnalysisResponse(**data)


@router.get("/health")
def health():
    cal_ok = False
    detail: Optional[str] = None
    if _client is not None:
        try:
            r = _client.get(CALIBRATOR_HEALTH_URL)
            cal_ok = r.status_code == 200
        except httpx.RequestError as e:
            detail = str(e)
    return {
        "status": "ok",
        "service": "gateway",
        "calibrator": cal_ok,
        "calibrator_url": CALIBRATOR_URL,
        "detail": detail,
    }


app.include_router(router)


if __name__ == "__main__":
    host = os.getenv("EXTERNAL_HOST", "0.0.0.0")
    port = int(os.getenv("EXTERNAL_PORT", "8004"))
    reload = os.getenv("UVICORN_RELOAD", "0").strip().lower() in ("1", "true", "yes", "on")
    uvicorn.run("main:app", host=host, port=port, reload=reload)

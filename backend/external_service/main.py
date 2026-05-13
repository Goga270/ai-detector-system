import os
import logging
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

app = FastAPI(title="External Gateway Service")

# ==================== CORS MIDDLEWARE ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем все источники (для продакшена ограничить)
    allow_credentials=True,
    allow_methods=["*"],  # Разрешаем все методы (GET, POST, OPTIONS и т.д.)
    allow_headers=["*"],  # Разрешаем все заголовки (Content-Type и т.д.)
)

router = APIRouter(prefix="/gateway", tags=["gateway"])

_client: Optional[httpx.Client] = None


@app.on_event("startup")
def startup():
    global _client
    _client = httpx.Client(timeout=TIMEOUT)
    logger.info(f"Gateway service started, calibrator URL: {CALIBRATOR_URL}")


@app.on_event("shutdown")
def shutdown():
    global _client
    if _client is not None:
        _client.close()
        _client = None
        logger.info("Gateway service shut down")


# ==================== PYDANTIC МОДЕЛИ (обновлены под новую структуру калибратора) ====================
class TextRequest(BaseModel):
    text: str


class SpanResultResponse(BaseModel):
    """Спан в ответе API (новая структура)."""
    text: str
    start: int
    end: int
    score: float
    label: str
    source: str


class AnalysisResponse(BaseModel):
    """Ответ API (адаптирован под новую структуру FinalResult)."""
    model_config = ConfigDict(extra="ignore")

    doc_score: float
    risk_level: str
    spans: List[SpanResultResponse] = []
    explanation: str
    top_signals: List[str] = []
    judge_agreement: float
    confidence: float
    needs_human_review: bool
    raw_signals: Dict[str, Any] = {}


def _call_calibrator(text: str) -> Dict[str, Any]:
    """Вызов калибратора с обработкой ошибок."""
    if _client is None:
        logger.error("HTTP client not ready")
        raise HTTPException(status_code=503, detail="HTTP client not ready")

    logger.info(f"Calling calibrator at {CALIBRATOR_URL}, text length: {len(text)} chars")
    try:
        r = _client.post(CALIBRATOR_URL, json={"text": text})
    except httpx.TimeoutException as e:
        logger.error(f"Calibrator timeout: {str(e)}")
        raise HTTPException(status_code=504, detail=f"Calibrator timeout after {TIMEOUT}s") from e
    except httpx.RequestError as e:
        logger.error(f"Calibrator request error: {str(e)}")
        raise HTTPException(status_code=502, detail=str(e)) from e

    if r.status_code >= 400:
        logger.error(f"Calibrator returned error {r.status_code}: {r.text}")
        raise HTTPException(status_code=r.status_code, detail=r.text or r.reason_phrase)

    logger.info("Calibrator response received successfully")
    return r.json()


# ==================== ЭНДПОИНТЫ ====================
@router.post("/detect/text", response_model=AnalysisResponse)
def detect_text(req: TextRequest):
    """Анализ текста через калибратор."""
    if not req.text.strip():
        logger.warning("Empty text received")
        raise HTTPException(status_code=400, detail="Empty text")

    logger.info(f"Processing text detection request, text length: {len(req.text)} chars")
    data = _call_calibrator(req.text.strip())

    # Преобразуем спаны в нужный формат (если пришли не в том виде)
    if "spans" in data and data["spans"]:
        formatted_spans = []
        for s in data["spans"]:
            # Поддерживаем оба формата (старый и новый)
            if "start_char" in s:
                formatted_spans.append(SpanResultResponse(
                    text=s.get("text", ""),
                    start=s["start_char"],
                    end=s["end_char"],
                    score=s.get("avg_confidence", s.get("score", 0.8)),
                    label=s.get("label", "ai"),
                    source=s.get("source", "llm_reasoner"),
                ))
            else:
                formatted_spans.append(SpanResultResponse(
                    text=s.get("text", ""),
                    start=s.get("start", 0),
                    end=s.get("end", 0),
                    score=s.get("score", 0.8),
                    label=s.get("label", "ai"),
                    source=s.get("source", "llm_reasoner"),
                ))
        data["spans"] = formatted_spans

    return AnalysisResponse(**data)


@router.post("/detect/pdf", response_model=AnalysisResponse)
async def detect_pdf(file: UploadFile = File(...)):
    """Анализ PDF-файла через калибратор."""
    if file.content_type != "application/pdf":
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Upload a PDF file")

    logger.info(f"Processing PDF file: {file.filename}")
    contents = await file.read()

    try:
        from pdf_utils import extract_text_from_pdf
    except ImportError as e:
        logger.error(f"pdf_utils import failed: {str(e)}")
        raise HTTPException(
            status_code=501,
            detail="PDF extraction not available (pdf_utils / pypdf missing)",
        ) from e

    try:
        text = extract_text_from_pdf(contents)
    except Exception as e:
        logger.error(f"PDF extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {str(e)}") from e

    if not text.strip():
        logger.warning("PDF contains no extractable text")
        raise HTTPException(status_code=400, detail="PDF contains no extractable text")

    logger.info(f"PDF extracted, text length: {len(text)} chars")
    data = _call_calibrator(text.strip())

    # Преобразуем спаны в нужный формат
    if "spans" in data and data["spans"]:
        formatted_spans = []
        for s in data["spans"]:
            if "start_char" in s:
                formatted_spans.append(SpanResultResponse(
                    text=s.get("text", ""),
                    start=s["start_char"],
                    end=s["end_char"],
                    score=s.get("avg_confidence", s.get("score", 0.8)),
                    label=s.get("label", "ai"),
                    source=s.get("source", "llm_reasoner"),
                ))
            else:
                formatted_spans.append(SpanResultResponse(
                    text=s.get("text", ""),
                    start=s.get("start", 0),
                    end=s.get("end", 0),
                    score=s.get("score", 0.8),
                    label=s.get("label", "ai"),
                    source=s.get("source", "llm_reasoner"),
                ))
        data["spans"] = formatted_spans

    return AnalysisResponse(**data)


@router.get("/health")
def health():
    """Проверка статуса сервиса и калибратора."""
    cal_ok = False
    detail: Optional[str] = None

    if _client is not None:
        try:
            r = _client.get(CALIBRATOR_HEALTH_URL)
            cal_ok = r.status_code == 200
            if not cal_ok:
                detail = f"Health check returned {r.status_code}"
        except httpx.RequestError as e:
            detail = str(e)
            logger.warning(f"Calibrator health check failed: {detail}")
    else:
        detail = "HTTP client not initialized"

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
    logger.info(f"Starting Gateway Service on {host}:{port}, reload={reload}")
    uvicorn.run("main:app", host=host, port=port, reload=reload)

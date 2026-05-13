import requests
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class SpanResult:
    """Спан, возвращённый LLMReasoner (уже скорректированный)"""
    text: str
    start: int
    end: int
    score: float
    label: str
    source: str = "llm_reasoner"

@dataclass
class FinalResult:
    """Итоговый результат калибратора"""
    doc_score: float
    risk_level: str
    spans: List[SpanResult]
    explanation: str
    top_signals: List[str]
    judge_agreement: float
    confidence: float
    needs_human_review: bool
    raw_signals: Dict[str, Any] = field(default_factory=dict)


class CalibratorService:
    """
    Сервис калибратора, вызывающий микросервисы:
    - BERT (spans)
    - DetectGPT (curvature)
    - LLM Reasoner
    - Judge D1 (audit)
    - Judge D2 (defense)

    Реализует новую логику калибровки (взвешенное комбинирование, штрафы судей, конфликт)
    """
    def __init__(
        self,
        confidence_threshold: float = 0.65,
        llm_weight: float = 0.8,
        detectgpt_weight: float = 0.2,
        penalty_d1: float = 0.3,
        penalty_d2: float = 0.5,
        bert_url: str = "http://127.0.0.1:8000/bert/predict",
        dgpt_url: str = "http://127.0.0.1:8001/detectgpt/predict",
        reasoner_url: str = "http://127.0.0.1:8002/arbiter/reasoner",
        audit_url: str = "http://127.0.0.1:8002/arbiter/audit",
        defense_url: str = "http://127.0.0.1:8002/arbiter/defend",
        timeout: int = 300,
    ):
        self.conf_threshold = confidence_threshold
        self.weights = {"llm": llm_weight, "detectgpt": detectgpt_weight}
        self.penalties = {"Judge-D1": penalty_d1, "Judge-D2": penalty_d2}
        self.thresholds = {"low": 0.3, "medium": 0.7}

        self.bert_url = bert_url
        self.dgpt_url = dgpt_url
        self.reasoner_url = reasoner_url
        self.audit_url = audit_url
        self.defense_url = defense_url
        self.timeout = timeout

        logger.info(f"CalibratorService initialized with thresholds: low={self.thresholds['low']}, medium={self.thresholds['medium']}")
        logger.info(f"Weights: LLM={self.weights['llm']}, DetectGPT={self.weights['detectgpt']}")
        logger.info(f"Penalties: D1={self.penalties['Judge-D1']}, D2={self.penalties['Judge-D2']}")

    def _call_bert(self, text: str) -> Dict[str, Any]:
        """Вызов BERT-сервиса для получения спанов"""
        text_preview = text[:200] + "..." if len(text) > 200 else text
        logger.debug(f"Calling BERT service at {self.bert_url}, text length: {len(text)} chars")
        try:
            resp = requests.post(
                self.bert_url,
                json={"text": text},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            result = resp.json()
            spans_count = len(result.get("spans", []))
            logger.info(f"BERT service responded successfully, spans found: {spans_count}")
            return result
        except requests.exceptions.Timeout:
            logger.error(f"BERT service timeout after {self.timeout}s")
            return {"num_spans": 0, "spans": [], "error": "Timeout"}
        except requests.exceptions.RequestException as e:
            logger.error(f"BERT service request failed: {str(e)}")
            return {"num_spans": 0, "spans": [], "error": str(e)}
        except Exception as e:
            logger.error(f"BERT service unexpected error: {str(e)}")
            return {"num_spans": 0, "spans": [], "error": str(e)}

    def _call_detectgpt(self, text: str) -> Dict[str, Any]:
        """Вызов DetectGPT-сервиса для получения кривизны."""
        text_preview = text[:200] + "..." if len(text) > 200 else text
        logger.debug(f"Calling DetectGPT service at {self.dgpt_url}, text length: {len(text)} chars")
        try:
            resp = requests.post(
                self.dgpt_url,
                json={"text": text},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            result = resp.json()
            curvature = result.get("normalized_curvature", 0.0)
            logger.info(f"DetectGPT service responded successfully, curvature: {curvature:.4f}")
            return result
        except requests.exceptions.Timeout:
            logger.error(f"DetectGPT service timeout after {self.timeout}s")
            return {"curvature": 0.0, "normalized_curvature": 0.0, "error": "Timeout"}
        except requests.exceptions.RequestException as e:
            logger.error(f"DetectGPT service request failed: {str(e)}")
            return {"curvature": 0.0, "normalized_curvature": 0.0, "error": str(e)}
        except Exception as e:
            logger.error(f"DetectGPT service unexpected error: {str(e)}")
            return {"curvature": 0.0, "normalized_curvature": 0.0, "error": str(e)}

    def _call_reasoner(
        self, text: str, bert_mean: float, dgpt_score: float, spans: List[Dict]
    ) -> Optional[Dict]:
        """Вызов LLM Reasoner."""
        text_preview = text[:200] + "..." if len(text) > 200 else text
        logger.debug(f"Calling Reasoner service at {self.reasoner_url}, text length: {len(text)} chars")
        logger.debug(f"Input params: bert_mean={bert_mean:.4f}, dgpt_score={dgpt_score:.4f}, spans_count={len(spans)}")

        payload = {
            "text": text,
            "bert_mean": bert_mean,
            "dgpt_score": dgpt_score,
            "spans": [
                {
                    "start_char": s["start_char"],
                    "end_char": s["end_char"],
                    "confidence": s.get("avg_confidence", 0.5),
                }
                for s in spans
            ],
        }
        try:
            resp = requests.post(self.reasoner_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()
            logger.info(f"Reasoner service responded successfully, verdict: {result.get('verdict', 'UNKNOWN')}, confidence: {result.get('confidence', 0):.4f}")
            return result
        except requests.exceptions.Timeout:
            logger.error(f"Reasoner service timeout after {self.timeout}s")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Reasoner service request failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Reasoner service unexpected error: {str(e)}")
            return None

    def _call_audit(
        self, text: str, reasoner_result: Dict, bert_mean: float, dgpt_score: float, spans: List[Dict]
    ) -> Optional[Dict]:
        """Вызов Judge D1 (auditor)."""
        text_preview = text[:200] + "..." if len(text) > 200 else text
        logger.debug(f"Calling Audit service at {self.audit_url}")

        payload = {
            "text": text,
            "reasoner_result": reasoner_result,
            "bert_mean": bert_mean,
            "dgpt_score": dgpt_score,
            "spans": [
                {
                    "start_char": s["start_char"],
                    "end_char": s["end_char"],
                    "confidence": s.get("avg_confidence", 0.5),
                }
                for s in spans
            ],
        }
        try:
            resp = requests.post(self.audit_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()
            audit_passed = result.get("audit_passed", False)
            logger.info(f"Audit service responded successfully, audit_passed={audit_passed}")
            return result
        except requests.exceptions.Timeout:
            logger.error(f"Audit service timeout after {self.timeout}s")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Audit service request failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Audit service unexpected error: {str(e)}")
            return None

    def _call_defense(
        self, text: str, reasoner_result: Dict, bert_mean: float, dgpt_score: float, spans: List[Dict]
    ) -> Optional[Dict]:
        """Вызов Judge D2 (defense)."""
        text_preview = text[:200] + "..." if len(text) > 200 else text
        logger.debug(f"Calling Defense service at {self.defense_url}")

        payload = {
            "text": text,
            "reasoner_result": reasoner_result,
            "bert_mean": bert_mean,
            "dgpt_score": dgpt_score,
            "spans": [
                {
                    "start_char": s["start_char"],
                    "end_char": s["end_char"],
                    "confidence": s.get("avg_confidence", 0.5),
                }
                for s in spans
            ],
        }
        try:
            resp = requests.post(self.defense_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()
            defense_possible = result.get("defense_possible", False)
            logger.info(f"Defense service responded successfully, defense_possible={defense_possible}")
            return result
        except requests.exceptions.Timeout:
            logger.error(f"Defense service timeout after {self.timeout}s")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Defense service request failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Defense service unexpected error: {str(e)}")
            return None


    def _compute_bert_mean(self, bert_result: Dict) -> float:
        """Вычисляет средний confidence по BERT-спанам"""
        spans = bert_result.get("spans", [])
        if not spans:
            logger.debug("No BERT spans found, mean confidence = 0.0")
            return 0.0
        confs = [s.get("avg_confidence", 0.0) for s in spans]
        mean_conf = sum(confs) / len(confs)
        logger.debug(f"BERT mean confidence computed: {mean_conf:.4f} from {len(spans)} spans")
        return mean_conf

    def _compute_doc_score(self, spans: List[SpanResult], total_tokens: int) -> float:
        """Доля AI-символов в тексте на основе скорректированных спанов"""
        if total_tokens <= 0:
            logger.debug("Total tokens <= 0, doc_score = 0.0")
            return 0.0
        ai_chars = sum(s.end - s.start for s in spans if s.label == "ai")
        doc_score = min(1.0, ai_chars / total_tokens)
        logger.debug(f"Doc score computed: {doc_score:.4f} (AI chars: {ai_chars}, total tokens: {total_tokens})")
        return doc_score

    def _base_prob_from_verdict(self, verdict: str, confidence: float) -> float:
        """Преобразует вердикт и уверенность в базовую вероятность AI"""
        if verdict == "HUMAN":
            prob = 0.0
        elif verdict == "AI":
            prob = 1.0
        else:  # MIXED
            prob = 0.5
        result = 0.5 + (prob - 0.5) * confidence
        logger.debug(f"Base prob from verdict: verdict={verdict}, confidence={confidence:.4f} -> prob={result:.4f}")
        return result

    def _compute_base_confidence(self, final_verdict: str, final_confidence: float,
                                 detectgpt_score: float) -> tuple:
        """
        Вычисляет базовую вероятность AI и уровень конфликта,
        комбинируя финальный вердикт (от get_final_verdict) и DetectGPT
        """
        llm_prob = self._base_prob_from_verdict(final_verdict, final_confidence)

        dgpt_prob = min(1.0, max(0.0, detectgpt_score / 20.0))

        logger.debug(f"LLM probability: {llm_prob:.4f}, DetectGPT probability: {dgpt_prob:.4f}")

        combined_prob = (llm_prob * self.weights["llm"]) + (dgpt_prob * self.weights["detectgpt"])

        conflict = abs(llm_prob - dgpt_prob) > 0.6
        if conflict:
            logger.debug(f"Conflict detected! LLM prob={llm_prob:.4f}, DG prob={dgpt_prob:.4f}, diff={abs(llm_prob - dgpt_prob):.4f}")

        confidence = abs(combined_prob - 0.5) * 2
        if conflict:
            confidence *= 0.7
            logger.debug(f"Conflict penalty applied, confidence reduced to {confidence:.4f}")

        logger.info(f"Base calibration: combined_prob={combined_prob:.4f}, confidence={confidence:.4f}, conflict={conflict}")
        return combined_prob, confidence, conflict

    def _apply_judges(self, prob: float, conf: float,
                      audit_passed: bool, audit_adjusted_verdict: Optional[str],
                      defense_possible: bool, defense_proposed_verdict: Optional[str]) -> tuple:
        """
        Корректирует вероятность и уверенность на основе судей.
        """
        final_prob = prob
        final_conf = conf

        logger.debug(f"Before judges: prob={prob:.4f}, conf={conf:.4f}")


        if not audit_passed:
            final_conf *= (1 - self.penalties["Judge-D1"])
            logger.debug(f"Audit failed, confidence reduced by {self.penalties['Judge-D1']*100:.0f}% to {final_conf:.4f}")
            if audit_adjusted_verdict:
                if audit_adjusted_verdict == "HUMAN":
                    old_prob = final_prob
                    final_prob = max(0.0, final_prob - 0.3)
                    logger.debug(f"Audit adjusted verdict to HUMAN: prob reduced from {old_prob:.4f} to {final_prob:.4f}")
                elif audit_adjusted_verdict == "AI":
                    old_prob = final_prob
                    final_prob = min(1.0, final_prob + 0.2)
                    logger.debug(f"Audit adjusted verdict to AI: prob increased from {old_prob:.4f} to {final_prob:.4f}")
        else:
            old_conf = final_conf
            final_conf = min(1.0, final_conf * 1.05)
            logger.debug(f"Audit passed, confidence increased from {old_conf:.4f} to {final_conf:.4f}")

        if defense_possible and defense_proposed_verdict:
            final_conf *= (1 - self.penalties["Judge-D2"] * (1 - prob))
            logger.debug(f"Defense possible, confidence reduced to {final_conf:.4f}")

            if defense_proposed_verdict == "HUMAN":
                old_prob = final_prob
                final_prob = max(0.0, final_prob - 0.4)
                logger.debug(f"Defense proposed HUMAN: prob reduced from {old_prob:.4f} to {final_prob:.4f}")
            elif defense_proposed_verdict == "MIXED":
                old_prob = final_prob
                final_prob = max(0.0, final_prob - 0.2)
                logger.debug(f"Defense proposed MIXED: prob reduced from {old_prob:.4f} to {final_prob:.4f}")

        logger.info(f"After judges: final_prob={final_prob:.4f}, final_conf={final_conf:.4f}")
        return max(0.0, min(1.0, final_prob)), max(0.0, min(1.0, final_conf))

    def _extract_signals(self, spans: List[SpanResult], detectgpt_score: float,
                         audit_passed: bool, defense_possible: bool) -> List[str]:
        """Извлекает топ признаков для объяснения."""
        signals = []
        if spans:
            signals.append(f"Найдено {len(spans)} AI-фрагментов")
        if detectgpt_score > 5.0:
            signals.append(f"Высокий DetectGPT ({detectgpt_score:.1f})")
        elif detectgpt_score > 2.0:
            signals.append(f"Средний DetectGPT ({detectgpt_score:.1f})")
        if not audit_passed:
            signals.append("Аудитор указал на логические ошибки")
        if defense_possible:
            signals.append("Адвокат обнаружил человеческие маркеры")

        logger.debug(f"Extracted signals: {signals[:4]}")
        return signals[:4]

    def _convert_to_span_results(self, spans: List[Dict], original_text: str) -> List[SpanResult]:
        """Преобразует сырые спаны в формат SpanResult."""
        span_results = []
        for s in spans:
            start = s.get("start_char", 0)
            end = s.get("end_char", 0)
            span_results.append(SpanResult(
                text=original_text[start:end],
                start=start,
                end=end,
                score=s.get("confidence", s.get("avg_confidence", 0.8)),
                label="ai",
                source="llm_reasoner"
            ))
        logger.debug(f"Converted {len(span_results)} spans to SpanResult objects")
        return span_results

    def calibrate(self, text: str) -> FinalResult:
        """
        Основной метод: вызывает все микросервисы, агрегирует результаты.
        Возвращает FinalResult с полями новой структуры.
        """
        logger.info("=" * 60)
        logger.info(f"Starting calibration for text (length: {len(text)} chars)")

        logger.info("Step 1: Calling BERT and DetectGPT services")
        bert_res = self._call_bert(text)
        dgpt_res = self._call_detectgpt(text)

        bert_spans = bert_res.get("spans", [])
        bert_mean = self._compute_bert_mean(bert_res)
        dgpt_score = dgpt_res.get("normalized_curvature", 0.0)

        logger.info(f"BERT: {len(bert_spans)} spans, mean confidence={bert_mean:.4f}")
        logger.info(f"DetectGPT: curvature={dgpt_score:.4f}")

        logger.info("Step 2: Calling Reasoner service")
        reasoner_json = self._call_reasoner(text, bert_mean, dgpt_score, bert_spans)
        if reasoner_json is None:
            logger.warning("Reasoner service failed, using fallback")
            reasoner_json = {
                "verdict": "MIXED",
                "confidence": 0.5,
                "ai_percentage": 0.5,
                "explanation": "LLM Arbiter unavailable",
                "detected_spans": [],
                "technical_consensus": "",
                "needs_human_review": True,
                "review_reason": "LLM service error",
            }

        detected_spans = reasoner_json.get("detected_spans", [])
        if not detected_spans:
            logger.debug("No detected spans from Reasoner, using BERT spans as fallback")
            detected_spans = bert_spans
        span_results = self._convert_to_span_results(detected_spans, text)

        logger.info("Step 3: Calling Audit and Defense services")
        audit_json = self._call_audit(text, reasoner_json, bert_mean, dgpt_score, bert_spans)
        defense_json = self._call_defense(text, reasoner_json, bert_mean, dgpt_score, bert_spans)

        if audit_json is None:
            logger.warning("Audit service failed, using fallback")
            audit_json = {
                "audit_passed": True,
                "adjusted_verdict": reasoner_json.get("verdict", "MIXED"),
                "adjusted_confidence": reasoner_json.get("confidence", 0.5),
                "explanation": "Audit service unavailable",
            }
        if defense_json is None:
            logger.warning("Defense service failed, using fallback")
            defense_json = {
                "defense_possible": False,
                "proposed_verdict": None,
                "defense_confidence": 0.0,
                "explanation": "Defense service unavailable",
            }

        logger.info("Step 4: Running calibration logic")

        final_verdict = reasoner_json.get("verdict", "MIXED")
        final_confidence = reasoner_json.get("confidence", 0.5)

        base_prob, base_conf, has_conflict = self._compute_base_confidence(
            final_verdict, final_confidence, dgpt_score
        )

        audit_passed = audit_json.get("audit_passed", True)
        audit_adjusted_verdict = audit_json.get("adjusted_verdict")
        defense_possible = defense_json.get("defense_possible", False)
        defense_proposed_verdict = defense_json.get("proposed_verdict")

        final_prob, final_conf = self._apply_judges(
            base_prob, base_conf,
            audit_passed, audit_adjusted_verdict,
            defense_possible, defense_proposed_verdict
        )

        if final_prob < self.thresholds["low"]:
            risk = RiskLevel.LOW.value
        elif final_prob < self.thresholds["medium"]:
            risk = RiskLevel.MEDIUM.value
        else:
            risk = RiskLevel.HIGH.value

        d2_strong = defense_possible and defense_proposed_verdict in ("HUMAN", "MIXED")
        needs_review = (final_conf < self.conf_threshold) or has_conflict or (not audit_passed and not d2_strong)

        explanation = (
            f"Ризонер: {reasoner_json.get('explanation', 'No explanation')} | "
            f"Финальный вердикт: {final_verdict} (уверенность {final_confidence:.2f}) | "
            f"После калибровки: {risk} с вероятностью AI {final_prob:.2f}"
        )

        judge_agreement = sum([
            1.0 if audit_passed else 0.5,
            1.0 if not defense_possible else 0.5
        ]) / 2

        total_tokens = len(text.split())
        doc_score_raw = self._compute_doc_score(span_results, total_tokens)

        result = FinalResult(
            doc_score=final_prob,
            risk_level=risk,
            spans=span_results,
            explanation=explanation,
            top_signals=self._extract_signals(span_results, dgpt_score, audit_passed, defense_possible),
            judge_agreement=judge_agreement,
            confidence=final_conf,
            needs_human_review=needs_review,
            raw_signals={
                "base_prob": base_prob,
                "base_conf": base_conf,
                "has_conflict": has_conflict,
                "doc_score_raw": doc_score_raw,
                "audit_passed": audit_passed,
                "defense_possible": defense_possible,
                "final_verdict_before_calibration": final_verdict,
                "final_confidence_before": final_confidence,
                "bert_mean_score": bert_mean,
                "detectgpt_curvature": dgpt_score,
            }
        )

        logger.info(f"Calibration completed: verdict={final_verdict}, risk={risk}, confidence={final_conf:.4f}, needs_review={needs_review}")
        logger.info("=" * 60)

        return result

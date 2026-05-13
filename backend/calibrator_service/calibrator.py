import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import requests

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
            if resp.status_code >= 400:
                body_preview = (resp.text or "")[:2000]
                logger.error(
                    "Reasoner HTTP %s: %s",
                    resp.status_code,
                    body_preview or "(empty body)",
                )
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

    def _compute_base_confidence(
        self,
        final_verdict: str,
        final_confidence: float,
        detectgpt_score: float,
        bert_span_count: int = 0,
    ) -> tuple:
        """
        Вычисляет базовую вероятность AI и уровень конфликта,
        комбинируя финальный вердикт (от get_final_verdict) и DetectGPT.

        Без BERT-спанов высокая кривизна DetectGPT не подтверждена токенным детектором —
        сильно снижаем вклад dgpt и потолок llm_prob (меньше «всегда 96%»).
        """
        llm_prob = self._base_prob_from_verdict(final_verdict, final_confidence)

        dgpt_prob = min(1.0, max(0.0, detectgpt_score / 20.0))

        if bert_span_count == 0:
            dgpt_prob *= 0.42
            if final_verdict == "AI":
                llm_prob = min(llm_prob, 0.62)
            elif final_verdict == "MIXED":
                llm_prob = min(llm_prob, 0.58)
            logger.info(
                "[CALIB] no BERT spans: damped dgpt_prob=%.4f llm_prob=%.4f (verdict=%s)",
                dgpt_prob,
                llm_prob,
                final_verdict,
            )

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

    def _repair_tail_if_single_prefix_span(self, spans: List[Dict], full_text: str) -> List[Dict]:
        """
        Ризонер часто ошибается в end_char при «спан на весь текст».
        Один спан с start=0, не до конца документа, непустой хвост и префикс уже существенный —
        расширяем end до len(full_text), чтобы spans.text совпадал с запросом.
        """
        n = len(full_text)
        if not spans or len(spans) != 1 or n <= 0:
            return spans
        s = dict(spans[0])
        try:
            st = int(s.get("start_char", 0))
            en = int(s.get("end_char", 0))
        except (TypeError, ValueError):
            return spans
        if st != 0 or en >= n:
            return spans
        tail = full_text[en:]
        if not tail.strip():
            return spans
        min_prefix = max(30, n // 5)
        if en < min_prefix:
            logger.debug(
                "[CALIB] span tail repair skipped: prefix_len=%s < min_prefix=%s",
                en,
                min_prefix,
            )
            return spans
        logger.info(
            "[CALIB] span tail repair: extend end %s -> %s (doc_len=%s tail_len=%s)",
            en,
            n,
            n,
            len(tail),
        )
        s["end_char"] = n
        return [s]

    def _convert_to_span_results(self, spans: List[Dict], original_text: str) -> List[SpanResult]:
        """Преобразует сырые спаны в формат SpanResult (текст всегда из original_text по индексам)."""
        n = len(original_text)
        span_results: List[SpanResult] = []
        for i, s in enumerate(spans):
            raw_start = s.get("start_char", 0)
            raw_end = s.get("end_char", 0)
            try:
                start = int(raw_start)
                end = int(raw_end)
            except (TypeError, ValueError):
                logger.warning("Span %s: non-int bounds start=%r end=%r, skip", i, raw_start, raw_end)
                continue
            start = max(0, min(start, n))
            end = max(start, min(end, n))
            if start != int(raw_start) or end != int(raw_end):
                logger.warning(
                    "Span %s: bounds clamped to document len=%s: (%s,%s) -> (%s,%s)",
                    i,
                    n,
                    raw_start,
                    raw_end,
                    start,
                    end,
                )
            sliced = original_text[start:end]
            llm_text = s.get("text")
            if isinstance(llm_text, str) and llm_text and llm_text != sliced:
                logger.info(
                    "Span %s: LLM text field length=%s != slice length=%s (indices %s:%s); using slice from document",
                    i,
                    len(llm_text),
                    len(sliced),
                    start,
                    end,
                )
            if end < n and (n - end) > 20:
                logger.debug(
                    "Span %s: partial coverage end=%s doc_len=%s (%.0f%% of doc)",
                    i,
                    end,
                    n,
                    100.0 * end / n if n else 0.0,
                )
            span_results.append(
                SpanResult(
                    text=sliced,
                    start=start,
                    end=end,
                    score=float(s.get("confidence", s.get("avg_confidence", 0.8))),
                    label="ai",
                    source="llm_reasoner",
                )
            )
        logger.info(
            "Converted %s spans (doc_len=%s); previews: %s",
            len(span_results),
            n,
            [x.text[:48].replace("\n", " ") + ("…" if len(x.text) > 48 else "") for x in span_results[:3]],
        )
        return span_results

    def calibrate(self, text: str) -> FinalResult:
        """
        Основной метод: вызывает все микросервисы, агрегирует результаты.
        Возвращает FinalResult с полями новой структуры.
        """
        logger.info("=" * 60)
        logger.info(
            "[CALIB] start text_len=%s text_sha256_prefix=%s",
            len(text),
            hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
        )

        logger.info("Step 1: Calling BERT and DetectGPT services")
        bert_res = self._call_bert(text)
        dgpt_res = self._call_detectgpt(text)

        bert_spans = bert_res.get("spans", [])
        bert_mean = self._compute_bert_mean(bert_res)
        dgpt_score = dgpt_res.get("normalized_curvature", 0.0)

        logger.info(
            "[CALIB] bert num_spans=%s mean_conf=%.4f err=%s | first_span=%s",
            len(bert_spans),
            bert_mean,
            bert_res.get("error"),
            bert_spans[0] if bert_spans else None,
        )
        logger.info(
            "[CALIB] detectgpt normalized_curvature=%.4f raw_err=%s",
            dgpt_score,
            dgpt_res.get("error"),
        )

        logger.info("Step 2: Calling Reasoner service")
        reasoner_json = self._call_reasoner(text, bert_mean, dgpt_score, bert_spans)
        if reasoner_json is None:
            logger.warning("Reasoner service failed, using fallback")
            reasoner_json = {
                "verdict": "MIXED",
                "confidence": 0.5,
                "ai_percentage": 0.5,
                "reasoning": "LLM Arbiter unavailable",
                "explanation": "LLM Arbiter unavailable",
                "detected_spans": [],
                "technical_consensus": "",
                "needs_human_review": True,
                "review_reason": "LLM service error",
            }

        detected_spans = reasoner_json.get("detected_spans", [])
        ds_bounds = [
            (x.get("start_char"), x.get("end_char"), len((x.get("text") or "")))
            for x in detected_spans[:8]
        ]
        logger.info(
            "[CALIB] reasoner verdict=%s conf=%.4f detected_spans=%s bounds(textlen)=%s reasoning_preview=%s",
            reasoner_json.get("verdict"),
            float(reasoner_json.get("confidence", 0) or 0),
            len(detected_spans),
            ds_bounds,
            (reasoner_json.get("reasoning") or reasoner_json.get("explanation") or "")[:160],
        )
        if not detected_spans:
            logger.debug("No detected spans from Reasoner, using BERT spans as fallback")
            detected_spans = bert_spans
        else:
            detected_spans = self._repair_tail_if_single_prefix_span(detected_spans, text)
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

        logger.info(
            "[CALIB] audit audit_passed=%s adjusted_verdict=%s critical=%s",
            audit_json.get("audit_passed"),
            audit_json.get("adjusted_verdict"),
            audit_json.get("critical_errors"),
        )
        logger.info(
            "[CALIB] defense would_overturn=%s proposed=%s conf=%.3f expl_preview=%s",
            defense_json.get("would_overturn"),
            defense_json.get("proposed_verdict"),
            float(defense_json.get("defense_confidence") or 0),
            str(defense_json.get("explanation") or "")[:120],
        )

        logger.info("Step 4: Running calibration logic")

        final_verdict = reasoner_json.get("verdict", "MIXED")
        final_confidence = reasoner_json.get("confidence", 0.5)

        llm_raw = self._base_prob_from_verdict(final_verdict, final_confidence)
        dgpt_raw = min(1.0, max(0.0, dgpt_score / 20.0))
        base_prob, base_conf, has_conflict = self._compute_base_confidence(
            final_verdict, final_confidence, dgpt_score, len(bert_spans)
        )
        logger.info(
            "[CALIB] scores raw_llm_prob=%.4f raw_dgpt_prob=%.4f (curv=%.3f) bert_spans=%s -> "
            "base_combined=%.4f base_conf=%.4f conflict=%s",
            llm_raw,
            dgpt_raw,
            dgpt_score,
            len(bert_spans),
            base_prob,
            base_conf,
            has_conflict,
        )
        audit_passed = audit_json.get("audit_passed", True)
        audit_adjusted_verdict = audit_json.get("adjusted_verdict")
        defense_possible = defense_json.get(
            "defense_possible",
            defense_json.get("would_overturn", False),
        )
        defense_proposed_verdict = defense_json.get("proposed_verdict")

        final_prob, final_conf = self._apply_judges(
            base_prob, base_conf,
            audit_passed, audit_adjusted_verdict,
            defense_possible, defense_proposed_verdict
        )
        logger.info(
            "[CALIB] after_judges final_prob=%.4f final_conf=%.4f (audit_passed=%s defense_possible=%s)",
            final_prob,
            final_conf,
            audit_passed,
            defense_possible,
        )
        if final_prob < self.thresholds["low"]:
            risk = RiskLevel.LOW.value
        elif final_prob < self.thresholds["medium"]:
            risk = RiskLevel.MEDIUM.value
        else:
            risk = RiskLevel.HIGH.value

        d2_strong = defense_possible and defense_proposed_verdict in ("HUMAN", "MIXED")
        needs_review = (final_conf < self.conf_threshold) or has_conflict or (not audit_passed and not d2_strong)

        reasoner_note = (
            reasoner_json.get("reasoning")
            or reasoner_json.get("explanation")
            or "No explanation"
        )
        explanation = (
            f"Ризонер: {reasoner_note} | "
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
                "bert_num_spans": len(bert_spans),
                "bert_service_error": bert_res.get("error"),
                "detectgpt_curvature": dgpt_score,
            }
        )

        logger.info(
            "[CALIB] done final_prob=%.4f final_conf=%.4f risk=%s needs_review=%s span_count=%s",
            final_prob,
            final_conf,
            risk,
            needs_review,
            len(span_results),
        )
        logger.info("=" * 60)

        return result

import requests
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

@dataclass
class FinalResult:
    verdict: str
    confidence: float
    ai_percentage: float
    risk_level: str
    spans: List[Dict[str, Any]]
    explanation: str
    technical_consensus: str
    judge_agreement: float
    needs_human_review: bool
    review_reason: str

class CalibratorService:
    def __init__(
        self,
        bert_url: str = "http://127.0.0.1:8000/bert/predict",
        dgpt_url: str = "http://127.0.0.1:8001/detectgpt/predict",
        reasoner_url: str = "http://127.0.0.1:8002/arbiter/reasoner",
        audit_url: str = "http://127.0.0.1:8002/arbiter/audit",
        defense_url: str = "http://127.0.0.1:8002/arbiter/defend",
        timeout: int = 300,
    ):
        self.bert_url = bert_url
        self.dgpt_url = dgpt_url
        self.reasoner_url = reasoner_url
        self.audit_url = audit_url
        self.defense_url = defense_url
        self.timeout = timeout

    def _call_bert(self, text: str) -> Dict[str, Any]:
        try:
            resp = requests.post(
                self.bert_url,
                json={"text": text},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"num_spans": 0, "spans": [], "error": str(e)}

    def _call_detectgpt(self, text: str) -> Dict[str, Any]:
        try:
            resp = requests.post(
                self.dgpt_url,
                json={"text": text},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {
                "curvature": 0.0,
                "normalized_curvature": 0.0,
                "error": str(e),
            }

    def _compute_bert_mean(self, bert_result: Dict) -> float:
        spans = bert_result.get("spans", [])
        if not spans:
            return 0.0
        confs = [s.get("avg_confidence", 0.0) for s in spans]
        return sum(confs) / len(confs)

    def _call_reasoner(
        self, text: str, bert_mean: float, dgpt_score: float, spans: List[Dict]
    ) -> Optional[Dict]:
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
            return resp.json()
        except Exception as e:
            return None

    def _call_audit(
        self, text: str, reasoner_result: Dict, bert_mean: float, dgpt_score: float, spans: List[Dict]
    ) -> Optional[Dict]:
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
            return resp.json()
        except Exception as e:
            return None

    def _call_defense(
        self, text: str, reasoner_result: Dict, bert_mean: float, dgpt_score: float, spans: List[Dict]
    ) -> Optional[Dict]:
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
            return resp.json()
        except Exception as e:
            return None

    def calibrate(self, text: str) -> FinalResult:
        bert_res = self._call_bert(text)
        dgpt_res = self._call_detectgpt(text)

        bert_spans = bert_res.get("spans", [])
        bert_mean = self._compute_bert_mean(bert_res)
        dgpt_score = dgpt_res.get("normalized_curvature", 0.0)

        reasoner_json = self._call_reasoner(text, bert_mean, dgpt_score, bert_spans)
        if reasoner_json is None:
            reasoner_json = {
                "verdict": "AI",
                "confidence": 0.5,
                "ai_percentage": 0.5,
                "reasoning": "LLM Arbiter unavailable",
                "technical_consensus": "",
                "needs_human_review": True,
                "review_reason": "LLM service error",
            }

        audit_json = self._call_audit(text, reasoner_json, bert_mean, dgpt_score, bert_spans)
        defense_json = self._call_defense(text, reasoner_json, bert_mean, dgpt_score, bert_spans)

        if audit_json is None:
            audit_json = {
                "audit_passed": True,
                "adjusted_verdict": reasoner_json["verdict"],
                "adjusted_confidence": reasoner_json["confidence"],
                "needs_human_review": reasoner_json.get("needs_human_review", False),
            }
        if defense_json is None:
            defense_json = {
                "defense_confidence": 0.0,
                "would_overturn": False,
            }

        base_verdict = reasoner_json["verdict"]
        base_confidence = reasoner_json["confidence"]
        base_ai_pct = reasoner_json["ai_percentage"]

        if audit_json.get("audit_passed", True):
            adjusted_verdict = base_verdict
            adjusted_confidence = base_confidence
        else:
            adjusted_verdict = audit_json.get("adjusted_verdict", base_verdict)
            adjusted_confidence = audit_json.get("adjusted_confidence", base_confidence) * 0.9

        defense_conf = defense_json.get("defense_confidence", 0.0)
        final_confidence = max(0.0, adjusted_confidence - 0.1 * defense_conf)
        if defense_json.get("would_overturn", False):
            final_confidence *= 0.7

        audit_conf = audit_json.get("adjusted_confidence", base_confidence)
        judge_agreement = max(0.0, 1.0 - abs(audit_conf - defense_conf))

        if final_confidence > 0.7:
            risk_level = "high" if adjusted_verdict == "AI" else "medium"
        elif final_confidence > 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"

        spans = bert_spans

        explanation = (
            f"Reasoner: {base_verdict} ({base_confidence:.2f}) | "
            f"Audit: {audit_json.get('adjusted_verdict', base_verdict)} ({audit_conf:.2f}) | "
            f"Defense confidence: {defense_conf:.2f}"
        )

        return FinalResult(
            verdict=adjusted_verdict,
            confidence=round(final_confidence, 4),
            ai_percentage=round(base_ai_pct, 4),
            risk_level=risk_level,
            spans=spans,
            explanation=explanation,
            technical_consensus=reasoner_json.get("technical_consensus", ""),
            judge_agreement=round(judge_agreement, 4),
            needs_human_review=reasoner_json.get("needs_human_review", False),
            review_reason=reasoner_json.get("review_reason", ""),
        )
    


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

class Calibrator:
    def __init__(
        self,
        confidence_threshold: float = 0.65,
        llm_weight: float = 0.8,
        detectgpt_weight: float = 0.2,
        penalty_d1: float = 0.3,
        penalty_d2: float = 0.5,
    ):
        self.conf_threshold = confidence_threshold
        self.weights = {"llm": llm_weight, "detectgpt": detectgpt_weight}
        self.penalties = {"Judge-D1": penalty_d1, "Judge-D2": penalty_d2}
        self.thresholds = {"low": 0.3, "medium": 0.7}

    def _compute_doc_score(self, spans: List[SpanResult], total_tokens: int) -> float:
        """Доля AI-символов в тексте на основе скорректированных спанов"""
        if total_tokens <= 0:
            return 0.0
        ai_chars = sum(s.end - s.start for s in spans if s.label == "ai")
        return min(1.0, ai_chars / total_tokens)

    def _base_prob_from_verdict(self, verdict: str, confidence: float) -> float:
        """Преобразует вердикт и уверенность в базовую вероятность AI"""
        if verdict == "HUMAN":
            prob = 0.0
        elif verdict == "AI":
            prob = 1.0
        else:
            prob = 0.5
        return 0.5 + (prob - 0.5) * confidence

    def _compute_base_confidence(self, final_verdict: str, final_confidence: float,
                                 detectgpt_score: float) -> tuple:
        """
        Вычисляет базовую вероятность AI и уровень конфликта, комбинируя финальный вердикт (от get_final_verdict) и DetectGPT
        """
        llm_prob = self._base_prob_from_verdict(final_verdict, final_confidence)
        
        dgpt_prob = min(1.0, max(0.0, detectgpt_score / 20.0))
        
        combined_prob = (llm_prob * self.weights["llm"]) + (dgpt_prob * self.weights["detectgpt"])
        
        conflict = abs(llm_prob - dgpt_prob) > 0.6
        
        confidence = abs(combined_prob - 0.5) * 2
        if conflict:
            confidence *= 0.7
        
        return combined_prob, confidence, conflict

    def _apply_judges(self, prob: float, conf: float,
                      audit_passed: bool, audit_adjusted_verdict: Optional[str],
                      defense_possible: bool, defense_proposed_verdict: Optional[str]) -> tuple:
        """
        Корректирует вероятность и уверенность на основе судей
        """
        final_prob = prob
        final_conf = conf
        
        if not audit_passed:
            final_conf *= (1 - self.penalties["Judge-D1"])
            if audit_adjusted_verdict:
                if audit_adjusted_verdict == "HUMAN":
                    final_prob = max(0.0, final_prob - 0.3)
                elif audit_adjusted_verdict == "AI":
                    final_prob = min(1.0, final_prob + 0.2)
        else:
            final_conf = min(1.0, final_conf * 1.05)
        
        if defense_possible and defense_proposed_verdict:
            final_conf *= (1 - self.penalties["Judge-D2"] * (1 - prob))
            if defense_proposed_verdict == "HUMAN":
                final_prob = max(0.0, final_prob - 0.4)
            elif defense_proposed_verdict == "MIXED":
                final_prob = max(0.0, final_prob - 0.2)
        
        return max(0.0, min(1.0, final_prob)), max(0.0, min(1.0, final_conf))

    def _extract_signals(self, spans: List[SpanResult], detectgpt_score: float,
                         audit_passed: bool, defense_possible: bool) -> List[str]:
        """Извлекает топ признаков для объяснения"""
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
        return signals[:4]

    def calibrate(
        self,
        corrected_spans: List[SpanResult],
        detectgpt_score: float,
        llm_explanation: str,
        final_verdict: str,
        final_confidence: float,
        audit_passed: bool,
        audit_adjusted_verdict: Optional[str],
        defense_possible: bool,
        defense_proposed_verdict: Optional[str],
        total_tokens: int,
        original_text: str = ""
    ) -> FinalResult:
        """
        Основной метод калибрации
        """
        doc_score_raw = self._compute_doc_score(corrected_spans, total_tokens)
        
        base_prob, base_conf, has_conflict = self._compute_base_confidence(
            final_verdict, final_confidence, detectgpt_score
        )
        
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
            f"Ризонер: {llm_explanation} | "
            f"Финальный вердикт: {final_verdict} (уверенность {final_confidence:.2f}) | "
            f"После калибровки: {risk} с вероятностью AI {final_prob:.2f}"
        )
        
        judge_agreement = np.mean([
            1.0 if audit_passed else 0.5,
            1.0 if not defense_possible else 0.5
        ])
        
        return FinalResult(
            doc_score=final_prob,
            risk_level=risk,
            spans=corrected_spans,
            explanation=explanation,
            top_signals=self._extract_signals(corrected_spans, detectgpt_score, audit_passed, defense_possible),
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
            }
        )
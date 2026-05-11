import requests
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

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
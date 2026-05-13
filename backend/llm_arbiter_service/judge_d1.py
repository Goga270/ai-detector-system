import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))


import json
import logging
from typing import List, Tuple, Optional, Dict, Any
from llm_client import YandexGPTClient

logger = logging.getLogger(__name__)

class JudgeD1:
    """Судья 1: Аудитор логической консистентности"""
    def __init__(self, client: YandexGPTClient):
        self.client = client
        self.system_prompt = """### ROLE
Аудитор заключений AI-детектора. Проверяете, не пропустил ли ризонер важные технические сигналы и не переоценил ли лингвистику.

### AUDIT RULES (по приоритету)

1. **BERT-спаны проигнорированы?** Если ризонер поставил HUMAN при наличии BERT-спанов длиной >30 токенов — это КРИТИЧЕСКАЯ ошибка (confirmation bias). Вердикт должен быть минимум MIXED.

2. **DetectGPT недооценён?** 
   - DetectGPT > 3.0, а вердикт HUMAN → ошибка
   - DetectGPT > 10.0, а вердикт не AI → вероятно ошибка

3. **Лингвистика перевесила технику?** Если ризонер написал «текст выглядит человеческим» при DetectGPT > 5.0 и наличии спанов — это переоценка лингвистики. Научные/философские тексты часто выглядят «человеческими» по стилю, но генерируются AI.

4. **Спаны правильно интерпретированы?** Проверьте, не назвал ли ризонер реальные AI-фрагменты «ложными срабатываниями BERT» без веских оснований.

### OUTPUT
{
  "audit_passed": true/false,
  "critical_errors": ["список критических ошибок"],
  "bert_spans_ignored": true/false,
  "detectgpt_underestimated": true/false,
  "adjusted_verdict": "AI|HUMAN|MIXED|null",
  "adjusted_confidence": 0.0-1.0,
  "needs_human_review": true/false
}

### THRESHOLDS ДЛЯ АУДИТА
- BERT-спан >30 токенов + вердикт HUMAN → audit_passed = false
- DetectGPT > 3.0 + вердикт HUMAN → audit_passed = false  
- DetectGPT > 10.0 + вердикт не AI → audit_passed = false
- Все сигналы слабые + вердикт AI → audit_passed = false (переоценка)"""

    def _format_spans(self, text: str, spans: List[Tuple[int, int, float]], top_k: int = 5) -> str:
        suspicious = sorted(spans, key=lambda x: x[2], reverse=True)[:top_k]
        if not suspicious:
            return "No spans provided."
        lines = []
        for s, e, score in suspicious:
            snippet = text[s:e].replace('\n', ' ')
            lines.append(f"- [{s}:{e}] {score:.2f}: \"{snippet}\"")
        return "\n".join(lines)

    async def audit(self, text: str,
                    reasoner_verdict: Dict[str, Any],
                    bert_score: float,
                    dgpt_score: float,
                    spans: Optional[List[Tuple[int, int, float]]] = None) -> Dict[str, Any]:
        truncated_text = text[:3000] if len(text) > 3000 else text
        spans_str = self._format_spans(truncated_text, spans) if spans else "No span data."
        user = f"""ORIGINAL TEXT (truncated): {truncated_text}

TECHNICAL SIGNALS:
- BERT Token Score: {bert_score:.3f}
- DetectGPT Curvature: {dgpt_score:.3f}
- Span anomalies:
{spans_str}

REASONER VERDICT:
{json.dumps(reasoner_verdict, ensure_ascii=False, indent=2)}"""

        try:
            raw = await self.client.generate(self.system_prompt, user)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
            if raw.endswith("```"):
                raw = raw[:-3].strip()
            data = json.loads(raw)
            return {
                "audit_passed": bool(data.get("audit_passed", True)),
                "critical_errors": list(data.get("critical_errors") or []),
                "bert_spans_ignored": bool(data.get("bert_spans_ignored", False)),
                "detectgpt_underestimated": bool(data.get("detectgpt_underestimated", False)),
                "adjusted_verdict": data.get("adjusted_verdict") or "MIXED",
                "adjusted_confidence": float(data.get("adjusted_confidence", 0.5)),
                "needs_human_review": bool(data.get("needs_human_review", False)),
            }
        except Exception as e:
            logger.error(f"JudgeD1 failed: {e}")
            return {
                "audit_passed": True,
                "critical_errors": [str(e)],
                "bert_spans_ignored": False,
                "detectgpt_underestimated": False,
                "adjusted_verdict": "MIXED",
                "adjusted_confidence": 0.5,
                "needs_human_review": True,
            }
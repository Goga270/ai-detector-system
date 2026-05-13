import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import json
import logging
from typing import List, Tuple, Optional, Dict, Any
from llm_client import YandexGPTClient

logger = logging.getLogger(__name__)

class JudgeD2:
    """Судья 2: Адвокат автора (поиск человеческих признаков)."""
    def __init__(self, client: YandexGPTClient):
        self.client = client
        self.system_prompt = """### ROLE
Адвокат автора. Ваша задача — найти доказательства, что текст написан ЧЕЛОВЕКОМ, НО только если для этого есть реальные основания. Не придумывайте аргументы, если технические сигналы однозначны.

### ПРАВИЛА ЗАЩИТЫ

**Когда защита ОБОСНОВАНА:**
- BERT-спаны короткие (<10 токенов) и не образуют связного текста
- DetectGPT < 1.0 (слабый сигнал)
- В тексте есть явные человеческие маркеры: опечатки, эмодзи, разговорная речь, личные истории
- BERT и DetectGPT противоречат друг другу

**Когда защита НЕОБОСНОВАНА** (не пытайтесь):
- BERT-спаны длинные (>50 токенов) + DetectGPT > 3.0 — это AI, не тратьте аргументы
- Текст состоит из штампов «в современном мире», «следует отметить» и т.д.
- BERT и DetectGPT оба указывают на AI

### СТРАТЕГИЯ
1. Если сигналы смешанные — ищите человеческие маркеры в НЕ-спановых частях текста
2. Если BERT нашёл спаны, но DetectGPT отрицательный — это сильный аргумент за HUMAN/MIXED
3. Не пытайтесь опровергнуть BERT-спаны длиной >100 токенов — это почти наверняка AI

### OUTPUT
{
  "defense_arguments": [
    {
      "type": "technical_counter|pragmatic|creative",
      "quote": "цитата",
      "argument": "суть",
      "strength": 0.0-1.0
    }
  ],
  "would_overturn": true/false,
  "defense_confidence": 0.0-1.0,
  "proposed_verdict": "HUMAN|MIXED|null",
  "explanation": "кратко"
}"""

    def _format_spans(self, text: str, spans: List[Tuple[int, int, float]], top_k: int = 5) -> str:
        suspicious = sorted(spans, key=lambda x: x[2], reverse=True)[:top_k]
        if not suspicious:
            return "No spans provided."
        lines = []
        for s, e, score in suspicious:
            snippet = text[s:e].replace('\n', ' ')
            lines.append(f"- [{s}:{e}] {score:.2f}: \"{snippet}\"")
        return "\n".join(lines)

    async def defend(self, text: str,
                     current_verdict: Optional[Dict[str, Any]] = None,
                     bert_score: float = 0.0,
                     dgpt_score: float = 0.0,
                     spans: Optional[List[Tuple[int, int, float]]] = None) -> Dict[str, Any]:
        truncated_text = text[:3000] if len(text) > 3000 else text
        spans_str = self._format_spans(truncated_text, spans) if spans else "No span data."
        verdict_str = json.dumps(current_verdict, ensure_ascii=False) if current_verdict else "No verdict yet."

        user = f"""TEXT (truncated): {truncated_text}

TECHNICAL SIGNALS:
- BERT: {bert_score:.3f}
- DetectGPT: {dgpt_score:.3f}
- Span anomalies:
{spans_str}

CURRENT VERDICT:
{verdict_str}"""

        try:
            raw = await self.client.generate(self.system_prompt, user)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
            if raw.endswith("```"):
                raw = raw[:-3].strip()
            data = json.loads(raw)
            return {
                "defense_arguments": list(data.get("defense_arguments") or []),
                "would_overturn": bool(data.get("would_overturn", False)),
                "defense_confidence": float(data.get("defense_confidence", 0.0)),
                "proposed_verdict": data.get("proposed_verdict"),
                "explanation": str(data.get("explanation") or ""),
            }
        except Exception as e:
            logger.error(f"JudgeD2 failed: {e}")
            return {
                "defense_arguments": [],
                "would_overturn": False,
                "defense_confidence": 0.0,
                "proposed_verdict": None,
                "explanation": f"Judge unavailable: {e}",
            }
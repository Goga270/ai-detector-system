import torch
import numpy as np
from typing import List, Dict, Any
from bert_token_scorer import AIDetectorInference

class BERTService:
    def __init__(
        self,
        model_path: str = "bert_model",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        window_size: int = 128,
        overlap_ratio: float = 0.8
    ):
        self.detector = AIDetectorInference(
            model_path=model_path,
            device=device,
            window_size=window_size,
            overlap_ratio=overlap_ratio
        )

    def predict_spans(
        self,
        text: str,
        min_confidence: float = 0.5,
        min_span_tokens: int = 3
    ) -> List[Dict[str, Any]]:
        return self.detector.predict_spans_sliding_window(
            text,
            min_confidence=min_confidence,
            min_span_tokens=min_span_tokens
        )
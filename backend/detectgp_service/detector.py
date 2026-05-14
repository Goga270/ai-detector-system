import threading
import torch
from typing import Dict, Any
from detectgpt_lightweight import LightweightDetectGPT

class DetectGPTService:
    def __init__(
        self,
        target_model_name: str = "gpt2",
        perturbation_model_name: str = "t5-base",
        num_perturbations: int = 5,
        batch_size: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mask_rate: float = 0.15,
        span_length: int = 4,
        max_length: int = 512,
        log_prob_type: str = "mean"
    ):
        self.detector = LightweightDetectGPT(
            target_model_name=target_model_name,
            perturbation_model_name=perturbation_model_name,
            num_perturbations=num_perturbations,
            batch_size=batch_size,
            device=device,
            mask_rate=mask_rate,
            span_length=span_length,
            max_length=max_length,
            log_prob_type=log_prob_type
        )
        self._infer_lock = threading.Lock()

    def predict(self, text: str) -> Dict[str, Any]:
        with self._infer_lock:
            results = self.detector.detect_batch([text])
        return results[0]

    def predict_batch(self, texts: list[str]) -> list[Dict[str, Any]]:
        with self._infer_lock:
            return self.detector.detect_batch(texts)

    @property
    def model_info(self) -> Dict[str, Any]:
        return {
            "target_model": self.detector.target_model_name,
            "perturbation_model": self.detector.perturbation_model_name,
            "device": str(self.detector.device),
            "num_perturbations": self.detector.num_perturbations,
            "max_length": self.detector.max_length
        }

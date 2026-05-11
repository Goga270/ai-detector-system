import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict


class AIDetectorInference:
    def __init__(
        self,
        model_path: str,
        device: str = None,
        window_size: int = 252,
        overlap_ratio: float = 0.5,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.window_size = window_size
        self.stride = int(window_size * (1 - overlap_ratio))

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        self.ai_labels = {"B-AI", "I-AI"}

    def _tokenize_with_offsets(self, text: str):
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=2048,
        )
        return encoding

    def _predict_window(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
            )
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        preds = np.argmax(probs, axis=-1)
        return preds, probs

    def predict_spans_sliding_window(
        self,
        text: str,
        min_confidence: float = 0.3,
        min_span_tokens: int = 3,
    ) -> List[Dict[str, Any]]:
        encoding = self._tokenize_with_offsets(text)
        full_input_ids = encoding["input_ids"][0].tolist()
        full_offset_mapping = encoding["offset_mapping"][0].numpy()
        total_tokens = len(full_input_ids)

        token_votes_ai = Counter()
        token_votes_total = Counter()
        token_confidence_sum = defaultdict(float)

        for start in range(0, max(1, total_tokens - self.window_size + self.stride), self.stride):
            end = min(start + self.window_size, total_tokens)

            window_ids = full_input_ids[start:end]
            window_mask = [1] * len(window_ids)

            if len(window_ids) < self.window_size:
                pad_len = self.window_size - len(window_ids)
                window_ids = window_ids + [self.tokenizer.pad_token_id] * pad_len
                window_mask = window_mask + [0] * pad_len

            input_ids = torch.tensor([window_ids])
            attention_mask = torch.tensor([window_mask])

            preds, probs = self._predict_window(input_ids, attention_mask)

            actual_len = min(self.window_size, end - start)

            for i in range(actual_len):
                token_idx = start + i
                label = self.id2label.get(preds[i], "O")

                if label in self.ai_labels:
                    ai_prob = probs[i][preds[i]]
                else:
                    ai_prob = 1.0 - probs[i][preds[i]]

                token_votes_total[token_idx] += 1

                if label in self.ai_labels and ai_prob >= min_confidence:
                    token_votes_ai[token_idx] += 1
                    token_confidence_sum[token_idx] += ai_prob

        ai_token_indices = []
        token_confidences = {}

        for idx in range(total_tokens):
            total_votes = token_votes_total.get(idx, 0)
            ai_votes = token_votes_ai.get(idx, 0)

            start_char, end_char = full_offset_mapping[idx]
            if start_char == 0 and end_char == 0:
                continue

            if total_votes > 0 and ai_votes / total_votes > 0.5:
                ai_token_indices.append(idx)
                avg_conf = token_confidence_sum[idx] / total_votes
                token_confidences[idx] = avg_conf

        return self._group_into_spans(
            text,
            ai_token_indices,
            full_offset_mapping,
            full_input_ids,
            token_confidences,
            min_span_tokens,
        )

    def _group_into_spans(
        self,
        text: str,
        ai_indices: List[int],
        offset_mapping,
        input_ids: List[int],
        token_confidences: Dict[int, float],
        min_tokens: int = 3,
    ) -> List[Dict[str, Any]]:
        if not ai_indices:
            return []

        spans = []
        current_span = [ai_indices[0]]

        for idx in ai_indices[1:]:
            if idx == current_span[-1] + 1:
                current_span.append(idx)
            else:
                if len(current_span) >= min_tokens:
                    span_info = self._create_span(
                        text, current_span, offset_mapping, input_ids, token_confidences
                    )
                    if span_info:
                        spans.append(span_info)
                current_span = [idx]

        if len(current_span) >= min_tokens:
            span_info = self._create_span(
                text, current_span, offset_mapping, input_ids, token_confidences
            )
            if span_info:
                spans.append(span_info)

        return spans

    def _create_span(
        self,
        text: str,
        token_indices: List[int],
        offset_mapping,
        input_ids: List[int],
        token_confidences: Dict[int, float],
    ) -> Optional[Dict[str, Any]]:
        start_char = offset_mapping[token_indices[0]][0]
        end_char = offset_mapping[token_indices[-1]][1]

        span_text = text[start_char:end_char].strip()
        if not span_text or len(span_text) < 5:
            return None

        confidences = [token_confidences.get(idx, 0.0) for idx in token_indices]
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        return {
            "start_char": int(start_char),
            "end_char": int(end_char),
            "text": span_text,
            "num_tokens": len(token_indices),
            "avg_confidence": avg_confidence,
            "min_confidence": float(np.min(confidences)) if confidences else 0.0,
            "max_confidence": float(np.max(confidences)) if confidences else 0.0,
        }

    def predict_tokens(self, text: str, return_probs: bool = False) -> Dict[str, Any]:
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
        )
        offset_mapping = encoding.pop("offset_mapping")[0].numpy()

        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in encoding.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits[0]

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        predictions = np.argmax(probs, axis=-1)
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

        return {
            "tokens": tokens,
            "labels": [self.id2label.get(p, "UNK") for p in predictions],
            "confidences": [float(max(p)) for p in probs],
        }

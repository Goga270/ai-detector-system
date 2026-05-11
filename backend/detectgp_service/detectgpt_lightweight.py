import re
import torch
from tqdm import tqdm
import numpy as np
import random
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightweightDetectGPT:
    def __init__(
        self,
        target_model_name: str = "gpt2",
        perturbation_model_name: str = "t5-base",
        num_perturbations: int = 10,
        batch_size: int = 4,
        device: Optional[str] = None,
        mask_rate: float = 0.15,
        span_length: int = 2,
        max_length: int = 512,
        log_prob_type: str = "mean",
    ):
        self.target_model_name = target_model_name
        self.perturbation_model_name = perturbation_model_name
        self.num_perturbations = num_perturbations
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mask_rate = mask_rate
        self.span_length = span_length
        self.max_length = max_length
        self.log_prob_type = log_prob_type

        logger.info(f"Loading target model: {self.target_model_name}")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.target_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.target_model.eval()
        self.target_tokenizer = AutoTokenizer.from_pretrained(self.target_model_name)
        if self.target_tokenizer.pad_token is None:
            self.target_tokenizer.pad_token = self.target_tokenizer.eos_token

        logger.info(f"Loading perturbation model: {self.perturbation_model_name}")
        self.perturb_model = T5ForConditionalGeneration.from_pretrained(
            self.perturbation_model_name
        ).to(self.device)
        self.perturb_model.eval()
        self.perturb_tokenizer = T5Tokenizer.from_pretrained(self.perturbation_model_name)

        self.max_sentinels = self.perturb_tokenizer._extra_ids
        logger.info(f"Maximum sentinel tokens available: {self.max_sentinels}")

    def _tokenize_for_masking(self, text: str):
        enc = self.perturb_tokenizer(text, add_special_tokens=False, return_tensors="pt")
        return enc["input_ids"][0]

    def _mask_tokens(self, input_ids: torch.Tensor) -> (str, List[int]):
        num_tokens = len(input_ids)
        if num_tokens == 0:
            return "", []

        num_to_mask = max(1, int(num_tokens * self.mask_rate))
        num_spans = max(1, num_to_mask // self.span_length)

        if num_spans > self.max_sentinels:
            num_spans = self.max_sentinels
            logger.warning(f"Reducing number of spans to {num_spans} due to sentinel limit.")

        masked_indices = set()
        attempts = 0
        max_attempts = num_spans * 10
        while len(masked_indices) < num_spans * self.span_length and attempts < max_attempts:
            start = random.randint(0, num_tokens - self.span_length)
            for i in range(start, start + self.span_length):
                masked_indices.add(i)
            attempts += 1

        masked_positions = sorted(masked_indices)

        spans = []
        if masked_positions:
            current_span = [masked_positions[0]]
            for pos in masked_positions[1:]:
                if pos == current_span[-1] + 1:
                    current_span.append(pos)
                else:
                    spans.append(current_span)
                    current_span = [pos]
            spans.append(current_span)

            spans = spans[:num_spans]

        sentinel_id = 0
        token_pieces = []
        last_idx = 0

        for span in spans:
            token_pieces.append(
                self.perturb_tokenizer.decode(
                    input_ids[last_idx:span[0]], skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            )
            sentinel_token = f"<extra_id_{sentinel_id}>"
            token_pieces.append(sentinel_token)
            sentinel_id += 1
            last_idx = span[-1] + 1

        if last_idx < num_tokens:
            token_pieces.append(
                self.perturb_tokenizer.decode(
                    input_ids[last_idx:], skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            )

        masked_text = " ".join(token_pieces).strip()
        return masked_text, spans

    def _fill_masked_text(self, masked_text: str, t5_output: str) -> str:
        sentinel_pattern = r"<extra_id_(\d+)>"
        parts = re.split(sentinel_pattern, t5_output)
        fills = {}
        i = 0
        while i < len(parts)-1:
            if parts[i] == "":
                try:
                    num = int(parts[i+1])
                    fill_text = parts[i+2] if i+2 < len(parts) else ""
                    fills[num] = fill_text.strip()
                    i += 3
                except ValueError:
                    i += 1
            else:
                i += 1

        filled = masked_text
        for num in sorted(fills.keys(), reverse=True):
            sentinel = f"<extra_id_{num}>"
            fill = fills[num]
            filled = filled.replace(sentinel, fill, 1)

        filled = re.sub(r"<extra_id_\d+>", "", filled)
        filled = re.sub(r"\s+", " ", filled).strip()
        return filled

    def generate_perturbations_batch(self, texts: List[str]) -> List[List[str]]:
        all_perturbations = []
        for batch_start in tqdm(range(0, len(texts), self.batch_size), desc="Perturbing texts"):
            batch_texts = texts[batch_start:batch_start + self.batch_size]
            batch_perturbations = [[] for _ in range(len(batch_texts))]

            for _ in range(self.num_perturbations):
                masked_texts = []
                for text in batch_texts:
                    input_ids = self._tokenize_for_masking(text)
                    masked, _spans = self._mask_tokens(input_ids)
                    masked_texts.append(masked)

                inputs = self.perturb_tokenizer(
                    masked_texts, return_tensors="pt", max_length=self.max_length,
                    truncation=True, padding=True
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.perturb_model.generate(
                        **inputs,
                        max_length=self.max_length,
                        do_sample=True,
                        temperature=1.0,
                        top_p=0.96,
                        num_return_sequences=1
                    )

                decoded_fills = self.perturb_tokenizer.batch_decode(outputs, skip_special_tokens=False)

                for i, (masked_text, fill) in enumerate(zip(masked_texts, decoded_fills)):
                    filled_text = self._fill_masked_text(masked_text, fill)
                    batch_perturbations[i].append(filled_text)

            all_perturbations.extend(batch_perturbations)
        return all_perturbations

    def compute_log_probs_batch(self, texts: List[str]) -> List[float]:
        if not texts:
            return []
        all_log_probs = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Computing log probs"):
            batch_texts = texts[i:i + self.batch_size]
            inputs = self.target_tokenizer(
                batch_texts, return_tensors="pt", truncation=True,
                max_length=self.max_length, padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.target_model(**inputs, labels=inputs["input_ids"])
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs["input_ids"][..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss.view(shift_labels.size())

                mask = (shift_labels != self.target_tokenizer.pad_token_id).float()
                if self.log_prob_type == "mean":
                    token_loss = (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                else:
                    token_loss = (loss * mask).sum(dim=1)

                all_log_probs.extend((-token_loss).cpu().tolist())
        return all_log_probs

    def detect_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        logger.info(f"Batch detecting {len(texts)} texts")
        all_perturbations = self.generate_perturbations_batch(texts)

        all_texts_for_logprobs = list(texts)
        for perts in all_perturbations:
            all_texts_for_logprobs.extend(perts)

        all_log_probs = self.compute_log_probs_batch(all_texts_for_logprobs)
        original_log_probs = all_log_probs[:len(texts)]

        results = []
        idx = len(texts)
        for i in range(len(texts)):
            original_lp = original_log_probs[i]
            perturbed_lps = all_log_probs[idx:idx + self.num_perturbations]
            idx += self.num_perturbations

            mean_perturbed = np.mean(perturbed_lps)
            std_perturbed = np.std(perturbed_lps) if len(perturbed_lps) > 1 else 1.0

            curvature = original_lp - mean_perturbed
            normalized_curvature = curvature / std_perturbed if std_perturbed > 0 else 0

            results.append({
                "curvature": curvature,
                "normalized_curvature": normalized_curvature,
                "original_log_prob": original_lp,
                "mean_perturbed_log_prob": mean_perturbed,
                "std_perturbed_log_prob": std_perturbed
            })

        return results

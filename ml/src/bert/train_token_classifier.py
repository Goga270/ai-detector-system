#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import inspect
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from datasets import Dataset, DatasetDict, load_from_disk
from sklearn.metrics import classification_report, precision_recall_fscore_support
from seqeval.metrics import (
    precision_score as seqeval_precision_score,
    recall_score as seqeval_recall_score,
    f1_score as seqeval_f1_score,
)
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

LABEL2ID = {
    "O": 0,
    "B-AI": 1,
    "I-AI": 2,
}

ID2LABEL = {
    0: "O",
    1: "B-AI",
    2: "I-AI",
}

IGNORE_INDEX = -100


@dataclass
class TrainConfig:
    # data sources
    hf_dataset_dir: str = ""
    bio_data_dir: str = "../../data/token_detector_bio"

    # model / output
    model_name: str = "microsoft/mdeberta-v3-base"
    output_dir: str = "./token_detector_model_v2"

    # mode
    train_mode: str = "head_only"   # head_only | partial_unfreeze | full
    partial_unfreeze_last_n_layers: int = 2

    # optimization
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    warmup_steps: int = 0
    lr_scheduler_type: str = "linear"
    max_grad_norm: float = 1.0

    # runtime
    logging_steps: int = 50
    save_total_limit: int = 1
    dataloader_num_workers: int = 2
    seed: int = 42
    fp16: bool = False
    bf16: bool = False
    cpu_only: bool = False
    gradient_checkpointing: bool = False
    group_by_length: bool = True

    # loss
    use_focal_loss: bool = True
    focal_gamma: float = 1.5
    use_class_balanced_alpha: bool = True
    effective_num_beta: float = 0.9999
    alpha_clip_max: float = 10.0
    manual_alpha: Optional[List[float]] = None

    # sampler
    use_weighted_sampler: bool = True
    sampler_human_weight: float = 1.0
    sampler_ai_weight: float = 1.5
    sampler_bai_weight: float = 4.0

    # save/eval
    eval_strategy: str = "epoch"
    save_strategy: str = "no"
    eval_steps: int = 500
    save_steps: int = 500
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_seqeval_f1"
    greater_is_better: bool = True

    # misc
    save_final_model_only: bool = True


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device_info(cpu_only: bool = False) -> Dict[str, Any]:
    mps_available = bool(
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    )
    cuda_available = torch.cuda.is_available() and not cpu_only

    return {
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "mps_available": mps_available and not cpu_only,
        "cpu_only": bool(cpu_only),
        "device_selected": (
            "cpu" if cpu_only else
            "cuda" if cuda_available else
            "mps" if mps_available else
            "cpu"
        ),
    }


def load_hf_dataset(dataset_dir: Path) -> DatasetDict:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    ds = load_from_disk(str(dataset_dir))
    required = {"train", "validation", "test"}
    missing = required - set(ds.keys())
    if missing:
        raise ValueError(f"Dataset missing splits: {missing}")
    return ds


def keep_only_training_columns(dataset: DatasetDict) -> DatasetDict:
    required_columns = {"input_ids", "attention_mask", "labels"}
    cleaned = {}

    for split_name, split_ds in dataset.items():
        missing = required_columns - set(split_ds.column_names)
        if missing:
            raise ValueError(f"Split {split_name} missing columns: {missing}")

        cols_to_remove = [c for c in split_ds.column_names if c not in required_columns]
        cleaned[split_name] = split_ds.remove_columns(cols_to_remove)

    return DatasetDict(cleaned)


def load_bio_jsonl_dataset(bio_data_dir: Path) -> DatasetDict:
    train_path = bio_data_dir / "bio_train.jsonl"
    val_path = bio_data_dir / "bio_val.jsonl"
    test_path = bio_data_dir / "bio_test.jsonl"

    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path)
    test_rows = read_jsonl(test_path)

    def to_minimal(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "input_ids": r["input_ids"],
                "attention_mask": r["attention_mask"],
                "labels": r["labels"],
            }
            for r in records
        ]

    return DatasetDict({
        "train": Dataset.from_list(to_minimal(train_rows)),
        "validation": Dataset.from_list(to_minimal(val_rows)),
        "test": Dataset.from_list(to_minimal(test_rows)),
    })


def validate_inmemory_dataset(ds: DatasetDict) -> DatasetDict:
    if not isinstance(ds, DatasetDict):
        raise TypeError("dataset_override must be a datasets.DatasetDict")

    required_splits = {"train", "validation", "test"}
    missing = required_splits - set(ds.keys())
    if missing:
        raise ValueError(f"dataset_override missing splits: {missing}")

    return keep_only_training_columns(ds)


def resolve_dataset(
    config: TrainConfig,
    dataset_override: Optional[DatasetDict] = None,
) -> Tuple[DatasetDict, str]:
    if dataset_override is not None:
        ds = validate_inmemory_dataset(dataset_override)
        return ds, "dataset_override"

    if config.hf_dataset_dir:
        ds = load_hf_dataset(Path(config.hf_dataset_dir))
        ds = keep_only_training_columns(ds)
        return ds, "hf_dataset"

    ds = load_bio_jsonl_dataset(Path(config.bio_data_dir))
    return ds, "bio_jsonl"


def count_label_frequencies(split_dataset) -> Dict[int, int]:
    counts = {0: 0, 1: 0, 2: 0}
    for ex in split_dataset:
        for label in ex["labels"]:
            label = int(label)
            if label == IGNORE_INDEX:
                continue
            counts[label] += 1
    return counts


def count_all_ignore_examples(split_dataset) -> int:
    n = 0
    for ex in split_dataset:
        valid = [int(x) for x in ex["labels"] if int(x) != IGNORE_INDEX]
        if len(valid) == 0:
            n += 1
    return n


def summarize_dataset_labels(dataset: DatasetDict) -> Dict[str, Any]:
    summary = {}
    for split_name, split_ds in dataset.items():
        lengths = [len(ex["labels"]) for ex in split_ds]
        summary[split_name] = {
            "num_examples": len(split_ds),
            "avg_seq_len": float(np.mean(lengths)) if lengths else 0.0,
            "label_counts": count_label_frequencies(split_ds),
            "all_ignore_examples": count_all_ignore_examples(split_ds),
            "first_labels_preview": split_ds[0]["labels"][:40] if len(split_ds) > 0 else [],
        }
    return summary


def build_example_sampling_weights(split_dataset, human_weight: float, ai_weight: float, bai_weight: float) -> List[float]:
    weights = []
    for ex in split_dataset:
        labels = [int(x) for x in ex["labels"] if int(x) != IGNORE_INDEX]
        if any(x == LABEL2ID["B-AI"] for x in labels):
            weights.append(float(bai_weight))
        elif any(x == LABEL2ID["I-AI"] for x in labels):
            weights.append(float(ai_weight))
        else:
            weights.append(float(human_weight))
    return weights


def compute_effective_number_alpha(
    label_counts: Dict[int, int],
    beta: float = 0.9999,
    clip_max: float = 10.0,
) -> torch.Tensor:
    counts = np.array([
        max(label_counts.get(0, 0), 1),
        max(label_counts.get(1, 0), 1),
        max(label_counts.get(2, 0), 1),
    ], dtype=np.float64)

    effective_num = 1.0 - np.power(beta, counts)
    alpha = (1.0 - beta) / effective_num
    alpha = alpha / np.mean(alpha)
    alpha = np.clip(alpha, 0.0, clip_max)

    return torch.tensor(alpha, dtype=torch.float32)


class TokenFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 1.5,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid_mask = targets != self.ignore_index
        if valid_mask.sum() == 0:
            return logits.new_tensor(0.0)

        logits = logits[valid_mask]
        targets = targets[valid_mask]

        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)

        target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        target_probs = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        ce = -target_log_probs
        focal_factor = torch.pow(1.0 - target_probs, self.gamma)
        loss = focal_factor * ce

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[targets]
            loss = alpha_t * loss

        return loss.mean()


def freeze_model_backbone(model) -> None:
    for _, p in model.named_parameters():
        p.requires_grad = False
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True


def partial_unfreeze_last_layers(model, last_n_layers: int) -> None:
    for _, p in model.named_parameters():
        p.requires_grad = False

    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True

    if last_n_layers > 0 and hasattr(model, "deberta") and hasattr(model.deberta, "encoder"):
        layers = model.deberta.encoder.layer
        total_layers = len(layers)
        start_idx = max(0, total_layers - last_n_layers)
        for idx in range(start_idx, total_layers):
            for p in layers[idx].parameters():
                p.requires_grad = True


def count_trainable_params(model) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_compute_metrics_fn():
    def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        y_true_flat: List[int] = []
        y_pred_flat: List[int] = []

        y_true_seq: List[List[str]] = []
        y_pred_seq: List[List[str]] = []

        for pred_row, label_row in zip(preds, labels):
            cur_true = []
            cur_pred = []

            for pred_id, label_id in zip(pred_row, label_row):
                if int(label_id) == IGNORE_INDEX:
                    continue
                y_true_flat.append(int(label_id))
                y_pred_flat.append(int(pred_id))
                cur_true.append(ID2LABEL[int(label_id)])
                cur_pred.append(ID2LABEL[int(pred_id)])

            y_true_seq.append(cur_true)
            y_pred_seq.append(cur_pred)

        if len(y_true_flat) == 0:
            return {
                "token_precision_macro": 0.0,
                "token_recall_macro": 0.0,
                "token_f1_macro": 0.0,
                "token_precision_weighted": 0.0,
                "token_recall_weighted": 0.0,
                "token_f1_weighted": 0.0,
                "ai_token_precision": 0.0,
                "ai_token_recall": 0.0,
                "ai_token_f1": 0.0,
                "seqeval_precision": 0.0,
                "seqeval_recall": 0.0,
                "seqeval_f1": 0.0,
                "class_O_precision": 0.0,
                "class_O_recall": 0.0,
                "class_O_f1": 0.0,
                "class_B-AI_precision": 0.0,
                "class_B-AI_recall": 0.0,
                "class_B-AI_f1": 0.0,
                "class_I-AI_precision": 0.0,
                "class_I-AI_recall": 0.0,
                "class_I-AI_f1": 0.0,
            }

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true_flat, y_pred_flat, average="macro", zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true_flat, y_pred_flat, average="weighted", zero_division=0
        )
        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
            y_true_flat, y_pred_flat, labels=[0, 1, 2], average=None, zero_division=0
        )

        y_true_ai = [0 if y == LABEL2ID["O"] else 1 for y in y_true_flat]
        y_pred_ai = [0 if y == LABEL2ID["O"] else 1 for y in y_pred_flat]
        ai_precision, ai_recall, ai_f1, _ = precision_recall_fscore_support(
            y_true_ai, y_pred_ai, average="binary", zero_division=0
        )

        seqeval_precision = seqeval_precision_score(y_true_seq, y_pred_seq)
        seqeval_recall = seqeval_recall_score(y_true_seq, y_pred_seq)
        seqeval_f1 = seqeval_f1_score(y_true_seq, y_pred_seq)

        return {
            "token_precision_macro": float(precision_macro),
            "token_recall_macro": float(recall_macro),
            "token_f1_macro": float(f1_macro),

            "token_precision_weighted": float(precision_weighted),
            "token_recall_weighted": float(recall_weighted),
            "token_f1_weighted": float(f1_weighted),

            "ai_token_precision": float(ai_precision),
            "ai_token_recall": float(ai_recall),
            "ai_token_f1": float(ai_f1),

            "seqeval_precision": float(seqeval_precision),
            "seqeval_recall": float(seqeval_recall),
            "seqeval_f1": float(seqeval_f1),

            "class_O_precision": float(per_class_precision[0]),
            "class_O_recall": float(per_class_recall[0]),
            "class_O_f1": float(per_class_f1[0]),

            "class_B-AI_precision": float(per_class_precision[1]),
            "class_B-AI_recall": float(per_class_recall[1]),
            "class_B-AI_f1": float(per_class_f1[1]),

            "class_I-AI_precision": float(per_class_precision[2]),
            "class_I-AI_recall": float(per_class_recall[2]),
            "class_I-AI_f1": float(per_class_f1[2]),
        }

    return compute_metrics


def build_token_classification_report(
    trainer: Trainer,
    dataset_split,
    output_path: Path,
) -> Dict[str, Any]:
    pred_out = trainer.predict(dataset_split)
    logits = pred_out.predictions
    labels = pred_out.label_ids
    preds = np.argmax(logits, axis=-1)

    y_true_flat: List[int] = []
    y_pred_flat: List[int] = []

    for pred_row, label_row in zip(preds, labels):
        for p, y in zip(pred_row, label_row):
            if int(y) == IGNORE_INDEX:
                continue
            y_true_flat.append(int(y))
            y_pred_flat.append(int(p))

    report = classification_report(
        y_true_flat,
        y_pred_flat,
        labels=[0, 1, 2],
        target_names=["O", "B-AI", "I-AI"],
        output_dict=True,
        zero_division=0,
    )
    write_json(output_path, report)
    return report


def _trainer_preprocess_kw(tokenizer: Any) -> Dict[str, Any]:
    params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in params:
        return {"processing_class": tokenizer}
    if "tokenizer" in params:
        return {"tokenizer": tokenizer}
    return {}


def build_training_arguments_compat(
    config: TrainConfig,
    output_dir: Path,
    use_cuda: bool,
    use_fp16: bool,
    use_bf16: bool,
) -> TrainingArguments:
    signature = inspect.signature(TrainingArguments.__init__)
    params = set(signature.parameters.keys())

    kwargs = {
        "output_dir": str(output_dir / "checkpoints"),
        "overwrite_output_dir": True,

        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_train_epochs": config.num_train_epochs,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "lr_scheduler_type": config.lr_scheduler_type,

        "logging_strategy": "steps",
        "logging_steps": config.logging_steps,
        "save_strategy": config.save_strategy,
        "save_total_limit": config.save_total_limit,

        "report_to": "none",
        "seed": config.seed,
        "remove_unused_columns": False,

        "fp16": use_fp16,
        "bf16": use_bf16,

        "use_cpu": bool(config.cpu_only),
        "dataloader_pin_memory": bool(use_cuda),
        "dataloader_num_workers": config.dataloader_num_workers,
        "group_by_length": config.group_by_length,
    }

    if config.warmup_steps > 0:
        kwargs["warmup_steps"] = config.warmup_steps
    else:
        kwargs["warmup_ratio"] = config.warmup_ratio

    if "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = config.eval_strategy
    elif "eval_strategy" in params:
        kwargs["eval_strategy"] = config.eval_strategy

    if config.eval_strategy == "steps":
        kwargs["eval_steps"] = config.eval_steps
    if config.save_strategy == "steps":
        kwargs["save_steps"] = config.save_steps

    if "gradient_checkpointing" in params:
        kwargs["gradient_checkpointing"] = config.gradient_checkpointing

    if config.load_best_model_at_end and config.save_strategy != "no":
        kwargs["load_best_model_at_end"] = True
        kwargs["metric_for_best_model"] = config.metric_for_best_model
        kwargs["greater_is_better"] = config.greater_is_better
    else:
        kwargs["load_best_model_at_end"] = False

    return TrainingArguments(**kwargs)


def build_model(model_name: str):
    return AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )


class TokenTrainer(Trainer):
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        focal_gamma: float = 1.5,
        use_focal_loss: bool = True,
        train_example_weights: Optional[List[float]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.focal_gamma = focal_gamma
        self.use_focal_loss = use_focal_loss
        self.train_example_weights = train_example_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        logits = outputs.logits.float()

        if self.use_focal_loss:
            loss_fct = TokenFocalLoss(
                alpha=self.alpha,
                gamma=self.focal_gamma,
                ignore_index=IGNORE_INDEX,
            )
            loss = loss_fct(
                logits.view(-1, model.config.num_labels),
                labels.view(-1).long(),
            )
        else:
            loss_fct = nn.CrossEntropyLoss(
                weight=self.alpha.to(logits.device) if self.alpha is not None else None,
                ignore_index=IGNORE_INDEX,
            )
            loss = loss_fct(
                logits.view(-1, model.config.num_labels),
                labels.view(-1).long(),
            )

        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer requires train_dataset.")

        if self.train_example_weights is None:
            return super().get_train_dataloader()

        sampler = WeightedRandomSampler(
            weights=torch.tensor(self.train_example_weights, dtype=torch.double),
            num_samples=len(self.train_example_weights),
            replacement=True,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
        )


def save_training_metadata(
    output_dir: Path,
    config: TrainConfig,
    device_info: Dict[str, Any],
    dataset: DatasetDict,
    dataset_source: str,
    dataset_label_summary: Dict[str, Any],
    alpha: Optional[torch.Tensor],
    total_params: int,
    trainable_params: int,
    train_example_weights_summary: Optional[Dict[str, float]],
) -> None:
    metadata = {
        "run_config": asdict(config),
        "device_info": device_info,
        "dataset_source": dataset_source,
        "dataset_shapes": {k: len(v) for k, v in dataset.items()},
        "dataset_label_summary": dataset_label_summary,
        "alpha": alpha.tolist() if alpha is not None else None,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_example_weights_summary": train_example_weights_summary,
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
    }
    write_json(output_dir / "run_metadata.json", metadata)


def run_training(
    config: Optional[TrainConfig] = None,
    dataset_override: Optional[DatasetDict] = None,
) -> None:
    if config is None:
        config = TrainConfig()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[TRAIN] Detecting device...")
    device_info = detect_device_info(cpu_only=config.cpu_only)
    print(json.dumps(device_info, ensure_ascii=False, indent=2))

    print("[TRAIN] Setting seeds...")
    seed_everything(config.seed)

    print(f"[TRAIN] Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)

    print("[TRAIN] Loading dataset...")
    dataset, dataset_source = resolve_dataset(config, dataset_override=dataset_override)
    print(f"[TRAIN] Dataset source: {dataset_source}")
    for split_name, split_ds in dataset.items():
        print(f"  {split_name}: {len(split_ds)}")

    print("[TRAIN] Building label summary...")
    dataset_label_summary = summarize_dataset_labels(dataset)
    print(json.dumps(dataset_label_summary, ensure_ascii=False, indent=2))

    train_label_counts = dataset_label_summary["train"]["label_counts"]

    alpha = None
    if config.manual_alpha is not None:
        alpha = torch.tensor(config.manual_alpha, dtype=torch.float32)
        print("[TRAIN] Using manual alpha:", alpha.tolist())
    elif config.use_class_balanced_alpha:
        alpha = compute_effective_number_alpha(
            label_counts=train_label_counts,
            beta=config.effective_num_beta,
            clip_max=config.alpha_clip_max,
        )
        print("[TRAIN] Using class-balanced alpha:", alpha.tolist())

    print(f"[TRAIN] Building model: {config.model_name}")
    model = build_model(config.model_name)

    if config.train_mode == "head_only":
        print("[TRAIN] Train mode = head_only")
        freeze_model_backbone(model)
    elif config.train_mode == "partial_unfreeze":
        print(f"[TRAIN] Train mode = partial_unfreeze (last {config.partial_unfreeze_last_n_layers} layers)")
        partial_unfreeze_last_layers(model, config.partial_unfreeze_last_n_layers)
    elif config.train_mode == "full":
        print("[TRAIN] Train mode = full")
    else:
        raise ValueError(f"Unsupported train_mode: {config.train_mode}")

    total_params, trainable_params = count_trainable_params(model)
    print(f"[TRAIN] Total params: {total_params:,}")
    print(f"[TRAIN] Trainable params: {trainable_params:,}")

    train_example_weights = None
    train_example_weights_summary = None
    if config.use_weighted_sampler:
        train_example_weights = build_example_sampling_weights(
            dataset["train"],
            human_weight=config.sampler_human_weight,
            ai_weight=config.sampler_ai_weight,
            bai_weight=config.sampler_bai_weight,
        )
        train_example_weights_summary = {
            "min": float(np.min(train_example_weights)),
            "mean": float(np.mean(train_example_weights)),
            "max": float(np.max(train_example_weights)),
        }
        print("[TRAIN] Weighted sampler summary:", train_example_weights_summary)

    save_training_metadata(
        output_dir=output_dir,
        config=config,
        device_info=device_info,
        dataset=dataset,
        dataset_source=dataset_source,
        dataset_label_summary=dataset_label_summary,
        alpha=alpha,
        total_params=total_params,
        trainable_params=trainable_params,
        train_example_weights_summary=train_example_weights_summary,
    )

    use_cuda = torch.cuda.is_available() and not config.cpu_only
    use_fp16 = bool(config.fp16 and use_cuda)
    use_bf16 = bool(config.bf16 and use_cuda)

    if use_cuda:
        print("[TRAIN] CUDA detected.")
    if config.gradient_checkpointing:
        print("[TRAIN] Gradient checkpointing enabled.")
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if use_cuda else None,
    )

    training_args = build_training_arguments_compat(
        config=config,
        output_dir=output_dir,
        use_cuda=use_cuda,
        use_fp16=use_fp16,
        use_bf16=use_bf16,
    )

    trainer = TokenTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=build_compute_metrics_fn(),
        alpha=alpha,
        focal_gamma=config.focal_gamma,
        use_focal_loss=config.use_focal_loss,
        train_example_weights=train_example_weights,
        **_trainer_preprocess_kw(tokenizer),
    )

    print("[TRAIN] Starting training...")
    train_result = trainer.train()

    train_metrics = train_result.metrics
    write_json(output_dir / "train_metrics.json", train_metrics)
    trainer.save_state()

    print("[TRAIN] Evaluating on validation...")
    val_metrics = trainer.evaluate(eval_dataset=dataset["validation"], metric_key_prefix="val")
    write_json(output_dir / "val_metrics.json", val_metrics)

    print("[TRAIN] Evaluating on test...")
    test_metrics = trainer.evaluate(eval_dataset=dataset["test"], metric_key_prefix="test")
    write_json(output_dir / "test_metrics.json", test_metrics)

    print("[TRAIN] Building classification reports...")
    build_token_classification_report(
        trainer=trainer,
        dataset_split=dataset["validation"],
        output_path=output_dir / "val_classification_report.json",
    )
    build_token_classification_report(
        trainer=trainer,
        dataset_split=dataset["test"],
        output_path=output_dir / "test_classification_report.json",
    )

    print("[TRAIN] Saving final model + tokenizer...")
    final_model_dir = output_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_model_dir), safe_serialization=False)
    tokenizer.save_pretrained(str(final_model_dir))

    summary = {
        "model_name": config.model_name,
        "dataset_source": dataset_source,
        "output_dir": str(output_dir),
        "final_model_dir": str(final_model_dir),
        "train_metrics_path": str(output_dir / "train_metrics.json"),
        "val_metrics_path": str(output_dir / "val_metrics.json"),
        "test_metrics_path": str(output_dir / "test_metrics.json"),
        "val_classification_report_path": str(output_dir / "val_classification_report.json"),
        "test_classification_report_path": str(output_dir / "test_classification_report.json"),
    }
    write_json(output_dir / "training_summary.json", summary)

    print("\n=== TRAINING FINISHED ===")
    print(f"Final model: {final_model_dir}")
    print(f"Training summary: {output_dir / 'training_summary.json'}")
    print("\nValidation metrics:")
    print(json.dumps(val_metrics, ensure_ascii=False, indent=2))
    print("\nTest metrics:")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение token classifier (BIO).")

    parser.add_argument("--hf-dataset-dir", type=str, default="")
    parser.add_argument("--bio-data-dir", type=str, default="../../data/token_detector_bio")

    parser.add_argument("--model-name", type=str, default="microsoft/mdeberta-v3-base")
    parser.add_argument("--output-dir", type=str, default="../../artifacts/token_detector_model_v2")

    parser.add_argument("--train-mode", type=str, default="head_only", choices=["head_only", "partial_unfreeze", "full"])
    parser.add_argument("--partial-unfreeze-last-n-layers", type=int, default=2)

    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--lr-scheduler-type", type=str, default="linear")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--group-by-length", action="store_true")

    parser.add_argument("--use-focal-loss", action="store_true")
    parser.add_argument("--focal-gamma", type=float, default=1.5)

    parser.add_argument("--use-class-balanced-alpha", action="store_true")
    parser.add_argument("--effective-num-beta", type=float, default=0.9999)
    parser.add_argument("--alpha-clip-max", type=float, default=10.0)
    parser.add_argument(
        "--manual-alpha",
        type=float,
        nargs=3,
        default=None,
        metavar=("A_O", "A_BAI", "A_IAI"),
    )

    parser.add_argument("--use-weighted-sampler", action="store_true")
    parser.add_argument("--sampler-human-weight", type=float, default=1.0)
    parser.add_argument("--sampler-ai-weight", type=float, default=1.5)
    parser.add_argument("--sampler-bai-weight", type=float, default=4.0)

    parser.add_argument("--eval-strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--save-strategy", type=str, default="no", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--load-best-model-at-end", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = TrainConfig(
        hf_dataset_dir=args.hf_dataset_dir,
        bio_data_dir=args.bio_data_dir,

        model_name=args.model_name,
        output_dir=args.output_dir,

        train_mode=args.train_mode,
        partial_unfreeze_last_n_layers=args.partial_unfreeze_last_n_layers,

        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,

        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,

        fp16=args.fp16,
        bf16=args.bf16,
        cpu_only=args.cpu_only,
        gradient_checkpointing=args.gradient_checkpointing,
        group_by_length=args.group_by_length,

        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        use_class_balanced_alpha=args.use_class_balanced_alpha,
        effective_num_beta=args.effective_num_beta,
        alpha_clip_max=args.alpha_clip_max,
        manual_alpha=args.manual_alpha,

        use_weighted_sampler=args.use_weighted_sampler,
        sampler_human_weight=args.sampler_human_weight,
        sampler_ai_weight=args.sampler_ai_weight,
        sampler_bai_weight=args.sampler_bai_weight,

        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        load_best_model_at_end=args.load_best_model_at_end,
    )

    run_training(config)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""JSONL из EDA → BIO-разметка по токенам (окна max_length/stride), summary и опционально HF DatasetDict.

С `--slim-jsonl` в JSONL не пишутся text/spans/offset_mapping. BIO в окне: первый AI-токен B-AI, далее I-AI."""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Optional

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from datasets import Dataset, DatasetDict, Features, Sequence, Value
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False
    Dataset = DatasetDict = Features = Sequence = Value = None  # type: ignore[misc, assignment]

REPO_ROOT = Path(__file__).resolve().parents[3]

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


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}, line {line_number}: {e}") from e
    return records


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def slim_bio_window_record(full: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "id",
        "group_id",
        "source_id",
        "split",
        "window_id",
        "window_index",
        "input_ids",
        "attention_mask",
        "labels",
        "window_char_start",
        "window_char_end",
        "meta",
        "ai_spans",
    )
    out = {k: full[k] for k in keys if k in full}
    for optional in ("tokens", "special_tokens_mask"):
        if optional in full:
            out[optional] = full[optional]
    return out


def append_debug_sample_if_needed(
    window_record: Dict[str, Any],
    tokenizer,
    split_tag: str,
    debug_acc: List[Dict[str, Any]],
    debug_max: int,
) -> None:
    if debug_max <= 0 or len(debug_acc) >= debug_max:
        return
    if LABEL2ID["B-AI"] not in window_record["labels"]:
        return
    input_ids = window_record["input_ids"]
    offsets = window_record["offset_mapping"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    label_names = [
        "IGNORE" if lid == IGNORE_INDEX else ID2LABEL.get(lid, str(lid))
        for lid in window_record["labels"]
    ]
    debug_acc.append({
        "window_id": window_record.get("window_id"),
        "split": split_tag,
        "doc_split": window_record.get("split"),
        "text": window_record["text"],
        "tokens": tokens,
        "labels": label_names,
        "label_ids": list(window_record["labels"]),
        "offset_mapping": offsets,
        "window_char_start": window_record.get("window_char_start"),
        "window_char_end": window_record.get("window_char_end"),
        "ai_spans": window_record.get("ai_spans"),
    })


def validate_record(record: Dict[str, Any]) -> None:
    required_top_keys = ["id", "group_id", "source_id", "text", "spans", "meta"]
    for key in required_top_keys:
        if key not in record:
            raise ValueError(f"Record missing required key: {key}")

    if not isinstance(record["text"], str) or len(record["text"]) == 0:
        raise ValueError(f"Record {record.get('id')} has empty or invalid text")

    if not isinstance(record["spans"], list) or len(record["spans"]) == 0:
        raise ValueError(f"Record {record.get('id')} has empty or invalid spans")

    text_len = len(record["text"])
    last_end = 0

    for i, span in enumerate(record["spans"]):
        for field in ["start", "end", "label"]:
            if field not in span:
                raise ValueError(f"Record {record.get('id')} span missing field: {field}")

        start = span["start"]
        end = span["end"]
        label = span["label"]

        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"Record {record.get('id')} span bounds must be int")

        if label not in ("human", "ai"):
            raise ValueError(f"Record {record.get('id')} has unsupported span label: {label}")

        if start < 0 or end <= start or end > text_len:
            raise ValueError(
                f"Record {record.get('id')} invalid span: start={start}, end={end}, text_len={text_len}"
            )

        if i == 0 and start != 0:
            raise ValueError(f"Record {record.get('id')} spans must start at 0")

        if start != last_end:
            raise ValueError(
                f"Record {record.get('id')} spans must fully cover text without gaps: "
                f"expected start={last_end}, got={start}"
            )

        last_end = end

    if last_end != text_len:
        raise ValueError(f"Record {record.get('id')} spans do not cover full text")


def extract_ai_spans(record: Dict[str, Any]) -> List[Tuple[int, int]]:
    ai_spans = []
    for span in record["spans"]:
        if span["label"] == "ai":
            ai_spans.append((span["start"], span["end"]))
    return ai_spans


def token_overlaps_span(token_start: int, token_end: int, span_start: int, span_end: int) -> bool:
    return not (token_end <= span_start or token_start >= span_end)


def build_window_labels(
    offsets: List[Tuple[int, int]],
    special_tokens_mask: List[int],
    ai_spans: List[Tuple[int, int]],
) -> List[int]:
    ai_mask: List[Optional[bool]] = []

    for (token_start, token_end), is_special in zip(offsets, special_tokens_mask):
        if is_special == 1 or token_end <= token_start:
            ai_mask.append(None)
            continue

        is_ai = any(
            token_overlaps_span(token_start, token_end, span_start, span_end)
            for span_start, span_end in ai_spans
        )
        ai_mask.append(is_ai)

    labels: List[int] = []
    prev_ai = False

    for is_ai in ai_mask:
        if is_ai is None:
            labels.append(IGNORE_INDEX)
            prev_ai = False
        elif not is_ai:
            labels.append(LABEL2ID["O"])
            prev_ai = False
        else:
            if prev_ai:
                labels.append(LABEL2ID["I-AI"])
            else:
                labels.append(LABEL2ID["B-AI"])
            prev_ai = True

    return labels


def assert_split_has_bio_tokens(split_summary: Dict[str, Any], split_name: str) -> None:
    tld = split_summary["token_label_distribution"]
    if tld.get("B-AI", 0) == 0:
        raise RuntimeError(f"No B-AI tokens found in split={split_name}")
    if tld.get("I-AI", 0) == 0:
        raise RuntimeError(f"No I-AI tokens found in split={split_name}")


def safe_window_char_bounds(offsets: List[List[int]], labels: List[int]) -> Tuple[Optional[int], Optional[int]]:
    valid_offsets = []
    for (start, end), label in zip(offsets, labels):
        if label == IGNORE_INDEX:
            continue
        valid_offsets.append((start, end))

    if not valid_offsets:
        return None, None

    return valid_offsets[0][0], valid_offsets[-1][1]


def stream_split_to_bio_jsonl(
    records: List[Dict[str, Any]],
    tokenizer,
    split_name: str,
    jsonl_path: Path,
    max_length: int,
    stride: int,
    save_tokens: bool = False,
    save_special_tokens_mask: bool = False,
    slim_jsonl: bool = False,
    debug_acc: Optional[List[Dict[str, Any]]] = None,
    debug_max: int = 0,
    debug_split_tag: str = "",
) -> Dict[str, Any]:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    label_counter: Counter = Counter()
    dataset_label_counter: Counter = Counter()
    strategy_counter: Counter = Counter()
    source_model_counter: Counter = Counter()

    docs_with_ai = 0
    docs_without_ai = 0
    windows_with_ai = 0
    windows_without_ai = 0
    windows_with_b_ai = 0
    multi_window_docs = 0
    n_windows = 0

    windows_per_doc: List[int] = []

    with jsonl_path.open("w", encoding="utf-8", buffering=1024 * 1024) as out_f:
        for record in tqdm(records, desc=f"Preparing BIO for split={split_name}"):
            validate_record(record)

            text = record["text"]
            ai_spans = extract_ai_spans(record)

            if len(ai_spans) > 0:
                docs_with_ai += 1
            else:
                docs_without_ai += 1

            dataset_label_counter[record["meta"].get("dataset_label_name", "unknown")] += 1
            strategy_counter[record["meta"].get("strategy", "unknown")] += 1
            source_model_counter[record["meta"].get("source_model", "unknown")] += 1

            encoding = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                stride=stride,
                return_offsets_mapping=True,
                return_overflowing_tokens=True,
                return_special_tokens_mask=True,
                padding=False,
            )

            n_windows_for_doc = len(encoding["input_ids"])
            windows_per_doc.append(n_windows_for_doc)

            if n_windows_for_doc > 1:
                multi_window_docs += 1

            for window_idx in range(n_windows_for_doc):
                input_ids = encoding["input_ids"][window_idx]
                attention_mask = encoding["attention_mask"][window_idx]
                offsets = encoding["offset_mapping"][window_idx]
                special_tokens_mask = encoding["special_tokens_mask"][window_idx]

                labels = build_window_labels(
                    offsets=list(offsets),
                    special_tokens_mask=list(special_tokens_mask),
                    ai_spans=ai_spans,
                )

                has_ai_in_window = False
                has_b_ai_in_window = False
                for label_id in labels:
                    if label_id == IGNORE_INDEX:
                        label_counter["IGNORE"] += 1
                    else:
                        label_counter[ID2LABEL[label_id]] += 1
                        if label_id == LABEL2ID["B-AI"]:
                            has_b_ai_in_window = True
                            has_ai_in_window = True
                        elif label_id == LABEL2ID["I-AI"]:
                            has_ai_in_window = True

                if has_ai_in_window:
                    windows_with_ai += 1
                else:
                    windows_without_ai += 1
                if has_b_ai_in_window:
                    windows_with_b_ai += 1

                offset_rows = [list(x) for x in offsets]
                window_char_start, window_char_end = safe_window_char_bounds(
                    offsets=offset_rows,
                    labels=labels,
                )

                window_record: Dict[str, Any] = {
                    "id": record["id"],
                    "group_id": record["group_id"],
                    "source_id": record["source_id"],
                    "split": split_name,
                    "window_id": f"{record['id']}__win_{window_idx}",
                    "window_index": window_idx,
                    "text": text,
                    "spans": record["spans"],
                    "ai_spans": [[s, e] for s, e in ai_spans],
                    "input_ids": list(input_ids),
                    "attention_mask": list(attention_mask),
                    "labels": labels,
                    "offset_mapping": offset_rows,
                    "window_char_start": window_char_start,
                    "window_char_end": window_char_end,
                    "meta": {
                        "dataset_label_name": record["meta"].get("dataset_label_name"),
                        "dataset_label": record["meta"].get("dataset_label"),
                        "strategy": record["meta"].get("strategy"),
                        "source_model": record["meta"].get("source_model"),
                        "judge_avg": record["meta"].get("judge_avg"),
                        "article_id": record["meta"].get("article_id"),
                        "chunk_index": record["meta"].get("chunk_index"),
                        "original_chunk_id": record["meta"].get("original_chunk_id"),
                        "char_count_actual": record["meta"].get("char_count_actual"),
                        "word_count_actual": record["meta"].get("word_count_actual"),
                        "ai_fraction_derived": record["meta"].get("ai_fraction_derived"),
                    },
                }

                if save_tokens:
                    window_record["tokens"] = tokenizer.convert_ids_to_tokens(input_ids)

                if save_special_tokens_mask:
                    window_record["special_tokens_mask"] = list(special_tokens_mask)

                if debug_acc is not None and debug_max > 0:
                    append_debug_sample_if_needed(
                        window_record,
                        tokenizer,
                        debug_split_tag or split_name,
                        debug_acc,
                        debug_max,
                    )

                to_write = slim_bio_window_record(window_record) if slim_jsonl else window_record
                out_f.write(json.dumps(to_write, ensure_ascii=False) + "\n")
                n_windows += 1

            del encoding

    split_summary = {
        "split_name": split_name,
        "n_documents": len(records),
        "n_windows": int(n_windows),
        "avg_windows_per_document": round(float(np.mean(windows_per_doc)), 6) if windows_per_doc else 0.0,
        "max_windows_per_document": int(max(windows_per_doc)) if windows_per_doc else 0,
        "multi_window_documents": int(multi_window_docs),
        "docs_with_ai": int(docs_with_ai),
        "docs_without_ai": int(docs_without_ai),
        "windows_with_ai": int(windows_with_ai),
        "windows_without_ai": int(windows_without_ai),
        "windows_with_b_ai": int(windows_with_b_ai),
        "token_label_distribution": {
            "O": int(label_counter.get("O", 0)),
            "B-AI": int(label_counter.get("B-AI", 0)),
            "I-AI": int(label_counter.get("I-AI", 0)),
            "IGNORE": int(label_counter.get("IGNORE", 0)),
        },
        "document_label_distribution": dict(dataset_label_counter),
        "strategy_distribution": dict(strategy_counter),
        "source_model_distribution": dict(source_model_counter),
        "slim_jsonl": bool(slim_jsonl),
    }

    return split_summary


def iter_bio_minimal_training_rows(jsonl_path: str) -> Iterator[Dict[str, List[int]]]:
    path = Path(jsonl_path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield {
                "input_ids": [int(x) for x in row["input_ids"]],
                "attention_mask": [int(x) for x in row["attention_mask"]],
                "labels": [int(x) for x in row["labels"]],
            }


def maybe_save_hf_dataset(
    output_dir: Path,
    train_jsonl: Path,
    val_jsonl: Path,
    test_jsonl: Path,
    enabled: bool,
) -> Optional[str]:
    if not enabled:
        return None

    if not HAS_DATASETS or Dataset is None or DatasetDict is None:
        raise ImportError(
            "Package 'datasets' is not installed. "
            "Install it or run without --save-hf-dataset."
        )

    hf_output_dir = output_dir / "hf_dataset"

    hf_features = Features(
        {
            "input_ids": Sequence(Value("int64")),
            "attention_mask": Sequence(Value("int64")),
            "labels": Sequence(Value("int64")),
        }
    )

    train_ds = Dataset.from_generator(
        iter_bio_minimal_training_rows,
        features=hf_features,
        gen_kwargs={"jsonl_path": str(train_jsonl)},
    )
    val_ds = Dataset.from_generator(
        iter_bio_minimal_training_rows,
        features=hf_features,
        gen_kwargs={"jsonl_path": str(val_jsonl)},
    )
    test_ds = Dataset.from_generator(
        iter_bio_minimal_training_rows,
        features=hf_features,
        gen_kwargs={"jsonl_path": str(test_jsonl)},
    )

    ds_dict = DatasetDict(
        {
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds,
        }
    )

    ds_dict.save_to_disk(str(hf_output_dir))
    gc.collect()
    return str(hf_output_dir)


def run_bio_preparation(
    input_dir: Path,
    output_dir: Path,
    tokenizer_name: str,
    max_length: int = 384,
    stride: int = 128,
    save_tokens: bool = False,
    save_special_tokens_mask: bool = False,
    save_hf_dataset: bool = False,
    debug_sample_count: int = 5,
    slim_jsonl: bool = False,
) -> Dict[str, Any]:
    train_path = input_dir / "train.jsonl"
    val_path = input_dir / "val.jsonl"
    test_path = input_dir / "test.jsonl"

    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[BIO] Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    if not tokenizer.is_fast:
        raise ValueError(
            f"Tokenizer '{tokenizer_name}' is not a fast tokenizer. "
            f"Fast tokenizer is required for offset_mapping."
        )

    bio_train_path = output_dir / "bio_train.jsonl"
    bio_val_path = output_dir / "bio_val.jsonl"
    bio_test_path = output_dir / "bio_test.jsonl"

    debug_acc: List[Dict[str, Any]] = []
    debug_max = max(0, debug_sample_count)

    print("[BIO] Reading train split...")
    train_records = read_jsonl(train_path)
    print("[BIO] Streaming train split to JSONL...")
    train_summary = stream_split_to_bio_jsonl(
        records=train_records,
        tokenizer=tokenizer,
        split_name="train",
        jsonl_path=bio_train_path,
        max_length=max_length,
        stride=stride,
        save_tokens=save_tokens,
        save_special_tokens_mask=save_special_tokens_mask,
        slim_jsonl=slim_jsonl,
        debug_acc=debug_acc,
        debug_max=debug_max,
    )
    assert_split_has_bio_tokens(train_summary, "train")
    del train_records
    gc.collect()

    print("[BIO] Reading val split...")
    val_records = read_jsonl(val_path)
    print("[BIO] Streaming val split to JSONL...")
    val_summary = stream_split_to_bio_jsonl(
        records=val_records,
        tokenizer=tokenizer,
        split_name="val",
        jsonl_path=bio_val_path,
        max_length=max_length,
        stride=stride,
        save_tokens=save_tokens,
        save_special_tokens_mask=save_special_tokens_mask,
        slim_jsonl=slim_jsonl,
        debug_acc=debug_acc,
        debug_max=debug_max,
    )
    assert_split_has_bio_tokens(val_summary, "val")
    del val_records
    gc.collect()

    print("[BIO] Reading test split...")
    test_records = read_jsonl(test_path)
    print("[BIO] Streaming test split to JSONL...")
    test_summary = stream_split_to_bio_jsonl(
        records=test_records,
        tokenizer=tokenizer,
        split_name="test",
        jsonl_path=bio_test_path,
        max_length=max_length,
        stride=stride,
        save_tokens=save_tokens,
        save_special_tokens_mask=save_special_tokens_mask,
        slim_jsonl=slim_jsonl,
        debug_acc=debug_acc,
        debug_max=debug_max,
    )
    assert_split_has_bio_tokens(test_summary, "test")
    del test_records
    gc.collect()

    debug_path: Optional[Path] = None
    if debug_max > 0:
        debug_path = output_dir / "bio_sample_debug.json"
        write_json(
            debug_path,
            {"n_samples": len(debug_acc), "samples": debug_acc},
        )

    hf_dataset_path = maybe_save_hf_dataset(
        output_dir=output_dir,
        train_jsonl=bio_train_path,
        val_jsonl=bio_val_path,
        test_jsonl=bio_test_path,
        enabled=save_hf_dataset,
    )

    summary = {
        "pipeline_info": {
            "name": "token_detector_bio_preparation",
            "version": "v1",
        },
        "config": {
            "tokenizer_name": tokenizer_name,
            "max_length": max_length,
            "stride": stride,
            "label2id": LABEL2ID,
            "id2label": ID2LABEL,
            "ignore_index": IGNORE_INDEX,
            "save_tokens": save_tokens,
            "save_special_tokens_mask": save_special_tokens_mask,
            "save_hf_dataset": save_hf_dataset,
            "debug_sample_count": debug_sample_count,
            "slim_jsonl": slim_jsonl,
        },
        "input_artifacts": {
            "train_jsonl": str(train_path),
            "val_jsonl": str(val_path),
            "test_jsonl": str(test_path),
        },
        "output_artifacts": {
            "bio_train_jsonl": str(output_dir / "bio_train.jsonl"),
            "bio_val_jsonl": str(output_dir / "bio_val.jsonl"),
            "bio_test_jsonl": str(output_dir / "bio_test.jsonl"),
            "bio_sample_debug_json": str(debug_path) if debug_path else None,
            "hf_dataset_dir": hf_dataset_path,
        },
        "split_summaries": {
            "train": train_summary,
            "val": val_summary,
            "test": test_summary,
        },
    }

    write_json(output_dir / "bio_dataset_summary.json", summary)

    print("\n=== BIO PREPARATION FINISHED ===")
    print(f"Output dir: {output_dir}")
    print(f"Summary:    {output_dir / 'bio_dataset_summary.json'}")
    print("\nTrain summary:")
    print(json.dumps(train_summary, ensure_ascii=False, indent=2))
    print("\nVal summary:")
    print(json.dumps(val_summary, ensure_ascii=False, indent=2))
    print("\nTest summary:")
    print(json.dumps(test_summary, ensure_ascii=False, indent=2))

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="BIO-датасет для token-level детектора.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "data" / "token_detector_eda" / "processed",
        help="Каталог с train.jsonl, val.jsonl, test.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "token_detector_bio",
        help="Каталог для bio_*.jsonl и summary",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="microsoft/mdeberta-v3-base",
        help="Имя токенизатора HuggingFace (fast)",
    )
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument(
        "--save-tokens",
        action="store_true",
        help="Сохранять в JSONL поле tokens (расширяет размер; для отладки)",
    )
    parser.add_argument(
        "--save-special-tokens-mask",
        action="store_true",
        help="Сохранять special_tokens_mask в каждом окне (для отладки)",
    )
    parser.add_argument(
        "--save-hf-dataset",
        action="store_true",
        help="Сохранить DatasetDict через datasets.save_to_disk",
    )
    parser.add_argument(
        "--debug-samples",
        type=int,
        default=5,
        help="Число примеров в bio_sample_debug.json (0 — не писать файл)",
    )
    parser.add_argument(
        "--slim-jsonl",
        action="store_true",
        help="Не писать в JSONL text/spans/offset_mapping (меньше RAM при сериализации и меньше диск; обучению достаточно)",
    )
    args = parser.parse_args()

    run_bio_preparation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer_name,
        max_length=args.max_length,
        stride=args.stride,
        save_tokens=args.save_tokens,
        save_special_tokens_mask=args.save_special_tokens_mask,
        save_hf_dataset=args.save_hf_dataset,
        debug_sample_count=args.debug_samples,
        slim_jsonl=args.slim_jsonl,
    )


if __name__ == "__main__":
    main()

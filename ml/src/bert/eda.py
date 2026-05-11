#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""EDA и подготовка JSONL для детектора на уровне токенов: аудит CSV, нормализация spans,
review/quarantine, дедупликация, train/val/test по article_id без утечки, отчёт eda_report.json.

Ожидаемые колонки CSV задаёт REQUIRED_COLUMNS. topic_flag_* — эвристика по ключевым словам, не тематический классификатор."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

REQUIRED_COLUMNS = [
    "article_id",
    "chunk_id",
    "text",
    "label",
    "ai_spans_json",
    "ai_fraction",
    "has_mixed_content",
    "strategy",
    "source_model",
    "judge_semantic_score",
    "judge_complexity_score",
    "judge_avg",
    "word_count",
    "char_count",
    "chunk_index",
    "total_chunks",
    "original_chunk_id",
    "created_at",
]

LABEL_MAP = {
    0: "human",
    1: "full_ai",
    2: "mixed",
}

REFUSAL_PATTERNS = [
    "я не могу помочь",
    "я не могу выполнить",
    "я не могу предоставить",
    "извините, но я не могу",
    "не могу помочь с этим",
    "i can't help with that",
    "i cannot help with that",
    "i can't assist with that",
    "i’m sorry, but i can’t",
    "i am sorry, but i can't",
]

TOPIC_KEYWORDS = {
    "religion": [
        "религ", "церков", "христиан", "ислам", "будд", "православ",
        "мусульман", "коран", "библи", "священ", "вера", "богослов", "бог"
    ],
    "politics": [
        "полит", "государств", "власть", "парти", "выбор", "депутат",
        "президент", "конституц", "санкц", "идеолог", "режим", "геополит"
    ],
    "philosophy": [
        "философ", "онтолог", "эпистем", "гносеолог", "метафиз",
        "экзистен", "сознани", "бытие", "этика", "аксиолог", "герменевт"
    ],
}

STRONG_PROMPT_LEAK_PATTERNS = [
    r"^\s*вот\s+переработан",
    r"^\s*вот\s+продолжени",
    r"^\s*ниже\s+приведен",
    r"^\s*конечно[,!\s]",
    r"^\s*output\s+only\b",
    r"^\s*translate\s+the\s+following\b",
    r"\byou\s+are\s+a\s+professional\s+translator\b",
    r"\bты\s+редактор\b",
    r"\bты\s+—\s+редактор\b",
    r"\bты\s+соавтор\b",
    r"\bтолько\s+текст\s+фрагмента\b",
    r"\bпродолжай\s+отсюда\b",
]

MEDIUM_PROMPT_LEAK_PATTERNS = [
    r"\btext:\b",
    r"\bfragment:\b",
    r"\bфрагмент:\b",
    r"\bтолько\s+продолжение\b",
    r"\bбез\s+пояснений\b",
]

SOFT_METADATA_FLAGS = {
    "reported_char_count_mismatch",
    "reported_word_count_mismatch",
}

SOFT_QUALITY_FLAGS = {
    "missing_judge_avg_for_synthetic",
    "ai_fraction_mismatch",
}

TOPIC_FLAG_PREFIX = "topic_flag_"


@dataclass
class ReviewPolicy:
    ignore_metadata_mismatches: bool = True
    allow_topic_flags_for_training: bool = True
    allow_ai_fraction_mismatch: bool = True
    allow_missing_judge_for_synthetic: bool = True
    relaxed_low_judge_threshold: float = 2.0
    allow_too_short_if_char_count_at_least: int = 180
    use_strict_prompt_leak_confirmation: bool = True
    deduplicate_exact_texts: bool = True
    clean_if_only_soft_flags_resolved: bool = True
    allow_remaining_soft_flags: bool = False


def ensure_dirs(base_dir: Path) -> Dict[str, Path]:
    processed_dir = base_dir / "processed"
    reports_dir = base_dir / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    return {
        "base": base_dir,
        "processed": processed_dir,
        "reports": reports_dir,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sanitize_for_json_report(obj: Any) -> Any:
    sensitive_exact = {
        "token",
        "hf_token",
        "huggingface_token",
        "access_token",
        "password",
        "secret",
        "api_key",
        "apikey",
        "authorization",
        "auth",
        "credentials",
    }

    def key_is_sensitive(k: str) -> bool:
        lower = k.lower()
        if lower in sensitive_exact:
            return True
        if lower.endswith("_token") or lower.endswith("_secret") or lower.endswith("_password"):
            return True
        if "api_key" in lower or "apikey" in lower:
            return True
        return False

    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            ks = str(k)
            if key_is_sensitive(ks):
                out[k] = "<redacted>"
            else:
                out[k] = sanitize_for_json_report(v)
        return out
    if isinstance(obj, list):
        return [sanitize_for_json_report(x) for x in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json_report(x) for x in obj]
    return obj


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    except Exception:
        return None


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return int(value)
    except Exception:
        return None


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_numeric_stats(values: List[float]) -> Dict[str, Any]:
    clean = [float(v) for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "max": None,
            "std": None,
        }

    arr = np.array(clean, dtype=float)
    return {
        "count": int(arr.size),
        "min": round(float(np.min(arr)), 6),
        "mean": round(float(np.mean(arr)), 6),
        "median": round(float(np.median(arr)), 6),
        "max": round(float(np.max(arr)), 6),
        "std": round(float(np.std(arr)), 6),
    }


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"В CSV отсутствуют обязательные колонки: {missing}")


def try_parse_json_spans(raw_value: Any) -> Tuple[Optional[List[List[int]]], Optional[str]]:
    if raw_value is None:
        return [], None
    if isinstance(raw_value, float) and math.isnan(raw_value):
        return [], None

    raw_str = str(raw_value).strip()
    if raw_str == "":
        return [], None

    try:
        parsed = json.loads(raw_str)
    except Exception as e:
        return None, f"invalid_ai_spans_json: {e}"

    if not isinstance(parsed, list):
        return None, "ai_spans_json_not_a_list"

    normalized = []
    for item in parsed:
        if not isinstance(item, list) or len(item) != 2:
            return None, "ai_spans_json_item_not_pair"
        try:
            start = int(item[0])
            end = int(item[1])
        except Exception:
            return None, "ai_spans_json_non_int_bounds"
        normalized.append([start, end])

    return normalized, None


def normalize_ai_spans(ai_spans: List[List[int]], text_len: int) -> Tuple[Optional[List[List[int]]], List[str]]:
    flags: List[str] = []

    if ai_spans is None:
        return None, ["ai_spans_is_none"]

    for start, end in ai_spans:
        if start < 0 or end < 0:
            return None, ["negative_ai_span_bound"]
        if end <= start:
            return None, ["zero_or_negative_length_ai_span"]
        if end > text_len:
            return None, ["ai_span_out_of_bounds"]

    spans_sorted = sorted(ai_spans, key=lambda x: (x[0], x[1]))
    merged: List[List[int]] = []

    for start, end in spans_sorted:
        if not merged:
            merged.append([start, end])
            continue

        last_start, last_end = merged[-1]

        if start < last_end:
            merged[-1][1] = max(last_end, end)
            flags.append("merged_overlapping_ai_spans")
        elif start == last_end:
            merged[-1][1] = end
            flags.append("merged_adjacent_ai_spans")
        else:
            merged.append([start, end])

    return merged, flags


def build_canonical_spans(text_len: int, ai_spans: List[List[int]]) -> List[Dict[str, Any]]:
    if text_len == 0:
        return []

    if not ai_spans:
        return [{"start": 0, "end": text_len, "label": "human"}]

    result: List[Dict[str, Any]] = []
    cursor = 0

    for start, end in ai_spans:
        if start > cursor:
            result.append({"start": cursor, "end": start, "label": "human"})
        result.append({"start": start, "end": end, "label": "ai"})
        cursor = end

    if cursor < text_len:
        result.append({"start": cursor, "end": text_len, "label": "human"})

    return result


def compute_ai_char_fraction(ai_spans: List[List[int]], text_len: int) -> float:
    if text_len == 0:
        return 0.0
    ai_chars = sum(end - start for start, end in ai_spans)
    return ai_chars / text_len


def detect_refusal(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in REFUSAL_PATTERNS)


def detect_topic_flags(text: str) -> List[str]:
    lower = text.lower()
    matched = []

    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(k in lower for k in keywords):
            matched.append(topic)

    return matched


def confirm_prompt_leak(text: str) -> Tuple[bool, List[str]]:
    lower = text.lower()

    strong_hits = []
    medium_hits = []

    for pattern in STRONG_PROMPT_LEAK_PATTERNS:
        if re.search(pattern, lower):
            strong_hits.append(pattern)

    for pattern in MEDIUM_PROMPT_LEAK_PATTERNS:
        if re.search(pattern, lower):
            medium_hits.append(pattern)

    if len(strong_hits) >= 1:
        return True, strong_hits + medium_hits

    if len(medium_hits) >= 2:
        return True, medium_hits

    return False, strong_hits + medium_hits


def assign_group_splits(
    records: List[Dict[str, Any]],
    val_size: float,
    test_size: float,
    seed: int
) -> Dict[str, str]:
    group_ids = sorted({str(r["group_id"]) for r in records})
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    n = len(group_ids)
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))

    if n >= 3:
        if n_test == 0:
            n_test = 1
        if n_val == 0:
            n_val = 1
        if n_test + n_val >= n:
            n_val = max(1, n_val - 1)

    test_groups = set(group_ids[:n_test])
    val_groups = set(group_ids[n_test:n_test + n_val])
    train_groups = set(group_ids[n_test + n_val:])

    split_map = {}
    for gid in train_groups:
        split_map[gid] = "train"
    for gid in val_groups:
        split_map[gid] = "val"
    for gid in test_groups:
        split_map[gid] = "test"

    return split_map


def build_record_from_row(
    row: pd.Series,
    low_judge_threshold: float
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    issues: List[str] = []
    fatal_issues: List[str] = []

    text = safe_str(row.get("text"))
    if text == "":
        fatal_issues.append("empty_text")

    text_len = len(text)
    actual_word_count = count_words(text)
    actual_char_count = len(text)

    chunk_id = safe_str(row.get("chunk_id"))
    article_id = safe_str(row.get("article_id"))
    original_chunk_id = safe_str(row.get("original_chunk_id")) or chunk_id

    label = safe_int(row.get("label"))
    if label not in (0, 1, 2):
        fatal_issues.append("invalid_label")

    reported_word_count = safe_int(row.get("word_count"))
    reported_char_count = safe_int(row.get("char_count"))
    reported_ai_fraction = safe_float(row.get("ai_fraction"))

    if reported_char_count is not None and reported_char_count != actual_char_count:
        issues.append("reported_char_count_mismatch")

    if reported_word_count is not None and abs(reported_word_count - actual_word_count) > 3:
        issues.append("reported_word_count_mismatch")

    parsed_ai_spans, parse_error = try_parse_json_spans(row.get("ai_spans_json"))
    if parse_error is not None:
        fatal_issues.append(parse_error)

    ai_spans: Optional[List[List[int]]] = None

    if label == 0:
        ai_spans = []
        if parsed_ai_spans not in ([], None):
            issues.append("label_0_but_ai_spans_present")

    elif label == 1:
        if text_len == 0:
            fatal_issues.append("label_1_with_empty_text")
        ai_spans = [[0, text_len]]
        if parsed_ai_spans not in ([], [[0, text_len]], None):
            issues.append("label_1_but_ai_spans_not_full_text")

    elif label == 2:
        if parsed_ai_spans is None:
            fatal_issues.append("label_2_but_ai_spans_unavailable")
        else:
            if len(parsed_ai_spans) == 0:
                start_char = safe_int(row.get("ai_span_start_char"))
                end_char = safe_int(row.get("ai_span_end_char"))
                if start_char is not None and end_char is not None:
                    ai_spans = [[start_char, end_char]]
                    issues.append("label_2_ai_span_restored_from_start_end_char")
                else:
                    fatal_issues.append("label_2_without_ai_spans")
            else:
                ai_spans = parsed_ai_spans

    if ai_spans is not None:
        ai_spans, span_flags = normalize_ai_spans(ai_spans, text_len)
        if ai_spans is None:
            fatal_issues.extend(span_flags)
        else:
            issues.extend(span_flags)

    if not fatal_issues and ai_spans is None:
        fatal_issues.append("ai_spans_resolution_failed")

    derived_ai_fraction = None
    if ai_spans is not None:
        derived_ai_fraction = compute_ai_char_fraction(ai_spans, text_len)
        if reported_ai_fraction is not None and abs(reported_ai_fraction - derived_ai_fraction) > 0.08:
            issues.append("ai_fraction_mismatch")

    has_mixed_content = row.get("has_mixed_content")
    has_mixed_bool = str(has_mixed_content).lower() in ("true", "1")

    if label == 2 and not has_mixed_bool:
        issues.append("label_2_but_has_mixed_content_not_true")

    if label in (0, 1) and has_mixed_bool:
        issues.append("non_mixed_label_but_has_mixed_content_true")

    if detect_refusal(text):
        fatal_issues.append("suspected_refusal_text")

    topic_flags = detect_topic_flags(text)
    for topic in topic_flags:
        issues.append(f"topic_flag_{topic}")

    prompt_leak_confirmed, prompt_patterns = confirm_prompt_leak(text)
    if prompt_leak_confirmed:
        issues.append("prompt_leak_pattern")

    strategy = safe_str(row.get("strategy"))
    source_model = safe_str(row.get("source_model"))

    judge_semantic_score = safe_float(row.get("judge_semantic_score"))
    judge_complexity_score = safe_float(row.get("judge_complexity_score"))
    judge_avg = safe_float(row.get("judge_avg"))

    is_synthetic = strategy != "original" or source_model.lower() != "human" or label in (1, 2)

    if is_synthetic and judge_avg is None:
        issues.append("missing_judge_avg_for_synthetic")

    if is_synthetic and judge_avg is not None and judge_avg < low_judge_threshold:
        issues.append("low_judge_avg")

    if actual_char_count < 50:
        issues.append("too_short_text")

    if fatal_issues:
        issue_entry = {
            "chunk_id": chunk_id,
            "article_id": article_id,
            "status": "drop",
            "fatal_issues": sorted(list(set(fatal_issues))),
            "issues": sorted(list(set(issues))),
        }
        return None, issue_entry

    canonical_spans = build_canonical_spans(text_len, ai_spans)
    ai_char_count = sum(end - start for start, end in ai_spans)

    quarantine_flags = set()

    for issue in issues:
        if issue.startswith("topic_flag_"):
            quarantine_flags.add(issue)

    soft_quarantine_markers = {
        "low_judge_avg",
        "prompt_leak_pattern",
        "missing_judge_avg_for_synthetic",
        "reported_char_count_mismatch",
        "reported_word_count_mismatch",
        "ai_fraction_mismatch",
        "too_short_text",
    }
    for issue in issues:
        if issue in soft_quarantine_markers:
            quarantine_flags.add(issue)

    status = "quarantine" if len(quarantine_flags) > 0 else "clean"

    record = {
        "id": chunk_id,
        "group_id": article_id if article_id != "" else original_chunk_id,
        "source_id": original_chunk_id,
        "text": text,
        "spans": canonical_spans,
        "meta": {
            "dataset_label": label,
            "dataset_label_name": LABEL_MAP[label],
            "strategy": strategy,
            "source_model": source_model,
            "ai_fraction_reported": reported_ai_fraction,
            "ai_fraction_derived": round(derived_ai_fraction, 6) if derived_ai_fraction is not None else None,
            "has_mixed_content": has_mixed_bool,
            "judge_semantic_score": judge_semantic_score,
            "judge_complexity_score": judge_complexity_score,
            "judge_avg": judge_avg,
            "word_count_reported": reported_word_count,
            "word_count_actual": actual_word_count,
            "char_count_reported": reported_char_count,
            "char_count_actual": actual_char_count,
            "article_id": row.get("article_id"),
            "chunk_index": safe_int(row.get("chunk_index")),
            "total_chunks": safe_int(row.get("total_chunks")),
            "original_chunk_id": original_chunk_id,
            "created_at": safe_str(row.get("created_at")),
            "ai_char_count": ai_char_count,
            "text_hash": hash_text(text),
            "prompt_patterns": prompt_patterns,
        },
        "quality": {
            "status": status,
            "flags": sorted(list(set(issues))),
        },
        "split": None,
    }

    issue_entry = {
        "chunk_id": chunk_id,
        "article_id": article_id,
        "status": status,
        "fatal_issues": [],
        "issues": sorted(list(set(issues))),
    }

    return record, issue_entry


def add_duplicate_flags(records: List[Dict[str, Any]], issues_table: List[Dict[str, Any]]) -> None:
    hash_to_ids = defaultdict(list)
    id_to_issue = {row["chunk_id"]: row for row in issues_table}

    for record in records:
        hash_to_ids[record["meta"]["text_hash"]].append(record["id"])

    duplicate_hashes = {h: ids for h, ids in hash_to_ids.items() if len(ids) > 1}

    for record in records:
        text_hash = record["meta"]["text_hash"]
        if text_hash in duplicate_hashes:
            if "exact_duplicate_text" not in record["quality"]["flags"]:
                record["quality"]["flags"].append("exact_duplicate_text")

            if record["quality"]["status"] == "clean":
                record["quality"]["status"] = "quarantine"

            issue_row = id_to_issue.get(record["id"])
            if issue_row is not None:
                if "exact_duplicate_text" not in issue_row["issues"]:
                    issue_row["issues"].append("exact_duplicate_text")
                if issue_row["status"] == "clean":
                    issue_row["status"] = "quarantine"


def choose_best_duplicate_record(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    label_priority = {
        "mixed": 3,
        "human": 2,
        "full_ai": 1,
    }

    def sort_key(r: Dict[str, Any]) -> Tuple[Any, ...]:
        status_score = 1 if r["quality"]["status"] == "clean" else 0
        judge_score = safe_float(r["meta"].get("judge_avg"))
        if judge_score is None:
            judge_score = -1.0
        label_score = label_priority.get(r["meta"].get("dataset_label_name"), 0)
        char_count = r["meta"].get("char_count_actual", 0)
        return (status_score, judge_score, label_score, char_count)

    return sorted(records, key=sort_key, reverse=True)[0]


def deduplicate_records(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_hash = defaultdict(list)
    for r in records:
        by_hash[r["meta"]["text_hash"]].append(r)

    deduped = []
    duplicate_report = []

    for text_hash, group in by_hash.items():
        if len(group) == 1:
            deduped.append(group[0])
            continue

        best = choose_best_duplicate_record(group)
        deduped.append(best)

        labels = sorted(set(r["meta"].get("dataset_label_name") for r in group))
        strategies = sorted(set(r["meta"].get("strategy") for r in group))
        ids = [r["id"] for r in group]

        duplicate_report.append({
            "text_hash": text_hash,
            "n_records": len(group),
            "kept_id": best["id"],
            "all_ids": ids,
            "labels_present": labels,
            "strategies_present": strategies,
            "has_label_conflict": len(labels) > 1,
        })

    return deduped, duplicate_report


def is_soft_flag(flag: str) -> bool:
    if flag in SOFT_METADATA_FLAGS:
        return True
    if flag in SOFT_QUALITY_FLAGS:
        return True
    if flag.startswith(TOPIC_FLAG_PREFIX):
        return True
    if flag == "too_short_text":
        return True
    if flag == "exact_duplicate_text":
        return True
    return False


def review_record(record: Dict[str, Any], policy: ReviewPolicy) -> Dict[str, Any]:
    r = copy.deepcopy(record)

    original_flags = set(r.get("quality", {}).get("flags", []))
    remaining_flags = set(original_flags)
    resolved_flags = []
    review_notes = []

    if policy.ignore_metadata_mismatches:
        for flag in list(remaining_flags):
            if flag in SOFT_METADATA_FLAGS:
                remaining_flags.remove(flag)
                resolved_flags.append(flag)

    if policy.allow_topic_flags_for_training:
        for flag in list(remaining_flags):
            if flag.startswith(TOPIC_FLAG_PREFIX):
                remaining_flags.remove(flag)
                resolved_flags.append(flag)

    if policy.allow_ai_fraction_mismatch and "ai_fraction_mismatch" in remaining_flags:
        remaining_flags.remove("ai_fraction_mismatch")
        resolved_flags.append("ai_fraction_mismatch")
        review_notes.append("use_derived_ai_fraction_instead_of_reported")

    if policy.allow_missing_judge_for_synthetic and "missing_judge_avg_for_synthetic" in remaining_flags:
        remaining_flags.remove("missing_judge_avg_for_synthetic")
        resolved_flags.append("missing_judge_avg_for_synthetic")

    if "low_judge_avg" in remaining_flags:
        judge_avg = safe_float(r["meta"].get("judge_avg"))
        if judge_avg is not None and judge_avg >= policy.relaxed_low_judge_threshold:
            remaining_flags.remove("low_judge_avg")
            resolved_flags.append("low_judge_avg")
            review_notes.append(f"judge_avg_kept_with_relaxed_threshold={policy.relaxed_low_judge_threshold}")

    if "too_short_text" in remaining_flags:
        char_count = r["meta"].get("char_count_actual", 0)
        if char_count >= policy.allow_too_short_if_char_count_at_least:
            remaining_flags.remove("too_short_text")
            resolved_flags.append("too_short_text")
            review_notes.append(f"kept_text_char_count={char_count}")

    if "prompt_leak_pattern" in remaining_flags and policy.use_strict_prompt_leak_confirmation:
        confirmed, matched_patterns = confirm_prompt_leak(r["text"])
        if not confirmed:
            remaining_flags.remove("prompt_leak_pattern")
            resolved_flags.append("prompt_leak_pattern")
            review_notes.append("prompt_leak_not_confirmed_by_strict_rules")
        else:
            review_notes.append("prompt_leak_confirmed_by_strict_rules")
            review_notes.append(f"prompt_patterns={matched_patterns}")

    if "exact_duplicate_text" in remaining_flags and policy.deduplicate_exact_texts:
        remaining_flags.remove("exact_duplicate_text")
        resolved_flags.append("exact_duplicate_text")
        review_notes.append("will_be_handled_by_deduplication")

    if len(remaining_flags) == 0:
        new_status = "clean"
    else:
        if policy.allow_remaining_soft_flags and all(is_soft_flag(f) for f in remaining_flags):
            new_status = "clean"
            review_notes.append("remaining_flags_are_soft_but_allowed")
        else:
            new_status = "quarantine"

    r["quality"]["status_original"] = r["quality"].get("status", "unknown")
    r["quality"]["status"] = new_status
    r["quality"]["flags_original"] = sorted(list(original_flags))
    r["quality"]["flags"] = sorted(list(remaining_flags))
    r["quality"]["resolved_flags"] = sorted(list(set(resolved_flags)))
    r["quality"]["review_notes"] = review_notes

    return r


def basic_dataset_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    statuses = Counter(r["quality"]["status"] for r in records)
    labels = Counter(r["meta"].get("dataset_label_name") for r in records)
    strategies = Counter(r["meta"].get("strategy") for r in records)
    source_models = Counter(r["meta"].get("source_model") for r in records)

    ai_fractions = [
        safe_float(r["meta"].get("ai_fraction_derived"))
        for r in records
        if safe_float(r["meta"].get("ai_fraction_derived")) is not None
    ]
    char_counts = [r["meta"].get("char_count_actual") for r in records if r["meta"].get("char_count_actual") is not None]
    word_counts = [r["meta"].get("word_count_actual") for r in records if r["meta"].get("word_count_actual") is not None]
    judge_avgs = [safe_float(r["meta"].get("judge_avg")) for r in records if safe_float(r["meta"].get("judge_avg")) is not None]

    total_chars = sum(r["meta"].get("char_count_actual", 0) for r in records)
    total_ai_chars = sum(r["meta"].get("ai_char_count", 0) for r in records)

    return {
        "n_records": len(records),
        "status_distribution": dict(statuses),
        "label_distribution": dict(labels),
        "strategy_distribution": dict(strategies),
        "source_model_distribution": dict(source_models),
        "char_count_stats": compute_numeric_stats(char_counts),
        "word_count_stats": compute_numeric_stats(word_counts),
        "ai_fraction_stats": compute_numeric_stats(ai_fractions),
        "judge_avg_stats": compute_numeric_stats(judge_avgs),
        "overall_ai_char_ratio": round(total_ai_chars / total_chars, 6) if total_chars > 0 else 0.0,
    }


def issue_counter(issues_table: List[Dict[str, Any]]) -> Dict[str, int]:
    cnt = Counter()
    for row in issues_table:
        for issue in row.get("issues", []):
            cnt[issue] += 1
        for issue in row.get("fatal_issues", []):
            cnt[issue] += 1
    return dict(cnt)


def build_quarantine_issue_distribution(records: List[Dict[str, Any]]) -> Dict[str, int]:
    cnt = Counter()
    for r in records:
        if r["quality"]["status"] != "quarantine":
            continue
        for flag in r["quality"].get("flags", []):
            cnt[flag] += 1
    return dict(cnt)


def build_split_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = {}
    for split in ("train", "val", "test"):
        subset = [r for r in records if r.get("split") == split]
        result[split] = {
            "rows": len(subset),
            "unique_groups": len({str(r["group_id"]) for r in subset}),
            "unique_source_id": len({str(r["source_id"]) for r in subset}),
            "label_distribution": dict(Counter(r["meta"].get("dataset_label_name") for r in subset)),
            "strategy_distribution": dict(Counter(r["meta"].get("strategy") for r in subset)),
        }
    return result


def build_split_source_family_audit(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    source_to_splits: Dict[str, set] = defaultdict(set)
    per_split: Dict[str, Dict[str, int]] = {}

    for split in ("train", "val", "test"):
        subset = [r for r in records if r.get("split") == split]
        sids = {str(r["source_id"]) for r in subset}
        gids = {str(r["group_id"]) for r in subset}
        per_split[split] = {
            "rows": len(subset),
            "unique_source_id": len(sids),
            "unique_group_id": len(gids),
        }
        for r in subset:
            source_to_splits[str(r["source_id"])].add(split)

    leaking = sorted(sid for sid, splits in source_to_splits.items() if len(splits) > 1)
    return {
        "per_split": per_split,
        "source_id_cross_split_leakage": {
            "n_source_ids_spanning_multiple_splits": len(leaking),
            "examples": leaking[:50],
        },
    }


def build_strategy_family_coverage(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    family_ids = set()
    strategy_to_families = defaultdict(set)

    for r in records:
        family_id = str(r["source_id"])
        family_ids.add(family_id)
        strategy_to_families[r["meta"]["strategy"]].add(family_id)

    total_families = max(1, len(family_ids))
    coverage = {}
    for strategy, fams in strategy_to_families.items():
        coverage[strategy] = {
            "families_covered": len(fams),
            "coverage_ratio": round(len(fams) / total_families, 6),
        }
    return coverage


def build_remaining_quarantine_preview(records: List[Dict[str, Any]], max_items: int = 20) -> List[Dict[str, Any]]:
    quarantine_records = [r for r in records if r["quality"]["status"] == "quarantine"]

    preview = []
    for r in quarantine_records[:max_items]:
        preview.append({
            "id": r["id"],
            "group_id": r["group_id"],
            "label_name": r["meta"].get("dataset_label_name"),
            "strategy": r["meta"].get("strategy"),
            "judge_avg": r["meta"].get("judge_avg"),
            "char_count": r["meta"].get("char_count_actual"),
            "remaining_flags": r["quality"].get("flags", []),
            "resolved_flags": r["quality"].get("resolved_flags", []),
            "review_notes": r["quality"].get("review_notes", []),
            "text_preview": r["text"][:500].replace("\n", " "),
        })
    return preview


def run_eda_pipeline(
    input_csv: Path,
    output_dir: Path,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
    initial_low_judge_threshold: float = 3.0,
    review_policy: Optional[ReviewPolicy] = None,
) -> Dict[str, Any]:
    if review_policy is None:
        review_policy = ReviewPolicy()

    dirs = ensure_dirs(output_dir)
    processed_dir = dirs["processed"]
    reports_dir = dirs["reports"]

    print(f"[EDA] Reading CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    validate_required_columns(df)

    raw_input_info = {
        "input_csv": str(input_csv),
        "input_rows_csv": int(len(df)),
        "unique_article_ids_input": int(df["article_id"].nunique(dropna=True)),
        "unique_chunk_ids_input": int(df["chunk_id"].nunique(dropna=True)),
        "unique_original_chunk_ids_input": int(df["original_chunk_id"].nunique(dropna=True)),
    }

    records_initial: List[Dict[str, Any]] = []
    issues_table: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        record, issue_entry = build_record_from_row(
            row=row,
            low_judge_threshold=initial_low_judge_threshold,
        )
        issues_table.append(issue_entry)
        if record is not None:
            records_initial.append(record)

    add_duplicate_flags(records_initial, issues_table)

    dropped_rows = [row for row in issues_table if row["status"] == "drop"]

    before_review_summary = basic_dataset_summary(records_initial)
    before_review_quarantine_issue_distribution = build_quarantine_issue_distribution(records_initial)

    reviewed_records = [review_record(r, review_policy) for r in records_initial]

    after_review_summary = basic_dataset_summary(reviewed_records)
    after_review_quarantine_issue_distribution = build_quarantine_issue_distribution(reviewed_records)

    clean_reviewed = [r for r in reviewed_records if r["quality"]["status"] == "clean"]
    quarantine_reviewed = [r for r in reviewed_records if r["quality"]["status"] == "quarantine"]

    duplicate_report = []
    if review_policy.deduplicate_exact_texts:
        deduped_clean, duplicate_report = deduplicate_records(clean_reviewed)
    else:
        deduped_clean = clean_reviewed

    if len(deduped_clean) == 0:
        raise RuntimeError("No clean records left after review and deduplication.")

    write_jsonl(processed_dir / "clean_records_before_split.jsonl", deduped_clean)

    split_map = assign_group_splits(
        records=deduped_clean,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )

    for r in deduped_clean:
        r["split"] = split_map[str(r["group_id"])]

    for r in quarantine_reviewed:
        r["split"] = None

    train_records = [r for r in deduped_clean if r["split"] == "train"]
    val_records = [r for r in deduped_clean if r["split"] == "val"]
    test_records = [r for r in deduped_clean if r["split"] == "test"]

    if len(train_records) == 0 or len(val_records) == 0 or len(test_records) == 0:
        raise RuntimeError(
            f"Empty split detected: train={len(train_records)}, val={len(val_records)}, test={len(test_records)}"
        )

    hard_cases_low_judge = [
        r for r in quarantine_reviewed
        if "low_judge_avg" in r["quality"].get("flags", [])
    ]

    final_clean_summary = basic_dataset_summary(deduped_clean)
    split_summary = build_split_summary(deduped_clean)
    split_source_family_audit = build_split_source_family_audit(deduped_clean)
    strategy_family_coverage = build_strategy_family_coverage(records_initial)

    duplicate_report_columns = [
        "text_hash",
        "n_records",
        "kept_id",
        "all_ids",
        "labels_present",
        "strategies_present",
        "has_label_conflict",
    ]
    duplicate_report_df = pd.DataFrame(duplicate_report, columns=duplicate_report_columns)

    write_jsonl(processed_dir / "all_records.jsonl", records_initial)
    write_jsonl(processed_dir / "reviewed_all_records.jsonl", reviewed_records)
    write_jsonl(processed_dir / "clean_records.jsonl", deduped_clean)
    write_jsonl(processed_dir / "quarantine_records.jsonl", quarantine_reviewed)
    write_jsonl(processed_dir / "hard_cases_low_judge.jsonl", hard_cases_low_judge)
    write_jsonl(processed_dir / "train.jsonl", train_records)
    write_jsonl(processed_dir / "val.jsonl", val_records)
    write_jsonl(processed_dir / "test.jsonl", test_records)

    pd.DataFrame(issues_table).to_csv(reports_dir / "issues.csv", index=False, encoding="utf-8")
    duplicate_report_df.to_csv(reports_dir / "duplicate_report.csv", index=False, encoding="utf-8")

    remaining_quarantine_rows = []
    for r in quarantine_reviewed:
        remaining_quarantine_rows.append({
            "id": r["id"],
            "group_id": r["group_id"],
            "label_name": r["meta"].get("dataset_label_name"),
            "strategy": r["meta"].get("strategy"),
            "judge_avg": r["meta"].get("judge_avg"),
            "char_count": r["meta"].get("char_count_actual"),
            "remaining_flags": ", ".join(r["quality"].get("flags", [])),
            "resolved_flags": ", ".join(r["quality"].get("resolved_flags", [])),
            "review_notes": " | ".join(r["quality"].get("review_notes", [])),
            "text_preview": r["text"][:600].replace("\n", " "),
        })
    pd.DataFrame(remaining_quarantine_rows).to_csv(
        reports_dir / "remaining_quarantine_samples.csv",
        index=False,
        encoding="utf-8"
    )

    eda_report = sanitize_for_json_report({
        "pipeline_info": {
            "name": "token_detector_eda_pipeline",
            "version": "v1",
        },
        "input_info": raw_input_info,
        "config": {
            "val_size": val_size,
            "test_size": test_size,
            "seed": seed,
            "initial_low_judge_threshold": initial_low_judge_threshold,
            "review_policy": asdict(review_policy),
        },
        "before_review": {
            "summary": before_review_summary,
            "quarantine_issue_distribution": before_review_quarantine_issue_distribution,
        },
        "after_review_before_dedup": {
            "summary": after_review_summary,
            "quarantine_issue_distribution": after_review_quarantine_issue_distribution,
        },
        "final_clean_dataset": {
            "summary": final_clean_summary,
            "split_summary": split_summary,
            "split_source_family_audit": split_source_family_audit,
            "strategy_family_coverage": strategy_family_coverage,
        },
        "quality_audit": {
            "issues_distribution_all_rows": issue_counter(issues_table),
            "dropped_rows_count": len(dropped_rows),
            "dropped_rows_preview": dropped_rows[:20],
            "remaining_quarantine_count": len(quarantine_reviewed),
            "remaining_quarantine_preview": build_remaining_quarantine_preview(quarantine_reviewed, max_items=20),
            "hard_cases_low_judge_count": len(hard_cases_low_judge),
        },
        "duplicates": {
            "duplicate_groups_removed_from_clean": int(len(duplicate_report_df)),
            "duplicate_report_preview": duplicate_report[:20],
        },
        "artifacts": {
            "all_records_jsonl": str(processed_dir / "all_records.jsonl"),
            "reviewed_all_records_jsonl": str(processed_dir / "reviewed_all_records.jsonl"),
            "clean_records_jsonl": str(processed_dir / "clean_records.jsonl"),
            "quarantine_records_jsonl": str(processed_dir / "quarantine_records.jsonl"),
            "hard_cases_low_judge_jsonl": str(processed_dir / "hard_cases_low_judge.jsonl"),
            "train_jsonl": str(processed_dir / "train.jsonl"),
            "val_jsonl": str(processed_dir / "val.jsonl"),
            "test_jsonl": str(processed_dir / "test.jsonl"),
            "issues_csv": str(reports_dir / "issues.csv"),
            "duplicate_report_csv": str(reports_dir / "duplicate_report.csv"),
            "remaining_quarantine_samples_csv": str(reports_dir / "remaining_quarantine_samples.csv"),
            "clean_records_before_split_jsonl": str(processed_dir / "clean_records_before_split.jsonl"),
        },
    })

    write_json(reports_dir / "eda_report.json", eda_report)

    print("\n=== EDA PIPELINE FINISHED ===")
    print(f"Processed dir: {processed_dir}")
    print(f"Reports dir:   {reports_dir}")
    print("\nFinal clean summary:")
    print(json.dumps(final_clean_summary, ensure_ascii=False, indent=2))
    print("\nSplit summary:")
    print(json.dumps(split_summary, ensure_ascii=False, indent=2))
    print(f"\nUnified report: {reports_dir / 'eda_report.json'}")

    return eda_report


def main() -> None:
    run_eda_pipeline(
        input_csv=REPO_ROOT / "data" / "final_dataset.csv",
        output_dir=REPO_ROOT / "data" / "token_detector_eda",
        val_size=0.1,
        test_size=0.1,
        seed=42,
        initial_low_judge_threshold=3.0,
        review_policy=ReviewPolicy(
            ignore_metadata_mismatches=True,
            allow_topic_flags_for_training=True,
            allow_ai_fraction_mismatch=True,
            allow_missing_judge_for_synthetic=True,
            relaxed_low_judge_threshold=2.0,
            allow_too_short_if_char_count_at_least=180,
            use_strict_prompt_leak_confirmation=True,
            deduplicate_exact_texts=True,
            clean_if_only_soft_flags_resolved=True,
            allow_remaining_soft_flags=False,
        ),
    )


if __name__ == "__main__":
    main()

"""Pytest-обёртка над кейсами cleaner (дублирует критичные проверки для CI)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cleaner import clean_dataframe, clean_text, clean_text_with_report  # noqa: E402
from src.cleaner_quality import analyze_text_quality, trim_document_head  # noqa: E402


def test_clean_text_with_report_has_before_after_types():
    rep = clean_text_with_report("Текст статьи " * 30)
    assert "probable_document_type_before" in rep
    assert "probable_document_type_after" in rep
    assert rep["quality_score_before"] == rep["quality_report_before"]["quality_score"]


def test_trim_head_confidence_short_doc_unchanged():
    short = "Короткий.\n"
    t2, n, c = trim_document_head(short)
    assert (t2, n, c) == (short, 0, 0.0)


def test_dataframe_quality_columns_when_pandas():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "text": ["Введение " * 40 + "\n\nAbstract\n\nThe paper.\n\nПродолжение " * 5],
            "source": ["CyberLeninka"],
            "text_source": ["html_ocr"],
            "language": ["ru"],
        }
    )
    out = clean_dataframe(
        df,
        text_col="text",
        annotation_col=None,
        source_col="source",
        text_source_col="text_source",
        language_col="language",
        with_quality_report=True,
        apply_source_profile=True,
    )
    for col in (
        "probable_document_type",
        "front_matter_density",
        "article_like_score",
        "bibliographic_like_score",
        "metadata_noise_detected",
        "meta_pass_flags_json",
    ):
        assert col in out.columns
    flags = json.loads(out.iloc[0]["meta_pass_flags_json"])
    assert isinstance(flags, list)


def test_document_type_article_like():
    t = "В данной статье " + ("рассматривается методология исследования. " * 80)
    q = analyze_text_quality(t)
    assert q["probable_document_type"] in ("article_like", "mixed_unknown")

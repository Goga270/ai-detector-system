from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from shared.schema import ARTICLE_COLUMNS, normalize_row
from shared.text_cleaner import apply_clean


def save_articles(
    df: pd.DataFrame,
    source_slug: str,
    data_dir: Path,
    ts: Optional[str] = None,
    clean: bool = False,
) -> Path:
    """CSV: {source_slug}_{YYYYMMDD_HHMMSS}.csv. При clean=True ещё *_cleaned.csv."""
    ts = ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    cols = [c for c in ARTICLE_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in ARTICLE_COLUMNS]
    df_out = df[cols + extra] if extra else df[cols]

    stem = data_dir / f"{source_slug}_{ts}"
    csv_path = stem.with_suffix(".csv")
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  {csv_path}  ({len(df_out)} строк)")

    if clean and "text" in df.columns:
        df_clean = apply_clean(df_out)
        clean_path = stem.parent / f"{stem.name}_cleaned.csv"
        df_clean.to_csv(clean_path, index=False, encoding="utf-8-sig")
        print(f"  {clean_path}")
        return clean_path
    return csv_path


def rows_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    normalized = [normalize_row(r) for r in rows]
    df = pd.DataFrame(normalized, columns=ARTICLE_COLUMNS)
    if "id" in df.columns and (df["id"].astype(str).str.strip() == "").all():
        df["id"] = range(1, len(df) + 1)
    for col in ("id", "authors_count", "text_length", "year"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
    return df

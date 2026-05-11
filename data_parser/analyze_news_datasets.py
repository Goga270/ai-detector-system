#!/usr/bin/env python3
"""Слияние и очистка новостных CSV + CyberLeninka → data/analyze/. Запуск: python analyze_news_datasets.py -h"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cyberleninka.src.cleaner import clean_dataframe, detect_artifacts  # noqa: E402
from shared.helpers import strip_html  # noqa: E402
from shared.schema import ARTICLE_COLUMNS  # noqa: E402

_RE_RBC_QUOTE = re.compile(r"rbc\.ru/quote/", re.IGNORECASE)

SOURCE_PREFIXES = ("lenta", "gazeta", "rbc", "newsapi", "mediacloud")
CYBER_CSV_PREFIX = "cyberleninka_articles"

ARTIFACT_LABELS = {
    "emails": "Email-адреса",
    "urls": "URL (http/https)",
    "udk_bbk": "Строки УДК/ББК",
    "doi": "DOI в теле",
    "access_mode": "«Режим доступа:»",
    "bibliography": "Заголовок списка литературы",
    "url_fragments": "Фрагменты URL-путей",
    "ocr_hyphens": "OCR-переносы слов",
    "shy_hyphens": "Мягкие переносы (U+00AD)",
    "ctrl_chars": "Управляющие символы",
    "email_lines": "Строки «E-mail:»",
    "bom": "BOM (U+FEFF)",
}


def normalize_url(u: object) -> str:
    if pd.isna(u):
        return ""
    return str(u).strip().lower().rstrip("/")


def find_latest_csv(data_dir: Path, prefix: str) -> Path | None:
    candidates = [
        p
        for p in data_dir.glob(f"{prefix}_*.csv")
        if "_cleaned" not in p.stem and p.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("text_length", "authors_count", "year", "id"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def align_to_article_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Старые выгрузки CyberLeninka (annotation/topic) → единая схема ARTICLE_COLUMNS."""
    df = df.copy()
    if "description" not in df.columns and "annotation" in df.columns:
        df["description"] = df["annotation"]
    if "query" not in df.columns and "topic" in df.columns:
        df["query"] = df["topic"]
    if "authors" not in df.columns and "author" in df.columns:
        df["authors"] = df["author"]
    src_empty = (
        "source" not in df.columns
        or df["source"].fillna("").astype(str).str.strip().eq("").all()
    )
    if src_empty:
        df["source"] = "CyberLeninka"
    sn_empty = (
        "source_name" not in df.columns
        or df["source_name"].fillna("").astype(str).str.strip().eq("").all()
    )
    if sn_empty:
        df["source_name"] = "CyberLeninka"
    lang_empty = (
        "language" not in df.columns
        or df["language"].fillna("").astype(str).str.strip().eq("").all()
    )
    if lang_empty:
        df["language"] = "ru"
    for col in ARTICLE_COLUMNS:
        if col not in df.columns:
            df[col] = 0 if col in ("authors_count", "text_length", "year") else ""
    return df


def is_effectively_html_only(s: object) -> bool:
    """
    Длинный фрагмент с множеством тегов, из которого strip_html извлекает мало текста
    (ошибочно попала разметка страницы вместо статьи / аннотации).
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return False
    t = str(s).strip()
    if len(t) < 500:
        return False
    if t.count("<") < 5:
        return False
    plain = strip_html(t).strip()
    if len(plain) < 150:
        return True
    if len(plain) / len(t) < 0.12:
        return True
    return False


def drop_rows_html_only_content(df: pd.DataFrame, stash) -> pd.DataFrame:
    """Убирает строки, где text или (при коротком text) description — по сути один HTML."""
    out = df
    m = pd.Series(False, index=out.index)
    if "text" in out.columns:
        m = m | out["text"].map(is_effectively_html_only)
    if "description" in out.columns:
        raw_tl = (
            out["text"].fillna("").astype(str).str.len()
            if "text" in out.columns
            else pd.Series(0, index=out.index)
        )
        m = m | (out["description"].map(is_effectively_html_only) & (raw_tl < 200))
    if not m.any():
        return out
    stash(out[m], "поле_только_html")
    return out.loc[~m].copy()


def strip_html_columns(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    """HTML → plain text (до clean_text / артефактов)."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].map(
                lambda x: strip_html(str(x)) if pd.notna(x) and str(x).strip() else ""
            )
    return out


def refresh_text_length_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "text" in out.columns:
        out["text_length"] = out["text"].fillna("").astype(str).str.len()
    return out


def flatten_multiline_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Одна логическая строка в CSV: переносы и табы в текстовых полях → пробел."""
    out = df.copy()
    for col in ("title", "description", "text", "authors", "query"):
        if col in out.columns:
            out[col] = (
                out[col]
                .fillna("")
                .astype(str)
                .str.replace(r"[\r\n]+", " ", regex=True)
                .str.replace(r"\t+", " ", regex=True)
                .str.replace(r" {2,}", " ", regex=True)
                .str.strip()
            )
    return out


def filter_for_export(
    df: pd.DataFrame,
    stash,
    *,
    keep_empty_text: bool,
    check_cyber_shell: bool,
    min_text_length: int,
) -> pd.DataFrame:
    """Котировки РБК, пустой text, слишком короткий text; для полного merge — пустые CyberLeninka."""
    out = df
    if "url" in out.columns:
        m = out["url"].fillna("").astype(str).str.contains(_RE_RBC_QUOTE, na=False)
        stash(out[m], "rbc_страница_котировок")
        out = out.loc[~m].copy()

    if not keep_empty_text and "text_length" in out.columns:
        tl = pd.to_numeric(out["text_length"], errors="coerce").fillna(0)
        m2 = tl <= 0
        stash(out[m2], "пустой_текст")
        out = out.loc[~m2].copy()

    if min_text_length > 0 and "text_length" in out.columns:
        tl = pd.to_numeric(out["text_length"], errors="coerce").fillna(0)
        m_short = tl < min_text_length
        stash(out[m_short], f"text_короче_{min_text_length}_симв")
        out = out.loc[~m_short].copy()

    if check_cyber_shell and "source" in out.columns:
        src = out["source"].fillna("").astype(str)
        is_cl = src.str.contains("CyberLeninka", case=False, na=False)
        tit = (
            out["title"].fillna("").astype(str).str.strip()
            if "title" in out.columns
            else pd.Series("", index=out.index)
        )
        txt = (
            out["text"].fillna("").astype(str).str.strip()
            if "text" in out.columns
            else pd.Series("", index=out.index)
        )
        m3 = is_cl & (tit == "") & (txt == "")
        stash(out[m3], "cyberleninka_без_заголовка_и_текста")
        out = out.loc[~m3].copy()

    return out


def dedupe_by_url_and_reindex(df: pd.DataFrame) -> pd.DataFrame:
    """Пустой URL убрать, дубликаты URL — одна строка, id с 1."""
    out = df.copy()
    if "url" not in out.columns:
        return out
    out["_norm_url"] = out["url"].map(normalize_url)
    out = out.loc[out["_norm_url"] != ""].copy()
    out = out.drop_duplicates(subset=["_norm_url"], keep="first")
    out = out.drop(columns=["_norm_url"], errors="ignore")
    out["id"] = range(1, len(out) + 1)
    return out[ARTICLE_COLUMNS]


def collect_artifact_stats(texts: pd.Series) -> tuple[dict[str, int], dict[str, int], int]:
    totals = {k: 0 for k in ARTIFACT_LABELS}
    per_doc = {k: 0 for k in ARTIFACT_LABELS}
    n = 0
    for t in texts.fillna("").astype(str):
        if not t.strip():
            continue
        n += 1
        art = detect_artifacts(t)
        for k in totals:
            v = int(art.get(k, 0))
            totals[k] += v
            if v > 0:
                per_doc[k] += 1
    return totals, per_doc, n


def print_artifact_table(
    out: StringIO,
    totals: dict[str, int],
    per_doc: dict[str, int],
    n_texts: int,
) -> None:
    out.write(f"\n{'Тип':<42} {'Вхождений':>12} {'Статей':>10} {'% статей':>10}\n")
    out.write("-" * 78 + "\n")
    for k, label in ARTIFACT_LABELS.items():
        tot = totals[k]
        docs = per_doc[k]
        pct = docs / n_texts * 100 if n_texts else 0.0
        out.write(f"  {label:<40} {tot:>12,} {docs:>10,} {pct:>9.1f}%\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Анализ и очистка 5 новостных CSV")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Папка с исходными CSV")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "analyze",
        help="Куда писать отчёт и результаты",
    )
    p.add_argument("--no-clean", action="store_true", help="Только анализ, без очистки текста")
    p.add_argument(
        "--keep-empty-text",
        action="store_true",
        help="Не удалять строки с пустым text в итоговых CSV (по умолч. удаляются)",
    )
    p.add_argument(
        "--min-text-length",
        type=int,
        default=300,
        help="Не сохранять строки с text_length ниже порога (0 = отключить)",
    )
    grp = p.add_argument_group("Опции очистки текста (как в cyberleninka/analyze.py)")
    grp.add_argument("--no-join-hyphens", action="store_true")
    grp.add_argument("--inline-refs", action="store_true")
    grp.add_argument("--keywords-line", action="store_true")
    grp.add_argument("--footnote-nums", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.min_text_length < 0:
        raise SystemExit("--min-text-length должен быть >= 0")
    data_dir = args.data_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    buf = StringIO()

    def L(s: str = "") -> None:
        buf.write(s + "\n")
        print(s)

    L("═" * 60)
    L("  Анализ новостных датасетов (5 источников)")
    L("═" * 60)
    L(f"Каталог данных: {data_dir}")
    L(f"Вывод:          {out_dir}")
    L(f"Метка времени:  {ts}")
    L()

    found: dict[str, Path] = {}
    missing = []
    for prefix in SOURCE_PREFIXES:
        p = find_latest_csv(data_dir, prefix)
        if p:
            found[prefix] = p
        else:
            missing.append(prefix)

    if missing:
        L(f"[!] Не найдены CSV для: {', '.join(missing)}")
        L(f"    Ожидаются файлы вида {data_dir}/<источник>_YYYYMMDD_HHMMSS.csv")
        sys.exit(1)

    L("Входные файлы (последние по дате изменения):")
    frames: list[pd.DataFrame] = []
    for prefix in SOURCE_PREFIXES:
        path = found[prefix]
        df = coerce_types(load_csv(path))
        df["_input_file"] = path.name
        frames.append(df)
        L(f"  {prefix:<12} {path.name}  ({len(df)} строк)")

    merged = pd.concat(frames, ignore_index=True)
    n0 = len(merged)
    L(f"\nОбъединено строк (до фильтрации): {n0}")

    merged["_norm_url"] = merged["url"].map(normalize_url) if "url" in merged.columns else ""

    removed_parts: list[pd.DataFrame] = []

    def stash(sub: pd.DataFrame, reason: str) -> None:
        if len(sub) == 0:
            return
        x = sub.copy()
        x["_remove_reason"] = reason
        removed_parts.append(x)

    df = merged

    if "url" not in df.columns:
        L("[!] Нет колонки url — фильтрация невозможна")
        sys.exit(1)

    m_empty_url = df["_norm_url"] == ""
    stash(df[m_empty_url], "пустой_url")
    df = df.loc[~m_empty_url].copy()

    m_dup = df.duplicated(subset=["_norm_url"], keep="first")
    stash(df[m_dup], "дубликат_url")
    df = df.loc[~m_dup].copy()

    if "title" in df.columns:
        m_no_title = df["title"].fillna("").astype(str).str.strip() == ""
        stash(df[m_no_title], "пустой_заголовок")
        df = df.loc[~m_no_title].copy()
    else:
        L("[!] Нет колонки title — пропуск фильтра по заголовку")

    L("\n─── Отсев строк: text/description по сути один HTML ───")
    n_h = len(df)
    df = drop_rows_html_only_content(df, stash)
    L(f"  Снято: {n_h - len(df)}")

    L("\n─── Снятие HTML из text/description (до очистки cleaner) ───")
    df = strip_html_columns(df, ("text", "description"))
    df = refresh_text_length_column(df)
    L(f"  Обработано строк: {len(df)}")

    if removed_parts:
        removed_df = pd.concat(removed_parts, ignore_index=True)
    else:
        removed_df = pd.DataFrame()

    L("\n─── Удаление записей ───")
    if len(removed_df):
        for reason, cnt in removed_df["_remove_reason"].value_counts().items():
            L(f"  {reason:<22} {int(cnt)}")
    L(f"  Всего снято строк:    {len(removed_df)}")
    L(f"  Осталось строк:       {len(df)}")

    L("\n─── Статистика по источникам (после отбора) ───")
    if "source" in df.columns and len(df):
        vc = df.groupby("source").size().rename("строк")
        L(vc.to_string())
    elif "source_name" in df.columns and len(df):
        L(df.groupby("source_name").size().rename("строк").to_string())

    L("\n─── text_source и покрытие текстом ───")
    if "text_source" in df.columns and len(df):
        total = len(df)
        for src, cnt in df["text_source"].fillna("").value_counts().items():
            L(f"  {repr(src):<20} {cnt:>6} ({cnt/total*100:.1f}%)")
    if "text_length" in df.columns and len(df):
        has_t = pd.to_numeric(df["text_length"], errors="coerce").fillna(0) > 0
        L(f"  С текстом (length>0): {has_t.sum()} ({has_t.mean()*100:.1f}%)")

    L("\n─── Год публикации (year) ───")
    if "year" in df.columns and len(df):
        y = pd.to_numeric(df["year"], errors="coerce")
        L(f"  year==0 (неизвестно): {(y.fillna(0) == 0).sum()}")
        yz = y[y > 0]
        if len(yz):
            L(f"  Диапазон (year>0):    {int(yz.min())} – {int(yz.max())}")

    L("\n─── Артефакты в text (до очистки) ───")
    if "text" in df.columns and len(df):
        tl = pd.to_numeric(df["text_length"], errors="coerce").fillna(0) if "text_length" in df.columns else pd.Series(0, index=df.index)
        texts = df.loc[tl > 0, "text"] if (tl > 0).any() else df["text"]
        totals, per_doc, n_txt = collect_artifact_stats(texts)
        tbuf = StringIO()
        print_artifact_table(tbuf, totals, per_doc, n_txt)
        tbl = tbuf.getvalue()
        buf.write(tbl)
        print(tbl, end="")
    else:
        L("  Нет колонки text или датасет пуст.")

    clean_opts = {
        "join_ocr_hyphens": not args.no_join_hyphens,
        "remove_inline_refs": args.inline_refs,
        "remove_keywords_line": args.keywords_line,
        "remove_footnote_numbers": args.footnote_nums,
    }

    if args.no_clean:
        L("\n(--no-clean) Очистка текста пропущена.")
        final = df.drop(columns=["_norm_url", "_input_file"], errors="ignore")
        final["id"] = range(1, len(final) + 1)
    else:
        L("\n─── Очистка текста (clean_dataframe) ───")
        L(f"  Опции: {clean_opts}")
        desc_col = "description" if "description" in df.columns else None
        df_work = df.drop(columns=["_norm_url"], errors="ignore")
        df_clean = clean_dataframe(
            df_work,
            text_col="text",
            annotation_col=desc_col,
            **clean_opts,
        )

        has_text = pd.to_numeric(df_work.get("text_length", 0), errors="coerce").fillna(0) > 0
        if has_text.any() and "text_length_clean" in df_clean.columns:
            olen = pd.to_numeric(df_work.loc[has_text, "text_length"], errors="coerce").fillna(0).sum()
            clen = pd.to_numeric(df_clean.loc[has_text, "text_length_clean"], errors="coerce").fillna(0).sum()
            removed_chars = int(olen - clen)
            pct = removed_chars / olen * 100 if olen else 0.0
            L(f"  Символов до очистки (где был текст): {int(olen):,}")
            L(f"  Символов после:                     {int(clen):,}")
            L(f"  Удалено символов:                   {removed_chars:,} ({pct:.1f}%)")

        df_clean["text"] = df_clean["text_clean"]
        df_clean["text_length"] = df_clean["text_length_clean"]
        if desc_col and f"{desc_col}_clean" in df_clean.columns:
            df_clean["description"] = df_clean[f"{desc_col}_clean"]

        drop_extra = [
            c
            for c in df_clean.columns
            if c.endswith("_clean") or c == "text_length_clean"
        ]
        final = df_clean.drop(columns=drop_extra, errors="ignore")
        final = final.drop(columns=["_input_file"], errors="ignore")
        final["id"] = range(1, len(final) + 1)

    for col in ARTICLE_COLUMNS:
        if col not in final.columns:
            final[col] = 0 if col in ("authors_count", "text_length", "year") else ""
    final = final[ARTICLE_COLUMNS]

    n_stash_before_export = len(removed_parts)
    L("\n─── Отбор перед сохранением: новости ───")
    final = filter_for_export(
        final,
        stash,
        keep_empty_text=args.keep_empty_text,
        check_cyber_shell=False,
        min_text_length=args.min_text_length,
    )
    final = flatten_multiline_fields(final)
    final["id"] = range(1, len(final) + 1)
    if len(removed_parts) > n_stash_before_export:
        exp = pd.concat(removed_parts[n_stash_before_export:], ignore_index=True)
        for reason, cnt in exp["_remove_reason"].value_counts().items():
            L(f"  {reason:<35} {int(cnt)}")

    # ── Объединение со всеми доступными CSV (новости уже в final + CyberLeninka) ──
    L("\n─── Полный merge: новости + CyberLeninka ───")
    merge_parts: list[pd.DataFrame] = [final.copy()]
    cyber_path = find_latest_csv(data_dir, CYBER_CSV_PREFIX)
    if cyber_path:
        c_raw = load_csv(cyber_path)
        L(f"  CyberLeninka: {cyber_path.name}  ({len(c_raw)} строк в файле)")
        c_df = align_to_article_columns(coerce_types(c_raw))
        n_ch = len(c_df)
        c_df = drop_rows_html_only_content(c_df, stash)
        L(f"  Отсев HTML-оболочек (CyberLeninka): {n_ch - len(c_df)} строк")
        c_df = strip_html_columns(c_df, ("text", "description"))
        c_df = refresh_text_length_column(c_df)
        if args.no_clean:
            c_final = c_df[ARTICLE_COLUMNS]
        else:
            desc_c = "description" if "description" in c_df.columns else None
            c_clean = clean_dataframe(
                c_df,
                text_col="text",
                annotation_col=desc_c,
                **clean_opts,
            )
            c_clean["text"] = c_clean["text_clean"]
            c_clean["text_length"] = c_clean["text_length_clean"]
            if desc_c and f"{desc_c}_clean" in c_clean.columns:
                c_clean["description"] = c_clean[f"{desc_c}_clean"]
            drop_c = [
                x
                for x in c_clean.columns
                if x.endswith("_clean") or x == "text_length_clean"
            ]
            c_final = c_clean.drop(columns=drop_c, errors="ignore")
            for col in ARTICLE_COLUMNS:
                if col not in c_final.columns:
                    c_final[col] = 0 if col in ("authors_count", "text_length", "year") else ""
            c_final = c_final[ARTICLE_COLUMNS]
        merge_parts.append(c_final)
    else:
        L(
            "  Файл cyberleninka_articles_*.csv в data/ не найден — "
            "в all_sources только новости."
        )

    all_merged = pd.concat(merge_parts, ignore_index=True)
    n_before_dedup = len(all_merged)
    all_merged = dedupe_by_url_and_reindex(all_merged)
    n_dedup = n_before_dedup - len(all_merged)
    if n_dedup:
        L(f"  Снято дубликатов URL между источниками: {n_dedup}")
    if len(all_merged) and "source" in all_merged.columns:
        L("  Итог по полю source:")
        L(all_merged.groupby("source").size().rename("строк").to_string())
    L(f"  Всего строк в полном merge: {len(all_merged)}")

    n_stash2 = len(removed_parts)
    L("\n─── Отбор перед сохранением: полный датасет ───")
    all_merged = filter_for_export(
        all_merged,
        stash,
        keep_empty_text=args.keep_empty_text,
        check_cyber_shell=True,
        min_text_length=args.min_text_length,
    )
    all_merged = flatten_multiline_fields(all_merged)
    all_merged["id"] = range(1, len(all_merged) + 1)
    if len(removed_parts) > n_stash2:
        exp2 = pd.concat(removed_parts[n_stash2:], ignore_index=True)
        for reason, cnt in exp2["_remove_reason"].value_counts().items():
            L(f"  {reason:<35} {int(cnt)}")
    L(f"  Строк в all_sources после отбора: {len(all_merged)}")

    report_path = out_dir / f"news_analysis_{ts}.txt"
    merged_path = out_dir / f"news_merged_cleaned_{ts}.csv"
    all_merged_path = out_dir / f"all_sources_merged_{ts}.csv"
    removed_path = out_dir / f"news_removed_rows_{ts}.csv"

    final.to_csv(merged_path, index=False, encoding="utf-8-sig")
    all_merged.to_csv(all_merged_path, index=False, encoding="utf-8-sig")
    L(f"\n─── Сохранение ───")
    L(f"  Отчёт:     {report_path}")
    L(f"  Новости:   {merged_path}  ({len(final)} строк)")
    L(f"  Все CSV:   {all_merged_path}  ({len(all_merged)} строк)")
    if removed_parts:
        removed_full = pd.concat(removed_parts, ignore_index=True)
        removed_out = removed_full.drop(columns=["_norm_url"], errors="ignore")
        removed_out = flatten_multiline_fields(removed_out)
        removed_out.to_csv(removed_path, index=False, encoding="utf-8-sig")
        L(f"  Удалённые: {removed_path}  ({len(removed_out)} строк)")
    else:
        L("  Удалённые строки: нет отдельного файла (ничего не снято)")

    L("\nГотово.")
    report_path.write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()

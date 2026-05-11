"""
Скрипт анализа и очистки собранного датасета CyberLeninka.

Запуск:
    python analyze.py                          # читает cyberleninka_articles.csv
    python analyze.py --input my_data.csv      # другой файл
    python analyze.py --no-clean               # только анализ, без очистки

Что делает скрипт:
    1. Полная статистика по DataFrame (размер, распределение тем, годы и т.д.).
    2. Анализ поля text_source — откуда получен текст, покрытие.
    3. Диагностика артефактов в текстах (email, URL, УДК, мягкие переносы и т.д.).
    4. Примеры «грязных» мест — показывает конкретные находки.
    5. Очистка текста через src.cleaner.clean_dataframe().
    6. Сравнение до/после очистки (длина текста, процент удалённых символов).
    7. Сохранение очищенного датасета в *_cleaned.csv.
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src import config
from src.cleaner import clean_dataframe, detect_artifacts


# ───────────────────────────────────────────────────────────────
# Вывод секций
# ───────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print("═" * 60)


def subsection(title: str) -> None:
    print(f"\n─── {title} ───")


# ───────────────────────────────────────────────────────────────
# 1. Полная статистика DataFrame
# ───────────────────────────────────────────────────────────────

def print_general_stats(df: pd.DataFrame) -> None:
    section("ОБЩАЯ СТАТИСТИКА ДАТАСЕТА")

    print(f"Строк:    {len(df)}")
    print(f"Колонок:  {len(df.columns)}")
    print(f"Колонки:  {list(df.columns)}")

    subsection("Типы данных")
    print(df.dtypes.to_string())

    subsection("Пропущенные значения")
    na = df.isna().sum()
    na = na[na > 0]
    if len(na):
        print(na.to_string())
    else:
        print("  Нет пропущенных значений.")

    subsection("Дубликаты (по article_id)")
    if "article_id" in df.columns:
        dupes = df["article_id"].duplicated().sum()
        print(f"  Дублирующихся строк: {dupes}")

    subsection("Распределение по темам (topic)")
    if "topic" in df.columns:
        print(df.groupby("topic", dropna=False)["id"].count().rename("статей").to_string())

    subsection("Год публикации")
    if "year" in df.columns:
        y = df["year"].dropna()
        print(f"  Диапазон: {int(y.min())} – {int(y.max())}")
        print(f"  Среднее:  {y.mean():.1f}")
        print(f"  Медиана:  {y.median():.0f}")
        print("\n  Топ-10 лет по числу статей:")
        print(
            df.groupby("year", dropna=False)["id"]
            .count()
            .sort_values(ascending=False)
            .head(10)
            .rename("статей")
            .to_string()
        )

    subsection("Авторы")
    if "authors_count" in df.columns:
        print(f"  Среднее авторов/статья: {df['authors_count'].mean():.2f}")
        print(f"  Макс. авторов:          {df['authors_count'].max()}")
        print(f"  Без авторов:            {(df['authors_count'] == 0).sum()}")

    subsection("Ключевые слова")
    if "keywords_count" in df.columns:
        print(f"  Среднее ключ. слов:  {df['keywords_count'].mean():.2f}")
        print(f"  Без ключ. слов:      {(df['keywords_count'] == 0).sum()}")

    subsection("Индексация")
    for col, label in [("rsci", "РИНЦ"), ("vak", "ВАК"), ("scopus", "Scopus")]:
        if col in df.columns:
            n = df[col].sum()
            pct = n / len(df) * 100
            print(f"  {label}: {n} ({pct:.1f}%)")

    subsection("Журналы")
    if "journal" in df.columns:
        journals = df["journal"].dropna().replace("", pd.NA).dropna()
        print(f"  Уникальных журналов:    {journals.nunique()}")
        print(f"  Без журнала:            {df['journal'].replace('', pd.NA).isna().sum()}")
        print("\n  Топ-10 журналов:")
        print(journals.value_counts().head(10).to_string())


# ───────────────────────────────────────────────────────────────
# 2. Анализ text_source
# ───────────────────────────────────────────────────────────────

def print_text_source_stats(df: pd.DataFrame) -> None:
    section("АНАЛИЗ text_source — ПОКРЫТИЕ ТЕКСТОМ")

    if "text_source" not in df.columns:
        print("  Колонка text_source отсутствует.")
        return

    total = len(df)
    counts = df["text_source"].fillna("").value_counts(dropna=False)
    print("Распределение источников текста:")
    for src, cnt in counts.items():
        label = {
            "html_ocr": "HTML OCR (со страницы статьи)",
            "pdf":       "PDF (скачан и извлечён pdfminer)",
            "":          "Текст НЕ получен",
        }.get(str(src), str(src))
        print(f"  {label:<40} {cnt:>5} ({cnt/total*100:.1f}%)")

    subsection("Длина текста по источнику")
    if "text_length" in df.columns:
        grp = df.groupby("text_source", dropna=False)["text_length"]
        print(grp.describe()[["count", "mean", "min", "max"]].to_string())

    subsection("Покрытие текстом по темам")
    if "topic" in df.columns and "text_length" in df.columns:
        has_text = df["text_length"] > 0
        cov = (
            df.groupby("topic")
            .apply(lambda g: (g["text_length"] > 0).mean() * 100)
            .rename("покрытие_%")
            .round(1)
        )
        print(cov.to_string())

    subsection("Аннотации")
    if "annotation" in df.columns:
        has_ann = df["annotation"].fillna("").str.len() > 0
        print(f"  Статей с аннотацией:  {has_ann.sum()} ({has_ann.mean()*100:.1f}%)")
        print(f"  Ср. длина аннотации:  {df.loc[has_ann, 'annotation'].str.len().mean():.0f} симв.")


# ───────────────────────────────────────────────────────────────
# 3. Диагностика артефактов
# ───────────────────────────────────────────────────────────────

def print_artifact_stats(df: pd.DataFrame) -> None:
    section("ДИАГНОСТИКА АРТЕФАКТОВ В ТЕКСТАХ")

    texts_with_content = df[df["text_length"] > 0].copy() if "text_length" in df.columns else df.copy()
    n = len(texts_with_content)

    if n == 0:
        print("  Нет статей с текстом.")
        return

    print(f"Анализируем {n} статей с текстом …\n")

    # Подсчёт по каждому виду артефактов
    artifact_totals: dict = {k: 0 for k in [
        "emails", "urls", "udk_bbk", "doi",
        "access_mode", "bibliography", "url_fragments",
        "ocr_hyphens", "shy_hyphens", "ctrl_chars", "email_lines", "bom",
    ]}
    artifact_docs: dict = {k: 0 for k in artifact_totals}

    for text in texts_with_content["text"].fillna(""):
        art = detect_artifacts(str(text))
        for k, v in art.items():
            if k in artifact_totals:
                artifact_totals[k] += v
                if v > 0:
                    artifact_docs[k] += 1

    labels = {
        "emails":        "Email-адреса (обфусцированные + обычные)",
        "urls":          "URL-ссылки (http/https)",
        "udk_bbk":       "Строки УДК/ББК/DOI",
        "doi":           "DOI-идентификаторы в теле текста",
        "access_mode":   "«Режим доступа:» с URL",
        "bibliography":  "Раздел «Список литературы»",
        "url_fragments": "Фрагменты URL-путей (хвосты после http://)",
        "ocr_hyphens":   "OCR-переносы (слово-\\nпродолжение)",
        "shy_hyphens":   "Мягкие переносы (U+00AD) от PDF",
        "ctrl_chars":    "Управляющие символы",
        "email_lines":   "Строки «E-mail:»",
        "bom":           "BOM-символы (U+FEFF)",
    }

    print(f"{'Тип артефакта':<50} {'Вхождений':>10} {'Статей':>8} {'% статей':>9}")
    print("-" * 80)
    for k, label in labels.items():
        tot = artifact_totals[k]
        docs = artifact_docs[k]
        pct = docs / n * 100 if n else 0
        print(f"  {label:<48} {tot:>10,} {docs:>8,} {pct:>8.1f}%")


# ───────────────────────────────────────────────────────────────
# 4. Примеры артефактов
# ───────────────────────────────────────────────────────────────

def print_artifact_examples(df: pd.DataFrame, n_examples: int = 5) -> None:
    section("ПРИМЕРЫ АРТЕФАКТОВ (до очистки)")

    texts = df[df["text_length"] > 0]["text"].fillna("").tolist() if "text_length" in df.columns else df["text"].fillna("").tolist()

    # Email-адреса (включая с пробелом вокруг @)
    subsection("Email-адреса")
    re_email = re.compile(r"[a-zA-Z0-9_.+-]+\s*@\s*[a-zA-Z0-9-]+(?:\s*\.\s*[a-zA-Z]{2,})+", re.IGNORECASE)
    found, shown = [], 0
    for t in texts:
        found.extend(re_email.findall(t))
    for ex in sorted(set(found))[:n_examples]:
        print(f"  {repr(ex)}")
        shown += 1
    if not shown:
        print("  Не найдено.")

    # Строки E-mail:
    subsection("Строки «E-mail:» (с аффилиацией)")
    re_email_line = re.compile(r"E[\s-]?mail\s*:\s*[^\n]*", re.IGNORECASE)
    shown = 0
    for t in texts:
        for m in re_email_line.finditer(t):
            snippet = m.group(0)[:120]
            print(f"  {repr(snippet)}")
            shown += 1
            if shown >= n_examples:
                break
        if shown >= n_examples:
            break
    if not shown:
        print("  Не найдено.")

    # URL + «Режим доступа»
    subsection("Примеры URL / «Режим доступа»")
    re_url = re.compile(r"Режим доступа\s*:\s*https?://[^\s\)\]\"\'<>]{0,80}", re.IGNORECASE)
    shown = 0
    for t in texts:
        for m in re_url.finditer(t):
            print(f"  {repr(m.group(0)[:120])}")
            shown += 1
            if shown >= n_examples:
                break
        if shown >= n_examples:
            break
    if not shown:
        print("  Не найдено.")

    # УДК/ББК строки
    subsection("Строки УДК/ББК в тексте")
    re_udk = re.compile(r"^[ \t]*(УДК|ББК|UDC|BBK|DOI)\b[^\n]*", re.MULTILINE | re.IGNORECASE)
    shown = 0
    for t in texts:
        for m in re_udk.finditer(t):
            print(f"  {repr(m.group(0)[:120])}")
            shown += 1
            if shown >= n_examples:
                break
        if shown >= n_examples:
            break
    if not shown:
        print("  Не найдено.")

    # BOM
    subsection("BOM-символы")
    bom_count = sum(t.count("\ufeff") for t in texts)
    print(f"  Всего BOM: {bom_count}")
    if bom_count:
        for t in texts:
            if "\ufeff" in t:
                idx = t.index("\ufeff")
                print(f"  Контекст: {repr(t[max(0,idx-10):idx+30])}")
                break


# ───────────────────────────────────────────────────────────────
# 5–6. Очистка и сравнение
# ───────────────────────────────────────────────────────────────

def run_cleaning(df: pd.DataFrame, clean_opts: dict | None = None) -> pd.DataFrame:
    section("ОЧИСТКА ТЕКСТА")
    opts = clean_opts or {}
    opts_str = ", ".join(f"{k}={v}" for k, v in opts.items()) if opts else "по умолчанию"
    print(f"Применяем clean_dataframe() [{opts_str}] …")

    df_clean = clean_dataframe(df, text_col="text", annotation_col="annotation", **opts)

    # Сравнение длин
    if "text_length" in df.columns and "text_length_clean" in df_clean.columns:
        has_text = df["text_length"] > 0
        orig_len = df.loc[has_text, "text_length"].sum()
        clean_len = df_clean.loc[has_text, "text_length_clean"].sum()
        removed = orig_len - clean_len
        pct = removed / orig_len * 100 if orig_len else 0

        subsection("Результат очистки")
        print(f"  Статей с текстом:       {has_text.sum()}")
        print(f"  Символов до очистки:    {orig_len:,}")
        print(f"  Символов после:         {clean_len:,}")
        print(f"  Удалено символов:       {removed:,} ({pct:.1f}%)")
        print(f"  Ср. длина ДО:           {df.loc[has_text, 'text_length'].mean():.0f} симв.")
        print(f"  Ср. длина ПОСЛЕ:        {df_clean.loc[has_text, 'text_length_clean'].mean():.0f} симв.")

        # Анализ артефактов после очистки
        subsection("Остаточные артефакты (после очистки)")
        texts_after = df_clean.loc[has_text, "text_clean"].fillna("")
        after_totals: dict = {k: 0 for k in [
            "emails", "urls", "udk_bbk", "doi", "access_mode", "bom"
        ]}
        for t in texts_after:
            art = detect_artifacts(str(t))
            for k in after_totals:
                after_totals[k] += art[k]
        for k, v in after_totals.items():
            print(f"  {k:<20} {v:>8,}")

    subsection("Пример: было → стало")
    col_before = "text"
    col_after  = "text_clean"
    if col_before in df.columns and col_after in df_clean.columns:
        mask = (df[col_before].fillna("").str.len() > 500)
        if mask.any():
            idx = df[mask].index[0]
            before = str(df.at[idx, col_before])[:400]
            after  = str(df_clean.at[idx, col_after])[:400]
            print(f"\n  [ДО]  {repr(before)}")
            print(f"\n  [ПОСЛЕ] {repr(after)}")

    return df_clean


# ───────────────────────────────────────────────────────────────
# Точка входа
# ───────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Анализ датасета CyberLeninka",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Базовый анализ + очистка
  python3 analyze.py --input data/cyberleninka_articles_20260307_124155.csv

  # Только статистика, без очистки
  python3 analyze.py --input data/... --no-clean

  # Включить удаление ссылок (1), [2] внутри текста
  python3 analyze.py --input data/... --inline-refs

  # Включить удаление строк «Ключевые слова:» из текста
  python3 analyze.py --input data/... --keywords-line

  # Включить нумерацию сносок (1., [1]) в начале строк
  python3 analyze.py --input data/... --footnote-nums

  # Отключить склейку OCR-переносов (включена по умолчанию)
  python3 analyze.py --input data/... --no-join-hyphens

  # Все агрессивные опции вместе
  python3 analyze.py --input data/... --inline-refs --keywords-line --footnote-nums
""",
    )
    p.add_argument(
        "--input", type=str, default="cyberleninka_articles.csv",
        help="Входной CSV (по умолч.: cyberleninka_articles.csv)",
    )
    p.add_argument(
        "--no-clean", action="store_true",
        help="Не запускать очистку — только статистика",
    )
    p.add_argument(
        "--examples", type=int, default=5,
        metavar="N",
        help="Кол-во примеров артефактов для вывода (по умолч.: 5)",
    )

    grp = p.add_argument_group("Опции очистки текста")
    grp.add_argument(
        "--no-join-hyphens", action="store_true",
        help="Не склеивать слова, разорванные OCR-переносом «слово-\\nпродолжение» (по умолч. склейка ВКЛ)",
    )
    grp.add_argument(
        "--inline-refs", action="store_true",
        help="Удалять внутритекстовые ссылки-номера (1), [2] "
             "(ОСТОРОЖНО: может удалить годы вида (2016))",
    )
    grp.add_argument(
        "--keywords-line", action="store_true",
        help="Удалять строку «Ключевые слова: …» из тела текста "
             "(ключевые слова уже есть в отдельной колонке)",
    )
    grp.add_argument(
        "--footnote-nums", action="store_true",
        help="Удалять номера сносок «1.», «[1]» в начале строк",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"[!] Файл не найден: {csv_path}")
        sys.exit(1)

    print(f"Загружаем {csv_path} …")
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)

    # Привести типы если нужно
    for col in ("text_length", "authors_count", "keywords_count"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for col in ("rsci", "vak", "scopus"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            ).fillna(False)

    # --- Разделы анализа ---
    print_general_stats(df)
    print_text_source_stats(df)
    print_artifact_stats(df)
    print_artifact_examples(df, n_examples=args.examples)

    if not args.no_clean:
        clean_opts = {
            "join_ocr_hyphens":        not args.no_join_hyphens,
            "remove_inline_refs":      args.inline_refs,
            "remove_keywords_line":    args.keywords_line,
            "remove_footnote_numbers": args.footnote_nums,
        }
        df_clean = run_cleaning(df, clean_opts=clean_opts)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Если входной файл уже лежит в data/, кладём рядом;
        # иначе — в config.DATA_DIR
        if csv_path.parent.name == config.DATA_DIR.name:
            out_path = csv_path.parent / f"{csv_path.stem}_cleaned{csv_path.suffix}"
        else:
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            out_path = config.DATA_DIR / f"{config.FILE_BASENAME}_{ts}_cleaned.csv"

        df_clean.to_csv(out_path, index=False, encoding="utf-8-sig")
        section("СОХРАНЕНИЕ")
        print(f"  Очищенный датасет → {out_path}")

    print("\nАнализ завершён.")


if __name__ == "__main__":
    main()

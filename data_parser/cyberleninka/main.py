"""Парсер CyberLeninka: выгрузка в data/, опционально --clean / --clean-only. См. python main.py -h."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from shared.schema import ARTICLE_COLUMNS

from src import config
from src.cleaner import clean_dataframe
from src.parser import fetch_all_topics


def _make_paths(ts: str) -> tuple:
    base = config.DATA_DIR / f"{config.FILE_BASENAME}_{ts}"
    return (
        base.with_suffix(".csv"),
        base.with_suffix(".json"),
        Path(str(base) + "_cleaned.csv"),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Парсер статей CyberLeninka",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--pages", type=int, default=config.MAX_PAGES,
        metavar="N",
        help=f"Страниц API на тему (10 статей/стр.). По умолч.: {config.MAX_PAGES}",
    )
    p.add_argument(
        "--topics", type=str, default=None,
        metavar="ТЕМЫ",
        help="Темы через запятую. По умолч.: 5 предустановленных тем.",
    )
    p.add_argument(
        "--filter", type=str, default="none",
        choices=["none", "rsci", "vak", "scopus"],
        dest="catalog_filter",
        help="Фильтр по базе индексации. По умолч.: none",
    )
    p.add_argument(
        "--delay", type=float, default=config.DELAY,
        metavar="SEC",
        help=f"Задержка между запросами (сек.). По умолч.: {config.DELAY}",
    )
    p.add_argument(
        "--pdf", action="store_true", default=False,
        help="Загружать PDF при отсутствии HTML-текста статьи.",
    )
    p.add_argument(
        "--no-json", action="store_true", default=False,
        help="Не сохранять JSON-файл.",
    )
    p.add_argument(
        "--clean", action="store_true", default=False,
        help="Применить очистку текста и сохранить *_cleaned.csv.",
    )
    p.add_argument(
        "--clean-only", action="store_true", default=False,
        help="Только очистить существующий CSV (--input), парсер не запускать.",
    )
    p.add_argument(
        "--input", type=str, default=None,
        metavar="FILE",
        help="Входной CSV для --clean-only.",
    )

    grp = p.add_argument_group("Опции очистки текста (работают с --clean и --clean-only)")
    grp.add_argument(
        "--no-join-hyphens", action="store_true",
        help="Не склеивать слова, разорванные OCR-переносом (по умолч. склейка ВКЛ)",
    )
    grp.add_argument(
        "--inline-refs", action="store_true",
        help="Удалять внутритекстовые ссылки (1), [2] (риск удалить годы вида (2016))",
    )
    grp.add_argument(
        "--keywords-line", action="store_true",
        help="Удалять строку «Ключевые слова: …» из тела текста",
    )
    grp.add_argument(
        "--footnote-nums", action="store_true",
        help="Удалять номера сносок «1.», «[1]» в начале строк",
    )
    return p


_FILTER_MAP = {"rsci": 22, "vak": 8, "scopus": 2, "none": None}


def _resolve_filter(name: str):
    return _FILTER_MAP.get(name.lower())


def _print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print(f"ИТОГО статей: {len(df)}")
    print()
    if "topic" in df.columns:
        print(df.groupby("topic")["id"].count().rename("статей").to_string())
        print()
    print(f"Входят в РИНЦ:   {df['rsci'].sum()}")
    print(f"Входят в ВАК:    {df['vak'].sum()}")
    print(f"Входят в Scopus: {df['scopus'].sum()}")
    if df["year"].notna().any():
        print(f"Средний год:     {df['year'].mean():.1f}")
    print(f"Ср. авторов:     {df['authors_count'].mean():.1f}")
    print(f"Ср. ключ. слов:  {df['keywords_count'].mean():.1f}")
    has_text = df["text_length"] > 0
    print(f"Со текстом:      {has_text.sum()} ({has_text.mean() * 100:.1f}%)")
    if has_text.any():
        sources = df.loc[has_text, "text_source"].value_counts().to_dict()
        print(f"  Источники:     {sources}")
        print(f"  Ср. длина:     {df.loc[has_text, 'text_length'].mean():.0f} симв.")
    print("=" * 60)


def _save(df: pd.DataFrame, csv_path: Path, json_path: Path, no_json: bool) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  CSV  → {csv_path}")
    if not no_json:
        df.to_json(json_path, orient="records", force_ascii=False, indent=2)
        print(f"  JSON → {json_path}")


def _build_clean_opts(args) -> dict:
    return {
        "annotation_col": "description",
        "join_ocr_hyphens": not args.no_join_hyphens,
        "remove_inline_refs": args.inline_refs,
        "remove_keywords_line": args.keywords_line,
        "remove_footnote_numbers": args.footnote_nums,
    }


def _fmt_opts(opts: dict) -> str:
    active = [k for k, v in opts.items() if v and k != "join_ocr_hyphens"]
    inactive = [] if opts.get("join_ocr_hyphens", True) else ["no-join-hyphens"]
    flags = active + inactive
    return f"[{', '.join(flags)}]" if flags else "[настройки по умолчанию]"


def main() -> None:
    args = build_parser().parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path, json_path, cleaned_path = _make_paths(ts)

    if args.clean_only:
        input_path = Path(args.input) if args.input else None
        if not input_path or not input_path.exists():
            print(f"[!] Укажите существующий файл через --input.")
            print(f"    Пример: python3 main.py --clean-only --input data/cyberleninka_articles_20260307_143022.csv")
            sys.exit(1)

        print(f"Загружаем {input_path} …")
        df = pd.read_csv(input_path, encoding="utf-8-sig")
        print(f"  Загружено строк: {len(df)}")

        stem = input_path.stem
        out_path = config.DATA_DIR / f"{stem}_cleaned.csv"
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)

        clean_opts = _build_clean_opts(args)
        print(f"Применяем очистку текста {_fmt_opts(clean_opts)} …")
        df_clean = clean_dataframe(df, **clean_opts)
        df_clean.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  Сохранено: {out_path}")
        return

    topics = (
        [t.strip() for t in args.topics.split(",")]
        if args.topics
        else config.TOPICS
    )
    filter_code = _resolve_filter(args.catalog_filter)

    print(f"Запуск: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Выходная папка: {config.DATA_DIR.resolve()}")
    print(f"Файлы будут сохранены с меткой: {ts}")
    print()
    print("Настройки:")
    print(f"  Темы:        {topics}")
    print(f"  Страниц:     {args.pages} (до {args.pages * config.ARTICLES_PER_PAGE} статей/тему)")
    print(f"  Фильтр:      {args.catalog_filter}")
    print(f"  Задержка:    {args.delay} с")
    print(f"  PDF-текст:   {'да' if args.pdf else 'нет'}")

    df = fetch_all_topics(
        topics=topics,
        max_pages=args.pages,
        filter_catalog=filter_code,
        fetch_pdf=args.pdf,
        delay=args.delay,
    )

    _print_summary(df)

    df["source"] = "CyberLeninka"
    df["source_name"] = "CyberLeninka"
    df["description"] = df["annotation"].astype(str)
    df["query"] = df["topic"].astype(str)
    df["language"] = "ru"
    df = df.drop(
        columns=["article_id", "journal", "keywords", "keywords_count", "rsci", "vak", "scopus", "annotation", "topic"],
        errors="ignore",
    )
    df = df[[c for c in ARTICLE_COLUMNS if c in df.columns]]

    print("\nСохранение …")
    _save(df, csv_path, json_path, args.no_json)

    if args.clean:
        clean_opts = _build_clean_opts(args)
        print(f"\nОчистка текста {_fmt_opts(clean_opts)} …")
        df_clean = clean_dataframe(df, **clean_opts)
        df_clean.to_csv(cleaned_path, index=False, encoding="utf-8-sig")
        print(f"  Очищенный CSV → {cleaned_path}")

    print(f"\nГотово. Файлы в папке: {config.DATA_DIR}/")


if __name__ == "__main__":
    main()

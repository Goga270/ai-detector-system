import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
from shared.io import save_articles, rows_to_dataframe
from grab import grab


def main():
    p = argparse.ArgumentParser(description="Сбор статей через NewsAPI.org")
    p.add_argument("--api-key", required=True, help="API key с newsapi.org")
    p.add_argument("--query", default="Россия")
    p.add_argument("--lang", default="ru")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--fetch-text", action="store_true", help="Подгружать полный текст по URL (медленнее)")
    p.add_argument("--clean", action="store_true", help="Очистка текста (cyberleninka cleaner)")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    args = p.parse_args()

    rows = grab(api_key=args.api_key, query=args.query, language=args.lang, limit=args.limit, fetch_text=args.fetch_text)
    df = rows_to_dataframe(rows)
    save_articles(df, "newsapi", args.data_dir, clean=args.clean)
    print(f"Готово: {len(df)} статей")


if __name__ == "__main__":
    main()

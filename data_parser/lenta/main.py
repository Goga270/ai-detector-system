import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
from shared.io import save_articles, rows_to_dataframe
from grab import grab


def main():
    p = argparse.ArgumentParser(description="Сбор новостей Lenta.ru (RSS + страницы статей)")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--clean", action="store_true")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    args = p.parse_args()

    rows = grab(limit=args.limit)
    df = rows_to_dataframe(rows)
    save_articles(df, "lenta", args.data_dir, clean=args.clean)
    print(f"Готово: {len(df)} статей")


if __name__ == "__main__":
    main()

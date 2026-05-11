"""
newsparser: RSS + optional EventRegistry/MediaCloud.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src import config
from src.eventregistry import fetch_eventregistry
from src.mediacloud import fetch_mediacloud
from src.rss import fetch_all_sources


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Сбор новостей из RSS и API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--sources", type=str, default=None,
        metavar="ИСТОЧНИКИ",
        help=(
            "Источники через запятую: lenta,gazeta,rbc,ria,kommersant. "
            f"По умолч.: все ({', '.join(config.DEFAULT_SOURCES)})"
        ),
    )
    p.add_argument(
        "--max", type=int, default=config.MAX_PER_SOURCE,
        dest="max_per_source",
        metavar="N",
        help=f"Статей на источник. По умолч.: {config.MAX_PER_SOURCE}",
    )
    p.add_argument(
        "--delay", type=float, default=config.DELAY,
        metavar="SEC",
        help=f"Пауза между запросами (сек.). По умолч.: {config.DELAY}",
    )
    p.add_argument(
        "--er-key", type=str, default=None,
        metavar="KEY",
        help="API-ключ EventRegistry (или задайте env ER_API_KEY).",
    )
    p.add_argument(
        "--mc-key", type=str, default=None,
        metavar="KEY",
        help="API-ключ MediaCloud (или задайте env MC_API_KEY).",
    )
    p.add_argument(
        "--lang", type=str, default="ru",
        metavar="LANG",
        help="Код языка для API-источников: ru, en, de… По умолч.: ru",
    )
    p.add_argument(
        "--no-json", action="store_true", default=False,
        help="Не сохранять JSON-файлы.",
    )
    p.add_argument(
        "--no-merge", action="store_true", default=False,
        help="Не создавать объединённый файл со всеми источниками.",
    )
    return p


def _save(
    articles: List[dict],
    name: str,
    ts: str,
    no_json: bool,
) -> Path:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    stem = config.DATA_DIR / f"{config.FILE_BASENAME}_{name}_{ts}"
    csv_path = stem.with_suffix(".csv")
    json_path = stem.with_suffix(".json")

    df = pd.DataFrame(articles)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  CSV  → {csv_path}  ({len(df)} строк)")

    if not no_json:
        df.to_json(json_path, orient="records", force_ascii=False, indent=2)
        print(f"  JSON → {json_path}")

    return csv_path


def _print_summary(results: Dict[str, List[dict]]) -> None:
    print("\n" + "=" * 60)
    print("ИТОГОВАЯ СТАТИСТИКА")
    total = 0
    for src, arts in results.items():
        print(f"  {src:<15} {len(arts):>5} статей")
        total += len(arts)
    print(f"  {'ИТОГО':<15} {total:>5}")
    print("=" * 60)


def main() -> None:
    args = build_parser().parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    sources = (
        [s.strip() for s in args.sources.split(",")]
        if args.sources
        else config.DEFAULT_SOURCES
    )

    print(f"Запуск: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Выходная папка: {config.DATA_DIR.resolve()}")
    print(f"Метка времени:  {ts}")
    print()
    print(f"RSS-источники:  {sources}")
    print(f"Макс. статей:   {args.max_per_source}/источник")
    print(f"EventRegistry:  {'да' if (args.er_key or __import__('os').environ.get('ER_API_KEY')) else 'нет'}")
    print(f"MediaCloud:     {'да' if (args.mc_key or __import__('os').environ.get('MC_API_KEY')) else 'нет'}")

    all_results: Dict[str, List[dict]] = {}

    print("\n" + "─" * 60)
    print("RSS-источники")
    print("─" * 60)
    rss_results = fetch_all_sources(
        sources=sources,
        max_per_source=args.max_per_source,
        delay=args.delay,
    )
    for src, arts in rss_results.items():
        all_results[src] = arts
        print(f"\nСохраняю {src} …")
        _save(arts, src, ts, args.no_json)

    if args.er_key or __import__("os").environ.get("ER_API_KEY"):
        print("\n" + "─" * 60)
        print("EventRegistry")
        print("─" * 60)
        er_arts = fetch_eventregistry(
            api_key=args.er_key,
            max_items=args.max_per_source,
            lang=args.lang,
        )
        if er_arts:
            all_results["eventregistry"] = er_arts
            _save(er_arts, "eventregistry", ts, args.no_json)

    if args.mc_key or __import__("os").environ.get("MC_API_KEY"):
        print("\n" + "─" * 60)
        print("MediaCloud")
        print("─" * 60)
        mc_arts = fetch_mediacloud(
            api_key=args.mc_key,
            max_items=args.max_per_source,
            lang=args.lang,
        )
        if mc_arts:
            all_results["mediacloud"] = mc_arts
            _save(mc_arts, "mediacloud", ts, args.no_json)

    if not args.no_merge and all_results:
        all_articles = [a for arts in all_results.values() for a in arts]
        if all_articles:
            print("\nСохраняю объединённый файл …")
            _save(all_articles, "all", ts, args.no_json)

    _print_summary(all_results)
    print(f"\nГотово. Файлы в папке: {config.DATA_DIR}/")


if __name__ == "__main__":
    main()

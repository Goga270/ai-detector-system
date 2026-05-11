"""RSS-ленты из config.RSS_FEEDS (lenta, gazeta, rbc, ria, kommersant)."""

import re
from time import sleep
from typing import Dict, List, Optional

import feedparser

from src.config import DELAY, MAX_PER_SOURCE, REQUEST_TIMEOUT, RSS_FEEDS


def _strip_html(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"<[^>]+>", " ", text).strip()


def parse_entry(entry) -> dict:
    title = entry.get("title", "") or ""
    link = entry.get("link", "") or entry.get("id", "") or ""
    published = (
        entry.get("published")
        or entry.get("updated")
        or entry.get("pubDate")
        or ""
    )
    content = ""
    if "content" in entry:
        try:
            content = entry.content[0].value or ""
        except (IndexError, AttributeError):
            pass
    if not content:
        content = entry.get("summary") or entry.get("description") or ""

    return {
        "title": title.strip(),
        "url": link.strip(),
        "published_at": published.strip(),
        "content": _strip_html(content),
    }


def fetch_source(
    source_name: str,
    feed_urls: List[str],
    max_items: int = MAX_PER_SOURCE,
    delay: float = DELAY,
) -> List[dict]:
    articles: List[dict] = []
    seen_urls: set = set()

    for feed_url in feed_urls:
        if len(articles) >= max_items:
            break

        try:
            parsed = feedparser.parse(
                feed_url,
                request_headers={"User-Agent": "Mozilla/5.0 (compatible; newsparser/1.0)"},
            )
        except Exception as exc:
            print(f"  [{source_name}] Ошибка загрузки {feed_url}: {exc}")
            continue

        if parsed.bozo:
            exc_msg = str(getattr(parsed, "bozo_exception", ""))
            print(f"  [{source_name}] feedparser.bozo: {exc_msg[:80]}")

        entries = parsed.get("entries", [])
        if not entries:
            print(f"  [{source_name}] Лента пуста: {feed_url}")
            continue

        for entry in entries:
            item = parse_entry(entry)
            if not item["url"]:
                continue
            if item["url"] in seen_urls:
                continue
            seen_urls.add(item["url"])
            item["id"] = item["url"]
            item["source"] = source_name
            articles.append(item)
            if len(articles) >= max_items:
                break

        sleep(delay)

    print(f"  [{source_name}] Собрано: {len(articles)} статей")
    return articles


def fetch_all_sources(
    sources: Optional[List[str]] = None,
    max_per_source: int = MAX_PER_SOURCE,
    delay: float = DELAY,
) -> Dict[str, List[dict]]:
    if sources is None:
        sources = list(RSS_FEEDS.keys())

    results: Dict[str, List[dict]] = {}
    for src in sources:
        if src not in RSS_FEEDS:
            print(f"[!] Источник '{src}' не в RSS_FEEDS, пропуск.")
            continue
        print(f"\nСобираю RSS: {src}")
        results[src] = fetch_source(
            source_name=src,
            feed_urls=RSS_FEEDS[src],
            max_items=max_per_source,
            delay=delay,
        )
    return results

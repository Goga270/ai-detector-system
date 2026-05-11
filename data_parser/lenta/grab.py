import feedparser

from shared.helpers import sleep, uniq, extract_article, collect_links

SOURCE = "lenta"
LENTA_RSS = [
    "https://lenta.ru/rss/news",
    "https://lenta.ru/rss/articles",
    "https://lenta.ru/rss/last24",
    "https://lenta.ru/rss/top7",
    "https://lenta.ru/rss/news/russia",
    "https://lenta.ru/rss/news/world",
]
SELECTORS = [
    "div.topic-body__content p",
    "div.topic-body__content-text p",
    "div.js-topic__text p",
    "article p",
]


def _row(url):
    r = extract_article(url, "Lenta.ru", selectors=SELECTORS)
    r["source"] = SOURCE
    r["source_name"] = "Lenta.ru"
    r["authors"] = r.get("author", "")
    r["text_length"] = len(r.get("text") or "")
    r["text_source"] = "html"
    r.setdefault("description", "")
    r.setdefault("query", "")
    r.setdefault("language", "")
    return r


def grab(limit: int = 100) -> list[dict]:
    links = []
    for feed_url in LENTA_RSS:
        try:
            feed = feedparser.parse(feed_url)
            links.extend([e.link for e in feed.entries if getattr(e, "link", None)])
        except Exception:
            pass
    if len(uniq(links)) < limit:
        links.extend(
            collect_links(
                "https://lenta.ru/",
                allowed_domain="lenta.ru",
                url_regex=r"lenta\.ru/(news|articles)/\d{4}/\d{2}/\d{2}/",
                limit=300,
            )
        )
    links = uniq(links)[:limit]
    rows = []
    for i, url in enumerate(links, 1):
        rows.append(_row(url))
        if i % 10 == 0:
            print(f"  {i}/{len(links)}")
        sleep(0.3)
    return rows

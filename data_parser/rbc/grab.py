import feedparser

from shared.helpers import sleep, uniq, extract_article, collect_links

SOURCE = "rbc"
RBC_RSS = ["https://rssexport.rbc.ru/rbcnews/news/30/full.rss"]
RBC_SEEDS = [
    "https://www.rbc.ru/",
    "https://www.rbc.ru/politics/",
    "https://www.rbc.ru/economics/",
    "https://www.rbc.ru/society/",
    "https://www.rbc.ru/business/",
    "https://www.rbc.ru/technology_and_media/",
    "https://www.rbc.ru/finances/",
    "https://www.rbc.ru/rbcfreenews/",
]
SKIP_DOMAINS = ("pro.rbc.ru", "plus.rbc.ru", "tv.rbc.ru", "quote.rbc.ru", "editorial.rbc.ru")
PATTERN = r"rbc\.ru/(?:rbcfreenews|[a-z_]+)/"
SELECTORS = [
    "div.article__text p",
    "div.article__content p",
    "div.l-col-main p",
    "article p",
]


def _row(url):
    r = extract_article(url, "РБК", selectors=SELECTORS)
    r["source"] = SOURCE
    r["source_name"] = "РБК"
    r["authors"] = r.get("author", "")
    r["text_length"] = len(r.get("text") or "")
    r["text_source"] = "html"
    r.setdefault("description", "")
    r.setdefault("query", "")
    r.setdefault("language", "")
    return r


def grab(limit: int = 100) -> list[dict]:
    links = []
    for feed_url in RBC_RSS:
        try:
            feed = feedparser.parse(feed_url)
            links.extend([e.link for e in feed.entries if getattr(e, "link", None)])
        except Exception:
            pass
    for seed in RBC_SEEDS:
        part = collect_links(seed, allowed_domain="rbc.ru", url_regex=PATTERN, limit=400)
        links.extend(part)
        if len(uniq(links)) >= limit:
            break
        sleep(0.3)
    links = [u for u in uniq(links) if not any(d in u for d in SKIP_DOMAINS)][:limit]
    rows = []
    for i, url in enumerate(links, 1):
        rows.append(_row(url))
        if i % 10 == 0:
            print(f"  {i}/{len(links)}")
        sleep(0.3)
    return rows

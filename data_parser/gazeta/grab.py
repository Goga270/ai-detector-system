import re
from shared.helpers import sleep, uniq, extract_article, collect_links

SOURCE = "gazeta"
GAZETA_SEEDS = [
    "https://www.gazeta.ru/news/",
    "https://www.gazeta.ru/politics/news/",
    "https://www.gazeta.ru/business/news/",
    "https://www.gazeta.ru/social/news/",
    "https://www.gazeta.ru/army/news/",
    "https://www.gazeta.ru/tech/news/",
    "https://www.gazeta.ru/culture/news/",
    "https://www.gazeta.ru/sport/news/",
]
PATTERN = re.compile(
    r"gazeta\.ru/[^/]+/news/\d{4}/\d{2}/\d{2}/\d+(?:\.shtml)?(?:\?|$)"
)
SELECTORS = [
    'div[itemprop="articleBody"] p',
    "div.b_article-text p",
    "div.article_text p",
    "article p",
]


def _row(url):
    r = extract_article(url, "Газета.Ru", selectors=SELECTORS, use_js=True)
    r["source"] = SOURCE
    r["source_name"] = "Газета.Ru"
    r["authors"] = r.get("author", "")
    r["text_length"] = len(r.get("text") or "")
    r["text_source"] = "html"
    r.setdefault("description", "")
    r.setdefault("query", "")
    r.setdefault("language", "")
    return r


def grab(limit: int = 100, debug: bool = False) -> list[dict]:
    links = []
    for seed in GAZETA_SEEDS:
        part = collect_links(seed, allowed_domain="gazeta.ru", url_regex=PATTERN, limit=300, debug=debug, use_js=True)
        links.extend(part)
        if len(uniq(links)) >= limit:
            break
        sleep(0.3)
    links = uniq(links)[:limit]
    if len(links) == 0:
        print("[gazeta] найдено 0 ссылок. Контент Газета.Ru подгружается через JS — нужен Playwright:")
        print("  pip install playwright && playwright install chromium")
        print("Диагностика по первой странице:")
        collect_links(GAZETA_SEEDS[0], allowed_domain="gazeta.ru", url_regex=PATTERN, limit=300, debug=True, use_js=True)
    rows = []
    for i, url in enumerate(links, 1):
        rows.append(_row(url))
        if i % 10 == 0:
            print(f"  {i}/{len(links)}")
        sleep(0.3)
    return rows

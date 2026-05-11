import requests

from shared.helpers import sleep, strip_html, extract_article

SOURCE = "newsapi"
SKIP_FETCH_DOMAINS = ("youtube.com", "youtu.be", "anekdot.ru")


def grab(api_key: str, query: str = "Россия", language: str = "ru", limit: int = 100, sort_by: str = "publishedAt", fetch_text: bool = False) -> list[dict]:
    url = "https://newsapi.org/v2/everything"
    rows = []
    page = 1
    while len(rows) < limit:
        page_size = min(100, limit - len(rows))
        params = {"q": query, "language": language, "sortBy": sort_by, "pageSize": page_size, "page": page}
        r = requests.get(url, params=params, headers={"X-Api-Key": api_key}, timeout=30)
        # Бесплатный Developer-план: часто только страница 1 (до 100 статей); page≥2 → 426 Upgrade Required
        if r.status_code == 426 and page > 1 and rows:
            try:
                detail = r.json().get("message", "")
            except Exception:
                detail = ""
            print(
                "  [NewsAPI] HTTP 426: пагинация недоступна на вашем тарифе "
                f"(страница {page}). Уже загружено {len(rows)} статей — продолжаем без падения."
            )
            if detail:
                print(f"            ({detail})")
            break
        r.raise_for_status()
        data = r.json()
        articles = data.get("articles", [])
        if not articles:
            break
        for a in articles:
            content = a.get("content") or ""
            t = strip_html(content)
            rows.append({
                "source": SOURCE,
                "source_name": (a.get("source") or {}).get("name") or "",
                "title": a.get("title"),
                "authors": a.get("author") or "",
                "published_at": a.get("publishedAt"),
                "url": a.get("url"),
                "description": a.get("description") or "",
                "text": t,
                "text_length": len(t),
                "text_source": "api",
                "query": query,
                "language": language,
            })
            if len(rows) >= limit:
                break
        page += 1
        sleep(0.5)
    if fetch_text:
        for i, row in enumerate(rows):
            url = row.get("url")
            if not url or any(d in url for d in SKIP_FETCH_DOMAINS):
                continue
            try:
                got = extract_article(url, row.get("source_name") or "NewsAPI", use_js=False)
                t = (got.get("text") or "").strip()
                if len(t) > 200 and "browser" not in t.lower()[:200]:
                    rows[i]["text"] = t
                    rows[i]["text_length"] = len(t)
                    rows[i]["text_source"] = "html"
            except Exception:
                pass
            if (i + 1) % 20 == 0:
                print(f"  загрузка текста: {i + 1}/{len(rows)}")
            sleep(0.3)
    return rows[:limit]

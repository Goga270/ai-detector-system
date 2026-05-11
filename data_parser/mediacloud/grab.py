import re
from datetime import date, timedelta

import mediacloud.api

from shared.helpers import extract_article, sleep

SOURCE = "mediacloud"


def grab(api_key: str, query: str = "Россия", limit: int = 100, start_date=None, end_date=None, fetch_text: bool = False) -> list[dict]:
    mc = mediacloud.api.SearchApi(api_key)
    end = end_date or date.today()
    start = start_date or (end - timedelta(days=30))
    if not hasattr(start, "isoformat"):
        start = date.fromisoformat(str(start)[:10])
    if not hasattr(end, "isoformat"):
        end = date.fromisoformat(str(end)[:10])
    rows = []
    pagination_token = None
    while len(rows) < limit:
        page, pagination_token = mc.story_list(query, start_date=start, end_date=end, pagination_token=pagination_token)
        if not page:
            break
        for story in page:
            lang = story.get("language")
            title = story.get("title") or ""
            if (lang == "ru") or re.search(r"[А-Яа-яЁё]", title):
                rows.append({
                    "source": SOURCE,
                    "source_name": story.get("media_name") or "",
                    "title": story.get("title"),
                    "authors": "",
                    "published_at": story.get("publish_date"),
                    "url": story.get("url"),
                    "description": "",
                    "text": "",
                    "text_length": 0,
                    "text_source": "",
                    "query": query,
                    "language": lang or "",
                })
            if len(rows) >= limit:
                break
        if pagination_token is None:
            break
    if fetch_text:
        for i, row in enumerate(rows):
            url = row.get("url")
            if not url:
                continue
            try:
                got = extract_article(url, row.get("source_name") or "Media Cloud", use_js=False)
                t = (got.get("text") or "").strip()
                if len(t) > 150:
                    rows[i]["text"] = t
                    rows[i]["text_length"] = len(t)
                    rows[i]["text_source"] = "html"
            except Exception:
                pass
            if (i + 1) % 20 == 0:
                print(f"  загрузка текста: {i + 1}/{len(rows)}")
            sleep(0.3)
    return rows[:limit]

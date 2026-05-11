"""Статьи через EventRegistry REST (ключ: ER_API_KEY или аргумент api_key)."""

import os
from typing import Any, Dict, List, Optional

import requests

from src.config import REQUEST_TIMEOUT

_API_URL = "https://eventregistry.org/api/v1/article/getArticles"


def _lang_param(lang: str) -> str:
    if lang in ("ru", "rus"):
        return "rus"
    if lang in ("en", "eng"):
        return "eng"
    return lang


def fetch_eventregistry(
    api_key: Optional[str] = None,
    max_items: int = 100,
    lang: str = "rus",
) -> List[dict]:
    key = api_key or os.getenv("ER_API_KEY")
    if not key:
        print("  [EventRegistry] API-ключ не передан. Пропускаю.")
        return []

    lang_er = _lang_param(lang)
    n = min(int(max_items), 100)
    body: Dict[str, Any] = {
        "action": "getArticles",
        "apiKey": key,
        "keyword": "новости",
        "articlesCount": n,
        "articlesSortBy": "date",
        "lang": lang_er,
        "resultType": "articles",
        "articlesArticleBodyLen": -1,
        "includeArticleBody": True,
        "includeArticleTitle": True,
    }

    try:
        resp = requests.post(_API_URL, json=body, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        print(f"  [EventRegistry] Сетевая ошибка: {exc}")
        return []

    if resp.status_code != 200:
        print(f"  [EventRegistry] HTTP {resp.status_code}: {resp.text[:300]}")
        return []

    try:
        data = resp.json()
    except ValueError:
        print("  [EventRegistry] Ответ не JSON.")
        return []

    raw = data.get("articles")
    if isinstance(raw, dict):
        arts = raw.get("results", [])
    elif isinstance(raw, list):
        arts = raw
    else:
        arts = []

    out: List[dict] = []
    for a in arts:
        if not isinstance(a, dict):
            continue
        src = a.get("source") or {}
        src_title = src.get("title", "") if isinstance(src, dict) else str(src)
        out.append(
            {
                "id": str(a.get("uri") or a.get("url") or ""),
                "source": "eventregistry",
                "title": (a.get("title") or "").strip(),
                "url": (a.get("url") or "").strip(),
                "published_at": str(a.get("date") or a.get("dateTime") or ""),
                "content": (a.get("body") or "").strip(),
            }
        )

    print(f"  [EventRegistry] Получено статей: {len(out)}")
    return out

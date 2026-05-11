"""Выгрузка статей через MediaCloud API (ключ MC_API_KEY или аргумент api_key)."""

import os
from typing import List, Optional


def fetch_mediacloud(
    api_key: Optional[str] = None,
    max_items: int = 100,
    lang: str = "ru",
) -> List[dict]:
    key = api_key or os.environ.get("MC_API_KEY", "")
    if not key:
        print("  [MediaCloud] API-ключ не передан. Пропуск.")
        return []

    try:
        import mediacloud.api as mc_api
    except ImportError:
        print("  [MediaCloud] pip install mediacloud")
        return []

    try:
        mc = mc_api.MediaCloud(key)
        result = mc.storyList(solr_query="*", rows=max_items, fq=f"language:{lang}")
        stories = result.get("stories", []) if isinstance(result, dict) else result

        articles = []
        for s in stories:
            articles.append(
                {
                    "id": str(s.get("stories_id") or s.get("url", "")),
                    "source": s.get("media_name", ""),
                    "title": s.get("title", ""),
                    "url": s.get("url", ""),
                    "published_at": s.get("publish_date", ""),
                    "content": s.get("story", "") or s.get("description", ""),
                }
            )

        print(f"  [MediaCloud] Получено статей: {len(articles)}")
        return articles[:max_items]

    except Exception as exc:
        print(f"  [MediaCloud] Ошибка API: {exc}")
        return []

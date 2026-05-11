import re

# Единый формат для слияния всех датасетов (CyberLeninka + новости)
ARTICLE_COLUMNS = [
    "id",
    "source",
    "source_name",
    "title",
    "authors",
    "authors_count",
    "year",
    "url",
    "description",
    "text",
    "text_length",
    "text_source",
    "query",
    "language",
]


def _year_from_published(s: str) -> int | str:
    if not s:
        return ""
    m = re.search(r"(\d{4})", str(s))
    return int(m.group(1)) if m else ""


def normalize_row(row: dict) -> dict:
    """Приводит строку к единой схеме. author→authors, published_at→year."""
    out = {
        "id": row.get("id") or "",
        "source": row.get("source") or "",
        "source_name": row.get("source_name") or "",
        "title": row.get("title") or "",
        "authors": row.get("authors") or row.get("author") or "",
        "authors_count": row.get("authors_count"),
        "year": row.get("year"),
        "url": row.get("url") or "",
        "description": row.get("description") or "",
        "text": row.get("text") or "",
        "text_length": row.get("text_length"),
        "text_source": row.get("text_source") or "",
        "query": row.get("query") or "",
        "language": row.get("language") or "",
    }
    if out["year"] is None or out["year"] == "":
        out["year"] = _year_from_published(row.get("published_at") or "")
    if out["authors_count"] is None or out["authors_count"] == "":
        a = out["authors"] or ""
        out["authors_count"] = len([x for x in a.split(",") if x.strip()]) if a else 0
    if out["text_length"] is None or out["text_length"] == "":
        out["text_length"] = len(out["text"] or "")
    return out

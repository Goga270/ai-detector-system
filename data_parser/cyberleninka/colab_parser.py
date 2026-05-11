# Colab-скрипт для CyberLeninka (requests, bs4, pandas). Не используется основным main.py.

import requests
import json
import re
import time
import io
import pandas as pd
import bs4
from typing import Optional, List

TOPICS: List[str] = [
    "Философия, этика, религиоведение",
    "История и археология",
    "Математика",
    "Право",
    "Политологические науки",
]

MAX_PAGES: int = 60

FILTER: Optional[int] = None
DELAY: float = 1.2

FETCH_PDF_TEXT: bool = False

API_URL           = "https://cyberleninka.ru/api/search"
BASE_URL          = "https://cyberleninka.ru"
ARTICLES_PER_PAGE = 10

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Content-Type": "application/json",
    "Referer": "https://cyberleninka.ru/",
}


# ==================== ПОИСК ЧЕРЕЗ API ====================

def _search_page(query: str, size: int, offset: int,
                 filters: Optional[List[int]] = None) -> dict:
    body: dict = {
        "mode": "articles",
        "size": size,
        "q": query,
        "from": offset,
    }
    if filters:
        body["catalogs"] = filters
    try:
        resp = requests.post(
            API_URL, data=json.dumps(body), headers=HEADERS, timeout=20
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"  [!] Ошибка API: {e}")
        return {}


def collect_links(query: str, max_pages: int,
                  filters: Optional[List[int]] = None) -> List[str]:
    """Собирает ссылки на статьи по всем страницам поиска."""
    links: List[str] = []

    first = _search_page(query, ARTICLES_PER_PAGE, 0, filters)
    if not first:
        return links

    total_found = first.get("found", 0)
    print(f"  Найдено на сервере: {total_found} статей")

    for art in first.get("articles", []):
        lnk = art.get("link", "")
        if lnk:
            links.append(BASE_URL + lnk)

    for page in range(1, max_pages):
        if len(links) >= total_found:
            break
        result = _search_page(
            query, ARTICLES_PER_PAGE, page * ARTICLES_PER_PAGE, filters
        )
        if not result:
            break
        batch = result.get("articles", [])
        if not batch:
            break
        for art in batch:
            lnk = art.get("link", "")
            if lnk:
                links.append(BASE_URL + lnk)
        time.sleep(DELAY)

    return links


# ==================== ИЗВЛЕЧЕНИЕ ТЕКСТА ИЗ PDF ====================

def _extract_pdf_text(article_url: str) -> str:
    """
    Скачивает PDF статьи и извлекает текст через pdfminer.
    Возвращает пустую строку при ошибке.
    """
    try:
        from pdfminer.high_level import extract_text as pdf_extract
    except ImportError:
        return ""

    try:
        resp = requests.get(article_url + "/pdf", headers=HEADERS,
                            stream=True, timeout=30)
        resp.raise_for_status()
        pdf_bytes = io.BytesIO(resp.content)
        text = pdf_extract(pdf_bytes)
        return text.strip() if text else ""
    except Exception:
        return ""


# ==================== ПАРСИНГ СТРАНИЦЫ СТАТЬИ ====================

def _article_slug(url: str) -> str:
    """Извлекает slug статьи из URL в качестве строкового ID."""
    m = re.search(r"/article/n/(.+?)/?$", url)
    return m.group(1) if m else url


def parse_article_page(url: str, fetch_pdf: bool = False) -> dict:
    """
    Скрапит страницу статьи и возвращает словарь с данными.
    Полный текст:
      - В первую очередь ищется OCR-текст на HTML-странице (div.ocr, article и т.д.)
      - Если FETCH_PDF_TEXT=True — дополнительно скачивается PDF
    """
    record = {
        "article_id":      _article_slug(url),
        "url":             url,
        "title":           "",
        "authors":         "",
        "authors_count":   0,
        "year":            None,
        "journal":         "",
        "annotation":      "",
        "keywords":        "",
        "keywords_count":  0,
        "rsci":            False,
        "vak":             False,
        "scopus":          False,
        "text":            "",
        "text_length":     0,
        "text_source":     "",   # "html_ocr" | "pdf" | ""
    }

    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"    [!] Ошибка страницы: {e}")
        return record

    soup = bs4.BeautifulSoup(resp.text, "html.parser")

    # --- Заголовок ---
    title_tag = soup.find("i")
    if title_tag:
        record["title"] = title_tag.get_text(strip=True)

    # --- Авторы ---
    right_title = soup.find("h2", {"class": "right-title"})
    if right_title:
        span = right_title.find("span")
        if span:
            txt = span.get_text(strip=True)
            dash = txt.find("—")
            authors_str = txt[dash + 2:].strip() if dash >= 0 else txt
            record["authors"] = authors_str
            record["authors_count"] = len(
                [a for a in authors_str.split(",") if a.strip()]
            )

    # --- Год и метки индексации ---
    labels_div = soup.find("div", {"class": "labels"})
    if labels_div:
        time_tag = labels_div.find("time")
        if time_tag:
            try:
                record["year"] = int(time_tag.get_text(strip=True))
            except ValueError:
                pass
        record["rsci"]   = bool(labels_div.find("div", {"class": "label rsci"}))
        record["vak"]    = bool(labels_div.find("div", {"class": "label vak"}))
        record["scopus"] = bool(labels_div.find("div", {"class": "label scopus"}))

    # --- Журнал ---
    journal_tag = soup.find("a", {"class": "link"})
    if journal_tag:
        record["journal"] = journal_tag.get_text(strip=True)

    # --- Аннотация ---
    for name, attrs in [
        ("div",     {"class": "abstract"}),
        ("section", {"class": "abstract"}),
        ("div",     {"itemprop": "description"}),
    ]:
        tag = soup.find(name, attrs)
        if tag:
            record["annotation"] = tag.get_text(" ", strip=True)
            break

    # --- Ключевые слова ---
    kw_found = False
    for name, attrs in [
        ("div",  {"class": "keywords"}),
        ("span", {"itemprop": "keywords"}),
        ("meta", {"name": "keywords"}),
    ]:
        tag = soup.find(name, attrs)
        if tag:
            kw_text = (
                tag.get("content", "")
                if name == "meta"
                else tag.get_text(", ", strip=True)
            )
            if kw_text:
                record["keywords"]       = kw_text
                record["keywords_count"] = len(
                    [k for k in kw_text.split(",") if k.strip()]
                )
                kw_found = True
                break

    if not kw_found:
        for elem in soup.find_all(["p", "div"]):
            txt = elem.get_text(" ", strip=True)
            if txt.startswith("Ключевые слова"):
                kw_text = re.sub(r"Ключевые\s*слова[:：]?\s*", "", txt).strip()
                record["keywords"]       = kw_text
                record["keywords_count"] = len(
                    [k for k in kw_text.split(",") if k.strip()]
                )
                break

    # --- Полный текст статьи ---
    # Сначала ищем OCR/HTML-текст прямо на странице
    html_text = ""
    for name, attrs in [
        ("div",     {"class": "ocr"}),          # CyberLeninka OCR-блок
        ("div",     {"itemprop": "articleBody"}),
        ("section", {"class": "body"}),
        ("div",     {"id": "body"}),
        ("article", {}),
    ]:
        tag = soup.find(name, attrs)
        if tag:
            candidate = tag.get_text(" ", strip=True)
            if len(candidate) > 300:             # Явно не пустышка
                html_text = candidate
                break

    if html_text:
        record["text"]        = html_text
        record["text_length"] = len(html_text)
        record["text_source"] = "html_ocr"
    elif fetch_pdf:
        pdf_text = _extract_pdf_text(url)
        if pdf_text:
            record["text"]        = pdf_text
            record["text_length"] = len(pdf_text)
            record["text_source"] = "pdf"

    return record


# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================

def fetch_topic(
    topic: str,
    max_pages: int = MAX_PAGES,
    filter_catalog: Optional[int] = FILTER,
    fetch_pdf: bool = FETCH_PDF_TEXT,
) -> pd.DataFrame:
    """Выгружает статьи по одной теме, возвращает DataFrame с колонкой topic."""
    filters = [filter_catalog] if filter_catalog else None

    print(f"\n{'=' * 60}")
    print(f"Тема: {topic}")
    print(f"Целевое кол-во: до {max_pages * ARTICLES_PER_PAGE} статей")
    print("=" * 60)

    links = collect_links(topic, max_pages, filters)
    print(f"  Ссылок собрано: {len(links)}\n")

    records = []
    for i, url in enumerate(links, 1):
        print(f"  [{i:>4}/{len(links)}] {url}")
        rec = parse_article_page(url, fetch_pdf=fetch_pdf)
        rec["topic"] = topic
        records.append(rec)
        time.sleep(DELAY)

    return pd.DataFrame(records)


def fetch_all_topics(
    topics: List[str] = TOPICS,
    max_pages: int = MAX_PAGES,
    filter_catalog: Optional[int] = FILTER,
    fetch_pdf: bool = FETCH_PDF_TEXT,
) -> pd.DataFrame:
    """Обходит все темы, объединяет в один DataFrame, добавляет id."""
    all_frames: List[pd.DataFrame] = []

    for topic in topics:
        df_topic = fetch_topic(topic, max_pages, filter_catalog, fetch_pdf)
        all_frames.append(df_topic)

    df = pd.concat(all_frames, ignore_index=True)

    # Сквозной числовой ID (начиная с 1)
    df.insert(0, "id", range(1, len(df) + 1))

    # Привести типы
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for col in ("authors_count", "keywords_count", "text_length"):
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            )

    return df


# ==================== ЗАПУСК ====================

df = fetch_all_topics()

# --- Статистика ---
print("\n" + "=" * 60)
print(f"ИТОГО статей: {len(df)}")
print()
print(df.groupby("topic")["id"].count().rename("статей").to_string())
print()
filter_names = {22: "РИНЦ", 8: "ВАК", 2: "Scopus"}
print(f"Входят в РИНЦ:           {df['rsci'].sum()}")
print(f"Входят в ВАК:            {df['vak'].sum()}")
print(f"Входят в Scopus:         {df['scopus'].sum()}")
if df["year"].notna().any():
    print(f"Средний год публикации:  {df['year'].mean():.1f}")
print(f"Ср. кол-во авторов:      {df['authors_count'].mean():.1f}")
print(f"Ср. кол-во ключ. слов:   {df['keywords_count'].mean():.1f}")
has_text = df["text_length"] > 0
print(f"Статей с текстом:        {has_text.sum()} "
      f"({has_text.mean() * 100:.1f}%)")
if has_text.any():
    print(f"Ср. длина текста:        {df.loc[has_text, 'text_length'].mean():.0f} симв.")

print("\nКолонки DataFrame:")
print(df.dtypes.to_string())

print("\nПервые 3 строки:")
display_cols = ["id", "topic", "article_id", "title", "authors", "year",
                "journal", "rsci", "vak", "scopus", "text_length"]
print(df[display_cols].head(3).to_string())

# --- Сохранение ---
CSV_FILE  = "cyberleninka_articles.csv"

df.to_csv(CSV_FILE,  index=False, encoding="utf-8-sig")

print(f"\nФайлы сохранены: {CSV_FILE}, {JSON_FILE}")

# Скачать из Google Colab
try:
    from google.colab import files
    files.download(CSV_FILE)
    files.download(JSON_FILE)
except ImportError:
    print("(Не в Colab — файлы сохранены локально)")

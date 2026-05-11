"""
Модуль парсинга статей с CyberLeninka.

Архитектура запроса данных:
1. Поиск через POST API → список ссылок на статьи.
2. Для каждой ссылки — GET HTML-страницы статьи → BeautifulSoup-разбор.
3. Опционально — загрузка PDF → извлечение текста через pdfminer.

CyberLeninka не предоставляет REST API для деталей статьи:
все поля (авторы, год, метки РИНЦ/ВАК/Scopus, текст) берутся
только с HTML-страницы и/или PDF.
"""

import io
import json
import re
import time
from typing import List, Optional

import bs4
import pandas as pd
import requests

from src.config import (
    API_URL, ARTICLES_PER_PAGE, BASE_URL, DELAY,
    HEADERS, FETCH_PDF_TEXT,
)


# ───────────────────────────────────────────────────────────────
# Вспомогательные функции
# ───────────────────────────────────────────────────────────────

def _article_slug(url: str) -> str:
    """
    Извлекает slug (строковый ID) статьи из её URL.

    Пример:
        'https://cyberleninka.ru/article/n/kvantovaya-fizika'
        → 'kvantovaya-fizika'

    Если URL не соответствует ожидаемому паттерну,
    возвращает исходный URL целиком.
    """
    m = re.search(r"/article/n/(.+?)/?$", url)
    return m.group(1) if m else url


# ───────────────────────────────────────────────────────────────
# Работа с поисковым API
# ───────────────────────────────────────────────────────────────

def search_page(
    query: str,
    size: int,
    offset: int,
    filters: Optional[List[int]] = None,
) -> dict:
    """
    Выполняет один POST-запрос к поисковому API CyberLeninka.

    API endpoint: POST https://cyberleninka.ru/api/search
    Тело запроса — JSON:
        {
            "mode": "articles",
            "size": <int>,          # статей на страницу
            "q": <str>,             # поисковый запрос
            "from": <int>,          # смещение (для пагинации)
            "catalogs": [<int>, …]  # необязательно, фильтры
        }

    Фильтры (catalogs):
        22 — РИНЦ
        8  — ВАК
        2  — Scopus

    Возвращает:
        dict с полями 'found' (общее число найденных) и
        'articles' (список словарей с ключом 'link').
        При ошибке сети возвращает пустой dict.
    """
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
            API_URL,
            data=json.dumps(body),
            headers=HEADERS,
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        print(f"  [!] Ошибка API (offset={offset}): {exc}")
        return {}


def collect_links(
    query: str,
    max_pages: int,
    filters: Optional[List[int]] = None,
) -> List[str]:
    """
    Собирает абсолютные URL статей по всем страницам поиска.

    Алгоритм:
        1. Запрашивает первую страницу, читает 'found' — общее число.
        2. Итерирует страницы (смещение += ARTICLES_PER_PAGE) пока не
           достигнет max_pages или не кончатся результаты.

    Аргументы:
        query     — поисковый запрос.
        max_pages — максимальное число страниц API (10 статей/страница).
        filters   — список кодов фильтров [22], [8], [2] или None.

    Возвращает:
        Список URL вида 'https://cyberleninka.ru/article/n/...'.
    """
    links: List[str] = []

    first = search_page(query, ARTICLES_PER_PAGE, 0, filters)
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
        result = search_page(
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


# ───────────────────────────────────────────────────────────────
# Извлечение текста из PDF
# ───────────────────────────────────────────────────────────────

def extract_pdf_text(article_url: str) -> str:
    """
    Скачивает PDF статьи и извлекает текст через pdfminer.six.

    URL PDF: <article_url>/pdf

    Зависимость: pdfminer.six (pip install pdfminer.six).
    Если библиотека не установлена — возвращает ''.

    Примечание:
        PDF-текст CyberLeninka часто содержит артефакты OCR:
        перенесённые слова (кото-\nрый), отсутствующие пробелы,
        «мусорные» строки из колонтитулов. Требует дополнительной
        очистки через cleaner.clean_text().

    Возвращает:
        Извлечённый текст или пустую строку при любой ошибке.
    """
    try:
        from pdfminer.high_level import extract_text as pdf_extract  # lazy import
    except ImportError:
        print("  [!] pdfminer.six не установлен. Установите: pip install pdfminer.six")
        return ""

    try:
        resp = requests.get(
            article_url + "/pdf",
            headers=HEADERS,
            stream=True,
            timeout=30,
        )
        resp.raise_for_status()
        pdf_bytes = io.BytesIO(resp.content)
        text = pdf_extract(pdf_bytes)
        return text.strip() if text else ""
    except Exception as exc:
        print(f"  [!] Ошибка PDF ({article_url}): {exc}")
        return ""


# ───────────────────────────────────────────────────────────────
# Парсинг HTML-страницы статьи
# ───────────────────────────────────────────────────────────────

def parse_article_page(url: str, fetch_pdf: bool = FETCH_PDF_TEXT) -> dict:
    """
    Скрапит HTML-страницу статьи и возвращает словарь с данными.

    Источники данных (в порядке приоритета):
        1. HTML-страница: заголовок, авторы, год, журнал,
           метки РИНЦ/ВАК/Scopus, аннотация, ключевые слова,
           OCR-текст (div.ocr, div[itemprop="articleBody"] и др.).
        2. PDF (если fetch_pdf=True и HTML-текст не найден):
           полный текст через pdfminer.

    Возвращаемый словарь:
        article_id    — slug из URL (уникальный строковый ключ).
        url           — полный URL страницы.
        title         — заголовок статьи.
        authors       — строка с именами авторов через запятую.
        authors_count — количество авторов.
        year          — год публикации (int или None).
        journal       — название журнала.
        annotation    — аннотация (abstract) статьи.
        keywords      — ключевые слова строкой.
        keywords_count— количество ключевых слов.
        rsci          — True если статья входит в РИНЦ.
        vak           — True если журнал входит в перечень ВАК.
        scopus        — True если журнал входит в Scopus.
        text          — полный текст статьи (может быть пустым).
        text_length   — длина text в символах.
        text_source   — 'html_ocr' | 'pdf' | '' (пусто = текст не найден).

    При сетевой ошибке возвращает запись с пустыми полями
    и логирует ошибку в stdout.
    """
    record: dict = {
        "article_id":     _article_slug(url),
        "url":            url,
        "title":          "",
        "authors":        "",
        "authors_count":  0,
        "year":           None,
        "journal":        "",
        "annotation":     "",
        "keywords":       "",
        "keywords_count": 0,
        "rsci":           False,
        "vak":            False,
        "scopus":         False,
        "text":           "",
        "text_length":    0,
        "text_source":    "",
    }

    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"    [!] Ошибка страницы ({url}): {exc}")
        return record

    soup = bs4.BeautifulSoup(resp.text, "html.parser")

    # --- Заголовок ---
    # CyberLeninka помещает заголовок в первый тег <i> на странице.
    title_tag = soup.find("i")
    if title_tag:
        record["title"] = title_tag.get_text(strip=True)

    # --- Авторы ---
    # Структура: <h2 class="right-title"><span>Журнал — Автор1, Автор2</span></h2>
    # Авторы идут после первого тире «—».
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
    # Структура: <div class="labels"><time>2023</time>
    #             <div class="label rsci">…</div>
    #             <div class="label vak">…</div> …</div>
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
    # Первая ссылка с классом 'link' обычно ведёт на страницу журнала.
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
        # Фолбэк: параграф, начинающийся с «Ключевые слова»
        for elem in soup.find_all(["p", "div"]):
            txt = elem.get_text(" ", strip=True)
            if txt.startswith("Ключевые слова"):
                kw_text = re.sub(r"Ключевые\s*слова[:：]?\s*", "", txt).strip()
                record["keywords"]       = kw_text
                record["keywords_count"] = len(
                    [k for k in kw_text.split(",") if k.strip()]
                )
                break

    # --- Полный текст статьи (OCR из HTML) ---
    # CyberLeninka размещает OCR-текст в специальных блоках.
    # Минимальный порог 300 символов защищает от пустых/служебных блоков.
    html_text = ""
    for name, attrs in [
        ("div",     {"class": "ocr"}),
        ("div",     {"itemprop": "articleBody"}),
        ("section", {"class": "body"}),
        ("div",     {"id": "body"}),
        ("article", {}),
    ]:
        tag = soup.find(name, attrs)
        if tag:
            candidate = tag.get_text(" ", strip=True)
            if len(candidate) > 300:
                html_text = candidate
                break

    if html_text:
        record["text"]        = html_text
        record["text_length"] = len(html_text)
        record["text_source"] = "html_ocr"
    elif fetch_pdf:
        pdf_text = extract_pdf_text(url)
        if pdf_text:
            record["text"]        = pdf_text
            record["text_length"] = len(pdf_text)
            record["text_source"] = "pdf"

    return record


# ───────────────────────────────────────────────────────────────
# Выгрузка по теме / по всем темам
# ───────────────────────────────────────────────────────────────

def fetch_topic(
    topic: str,
    max_pages: int,
    filter_catalog: Optional[int] = None,
    fetch_pdf: bool = FETCH_PDF_TEXT,
    delay: float = DELAY,
) -> pd.DataFrame:
    """
    Выгружает статьи по одной теме и возвращает DataFrame.

    Добавляет колонку 'topic' со значением темы в каждую строку.

    Аргументы:
        topic          — поисковый запрос (тема).
        max_pages      — максимум страниц API (10 статей/страница).
        filter_catalog — код фильтра (22/8/2) или None.
        fetch_pdf      — загружать ли PDF при отсутствии HTML-текста.
        delay          — задержка между HTTP-запросами (секунды).

    Возвращает:
        pd.DataFrame с колонками из parse_article_page() + 'topic'.
    """
    filters = [filter_catalog] if filter_catalog else None
    filter_names = {22: "РИНЦ", 8: "ВАК", 2: "Scopus"}

    print(f"\n{'=' * 60}")
    print(f"Тема:        {topic}")
    print(f"До статей:   {max_pages * ARTICLES_PER_PAGE}")
    if filter_catalog:
        print(f"Фильтр:      {filter_names.get(filter_catalog, filter_catalog)}")
    print("=" * 60)

    links = collect_links(topic, max_pages, filters)
    print(f"  Ссылок собрано: {len(links)}\n")

    records = []
    for i, url in enumerate(links, 1):
        print(f"  [{i:>4}/{len(links)}] {url}")
        rec = parse_article_page(url, fetch_pdf=fetch_pdf)
        rec["topic"] = topic
        records.append(rec)
        time.sleep(delay)

    return pd.DataFrame(records)


def fetch_all_topics(
    topics: List[str],
    max_pages: int,
    filter_catalog: Optional[int] = None,
    fetch_pdf: bool = FETCH_PDF_TEXT,
    delay: float = DELAY,
) -> pd.DataFrame:
    """
    Обходит список тем, объединяет DataFrame-ы в один.

    После объединения:
        - Добавляет сквозную числовую колонку 'id' (начиная с 1).
        - Приводит числовые колонки к корректным типам.

    Колонки итогового DataFrame:
        id, article_id, url, title, authors, authors_count,
        year, journal, annotation, keywords, keywords_count,
        rsci, vak, scopus, text, text_length, text_source, topic.

    Аргументы:
        topics         — список поисковых запросов.
        max_pages      — максимум страниц API на тему.
        filter_catalog — единый фильтр для всех тем (или None).
        fetch_pdf      — загружать ли PDF.
        delay          — задержка между запросами.

    Возвращает:
        pd.DataFrame со всеми статьями по всем темам.
    """
    frames: List[pd.DataFrame] = []
    for topic in topics:
        df_t = fetch_topic(topic, max_pages, filter_catalog, fetch_pdf, delay)
        frames.append(df_t)

    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "id", range(1, len(df) + 1))

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for col in ("authors_count", "keywords_count", "text_length"):
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(0)
                .astype(int)
            )

    return df

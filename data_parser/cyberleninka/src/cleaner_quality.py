"""
Эвристики шумности текста, построчная мета-очистка (колонтитулы, ISSN, abstract).

Отделено от cleaner.py. Паттерны ограничены по длине — без жадного DOTALL на весь документ.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

# ── Построчные маркеры метаданных (начало строки, после strip) ─────────
_FM_MAX_LINE_LEN = 240

_RE_ISSN_LINE = re.compile(
    r"^(?:ISSN|ИССН|E-ISSN)\s*[\d\-–—X]{6,}",
    re.IGNORECASE,
)
_RE_ISSN_FULL_LINE = re.compile(
    r"^(?:ISSN|ИССН|E-ISSN)\s*[\d\-–—X]{6,}\s*$",
    re.IGNORECASE,
)
_RE_EDITORIAL_LINE = re.compile(
    r"^(?:Главный\s+редактор|Редакционн(?:ый|ая)\s+(?:совет|коллегия)|"
    r"Редакция|Адрес\s+редакции|Издатель(?:ство)?|Постоянный\s+адрес\s+статьи)\b",
    re.IGNORECASE,
)
_RE_COPYRIGHT_LINE = re.compile(
    r"^©\s*.+$",
    re.IGNORECASE,
)
_RE_ANNOTATION_HEADER = re.compile(
    r"^(?:Аннотация|Abstract|Ключевые\s+слова|Keywords?|Key\s+words)\s*:?\s*$",
    re.IGNORECASE,
)
_RE_METADATA_NOISE = re.compile(
    r"^(?:УДК|ББК|UDC|BBK)\s+[0-9./\[\]\s\-–—]+$",
    re.IGNORECASE,
)
# Расширенный front matter (Кейс A)
_RE_FM_ADDRESS_SOURCE = re.compile(
    r"^(?:Адрес\s+(?:статьи|журнала)|Источник)\s*[:\.]",
    re.IGNORECASE,
)
_RE_FM_PUBLICATION_META = re.compile(
    r"^(?:Содержание\s+данного\s+номера|"
    r"Дата\s+поступления\s+рукописи|"
    r"Информация\s+о\s+возможности\s+публикации)\b",
    re.IGNORECASE,
)
_RE_FM_DOI_LINE = re.compile(
    r"^DOI\s*[:\s]?\s*10\.\d{4,}/\S+",
    re.IGNORECASE,
)
_RE_FM_AFFILIATION_HINT = re.compile(
    r"^(?:ORCID|http://orcid\.org|https://orcid\.org)\b",
    re.IGNORECASE,
)

# Известные шаблоны колонтитулов / шапок (можно резать при 2+ повторах)
_RE_KNOWN_BOILERPLATE = re.compile(
    r"(?:"
    r"РЕФЕРАТИВНЫЙ\s+ЖУРНАЛ|"
    r"РОССИЙСКАЯ\s+АКАДЕМИЯ\s+НАУК|"
    r"IZVESTIYA\s+VUZOV|"
    r"НОВЫЕ\s+ИССЛЕДОВАНИЯ\s*>|"
    r"Студенческий\s+научный\s+электронный\s+журнал|"
    r"электронный\s+научный\s+журнал\s+www\.|"
    r"ISSN\s+[\d\-–—X]{6,}\s+[A-Za-zА-Яа-яЁё]"
    r")",
    re.IGNORECASE,
)

# Веб-вставки портала
_RE_PORTAL_PROMPT_RU = re.compile(
    r"Не\s+можете\s+найти\s+то,\s*что\s+вам\s+нужно\?[^\n]{0,240}",
    re.IGNORECASE,
)

_RE_PAGE_NUM_ONLY = re.compile(r"^\d{1,4}$")
# Каталожные / колоночные остатки (агрессивный режим)
_RE_CATALOG_LINE = re.compile(
    r"^\s*-?\d+\s*\.\s*\d+\s*$|^\s*\d{2}\.\d{2}\.\d{2,4}\.?\s*$|^\s*\d{3}\s*$",
)

_RE_OCR_CATALOG_JUNK = re.compile(
    r"\b(?:2Q{2,}\w*|[A-Z]?\d*Q\d+(?:\.\d+){1,4})\b",
    re.IGNORECASE,
)
_RE_DIGIT_LETTER_MIX = re.compile(
    r"\b\d+[A-Za-zА-Яа-яЁё]{1,3}\b|\b[A-Za-zА-Яа-яЁё]{1,3}\d+\b",
)

# Медиа-подписи (новости HTML) — только при явном флаге в cleaner
_RE_MEDIA_CAPTION_LINE = re.compile(
    r"^(?:Фото|Кадр|Видео|Снимок|Иллюстрация)\s*:\s*.{0,400}$",
    re.IGNORECASE | re.MULTILINE,
)


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def is_metadata_or_boilerplate_line(s: str) -> bool:
    """Строка похожа на метаданные или известный boilerplate (для body-start и подсчётов)."""
    st = s.strip()
    if not st or len(st) > _FM_MAX_LINE_LEN:
        return False
    if _RE_ISSN_LINE.match(st):
        return True
    if _RE_EDITORIAL_LINE.match(st):
        return True
    if _RE_COPYRIGHT_LINE.match(st):
        return True
    if _RE_ANNOTATION_HEADER.match(st):
        return True
    if _RE_METADATA_NOISE.match(st):
        return True
    if _RE_FM_ADDRESS_SOURCE.match(st):
        return True
    if _RE_FM_PUBLICATION_META.match(st):
        return True
    if _RE_FM_DOI_LINE.match(st):
        return True
    if _RE_FM_AFFILIATION_HINT.match(st):
        return True
    if _RE_KNOWN_BOILERPLATE.search(st):
        return True
    return False


def remove_web_portal_prompts(text: str) -> str:
    return _RE_PORTAL_PROMPT_RU.sub(" ", text)


def remove_media_caption_lines(text: str) -> str:
    """Удаляет строки «Фото: …», «Кадр: …» и т.п. (целиком строка, до 400 симв.)."""
    lines = text.split("\n")
    out: list[str] = []
    for line in lines:
        if _RE_MEDIA_CAPTION_LINE.match(line.strip()):
            continue
        out.append(line)
    return "\n".join(out)


def remove_standalone_page_number_lines(
    text: str,
    *,
    aggressive: bool = False,
) -> str:
    """
    Убирает строки из одних цифр (1–4) и при aggressive — каталожные «-2.6», «96.01.022.», «240».
    """
    lines = text.split("\n")
    out: list[str] = []
    for line in lines:
        st = line.strip()
        if not st:
            out.append(line)
            continue
        if _RE_PAGE_NUM_ONLY.match(st):
            continue
        if aggressive and _RE_CATALOG_LINE.match(st):
            continue
        out.append(line)
    return "\n".join(out)


def remove_repeated_boilerplate_lines(
    text: str,
    *,
    min_repeats: int = 4,
    min_len: int = 18,
    max_len: int = 220,
    known_boilerplate_min_repeats: int = 2,
) -> tuple[str, int]:
    """
    Повторы >= min_repeats в окне длины; известные шаблоны — >= known_boilerplate_min_repeats.
    """
    lines = text.split("\n")
    if len(lines) < 2:
        return text, 0

    stripped = [ln.strip() for ln in lines]
    counts = Counter(s for s in stripped if s)

    bad_norms: set[str] = set()
    for s, c in counts.items():
        if not (min_len <= len(s) <= max_len):
            continue
        if c >= min_repeats:
            bad_norms.add(s)
        elif c >= known_boilerplate_min_repeats and _RE_KNOWN_BOILERPLATE.search(s):
            bad_norms.add(s)

    removed = 0
    out: list[str] = []
    for ln, st in zip(lines, stripped, strict=False):
        if st in bad_norms:
            removed += 1
            continue
        out.append(ln)
    return "\n".join(out), removed


def remove_front_matter_lines(
    text: str,
    *,
    max_scan_lines: int = 100,
) -> tuple[str, int]:
    """Первые max_scan_lines: метаданные и короткие служебные строки."""
    lines = text.split("\n")
    if not lines:
        return text, 0

    removed = 0
    head = lines[:max_scan_lines]
    tail = lines[max_scan_lines:]

    def is_fm_line(raw: str) -> bool:
        s = raw.strip()
        if not s:
            return False
        if len(s) > _FM_MAX_LINE_LEN:
            return False
        return is_metadata_or_boilerplate_line(s)

    new_head: list[str] = []
    for ln in head:
        if is_fm_line(ln):
            removed += 1
            continue
        new_head.append(ln)
    return "\n".join(new_head + tail), removed


def _body_line_score(line: str) -> float:
    st = line.strip()
    if len(st) < 42:
        return 0.0
    if is_metadata_or_boilerplate_line(st):
        return 0.0
    words = len(st.split())
    if words < 7:
        return 0.0
    letters = sum(1 for c in st if c.isalpha())
    if letters < 28:
        return 0.0
    cyr = sum(1 for c in st if "\u0400" <= c <= "\u04ff")
    lat = sum(1 for c in st if ("a" <= c <= "z") or ("A" <= c <= "Z"))
    if cyr + lat < 22:
        return 0.0
    cyr_ratio = cyr / max(letters, 1)
    score = min(1.0, len(st) / 130.0) * min(1.0, words / 16.0)
    if cyr_ratio >= 0.28:
        score *= 1.25
    if lat > cyr * 3 and cyr < 8:
        score *= 0.35
    return min(1.0, score)


def trim_document_head(
    text: str,
    *,
    max_scan_lines: int = 110,
    min_original_len: int = 850,
    min_keep_ratio: float = 0.20,
) -> tuple[str, int, float]:
    """
    Ищет начало связного «тела» статьи; обрезает строки до него при достаточной уверенности.

    Возвращает (новый_текст, число_отрезанных_строк_сверху, confidence 0..1).
    Безопасность: короткие тексты не трогаем; не оставляем < min_keep_ratio исходной длины.
    """
    if len(text) < min_original_len:
        return text, 0, 0.0

    lines = text.split("\n")
    if len(lines) < 10:
        return text, 0, 0.0

    lim = min(max_scan_lines, len(lines))
    start_idx: int | None = None
    confidence = 0.0

    for i in range(lim):
        s1 = _body_line_score(lines[i])
        s2 = _body_line_score(lines[i + 1]) if i + 1 < len(lines) else 0.0
        if i + 1 < len(lines) and s1 >= 0.52 and s2 >= 0.38:
            start_idx = i
            confidence = 0.78 if i <= 25 else 0.62
            break
        if s1 >= 0.82 and len(lines[i].strip()) >= 100:
            start_idx = i
            confidence = 0.55
            break

    if start_idx is None or start_idx == 0:
        return text, 0, 0.0

    new_text = "\n".join(lines[start_idx:])
    if len(new_text.strip()) < len(text) * min_keep_ratio:
        return text, 0, 0.0

    return new_text, start_idx, confidence


def _latin_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    lat = sum(1 for c in letters if "a" <= c.lower() <= "z")
    return lat / len(letters)


def _strip_one_english_block(
    text: str,
    chunk_start: int,
    max_block_chars: int,
) -> tuple[str, int] | None:
    """Вырезать один блок от заголовка EN-метаданных. Возвращает (новый_текст, delta_len) или None."""
    chunk = text[chunk_start:]
    m = re.search(
        r"(?:^|\n)(Abstract|Summary|Annotation)\s*[:\.]?\s*\n",
        chunk,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not m:
        m = re.search(
            r"\n(Abstract|Summary|Annotation)\s*[:\.]\s+[A-Za-z]",
            chunk,
            flags=re.IGNORECASE,
        )
    if not m:
        m = re.search(
            r"(?:^|\n)Key\s*words?\s*[:\.]\s*\S",
            chunk,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    if not m:
        return None

    abs_start = chunk_start + m.start()
    rest = text[abs_start:]
    cut = min(max_block_chars, len(rest))
    end_rel = cut

    boundary = re.search(
        r"\n\n(?=[ \t]*[А-ЯЁа-яё][^\n]{28,})",
        rest[:cut],
    )
    if boundary:
        end_rel = boundary.start()
    else:
        boundary2 = re.search(
            r"\n(?=[А-ЯЁ][а-яё]{4,}\s)",
            rest[80:cut],
        )
        if boundary2:
            end_rel = 80 + boundary2.start()

    block = rest[:end_rel]
    cyr = sum(1 for c in block if "\u0400" <= c <= "\u04ff")
    lat = sum(1 for c in block if "a" <= c.lower() <= "z")
    if cyr > 95 and cyr > lat * 0.35:
        return None
    if _latin_ratio(block) < 0.45 and len(block) > 120:
        return None

    new_text = text[:abs_start] + rest[end_rel:]
    return new_text, len(text) - len(new_text)


def strip_english_abstract_block(
    text: str,
    *,
    max_block_chars: int = 4000,
    min_match_pos: int = 20,
    language: str | None = "ru",
    max_passes: int = 3,
) -> tuple[str, bool]:
    """
    Удаляет локализованные англоязычные Abstract / Summary / Key words (для ru-корпуса).

    language: только если начинается с «ru» или None (эвристика по латинице в блоке).
    min_match_pos: не искать блок раньше (типично короткий русский ввод перед Abstract).
    """
    lang = (language or "").strip().lower()
    if lang and not lang.startswith("ru"):
        return text, False

    t = text
    any_cut = False
    pos_floor = min_match_pos
    for _ in range(max_passes):
        if len(t) < pos_floor + 30:
            break
        res = _strip_one_english_block(t, pos_floor, max_block_chars)
        if res is None:
            break
        t, _d = res
        any_cut = True
        pos_floor = min_match_pos

    return t, any_cut


def strip_leading_duplicate_prefix(text: str, prefix: str | None) -> tuple[str, bool]:
    if not prefix or not text:
        return text, False
    p = _nfc(prefix.strip())
    if len(p) < 12:
        return text, False
    t = text.lstrip()
    if t.lower().startswith(p.lower()):
        rest = text[len(text) - len(t) + len(p) :].lstrip()
        if rest.startswith("\n"):
            rest = rest[1:].lstrip()
        return rest, True
    return text, False


def count_front_matter_hits_in_head(text: str, *, max_lines: int = 80) -> int:
    lines = text.split("\n")[:max_lines]
    return sum(1 for ln in lines if is_metadata_or_boilerplate_line(ln.strip()))


def detect_noise_patterns(text: str) -> dict[str, Any]:
    if not text:
        return _empty_noise()

    ocr_cat = len(_RE_OCR_CATALOG_JUNK.findall(text))
    dlm = len(_RE_DIGIT_LETTER_MIX.findall(text))

    lines = text.split("\n")
    stripped = [ln.strip() for ln in lines if ln.strip()]
    counts = Counter(stripped)
    repeated_kinds = sum(1 for s, c in counts.items() if c >= 4 and 18 <= len(s) <= 220)
    repeated_kinds_soft = sum(1 for s, c in counts.items() if c >= 2 and 18 <= len(s) <= 220)

    page_like = sum(1 for s in stripped if _RE_PAGE_NUM_ONLY.match(s))
    catalog_like = sum(1 for s in stripped if _RE_CATALOG_LINE.match(s))
    issn_like = sum(1 for s in stripped if _RE_ISSN_FULL_LINE.match(s))

    suspicious = 0
    for s in stripped[:120]:
        if _RE_EDITORIAL_LINE.match(s) or _RE_COPYRIGHT_LINE.match(s):
            suspicious += 1

    eng_abs = 1 if re.search(r"(?:^|\n)Abstract\s*[:\.]?\s*\n", text, re.I | re.M) else 0
    reflective = sum(1 for s in stripped[:60] if _RE_KNOWN_BOILERPLATE.search(s))
    fm_hits = count_front_matter_hits_in_head(text, max_lines=80)

    return {
        "ocr_catalog_like_tokens": ocr_cat,
        "digit_letter_mixed_tokens": dlm,
        "suspicious_lines": suspicious,
        "repeated_line_kinds": repeated_kinds,
        "repeated_line_kinds_soft": repeated_kinds_soft,
        "standalone_page_like_lines": page_like,
        "catalog_like_lines": catalog_like,
        "issn_like_lines": issn_like,
        "english_abstract_marker": eng_abs,
        "reflective_journal_markers": reflective,
        "front_matter_line_hits_head": fm_hits,
    }


def _empty_noise() -> dict[str, Any]:
    return {
        "ocr_catalog_like_tokens": 0,
        "digit_letter_mixed_tokens": 0,
        "suspicious_lines": 0,
        "repeated_line_kinds": 0,
        "repeated_line_kinds_soft": 0,
        "standalone_page_like_lines": 0,
        "catalog_like_lines": 0,
        "issn_like_lines": 0,
        "english_abstract_marker": 0,
        "reflective_journal_markers": 0,
        "front_matter_line_hits_head": 0,
    }


def _score_document_type(text: str, noise: dict[str, Any], n: int) -> tuple[str, float, float]:
    """
    probable_document_type, article_like_score, bibliographic_like_score (0..1).
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    n_lines = max(1, len(lines))
    avg_len = sum(len(x) for x in lines[: min(40, len(lines))]) / min(40, max(len(lines), 1))

    biblio = 0.0
    if noise["reflective_journal_markers"] >= 2:
        biblio += 0.35
    if noise["ocr_catalog_like_tokens"] > 15:
        biblio += 0.25
    if noise["catalog_like_lines"] > 5:
        biblio += 0.2
    if noise["front_matter_line_hits_head"] > 18 and n > 1500:
        biblio += 0.15
    biblio = min(1.0, biblio)

    article = 0.0
    if avg_len > 55 and noise["repeated_line_kinds"] < 6:
        article += 0.3
    if n > 2500 and noise["front_matter_line_hits_head"] < 25:
        article += 0.25
    cyr = sum(1 for c in text if "\u0400" <= c <= "\u04ff")
    if cyr > 400:
        article += 0.25
    if noise["english_abstract_marker"] and cyr > 300:
        article += 0.1
    article = min(1.0, article)

    noisy = 0.0
    if noise["repeated_line_kinds"] > 5 or noise["standalone_page_like_lines"] > 6:
        noisy += 0.35
    if noise["ocr_catalog_like_tokens"] > 25:
        noisy += 0.3
    if noise["digit_letter_mixed_tokens"] > 40:
        noisy += 0.2
    noisy = min(1.0, noisy)

    html_news = 0.0
    if n < 4500 and _RE_MEDIA_CAPTION_LINE.search(text):
        html_news += 0.25
    if n < 3500 and avg_len < 42 and n_lines > 8:
        html_news += 0.2
    html_news = min(1.0, html_news)

    scores = {
        "bibliographic_review_like": biblio,
        "article_like": article,
        "noisy_ocr_like": noisy,
        "html_news_like": html_news,
    }
    best = max(scores, key=lambda k: scores[k])
    if scores[best] < 0.28:
        doc_type = "mixed_unknown"
    elif best == "bibliographic_review_like" and biblio >= article + 0.12:
        doc_type = "bibliographic_review_like"
    elif best == "noisy_ocr_like" and noisy >= article:
        doc_type = "noisy_ocr_like"
    elif best == "html_news_like" and html_news >= 0.35:
        doc_type = "html_news_like"
    elif article >= 0.35:
        doc_type = "article_like"
    else:
        doc_type = "mixed_unknown"

    return doc_type, round(article, 4), round(biblio, 4)


def analyze_text_quality(
    text: str,
    *,
    artifacts: dict[str, Any] | None = None,
    removed_ratio_hint: float | None = None,
) -> dict[str, Any]:
    noise = detect_noise_patterns(text)
    art = artifacts or {}

    n = len(text)
    n_lines = max(1, text.count("\n") + 1)
    non_ws = sum(1 for c in text if not c.isspace())
    weird = sum(
        1
        for c in text
        if unicodedata.category(c) in ("So", "Sk", "Sm") and c not in "«»—–-"
    )
    suspicious_symbol_ratio = (weird / non_ws) if non_ws else 0.0

    fm_density = min(1.0, noise["front_matter_line_hits_head"] / 45.0)
    rb_density = min(1.0, noise["repeated_line_kinds"] / 12.0)
    en_density = 0.0
    sample = text[: min(4000, n)]
    lat = sum(1 for c in sample if "a" <= c.lower() <= "z")
    letters = sum(1 for c in sample if c.isalpha())
    if letters:
        en_density = min(1.0, (lat / letters) * (1.2 if noise["english_abstract_marker"] else 0.7))

    meta_density = min(
        1.0,
        (noise["issn_like_lines"] * 0.15 + noise["suspicious_lines"] * 0.08 + fm_density * 0.5),
    )

    biblio = int(art.get("bibliography", 0))
    urls = int(art.get("urls", 0))
    emails = int(art.get("emails", 0))

    page_noise = int(noise["standalone_page_like_lines"] > 2 or noise["repeated_line_kinds"] > 2)
    metadata_noise = int(noise["issn_like_lines"] > 0 or noise["suspicious_lines"] > 1)
    english_block = int(noise["english_abstract_marker"] > 0)

    cyr_all = sum(1 for c in text if "\u0400" <= c <= "\u04ff")
    lat_all = sum(1 for c in text if "a" <= c.lower() <= "z")
    probable_language_mismatch = False
    if n > 800 and english_block and cyr_all > 200 and lat_all > cyr_all * 0.4:
        probable_language_mismatch = True

    doc_type, article_like_score, bibliographic_like_score = _score_document_type(text, noise, n)

    pikabu_ph = detect_portal_chrome_hits(text, "pikabu_strong", head_lines=52, tail_lines=78)
    portal_chrome_heavy = pikabu_ph >= 10 and n > 550
    if portal_chrome_heavy and pikabu_ph >= 14:
        doc_type = "portal_noise_heavy"
    elif portal_chrome_heavy and pikabu_ph >= 11 and doc_type in ("mixed_unknown", "html_news_like"):
        doc_type = "portal_noise_heavy"

    score = 100.0
    score -= min(25, noise["ocr_catalog_like_tokens"] * 2)
    score -= min(15, noise["digit_letter_mixed_tokens"])
    score -= min(15, noise["repeated_line_kinds"] * 3)
    score -= min(10, noise["standalone_page_like_lines"])
    score -= min(10, suspicious_symbol_ratio * 200)
    score -= biblio * 3
    score -= min(10, urls // 2)
    score -= min(8, emails * 2)
    score -= fm_density * 18
    score -= rb_density * 12
    if probable_language_mismatch:
        score -= 12
    if doc_type == "bibliographic_review_like":
        score -= 10
    elif doc_type == "noisy_ocr_like":
        score -= 8
    if doc_type == "portal_noise_heavy":
        score -= 14
    score = max(0.0, min(100.0, score))

    coherent_len = len(text.strip())
    portal_hits = max(
        detect_portal_chrome_hits(text, "pravda"),
        detect_portal_chrome_hits(text, "pravda_ua", head_lines=52, tail_lines=62),
        detect_portal_chrome_hits(text, "pikabu_strong"),
        detect_portal_chrome_hits(text, "mail_ru"),
    )
    next_bleed = detect_next_article_bleed(text)
    portal_chrome_detected = portal_hits >= 2

    quarantine = (
        score < 42
        or noise["ocr_catalog_like_tokens"] > 40
        or suspicious_symbol_ratio > 0.085
        or (noise["repeated_line_kinds"] > 8 and n > 500)
        or (doc_type == "bibliographic_review_like" and bibliographic_like_score > 0.72)
        or (coherent_len < 380 and n > 1200)
        or (portal_hits > 14 and n < 2800)
        or (next_bleed and n > 2200 and doc_type != "article_like")
        or doc_type == "portal_noise_heavy"
        or (portal_chrome_heavy and pikabu_ph >= 16)
    )
    if removed_ratio_hint is not None and removed_ratio_hint > 0.62 and coherent_len < 500:
        quarantine = True

    return {
        **noise,
        "portal_chrome_hits": portal_hits,
        "portal_chrome_detected": portal_chrome_detected,
        "portal_chrome_heavy": bool(portal_chrome_heavy),
        "next_article_bleed_detected": next_bleed,
        "bibliography_detected": bool(biblio),
        "page_noise_detected": bool(page_noise),
        "metadata_noise_detected": bool(metadata_noise),
        "english_block_detected": bool(english_block),
        "probable_language_mismatch": probable_language_mismatch,
        "suspicious_symbol_ratio": round(suspicious_symbol_ratio, 5),
        "front_matter_density": round(fm_density, 4),
        "repeated_boilerplate_density": round(rb_density, 4),
        "english_block_density": round(en_density, 4),
        "metadata_noise_density": round(meta_density, 4),
        "article_like_score": article_like_score,
        "bibliographic_like_score": bibliographic_like_score,
        "probable_document_type": doc_type,
        "quality_score": round(score, 2),
        "quarantine_candidate": bool(quarantine),
    }


# ── v3: границы статьи, хвост, portal chrome (построчно, без DOTALL на весь текст) ──

_RE_TAIL_SECTION_HEADER = re.compile(
    r"^(?:"
    r"Список\s+(?:литературы|источников?|использованн\S+\s+литературы)"
    r"|Библиографический\s+список"
    r"|Библиография"
    r"|Литература\s*$"
    r"|Литература\s+\d"
    r"|References?\s*$"
    r"|Bibliography\s*$"
    r"|Сведения\s+об\s+авторах?"
    r"|Information\s+about\s+(?:the\s+)?authors?"
    r"|About\s+(?:the\s+)?authors?"
    r"|Для\s+цитирования"
    r"|How\s+to\s+cite(?:\s+this\s+article)?\s*$"
    r"|Статья\s+поступила"
    r"|Принята\s+к\s+публикации"
    r"|Annotation\s*$"
    r"|Abstract\s*$"
    r"|Keywords?\s*$"
    r"|Key\s+words\s*$"
    r")",
    re.IGNORECASE,
)

_RE_TAIL_NEXT_ARTICLE_HINT = re.compile(
    r"^(?:УДК|ББК|UDC|BBK)\s+[\d./\[\]\s\-–—]+$"
    r"|^DOI\s*[:\s]?\s*10\.\d{4,}/\S+$"
    r"|^(?:ISSN|ИССН|E-ISSN)\s*[\d\-–—X]{6,}\s*$",
    re.IGNORECASE,
)

_RE_TAIL_CAPS_TITLE = re.compile(
    r"^[А-ЯЁA-Z][А-ЯЁA-Z0-9\s\-–—:,.«»\"]{14,140}$",
)


def is_tail_section_header_line(s: str) -> bool:
    st = s.strip()
    if not st or len(st) > _FM_MAX_LINE_LEN:
        return False
    return bool(_RE_TAIL_SECTION_HEADER.match(st))


def detect_probable_article_start(
    text: str,
    *,
    max_scan_lines: int = 130,
    min_text_len: int = 650,
) -> tuple[int | None, float]:
    """
    Индекс строки, с которой вероятно начинается тело статьи (после хвоста предыдущей), и уверенность 0..1.
    """
    if len(text) < min_text_len:
        return None, 0.0
    lines = text.split("\n")
    if len(lines) < 14:
        return None, 0.0
    head = lines[: min(max_scan_lines, len(lines))]
    if not head:
        return None, 0.0
    short = sum(1 for L in head if len(L.strip()) < 36)
    thin_ratio = short / max(len(head), 1)
    conf_boost = 0.15 if thin_ratio >= 0.42 else 0.0

    lim = min(max_scan_lines, len(lines))
    for i in range(lim):
        s1 = _body_line_score(lines[i])
        s2 = _body_line_score(lines[i + 1]) if i + 1 < len(lines) else 0.0
        if i + 1 < len(lines) and s1 >= 0.48 and s2 >= 0.35:
            return i, min(1.0, 0.62 + conf_boost)
        if s1 >= 0.78 and len(lines[i].strip()) >= 88:
            return i, min(1.0, 0.52 + conf_boost)
    return None, 0.0


def detect_probable_article_end(
    text: str,
    *,
    tail_max_lines: int = 110,
    min_lines: int = 22,
    min_pos_frac: float = 0.40,
) -> tuple[int | None, float]:
    """
    Индекс строки, с которой вероятно начинается хвост (библиография / авторы / следующая статья).
    Обрезка: оставить text[:line_index].
    """
    lines = text.split("\n")
    n = len(lines)
    if n < min_lines:
        return None, 0.0
    min_idx = max(int(n * min_pos_frac), 8)
    tail_start = max(0, n - tail_max_lines)
    for i in range(tail_start, n):
        if i < min_idx:
            continue
        st = lines[i].strip()
        if is_tail_section_header_line(st):
            return i, 0.78
        if _RE_TAIL_NEXT_ARTICLE_HINT.match(st):
            return i, 0.55
        if _RE_TAIL_CAPS_TITLE.match(st) and i >= int(n * 0.55):
            prev = lines[i - 1].strip() if i > 0 else ""
            prev2 = lines[i - 2].strip() if i > 1 else ""
            if (not prev and not prev2) or _latin_ratio("\n".join((prev, prev2))) > 0.55:
                return i, 0.42
    return None, 0.0


def trim_leading_article_bleed(
    text: str,
    *,
    min_confidence: float = 0.58,
    min_keep_ratio: float = 0.18,
) -> tuple[str, list[str]]:
    """Убирает вероятный хвост предыдущей статьи в начале (осторожно)."""
    idx, conf = detect_probable_article_start(text)
    flags: list[str] = []
    if idx is None or idx <= 0 or conf < min_confidence:
        return text, flags
    lines = text.split("\n")
    new_text = "\n".join(lines[idx:]).strip()
    if len(new_text) < len(text) * min_keep_ratio:
        return text, flags
    flags.append(f"article_start_trim:line={idx}:conf={conf:.2f}")
    return new_text, flags


def trim_tail_sections(
    text: str,
    *,
    tail_max_lines: int = 110,
    min_lines: int = 22,
    min_pos_frac: float = 0.40,
    min_keep_ratio: float = 0.12,
) -> tuple[str, list[str]]:
    """Отрезает хвост от первого надёжного маркера в tail-zone."""
    idx, conf = detect_probable_article_end(
        text, tail_max_lines=tail_max_lines, min_lines=min_lines, min_pos_frac=min_pos_frac
    )
    flags: list[str] = []
    if idx is None or conf < 0.40:
        return text, flags
    lines = text.split("\n")
    new_text = "\n".join(lines[:idx]).rstrip()
    if len(new_text) < len(text) * min_keep_ratio and len(new_text) < 120:
        return text, flags
    flags.append(f"tail_section_trim:line={idx}:conf={conf:.2f}")
    return new_text, flags


def strip_trailing_english_metadata_block(
    text: str,
    *,
    language: str | None = "ru",
    tail_window: int = 4800,
    max_block: int = 3200,
) -> tuple[str, bool]:
    """
    Англ. Abstract / Key words только в хвостовом окне (после русского основного массива).
    """
    lang = (language or "").strip().lower()
    if lang and not lang.startswith("ru"):
        return text, False
    if len(text) < 400:
        return text, False
    win_start = max(0, len(text) - tail_window)
    win = text[win_start:]
    m = re.search(
        r"(?:^|\n)(Abstract|Summary|Annotation|Key\s*words?)\s*[:\.]?\s*\n",
        win,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not m:
        m = re.search(
            r"\n(Abstract|Summary)\s*[:\.]\s+[A-Za-z]",
            win,
            flags=re.IGNORECASE,
        )
    if not m:
        return text, False
    abs_rel = win_start + m.start()
    if abs_rel < len(text) * 0.35:
        return text, False
    rest = text[abs_rel:]
    cut = min(max_block, len(rest))
    end_rel = cut
    boundary = re.search(
        r"\n\n(?=[ \t]*[А-ЯЁа-яё][^\n]{24,})",
        rest[:cut],
    )
    if boundary:
        end_rel = boundary.start()
    block = rest[:end_rel]
    if _latin_ratio(block) < 0.40 and len(block) > 80:
        return text, False
    new_t = text[:abs_rel].rstrip() + rest[end_rel:]
    return new_t, True


def trim_to_article_boundaries(
    text: str,
    *,
    trim_leading: bool = True,
    trim_trailing: bool = True,
    strip_trailing_en: bool = False,
    strip_english_language: str | None = "ru",
) -> tuple[str, list[str]]:
    """Комбинация: начало + конец + опционально англ. хвост."""
    flags: list[str] = []
    t = text
    if trim_leading:
        t, f = trim_leading_article_bleed(t)
        flags.extend(f)
    if trim_trailing:
        t, f = trim_tail_sections(t)
        flags.extend(f)
    if strip_trailing_en:
        t2, ok = strip_trailing_english_metadata_block(t, language=strip_english_language)
        if ok:
            flags.append("trailing_english_metadata_removed")
        t = t2
    return t, flags


# ── RBC: footer CTA (только хвост, source-specific) ───────────────────
_RE_RBC_FOOTER_CTA = re.compile(
    r"(?:Оставайтесь\s+на\s+связи\s+с\s+РБК|РБК\s+в\s+M[аa]х|РБК\s+в\s+Max|"
    r"мессенджер[еа]?\s+M[аa]x|мессенджер[еа]?\s+Max|скачайте\s+приложение\s+РБК)",
    re.IGNORECASE,
)


def remove_rbc_footer_cta(text: str, *, tail_lines: int = 38) -> tuple[str, list[str]]:
    """Удаляет строки с призывом «Оставайтесь на связи с РБК в Max» и аналоги (только хвост)."""
    lines = text.split("\n")
    n = len(lines)
    if n < 2:
        return text, []
    start = max(0, n - tail_lines)
    out: list[str] = []
    removed = 0
    for i, ln in enumerate(lines):
        st = ln.strip()
        if i >= start and (
            _RE_RBC_FOOTER_CTA.search(st)
            or re.search(r"rbc\.ru/(?:app|max)\b", st, re.IGNORECASE)
        ):
            removed += 1
            continue
        out.append(ln)
    flags = [f"rbc_footer_cta_removed:{removed}"] if removed else []
    return "\n".join(out), flags


# Portal chrome: только первые/последние строки документа
_RE_PORTAL_PRAVDA = re.compile(
    r"(?:"
    r"^Использование\s+материалов\s+сайта\b"
    r"|^Реклама\s+на\s+сайте\b"
    r"|^Правил[аи]\s+использования\b"
    r"|^Свидетельство\s+о\s+регистрации\b"
    r"|^Новости\s*$|^Публикации\s*$|^Колонки\s*$|^Спецпроекты\s*$|^Радио\s*$"
    r"|^Мультимедиа\s*$|^Подписка\s*$"
    r"|редакци[ия]\s*[:\.]\s*"
    r"|^©\s*\d{4}\b.*Правда"
    r")",
    re.IGNORECASE,
)

# Pravda.com.ua (Украинская правда): верхнее меню одной строкой + юридический футер
_RE_PORTAL_PRAVDA_UA_LINE = re.compile(
    r"(?:"
    r"^Реклама\s+на\s+сайте\b"
    r"|^Правила\s+использования\s+материалов\b"
    r"|^Политика\s+ИИ\b|^Гендерная\s+политика\b"
    r"|^Как\s+писать\s+для\s+УП\b|^Стажировка\b"
    r"|^Использование\s+материалов\s+сайта\b"
    r"|^Умови\s+використання\b|^Політика\s+ІІ\b"  # ua
    r"|редакци[ия]\s*[—\-]\s*газет|Редакція\s*[:\-]|^redaction@|^press@"
    r"|@pravda\.com\.ua\b"
    r"|^Тел\.?\s*[:\-]?\s*\+|Tel\.?\s*[:\-]?\s*\+380"
    r")",
    re.IGNORECASE,
)
_RE_PORTAL_PRAVDA_UA_TOP_NAV = re.compile(
    r"(?:\bНовости\s+Публикации\s+Колонки\b|"
    r"\bПубликации\s+Колонки\s+Интервью\b|"
    r"\bНовости\s+Публикации\s+Колонки\s+Интервью\b|"
    r"\bАрхів\s+новин\b|\bГоловна\s+—\s+УП\b)",
    re.IGNORECASE,
)

_RE_PORTAL_PIKABU = re.compile(
    r"(?:"
    r"Подписаться|Спасибо,\s*что\s+подписались|Лучшие\s+посты\s+недели"
    r"|Реклам[аы]\s+на\s+Пикабу|Пикабу[\s\-]*пост|Сообщество\s+авторов"
    r"|^Теги:?\s*$|^Похожие\s+посты|^Ещё\s+посты|^Комментарии\s+\d"
    r"|^Юмор\s*$|^Здоровье\s*$|^Бизнес\s*$|^Транспорт\s*$|^Наука\s*$|^Моё\s*$|^IT\s*$"
    r")",
    re.IGNORECASE,
)

_RE_PORTAL_MAIL = re.compile(
    r"(?:"
    r"^Mail\.Ru\b|@mail\.ru\s*$|Почта\s+Mail"
    r"|Пользовательское\s+соглашение"
    r"|^Реклама\s*$"
    r")",
    re.IGNORECASE,
)


def _line_is_portal_chrome(st: str, variant: str) -> bool:
    if not st or len(st) > 260:
        return False
    v = variant.lower().replace("-", "_").replace(".", "_")
    if v in ("pravda_ua", "pravda_com_ua"):
        if _RE_PORTAL_PRAVDA_UA_LINE.search(st):
            return True
        return bool(len(st) < 220 and _RE_PORTAL_PRAVDA_UA_TOP_NAV.search(st))
    if v in ("pravda", "pravda_ru"):
        return bool(_RE_PORTAL_PRAVDA.search(st))
    if v in ("pikabu", "pikabu_strong"):
        if _RE_PORTAL_PIKABU.search(st):
            return True
        if v == "pikabu_strong" or v == "pikabu":
            seps = st.count("·") + st.count(" | ") + st.count(" • ")
            if seps >= 4 and len(st) < 210 and st.count(" ") < 35:
                return True
    if v in ("mail_ru", "mail", "mailru"):
        return bool(_RE_PORTAL_MAIL.search(st))
    return False


def trim_pikabu_trailing_portal_lines(text: str, *, max_scan: int = 90) -> tuple[str, list[str]]:
    """Срезает подряд идущие в конце строки тегов/портала Pikabu (осторожно)."""
    lines = text.split("\n")
    n = len(lines)
    if n < 10:
        return text, []
    start = max(0, n - max_scan)
    j = n - 1
    dropped = 0
    while j >= start:
        st = lines[j].strip()
        if not st:
            j -= 1
            continue
        if _line_is_portal_chrome(st, "pikabu_strong"):
            dropped += 1
            j -= 1
            continue
        break
    if dropped == 0:
        return text, []
    new_lines = lines[: j + 1]
    return "\n".join(new_lines).rstrip(), [f"pikabu_trailing_trim:{dropped}"]


def remove_portal_chrome(
    text: str,
    variant: str,
    *,
    head_lines: int = 45,
    tail_lines: int = 55,
) -> tuple[str, list[str]]:
    """Удаление типовых строк меню/footer портала (только верх/низ документа)."""
    lines = text.split("\n")
    n = len(lines)
    if n < 4:
        return text, []
    head_idx = set(range(0, min(head_lines, n)))
    tail_idx = set(range(max(0, n - tail_lines), n))
    target = head_idx | tail_idx
    out: list[str] = []
    removed = 0
    for i, ln in enumerate(lines):
        st = ln.strip()
        if i in target and _line_is_portal_chrome(st, variant):
            removed += 1
            continue
        out.append(ln)
    flags = [f"portal_chrome_removed:{removed}:{variant}"] if removed else []
    return "\n".join(out), flags


def detect_portal_chrome_hits(text: str, variant: str = "pravda", *, head_lines: int = 40, tail_lines: int = 45) -> int:
    """Счётчик строк, похожих на portal chrome (для quality)."""
    lines = text.split("\n")
    n = len(lines)
    if n < 3:
        return 0
    c = 0
    for i, ln in enumerate(lines):
        if i >= head_lines and i < n - tail_lines:
            continue
        if _line_is_portal_chrome(ln.strip(), variant):
            c += 1
    return c


def detect_next_article_bleed(text: str, *, tail_frac: float = 0.38) -> bool:
    """Грубая эвристика: в хвосте документа признаки начала другой статьи."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    n = len(lines)
    if n < 18:
        return False
    start = int(n * (1.0 - tail_frac))
    hits = 0
    for st in lines[start:]:
        if _RE_TAIL_NEXT_ARTICLE_HINT.match(st):
            hits += 2
        if is_tail_section_header_line(st):
            hits += 1
    return hits >= 3


def apply_meta_clean_pass(
    text: str,
    *,
    remove_front_matter: bool = False,
    remove_page_noise: bool = False,
    remove_repeated_boilerplate: bool = False,
    remove_web_prompts: bool = False,
    remove_english_abstract: bool = False,
    strip_title_prefix: str | None = None,
    trim_head_body_start: bool = False,
    aggressive_page_noise: bool = False,
    strip_english_language: str | None = "ru",
    remove_media_captions: bool = False,
) -> tuple[str, list[str]]:
    flags: list[str] = []
    t = text

    if remove_media_captions:
        t2 = remove_media_caption_lines(t)
        if t2 != t:
            flags.append("media_caption_lines_removed")
        t = t2

    if strip_title_prefix:
        t, ok = strip_leading_duplicate_prefix(t, strip_title_prefix)
        if ok:
            flags.append("strip_title_prefix")

    if remove_web_prompts:
        t2 = remove_web_portal_prompts(t)
        if t2 != t:
            flags.append("web_prompt_removed")
        t = t2

    if trim_head_body_start:
        t2, ntrim, conf = trim_document_head(t)
        if ntrim > 0:
            flags.append(f"trim_document_head:{ntrim}:conf={conf:.2f}")
        t = t2

    if remove_front_matter:
        t, n = remove_front_matter_lines(t)
        if n:
            flags.append(f"front_matter_lines_removed:{n}")

    if remove_english_abstract:
        t, ok = strip_english_abstract_block(t, language=strip_english_language)
        if ok:
            flags.append("english_abstract_block_removed")

    if remove_repeated_boilerplate:
        t, n = remove_repeated_boilerplate_lines(t)
        if n:
            flags.append(f"repeated_boilerplate_lines_removed:{n}")

    if remove_page_noise:
        t2 = remove_standalone_page_number_lines(t, aggressive=aggressive_page_noise)
        if t2 != t:
            flags.append(
                "page_number_lines_removed"
                + (":aggressive" if aggressive_page_noise else "")
            )
        t = t2

    return t, flags

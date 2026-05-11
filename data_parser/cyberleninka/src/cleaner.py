"""
Модуль очистки текста статей CyberLeninka.

Проблемы, которые решает этот модуль:
──────────────────────────────────────
1.  BOM (U+FEFF) — часто встречается в начале OCR-текстов.
2.  Email-адреса — публикуются в блоках сведений об авторах.
    Бывают «обфусцированные»: myagkova.post@ gmail.com,
    hirulin58@mail. ru (пробел внутри домена).
3.  URL / ссылки вида «Режим доступа: http://…»
    Включая URL с пробелом после :// (артефакт OCR): «http:// www.site.ru/…»
4.  УДК / ББК / DOI строки — метаданные, не относящиеся к тексту.
5.  Блоки сведений об авторах в конце статьи (русский + английский).
6.  Раздел «Список литературы» и его варианты — удаляется полностью
    вместе со всеми пронумерованными ссылками.
7.  Фрагменты URL-путей, оставшиеся после удаления http://-части
    (например: «councils/chrstuni/…_en.html», «tuni_doc_19980316.html»).
8.  Мягкий перенос (U+00AD), неразрывный пробел (U+00A0).
9.  Символы из семейства «широких» кавычек и тире → нормализация.
10. Лишние пробелы, пустые строки, управляющие символы.
11. Повтор заголовка «Ключевые слова:» в теле текста.
12. Нумерованные сноски вида «1.\t» в начале строк.
13. OCR-перенос: слова, разорванные дефисом через перевод строки
    («слово-\nпродолжение» → «слово продолжение»).
14. Внутритекстовые ссылки-номера вида «(1)», «[2]» (опционально).
15. Опционально: front matter, колонтитулы, англ. abstract, веб-подсказки (флаги).

Стратегия: максимально сохранить содержательный текст,
удаляя только артефакты вёрстки и личные данные.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Optional

import bs4
import pandas as pd

from .cleaner_quality import (
    analyze_text_quality,
    apply_meta_clean_pass,
    detect_noise_patterns,
    remove_portal_chrome,
    remove_rbc_footer_cta as strip_rbc_footer_cta_lines,
    strip_trailing_english_metadata_block,
    trim_leading_article_bleed,
    trim_pikabu_trailing_portal_lines,
    trim_tail_sections,
)

# Публичные алиасы для «import ... cleaner as C»
__all__ = [
    "analyze_text_quality",
    "clean_dataframe",
    "clean_kwargs_for_source",
    "clean_text",
    "clean_text_with_report",
    "detect_artifacts",
    "detect_noise_patterns",
    "get_source_clean_profile",
]


# ───────────────────────────────────────────────────────────────
# Регулярные выражения (компилируются один раз при импорте)
# ───────────────────────────────────────────────────────────────

# Email: классический + с пробелом вокруг @/@
_RE_EMAIL = re.compile(
    r"[a-zA-Z0-9_.+-]+\s*@\s*[a-zA-Z0-9-]+(?:\s*\.\s*[a-zA-Z]{2,})+",
    re.IGNORECASE,
)

# URL: http/https/ftp.
# Допускаем пробел после "://" — артефакт OCR («http:// www.site.ru/…»).
_RE_URL = re.compile(
    r"https?\s*://\s*[^\s\)\]\}\"\'<>]+|ftp://[^\s]+",
    re.IGNORECASE,
)

# «Режим доступа:» — вся фраза до конца строки (URL удаляется отдельно).
# ВАЖНО: ограничиваем до 500 символов, чтобы не съесть весь текст в
# однострочных OCR-документах (где нет \n).
_RE_ACCESS_MODE = re.compile(
    r"[-–—]?\s*Режим доступа\s*:[^\n]{0,500}",
    re.IGNORECASE,
)

# «Дата обращения: DD.MM.YYYY»
_RE_ACCESS_DATE = re.compile(
    r"\(?Дата обращения\s*:\s*\d{1,2}\.\d{1,2}\.\d{4}\.?\)?",
    re.IGNORECASE,
)

# ── Список литературы ──────────────────────────────────────────
# Два паттерна для обнаружения библиографии. Вместо re.sub используем
# re.search + обрезку строки (text[:pos]), чтобы оба случая работали:
#
# Случай A — МНОГОСТРОЧНЫЙ текст (есть \n):
#   Заголовок стоит на отдельной строке: "\nСписок литературы\n"
#
# Случай B — ОДНОСТРОЧНЫЙ текст (OCR без \n):
#   Заголовок встроен в строку после точки: "...текст. Список литературы 1."
#
# Для «Литература» без уточнений — требуем, чтобы за заголовком сразу шла
# цифра или конец строки, чтобы не удалить «в современной литературе...».

_BIB_HEADERS = (
    r"(?:"
    r"Список\s+(?:литературы|источников?|использованн\S+\s+(?:литературы|источников?))"
    r"|Литература"
    r"|Библиографический\s+список"
    r"|Библиография"
    r"|References?"
    r"|Bibliography"
    r")"
)

# Случай A: заголовок на отдельной строке (многострочный текст)
_RE_BIBLIOGRAPHY_ML = re.compile(
    r"(?:^|\n)[ \t]*" + _BIB_HEADERS + r"[ \t]*(?:\n|$)",
    re.IGNORECASE,
)

# Случай B: заголовок встроен в однострочный текст после конца предложения
# Требуем: предшествует ". " или ".\t", а следует пробел + цифра (нумерованный пункт)
_RE_BIBLIOGRAPHY_INLINE = re.compile(
    r"(?<=[.!?])\s{1,5}" + _BIB_HEADERS + r"(?=[ \t]{0,5}\d)",
    re.IGNORECASE,
)

# ── Фрагменты URL-путей ────────────────────────────────────────
# «Хвосты» ссылок, оставшихся после удаления http://-части.
# Пример: «councils/chrstuni/rc_pc_chrstuni_doc_19741201_en.html»
#
# ВАЖНО: используем `[\w\-_.%/]+` (один квантификатор, включая слеши)
# вместо `(?:/[\w\-_.%]+)+` (вложенный), чтобы ИСКЛЮЧИТЬ катастрофический
# backtracking на длинных текстах.
# Паттерн: первая часть — без слеша, ровно один слеш, затем хвост (со слешами).
_RE_URL_FRAGMENT = re.compile(
    r"[\w\-_.%]+/[\w\-_.%/]+\.(?:html?|pdf|php|xml|asp|aspx|cfm|do|shtml)\b",
    re.IGNORECASE,
)

# Одиночный путь начинающийся со слеша без хоста (/archive/index.html)
_RE_URL_ABS_PATH = re.compile(
    r"(?<!\w)/[\w\-_.%/]+\.(?:html?|pdf|xml|php)\b",
    re.IGNORECASE,
)

# УДК / ББК / DOI строки
_RE_UDK_BBK = re.compile(
    r"^[ \t]*(УДК|ББК|UDC|BBK|DOI)\b[^\n]*",
    re.MULTILINE | re.IGNORECASE,
)

# DOI-идентификатор в теле текста (10.XXXX/...)
_RE_DOI = re.compile(
    r"\b10\.\d{4,}/\S+",
    re.IGNORECASE,
)

# Блок «Сведения об авторах» (русский и английский варианты)
_RE_AUTHOR_BLOCK_RU = re.compile(
    r"Сведения об авторах?\s*\n.+",
    re.DOTALL | re.IGNORECASE,
)
_RE_AUTHOR_BLOCK_EN = re.compile(
    r"(Information about (the )?authors?|About (the )?authors?)\s*\n.+",
    re.DOTALL | re.IGNORECASE,
)

# Строки только из заглавных букв (возможно, название журнала/раздела из OCR)
_RE_ALL_CAPS_LINE = re.compile(
    r"^[А-ЯЁA-Z\s\d«»\(\)\-–—:,\.]{10,}\n",
    re.MULTILINE,
)

# Мягкий перенос (shy hyphen) — часто из PDF
_RE_SHY = re.compile(r"\xad")

# Управляющие символы кроме \n, \t
_RE_CTRL = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")

# Три и более переводов строк → два
_RE_MULTI_NL = re.compile(r"\n{3,}")

# Несколько пробелов подряд
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")

# Нумерованные сноски в начале строки: «1.\t», «2. », «[1]»
_RE_FN_NUM = re.compile(r"^\s*\d+[\.\)]\s+", re.MULTILINE)
_RE_FN_BRACKET = re.compile(r"^\s*\[\d+\]\s*", re.MULTILINE)

# Повтор «Ключевые слова:» в теле (если есть отдельная колонка).
#
# Умное определение конца строки ключевых слов — три варианта по приоритету:
#
# A) Многострочный текст (есть \n): читаем до конца строки (≤1000 симв.).
# B) Однострочный OCR-текст: не-жадный поиск до первой точки,
#    за которой стоит пробел + заглавная буква (граница предложения).
#    Максимум 800 симв. — перекрывает даже длинные списки ключевых слов.
# C) Резервный: не более 300 симв. — гарантирует, что тело статьи не пострадает.
#
# Также учитываем вариант «Ключевые слова и фразы:».
_RE_KEYWORDS_LINE = re.compile(
    r"Ключевые\s+(?:слова\s+и\s+фразы|слова)\s*[:\uf03a][,\s]*"
    r"(?:"
    r"[^\n]{0,1000}\n"                              # A: до переноса строки
    r"|[^\n]{0,800}?\.[ \t]*(?=\n|$|[А-ЯЁA-Z])"   # B: до точки + заглавная
    r"|[^\n]{0,300}"                                # C: резервный максимум
    r")",
    re.IGNORECASE,
)

# Строка «E-mail:» (различные варианты).
# Ограничиваем до 300 символов по той же причине.
_RE_EMAIL_LINE = re.compile(
    r"E[\s-]?mail\s*:\s*[^\n]{0,300}\n?",
    re.IGNORECASE,
)

# ── OCR: перенос через пробел-дефис-пробел ──────────────────────
# В однострочных OCR-текстах строки без \n переносы сохраняются как
# «word- nextword» (дефис + пробел). Это НЕ то же самое, что составные
# слова (военно-политическое — без пробела), поэтому убираем дефис+пробел.
# Пример: «подчер- кивалось» → «подчеркивалось»
#          «обра- тить»      → «обратить»
_RE_OCR_SPACE_HYPHEN = re.compile(
    r"([а-яА-ЯёЁa-zA-Z])- +([а-яА-ЯёЁa-zA-Z])",
    re.UNICODE,
)

# ── Лишний обратный слеш ─────────────────────────────────────────
# OCR иногда вставляет «\» после закрывающей кавычки/скобки перед пробелом:
# «(Рим. 11:29)»\ Она» → «(Рим. 11:29)» Она»
_RE_STRAY_BACKSLASH = re.compile(
    r"(?<=[»\)\]\"])[ \t]*\\[ \t]*",
)

# ── Блок «Как цитировать статью:» ───────────────────────────────
# Некоторые журналы вставляют этот блок прямо в тело OCR-текста.
# Убираем от «Как цитировать…» до конца строки (≤800 симв.).
_RE_HOW_TO_CITE = re.compile(
    r"Как\s+цитировать\s+(?:статью|работу|публикацию)\s*:[^\n]{0,800}\n?",
    re.IGNORECASE,
)

# «How to cite this article:» — опционально (не включать по умолчанию)
_RE_HOW_TO_CITE_EN = re.compile(
    r"How\s+to\s+cite(?:\s+this\s+article)?\s*:[^\n]{0,400}\n?",
    re.IGNORECASE,
)

# ── OCR-перенос ────────────────────────────────────────────────
# OCR часто разрывает слово дефисом через перевод строки:
#   «иудей-\nско-католического» → «иудейско-католического»
#   «не-\nпостижимой» → «непостижимой»
# Применяется ДО нормализации пробелов — иначе \n уже заменён.
# Захватываем букву до дефиса и букву после переноса строки.
_RE_OCR_HYPHEN = re.compile(
    r"([а-яА-ЯёЁa-zA-Z])-[ \t]*\n[ \t]*([а-яА-ЯёЁa-zA-Z])",
    re.UNICODE,
)

# ── Внутритекстовые ссылки-номера ─────────────────────────────
# Вида «(1)», «(2, Преамбула)», «[1]», «[2, 3]» — ссылки
# на библиографию внутри текста.
# ОСТОРОЖНО: могут совпадать с «(2016)» и т.п. — включается
# только при remove_inline_refs=True (по умолч. False).
# Паттерн: скобка, одна или несколько цифр через запятую.
_RE_INLINE_REF_ROUND = re.compile(r"\(\s*\d+(?:\s*[,;]\s*\d+)*\s*\)")
_RE_INLINE_REF_SQUARE = re.compile(r"\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]")


# ───────────────────────────────────────────────────────────────
# Нормализация Unicode
# ───────────────────────────────────────────────────────────────

_QUOTES_MAP = str.maketrans({
    "\u201c": '"', "\u201d": '"',   # "левая/правая двойная"
    "\u00ab": '"', "\u00bb": '"',   # «»
    "\u2018": "'", "\u2019": "'",   # 'одинарные'
    "\u2014": " — ",                # длинное тире → пробел-тире-пробел
    "\u2013": "–",                  # короткое тире оставляем
    "\u00a0": " ",                  # неразрывный пробел → обычный
    "\ufeff": "",                   # BOM
})


def _strip_html_to_text(text: str) -> str:
    """Если в строке есть теги — извлекаем видимый текст (ошибочный HTML в полях)."""
    if not text or "<" not in text or ">" not in text:
        return text
    try:
        return bs4.BeautifulSoup(text, "lxml").get_text(" ", strip=True)
    except Exception:
        return text


def _normalize_unicode(text: str) -> str:
    """
    Нормализует юникодные символы:
    - убирает BOM,
    - заменяет типографские кавычки на ASCII,
    - неразрывные пробелы → обычные,
    - применяет NFC-нормализацию.
    """
    text = text.translate(_QUOTES_MAP)
    return unicodedata.normalize("NFC", text)


# ───────────────────────────────────────────────────────────────
# Публичные функции
# ───────────────────────────────────────────────────────────────

def detect_artifacts(text: str, *, extended: bool = False) -> dict[str, Any]:
    """
    Анализирует текст на наличие характерных артефактов.

    extended=True — добавляет целочисленные счётчики из detect_noise_patterns
    (OCR-мусор, повторы строк, маркеры Abstract и т.д.).

    Используйте для диагностики качества текста перед очисткой.
    """
    base: dict[str, Any] = {
        "emails":        len(_RE_EMAIL.findall(text)),
        "urls":          len(_RE_URL.findall(text)),
        "udk_bbk":       len(_RE_UDK_BBK.findall(text)),
        "doi":           len(_RE_DOI.findall(text)),
        "access_mode":   len(_RE_ACCESS_MODE.findall(text)),
        "bibliography":  1 if (_RE_BIBLIOGRAPHY_ML.search(text) or _RE_BIBLIOGRAPHY_INLINE.search(text)) else 0,
        "url_fragments": len(_RE_URL_FRAGMENT.findall(text)),
        "ocr_hyphens":   len(_RE_OCR_HYPHEN.findall(text)),
        "shy_hyphens":   len(_RE_SHY.findall(text)),
        "ctrl_chars":    len(_RE_CTRL.findall(text)),
        "email_lines":   len(_RE_EMAIL_LINE.findall(text)),
        "bom":           text.count("\ufeff"),
    }
    if extended:
        for k, v in detect_noise_patterns(text).items():
            base[f"noise_{k}"] = int(v) if isinstance(v, bool) else v
    return base


def _with_profile_name(kwargs: dict[str, Any], name: str) -> dict[str, Any]:
    d = dict(kwargs)
    d["_source_profile_name"] = name
    return d


def clean_kwargs_for_source(
    source: str | None = None,
    text_source: str | None = None,
    language: str | None = None,
    *,
    outlet: str | None = None,
) -> dict[str, Any]:
    """
    Рекомендуемые флаги clean_text по источнику (CyberLeninka, Lenta, newsapi+outlet).

    outlet — необязательная подсказка (домен/сайт), например «pravda.ru».
    В словаре служебный ключ _source_profile_name снимите через get_source_clean_profile.
    """
    s = (source or "").replace(" ", "").lower()
    ol = (outlet or "").replace(" ", "").lower()
    blob = f"{source or ''} {text_source or ''} {outlet or ''}".lower()
    ts = (text_source or "").strip().lower().replace("-", "_")
    lang = (language or "").strip().lower()
    out: dict[str, Any] = {}

    if "cyberleninka" in s and ts in ("html_ocr", "htmlocr", "ocr"):
        out.update(
            {
                "remove_front_matter": True,
                "remove_page_noise": True,
                "remove_repeated_boilerplate": True,
                "remove_web_prompts": True,
                "trim_head_body_start": True,
                "trim_article_boundaries": True,
            }
        )
        if lang.startswith("ru"):
            out["remove_english_abstract"] = True
        return _with_profile_name(out, "cyberleninka_html_ocr")

    if "lenta" in s and ts in ("html", "web") and "ocr" not in ts:
        return _with_profile_name({"remove_media_captions": True}, "lenta_html")

    if ("rbc" in s or "росбизнесконсалтинг" in s) and ts in ("html", "web") and "ocr" not in ts:
        return _with_profile_name(
            {
                "remove_rbc_footer_cta": True,
                "remove_media_captions": True,
            },
            "rbc_html",
        )

    if "newsapi" in s or "news_api" in s or "newsapi" in ol:
        if "pikabu" in blob:
            return _with_profile_name(
                {
                    "remove_portal_chrome": True,
                    "portal_chrome_variant": "pikabu_strong",
                    "trim_pikabu_trailing_noise": True,
                    "remove_repeated_boilerplate": True,
                    "remove_page_noise": True,
                },
                "newsapi_pikabu",
            )
        if "pravda" in blob:
            blob_compact = blob.replace(" ", "").replace("/", "")
            if "com.ua" in blob or "pravda.com.ua" in blob.replace(" ", "") or "pravdacomua" in blob_compact:
                return _with_profile_name(
                    {
                        "remove_portal_chrome": True,
                        "portal_chrome_variant": "pravda_ua",
                        "remove_media_captions": True,
                    },
                    "newsapi_pravda_ua",
                )
            return _with_profile_name(
                {
                    "remove_portal_chrome": True,
                    "portal_chrome_variant": "pravda",
                    "remove_media_captions": True,
                },
                "newsapi_pravda",
            )
        if "mail" in blob or "mail.ru" in blob:
            return _with_profile_name(
                {
                    "remove_portal_chrome": True,
                    "portal_chrome_variant": "mail_ru",
                },
                "newsapi_mail_ru",
            )

    return out


def get_source_clean_profile(
    source: str | None = None,
    text_source: str | None = None,
    language: str | None = None,
    *,
    outlet: str | None = None,
) -> dict[str, Any]:
    """Имя профиля и kwargs без служебных ключей (удобно для логов / EDA)."""
    kw = clean_kwargs_for_source(source, text_source, language, outlet=outlet)
    name = kw.pop("_source_profile_name", None)
    return {"profile_name": name, "clean_kwargs": kw}


def clean_text_with_report(
    text: str,
    *,
    source: str | None = None,
    text_source: str | None = None,
    language: str | None = None,
    apply_source_profile: bool = False,
    strip_duplicate_title: str | None = None,
    strip_annotation_prefix: bool = False,
    annotation: str | None = None,
    quality_use_extended_artifacts: bool = True,
    **clean_kwargs: Any,
) -> dict[str, Any]:
    """
    Очистка текста с отчётом для EDA: длины, артефакты до/после, quality_score, карантин.

    strip_annotation_prefix=True — взять из annotation первую строку (до 200 симв.)
    как префикс для strip_duplicate_title, если заголовок дублируется в начале text.
    """
    raw = text if isinstance(text, str) else ""
    raw_len = len(raw)

    profile = (
        clean_kwargs_for_source(source, text_source, language) if apply_source_profile else {}
    )
    merged: dict[str, Any] = {**profile, **clean_kwargs}

    eff_title = strip_duplicate_title
    if strip_annotation_prefix and annotation and isinstance(annotation, str):
        ann = annotation.strip().split("\n", 1)[0].strip()
        if len(ann) >= 20:
            eff_title = eff_title or ann[:200]

    if eff_title:
        merged["strip_duplicate_title"] = eff_title

    merged.pop("meta_pass_flags", None)
    source_profile_name: str | None = merged.pop("_source_profile_name", None)

    cleaning_flags: list[str] = []
    if apply_source_profile and profile:
        cleaning_flags.append("apply_source_profile")

    meta_pass_flags: list[str] = []

    ext = quality_use_extended_artifacts
    artifacts_before = detect_artifacts(raw, extended=ext)
    quality_before = analyze_text_quality(raw, artifacts=artifacts_before)

    cleaned = clean_text(raw, meta_pass_flags=meta_pass_flags, **merged)
    cleaned_len = len(cleaned)
    removed = raw_len - cleaned_len
    ratio = (removed / raw_len) if raw_len else 0.0

    artifacts_after = detect_artifacts(cleaned, extended=ext)
    quality_after = analyze_text_quality(
        cleaned,
        artifacts=artifacts_after,
        removed_ratio_hint=ratio,
    )

    cleaning_flags.extend(meta_pass_flags)

    art_trim = any(
        "article_start_trim" in x or "trim_document_head" in x for x in meta_pass_flags
    )
    tail_applied = any(x.startswith("tail_section_trim") for x in meta_pass_flags)
    portal_strip = any(x.startswith("portal_chrome_removed:") for x in meta_pass_flags)

    quality_before_enriched = {
        **quality_before,
        "article_boundary_trim_applied": False,
        "tail_cut_applied": False,
        "portal_chrome_strip_applied": False,
        "source_profile_name": source_profile_name,
    }
    quality_after_enriched = {
        **quality_after,
        "article_boundary_trim_applied": art_trim,
        "tail_cut_applied": tail_applied,
        "portal_chrome_strip_applied": portal_strip,
        "source_profile_name": source_profile_name,
    }

    is_q = bool(quality_after_enriched.get("quarantine_candidate"))

    return {
        "cleaned_text": cleaned,
        "raw_length": raw_len,
        "cleaned_length": cleaned_len,
        "removed_chars": removed,
        "removed_ratio": round(ratio, 6),
        "artifacts_before": artifacts_before,
        "artifacts_after": artifacts_after,
        "quality_report": quality_after_enriched,
        "quality_report_before": quality_before_enriched,
        "quality_report_after": quality_after_enriched,
        "probable_document_type_before": quality_before.get("probable_document_type"),
        "probable_document_type_after": quality_after.get("probable_document_type"),
        "quality_score_before": float(quality_before.get("quality_score", 0.0)),
        "quality_score_after": float(quality_after.get("quality_score", 0.0)),
        "quarantine_candidate_before": bool(quality_before.get("quarantine_candidate")),
        "quarantine_candidate_after": bool(quality_after.get("quarantine_candidate")),
        "cleaning_flags": cleaning_flags,
        "meta_pass_flags": meta_pass_flags,
        "is_quarantine_candidate": is_q,
        "quality_score": float(quality_after.get("quality_score", 0.0)),
        "source_profile_name": source_profile_name,
        "article_boundary_trim_applied": art_trim,
        "tail_cut_applied": tail_applied,
        "portal_chrome_strip_applied": portal_strip,
    }


def clean_text(
    text: str,
    remove_emails: bool = True,
    remove_urls: bool = True,
    remove_udk: bool = True,
    remove_author_blocks: bool = True,
    remove_bibliography: bool = True,
    remove_url_fragments: bool = True,
    join_ocr_hyphens: bool = True,
    remove_keywords_line: bool = False,
    remove_footnote_numbers: bool = False,
    remove_inline_refs: bool = False,
    *,
    remove_front_matter: bool = False,
    remove_page_noise: bool = False,
    remove_repeated_boilerplate: bool = False,
    remove_web_prompts: bool = False,
    remove_english_abstract: bool = False,
    remove_how_to_cite_en: bool = False,
    strip_duplicate_title: str | None = None,
    remove_media_captions: bool = False,
    trim_head_body_start: bool = False,
    aggressive_page_noise: bool = False,
    strip_english_language: str | None = "ru",
    meta_pass_flags: list[str] | None = None,
    trim_article_boundaries: bool = False,
    remove_portal_chrome: bool = False,
    portal_chrome_variant: str | None = None,
    remove_rbc_footer_cta: bool = False,
    trim_pikabu_trailing_noise: bool = False,
) -> str:
    """
    Очищает текст статьи от артефактов вёрстки и личных данных.

    Аргументы:
        text                    — исходный текст.
        remove_emails           — удалить email-адреса (по умолч. True).
        remove_urls             — удалить URL-ссылки (по умолч. True).
        remove_udk              — удалить строки УДК/ББК/DOI (True).
        remove_author_blocks    — удалить блоки «Сведения об авторах» (True).
        remove_bibliography     — удалить раздел «Список литературы» и
                                  всё после него (по умолч. True).
        remove_url_fragments    — удалить хвосты URL-путей, оставшиеся
                                  после удаления http://-части (True).
        join_ocr_hyphens        — склеить слова, разорванные OCR-переносом:
                                  «слово-\\nпродолжение» → «слово продолжение»
                                  (по умолч. True).
        remove_keywords_line    — удалить строку «Ключевые слова:» из тела
                                  (False — ключевые слова хранятся в отдельной
                                  колонке, но строка в тексте тоже несёт смысл).
        remove_footnote_numbers — удалить нумерацию сносок в начале строк
                                  «1.», «[1]» (False — риск удалить нужные числа).
        remove_inline_refs      — удалить внутритекстовые ссылки-номера вида
                                  «(1)», «[2, 3]» (False — риск совпадения
                                  с годами: «(2016)»).
        remove_front_matter     — первые ~80 строк: ISSN, «Главный редактор»,
                                  «Аннотация:» как отдельная строка и т.п. (False).
        remove_page_noise       — строки только из 1–4 цифр (False).
        remove_repeated_boilerplate — строки, повторённые ≥4 раза, 18–200 симв. (False).
        remove_web_prompts      — подсказки портала CyberLeninka (False).
        remove_english_abstract — отдельный блок Abstract на англ. (False; для ru-корпуса).
        remove_how_to_cite_en   — строка «How to cite…» (False).
        strip_duplicate_title   — убрать ведущий дубликат заголовка (если передан).
        remove_media_captions   — строки «Фото:», «Кадр:»… для новостного HTML (False).
        trim_head_body_start    — эвристика обрезки шумной шапки до тела статьи (False).
        aggressive_page_noise   — также убирать каталожные строки «-2.6», «96.01.022.» (False).
        strip_english_language  — для remove_english_abstract: «ru» или None (по умолч. «ru»).
        meta_pass_flags         — если list, дополняется метками сработавших meta-проходов.
        trim_article_boundaries — эвристика хвоста предыдущей статьи + tail-секции + англ. хвост (False).
        remove_portal_chrome    — снять меню/footer в head/tail (False; только с portal_chrome_variant).
        portal_chrome_variant   — «pravda» | «pravda_ua» | «pikabu_strong» | «mail_ru» и т.п.
        remove_rbc_footer_cta   — хвост «Оставайтесь на связи с РБК…» (False; только RBC).
        trim_pikabu_trailing_noise — срез хвоста тегов/портала Pikabu (False).

    Порядок операций:
        1.   Нормализация Unicode (BOM, кавычки, тире, NBSP).
        2.   Мягкие переносы (U+00AD) + управляющие символы.
        2.5. Лишний обратный слеш (»\\ → » ) — OCR-артефакт.
        3.   OCR-перенос через «дефис + \\n» — склейка слов.
        3.5. OCR-перенос через «дефис + пробел» (word- next → wordnext).
        4.   Раздел «Список литературы» — удаляется ПЕРВЫМ, целиком до конца
             текста, чтобы не оставлять «голые» фрагменты URL.
        5.   Блоки «Сведения об авторах».
        6.   УДК/ББК/DOI строки.
        7.   URL-ссылки (включая http:// с пробелом после ://).
        8.   «Режим доступа:» (вся строка) + «Дата обращения:».
        9.   Строки «E-mail:».
        9.5. Блок «Как цитировать статью:» — служебный блок журнала.
        10.  Email-адреса.
        11.  DOI-идентификаторы в теле.
        12.  Фрагменты URL-путей (хвосты после удалённых http://-частей).
        13.  Строка «Ключевые слова:» (если флаг включён).
        14.  Нумерация сносок (если флаг включён).
        15.  Внутритекстовые ссылки (1), [1] (если флаг включён).
        15.5. Опционально: «How to cite» (EN), затем meta-pass (front matter, abstract, …).
        16.  Нормализация пробелов и переводов строк.

    Возвращает:
        Очищенный текст.
    """
    if not text or not isinstance(text, str):
        return text

    # 0. Сырой HTML в text/description → plain text до остальных шагов
    text = _strip_html_to_text(text)

    # 1. Нормализация Unicode
    text = _normalize_unicode(text)

    # 2. Мягкие переносы + управляющие символы
    text = _RE_SHY.sub("", text)
    text = _RE_CTRL.sub("", text)

    # 2.5. Лишний обратный слеш после закрывающих кавычек/скобок (OCR-артефакт).
    text = _RE_STRAY_BACKSLASH.sub(" ", text)

    # 2.6. Portal chrome (только head/tail) — source-specific.
    if remove_portal_chrome and portal_chrome_variant:
        pv = (portal_chrome_variant or "").strip().lower().replace(".", "_")
        if pv in ("pravda_ua", "pravda_com_ua"):
            text, _pc = remove_portal_chrome(text, "pravda_ua", head_lines=56, tail_lines=70)
        else:
            text, _pc = remove_portal_chrome(text, portal_chrome_variant)
        if meta_pass_flags is not None:
            meta_pass_flags.extend(_pc)

    if remove_rbc_footer_cta:
        text, _rbc = strip_rbc_footer_cta_lines(text)
        if meta_pass_flags is not None:
            meta_pass_flags.extend(_rbc)

    if trim_pikabu_trailing_noise:
        text, _pkt = trim_pikabu_trailing_portal_lines(text)
        if meta_pass_flags is not None:
            meta_pass_flags.extend(_pkt)

    # 3. OCR-перенос: «слово-\nпродолжение» → «словопродолжение» (без пробела,
    #    так как перенос означает склейку: «иудей-\nско» → «иудейско»).
    #    Применяем ДО нормализации пробелов, пока \n ещё в тексте.
    if join_ocr_hyphens:
        text = _RE_OCR_HYPHEN.sub(r"\1\2", text)

    # 3.5. OCR-перенос через пробел-дефис: «подчер- кивалось» → «подчеркивалось».
    #      В однострочных OCR-текстах перенос строки преобразован в пробел.
    #      Всегда включено (явный артефакт, не путать с законными «военно-политическое»,
    #      где пробела после дефиса нет).
    text = _RE_OCR_SPACE_HYPHEN.sub(r"\1\2", text)

    # 3.7. Вероятный хвост предыдущей статьи в начале (OCR-стримы).
    if trim_article_boundaries:
        text, _lb = trim_leading_article_bleed(text)
        if meta_pass_flags is not None:
            meta_pass_flags.extend(_lb)

    # 4. Список литературы — удаляем целиком ПЕРВЫМ, до URL-паттернов,
    #    чтобы не оставлять «голые» фрагменты путей и разорванные ссылки.
    #
    #    Используем search + обрезку вместо sub, чтобы корректно работать в двух
    #    ситуациях:
    #    A) многострочный текст (заголовок на отдельной строке),
    #    B) однострочный OCR-текст (заголовок встроен в строку).
    #
    #    Защита: не обрезаем если результат < 10% исходного — признак ложного
    #    срабатывания (например, слово «литература» в середине вводной части).
    if remove_bibliography:
        m_ml = _RE_BIBLIOGRAPHY_ML.search(text)
        m_il = _RE_BIBLIOGRAPHY_INLINE.search(text)
        best: tuple[int, str] | None = None
        if m_ml:
            best = (m_ml.start(), "ml")
        if m_il and (best is None or m_il.start() < best[0]):
            best = (m_il.start(), "il")
        if best:
            pos, mode = best
            candidate = text[:pos].rstrip()
            # ML: заголовок на строке — мягкий порог (короткие статьи в тестах и выборки).
            # IL: встроенный в строку — жёстче, чтобы не резать «... в литературе 1.».
            if mode == "ml":
                min_need = max(len(text) * 0.06, 5.0)
            else:
                _floor = 100 if len(text) > 900 else max(28, int(len(text) * 0.22))
                min_need = max(len(text) * 0.10, _floor)
            if len(candidate) >= min_need:
                text = candidate

    # 5. Блоки сведений об авторах
    if remove_author_blocks:
        text = _RE_AUTHOR_BLOCK_RU.sub("", text)
        text = _RE_AUTHOR_BLOCK_EN.sub("", text)

    # 6. УДК/ББК/DOI строки
    if remove_udk:
        text = _RE_UDK_BBK.sub("", text)

    # 7. URL-ссылки (в т.ч. с пробелом после "://") — до «Режим доступа»,
    #    чтобы остаток фразы потом убрался одним паттерном.
    if remove_urls:
        text = _RE_URL.sub("", text)

    # 8. «Режим доступа:» — убираем всю строку (URL уже ушли на шаге 7)
    #    + «Дата обращения:»
    if remove_urls:
        text = _RE_ACCESS_MODE.sub("", text)
        text = _RE_ACCESS_DATE.sub("", text)

    # 9. Строки «E-mail:»
    if remove_emails:
        text = _RE_EMAIL_LINE.sub("", text)

    # 9.5. Блок «Как цитировать статью:» — всегда удаляем, это служебный блок
    #      некоторых журналов, вклинивающийся в основной текст.
    text = _RE_HOW_TO_CITE.sub("", text)

    # 10. Email-адреса
    if remove_emails:
        text = _RE_EMAIL.sub("", text)

    # 11. DOI в теле
    if remove_udk:
        text = _RE_DOI.sub("", text)

    # 12. Фрагменты URL-путей — «хвосты» разорванных OCR-ом ссылок
    if remove_url_fragments:
        text = _RE_URL_FRAGMENT.sub("", text)
        text = _RE_URL_ABS_PATH.sub("", text)

    # 13. Строка «Ключевые слова:»
    if remove_keywords_line:
        text = _RE_KEYWORDS_LINE.sub("", text)

    # 14. Нумерация сносок
    if remove_footnote_numbers:
        text = _RE_FN_NUM.sub("", text)
        text = _RE_FN_BRACKET.sub("", text)

    # 15. Внутритекстовые ссылки (1), [2] — только при явном включении.
    #     По умолч. выключено: риск удалить «(2016)», «(с. 249)» и т.п.
    if remove_inline_refs:
        text = _RE_INLINE_REF_ROUND.sub("", text)
        text = _RE_INLINE_REF_SQUARE.sub("", text)

    if remove_how_to_cite_en:
        text = _RE_HOW_TO_CITE_EN.sub("", text)

    meta_on = (
        remove_front_matter
        or remove_page_noise
        or remove_repeated_boilerplate
        or remove_web_prompts
        or remove_english_abstract
        or remove_media_captions
        or trim_head_body_start
        or (strip_duplicate_title and len((strip_duplicate_title or "").strip()) >= 12)
    )
    if meta_on:
        text, mflags = apply_meta_clean_pass(
            text,
            remove_front_matter=remove_front_matter,
            remove_page_noise=remove_page_noise,
            remove_repeated_boilerplate=remove_repeated_boilerplate,
            remove_web_prompts=remove_web_prompts,
            remove_english_abstract=remove_english_abstract,
            strip_title_prefix=strip_duplicate_title,
            trim_head_body_start=trim_head_body_start,
            aggressive_page_noise=aggressive_page_noise,
            strip_english_language=strip_english_language,
            remove_media_captions=remove_media_captions,
        )
        if meta_pass_flags is not None:
            meta_pass_flags.extend(mflags)

    # 15.7. Хвост: библиография/авторы/след. статья/англ. метаданные (после основных шагов).
    if trim_article_boundaries:
        text, _tf = trim_tail_sections(text)
        if meta_pass_flags is not None:
            meta_pass_flags.extend(_tf)
        if remove_english_abstract:
            text, _en_tail = strip_trailing_english_metadata_block(
                text, language=strip_english_language
            )
            if _en_tail and meta_pass_flags is not None:
                meta_pass_flags.append("trailing_english_metadata_removed")

    # 16. Нормализация пробелов
    text = _RE_MULTI_SPACE.sub(" ", text)
    text = _RE_MULTI_NL.sub("\n\n", text)
    text = text.strip()

    return text


def clean_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    annotation_col: Optional[str] = "annotation",
    *,
    source_col: Optional[str] = None,
    text_source_col: Optional[str] = None,
    language_col: Optional[str] = None,
    title_col: Optional[str] = None,
    apply_source_profile: bool = False,
    with_quality_report: bool = False,
    strip_annotation_prefix: bool = False,
    quality_use_extended_artifacts: bool = True,
    **clean_kwargs: Any,
) -> pd.DataFrame:
    """
    Применяет clean_text() к text и (опционально) annotation/description.

    Если with_quality_report=True — по строкам вызывается clean_text_with_report()
    и добавляются метрики (медленнее). Иначе при необходимости построчный merge
    профиля источника (apply_source_profile + source_col / text_source_col / language_col).

    Новые колонки при with_quality_report:
        clean_removed_chars, clean_removed_ratio, cleaning_flags (JSON),
        is_quarantine_candidate, quality_score,
        artifacts_before_json, artifacts_after_json,
        probable_document_type, probable_document_type_before,
        front_matter_density, article_like_score, bibliographic_like_score,
        metadata_noise_detected, meta_pass_flags_json.
    """
    df = df.copy()

    rowwise = (
        apply_source_profile
        or with_quality_report
        or strip_annotation_prefix
        or (title_col is not None and title_col in df.columns)
    )

    def _cell_str(val: Any) -> str:
        return str(val) if pd.notna(val) else ""

    def _row_profile_kwargs(row: pd.Series) -> dict[str, Any]:
        kw = dict(clean_kwargs)
        if apply_source_profile:
            s = _cell_str(row[source_col]) if source_col and source_col in row.index else ""
            ts = _cell_str(row[text_source_col]) if text_source_col and text_source_col in row.index else ""
            lg = _cell_str(row[language_col]) if language_col and language_col in row.index else ""
            prof = dict(clean_kwargs_for_source(s or None, ts or None, lg or None))
            prof.pop("_source_profile_name", None)
            merged = {**prof, **kw}
            return merged
        return kw

    if text_col in df.columns:
        if with_quality_report:

            def _one_rep_fixed(row: pd.Series) -> dict[str, Any]:
                raw_t = row[text_col]
                t = _cell_str(raw_t)
                ann = row[annotation_col] if annotation_col and annotation_col in row.index else None
                tit = row[title_col] if title_col and title_col in row.index else None
                stp: str | None = None
                if tit is not None and pd.notna(tit):
                    ts = str(tit).strip()
                    if ts:
                        stp = ts
                base_kw = dict(clean_kwargs)
                return clean_text_with_report(
                    t,
                    source=_cell_str(row[source_col]) if source_col and source_col in row.index else None,
                    text_source=_cell_str(row[text_source_col])
                    if text_source_col and text_source_col in row.index
                    else None,
                    language=_cell_str(row[language_col])
                    if language_col and language_col in row.index
                    else None,
                    apply_source_profile=apply_source_profile,
                    strip_duplicate_title=stp,
                    strip_annotation_prefix=strip_annotation_prefix,
                    annotation=_cell_str(ann) if ann is not None and pd.notna(ann) else None,
                    quality_use_extended_artifacts=quality_use_extended_artifacts,
                    **base_kw,
                )

            reps = df.apply(_one_rep_fixed, axis=1)
            df[f"{text_col}_clean"] = reps.map(lambda r: r["cleaned_text"])
            df["text_length_clean"] = df[f"{text_col}_clean"].str.len().fillna(0).astype(int)
            df["clean_removed_chars"] = reps.map(lambda r: r["removed_chars"])
            df["clean_removed_ratio"] = reps.map(lambda r: r["removed_ratio"])
            df["cleaning_flags"] = reps.map(lambda r: json.dumps(r["cleaning_flags"], ensure_ascii=False))
            df["is_quarantine_candidate"] = reps.map(lambda r: r["is_quarantine_candidate"])
            df["quality_score"] = reps.map(lambda r: r["quality_score"])
            df["artifacts_before_json"] = reps.map(
                lambda r: json.dumps(r["artifacts_before"], ensure_ascii=False)
            )
            df["artifacts_after_json"] = reps.map(
                lambda r: json.dumps(r["artifacts_after"], ensure_ascii=False)
            )
            df["probable_document_type"] = reps.map(
                lambda r: r.get("probable_document_type_after", "")
            )
            df["probable_document_type_before"] = reps.map(
                lambda r: r.get("probable_document_type_before", "")
            )
            df["front_matter_density"] = reps.map(
                lambda r: float(r["quality_report_after"].get("front_matter_density", 0.0))
            )
            df["article_like_score"] = reps.map(
                lambda r: float(r["quality_report_after"].get("article_like_score", 0.0))
            )
            df["bibliographic_like_score"] = reps.map(
                lambda r: float(r["quality_report_after"].get("bibliographic_like_score", 0.0))
            )
            df["metadata_noise_detected"] = reps.map(
                lambda r: bool(r["quality_report_after"].get("metadata_noise_detected"))
            )
            df["meta_pass_flags_json"] = reps.map(
                lambda r: json.dumps(r.get("meta_pass_flags", []), ensure_ascii=False)
            )
            df["source_profile_name"] = reps.map(lambda r: r.get("source_profile_name"))
            df["article_boundary_trim_applied"] = reps.map(
                lambda r: bool(r.get("article_boundary_trim_applied"))
            )
            df["tail_cut_applied"] = reps.map(lambda r: bool(r.get("tail_cut_applied")))
            df["next_article_bleed_detected"] = reps.map(
                lambda r: bool(r["quality_report_after"].get("next_article_bleed_detected"))
            )
            df["portal_chrome_detected"] = reps.map(
                lambda r: bool(r["quality_report_after"].get("portal_chrome_detected"))
            )
        elif rowwise:

            def _one_clean(row: pd.Series) -> str:
                raw_t = row[text_col]
                if not pd.notna(raw_t):
                    return ""
                kw = _row_profile_kwargs(row)
                tit = row[title_col] if title_col and title_col in row.index else None
                ann = row[annotation_col] if annotation_col and annotation_col in row.index else None
                if tit is not None and pd.notna(tit) and str(tit).strip():
                    kw.setdefault("strip_duplicate_title", str(tit).strip())
                if strip_annotation_prefix and ann is not None and pd.notna(ann):
                    a0 = str(ann).strip().split("\n", 1)[0].strip()
                    if len(a0) >= 20:
                        kw.setdefault("strip_duplicate_title", a0[:200])
                return clean_text(str(raw_t), **kw)

            df[f"{text_col}_clean"] = df.apply(_one_clean, axis=1)
            df["text_length_clean"] = df[f"{text_col}_clean"].str.len().fillna(0).astype(int)
        else:
            df[f"{text_col}_clean"] = df[text_col].apply(
                lambda t: clean_text(str(t), **clean_kwargs) if pd.notna(t) else ""
            )
            df["text_length_clean"] = df[f"{text_col}_clean"].str.len().fillna(0).astype(int)

    if annotation_col and annotation_col in df.columns:
        df[f"{annotation_col}_clean"] = df[annotation_col].apply(
            lambda t: clean_text(str(t), **clean_kwargs) if pd.notna(t) else ""
        )

    return df

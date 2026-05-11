"""Точечные регрессии: RBC footer, Pravda.com.ua, Pikabu portal noise."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cleaner import clean_kwargs_for_source, clean_text  # noqa: E402
from src.cleaner_quality import (  # noqa: E402
    analyze_text_quality,
    remove_portal_chrome,
    remove_rbc_footer_cta,
)


def test_rbc_max_footer_line_removed():
    body = "Банк России сохранил ключевую ставку на прежнем уровне по итогам заседания."
    raw = body + "\n\nОставайтесь на связи с РБК в Max .\n"
    out, flags = remove_rbc_footer_cta(raw)
    assert "Оставайтесь на связи" not in out
    assert body.strip() in out
    assert flags


def test_rbc_profile_and_clean_text():
    kw = clean_kwargs_for_source("rbc.ru", "html", "ru")
    assert kw.get("_source_profile_name") == "rbc_html"
    assert kw.get("remove_rbc_footer_cta") is True
    kw.pop("_source_profile_name", None)
    raw = "Новость дня кратко изложена в этом абзаце для теста.\n\nОставайтесь на связи с РБК в Mах.\n"
    out = clean_text(raw, **kw)
    assert "Оставайтесь" not in out
    assert "Новость дня" in out


def test_pravda_ua_top_nav_and_footer_removed():
    raw = (
        "Новости Публикации Колонки Интервью Архив\n\n"
        "Главная новость: правительство объявило о мерах поддержки граждан в регионах "
        "и уточнило сроки реализации программы.\n\n"
        "Реклама на сайте\n"
        "Правила использования материалов\n"
        "Политика ИИ\n"
    )
    out, flags = remove_portal_chrome(raw, "pravda_ua", head_lines=56, tail_lines=65)
    assert "Главная новость" in out
    assert "Новости Публикации Колонки" not in out
    assert "Реклама на сайте" not in out
    assert "Правила использования материалов" not in out
    assert flags


def test_pravda_ua_profile():
    kw = clean_kwargs_for_source("NewsAPI", "html", "uk", outlet="pravda.com.ua")
    assert kw.get("_source_profile_name") == "newsapi_pravda_ua"
    assert kw.get("portal_chrome_variant") == "pravda_ua"


def test_pikabu_portal_noise_heavy_quarantine_or_type():
    wall_lines = ["Юмор · Здоровье · Бизнес · Транспорт · Наука · Игры · Моё"] * 14
    body = "Основной текст поста с развёрнутым содержанием и мнением автора по теме дня."
    raw = "\n".join(["Подписаться", "Спасибо, что подписались"] + wall_lines + [body, "Теги:"] + wall_lines[:8])
    q = analyze_text_quality(raw)
    assert q.get("portal_chrome_heavy") is True
    assert q.get("probable_document_type") == "portal_noise_heavy" or q.get("quarantine_candidate") is True

"""Patch v3: границы статьи, хвост, portal chrome, профили источников."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cleaner import (
    clean_kwargs_for_source,
    clean_text,
    clean_text_with_report,
    get_source_clean_profile,
)
from src.cleaner_quality import (
    detect_next_article_bleed,
    remove_portal_chrome,
    trim_tail_sections,
)


class TestCleanerV3CyberleninkaOCR(unittest.TestCase):
    def test_bibliography_then_tail_bleed_removed(self):
        core = (
            "Введение в исследование проблемы содержит постановку вопроса и обзор "
            "литературы по теме. Основная часть раскрывает методы и результаты."
        )
        raw = (
            "короткий хвост\n123\n\n"
            + core
            + "\n\nСписок литературы\n1. Иванов И.И. Книга.\n\n"
            "Сведения об авторах\nИванов — кандидат наук.\n\n"
            "Abstract\n\nThe study presents results.\n\n"
            "УДК 123.4\nНОВЫЙ ЗАГОЛОВОК СТАТЬИ В ВЕРХНЕМ РЕГИСТРЕ\n"
        )
        out = clean_text(
            raw,
            remove_author_blocks=True,
            remove_bibliography=True,
            trim_article_boundaries=True,
            remove_english_abstract=True,
        )
        self.assertIn("Основная часть", out)
        self.assertNotIn("Список литературы", out)

    def test_tail_trim_without_bib_header_udc_bleed(self):
        paras = [
            f"Абзац {i}: развёрнутое изложение темы на русском языке с примерами и выводами. " * 2
            for i in range(38)
        ]
        core = "\n\n".join(paras)
        raw = (
            core
            + "\n\nAbstract\n\nThe English summary paragraph without cyrillic letters here.\n\n"
            "УДК 111.22\nНОВАЯ СТАТЬЯ ЗАГОЛОВОК В ВЕРХНЕМ РЕГИСТРЕ ДОСТАТОЧНО ДЛИННЫЙ\n"
        )
        out = clean_text(
            raw,
            remove_bibliography=False,
            remove_author_blocks=False,
            trim_article_boundaries=True,
            remove_english_abstract=True,
        )
        self.assertNotIn("УДК 111.22", out)
        self.assertNotIn("НОВАЯ СТАТЬЯ ЗАГОЛОВОК", out)
        self.assertNotIn("The English summary", out)
        self.assertIn("Абзац 0", out)

    def test_profile_cyberleninka_has_boundaries(self):
        kw = clean_kwargs_for_source("CyberLeninka", "html_ocr", "ru")
        self.assertEqual(kw.get("_source_profile_name"), "cyberleninka_html_ocr")
        self.assertTrue(kw.get("trim_article_boundaries"))


class TestCleanerV3Lenta(unittest.TestCase):
    def test_lenta_like_not_over_stripped(self):
        body = (
            "Министр сообщил о новых мерах поддержки экономики в регионе. "
            "По его словам, программа продлится до конца года."
        )
        out = clean_text(
            body,
            remove_media_captions=True,
            trim_article_boundaries=False,
            remove_portal_chrome=False,
        )
        self.assertIn("Министр", out)
        self.assertGreater(len(out), len(body) * 0.85)

    def test_lenta_profile_minimal(self):
        p = get_source_clean_profile("lenta.ru", "html", "ru")
        self.assertEqual(p["profile_name"], "lenta_html")
        self.assertIn("remove_media_captions", p["clean_kwargs"])


class TestCleanerV3Pravda(unittest.TestCase):
    def test_portal_chrome_pravda(self):
        raw = (
            "Новости\nПубликации\nКолонки\n\n"
            "Важная новость дня состоит в принятии решения правительства "
            "о дальнейших шагах по реформе отрасли и поддержке граждан.\n\n"
            "Использование материалов сайта допускается только с согласия редакции.\n"
            "Реклама на сайте\n"
        )
        out, flags = remove_portal_chrome(raw, "pravda")
        self.assertNotIn("Использование материалов", out)
        self.assertNotIn("Новости", out.split("\n")[0])
        self.assertIn("Важная новость", out)
        self.assertTrue(any("portal_chrome_removed" in f for f in flags))


class TestCleanerV3Pikabu(unittest.TestCase):
    def test_tag_wall_reduced(self):
        wall = " · ".join(["Юмор", "Здоровье", "Бизнес", "Транспорт", "Наука", "Игры"])
        raw = (
            "Подписаться\nСпасибо, что подписались\n"
            + wall
            + "\n\n"
            "Основной текст поста описывает интересный случай из жизни "
            "и содержит развёрнутое мнение автора по обсуждаемой теме."
        )
        out, flags = remove_portal_chrome(raw, "pikabu_strong")
        self.assertLess(out.count("·"), raw.count("·"))
        self.assertIn("Основной текст", out)
        self.assertTrue(any("portal_chrome_removed" in f for f in flags))

    def test_newsapi_pikabu_profile(self):
        kw = clean_kwargs_for_source("NewsAPI", "html", "ru", outlet="pikabu.ru")
        self.assertEqual(kw.get("_source_profile_name"), "newsapi_pikabu")
        self.assertTrue(kw.get("remove_portal_chrome"))


class TestCleanerV3QualityFlags(unittest.TestCase):
    def test_bleed_detection(self):
        tail = "\n".join(["References", "1. A.", "УДК 111.2", "ISSN 2222-3333"])
        body = "\n".join([f"Русский абзац {i} с содержанием и развитием мысли." for i in range(22)])
        long_ru = body + "\n\n" + tail
        self.assertTrue(detect_next_article_bleed(long_ru))

    def test_report_has_v3_fields(self):
        rep = clean_text_with_report("Текст " * 80, apply_source_profile=False)
        self.assertIn("source_profile_name", rep)
        self.assertIn("article_boundary_trim_applied", rep)
        self.assertIn("tail_cut_applied", rep)
        self.assertIn("next_article_bleed_detected", rep["quality_report_after"])


class TestTailOnlyInZone(unittest.TestCase):
    def test_no_early_tail_cut(self):
        lines = ["В тексте упоминается Литература как понятие в культуре."] + [
            "Продолжение абзаца с развёрнутым смыслом и примерами из практики. " * 2
        ] * 25
        raw = "\n".join(lines)
        out, flags = trim_tail_sections(raw)
        self.assertFalse(flags)
        self.assertIn("Литература", out)


if __name__ == "__main__":
    unittest.main()

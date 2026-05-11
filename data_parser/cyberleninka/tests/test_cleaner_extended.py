"""Расширенные тесты cleaner / cleaner_quality (unittest, без зависимости от pytest)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

from src.cleaner import (
    clean_dataframe,
    clean_kwargs_for_source,
    clean_text,
    clean_text_with_report,
    detect_artifacts,
)
from src.cleaner_quality import (
    analyze_text_quality,
    strip_english_abstract_block,
    trim_document_head,
    is_metadata_or_boilerplate_line,
)


class TestCleanerExtended(unittest.TestCase):
    def test_email_and_url_removal(self):
        s = "Связь: user@mail.ru и ссылка https://example.com/path end."
        out = clean_text(s)
        self.assertNotIn("mail.ru", out)
        self.assertNotIn("example.com", out)

    def test_bibliography_cut(self):
        sample = (
            "Основной текст статьи. Выводы.\n\n"
            "Список литературы\n"
            "1. Автор. Книга.\n"
        )
        out = clean_text(sample)
        self.assertNotIn("Список литературы", out)
        self.assertIn("Основной текст", out)

    def test_ocr_hyphen_join(self):
        s = "подчер-\nкивалось в тексте"
        out = clean_text(s)
        self.assertIn("подчеркивалось", out.replace(" ", ""))

    def test_repeated_boilerplate_removed(self):
        line = (
            "НОВЫЕ ИССЛЕДОВАНИЯ > ТУВЫ электронный научный журнал "
            "www.tuva.asia № 4 2009 г."
        )
        body = "Это содержательный абзац статьи про науку в регионе. " * 3
        parts = [line] * 5 + [body]
        raw = "\n".join(parts)
        out = clean_text(raw, remove_repeated_boilerplate=True, remove_urls=True)
        self.assertNotIn(line, out)
        self.assertIn("содержательный", out)

    def test_page_number_lines(self):
        raw = "Введение в тему.\n\n245\n\n246\n\nПродолжение основного текста здесь."
        out = clean_text(raw, remove_page_noise=True)
        self.assertNotIn("245", out)
        self.assertIn("Продолжение", out)

    def test_english_abstract_removed_when_flag(self):
        raw = (
            "В статье рассматривается проблема.\n\n"
            "Abstract\n"
            "The article discusses the problem in detail and provides conclusions.\n\n"
            "Далее идёт основной русский текст с выводами и анализом материала."
        )
        out = clean_text(raw, remove_english_abstract=True)
        self.assertNotIn("Abstract", out)
        self.assertNotIn("The article", out)
        self.assertIn("основной русский", out)

    def test_web_prompt_removal(self):
        raw = "Текст статьи. Не можете найти то, что вам нужно? Попробуйте сервис подбора."
        out = clean_text(raw, remove_web_prompts=True)
        self.assertNotIn("Не можете найти", out)

    def test_clean_text_with_report_structure(self):
        raw = "Короткий текст без артефактов."
        rep = clean_text_with_report(raw)
        keys = {
            "cleaned_text",
            "raw_length",
            "cleaned_length",
            "removed_chars",
            "removed_ratio",
            "artifacts_before",
            "artifacts_after",
            "quality_report",
            "quality_report_before",
            "quality_report_after",
            "probable_document_type_before",
            "probable_document_type_after",
            "quality_score_before",
            "quality_score_after",
            "quarantine_candidate_before",
            "quarantine_candidate_after",
            "meta_pass_flags",
            "cleaning_flags",
            "is_quarantine_candidate",
            "quality_score",
        }
        self.assertTrue(keys.issubset(rep.keys()))
        self.assertEqual(rep["raw_length"], len(raw))
        self.assertIsInstance(rep["quality_report"], dict)
        self.assertIn("probable_document_type", rep["quality_report_after"])

    def test_front_matter_address_source(self):
        raw = (
            "Адрес статьи: https://example.org/x\n"
            "Источник: Журнал Наука\n"
            "ISSN 1234-5678\n"
            "Основной текст статьи с достаточной длиной и содержанием для теста очистки."
        )
        out = clean_text(raw, remove_front_matter=True, remove_urls=True)
        self.assertNotIn("Адрес статьи", out)
        self.assertNotIn("Источник:", out)
        self.assertIn("Основной текст", out)

    def test_trim_document_head_body_start(self):
        noise = "\n".join(["ISSN 1111-2222", "Главный редактор Иванов", "Редакция"]) + "\n\n"
        body = (
            "В данной работе исследуется взаимосвязь нескольких факторов "
            "и приводятся экспериментальные результаты с подробным описанием методики. "
            "Второй абзац продолжает изложение с выводами и сравнением с литературой."
        )
        raw = noise * 15 + "\n" + body
        t2, ntrim, conf = trim_document_head(raw)
        self.assertGreater(ntrim, 0)
        self.assertGreater(conf, 0.4)
        self.assertIn("В данной работе", t2)

    def test_aggressive_page_noise(self):
        raw = "Текст.\n\n-2.6\n\n96.01.022.\n\nПродолжение абзаца с содержанием."
        out = clean_text(raw, remove_page_noise=True, aggressive_page_noise=True)
        self.assertNotIn("-2.6", out)
        self.assertIn("Продолжение", out)

    def test_media_caption_cleanup(self):
        raw = "Новость о событии.\n\nФото: РИА Новости / Иванов\n\nПодробности далее."
        out = clean_text(raw, remove_media_captions=True)
        self.assertNotIn("Фото:", out)
        self.assertIn("Подробности", out)

    def test_english_keywords_block_removal(self):
        raw = (
            "Введение по-русски с достаточным количеством текста для порога.\n\n"
            "Key words: linguistics, corpus, analysis\n\n"
            "Основная часть на русском языке с развёрнутым изложением темы и аргументами."
        )
        out = clean_text(raw, remove_english_abstract=True)
        self.assertNotIn("Key words", out)
        self.assertIn("Основная часть", out)

    def test_probable_document_type_reflective(self):
        head = "РЕФЕРАТИВНЫЙ ЖУРНАЛ НАУКИ\n" * 5
        body = "x " * 200
        q = analyze_text_quality(head + body)
        self.assertIn(q["probable_document_type"], ("bibliographic_review_like", "mixed_unknown", "noisy_ocr_like"))

    def test_is_metadata_line(self):
        self.assertTrue(is_metadata_or_boilerplate_line("Адрес журнала: г. Москва"))
        self.assertFalse(is_metadata_or_boilerplate_line("В данной статье рассматривается проблема."))

    @unittest.skipUnless(pd is not None, "pandas не установлен")
    def test_clean_dataframe_quality_columns(self):
        df = pd.DataFrame(  # type: ignore[union-attr]
            {
                "text": ["Hello user@test.ru http://a.b"],
                "source": ["CyberLeninka"],
                "text_source": ["html_ocr"],
                "language": ["ru"],
            }
        )
        out = clean_dataframe(
            df,
            text_col="text",
            annotation_col=None,
            source_col="source",
            text_source_col="text_source",
            language_col="language",
            with_quality_report=True,
            apply_source_profile=True,
        )
        self.assertIn("text_clean", out.columns)
        self.assertIn("clean_removed_chars", out.columns)
        self.assertIn("quality_score", out.columns)
        self.assertIn("artifacts_before_json", out.columns)
        self.assertIn("probable_document_type", out.columns)
        self.assertIn("front_matter_density", out.columns)
        self.assertIn("article_like_score", out.columns)
        self.assertIn("bibliographic_like_score", out.columns)
        self.assertIn("metadata_noise_detected", out.columns)
        self.assertIn("meta_pass_flags_json", out.columns)
        json.loads(out.iloc[0]["artifacts_before_json"])

    def test_clean_kwargs_for_source_cyberleninka(self):
        kw = clean_kwargs_for_source("CyberLeninka", "html_ocr", "ru")
        self.assertEqual(kw.get("_source_profile_name"), "cyberleninka_html_ocr")
        self.assertTrue(kw.get("remove_front_matter"))
        self.assertTrue(kw.get("trim_head_body_start"))
        self.assertTrue(kw.get("trim_article_boundaries"))
        self.assertTrue(kw.get("remove_english_abstract"))
        kw2 = clean_kwargs_for_source("Other", "html_ocr", "ru")
        self.assertEqual(kw2, {})

    def test_detect_artifacts_extended(self):
        t = "ISSN 1234-5678\n\nAbstract\n\nText"
        d = detect_artifacts(t, extended=True)
        self.assertIn("noise_issn_like_lines", d)

    def test_analyze_text_quality_keys(self):
        q = analyze_text_quality("Нормальный русский текст " * 50)
        self.assertIn("quality_score", q)
        self.assertIn("quarantine_candidate", q)

    def test_strip_english_abstract_block_unit(self):
        t = (
            "Русский.\n\nAbstract\n\nEnglish only here.\n\n\n"
            "Русский продолжение длинного абзаца " * 2
        )
        new_t, ok = strip_english_abstract_block(t, min_match_pos=0)
        self.assertTrue(ok)
        self.assertNotIn("English only", new_t)

    def test_regression_long_journal_like_csv_snippet(self):
        hdr = (
            "Студенческий научный электронный журнал StudArctic Forum "
            "ПЕТРОЗАВОДСКИЙ ГОСУДАРСТВЕННЫЙ УНИВЕРСИТЕТ"
        )
        chunks = []
        for p in (241, 242, 243, 244, 245):
            chunks.append(hdr)
            chunks.append(str(p))
        chunks.append(
            "Экспрессионизм - явление, возникшее в Германии в первое десятилетие XX века, "
            "ярко контрастировало с французским импрессионизмом."
        )
        raw = "\n".join(chunks)
        out = clean_text(
            raw,
            remove_repeated_boilerplate=True,
            remove_page_noise=True,
            remove_urls=False,
        )
        self.assertIn("Экспрессионизм", out)
        self.assertNotIn("241", out.split())
        self.assertLess(out.count(hdr), raw.count(hdr))

    def test_regression_tuva_footer_style(self):
        line = (
            '247 ""НОВЫЕ ИССЛЕДОВАНИЯ > ТУВЫ"" электронный научный журнал '
            "www.tuva.asia № 4 2009 г."
        )
        core = "Современные коммуникативные технологии способствуют распространению информации."
        raw = "\n".join([line] * 6 + [core])
        out = clean_text(raw, remove_repeated_boilerplate=True, remove_urls=True)
        self.assertIn(core, out)
        self.assertLess(out.count(line), raw.count(line))


if __name__ == "__main__":
    unittest.main()

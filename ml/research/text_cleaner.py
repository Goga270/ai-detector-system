"""
text_cleaner.py
===============
Очистка академических текстов из CyberLeninka (PDF → CSV pipeline).

Артефакты, которые убираем:
  1. PDF-артефакты переноса строк (word-\nbreak → wordbreak)
  2. Страничные колонтитулы (номера страниц, названия журналов)
  3. Ссылки: [1], [1, 2], [1–5], (Иванов, 2021), (Smith et al., 2020)
  4. Разметка таблиц / рисунков: «Таблица 1.», «Рис. 3а», «Figure 2»
  5. Гомоглифы: латинские буквы внутри кириллических слов (с, а, е, о, р и т.д.)
  6. Пунктуационный мусор: «. . .», «,,», «..» (не «...»), лишние дефисы
  7. URL / DOI / email
  8. Non-breaking spaces, zero-width chars, BOM
  9. Подписи к рисункам/таблицам (однострочные капслок-заголовки)
  10. Оставшийся мусор после PDF-конвертации (случайные символы)

API:
    cleaner = TextCleaner()
    clean = cleaner.clean(raw_text)                 # одна строка
    df["text_clean"] = df["text_raw"].map(cleaner.clean)   # pandas
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

# ══════════════════════════════════════════════════════════════════
# HOMOGLYPH MAP (Латынь → Кириллица)
# Используется только когда символ стоит внутри/рядом с кирилл. словом
# ══════════════════════════════════════════════════════════════════

_LAT_TO_CYR: dict[str, str] = {
    "a": "а", "A": "А",
    "e": "е", "E": "Е",
    "o": "о", "O": "О",
    "p": "р", "P": "Р",
    "c": "с", "C": "С",
    "x": "х", "X": "Х",
    "B": "В",
    "H": "Н",
    "T": "Т",
    "M": "М",
    "K": "К",
    "y": "у",   # редко, но бывает в pdf
}

# Строим regex: кириллический контекст (≥1 кир. буква) вокруг латинского гомоглифа
_HOMOGLYPH_PATTERN = re.compile(
    r"(?<=[а-яёА-ЯЁ\-])([" + "".join(re.escape(k) for k in _LAT_TO_CYR) + r"])(?=[а-яёА-ЯЁ\-])"
)


def _fix_homoglyphs(text: str) -> str:
    return _HOMOGLYPH_PATTERN.sub(lambda m: _LAT_TO_CYR.get(m.group(1), m.group(1)), text)


# ══════════════════════════════════════════════════════════════════
# COMPILED PATTERNS (compile один раз при импорте)
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _Patterns:

    # 1. Перенос слова через дефис + перевод строки: «пре-\nдложение» → «предложение»
    hyphen_break: re.Pattern = field(
        default_factory=lambda: re.compile(r"(\w)-\s*\n\s*(\w)")
    )

    # 2. Обычный перенос строки внутри абзаца (не двойной)
    single_newline: re.Pattern = field(
        default_factory=lambda: re.compile(r"(?<!\n)\n(?!\n)")
    )

    # 3. Номера страниц в колонтитуле: строка из 1–3 цифр окружённая пустыми строками
    page_number: re.Pattern = field(
        default_factory=lambda: re.compile(r"^\s*\d{1,3}\s*$", re.MULTILINE)
    )

    # 4. Цифровые ссылки: [1], [1,2], [1, 2, 3], [1–5], [1-5]
    numeric_ref: re.Pattern = field(
        default_factory=lambda: re.compile(r"\[\d+(?:[,;\–\-–—]\s*\d+)*\]")
    )

    # 5. Авторские ссылки: (Иванов, 2021), (Smith et al., 2020), (Иванов и др., 2021)
    author_ref: re.Pattern = field(
        default_factory=lambda: re.compile(
            r"\([А-ЯЁA-Z][а-яёa-z]+(?:\s+et\s+al\.?|\s+и\s+др\.?)?,?\s*\d{4}[a-z]?\)"
        )
    )

    # 6. Таблицы и рисунки: «Таблица 1.», «Рис. 3а», «Figure 2», «Tab. 1»
    table_fig: re.Pattern = field(
        default_factory=lambda: re.compile(
            r"\b(?:Таблица|Рисунок|Рис\.|Figure|Fig\.|Tab\.)\s*\d+[\w\.]*",
            re.IGNORECASE,
        )
    )

    # 7. URL
    url: re.Pattern = field(
        default_factory=lambda: re.compile(
            r"https?://\S+|www\.\S+",
            re.IGNORECASE,
        )
    )

    # 8. DOI
    doi: re.Pattern = field(
        default_factory=lambda: re.compile(r"doi\s*:\s*\S+", re.IGNORECASE)
    )

    # 9. Email
    email: re.Pattern = field(
        default_factory=lambda: re.compile(r"\S+@\S+\.\S+")
    )

    # 10. Пунктуационный мусор
    dot_space_dot: re.Pattern = field(
        default_factory=lambda: re.compile(r"\.\s\.\s\.")    # ". . ." → "..."
    )
    double_comma: re.Pattern = field(
        default_factory=lambda: re.compile(r",{2,}")
    )
    double_dot: re.Pattern = field(
        # «..» но не «...» и не «....»
        default_factory=lambda: re.compile(r"(?<!\.)\.\.(?!\.)")
    )

    # 11. Случайные одиночные латинские буквы между кириллическими словами
    # (артефакт OCR: «исследо v вания» → «исследо вания»)
    lone_latin: re.Pattern = field(
        default_factory=lambda: re.compile(
            r"(?<=[а-яёА-ЯЁ\s])\b[a-zA-Z]\b(?=[а-яёА-ЯЁ\s])"
        )
    )

    # 12. Подписи-заголовки капслоком (строки из 2–10 слов, всё заглавными)
    # Пример: «СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ», «ВВЕДЕНИЕ»
    caps_header: re.Pattern = field(
        default_factory=lambda: re.compile(
            r"^[А-ЯЁ\s\-]{5,80}$", re.MULTILINE
        )
    )

    # 13. Множественные пробелы/табы
    multi_space: re.Pattern = field(
        default_factory=lambda: re.compile(r"[ \t]{2,}")
    )

    # 14. Trailing whitespace на строке
    trailing_space: re.Pattern = field(
        default_factory=lambda: re.compile(r"[ \t]+$", re.MULTILINE)
    )

    # 15. Тройные+ переводы строки → двойные
    multi_newline: re.Pattern = field(
        default_factory=lambda: re.compile(r"\n{3,}")
    )

    # 16. Нечитаемые символы (суррогаты, control chars кроме \n\t)
    control_chars: re.Pattern = field(
        default_factory=lambda: re.compile(
            r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x80-\x9f]"
        )
    )

    # 17. Список литературы в конце (эвристика: «Список литературы» до конца)
    bibliography: re.Pattern = field(
        default_factory=lambda: re.compile(
            r"\n\s*(?:Список\s+(?:использованных\s+)?(?:литературы|источников|ссылок)"
            r"|References|ЛИТЕРАТУРА|REFERENCES)"
            r"[\s\S]*$",
            re.IGNORECASE,
        )
    )

    # 18. Формульные заглушки из PDF: «(1)», «(2)», «(1.2)» на отдельной строке
    formula_label: re.Pattern = field(
        default_factory=lambda: re.compile(r"^\s*\(\d+(?:\.\d+)?\)\s*$", re.MULTILINE)
    )


_P = _Patterns()  # singleton


# ══════════════════════════════════════════════════════════════════
# ZERO-WIDTH & SPECIAL UNICODE CHARS
# ══════════════════════════════════════════════════════════════════

_ZERO_WIDTH = str.maketrans(
    "",
    "",
    "\u200b\u200c\u200d\ufeff\u00ad"  # ZWSP, ZWNJ, ZWJ, BOM, soft-hyphen
    "\u2060\u2061\u2062\u2063",        # word-joiner и математические invisible
)

_NBSP_TABLE = str.maketrans("\u00a0\u202f\u2007", "   ")  # NBSP → обычный пробел


# ══════════════════════════════════════════════════════════════════
# MAIN CLEANER
# ══════════════════════════════════════════════════════════════════

class TextCleaner:
    """
    Stateless pipeline очистки. Порядок шагов важен.

    Параметры:
        remove_bibliography  — убрать список литературы в конце (default: True)
        remove_caps_headers  — убрать КАПСЛОК-заголовки секций (default: True)
        fix_homoglyphs       — исправить латинские гомоглифы (default: True)
        min_length           — если после чистки текст < N символов, вернуть ""
    """

    def __init__(
        self,
        remove_bibliography: bool = True,
        remove_caps_headers: bool = True,
        fix_homoglyphs: bool = True,
        min_length: int = 50,
    ) -> None:
        self.remove_bibliography = remove_bibliography
        self.remove_caps_headers = remove_caps_headers
        self.fix_homoglyphs = fix_homoglyphs
        self.min_length = min_length

        # Строим pipeline как список (step_name, callable)
        self._pipeline: list[tuple[str, Callable[[str], str]]] = [
            ("unicode_normalize",    self._unicode_normalize),
            ("zero_width",          self._remove_zero_width),
            ("nbsp",                self._normalize_nbsp),
            ("control_chars",       self._remove_control_chars),
            ("bibliography",        self._remove_bibliography),
            ("hyphen_break",        self._fix_hyphen_breaks),
            ("url_doi_email",       self._remove_url_doi_email),
            ("refs",                self._remove_refs),
            ("table_fig",           self._remove_table_fig),
            ("formula_labels",      self._remove_formula_labels),
            ("page_numbers",        self._remove_page_numbers),
            ("caps_headers",        self._remove_caps_headers_step),
            ("homoglyphs",          self._fix_homoglyphs_step),
            ("lone_latin",          self._remove_lone_latin),
            ("punct_cleanup",       self._fix_punctuation),
            ("whitespace",          self._normalize_whitespace),
            ("final_strip",         str.strip),
        ]

    # ── STEPS ────────────────────────────────────────────────────

    @staticmethod
    def _unicode_normalize(text: str) -> str:
        # NFC: composed форма, чтобы «е» не была е+combining accent
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def _remove_zero_width(text: str) -> str:
        return text.translate(_ZERO_WIDTH)

    @staticmethod
    def _normalize_nbsp(text: str) -> str:
        return text.translate(_NBSP_TABLE)

    @staticmethod
    def _remove_control_chars(text: str) -> str:
        return _P.control_chars.sub("", text)

    def _remove_bibliography(self, text: str) -> str:
        if self.remove_bibliography:
            text = _P.bibliography.sub("", text)
        return text

    @staticmethod
    def _fix_hyphen_breaks(text: str) -> str:
        # «пре-\nдложе» → «предложе»
        return _P.hyphen_break.sub(r"\1\2", text)

    @staticmethod
    def _remove_url_doi_email(text: str) -> str:
        text = _P.url.sub(" ", text)
        text = _P.doi.sub(" ", text)
        text = _P.email.sub(" ", text)
        return text

    @staticmethod
    def _remove_refs(text: str) -> str:
        text = _P.numeric_ref.sub("", text)
        text = _P.author_ref.sub("", text)
        return text

    @staticmethod
    def _remove_table_fig(text: str) -> str:
        return _P.table_fig.sub("", text)

    @staticmethod
    def _remove_formula_labels(text: str) -> str:
        return _P.formula_label.sub("\n", text)

    @staticmethod
    def _remove_page_numbers(text: str) -> str:
        return _P.page_number.sub("", text)

    def _remove_caps_headers_step(self, text: str) -> str:
        if self.remove_caps_headers:
            # Оставляем только если строка ≥ 4 слов (больше похоже на текст)
            def _keep_or_drop(m: re.Match) -> str:
                line = m.group(0).strip()
                return "" if len(line.split()) <= 6 else m.group(0)
            text = _P.caps_header.sub(_keep_or_drop, text)
        return text

    def _fix_homoglyphs_step(self, text: str) -> str:
        if self.fix_homoglyphs:
            return _fix_homoglyphs(text)
        return text

    @staticmethod
    def _remove_lone_latin(text: str) -> str:
        return _P.lone_latin.sub(" ", text)

    @staticmethod
    def _fix_punctuation(text: str) -> str:
        text = _P.dot_space_dot.sub("...", text)
        text = _P.double_comma.sub(",", text)
        text = _P.double_dot.sub(".", text)
        # «–» и «—» в середине слова (OCR артефакт) → пробел
        text = re.sub(r"(?<=\w)[–—](?=\s)", " ", text)
        return text

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        text = _P.single_newline.sub(" ", text)    # одиночный \n → пробел
        text = _P.trailing_space.sub("", text)
        text = _P.multi_space.sub(" ", text)
        text = _P.multi_newline.sub("\n\n", text)
        return text

    # ── PUBLIC ───────────────────────────────────────────────────

    def clean(self, text: str) -> str:
        """Прогоняет текст через весь pipeline."""
        if not isinstance(text, str):
            return ""
        for _name, step in self._pipeline:
            text = step(text)
        if len(text) < self.min_length:
            return ""
        return text

    def clean_series(self, series) -> "pd.Series":
        """Удобный метод для pandas: df['text_clean'] = cleaner.clean_series(df['text'])"""
        return series.map(self.clean)

    def audit(self, text: str) -> dict:
        """
        Диагностика: показывает, что именно было найдено в тексте.
        Удобно для проверки на небольшой выборке.
        """
        return {
            "numeric_refs":     len(_P.numeric_ref.findall(text)),
            "author_refs":      len(_P.author_ref.findall(text)),
            "urls":             len(_P.url.findall(text)),
            "dois":             len(_P.doi.findall(text)),
            "table_figs":       len(_P.table_fig.findall(text)),
            "caps_headers":     len(_P.caps_header.findall(text)),
            "homoglyphs":       len(_HOMOGLYPH_PATTERN.findall(text)),
            "lone_latins":      len(_P.lone_latin.findall(text)),
            "page_numbers":     len(_P.page_number.findall(text)),
            "has_bibliography": bool(_P.bibliography.search(text)),
            "zero_width_chars": sum(text.count(c) for c in "\u200b\u200c\u200d\ufeff"),
            "control_chars":    len(_P.control_chars.findall(text)),
            "raw_length":       len(text),
        }


# ══════════════════════════════════════════════════════════════════
# INTEGRATION PATCH для generator_pipeline2.py
# ══════════════════════════════════════════════════════════════════

"""
В generator_pipeline2.py замените метод AdaptiveEqualChunker._clean():

  @staticmethod
  def _clean(text: str) -> str:
      text = re.sub(r"\s+", " ", str(text))
      text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)
      return text.strip()

На:

  _text_cleaner = TextCleaner()  # один инстанс на класс

  @staticmethod
  def _clean(text: str) -> str:
      return AdaptiveEqualChunker._text_cleaner.clean(text)

И в main() перед chunker.process_dataframe():

  cleaner = TextCleaner()
  df_raw["text_clean"] = cleaner.clean_series(df_raw["text_clean"])
  logger.info("Text cleaning done.")
"""


# ══════════════════════════════════════════════════════════════════
# QUICK SELF-TEST
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sample = """
ВВЕДЕНИЕ

    Данная статья посвящена исслeдованию (Иванов, 2021) алгоритмов [1, 2, 3]
    кластери-
    зации данных. Рис. 3а показывает результаты.

    Метод демонстрирует высoкую (Smith et al., 2020) эффективность. . .
    Подробнее: https://example.com/paper doi: 10.1234/test  author@uni.ru

(1)

    СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ
    1. Иванов И.И. // Журнал. — 2021. — С. 1–10.
    2. Smith J. et al. // Nature. — 2020.
"""

    cleaner = TextCleaner()
    print("=== AUDIT ===")
    for k, v in cleaner.audit(sample).items():
        print(f"  {k}: {v}")

    print("\n=== CLEANED ===")
    print(cleaner.clean(sample))

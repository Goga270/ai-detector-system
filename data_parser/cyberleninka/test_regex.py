"""
Тест regex-паттернов из cleaner.py.
Запускать: python3 test_regex.py
"""
import re
import time

# ── Паттерны (скопированы из cleaner.py) ────────────────────────
BIB_RE = re.compile(
    r"^[ \t]*"
    r"(?:"
    r"Список\s+(?:литературы|источников?|использованн\S+\s+(?:литературы|источников?))|"
    r"Литература|"
    r"Библиографический\s+список|"
    r"Библиография|"
    r"References?|"
    r"Bibliography"
    r")"
    r"[ \t]*\n"
    r".*",
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)

FRAG_RE = re.compile(
    r"[\w\-_.%]+/[\w\-_.%/]+\.(?:html?|pdf|php|xml|asp|aspx|cfm|do|shtml)\b",
    re.IGNORECASE,
)

URL_RE = re.compile(
    r"https?\s*://\s*[^\s\)\]\}\"\'<>]+|ftp://[^\s]+",
    re.IGNORECASE,
)

ACCESS_RE = re.compile(r"[-–—]?\s*Режим доступа\s*:[^\n]*", re.IGNORECASE)
DATE_RE = re.compile(r"\(?Дата обращения\s*:\s*\d{1,2}\.\d{1,2}\.\d{4}\.?\)?", re.IGNORECASE)

# ── Тест 1: нет backtracking на длинном тексте без слешей ────────
long_text = (
    "Ватиканский собор — Декларация Nostraaetate — "
    "межконфессиональные отношения — философия религии. "
) * 50
t = time.time()
FRAG_RE.sub("", long_text)
elapsed = time.time() - t
assert elapsed < 1.0, f"SLOW backtracking: {elapsed:.2f}s"
print(f"[OK] FRAG_RE backtracking: {elapsed:.4f}s")

# ── Тест 2: удаление библиографии ────────────────────────────────
sample = (
    "Основной текст статьи.\n\n"
    "Список литературы\n"
    "1. Второй Ватиканский собор. "
    "Режим доступа: http:// www.vatican.va/archive/index.html "
    "(Дата обращения: 28.06.2016.)\n"
    "2. Курия Ватикана. councils/chrstuni/doc_19741201_en.html\n"
    "3. Мы помним. tuni_doc_16031998_shoah_en.html\n"
)
t = time.time()
result = BIB_RE.sub("", sample)
elapsed = time.time() - t
assert "Список литературы" not in result, "Список литературы не удалён!"
assert "councils/chrstuni" not in result, "URL-фрагмент не удалён!"
print(f"[OK] Библиография удалена: {elapsed:.4f}s")
print("Результат:", repr(result.strip()))

# ── Тест 3: варианты заголовков ───────────────────────────────────
for header in [
    "Литература\n",
    "References\n",
    "СПИСОК ЛИТЕРАТУРЫ\n",
    "Библиографический список\n",
    "Список использованных источников\n",
    "Bibliography\n",
]:
    text = "Текст.\n\n" + header + "1. Автор. Книга.\n"
    r = BIB_RE.sub("", text)
    ok = header.strip() not in r
    print(f"[{'OK' if ok else 'FAIL'}] '{header.strip()}'")

# ── Тест 4: URL с пробелом после :// ─────────────────────────────
url_text = "см. http:// www.site.ru/page.html здесь"
r = URL_RE.sub("", url_text)
ok = "www.site.ru" not in r
print(f"[{'OK' if ok else 'FAIL'}] URL с пробелом после ://")

# ── Тест 5: «Режим доступа» без URL ──────────────────────────────
access_text = "Режим доступа: остаток строки без URL\n"
r = ACCESS_RE.sub("", access_text)
ok = "Режим доступа" not in r
print(f"[{'OK' if ok else 'FAIL'}] «Режим доступа» без URL")

print("\nВсе тесты завершены.")

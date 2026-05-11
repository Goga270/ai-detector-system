"""Быстрый тест функции clean_text."""
import sys, time
sys.path.insert(0, ".")
from src.cleaner import clean_text

# --- Тест 1: скорость (нет backtracking) --------------------------------
bad = ("Ватиканский собор - Декларация Nostraaetate - "
       "межконфессиональные отношения - философия религии ") * 30
t = time.time()
clean_text(bad)
elapsed = time.time() - t
assert elapsed < 2.0, f"Слишком долго: {elapsed:.2f}s — возможен backtracking!"
print(f"[OK] Backtracking test: {elapsed:.3f}s")

# --- Тест 2: удаление библиографии -----------------------------------
sample = (
    "Основной текст статьи. Далее идут выводы исследования.\n\n"
    "Список литературы\n"
    "1. Второй Ватиканский собор. Режим доступа: "
    "http:// www.vatican.va/archive/index.html (Дата обращения: 28.06.2016.)\n"
    "2. Курия Ватикана. councils/chrstuni/rc_pc_chrstuni_doc_19741201_en.html\n"
    "3. Мы помним. tuni_doc_16031998_shoah_en.html\n"
)
result = clean_text(sample)
print("\n=== После очистки ===")
print(result)
print()
assert "Список литературы" not in result, "Список литературы НЕ удалён!"
assert "vatican.va" not in result, "URL НЕ удалён!"
assert "nostra-aetate_en.html" not in result.lower(), "URL-фрагмент НЕ удалён!"
print("[OK] Список литературы удалён")
print("[OK] URL удалены")

# --- Тест 3: варианты заголовков библиографии -----------------------------------
for header in ["Литература\n", "References\n", "СПИСОК ЛИТЕРАТУРЫ\n",
               "Библиографический список\n"]:
    t2 = "Текст.\n\n" + header + "1. Автор. Книга.\n"
    r2 = clean_text(t2)
    assert header.strip() not in r2, f"Заголовок '{header.strip()}' НЕ удалён!"
    print(f"[OK] '{header.strip()}' удалён")

print("\nВсе тесты прошли!")

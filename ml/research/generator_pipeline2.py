"""
generator_pipeline_v2.py
========================
Adversarial Dataset Generation Pipeline для детекции ИИ-текстов.

DATASET SCHEMA:
─────────────────────────────────────────────
article_id            — ID исходной статьи
chunk_id              — формат: {article_id}_chunk{i}_{human|ai_{strategy}}
text                  — текст чанка (финальный, готовый к обучению)
label                 — 0=Human | 1=Full_AI | 2=Mixed (частичная вставка)
ai_spans_json         — JSON [[start_char, end_char]] — ground truth для span detection BERT
                        label=0: "[]"  |  label=1: "[[0, N]]"  |  label=2: "[[s, e]]"
ai_span_start_char    — начало AI-части в символах (int или null)
ai_span_end_char      — конец AI-части в символах (int или null)
ai_span_start_word    — начало AI-части в словах (int или null)
ai_span_end_word      — конец AI-части в словах (int или null)
ai_fraction           — доля AI-текста в чанке (0.0 → 1.0)
has_mixed_content     — bool: True только для label=2
strategy              — стратегия генерации или 'original'
source_model          — модель-генератор или 'Human'
judge_semantic_score  — оценка семантической верности (1–5), null для human
judge_complexity_score— оценка натуральности / отсутствия AI-штампов (1–5), null для human
judge_avg             — среднее двух судей
word_count            — слов в финальном тексте
char_count            — символов в финальном тексте
chunk_index           — порядковый номер чанка внутри статьи (0-based)
total_chunks          — сколько чанков получено из этой статьи
original_chunk_id     — chunk_id исходного human-чанка (трейсабилити)
created_at            — ISO 8601 UTC

TASK TYPES ДЛЯ BERT-КОЛЛЕГ:
─────────────────────────────
- Бинарная классификация:  label ∈ {0, 1|2}
- 3-классовая:             label ∈ {0, 1, 2}
- Span detection:          use ai_spans_json where label == 2 (самые ценные примеры!)
"""

import asyncio
import json
import logging
import os
import re
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import pandas as pd
from razdel import sentenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.asyncio import tqdm
from pathlib import Path
from dotenv import load_dotenv

from llm_router import call_llm, RouterConfig, HardwareProfile
from text_cleaner import TextCleaner

# ══════════════════════════════════════════════════════════════════
# 0. LOGGING
# ══════════════════════════════════════════════════════════════════
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/generation.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# 1. INFRA CONFIG
# ══════════════════════════════════════════════════════════════════
current_file = Path(__file__).resolve()

project_root = current_file.parent.parent.parent
env_path = project_root / '.env'

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info(f"✅ Переменные окружения загружены из {env_path}")
else:
    logger.warning(f"⚠️ Файл .env не найден по пути {env_path}. Используются системные переменные.")

YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

LOCAL_NODES = ["http://localhost:8001/v1/chat/completions"]

YANDEX_MODELS = [
    f"gpt://{YANDEX_FOLDER_ID}/yandexgpt-5-lite"
]

class RateLimiter:
    """Максимум N запросов в секунду"""
    def __init__(self, rps: int = 15):
        self._sem = asyncio.Semaphore(rps)

    async def acquire(self):
        await self._sem.acquire()
        asyncio.get_event_loop().call_later(1.0, self._sem.release)

rate_limiter = RateLimiter(rps=15)

# ══════════════════════════════════════════════════════════════════
# 2. GOLDEN SETS — размечены вручную, краевые случаи
# ══════════════════════════════════════════════════════════════════

SEMANTIC_GOLDEN_SET = """
<golden_examples>

<example id="S1" score="5" verdict="PERFECT_PRESERVATION">
  <original>Исследование показало, что применение алгоритма k-средних позволило сократить время кластеризации на 34% при сохранении точности в 91,2%. Авторы отмечают, что метод устойчив к выбросам при размере выборки свыше 10 000 наблюдений.</original>
  <rewrite>Согласно результатам эксперимента, алгоритм k-means обеспечил ускорение процесса кластеризации примерно на треть, сохранив точность классификации на уровне 91%. Устойчивость к выбросам проявляется при объёме данных свыше десяти тысяч наблюдений.</rewrite>
  <analysis>Все ключевые факты сохранены: метод (k-means), прирост скорости (~34%), точность (~91%), условие устойчивости (10k+ наблюдений). Незначительные округления не искажают смысл. Причинно-следственная логика сохранена полностью. Score=5.</analysis>
</example>

<example id="S2" score="1" verdict="DOUBLE_HALLUCINATION">
  <original>Авторы предложили трёхэтапную модель адаптации персонала, основанную на теории Выготского о зоне ближайшего развития.</original>
  <rewrite>Авторы предложили пятиэтапную модель профессиональной адаптации, базирующуюся на концепции деятельностного подхода Леонтьева.</rewrite>
  <analysis>ДВОЙНАЯ ГАЛЛЮЦИНАЦИЯ: (1) «три этапа» → «пять этапов» — фактическая ошибка; (2) теоретическая база подменена с Выготского на Леонтьева. Полная подмена смысла. Score=1.</analysis>
</example>

<example id="S3" score="3" verdict="PARTIAL_NUANCE_LOSS">
  <original>Метод продемонстрировал ограниченную эффективность именно в задачах многоклассовой классификации при несбалансированных выборках, тогда как в бинарных задачах результаты были сопоставимы с baseline.</original>
  <rewrite>Метод оказался недостаточно эффективным для задач классификации при несбалансированных данных.</rewrite>
  <analysis>Потеряны критически важные уточнения: (1) «многоклассовой» — убрано, обобщено; (2) полностью удалён контраст с бинарными задачами и сравнение с baseline. Смысл частично сохранён, ключевые нюансы утрачены. Score=3.</analysis>
</example>

<example id="S4" score="2" verdict="SEMANTIC_INVERSION">
  <original>Авторы осторожно предполагают, что корреляция может объясняться третьей переменной, не включённой в модель.</original>
  <rewrite>Авторы утверждают, что обнаруженная корреляция доказывает причинно-следственную связь между переменными.</rewrite>
  <analysis>ИНВЕРСИЯ: «осторожно предполагают» → «утверждают»; «может объясняться третьей переменной» → «доказывает причинно-следственную связь». Полная противоположность исходному тезису. Score=2.</analysis>
</example>

<example id="S5" score="4" verdict="MINOR_SPECIFICITY_LOSS">
  <original>Выборка состояла из 847 студентов московских вузов в возрасте от 18 до 23 лет, обучающихся на технических специальностях.</original>
  <rewrite>В исследовании приняли участие около 850 студентов технических направлений московских университетов.</rewrite>
  <analysis>Число округлено (847→~850) — допустимо. Возрастной диапазон утрачен — это потеря детали, но не смысла. Основной факт (выборка, специальности, город) сохранён. Score=4.</analysis>
</example>

</golden_examples>
"""

COMPLEXITY_GOLDEN_SET = """
<golden_examples>

<example id="C1" score="5" verdict="AUTHENTIC_HUMAN_VOICE">
  <text>Полученные данные, строго говоря, не позволяют нам однозначно отвергнуть нулевую гипотезу — слишком мала выборка. Но вот что любопытно: даже при таком ограничении эффект воспроизводится в трёх независимых когортах. Это, на наш взгляд, говорит само за себя.</text>
  <analysis>Признаки человека: «строго говоря» и «на наш взгляд» — живое хеджирование; «Но вот что любопытно» — риторический поворот; резкая смена длины предложений; конкретика («три когорты»); ноль AI-штампов. Score=5.</analysis>
</example>

<example id="C2" score="1" verdict="TEXTBOOK_AI">
  <text>Важно отметить, что данный подход является весьма эффективным инструментом решения поставленных задач. Следует подчеркнуть, что полученные результаты свидетельствуют о высокой степени достоверности предложенного метода. Таким образом, можно сделать вывод о перспективности дальнейших исследований в данной области.</text>
  <analysis>ДЕТЕКЦИЯ ПО 5 МАРКЕРАМ: «важно отметить» + «следует подчеркнуть» + «таким образом, можно сделать вывод» + «данной области» + абстрактные похвалы без единой конкретной цифры. Классический AI-шаблон. Score=1.</analysis>
</example>

<example id="C3" score="3" verdict="BORDERLINE_HYBRID">
  <text>Результаты исследования показывают, что предложенный алгоритм превосходит существующие методы по ряду ключевых показателей. В частности, точность классификации составила 87,3%, что на 12% выше, чем у алгоритма-конкурента. Данные результаты открывают перспективы для применения метода в прикладных задачах.</text>
  <analysis>СМЕШАННАЯ КАРТИНА: есть конкретные числа (87,3%, 12%) — зелёный флаг; но «ряду ключевых показателей» — AI-размытость, «данные результаты открывают перспективы» — AI-клише. Опытный читатель заподозрит. Score=3.</analysis>
</example>

<example id="C4" score="2" verdict="THIN_DISGUISE">
  <text>Необходимо отметить, что проведённый анализ выявил ряд существенных закономерностей. Полученные данные свидетельствуют о том, что исследуемая область характеризуется высоким потенциалом развития. В рамках данной работы было установлено, что применение предложенного подхода обеспечивает значительное повышение эффективности.</text>
  <analysis>ЧЕТЫРЕ AI-МАРКЕРА: «необходимо отметить» + «ряд существенных закономерностей» + «высоким потенциалом развития» + «значительное повышение эффективности». Ноль конкретных чисел. Score=2.</analysis>
</example>

<example id="C5" score="4" verdict="ALMOST_HUMAN">
  <text>Строго говоря, мы не ожидали такого разброса результатов. Дисперсия в группе контроля оказалась втрое выше предсказанной моделью — и это при том, что условия эксперимента были стандартизированы по всем параметрам. Объяснение, по всей видимости, кроется в неоднородности выборки.</text>
  <analysis>Почти идеально: «строго говоря», «по всей видимости» — живое хеджирование; конкретика («втрое выше»); риторическое «— и это при том, что...». Минус: последнее предложение слегка шаблонно («кроется в»). Score=4.</analysis>
</example>

</golden_examples>
"""


# ══════════════════════════════════════════════════════════════════
# 3. AI STAMP BLACKLIST
# ══════════════════════════════════════════════════════════════════
_BLACKLIST = (
    "важно отметить | следует отметить | необходимо отметить | следует подчеркнуть | "
    "необходимо подчеркнуть | таким образом можно сделать вывод | таким образом | "
    "в рамках данной работы | данная работа посвящена | в заключение следует | "
    "подводя итог | является весьма | является крайне | открывает широкие перспективы | "
    "высокий потенциал развития | значительное повышение эффективности | "
    "ряд ключевых показателей | обеспечивает значительное | характеризуется высоким | "
    "свидетельствует о том что | данные результаты | в ходе исследования было установлено"
)


# ══════════════════════════════════════════════════════════════════
# 4. EDIT STRATEGIES (2 full + 2 partial)
# ══════════════════════════════════════════════════════════════════
EDIT_STRATEGIES: Dict[str, Dict] = {

    # ── FULL STRATEGY 1: Глубокий академический парафраз ──────────
    "paraphrase_deep": {
        "output_handling": "replace_full",
        "system": "Ты академический редактор. Строго следуй инструкциям. Выдавай только текст.",
        "prompt": (
            "Ты — академический редактор с 20-летним стажем работы в рецензируемых журналах.\n\n"
            "ЗАДАЧА: Сделай глубокий смысловой парафраз текста.\n\n"
            "═══ ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА ═══\n"
            "1. Сохрани ВСЕ: числа, имена авторов, названия методов, причинно-следственные связи и степень уверенности авторов (хеджирование).\n"
            "2. Полностью перестрой синтаксис: измени порядок придаточных, переключи залог (актив ↔ пассив), разбей длинные предложения или объедини короткие.\n"
            "3. Замени минимум 75% лексики синонимами или перифразами.\n"
            "4. Итоговый объём — в пределах ±15% от исходного.\n\n"
            "═══ АБСОЛЮТНЫЕ ЗАПРЕТЫ (нарушение = провал) ═══\n"
            f"- Запрещённые штампы: {_BLACKLIST}\n"
            "- Запрещено добавлять информацию, которой нет в оригинале.\n"
            "- Запрещено начинать с «Конечно», «Разумеется», «Вот», «Безусловно» или любых метакомментариев.\n"
            "- Запрещены приветствия, объяснения, пояснения к тексту.\n\n"
            "ФОРМАТ: Выдай ТОЛЬКО итоговый текст. Ничего кроме текста.\n\n"
            "ТЕКСТ:\n{text}"
        ),
    },

    # ── FULL STRATEGY 2: Стилизация под живого учёного ────────────
    "paraphrase_human_voice": {
        "output_handling": "replace_full",
        "system": "Ты учёный, пишущий черновик статьи. Только текст, без пояснений.",
        "prompt": (
            "Ты — исследователь, который пишет раздел своей статьи. Ты думаешь вслух, "
            "формулируешь неравномерно — как живой человек, а не языковая модель.\n\n"
            "ЗАДАЧА: Перепиши текст так, чтобы он звучал как живая академическая речь.\n\n"
            "═══ ОБЯЗАТЕЛЬНЫЕ ТЕХНИКИ ═══\n"
            "1. Используй риторические вопросы или самоперебивания:\n"
            "   «Но здесь возникает вопрос...», «Точнее сказать...», «— а это, кстати, нетривиальный момент —»\n"
            "2. Резко варьируй длину предложений: одно короткое (5–8 слов) → следующее развёрнутое (20–30 слов).\n"
            "3. Используй живое хеджирование: «на наш взгляд», «по всей видимости», «если не ошибаемся».\n"
            "4. Сохрани ВСЕ ключевые факты, числа и термины оригинала. Не выдумывай новые.\n"
            "5. Один раз можно сослаться на личный опыт или наблюдение автора — если это органично.\n\n"
            "═══ АБСОЛЮТНЫЕ ЗАПРЕТЫ ═══\n"
            f"- Запрещённые штампы: {_BLACKLIST}\n"
            "- Запрещено: «В заключение», «Подводя итог», «Таким образом» как вводные.\n"
            "- Запрещено начинать с «Конечно», «Разумеется», «Вот переработанный текст».\n\n"
            "ФОРМАТ: Выдай ТОЛЬКО итоговый текст.\n\n"
            "ИСХОДНЫЙ ТЕКСТ:\n{text}"
        ),
    },

    # ── PARTIAL STRATEGY 1: Вставка ИИ в середину ─────────────────
    "partial_middle_insert": {
        "output_handling": "replace_fragment",
        "fraction": 0.40,
        "system": "Ты редактор, переписывающий фрагмент. Только текст фрагмента, без пояснений.",
        "prompt": (
            "Ты — редактор, который переписывает вырванный фрагмент из середины академического текста.\n\n"
            "ЗАДАЧА: Перепиши приведённый фрагмент другими словами, сохранив его функцию.\n\n"
            "═══ ПРАВИЛА ═══\n"
            "1. Объём переписанного фрагмента — в пределах ±15% от оригинала.\n"
            "2. Сохрани ВСЕ факты, числа, термины и логические связи.\n"
            "3. Стиль должен бесшовно стыковаться с соседними предложениями — "
            "   никаких резких тематических переходов, никаких вводных к новой теме.\n"
            "4. Не добавляй заключений или обобщений — это середина текста, не финал.\n\n"
            "═══ ЗАПРЕТЫ ═══\n"
            f"- Штампы: {_BLACKLIST}\n"
            "- Запрещены любые метакомментарии («Вот переработанный фрагмент:», «Конечно,»).\n\n"
            "ФОРМАТ: Выдай ТОЛЬКО переписанный фрагмент. Без кавычек, без объяснений.\n\n"
            "ФРАГМЕНТ:\n{text}"
        ),
    },

    # ── PARTIAL STRATEGY 2: Продолжение со второй половины ────────
    "partial_continuation": {
        "output_handling": "append",
        "fraction": 0.50,
        "system": "Ты соавтор, дописывающий абзац. Только продолжение, без пояснений.",
        "prompt": (
            "Ты — соавтор академической статьи. Тебе дана первая половина абзаца. "
            "Нужно написать органичное продолжение.\n\n"
            "═══ ПРАВИЛА ═══\n"
            "1. Продолжай РОВНО с того места, где обрывается текст. Не повторяй последнюю фразу.\n"
            "2. Объём продолжения ≈ объёму данного фрагмента (±20%).\n"
            "3. Развивай мысль логически: добавь пример, уточни механизм, обозначь ограничение или контраст.\n"
            "4. Живой академический стиль: варьируй длину предложений, используй хеджирование.\n"
            "5. Заверши мысль абзаца — но не делай глобальных выводов (это не финал статьи).\n\n"
            "═══ ЗАПРЕТЫ ═══\n"
            f"- Штампы: {_BLACKLIST}\n"
            "- «В заключение», «Подводя итог», «Таким образом, можно сделать вывод».\n"
            "- Любые вводные фразы («Конечно, продолжу», «Вот продолжение:»).\n\n"
            "ФОРМАТ: Выдай ТОЛЬКО текст продолжения.\n\n"
            "НАЧАЛО (продолжай отсюда):\n{text}"
        ),
    },
    "back_translation": {
        "output_handling": "replace_full",
        "system": "You are a professional translator. Translate accurately.",
        "prompt": (
            "Translate the following Russian academic text to English.\n"
            "Preserve all facts, names, numbers, and academic tone.\n"
            "Output ONLY the translation, nothing else.\n\n"
            "TEXT:\n{text}"
        ),
        "_back_prompt": (
            "Translate the following English academic text back to Russian.\n"
            "Use natural academic Russian, vary sentence structure.\n"
            "Preserve all facts, terminology, and author hedging.\n"
            "Output ONLY the translation, nothing else.\n\n"
            "TEXT:\n{text}"
        )
    },
    "neutral_editing": {
        "output_handling": "replace_full",
        "system": (
            "Ты — научный редактор журнала ВАК. "
            "Твоя задача — стилистическая правка академических текстов. "
            "Отвечай только отредактированным текстом."
        ),
        "prompt": (
            "Ниже приведён черновик раздела научной статьи. "
            "Выполни стилистическую правку: улучши читаемость, "
            "устрани канцеляризмы, сделай изложение более живым "
            "не теряя академической строгости. "
            "Все факты, цифры и термины сохрани без изменений.\n\n"
            "Верни ТОЛЬКО отредактированный текст.\n\n"
            "Черновик:\n{text}"
        ),
    }
}

FULL_STRATEGIES = ["paraphrase_deep", "paraphrase_human_voice"]
PARTIAL_STRATEGIES = ["partial_middle_insert", "partial_continuation"]
ALL_STRATEGIES = FULL_STRATEGIES + PARTIAL_STRATEGIES


# ══════════════════════════════════════════════════════════════════
# 5. JUDGE PROMPTS
# ══════════════════════════════════════════════════════════════════

SEMANTIC_JUDGE_PROMPT = f"""Ты — строгий эксперт по семантической верификации академических текстов. Роль: беспристрастный судья фактической точности.

{SEMANTIC_GOLDEN_SET}

КРИТЕРИИ (1–5):
  5 — Все факты, числа, причинно-следственные связи и нюансы сохранены. Возможны незначительные округления.
  4 — Основной смысл сохранён. Потеряны 1–2 второстепенные детали, не влияющие на главный тезис.
  3 — Главная мысль угадывается, но утрачены важные уточнения или контрасты.
  2 — Существенные факты искажены или подменены. Смысл частично изменён.
  1 — Галлюцинации, инверсия утверждений, полная подмена смысла.

ИНСТРУКЦИЯ:
1. Найди конкретные расхождения (числа, имена, методы, причинно-следственные связи, степень уверенности).
2. Выдай строгий JSON. НИКАКОГО текста до или после JSON.

ФОРМАТ:
{{"score": <int 1-5>, "issues": ["<проблема 1>", "<проблема 2>"], "verdict": "<одно предложение>"}}

ОРИГИНАЛ:
{{original}}

РЕРАЙТ:
{{rewrite}}"""


COMPLEXITY_JUDGE_PROMPT = f"""Ты — судебный лингвист и эксперт по детекции ИИ-текстов. Задача: оценить, насколько убедительно текст имитирует живую академическую речь человека.

{COMPLEXITY_GOLDEN_SET}

ЧЕКЛИСТ (проверяй каждый пункт):

🔴 КРАСНЫЕ ФЛАГИ (каждый снижает оценку):
  □ AI-штампы: «важно отметить», «следует подчеркнуть», «таким образом можно сделать вывод»,
    «в рамках данной работы», «открывает широкие перспективы», «значительное повышение эффективности»,
    «ряд ключевых показателей», «высокий потенциал», «свидетельствует о том что»
  □ Абстрактные похвалы без конкретики: «весьма эффективный», «перспективное направление»
  □ Монотонный ритм: все предложения одинаковой длины
  □ Нулевая информационная плотность: много слов, мало фактов
  □ Отсутствие живого авторского голоса

🟢 ЗЕЛЁНЫЕ ФЛАГИ (каждый повышает оценку):
  □ Конкретные числа, имена, сравнения
  □ Риторические вопросы и самоперебивания
  □ Резкая вариация длины предложений
  □ Живое хеджирование: «на наш взгляд», «по всей видимости», «строго говоря»
  □ Уточнения типа «— а это, кстати, важно —», «точнее сказать...»

ШКАЛА:
  5 — Убедительно человеческий. Коллегия лингвистов не определит ИИ.
  4 — Почти человеческий. 1–2 подозрительных паттерна, не критичных.
  3 — Микс. AI-маркеры есть, но частично замаскированы.
  2 — Слабая маскировка. Опытный читатель сразу заподозрит ИИ.
  1 — Очевидный ИИ. Штампы, пустые фразы, монотонный ритм.

ПРАВИЛО: Не давай оценку выше 3, если есть хоть один красный флаг без компенсации зелёными.

ИНСТРУКЦИЯ: Будь безжалостен. Выдай строгий JSON. НИКАКОГО текста до или после JSON.

ФОРМАТ:
{{"score": <int 1-5>, "red_flags": ["<флаг>"], "green_flags": ["<флаг>"], "verdict": "<одно предложение>"}}

ТЕКСТ ДЛЯ ОЦЕНКИ:
{{text}}"""


# ══════════════════════════════════════════════════════════════════
# 6. LLM ROUTER (Local vLLM → Yandex fallback)
# ══════════════════════════════════════════════════════════════════
# @retry(
#     stop=stop_after_attempt(4),
#     wait=wait_exponential(multiplier=1, min=2, max=10),
#     retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
#     before_sleep=before_sleep_log(logger, logging.WARNING), # Автоматически залогирует "Retrying..."
#     reraise=True # Если попытки кончились, пробрасываем ошибку дальше
# )
# async def call_llm(
#     session: aiohttp.ClientSession,
#     system_prompt: str,
#     user_prompt: str,
#     temperature: Optional[float] = None,
# ) -> Tuple[Optional[str], Optional[str]]:
#     """Маршрутизатор: Local vLLM → Yandex Cloud fallback."""
#     if temperature is None:
#         temperature = random.uniform(0.6, 0.9)
#
#     payload = {
#         "messages": [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt},
#         ],
#         "temperature": temperature,
#         "max_tokens": 1500,
#     }
#
#     # nodes = LOCAL_NODES[:]
#     # random.shuffle(nodes)
#     # for node_url in nodes:
#     #     payload["model"] = "auto"
#     #     try:
#     #         async with session.post(node_url, json=payload, timeout=aiohttp.ClientTimeout(total=7.0)) as resp:
#     #             if resp.status == 200:
#     #                 data = await resp.json()
#     #                 return data["choices"][0]["message"]["content"], f"Local_{node_url}"
#     #     except (asyncio.TimeoutError, aiohttp.ClientError):
#     #         continue
#
#     # Yandex fallback
#     y_model = random.choice(YANDEX_MODELS)
#     yc_payload = {
#         "modelUri": y_model,
#         "completionOptions": {"stream": False, "temperature": temperature},
#         "messages": [
#             {"role": "system", "text": system_prompt},
#             {"role": "user", "text": user_prompt},
#         ],
#     }
#     yc_headers = {
#         "Authorization": f"Api-Key {YANDEX_API_KEY}",
#         "x-folder-id": YANDEX_FOLDER_ID,
#     }
#     logger.info(f"Запущен процесс генерации чанка({y_model}): {yc_payload} - {yc_headers}")
#     # async with session.post(
#     #         YANDEX_URL, json=yc_payload, headers=yc_headers,
#     #         ssl=False, timeout=aiohttp.ClientTimeout(total=15.0)
#     # ) as resp:
#     #     if resp.status == 200:
#     #         data = await resp.json()
#     #         logger.info(f"Сгенерированный текст {data}")
#     #         return data["result"]["alternatives"][0]["message"]["text"], f"Yandex_{y_model}"
#     try:
#         async with session.post(
#             YANDEX_URL, json=yc_payload, headers=yc_headers,
#             ssl=False, timeout=aiohttp.ClientTimeout(total=60.0)
#         ) as resp:
#             if resp.status == 200:
#                 data = await resp.json()
#                 logger.info(f"Сгенерированный текст {data}")
#                 return data["result"]["alternatives"][0]["message"]["text"], f"Yandex_{y_model}"
#     except Exception as e:
#         logger.error(f"Все LLM недоступны: {e}")
#
#     return None, None


# ══════════════════════════════════════════════════════════════════
# 7. DUAL JUDGE
# ══════════════════════════════════════════════════════════════════

def _parse_judge_json(response_text: Optional[str]) -> Optional[Dict]:
    if not response_text:
        return None
    try:
        clean = re.sub(r"```json|```", "", response_text).strip()
        parsed = json.loads(clean)
    except Exception:
        return None

    if isinstance(parsed, list):
        if len(parsed) > 0 and isinstance(parsed[0], dict):
            return parsed[0]  # Берем первый словарь из списка
        return None

        # Если вернулся нормальный словарь
    elif isinstance(parsed, dict):
        return parsed

    return None


async def call_semantic_judge(
    session: aiohttp.ClientSession, original: str, rewrite: str
) -> Tuple[int, List[str], str]:
    prompt = (SEMANTIC_JUDGE_PROMPT
              .replace("{original}", original)
              .replace("{rewrite}", rewrite))
    resp, _ = await call_llm(session, "Ты беспристрастный судья. Отвечай строго JSON.", prompt, temperature=0.05, role="judge")
    if not resp or is_refusal(resp):
        return 0, ["Судья недоступен, чанк не был сгенерирован в виду запрета ЛЛМ"], "Судья не сработал"
    data = _parse_judge_json(resp)
    if not data:
        return 3, ["Ошибка парсинга"], "Судья не ответил"
    logger.info(f"Итоговая data(call_semantic_judge): {data}")
    raw_score = data.get("score")
    score = int(raw_score) if raw_score is not None else 3
    return score, data.get("issues", []), data.get("verdict", "")


async def call_complexity_judge(
    session: aiohttp.ClientSession, text: str
) -> Tuple[int, List[str], List[str], str]:
    prompt = COMPLEXITY_JUDGE_PROMPT.replace("{text}", text)
    resp, _ = await call_llm(session, "Ты беспристрастный судья. Отвечай строго JSON.", prompt, temperature=0.05, role="judge")
    if not resp or is_refusal(resp):
        return 0, ["Судья недоступен, чанк не был сгенерирован в виду запрета ЛЛМ"], [], "Судья не сработал"
    data = _parse_judge_json(resp)
    if not data:
        return 3, ["Ошибка парсинга"], [], "Судья не ответил"

    logger.info(f"Итоговая data(call_complexity_judge): {data}")
    raw_score = data.get("score")
    score = int(raw_score) if raw_score is not None else 3
    return (
        score,
        data.get("red_flags", []),
        data.get("green_flags", []),
        data.get("verdict", ""),
    )


async def dual_judge(
    session: aiohttp.ClientSession,
    original_text: str,
    full_draft: str,
    ai_part: str,
) -> Tuple[int, str, int, str, float]:
    """Параллельный запуск двух судей. Возвращает (sem_score, sem_fb, compl_score, compl_fb, avg)."""
    sem_task = call_semantic_judge(session, original_text, full_draft)
    compl_task = call_complexity_judge(session, ai_part)

    (sem_score, sem_issues, sem_verdict), (compl_score, compl_red, _, compl_verdict) = \
        await asyncio.gather(sem_task, compl_task)

    sem_fb = f"{sem_verdict} | Issues: {'; '.join(sem_issues[:2])}" if sem_issues else sem_verdict
    compl_fb = f"{compl_verdict} | Red flags: {'; '.join(compl_red[:2])}" if compl_red else compl_verdict
    avg = (sem_score + compl_score) / 2.0

    return sem_score, sem_fb, compl_score, compl_fb, avg


# ══════════════════════════════════════════════════════════════════
# 8. SPAN COMPUTATION
# ══════════════════════════════════════════════════════════════════

def compute_spans(
    full_draft: str,
    ai_part: str,
    output_handling: str,
    prefix_text: str = "",
    human_prefix_text: str = "",   # для append: человеческая первая половина
) -> Dict:
    """
    Вычисляет символьные и словесные оффсеты AI-части в финальном тексте.
    prefix_text  — текст перед AI-частью (для replace_fragment).
    human_prefix_text — человеческая часть (для append).
    """
    total_words = len(full_draft.split())

    if output_handling == "replace_full":
        return {
            "ai_span_start_char": 0,
            "ai_span_end_char": len(full_draft),
            "ai_span_start_word": 0,
            "ai_span_end_word": total_words,
            "ai_fraction": 1.0,
            "has_mixed_content": False,
            "ai_spans_json": json.dumps([[0, len(full_draft)]]),
            "label": 1,
        }

    elif output_handling == "replace_fragment":
        if prefix_text:
            start_char = len(prefix_text) + 1   # +1 = пробел-разделитель
            start_word = len(prefix_text.split())
        else:
            start_char = 0
            start_word = 0
        end_char = start_char + len(ai_part)
        end_word = start_word + len(ai_part.split())
        frac = round((end_word - start_word) / total_words, 3) if total_words else 0.0
        return {
            "ai_span_start_char": start_char,
            "ai_span_end_char": end_char,
            "ai_span_start_word": start_word,
            "ai_span_end_word": end_word,
            "ai_fraction": frac,
            "has_mixed_content": True,
            "ai_spans_json": json.dumps([[start_char, end_char]]),
            "label": 2,
        }

    elif output_handling == "append":
        # full_draft = human_prefix_text + " " + ai_part
        start_char = len(human_prefix_text) + 1
        end_char = len(full_draft)
        start_word = len(human_prefix_text.split())
        end_word = total_words
        frac = round((end_word - start_word) / total_words, 3) if total_words else 0.0
        return {
            "ai_span_start_char": start_char,
            "ai_span_end_char": end_char,
            "ai_span_start_word": start_word,
            "ai_span_end_word": end_word,
            "ai_fraction": frac,
            "has_mixed_content": True,
            "ai_spans_json": json.dumps([[start_char, end_char]]),
            "label": 2,
        }

    # Fallback
    return {
        "ai_span_start_char": None, "ai_span_end_char": None,
        "ai_span_start_word": None, "ai_span_end_word": None,
        "ai_fraction": 0.0, "has_mixed_content": False,
        "ai_spans_json": "[]", "label": 0,
    }


# ══════════════════════════════════════════════════════════════════
# 9. PREAMBLE STRIPPER
# ══════════════════════════════════════════════════════════════════
_PREAMBLE_RE = re.compile(
    r"^(Конечно|Разумеется|Безусловно|Вот|Пожалуйста|Хорошо|Понял|Ок)[^.!?\n]{0,80}[:\.]\s*",
    re.IGNORECASE,
)
_CODE_FENCE_RE = re.compile(r"^```[^\n]*\n|```\s*$", re.MULTILINE)


def strip_ai_preamble(text: str) -> str:
    text = _CODE_FENCE_RE.sub("", text)
    text = _PREAMBLE_RE.sub("", text)
    return text.strip()

_REFUSAL_PATTERNS = re.compile(
    r"(не могу обсуждать|давайте поговорим о чём-нибудь|не могу помочь|"
    r"я не могу|cannot discuss|I can't|отказываюсь|не буду)",
    re.IGNORECASE
)

def is_refusal(text: str) -> bool:
    """Детектим отказ модели."""
    return bool(_REFUSAL_PATTERNS.search(text)) or len(text.strip()) < 50

# ══════════════════════════════════════════════════════════════════
# FAILED RECORDS SCHEMA
# ══════════════════════════════════════════════════════════════════
FAILED_SCHEMA_COLUMNS = [
    "chunk_id", "article_id", "strategy", "reason",
    "attempts_made", "best_avg_score",
    "original_text_preview",  # первые 200 символов
    "created_at",
]

FAIL_REASONS = {
    "ALL_REFUSALS": "Модель отказала во всех 3 попытках (цензура/контентный фильтр)",
    "ALL_JUDGE_FAIL": "Судья не смог оценить ни одну попытку (тоже цензура)",
    "LOW_SCORE": "Лучший avg score <= 3 после 3 попыток",
    "NO_LLM": "LLM недоступна (таймаут/сетевая ошибка)",
}


def save_failed_record(
    chunk_id: str,
    article_id: str,
    strategy: str,
    reason: str,
    attempts_made: int,
    best_avg: float,
    original_text: str,
    filepath: str = "failed_chunks.csv",
) -> None:
    record = {
        "chunk_id": chunk_id,
        "article_id": article_id,
        "strategy": strategy,
        "reason": reason,
        "attempts_made": attempts_made,
        "best_avg_score": round(best_avg, 2),
        "original_text_preview": original_text[:200].replace("\n", " "),
        "created_at": datetime.utcnow().isoformat(),
    }
    pd.DataFrame([record])[FAILED_SCHEMA_COLUMNS].to_csv(
        filepath, mode="a", index=False,
        header=not os.path.exists(filepath)
    )


# ══════════════════════════════════════════════════════════════════
# 10. BACK TRANSLATION
# ══════════════════════════════════════════════════════════════════
async def generate_back_translation(
        session: aiohttp.ClientSession,
        original_text: str,
        base_chunk_id: str,
        chunk_meta: Dict,
        failed_filepath: str = "failed_chunks.csv",
) -> Optional[Dict]:
    """
    Rescue strategy: RU → EN → RU.
    Применяется к чанкам, которые не удалось сгенерировать основными стратегиями.
    Порог судьи снижен до >= 3 (вместо > 3).
    """
    strategy = EDIT_STRATEGIES["back_translation"]

    # Шаг 1: RU → EN
    en_text, source = await call_llm(
        session,
        strategy["system"],
        strategy["prompt"].format(text=original_text),
        temperature=0.3
    )

    if not en_text or is_refusal(en_text):
        logger.warning(f"[{base_chunk_id}|back_translation] EN перевод не удался.")
        return None

    # Шаг 2: EN → RU
    ru_text, _ = await call_llm(
        session,
        strategy["system"],
        strategy["_back_prompt"].format(text=en_text),
        temperature=0.3,
    )

    if not ru_text or is_refusal(ru_text):
        logger.warning(f"[{base_chunk_id}|back_translation] RU перевод не удался.")
        return None

    ru_text = strip_ai_preamble(ru_text)

    # Судья — порог >= 3 (включительно)
    sem_score, sem_fb, compl_score, compl_fb, avg = await dual_judge(
        session, original_text, ru_text, ru_text
    )

    logger.info(f"[{base_chunk_id}|back_translation] sem={sem_score} compl={compl_score} avg={avg:.2f}")

    if avg < 3.0:
        logger.warning(f"[{base_chunk_id}|back_translation] avg={avg:.2f} < 3.0, пропускаем.")
        return None

    spans = compute_spans(
        full_draft=ru_text,
        ai_part=ru_text,
        output_handling="replace_full"
    )

    return {
        "article_id": chunk_meta["article_id"],
        "chunk_id": f"{base_chunk_id}_ai_back_translation",
        "text": ru_text,
        "label": spans["label"],
        "ai_spans_json": spans["ai_spans_json"],
        "ai_span_start_char": spans["ai_span_start_char"],
        "ai_span_end_char": spans["ai_span_end_char"],
        "ai_span_start_word": spans["ai_span_start_word"],
        "ai_span_end_word": spans["ai_span_end_word"],
        "ai_fraction": spans["ai_fraction"],
        "has_mixed_content": spans["has_mixed_content"],
        "strategy": "back_translation",
        "source_model": source,
        "judge_semantic_score": sem_score,
        "judge_complexity_score": compl_score,
        "judge_avg": round(avg, 2),
        "word_count": len(ru_text.split()),
        "char_count": len(ru_text),
        "chunk_index": chunk_meta["chunk_index"],
        "total_chunks": chunk_meta["total_chunks"],
        "original_chunk_id": base_chunk_id,
        "created_at": datetime.utcnow().isoformat(),
    }

# ══════════════════════════════════════════════════════════════════
# 12. SINGLE-STRATEGY GENERATOR WITH REFLEXION
# ══════════════════════════════════════════════════════════════════

async def generate_with_strategy(
    session: aiohttp.ClientSession,
    original_text: str,
    base_chunk_id: str,
    chunk_meta: Dict,
    strategy_name: str,
) -> Optional[Dict]:
    """Генерирует один AI-вариант чанка с рефлексией (макс 3 попытки)."""

    strategy = EDIT_STRATEGIES[strategy_name]
    output_handling = strategy["output_handling"]

    sentences = [s.text for s in sentenize(original_text)]
    n_sents = len(sentences)
    word_count = len(original_text.split())

    # ── Подготовка target_text в зависимости от стратегии ──
    prefix_text = ""
    suffix_text = ""
    human_prefix_text = ""

    if output_handling == "replace_full":
        target_text = original_text

    elif output_handling == "replace_fragment":
        # Чанк слишком короткий — AI-фрагмент будет < 100 слов, нет смысла
        if word_count < AdaptiveEqualChunker.MIN_WORDS_FOR_PARTIAL or n_sents < 6:
            logger.debug(f"[{base_chunk_id}] Chunk {word_count}w < MIN_WORDS_FOR_PARTIAL, fallback → paraphrase_deep")
            strategy_name = "paraphrase_deep"
            strategy = EDIT_STRATEGIES[strategy_name]
            output_handling = "replace_full"
            target_text = original_text
        else:
            fraction = strategy["fraction"]
            num_sents = max(1, int(n_sents * fraction))
            # Берём середину
            start_idx = max(1, (n_sents - num_sents) // 2)
            start_idx = min(start_idx, n_sents - num_sents - 1)
            target_text = " ".join(sentences[start_idx: start_idx + num_sents])
            prefix_text = " ".join(sentences[:start_idx])
            suffix_text = " ".join(sentences[start_idx + num_sents:])

    elif output_handling == "append":
        if word_count < AdaptiveEqualChunker.MIN_WORDS_FOR_PARTIAL or n_sents < 5:
            logger.debug(f"[{base_chunk_id}] Chunk {word_count}w < MIN_WORDS_FOR_PARTIAL, fallback → paraphrase_human_voice")
            strategy_name = "paraphrase_human_voice"
            strategy = EDIT_STRATEGIES[strategy_name]
            output_handling = "replace_full"
            target_text = original_text
        else:
            mid = n_sents // 2
            human_prefix_text = " ".join(sentences[:mid])
            target_text = human_prefix_text

    base_prompt = strategy["prompt"].format(text=target_text)
    current_prompt = base_prompt
    MAX_ATTEMPTS = 3

    best: Optional[Dict] = None
    best_avg = 0.0

    # Трекинг причин отказа
    refusal_count = 0
    judge_fail_count = 0
    no_llm_count = 0

    for attempt in range(1, MAX_ATTEMPTS + 1):

        # 1. Генерация
        raw_text, source = await call_llm(session, strategy["system"], current_prompt)
        if not raw_text:
            no_llm_count += 1
            await asyncio.sleep(2)
            continue
        ai_part = strip_ai_preamble(raw_text)

        logger.info(f"Сгенерирован чанк: {ai_part}")

        if is_refusal(ai_part):
            logger.warning(f"[{base_chunk_id}|{strategy_name}] Модель отказала, пропускаем попытку.")
            refusal_count += 1
            break  # или попробовать другую модель

        # 2. Сборка финального текста
        if output_handling == "replace_full":
            full_draft = ai_part

        elif output_handling == "replace_fragment":
            parts = ([prefix_text] if prefix_text else []) + [ai_part] + ([suffix_text] if suffix_text else [])
            full_draft = " ".join(parts)

        elif output_handling == "append":
            full_draft = f"{human_prefix_text} {ai_part}"

        # 3. Dual judge (параллельно)
        sem_score, sem_fb, compl_score, compl_fb, avg = await dual_judge(
            session, original_text, full_draft, ai_part
        )

        logger.info(
            f"[{base_chunk_id}|{strategy_name}] attempt={attempt}/3 "
            f"sem={sem_score} compl={compl_score} avg={avg:.2f}"
        )

        if sem_score == 0 and compl_score == 0:
            logger.warning(
                f"[{base_chunk_id}|{strategy_name}] Оба судьи вернули 0 — "
                f"цензура или недоступность. Прерываем."
            )
            break

        if avg > best_avg:
            best_avg = avg
            best = {
                "ai_part": ai_part,
                "full_draft": full_draft,
                "source": source,
                "sem_score": sem_score,
                "compl_score": compl_score,
                "avg": avg,
            }

        accept_threshold = 3.0 if attempt == MAX_ATTEMPTS else 3.0
        accept_condition = avg >= accept_threshold if attempt == MAX_ATTEMPTS else avg > 3.0

        if accept_condition:
            logger.info(
                f"[{base_chunk_id}|{strategy_name}] Принят на попытке {attempt} "
                f"(avg={avg:.2f}, порог={'>=3.0' if attempt == MAX_ATTEMPTS else '>3.0'})"
            )
            break

        # 4. Рефлексия
        current_prompt = (
            f"{base_prompt}\n\n"
            f"--- ФИДБЕК РЕДАКТОРОВ (попытка {attempt}) ---\n"
            f"Семантический судья [{sem_score}/5]: {sem_fb}\n"
            f"Судья по стилю [{compl_score}/5]: {compl_fb}\n"
            f"Исправь ВСЕ указанные проблемы и перепиши заново."
        )

    if not best:
        if no_llm_count == MAX_ATTEMPTS:
            reason = FAIL_REASONS["NO_LLM"]
        elif refusal_count == MAX_ATTEMPTS:
            reason = FAIL_REASONS["ALL_REFUSALS"]
        elif judge_fail_count >= 2:
            reason = FAIL_REASONS["ALL_JUDGE_FAIL"]
        else:
            reason = FAIL_REASONS["LOW_SCORE"]

        logger.warning(f"[{base_chunk_id}|{strategy_name}] FAILED: {reason}")
        save_failed_record(
            chunk_id=base_chunk_id,
            article_id=chunk_meta["article_id"],
            strategy=strategy_name,
            reason=reason,
            attempts_made=MAX_ATTEMPTS,
            best_avg=best_avg,
            original_text=original_text,
            filepath="failed_chunks.csv",
        )
        return None

    # ── Вычисление спанов ──
    spans = compute_spans(
        full_draft=best["full_draft"],
        ai_part=best["ai_part"],
        output_handling=output_handling,
        prefix_text=prefix_text,
        human_prefix_text=human_prefix_text,
    )

    return {
        "article_id": chunk_meta["article_id"],
        "chunk_id": f"{base_chunk_id}_ai_{strategy_name}",
        "text": best["full_draft"],
        "label": spans["label"],
        "ai_spans_json": spans["ai_spans_json"],
        "ai_span_start_char": spans["ai_span_start_char"],
        "ai_span_end_char": spans["ai_span_end_char"],
        "ai_span_start_word": spans["ai_span_start_word"],
        "ai_span_end_word": spans["ai_span_end_word"],
        "ai_fraction": spans["ai_fraction"],
        "has_mixed_content": spans["has_mixed_content"],
        "strategy": strategy_name,
        "source_model": best["source"],
        "judge_semantic_score": best["sem_score"],
        "judge_complexity_score": best["compl_score"],
        "judge_avg": round(best["avg"], 2),
        "word_count": len(best["full_draft"].split()),
        "char_count": len(best["full_draft"]),
        "chunk_index": chunk_meta["chunk_index"],
        "total_chunks": chunk_meta["total_chunks"],
        "original_chunk_id": base_chunk_id,
        "created_at": datetime.utcnow().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════
# 13. RUN RESCUE PASS
# ══════════════════════════════════════════════════════════════════

def dedup_failed_chunks(filepath: str = "failed_chunks.csv") -> None:
    """Оставляет последнюю запись по дате для каждого chunk_id."""
    if not os.path.exists(filepath):
        return

    df = pd.read_csv(filepath)
    before = len(df)

    df["created_at"] = pd.to_datetime(df["created_at"])
    df = (
        df.sort_values("created_at", ascending=True)
          .drop_duplicates(subset=["chunk_id"], keep="last")
          .reset_index(drop=True)
    )

    df.to_csv(filepath, index=False)
    logger.info(f"Дедупликация failed_chunks: {before} → {len(df)} записей.")

async def run_rescue_pass(
        human_chunks_df: pd.DataFrame,
        filepath: str = "final_dataset.csv",
        failed_filepath: str = "failed_chunks.csv",
        concurrency: int = 5,
) -> None:
    """Второй проход: back-translation для всех упавших чанков."""
    dedup_failed_chunks(failed_filepath)

    if not os.path.exists(failed_filepath):
        logger.info("ℹ️ Нет failed_chunks.csv — rescue pass пропускается.")
        return

    failed_df = pd.read_csv(failed_filepath)
    processed_ids = set(pd.read_csv(filepath)["chunk_id"].unique())

    # Берём уникальные chunk_id у которых ещё нет back_translation
    to_rescue_bt = [
        row for _, row in failed_df.iterrows()
        if f"{row['chunk_id']}_ai_back_translation" not in processed_ids
    ]

    logger.info(f"🚑 Rescue pass: {len(to_rescue_bt)} чанков на back-translation.")

    # Строим маппинг chunk_id → оригинальный текст
    chunk_text_map = dict(zip(
        human_chunks_df["chunk_id"],
        human_chunks_df["text"]
    ))
    chunk_meta_map = {
        row["chunk_id"]: {
            "article_id": row["article_id"],
            "chunk_index": row["chunk_index"],
            "total_chunks": row["total_chunks"],
        }
        for _, row in human_chunks_df.iterrows()
    }

    sem = asyncio.Semaphore(concurrency)

    async def rescue_worker(row: Dict) -> Optional[Dict]:
        async with sem:
            chunk_id = row["chunk_id"]
            if chunk_id not in chunk_text_map:
                return None
            result = await generate_back_translation(
                session,
                chunk_text_map[chunk_id],
                chunk_id,
                chunk_meta_map[chunk_id],
            )
            if result:
                save_record(result, filepath)
            return result

    connector = aiohttp.TCPConnector(limit=20)
    async with aiohttp.ClientSession(connector=connector) as session:
        coroutines = [rescue_worker(row) for row in to_rescue_bt]
        results = await tqdm.gather(*coroutines, desc="Rescue back_translation")

    success_bt = sum(1 for r in results if r is not None)
    logger.info(f"🚑 Rescue done. {success_bt}/{len(to_rescue_bt)} recovered.")

    processed_ids = set(pd.read_csv(filepath)["chunk_id"].unique())

    to_rescue_ne = [
        row for _, row in failed_df.iterrows()
        if (
                f"{row['chunk_id']}_ai_back_translation" not in processed_ids
                and f"{row['chunk_id']}_ai_neutral_editing" not in processed_ids
        )
    ]
    logger.info(f"🚑 Rescue pass 2 (neutral_editing): {len(to_rescue_ne)} чанков.")

    connector2 = aiohttp.TCPConnector(limit=20)
    async with aiohttp.ClientSession(connector=connector2) as session:
        async def rescue_worker_ne(row: Dict) -> Optional[Dict]:
            async with sem:
                try:
                    chunk_id = row["chunk_id"]
                    if chunk_id not in chunk_text_map:
                        return None
                    result = await generate_with_strategy(
                        session,
                        chunk_text_map[chunk_id],
                        chunk_id,
                        chunk_meta_map[chunk_id],
                        "neutral_editing",
                    )
                    if result:
                        save_record(result, filepath)
                    return result
                except Exception as e:
                    logger.error(f"[rescue_ne|{chunk_id}] {e}")
                    return None

        results_ne = await tqdm.gather(
            *[rescue_worker_ne(row) for row in to_rescue_ne],
            desc="Rescue neutral_editing"
        )

    success_ne = sum(1 for r in results_ne if r is not None)
    logger.info(f"🚑 neutral_editing: {success_ne}/{len(to_rescue_ne)} recovered.")
    logger.info(
        f"🏁 Rescue total: {success_bt + success_ne}/"
        f"{len(to_rescue_bt)} recovered."
    )


# ══════════════════════════════════════════════════════════════════
# 14. ADAPTIVE EQUAL CHUNKER
# ══════════════════════════════════════════════════════════════════

class AdaptiveEqualChunker:
    """
    Определяет оптимальное N чанков по длине текста,
    затем делит текст на N равных (по словам) частей
    с учётом семантических границ.

    Целевой диапазон чанка: 500–750 слов.
    Обоснование: partial edit берёт 40% → 200–300 слов AI-фрагмент.
    Это минимум для качественной генерации и надёжного span detection BERT.

    Таблица N:
    < 500 слов  → 1 чанк  (полный текст, full-strategy only)
    500–1099    → 2 чанка (~550 слов каждый)
    1100–1749   → 2 чанка (~575–875 слов, держим в диапазоне)
    1750–2399   → 3 чанка (~580–800 слов)
    2400–3199   → 4 чанка (~600–800 слов)
    3200–3999   → 5 чанков
    4000+       → floor(words / 600), макс 10

    Граница fallback на full-strategy (MIN_WORDS_FOR_PARTIAL):
    Чанк < 400 слов → не запускаем partial стратегии, только full.
    """

    _N_TABLE = [
        (0,    500,  1),
        (500,  1100, 2),
        (1100, 1750, 2),   # лучше 2 крупных, чем 3 мелких
        (1750, 2400, 3),
        (2400, 3200, 4),
        (3200, 4000, 5),
    ]
    _MAX_CHUNKS = 10
    _WORDS_PER_CHUNK_LARGE = 600   # для текстов 4000+ слов
    MIN_WORDS_FOR_PARTIAL = 400    # ниже этого — только full-стратегии

    def __init__(self, model_name: str = "cointegrated/rubert-tiny2"):
        logger.info(f"📥 AdaptiveChunker: loading {model_name}...")
        self.encoder = SentenceTransformer(model_name, device="cpu")
        self.text_cleaner = TextCleaner()

    # @staticmethod
    # def _clean(self, text: str) -> str:
    #     return self.text_cleaner.clean(text)

    def _get_n(self, word_count: int) -> int:
        for lo, hi, n in self._N_TABLE:
            if lo <= word_count < hi:
                return n
        return min(self._MAX_CHUNKS, word_count // self._WORDS_PER_CHUNK_LARGE)

    def chunk_text(self, text: str) -> List[str]:
        # text = self._clean(self, text)
        sentences = [s.text for s in sentenize(text)]

        if len(sentences) <= 2:
            return [text]

        word_count = len(text.split())
        n = self._get_n(word_count)

        if n <= 1 or len(sentences) < n * 2:
            return [text]

        # Cumulative word counts at each sentence end
        cum_words: List[int] = []
        running = 0
        for s in sentences:
            running += len(s.split())
            cum_words.append(running)
        total_words = cum_words[-1]

        # Semantic similarities between adjacent sentences
        embeddings = self.encoder.encode(sentences, show_progress_bar=False)
        sims = np.array([
            float(cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0])
            for i in range(len(embeddings) - 1)
        ] + [1.0])  # dummy для последнего предложения

        # Find n-1 split points
        target_words = [total_words * k // n for k in range(1, n)]
        split_indices: List[int] = []
        used: set = set()

        for target in target_words:
            best_idx, best_score = None, float("inf")
            for i in range(1, len(sentences)):
                if i in used:
                    continue
                dist_norm = abs(cum_words[i - 1] - target) / total_words
                sem_penalty = float(sims[i - 1])     # низкий sim = хорошая граница
                score = dist_norm * 0.65 + sem_penalty * 0.35
                if score < best_score:
                    best_score = score
                    best_idx = i
            if best_idx is not None:
                split_indices.append(best_idx)
                used.add(best_idx)

        # Собираем чанки
        chunks: List[str] = []
        prev = 0
        for idx in sorted(split_indices):
            chunk = " ".join(sentences[prev:idx]).strip()
            if chunk:
                chunks.append(chunk)
            prev = idx
        tail = " ".join(sentences[prev:]).strip()
        if tail:
            chunks.append(tail)

        return chunks or [text]

    def process_dataframe(
        self, df: pd.DataFrame, text_col: str = "text", id_col: str = "article_id"
    ) -> pd.DataFrame:
        logger.info(f"🧠 Chunking {len(df)} articles...")
        records: List[Dict] = []

        for _, row in df.iterrows():
            text = str(row[text_col])
            chunks = self.chunk_text(text)
            n_chunks = len(chunks)

            for i, chunk in enumerate(chunks):
                wc = len(chunk.split())
                if wc < 20:
                    continue
                records.append({
                    "article_id": row[id_col],
                    "chunk_id": f"{row[id_col]}_chunk{i}_human",
                    "text": chunk,
                    "label": 0,
                    "ai_spans_json": "[]",
                    "ai_span_start_char": None,
                    "ai_span_end_char": None,
                    "ai_span_start_word": None,
                    "ai_span_end_word": None,
                    "ai_fraction": 0.0,
                    "has_mixed_content": False,
                    "strategy": "original",
                    "source_model": "Human",
                    "judge_semantic_score": None,
                    "judge_complexity_score": None,
                    "judge_avg": None,
                    "word_count": wc,
                    "char_count": len(chunk),
                    "chunk_index": i,
                    "total_chunks": n_chunks,
                    "original_chunk_id": f"{row[id_col]}_chunk{i}_human",
                    "created_at": datetime.utcnow().isoformat(),
                })

        logger.info(f"✅ Got {len(records)} human chunks.")
        return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════
# 15. PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════

SCHEMA_COLUMNS = [
    "article_id", "chunk_id", "text", "label",
    "ai_spans_json",
    "ai_span_start_char", "ai_span_end_char",
    "ai_span_start_word", "ai_span_end_word",
    "ai_fraction", "has_mixed_content",
    "strategy", "source_model",
    "judge_semantic_score", "judge_complexity_score", "judge_avg",
    "word_count", "char_count",
    "chunk_index", "total_chunks",
    "original_chunk_id", "created_at",
]


def save_record(record: Dict, filepath: str) -> None:
    row = pd.DataFrame([record])[SCHEMA_COLUMNS]
    row.to_csv(filepath, mode="a", index=False, header=not os.path.exists(filepath))


def _get_missing_strategies(chunk_id: str, processed_ids: set) -> List[str]:
    return [s for s in ALL_STRATEGIES if f"{chunk_id}_ai_{s}" not in processed_ids]


async def run_pipeline(
    human_chunks_df: pd.DataFrame,
    filepath: str = "final_dataset.csv",
    concurrency: int = 5,
) -> None:
    # ── Инициализация файла с human-чанками ──
    processed_ids: set = set()
    if os.path.exists(filepath):
        processed_ids = set(pd.read_csv(filepath)["chunk_id"].unique())
        logger.info(f"♻️  Resumed. Found {len(processed_ids)} existing records.")
    else:
        human_df = human_chunks_df[SCHEMA_COLUMNS]
        human_df.to_csv(filepath, index=False)
        processed_ids.update(human_chunks_df["chunk_id"].tolist())
        logger.info(f"🆕 Created dataset with {len(human_chunks_df)} human chunks.")

    # ── Формируем задачи: (row, strategy_name) ──
    tasks_to_run: List[Tuple[Dict, str]] = []
    for _, row in human_chunks_df.iterrows():
        missing = _get_missing_strategies(row["chunk_id"], processed_ids)
        for strategy_name in missing:
            tasks_to_run.append((row.to_dict(), strategy_name))

    logger.info(f"🚀 Tasks to run: {len(tasks_to_run)} ({len(human_chunks_df)} chunks × {len(ALL_STRATEGIES)} strategies)")

    sem = asyncio.Semaphore(concurrency)

    async def worker(row: Dict, strategy_name: str) -> Optional[Dict]:
        async with sem:
            chunk_meta = {
                "article_id": row["article_id"],
                "chunk_index": row["chunk_index"],
                "total_chunks": row["total_chunks"],
            }
            result = await generate_with_strategy(
                session, row["text"], row["chunk_id"], chunk_meta, strategy_name
            )
            if result:
                save_record(result, filepath)
            return result

    connector = aiohttp.TCPConnector(limit=20)
    async with aiohttp.ClientSession(connector=connector) as session:
        coroutines = [worker(row, sname) for row, sname in tasks_to_run]
        results = await tqdm.gather(*coroutines, desc="Generating")

    success = sum(1 for r in results if r is not None)
    logger.info(f"🏁 Done. {success}/{len(tasks_to_run)} successful.")

# ══════════════════════════════════════════════════════════════════
# 16. ANALYZE
# ══════════════════════════════════════════════════════════════════

def analyze_dataset(
        filepath: str = "final_dataset.csv",
        failed_filepath: str = "failed_chunks.csv"
) -> None:
    """Полный анализ финального датасета."""
    if not os.path.exists(filepath):
        logger.error("final_dataset.csv не найден.")
        return

    df = pd.read_csv(filepath)

    # ── Дедупликация на случай повторных запусков ─────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["chunk_id"], keep="last")
    if before != len(df):
        print(f"⚠️  Удалено дублей: {before - len(df)}")

    print("\n" + "═" * 60)
    print("  АНАЛИЗ ФИНАЛЬНОГО ДАТАСЕТА")
    print("═" * 60)

    # ── 1. Общая статистика ───────────────────────────────────────
    total = len(df)
    human = df[df["label"] == 0]
    full_ai = df[df["label"] == 1]
    mixed = df[df["label"] == 2]

    print(f"\n📊 РАСПРЕДЕЛЕНИЕ ЧАНКОВ (всего: {total})")
    print(f"  Human  (label=0): {len(human):>6}  ({len(human) / total * 100:.1f}%)")
    print(f"  Full AI (label=1): {len(full_ai):>6}  ({len(full_ai) / total * 100:.1f}%)")
    print(f"  Mixed  (label=2): {len(mixed):>6}  ({len(mixed) / total * 100:.1f}%)")

    # ── 2. По стратегиям ──────────────────────────────────────────
    print(f"\n📋 РАСПРЕДЕЛЕНИЕ ПО СТРАТЕГИЯМ")
    strategy_counts = df.groupby("strategy")["chunk_id"].count().sort_values(ascending=False)
    for strategy, count in strategy_counts.items():
        print(f"  {strategy:<35} {count:>6}  ({count / total * 100:.1f}%)")

    # ── 3. Сколько не сгенерировалось ─────────────────────────────
    human_ids = set(human["chunk_id"].unique())
    all_strategies = ["paraphrase_deep", "paraphrase_human_voice",
                      "partial_middle_insert", "partial_continuation"]

    expected_ai = len(human_ids) * len(all_strategies)
    actual_ai = len(df[df["label"].isin([1, 2])])

    print(f"\n❌ ПОКРЫТИЕ ГЕНЕРАЦИИ")
    print(f"  Human чанков уникальных:    {len(human_ids):>6}")
    print(f"  Ожидалось AI чанков:        {expected_ai:>6}  ({len(all_strategies)} стратегии × {len(human_ids)})")
    print(f"  Сгенерировано AI чанков:    {actual_ai:>6}")
    print(
        f"  Не сгенерировано:           {expected_ai - actual_ai:>6}  ({(expected_ai - actual_ai) / expected_ai * 100:.1f}%)")

    # ── 4. Покрытие по стратегиям ─────────────────────────────────
    print(f"\n🎯 ПОКРЫТИЕ ПО КАЖДОЙ СТРАТЕГИИ")
    ai_df = df[df["label"].isin([1, 2])]
    for strategy in all_strategies:
        count = len(ai_df[ai_df["strategy"] == strategy])
        pct = count / len(human_ids) * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {strategy:<35} {bar} {count:>5}/{len(human_ids)} ({pct:.1f}%)")

    # ── 5. Качество по судье ──────────────────────────────────────
    ai_with_scores = ai_df[ai_df["judge_avg"].notna()]
    if len(ai_with_scores) > 0:
        print(f"\n⭐ КАЧЕСТВО (judge_avg) — только AI чанки")
        print(f"  Среднее:   {ai_with_scores['judge_avg'].mean():.2f}")
        print(f"  Медиана:   {ai_with_scores['judge_avg'].median():.2f}")
        print(f"  Мин:       {ai_with_scores['judge_avg'].min():.2f}")
        print(f"  Макс:      {ai_with_scores['judge_avg'].max():.2f}")

        print(f"\n  Распределение оценок:")
        bins = [(0, 2), (2, 3), (3, 4), (4, 5.1)]
        labels = ["0–2 (плохо)", "2–3 (слабо)", "3–4 (норм)", "4–5 (хорошо)"]
        for (lo, hi), label in zip(bins, labels):
            cnt = len(ai_with_scores[
                          (ai_with_scores["judge_avg"] >= lo) &
                          (ai_with_scores["judge_avg"] < hi)
                          ])
            print(f"    {label:<20} {cnt:>6}  ({cnt / len(ai_with_scores) * 100:.1f}%)")

    # ── 6. Статистика по текстам ──────────────────────────────────
    print(f"\n📏 ДЛИНА ТЕКСТОВ (слов)")
    for label, name in [(0, "Human"), (1, "Full AI"), (2, "Mixed")]:
        subset = df[df["label"] == label]["word_count"]
        if len(subset) > 0:
            print(f"  {name:<10} avg={subset.mean():.0f}  "
                  f"min={subset.min()}  max={subset.max()}")

    # ── 7. Failed chunks ──────────────────────────────────────────
    if os.path.exists(failed_filepath):
        failed_df = pd.read_csv(failed_filepath)
        failed_df = failed_df.drop_duplicates(subset=["chunk_id"], keep="last")

        print(f"\n💀 FAILED CHUNKS (после всех rescue pass)")
        print(f"  Всего упавших уникальных: {len(failed_df)}")

        reason_counts = failed_df["reason"].value_counts()
        for reason, count in reason_counts.items():
            print(f"  {reason[:50]:<50} {count:>5}")

    # ── 8. Уникальные статьи ──────────────────────────────────────
    print(f"\n📰 СТАТЬИ")
    print(f"  Уникальных статей в датасете: {df['article_id'].nunique()}")
    print(f"  Среднее чанков на статью:     {df.groupby('article_id').size().mean():.1f}")

    print("\n" + "═" * 60)

# ══════════════════════════════════════════════════════════════════
# 17. ENTRYPOINT
# ══════════════════════════════════════════════════════════════════

async def main() -> None:
    RouterConfig.set_profile(HardwareProfile.YANDEX_ONLY)
    logger.info("Loading dataset...")
    df_raw = pd.read_csv("./datasets/cyberleninka_articles.csv")
    logger.info(f"Articles loaded: {len(df_raw)}")

    chunker = AdaptiveEqualChunker()
    human_chunks_df = chunker.process_dataframe(df_raw, text_col="text", id_col="id")

    # await run_pipeline(human_chunks_df, filepath="final_dataset.csv", concurrency=20)

    await run_rescue_pass(human_chunks_df, filepath="final_dataset.csv", concurrency=10)

    analyze_dataset(filepath="final_dataset.csv", failed_filepath="failed_chunks.csv")

if __name__ == "__main__":
    # import nest_asyncio
    # nest_asyncio.apply()
    asyncio.run(main())
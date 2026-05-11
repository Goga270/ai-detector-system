"""
Централизованная конфигурация newsparser.

Содержит:
- RSS-ленты для каждого источника
- Параметры выгрузки
- Пути к выходным файлам
"""

from pathlib import Path
from typing import Dict, List

# -------------------------------------------------------
# RSS-ленты по источникам
# -------------------------------------------------------
# Ключ — короткое имя источника (используется как имя файла).
# Значение — список URL RSS-лент (парсер пробует их по порядку).

RSS_FEEDS: Dict[str, List[str]] = {
    "lenta": [
        "https://lenta.ru/rss",
        "https://lenta.ru/rss/news",
        "https://lenta.ru/rss/top7",
    ],
    "gazeta": [
        "https://www.gazeta.ru/export/rss/lenta.xml",
        "https://www.gazeta.ru/export/rss/social.xml",
    ],
    "rbc": [
        "https://www.rbc.ru/rbcnews.rss",
        "https://rss.rbc.ru/rbcnews.rss",
        "https://www.rbc.ru/rss/news",
    ],
    "ria": [
        "https://ria.ru/export/rss2/archive/index.xml",
        "https://ria.ru/export/rss2/world/index.xml",
    ],
    "kommersant": [
        "https://www.kommersant.ru/RSS/news.xml",
        "https://www.kommersant.ru/RSS/main.xml",
    ],
}

# Все источники, которые парсятся по умолчанию
DEFAULT_SOURCES: List[str] = list(RSS_FEEDS.keys())

# -------------------------------------------------------
# Параметры выгрузки
# -------------------------------------------------------

# Максимум статей на один источник
MAX_PER_SOURCE: int = 100

# Задержка между запросами к разным фидам одного источника (сек.)
DELAY: float = 0.5

# Таймаут HTTP-запроса (сек.)
REQUEST_TIMEOUT: int = 20

# -------------------------------------------------------
# Выходные файлы
# -------------------------------------------------------

# Все файлы сохраняются в эту папку (создаётся автоматически).
DATA_DIR = Path("data")

# Базовое имя файла (без даты и расширения).
# Итог: data/news_lenta_20260307_143022.csv
FILE_BASENAME = "news"

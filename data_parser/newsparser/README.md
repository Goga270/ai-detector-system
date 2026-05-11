# newsparser

Сборщик новостных статей из RSS-лент и внешних API.

**Источники:**
- **RSS** — lenta.ru, gazeta.ru, rbc.ru, ria.ru, kommersant.ru *(без ключа)*
- **EventRegistry** — новостная база, ~2 500 запросов/мес. бесплатно
- **MediaCloud** — архив новостных медиа

---

## Структура проекта

```
newsparser/
├── main.py                  # Точка входа — запуск сборщика
├── requirements.txt
├── README.md
├── data/                    # Выходные файлы (создаётся автоматически)
└── src/
    ├── __init__.py
    ├── config.py            # RSS-ленты, настройки, пути
    ├── rss.py               # Парсинг RSS через feedparser
    ├── eventregistry.py     # EventRegistry REST API
    └── mediacloud.py        # MediaCloud Python-клиент
```

---

## Быстрый старт

```bash
cd newsparser

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
# python -m venv .venv
# .venv\Scripts\activate.bat

# Базовые зависимости (только RSS)
pip install feedparser requests pandas

# Все зависимости (включая API-клиенты)
pip install -r requirements.txt

# Запуск — все RSS-источники, 100 статей/источник
python3 main.py
```

Файлы сохраняются в `data/` с меткой времени:
```
data/
├── news_lenta_20260307_143022.csv
├── news_gazeta_20260307_143022.csv
├── news_rbc_20260307_143022.csv
├── news_ria_20260307_143022.csv
├── news_kommersant_20260307_143022.csv
└── news_all_20260307_143022.csv        ← объединённый файл
```

---

## Параметры запуска

| Параметр | По умолч. | Описание |
|---|---|---|
| `--sources lenta,rbc` | все | Источники через запятую |
| `--max N` | 100 | Статей на источник |
| `--delay SEC` | 0.5 | Пауза между запросами (сек.) |
| `--er-key KEY` | — | API-ключ EventRegistry |
| `--mc-key KEY` | — | API-ключ MediaCloud |
| `--lang LANG` | ru | Язык для API-источников |
| `--no-json` | — | Не сохранять JSON |
| `--no-merge` | — | Не создавать объединённый файл |

### Примеры

```bash
# Только lenta и rbc, 50 статей
python3 main.py --sources lenta,rbc --max 50

# Все RSS + EventRegistry
python3 main.py --er-key ВАШ_КЛЮЧ

# Все источники с обоими API
python3 main.py --er-key ВАШ_КЛЮЧ --mc-key ВАШ_КЛЮЧ

# Ключи через переменные окружения (безопаснее)
export ER_API_KEY=ВАШ_КЛЮЧ
export MC_API_KEY=ВАШ_КЛЮЧ
python3 main.py

# Только RSS, без JSON
python3 main.py --no-json
```

---

## Структура выходного CSV

| Колонка | Описание |
|---|---|
| `id` | URL статьи (уникальный ключ) |
| `source` | Имя источника (lenta, rbc, eventregistry…) |
| `title` | Заголовок статьи |
| `url` | Полный URL |
| `published_at` | Дата публикации (строка из RSS, не нормализуется) |
| `content` | Текст/аннотация из RSS или тело из API |

---

## Добавить новый RSS-источник

Откройте `src/config.py` и добавьте запись в `RSS_FEEDS`:

```python
RSS_FEEDS = {
    ...
    "vedomosti": [
        "https://www.vedomosti.ru/rss/news.xml",
    ],
}
```

Затем запустите:
```bash
python3 main.py --sources vedomosti
```

---

## Получить API-ключи

| Сервис | Ссылка |
|---|---|
| EventRegistry | https://eventregistry.org/register |
| MediaCloud | https://tools.mediacloud.org/#/user/signup |

---

## Возможные проблемы

**RSS-лента не загружается (`bozo` предупреждение):**  
Скрипт продолжит работу — `feedparser` часто парсит даже «сломанные» ленты.  
Если лента пуста — проверьте URL в браузере и обновите `config.py`.

**`ModuleNotFoundError: feedparser`:**
```bash
pip install feedparser
```

**EventRegistry / MediaCloud: `API-ключ не передан`:**  
Передайте ключ через аргумент `--er-key` или задайте переменную окружения:
```bash
export ER_API_KEY=ваш_ключ
```

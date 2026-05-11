# Газета.Ru

Контент списка новостей на сайте подгружается через **JavaScript**, поэтому обычный HTTP-запрос отдаёт пустую оболочку без ссылок. Для сбора ссылок используется **Playwright** (headless Chromium).

Установка:
```bash
pip install playwright
playwright install chromium
```

Запуск:
```bash
python gazeta/main.py
python gazeta/main.py --limit 50 --clean
```

Выход: `data/gazeta_YYYYMMDD_HHMMSS.csv`.

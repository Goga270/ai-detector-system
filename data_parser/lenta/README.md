# Lenta.ru

Сбор по RSS-лентам и выгрузка полного текста со страниц статей. Ключ не нужен.

RSS: lenta.ru/rss/news, /rss/articles, /rss/last24 и др.

## Запуск

```bash
python lenta/main.py
python lenta/main.py --limit 50 --clean
```

Выход: `data/lenta_YYYYMMDD_HHMMSS.csv`.

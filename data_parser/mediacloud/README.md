# Media Cloud

Метаданные и URL статей через [Media Cloud](https://mediacloud.org/). Полный текст по их правилам из API не отдаётся (copyright).

## Ключ

Регистрация: https://tools.mediacloud.org/#/user/signup  
В профиле — API key.

## Запуск

```bash
pip install mediacloud
python mediacloud/main.py --api-key ВАШ_КЛЮЧ
python mediacloud/main.py --api-key ВАШ_КЛЮЧ --query "политика" --limit 50
python mediacloud/main.py --api-key ВАШ_КЛЮЧ --fetch-text   # подгрузка текста по URL
```

Выход: `data/mediacloud_YYYYMMDD_HHMMSS.csv`. По умолчанию поле `text` пустое; с флагом `--fetch-text` подгружается полный текст по каждой ссылке (медленнее).

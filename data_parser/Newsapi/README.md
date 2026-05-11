# NewsAPI.org

Сбор новостей через [NewsAPI.org](https://newsapi.org). Нужен API key. Поле `content` в API часто обрезано или с HTML — из него убираются теги. Для полного текста используйте `--fetch-text` (подгрузка по URL, медленнее).

## Ключ

Зарегистрироваться: https://newsapi.org/register  
Бесплатный тариф: 100 запросов/день, до 100 статей за запрос (`pageSize`).

## Запуск

```bash
# из корня репозитория
python newsapi/main.py --api-key ВАШ_КЛЮЧ

# с подгрузкой полного текста по ссылке (медленно)
python newsapi/main.py --api-key ВАШ_КЛЮЧ --fetch-text

# опции
python newsapi/main.py --api-key ВАШ_КЛЮЧ --query "философия" --limit 50 --clean
```

Результат: `data/newsapi_YYYYMMDD_HHMMSS.csv` (и при `--clean` ещё `*_cleaned.csv`).

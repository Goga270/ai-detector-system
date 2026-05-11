#!/usr/bin/env bash
# Запуск всех парсеров с лимитом 100. Выполнять из корня репозитория.
# Использует .venv в корне: на macOS/Homebrew системный pip часто заблокирован (PEP 668).

set -e
cd "$(dirname "$0")"
ROOT="$(pwd)"
VENV="$ROOT/.venv"
LIMIT=1000
QUERY="Россия"

if command -v python3 >/dev/null 2>&1; then
  PY_BOOTSTRAP=python3
elif command -v python >/dev/null 2>&1; then
  PY_BOOTSTRAP=python
else
  echo "Не найден python3 или python в PATH" >&2
  exit 1
fi

if [ ! -x "$VENV/bin/python" ]; then
  echo "Создаю виртуальное окружение $VENV ..."
  "$PY_BOOTSTRAP" -m venv "$VENV"
fi
PYTHON="$VENV/bin/python"

if ! "$PYTHON" -c "import pandas" 2>/dev/null; then
  echo "Устанавливаю зависимости: pip install -r requirements.txt ..."
  "$PYTHON" -m pip install -q -U pip
  "$PYTHON" -m pip install -q -r "$ROOT/requirements.txt"
fi

echo "=== Lenta ==="
"$PYTHON" lenta/main.py --limit "$LIMIT"

echo ""
echo "=== Gazeta ==="
"$PYTHON" gazeta/main.py --limit "$LIMIT"

echo ""
echo "=== RBC ==="
"$PYTHON" rbc/main.py --limit "$LIMIT"

echo ""
echo "=== NewsAPI ==="
"$PYTHON" newsapi/main.py --api-key 27a0d8b6dd8e46c3ae86966b9d2cb631 --limit "$LIMIT" --query "$QUERY" --fetch-text

echo ""
echo "=== Media Cloud ==="
"$PYTHON" mediacloud/main.py --api-key 2f9d85ef24b7d7425084732034dd030eaf2ae75d --limit "$LIMIT" --query "$QUERY" --fetch-text

echo ""
echo "=== Анализ и объединение 5 CSV → data/analyze/ ==="
"$PYTHON" analyze_news_datasets.py

echo ""
echo "Готово. Сырые CSV в data/, отчёт и merged — в data/analyze/"

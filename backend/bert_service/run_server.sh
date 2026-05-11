#!/usr/bin/env bash
set -euo pipefail
SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_ROOT="$(cd "$SERVICE_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SERVICE_DIR/../.." && pwd)"
if [[ -f "$BACKEND_ROOT/.env" ]]; then set -a && source "$BACKEND_ROOT/.env" && set +a; fi
export UVICORN_RELOAD=0
VENV="${AI_DETECTOR_VENV:-$REPO_ROOT/.venv}"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "no venv: $VENV" >&2
  exit 1
fi
cd "$SERVICE_DIR"
exec "$VENV/bin/python" -m uvicorn main:app \
  --host "${BERT_HOST:-0.0.0.0}" \
  --port "${BERT_PORT:-8000}"

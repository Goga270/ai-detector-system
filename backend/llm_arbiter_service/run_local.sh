#!/usr/bin/env bash
set -euo pipefail
SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_ROOT="$(cd "$SERVICE_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SERVICE_DIR/../.." && pwd)"
if [[ -f "$BACKEND_ROOT/.env" ]]; then set -a && source "$BACKEND_ROOT/.env" && set +a; fi
VENV="${AI_DETECTOR_VENV:-$REPO_ROOT/.venv}"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "no venv: $VENV" >&2
  exit 1
fi
cd "$SERVICE_DIR"
case "${UVICORN_RELOAD:-0}" in 1|true|yes|TRUE|Yes|on|ON)
  exec "$VENV/bin/python" -m uvicorn main:app \
    --host "${LLM_ARBITER_HOST:-127.0.0.1}" \
    --port "${LLM_ARBITER_PORT:-8002}" \
    --reload
  ;;
*)
  exec "$VENV/bin/python" -m uvicorn main:app \
    --host "${LLM_ARBITER_HOST:-127.0.0.1}" \
    --port "${LLM_ARBITER_PORT:-8002}"
  ;;
esac

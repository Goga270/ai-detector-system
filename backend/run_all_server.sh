#!/usr/bin/env bash
set -euo pipefail
BACKEND="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$BACKEND/.env" ]]; then set -a && source "$BACKEND/.env" && set +a; fi
mkdir -p "$BACKEND/logs"
PIDS="$BACKEND/logs/server.pids"
chmod +x "$BACKEND/bert_service/run_local.sh" \
  "$BACKEND/detectgp_service/run_local.sh" \
  "$BACKEND/llm_arbiter_service/run_local.sh" \
  "$BACKEND/calibrator_service/run_local.sh" \
  "$BACKEND/external_service/run_local.sh" \
  "$BACKEND/bert_service/run_server.sh" \
  "$BACKEND/detectgp_service/run_server.sh" \
  "$BACKEND/llm_arbiter_service/run_server.sh" \
  "$BACKEND/calibrator_service/run_server.sh" \
  "$BACKEND/external_service/run_server.sh" \
  "$BACKEND/run_all_local.sh" \
  "$BACKEND/run_all_server.sh" 2>/dev/null || true

if [[ "${1:-}" == "stop" ]]; then
  if [[ -f "$PIDS" ]]; then
    while read -r pid; do kill "$pid" 2>/dev/null || true; done <"$PIDS"
    rm -f "$PIDS"
    echo stopped
  else
    echo "no pids"
  fi
  exit 0
fi

rm -f "$PIDS"
launch() {
  local svc=$1
  (
    cd "$BACKEND/$svc"
    exec ./run_server.sh
  ) >>"$BACKEND/logs/${svc}-server.log" 2>&1 &
  echo $! >>"$PIDS"
  echo "$svc $!"
}

launch bert_service
launch detectgp_service
launch llm_arbiter_service
sleep 3
launch calibrator_service
sleep 2
launch external_service
echo "logs: $BACKEND/logs/"
echo "$0 stop"
wait

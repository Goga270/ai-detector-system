# Backend

| Сервис | Порт | Префикс |
|--------|------|-----------|
| bert_service | 8000 | `/bert` |
| detectgp_service | 8001 | `/detectgpt` |
| llm_arbiter_service | 8002 | `/arbiter` |
| calibrator_service | 8003 | `/calibrator` |
| external_service | 8004 | `/gateway` |

Внешний вход: `POST http://127.0.0.1:8004/gateway/detect/text` (и `/gateway/detect/pdf`). Межсервисные URL — в `env.example`.

## Локально

```bash
cd /path/to/ai-detector-system
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip && pip install -r requirements.txt
cp backend/env.example backend/.env
export AI_DETECTOR_VENV="$PWD/.venv"
cd backend && ./run_all_local.sh
```

`./run_all_local.sh stop`, логи: `backend/logs/<service>.log`.

## Деплой

```bash
cp dev.env.example dev.env
source dev.env
cd backend && ./sdeploy-backend.sh
```

На ВМ: venv, `pip install -r ~/.requirements-deploy.txt` из выгруженного каталога, `.env`, `./run_all_server.sh`.

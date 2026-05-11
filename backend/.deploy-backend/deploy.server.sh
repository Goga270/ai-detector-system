: "${DEP_SERVER_NAME_U:?}"
: "${TIMESTAMP:?}"
: "${SERVICE_NAME:?}"

REMOTE_APP_DIR="${REMOTE_APP_DIR:-ai-detector-backend}"
DEV_SERVICE_MEMORY_LIMIT="${DEV_SERVICE_MEMORY_LIMIT:-2G}"
TAR_NAME="${SERVICE_NAME}-${TIMESTAMP}.tgz"
TMP_TAR="/tmp/${TAR_NAME}"

if [[ -z "${DEPLOY_NONINTERACTIVE:-}" ]]; then
  read -r -p "deploy? [y/N] " ans
  case "$ans" in y|Y|yes|Yes) ;; *) exit 1 ;; esac
fi

REQ_SNAP="$REPO_ROOT/backend/.requirements-deploy.txt"
cp "$REPO_ROOT/requirements.txt" "$REQ_SNAP"
trap 'rm -f "$REQ_SNAP"' EXIT

echo "tar ${TMP_TAR}"
(cd "$REPO_ROOT/backend" && tar czf "${TMP_TAR}" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.venv' \
  --exclude='logs' \
  .)

echo "scp"
scp "${TMP_TAR}" "${DEP_SERVER_NAME_U}:~/"

ssh "${DEP_SERVER_NAME_U}" bash <<EOF
set -euo pipefail
mkdir -p "\$HOME/${REMOTE_APP_DIR}"
tar xzf "\$HOME/${TAR_NAME}" -C "\$HOME/${REMOTE_APP_DIR}"
chmod +x "\$HOME/${REMOTE_APP_DIR}/bert_service/run_server.sh" "\$HOME/${REMOTE_APP_DIR}/detectgp_service/run_server.sh" "\$HOME/${REMOTE_APP_DIR}/llm_arbiter_service/run_server.sh" "\$HOME/${REMOTE_APP_DIR}/calibrator_service/run_server.sh" "\$HOME/${REMOTE_APP_DIR}/external_service/run_server.sh" "\$HOME/${REMOTE_APP_DIR}/run_all_server.sh" "\$HOME/${REMOTE_APP_DIR}/run_all_local.sh" 2>/dev/null || true
rm -f "\$HOME/${TAR_NAME}"
EOF

rm -f "${TMP_TAR}"
echo "→ ~/${REMOTE_APP_DIR}"

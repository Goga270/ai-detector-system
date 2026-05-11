#!/usr/bin/env bash
set -euo pipefail
BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "$BACKEND_DIR/.." && pwd)"
DEPLOY_TMP="${BACKEND_DIR}/.k8s_dep"
mkdir -p "${DEPLOY_TMP}"
cat "${BACKEND_DIR}/.deploy-backend/deploy.common_envs.sh" \
  "${BACKEND_DIR}/.deploy-backend/deploy.envs.sh" \
  "${BACKEND_DIR}/.deploy-backend/deploy.server.sh" \
  >"${DEPLOY_TMP}/deploy.full.sh"
chmod +x "${DEPLOY_TMP}/deploy.full.sh"
( cd "${BACKEND_DIR}" && "${DEPLOY_TMP}/deploy.full.sh" )
code=$?
rm -rf "${DEPLOY_TMP}"
if [[ "$code" -ne 0 ]]; then
  echo "${BACKEND_DIR}: deploy failed"
  exit "$code"
fi

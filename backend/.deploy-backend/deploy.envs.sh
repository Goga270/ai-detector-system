export TIMESTAMP
TIMESTAMP="$(date +%Y%m%d%H%M%S)"
export SERVICE_NAME="${SERVICE_NAME:-ai-detector-backend}"
export REMOTE_APP_DIR="${REMOTE_APP_DIR:-ai-detector-backend}"
export GIT_HEAD
GIT_HEAD="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
export GIT_BRANCH
GIT_BRANCH="$(git -C "$REPO_ROOT" branch --show-current 2>/dev/null || echo unknown)"
export GIT_STATUS
GIT_STATUS="$(git -C "$REPO_ROOT" status --short 2>/dev/null | tr '/' '_' | head -c 200 || true)"

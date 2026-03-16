#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE_NAME="${IMAGE_NAME:-interactive-world-sim}"
# Match build_docker.sh default for local desktop runs.
IMAGE_TAG="${IMAGE_TAG:-amd64}"
CLEANUP_XHOST=0
DOCKER_ARGS=(
  --rm
  -it
  --gpus all
  -v "${REPO_DIR}:/workspace"
  -w /workspace
)

# Forward X11 display access when available for GUI/OpenCV workflows.
if [ -n "${DISPLAY:-}" ] && [ -d "/tmp/.X11-unix" ]; then
  if command -v xhost >/dev/null 2>&1; then
    if xhost +local:docker >/dev/null 2>&1; then
      CLEANUP_XHOST=1
      cleanup_xhost() {
        if [ "${CLEANUP_XHOST}" -eq 1 ]; then
          xhost -local:docker >/dev/null 2>&1 || \
            echo "[warn] Failed to run 'xhost -local:docker' during cleanup." >&2
        fi
      }
      trap cleanup_xhost EXIT
    else
      echo "[warn] Failed to run 'xhost +local:docker'; GUI apps may not open." >&2
    fi
  else
    echo "[warn] 'xhost' not found; GUI apps in docker may fail to open." >&2
  fi
  DOCKER_ARGS+=(
    -e "DISPLAY=${DISPLAY}"
    -e "QT_X11_NO_MITSHM=1"
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw"
  )
fi

if [ "$#" -eq 0 ]; then
  docker run "${DOCKER_ARGS[@]}" \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    bash
else
  docker run "${DOCKER_ARGS[@]}" \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    "$@"
fi
